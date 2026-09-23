"""Tune the ESKF fusion settings for a real-data SpeedNet checkpoint -- on VAL drives only.

    PYTHONPATH=. python3 -m model.tune_eskf --data ~/OffMaps-data/IO-VNBD-sync --ckpt model/nn_real.pt

Why: with the real-trained net the ESKF was WORSE than the net it fuses (41.6 %
vs 19.1 % median drift at 60 s on test). The fusion loop's settings were chosen on
synthetic data, where the accelerometer is clean: the lateral-accel/yaw-rate
curvature speed update, the ZUPT/ZARU trigger (NN v < 0.5 -> clamp v=0 and take
the gyro as pure bias), the speed random walk and the trust in the NN sigma.

Method: coordinate descent over those knobs (core_bridge.ESKF_DEFAULT), each one
swept with the others held, two passes. Objective = mean over outage durations of
the median drift % on the VALIDATION drives' outage windows (seed 0, fixed before
tuning). The test drives are never loaded here. The chosen settings are written
into the checkpoint as "eskf_cfg" (with the whole sweep in meta), so every
consumer of that checkpoint -- validate_realdata --nn-ckpt included -- uses them;
model/nn.pt carries none and keeps the synthetic defaults.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model.nn_model as NNM
import core_bridge as CB
from data.iovnbd_sync import load_sync_dir
from data.outage import make_outages, DURATIONS
from eval.metrics import score_outage
from model.train_real import split_drives

GRID = {
    "curv":      [True, False],
    "curv_sigma": [2.0, 5.0, 10.0],
    "zupt_v":    [0.5, 0.2, None],
    "srw":       [0.3, 1.0, 3.0, 6.0, 12.0, 24.0],
    "sig_scale": [0.25, 0.5, 1.0, 2.0],
}


def _cache_predictions():
    """The NN forward is identical for every fusion config: compute it once per window."""
    orig = CB.predict_steps
    memo = {}

    def cached(net, drive, i0, i1, calib=None):
        k = (id(drive), i0, i1)
        if k not in memo:
            memo[k] = orig(net, drive, i0, i1, calib)
        return memo[k]
    CB.predict_steps = cached


def objective(windows, cfg):
    per = {}
    for d, o in windows:
        r = score_outage(d, o, CB._run(d, o, cfg=cfg))
        per.setdefault(o.duration_s, []).append(r["drift_pct"])
    med = {k: float(np.nanmedian(v)) for k, v in sorted(per.items())}
    return float(np.mean(list(med.values()))), med


def tune(data, ckpt, n_per=30, passes=2):
    NNM.DEFAULT = os.path.abspath(ckpt)
    NNM._CACHE.clear(); NNM._CALIB.clear(); NNM._ESKF.clear()
    val = split_drives(load_sync_dir(data, verbose=False))["val"]      # VAL ONLY
    windows = [(d, o) for d in val for o in make_outages(d, DURATIONS, n_per, 0)]
    print(f"val drives {[d.vehicle_id for d in val]}  windows {len(windows)}")
    _cache_predictions()

    ref = {k: v for k, v in CB.ESKF_DEFAULT.items()}
    # NN alone on the same windows, as the bar fusion should clear
    from eval.models import REGISTRY
    per = {}
    for d, o in windows:
        per.setdefault(o.duration_s, []).append(score_outage(d, o, REGISTRY["nn"](d, o))["drift_pct"])
    nn_med = {k: float(np.nanmedian(v)) for k, v in sorted(per.items())}
    print(f"nn alone (val): mean {np.mean(list(nn_med.values())):.1f}%  {nn_med}")

    cur = dict(ref)
    best, best_med = objective(windows, cur)
    print(f"defaults      : mean {best:.1f}%  {best_med}")
    log = [dict(cfg=dict(cur), score=best, med=best_med)]
    for p in range(passes):
        for knob, vals in GRID.items():
            for v in vals:
                if cur[knob] == v or (knob == "curv_sigma" and not cur["curv"]):
                    continue
                cand = {**cur, knob: v}
                t0 = time.time()
                sc, med = objective(windows, cand)
                log.append(dict(cfg=cand, score=sc, med=med))
                flag = "  <- better" if sc < best - 1e-9 else ""
                print(f"  pass{p} {knob}={v!s:5s}: mean {sc:5.1f}%  ({time.time()-t0:.0f}s){flag}",
                      flush=True)
                if sc < best - 1e-9:
                    best, best_med, cur = sc, med, cand
    print(f"chosen (val)  : mean {best:.1f}%  {best_med}\n  cfg {cur}")
    ck = torch.load(ckpt, map_location="cpu")
    ck["eskf_cfg"] = {k: v for k, v in cur.items() if v != ref[k]} or {}
    ck.setdefault("meta", {})["eskf_tuning"] = dict(
        split="val", n_windows=len(windows), objective="mean over durations of median drift %",
        nn_alone_val=nn_med, default_val=log[0]["med"], chosen_val=best_med, chosen=cur, sweep=log)
    torch.save(ck, ckpt)
    with open(ckpt.replace(".pt", ".json"), "w") as f:
        json.dump(ck["meta"], f, indent=2, default=str)
    print(f"wrote eskf_cfg into {ckpt}: {ck['eskf_cfg']}")
    return cur


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-per-duration", type=int, default=30)
    ap.add_argument("--passes", type=int, default=2)
    a = ap.parse_args()
    tune(a.data, a.ckpt, a.n_per_duration, a.passes)


if __name__ == "__main__":
    main()
