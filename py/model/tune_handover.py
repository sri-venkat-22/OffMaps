"""Tune the physics->NN hand-over for a real-data checkpoint -- on VAL drives only.

    PYTHONPATH=. python3 -m model.tune_handover --data ~/OffMaps-data/IO-VNBD-sync --ckpt model/nn_real.pt

Why: on real drives `physics` (hold the last Doppler speed, steer by gyro) beats
the NN for short outages and the NN wins for long ones. The hand-over holds the
entry speed for the first `handover_s` seconds of dead-reckoning, then fuses the
NN (settings from tune_eskf). One knob, swept with everything else fixed.

Scored in the LIVE loop the phone runs (phase6_check.run, the FusionEngine
mirror), not the cold-start harness: on the phone every outage follows minutes of
GNSS, which warms up the Doppler self-cal. Outages of 10/30/60/120 s, 3 min of
GNSS between them, on the val drives; windows averaging < 5 m/s are skipped
(% drift is meaningless when parked). Objective = mean over durations of the
median drift %. The test drives are never loaded. Writes eskf_cfg["handover_s"]
and meta["handover_tuning"] into the checkpoint; then re-export the phone
profile (model/export.py --profile).
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phase6_check as P
from data.iovnbd_sync import load_sync_dir
from model.train_real import split_drives

DURATIONS = [10, 30, 60, 120]
GRID = [0, 5, 8, 10, 12, 15, 20, 30, 45, 60, 1000]      # 1000 = never hand over (pure physics)


def live_drift(val, profile, durations=DURATIONS, gap=180.0):
    """Median drift % per outage duration, live loop, given a profile dict."""
    orig = P.load_profile
    P.load_profile = lambda name=None: profile
    try:
        out = {}
        for D in durations:
            pct = []
            for d in val:
                t0s = np.arange(d.t[0] + gap, d.t[-1] - D - 10, D + gap)
                outs = [(a, a + D) for a in t0s]
                err = P.run(d, outage=outs)[0]
                for a, b in outs:
                    i0, i1 = np.searchsorted(d.t, [a, b]); i1 -= 1
                    dist = np.trapezoid(d.speed[i0:i1], d.t[i0:i1])
                    if dist < 5 * D:
                        continue
                    pct.append(100 * (err[i1] - err[i0]) / dist)
            out[D] = dict(median=float(np.median(pct)), n=len(pct))
        return out
    finally:
        P.load_profile = orig


def tune(data, ckpt):
    import core_bridge as CB
    val = split_drives(load_sync_dir(data, verbose=False))["val"]      # VAL ONLY
    ck = torch.load(ckpt, map_location="cpu")
    base = dict(model="(tuning)", checkpoint=os.path.basename(ckpt),
                calib=ck["calib"], eskf={**CB.ESKF_DEFAULT, **(ck.get("eskf_cfg") or {})})
    log = []
    for h in GRID:
        r = live_drift(val, {**base, "eskf": {**base["eskf"], "handover_s": float(h)}})
        score = float(np.mean([x["median"] for x in r.values()]))
        log.append(dict(handover_s=h, score=score, per_duration=r))
        print(f"handover {h:5.0f}s  " + "  ".join(f"{D}s {x['median']:5.1f}% (n={x['n']})" for D, x in r.items())
              + f"  | mean {score:5.1f}", flush=True)
    best = min(log, key=lambda x: x["score"])
    ck.setdefault("eskf_cfg", {})["handover_s"] = float(best["handover_s"])
    ck.setdefault("meta", {})["handover_tuning"] = dict(
        split="val", loop="live (phase6_check.run)", durations=DURATIONS,
        objective="mean over durations of median drift %", chosen=best["handover_s"], sweep=log)
    torch.save(ck, ckpt)
    with open(ckpt.replace(".pt", ".json"), "w") as f:
        json.dump(ck["meta"], f, indent=2, default=str)
    print(f"chosen handover_s={best['handover_s']} (val mean {best['score']:.1f}%) -> {ckpt}")
    return best["handover_s"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    tune(a.data, a.ckpt)


if __name__ == "__main__":
    main()
