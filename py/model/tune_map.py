"""Tune road snapping (map aid) for a real-data checkpoint -- on VAL drives only.

    PYTHONPATH=. python3 -m model.tune_map --data ~/OffMaps-data/IO-VNBD-sync --ckpt model/nn_real.pt

Why: with the real-data settings (srw=24) the phone's road snapping made real
outages WORSE (60 s median drift 12.9 % -> 19.2 %). A crosstrack innovation leaked
into speed through the heading/position correlation. idr_set_map_keep_speed stops
road updates from touching v; this sweeps it with the open-road crosstrack and
heading sigmas. Scored in the LIVE loop (phase6_check.run, the FusionEngine
mirror), 30/60/120 s outages, 3 min of GNSS between, windows averaging < 5 m/s
skipped. The map is each drive's own GNSS track (an oracle map, like the harness
eskf_map), so these numbers are an UPPER BOUND on what a real OSM map gives.
Objective = mean over durations of median drift %. Test drives never loaded.
Writes the mm_* keys into eskf_cfg and meta["map_tuning"] into the checkpoint;
then re-export the phone profile (model/export.py --profile).
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phase6_check as P
from core_bridge import MapMatcher
from mapdata import map_from_path
from data.iovnbd_sync import load_sync_dir
from model.train_real import split_drives

DURATIONS = [30, 60, 120]
GRID = [dict(map=False)] + [dict(mm_keep_speed=False, mm_cross_sigma=1.5, mm_heading_sigma_deg=3.0)] + [
    dict(mm_keep_speed=True, mm_cross_sigma=cs, mm_heading_sigma_deg=hd)
    for cs in (1.5, 3.0, 5.0, 8.0) for hd in (2.0, 3.0, 5.0)]


def oracle_matcher(d):
    m = MapMatcher()
    for e, n, tunnel, oneway in map_from_path(d.e, d.n):
        m.add_way(e, n, tunnel, oneway)
    return m


def live_map_drift(val, profile, use_map=True, durations=DURATIONS, gap=180.0):
    orig = P.load_profile
    P.load_profile = lambda name=None: profile
    try:
        out = {}
        for D in durations:
            pct = []
            for d in val:
                t0s = np.arange(d.t[0] + gap, d.t[-1] - D - 10, D + gap)
                outs = [(a, a + D) for a in t0s]
                err = P.run(d, outage=outs, matcher=oracle_matcher(d) if use_map else None)[0]
                for a, b in outs:
                    i0, i1 = np.searchsorted(d.t, [a, b]); i1 -= 1
                    dist = np.trapezoid(d.speed[i0:i1], d.t[i0:i1])
                    if dist >= 5 * D:
                        pct.append(100 * err[i1] / dist)
            out[D] = dict(median=float(np.median(pct)), n=len(pct))
        return out
    finally:
        P.load_profile = orig


def tune(data, ckpt):
    import core_bridge as CB
    val = split_drives(load_sync_dir(data, verbose=False))["val"]      # VAL ONLY
    ck = torch.load(ckpt, map_location="cpu")
    eskf = {**CB.ESKF_DEFAULT, **(ck.get("eskf_cfg") or {})}
    base = dict(model="(tuning)", checkpoint=os.path.basename(ckpt), calib=ck["calib"])
    log = []
    for g in GRID:
        use_map = g.get("map", True)
        cfg = {**eskf, **{k: v for k, v in g.items() if k != "map"}}
        r = live_map_drift(val, {**base, "eskf": cfg}, use_map)
        score = float(np.mean([x["median"] for x in r.values()]))
        log.append(dict(setting=g, score=score, per_duration=r))
        print(f"{json.dumps(g):80s} " + " / ".join(f"{x['median']:5.1f}%" for x in r.values())
              + f"  mean {score:5.1f}", flush=True)
    best = min((x for x in log if x["setting"].get("map", True)), key=lambda x: x["score"])
    ck.setdefault("eskf_cfg", {}).update(best["setting"])
    ck.setdefault("meta", {})["map_tuning"] = dict(
        split="val", loop="live (phase6_check.run)", map="oracle (drive's own track)",
        durations=DURATIONS, objective="mean over durations of median drift %",
        no_map=log[0], chosen=best, sweep=log)
    torch.save(ck, ckpt)
    with open(ckpt.replace(".pt", ".json"), "w") as f:
        json.dump(ck["meta"], f, indent=2, default=str)
    print(f"chosen {best['setting']} (val mean {best['score']:.1f}% vs no map {log[0]['score']:.1f}%) -> {ckpt}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    tune(a.data, a.ckpt)


if __name__ == "__main__":
    main()
