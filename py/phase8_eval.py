"""Phase 8 live-loop evaluation on real IO-VNBD drives (edge_engine = the phone's loop).

    PYTHONPATH=. python3 phase8_eval.py --split val  [--configs shipped,head] [--out ../out/phase8_val.json]
    PYTHONPATH=. python3 phase8_eval.py --split test --configs shipped,head   # ONCE, after every choice is fixed

Outages of 10/30/60/120 s with 3 min of GNSS before each (the Doppler self-cal and
the head's context warm up exactly as on the phone), windows averaging < 5 m/s
skipped (% of distance is meaningless when parked), scored against the vehicle's
survey GNSS. To get more than ~12 windows per duration out of two drives, each
drive is replayed with `--offsets` staggered outage schedules (0/60/120 s shifts):
windows from different schedules never overlap in time within one replay, and
each replay is an independent live run. Metric: median drift % of distance.

Configs:
  shipped   the loop before Phase 8 (nn_real profile: hold Doppler 10 s, then SpeedNet)
  head      + the learned fusion head (model/fusion_head.pt)
  head_zupt + the strict stop detector
"""
from __future__ import annotations
import argparse, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from edge_engine import EdgeEngine, run

DURATIONS = (10, 30, 60, 120)


def streams(d):
    k = math.pi / 180 * 6_371_000.0
    lat0, lon0 = d.meta["lat0"], d.meta["lon0"]
    lat = lat0 + d.n / k; lon = lon0 + d.e / (math.cos(math.radians(lat0)) * k)
    g = d.gyro.copy(); g[:, 2] = -g[:, 2]                       # compass yaw -> device CCW rate
    imu = dict(t=d.t, acc=d.acc, gyro=g, mag=np.full((len(d), 3), np.nan))
    i = np.arange(0, len(d), 10); n = len(i)
    gn = dict(t=d.t[i], lat=lat[i], lon=lon[i], speed=d.speed[i], bearing=np.degrees(d.heading[i]) % 360,
              cn0=np.full(n, 42.0), sv=np.full(n, 9), navic=np.full(n, 3), masked=np.zeros(n))
    return imu, gn


def live(drives, factory, durations=DURATIONS, gap=180.0, offsets=(0.0, 60.0, 120.0), raw=False):
    res = {}
    for D in durations:
        pct = []
        for d in drives:
            imu, gn = streams(d)
            for off in offsets:
                t0s = np.arange(d.t[0] + gap + off, d.t[-1] - D - 10, D + gap)
                outs = [(a, a + D) for a in t0s]
                if not outs:
                    continue
                o = run(factory(), imu, gn, outs)
                err = np.hypot(o[:, 1] - d.e, o[:, 2] - d.n)
                for a, b in outs:
                    i0, i1 = np.searchsorted(d.t, [a, b]); i1 -= 1
                    dist = np.trapezoid(d.speed[i0:i1], d.t[i0:i1])
                    if dist >= 5 * D:
                        pct.append(100 * err[i1] / dist)
        p = np.asarray(pct)
        if raw:
            res[D] = pct; continue
        res[D] = dict(median=float(np.median(p)), p75=float(np.percentile(p, 75)),
                      under10=float((p < 10).mean()), n=int(len(p)))
    return res


def lodo(a):
    import re
    from data.iovnbd_sync import load_sync_dir
    from model.train_real import split_drives
    from model.fusion_head import load_head, FOLDS
    from edge_engine import TorchSpeedNet
    oof = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "oof")
    train = split_drives(load_sync_dir(a.data, verbose=False))["train"]
    fold_of = lambda d: [k for k, rx in FOLDS.items() if re.search(rx, d.vehicle_id.split("#")[0])][0]
    res = {}
    for c in a.configs.split(","):
        per = {D: [] for D in DURATIONS}
        for fk in FOLDS:
            drv = [d for d in train if fold_of(d) == fk]
            net = TorchSpeedNet(os.path.join(oof, f"nn_oof_{fk}.pt"))
            head = load_head(os.path.join(oof, f"fusion_head_oof_{fk}.pt"))
            fac = {"shipped": lambda: EdgeEngine(head=None, speed_net=net, yaw_mode="fast"),
                   "head": lambda: EdgeEngine(head=head, speed_net=net),
                   "ho_head": lambda: EdgeEngine(head=head, speed_net=net, head_handover=True)}[c]
            r = live(drv, fac, raw=True)
            for D in DURATIONS:
                per[D] += r[D]
        res[c] = {D: dict(median=float(np.median(p)), p75=float(np.percentile(p, 75)),
                          under10=float((np.asarray(p) < 10).mean()), n=len(p)) for D, p in per.items()}
        print(f"lodo  {c:10s} " + "  ".join(f"{D}s {x['median']:5.1f}% (p75 {x['p75']:5.1f}, <10%: {100*x['under10']:3.0f}%, n={x['n']})"
                                           for D, x in res[c].items())
              + f"  | mean {np.mean([x['median'] for x in res[c].values()]):5.1f}", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(dict(split="train, leave-one-drive-out (OOF SpeedNet + fold head)", results=res), f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.path.expanduser("~/OffMaps-data/IO-VNBD-sync"))
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--configs", default="shipped,head")
    ap.add_argument("--head", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "fusion_head.pt"))
    ap.add_argument("--out")
    ap.add_argument("--lodo", action="store_true",
                    help="train split, leave-one-drive-out: each drive runs with its out-of-fold SpeedNet "
                         "and a head trained without it (model/oof/)")
    a = ap.parse_args()
    if a.lodo:
        return lodo(a)
    from data.iovnbd_sync import load_sync_dir
    from model.train_real import split_drives
    from model.fusion_head import load_head
    drives = split_drives(load_sync_dir(a.data, verbose=False))[a.split]
    head = load_head(a.head) if os.path.exists(a.head) else None
    CONFIGS = {
        "shipped": lambda: EdgeEngine(head=None, yaw_mode="fast"),     # the pre-Phase-8 loop
        "head": lambda: EdgeEngine(head=head),
        "head_zupt": lambda: EdgeEngine(head=head, zupt="strict"),
        "ho_head": lambda: EdgeEngine(head=head, head_handover=True),
    }
    out = {}
    for c in a.configs.split(","):
        r = live(drives, CONFIGS[c])
        out[c] = r
        print(f"{a.split:5s} {c:10s} " + "  ".join(f"{D}s {x['median']:5.1f}% (p75 {x['p75']:5.1f}, <10%: {100*x['under10']:3.0f}%, n={x['n']})"
                                             for D, x in r.items())
              + f"  | mean {np.mean([x['median'] for x in r.values()]):5.1f}", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(dict(split=a.split, drives=[d.vehicle_id for d in drives], results=out), f, indent=2)


if __name__ == "__main__":
    main()
