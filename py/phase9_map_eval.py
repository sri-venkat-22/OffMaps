"""Phase 9: map-matching fixes (DrishtiNav's gating) on real roads, live loop.

    PYTHONPATH=. python3 phase9_map_eval.py --split val [--configs off,greedy,...] [--out ../out/phase9/val.json]
    PYTHONPATH=. python3 phase9_map_eval.py --lodo [--configs ...]      # train, leave-one-drive-out

Same protocol and metric as phase8_eval.py (live edge engine = the phone's loop,
shipped profile + fusion head, outages 10/30/60/120 s with 3 min of GNSS before
each, median drift % of distance). The roads are the real OSM network of the
IO-VNBD area (map/coventry/roads.bin, now carrying the highway class).

Configs, each one fix on top of the previous where it says "+":
  off          no road snapping (the shipped default)
  greedy       the app's road snapping when switched on (Phase 8 safe settings)
  g_nosvc      + service roads removed from the matcher
  g_sigma      + snap only while filter position sigma <= 25 m
  g_keep       + road updates move neither speed nor gyro bias, 5 m sigma everywhere
  hmm_all      HMM forward filter (road_hmm.py), all road classes, DrishtiNav gating
  hmm          HMM, service roads removed (DrishtiNav's setup)
  hmm_keep     hmm + road updates keep gyro bias too
  hmm_nohead   hmm_keep without the road-heading update
  hmm_c95      hmm_keep with posterior >= 0.95
  hmm_gate     hmm_keep + chi2 innovation gate (1 dof, 99 %) on each road update, as DrishtiNav;
               this is edge_engine.MAP_HMM, the preset behind --map-mode hmm
  hmm_s15/20/40/60/inf  hmm_keep with the filter-sigma gate at 15/20/40/60 m / none (this filter's
               GNSS-aided sigma is ~9 m, DrishtiNav's ~2.5 m, so their 25 m is not ours)
"""
from __future__ import annotations
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import numpy as np

from edge_engine import EdgeEngine
from phase8_eval import live, DURATIONS

HERE = os.path.dirname(os.path.abspath(__file__))
ROADS = os.path.join(HERE, "..", "map", "coventry", "roads.bin")
NOSVC = ("service",)
SAFE = dict(heading=False, unique=True, gate_deg=15.0)          # the shipped live block
HMM = dict(sigma_max=25.0, heading=True, chi2=None, exclude=(), keep=1)   # explicit: no MAP_HMM defaults
OPTS = {
    "off": None,
    "greedy": ("greedy", SAFE),
    "g_nosvc": ("greedy", {**SAFE, "exclude": NOSVC}),
    "g_sigma": ("greedy", {**SAFE, "exclude": NOSVC, "sigma_max": 25.0}),
    "g_keep": ("greedy", {**SAFE, "exclude": NOSVC, "sigma_max": 25.0, "keep": 3, "corridor_sigma": 5.0}),
    "hmm_all": ("hmm", {**HMM, "keep": 1}),
    "hmm": ("hmm", {**HMM, "exclude": NOSVC, "keep": 1}),
    "hmm_keep": ("hmm", {**HMM, "exclude": NOSVC, "keep": 3}),
    "hmm_nohead": ("hmm", {**HMM, "exclude": NOSVC, "keep": 3, "heading": False}),
    "hmm_c95": ("hmm", {**HMM, "exclude": NOSVC, "keep": 3, "conf": 0.95}),
    "hmm_gate": ("hmm", {}),                                    # == edge_engine.MAP_HMM (shipped preset)
    "hmm_s15": ("hmm", {**HMM, "exclude": NOSVC, "keep": 3, "sigma_max": 15.0}),
    "hmm_s20": ("hmm", {**HMM, "exclude": NOSVC, "keep": 3, "sigma_max": 20.0}),
    "hmm_s40": ("hmm", {**HMM, "exclude": NOSVC, "keep": 3, "sigma_max": 40.0}),
    "hmm_s60": ("hmm", {**HMM, "exclude": NOSVC, "keep": 3, "sigma_max": 60.0}),
    "hmm_sinf": ("hmm", {**HMM, "exclude": NOSVC, "keep": 3, "sigma_max": None}),
}


def factory(name, roads, head, net=None):
    o = OPTS[name]
    if o is None:
        return lambda: EdgeEngine(head=head, speed_net=net, map_mode="off")
    mode, opts = o
    return lambda: EdgeEngine(head=head, speed_net=net, roads=roads, map_mode=mode, map_opts=opts)


def summary(per):
    return {D: dict(median=float(np.median(p)), p75=float(np.percentile(p, 75)),
                    under10=float((np.asarray(p) < 10).mean()), n=len(p)) for D, p in per.items()}


def line(tag, c, r):
    return (f"{tag:5s} {c:10s} " + "  ".join(f"{D}s {x['median']:5.1f}% (p75 {x['p75']:5.1f}, <10%: {100*x['under10']:3.0f}%)"
                                            for D, x in r.items())
            + f"  | mean {np.mean([x['median'] for x in r.values()]):5.1f}")


def paired(path, base="off"):
    """Paired comparison against `base` over identical outage windows: median of the
    per-outage drift difference (points) and a bootstrap 90 % interval, per duration."""
    J = json.load(open(path)); P = J["per_outage"]; rng = np.random.default_rng(0)
    for c in P:
        if c == base:
            continue
        row = []
        for D in P[c]:
            x = np.asarray(P[c][D]) - np.asarray(P[base][D])
            bs = [np.median(rng.choice(x, len(x))) for _ in range(2000)]
            row.append(f"{D}s {np.median(x):+5.1f} [{np.percentile(bs, 5):+5.1f},{np.percentile(bs, 95):+5.1f}] "
                       f"better {100 * (x < 0).mean():3.0f}%")
        print(f"{c:10s} vs {base}: " + "  ".join(row))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.path.expanduser("~/OffMaps-data/IO-VNBD-sync"))
    ap.add_argument("--split", choices=["train", "val"], default="val")   # test stays untouched
    ap.add_argument("--lodo", action="store_true")
    ap.add_argument("--configs", default=",".join(OPTS))
    ap.add_argument("--roads", default=ROADS)
    ap.add_argument("--out")
    ap.add_argument("--paired", help="a results json with per_outage: print paired differences vs off")
    a = ap.parse_args()
    if a.paired:
        return paired(a.paired)
    from osm_layers import read_roads_bin
    from data.iovnbd_sync import load_sync_dir
    from model.train_real import split_drives
    from model.fusion_head import load_head
    roads = read_roads_bin(a.roads, with_class=True)
    res, raw = {}, {}
    if a.lodo:
        from model.fusion_head import FOLDS
        from edge_engine import TorchSpeedNet
        oof = os.path.join(HERE, "model", "oof")
        train = split_drives(load_sync_dir(a.data, verbose=False))["train"]
        fold_of = lambda d: [k for k, rx in FOLDS.items() if re.search(rx, d.vehicle_id.split("#")[0])][0]
        for c in a.configs.split(","):
            per = {D: [] for D in DURATIONS}
            for fk in FOLDS:
                drv = [d for d in train if fold_of(d) == fk]
                net = TorchSpeedNet(os.path.join(oof, f"nn_oof_{fk}.pt"))
                head = load_head(os.path.join(oof, f"fusion_head_oof_{fk}.pt"))
                r = live(drv, factory(c, roads, head, net), raw=True)
                for D in DURATIONS:
                    per[D] += r[D]
            raw[c] = per
            res[c] = summary(per)
            print(line("lodo", c, res[c]), flush=True)
    else:
        drives = split_drives(load_sync_dir(a.data, verbose=False))[a.split]
        head = load_head(os.path.join(HERE, "model", "fusion_head.pt"))
        for c in a.configs.split(","):
            raw[c] = live(drives, factory(c, roads, head), raw=True)
            res[c] = summary(raw[c])
            print(line(a.split, c, res[c]), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(dict(split="lodo" if a.lodo else a.split, roads=os.path.relpath(a.roads, HERE),
                           configs={c: repr(OPTS[c]) for c in res}, results=res,
                           per_outage=raw), f, indent=2)   # same windows in the same order: pair them


if __name__ == "__main__":
    main()
