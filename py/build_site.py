"""Data for the hosted demo page (docs/, GitHub Pages): every validation-drive outage,
replayed through the live loop in the four ablation stages.

    PYTHONPATH=. python3 build_site.py [--data ~/OffMaps-data/IO-VNBD-sync]

Writes docs/data/outages.json (per outage: the reference track, each stage's track, the
shipped stage's filter sigma and road-snap flags, end drift per stage), docs/data/roads.json
(the real OSM roads around those outages) and docs/data/ablation.json (out/ablation summaries).

Same protocol as ablation.py / phase8_eval.py (outages 30/60/120 s after >= 3 min of GNSS,
three staggered schedules, windows averaging < 5 m/s skipped), so the medians printed
at the end equal ablation.py's validation table. Validation drives only: the models never
trained on them (they were used for tuning); the test split is not used.
Tracks are stored at 2 Hz in decimetres, in the drive's own ENU frame.
"""
from __future__ import annotations
import argparse, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs", "data")
DURATIONS = (30, 60, 120)
OFFSETS = (0.0, 60.0, 120.0)
GAP, PRE, POST, HZ_OUT = 180.0, 30.0, 20.0, 2
ROAD_MARGIN = 350.0


def dm(a):
    return [int(round(x * 10)) for x in a]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.environ.get("OFFMAPS_IOVNBD", os.path.expanduser("~/OffMaps-data/IO-VNBD-sync")))
    a = ap.parse_args()
    from osm_layers import read_roads_bin, HW_CODE
    from data.iovnbd_sync import load_sync_dir
    from model.train_real import split_drives
    from model.fusion_head import load_head
    from edge_engine import run
    from phase8_eval import streams
    from ablation import factory, STAGES, ROADS, LABEL

    drives = split_drives(load_sync_dir(a.data, verbose=False))["val"]
    ways = read_roads_bin(ROADS, with_class=True)
    head = load_head(os.path.join(HERE, "model", "fusion_head.pt"))
    outages, frames, per = [], {}, {s: {D: [] for D in DURATIONS} for s in STAGES}
    for d in drives:
        imu, gn = streams(d)
        frames[d.vehicle_id] = (d.meta["lat0"], d.meta["lon0"])
        step = int(round(1 / (d.t[1] - d.t[0]) / HZ_OUT))
        for D in DURATIONS:
            for off in OFFSETS:
                t0s = np.arange(d.t[0] + GAP + off, d.t[-1] - D - 10, D + GAP)
                outs = [(x, x + D) for x in t0s]
                if not outs:
                    continue
                runs = {s: run(factory(s, head, ways)(), imu, gn, outs) for s in STAGES}
                for x0, x1 in outs:
                    i0, i1 = np.searchsorted(d.t, [x0, x1]); i1 -= 1
                    dist = float(np.trapezoid(d.speed[i0:i1], d.t[i0:i1]))
                    if dist < 5 * D:
                        continue
                    w0, w1 = np.searchsorted(d.t, [x0 - PRE, x1 + POST])
                    idx = np.arange(w0, min(w1, len(d.t)), step)
                    rec = dict(drive=d.vehicle_id, dur=D, dist=round(dist, 1), t0=round(float(d.t[i0] - d.t[0]), 1),
                               hz=HZ_OUT, denied=[int(np.searchsorted(idx, i0)), int(np.searchsorted(idx, i1))],
                               truth=[dm(d.e[idx]), dm(d.n[idx])], stages={})
                    for s in STAGES:
                        o = runs[s]
                        err = float(np.hypot(o[i1, 1] - d.e[i1], o[i1, 2] - d.n[i1]))
                        pct = 100 * err / dist
                        per[s][D].append(pct)
                        st = dict(e=dm(o[idx, 1]), n=dm(o[idx, 2]), err=round(err, 1), drift=round(pct, 2))
                        if s == "map":        # the shipped app: filter sigma (m) and road-snap flags
                            st["sigma"] = [round(float(math.sqrt(max(o[j, 5] + o[j, 6], 0) / 2)), 1) for j in idx]
                            snap = o[:, 8] > 0
                            st["snap"] = [int(snap[j:j + step].any()) for j in idx]
                        rec["stages"][s] = st
                    outages.append(rec)
        print(f"{d.vehicle_id}: {sum(o['drive'] == d.vehicle_id for o in outages)} outages", flush=True)

    # roads near any outage window, in each drive's frame (decimetres), by class
    k = math.pi / 180 * 6_371_000.0
    roads = {}
    for vid, (lat0, lon0) in frames.items():
        pts = np.concatenate([np.c_[o["truth"][0], o["truth"][1]] for o in outages if o["drive"] == vid]) / 10.0
        if not len(pts):
            continue
        cl = math.cos(math.radians(lat0))
        lo_e, lo_n = pts.min(0) - ROAD_MARGIN; hi_e, hi_n = pts.max(0) + ROAD_MARGIN
        cell = 100.0
        near = {(int(x // cell), int(y // cell)) for x, y in pts}
        r = int(ROAD_MARGIN // cell) + 1
        near = {(cx + i, cy + j) for cx, cy in near for i in range(-r, r + 1) for j in range(-r, r + 1)}
        out = []
        for la, lo, tun, one, cls in ways:
            e = (lo - lon0) * cl * k; n = (la - lat0) * k
            if e.max() < lo_e or e.min() > hi_e or n.max() < lo_n or n.min() > hi_n:
                continue
            if not any((int(x // cell), int(y // cell)) in near for x, y in zip(e, n)):
                continue
            out.append([HW_CODE.get(cls, 0), int(tun)] + [v for x, y in zip(dm(e), dm(n)) for v in (x, y)])
        roads[vid] = out
        print(f"{vid}: {len(out)} road pieces", flush=True)

    summary = {s: {str(D): round(float(np.median(per[s][D])), 2) for D in DURATIONS} for s in STAGES}
    for s in STAGES:
        print(f"{s:9s} " + "  ".join(f"{D}s {summary[s][str(D)]:5.1f}%" for D in DURATIONS)
              + f"  (n = {', '.join(str(len(per[s][D])) for D in DURATIONS)})")
    abl = {}
    for which in ("lodo", "val"):
        J = json.load(open(os.path.join(HERE, "..", "out", "ablation", f"{which}.json")))
        abl[which] = dict(title=J["title"], summary=J["summary"])
    os.makedirs(DOCS, exist_ok=True)
    meta = dict(stages=list(STAGES), labels={s: LABEL[s] for s in STAGES},
                source="IO-VNBD validation drives (Onyekpe et al., 2021); roads (c) OpenStreetMap contributors, ODbL",
                durations=list(DURATIONS), summary=summary)
    for name, obj in (("outages.json", dict(meta=meta, outages=outages)), ("roads.json", roads),
                      ("ablation.json", abl)):
        p = os.path.join(DOCS, name)
        with open(p, "w") as f:
            json.dump(obj, f, separators=(",", ":"))
        print(f"-> {os.path.relpath(p)} ({os.path.getsize(p) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
