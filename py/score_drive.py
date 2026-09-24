"""Score a real phone drive with the LIVE loop -- one command after the Redmi drive.

    cd py
    adb pull /sdcard/Android/data/com.offmaps/files/drive_<ms> ../drives/
    PYTHONPATH=. python3 -m score_drive --dir ../drives/drive_<ms> [--roads ../map/hyderabad/roads.bin]

The nav app records every session it navigates (imu.csv + gnss.csv in the logger
schema, data/phone_log.py) next to the Phase-1 logger's drives. This replays that
recording through the edge engine (edge_engine.py: the phone's FusionEngine loop,
same core, same ONNX SpeedNet, same profile) and answers the three questions no
IO-VNBD number can:

 1. Drift on THIS phone and car. Simulated outages of 10/30/60/120 s, 3 min of
    GNSS between them (as model/tune_handover.live_drift does on IO-VNBD), scored
    against the phone's own GNSS -- median drift % of distance per duration, for
    the shipped loop, the learned fusion head (if model/fusion_head.pt exists) and
    road snapping (if --roads). Windows averaging < 5 m/s are skipped.
    "Truth" is phone GNSS (~3-5 m), not survey GNSS: fine for 60 s outages of
    hundreds of metres, noted in the report.
 2. Stops (item 6: should ZUPT come back?). Every parked span (GNSS speed < 0.3 m/s
    for >= 20 s) is replayed as an outage with ZUPT off vs the strict detector:
    drift accumulated while parked, SpeedNet's speed while parked, gyro noise.
    Recommendation = strict ZUPT on only if it reduces parked drift AND does not
    worsen the moving-outage drift of (1).
 3. Sanity of this phone's sensors: IMU rate, GNSS rate / C/N0 / trust, heading
    agreement with GNSS course while moving (a mirrored yaw sign shows up here).

Writes out/phone/<name>/REPORT.md + results.json.
"""
from __future__ import annotations
import argparse, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from edge_engine import EdgeEngine, read_inputs, run
from data.outage import tilde_home

DURATIONS = (10, 30, 60, 120)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gnss_truth(eng_factory, imu, gn):
    """Run with GNSS on to get the ENU frame, then truth = GNSS fixes in that frame."""
    eng = eng_factory(); out = run(eng, imu, gn)
    k = math.pi / 180 * 6_371_000.0
    e = (gn["lon"] - eng.lon0) * math.cos(math.radians(eng.lat0)) * k
    n = (gn["lat"] - eng.lat0) * k
    return eng, out, e, n


def err_at_fixes(out, imu_t, gt, ge, gn_):
    i = np.clip(np.searchsorted(imu_t, gt) - 1, 0, len(imu_t) - 1)
    return np.hypot(out[i, 1] - ge, out[i, 2] - gn_)


def outage_drift(factory, imu, gn, ge, gnn, D, gap=180.0):
    gt, v = gn["t"], np.nan_to_num(gn["speed"])
    t0s = np.arange(gt[0] + gap, gt[-1] - D - 10, D + gap)
    outs = [(a, a + D) for a in t0s]
    if not outs:
        return []
    out = run(factory(), imu, gn, outs)
    err = err_at_fixes(out, imu["t"], gt, ge, gnn)
    pct = []
    for a, b in outs:
        j0, j1 = np.searchsorted(gt, [a, b]); j1 -= 1
        dist = np.trapezoid(v[j0:j1 + 1], gt[j0:j1 + 1])
        if dist >= 5 * D:
            pct.append(100 * err[j1] / dist)
    return pct


def parked_spans(gn, min_s=20.0):
    gt, v = gn["t"], np.nan_to_num(gn["speed"], nan=99)
    still = v < 0.3
    spans, i = [], 0
    while i < len(gt):
        if still[i]:
            j = i
            while j + 1 < len(gt) and still[j + 1]:
                j += 1
            if gt[j] - gt[i] >= min_s and gt[i] > gt[0] + 60:
                spans.append((gt[i], gt[j]))
            i = j + 1
        else:
            i += 1
    return spans


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--roads")
    ap.add_argument("--head", default=os.path.join(ROOT, "py", "model", "fusion_head.pt"))
    ap.add_argument("--vehicle", choices=["car", "two_wheeler"], default="car")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    name = os.path.basename(os.path.normpath(a.dir))
    outdir = a.out or os.path.join(ROOT, "out", "phone", name)
    os.makedirs(outdir, exist_ok=True)
    imu, gn = read_inputs(os.path.join(a.dir, "imu.csv"), os.path.join(a.dir, "gnss.csv"))
    ok = np.isfinite(gn["lat"])
    gn = {k: np.asarray(v)[ok] for k, v in gn.items()}
    gn["masked"] = np.zeros(len(gn["t"]))          # the recording's own outage toggles are ignored: we simulate
    roads = None
    if a.roads:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        from osm_layers import read_roads_bin
        roads = read_roads_bin(a.roads)
    head = None
    if a.head and os.path.exists(a.head):
        from model.fusion_head import load_head
        head = load_head(a.head)

    base = lambda **kw: (lambda: EdgeEngine(vehicle=a.vehicle, **kw))
    eng, out_on, ge, gnn = gnss_truth(base(), imu, gn)
    span = imu["t"][-1] - imu["t"][0]
    moving = np.nan_to_num(gn["speed"]) > 3
    brg = np.radians(gn["bearing"])
    i = np.clip(np.searchsorted(imu["t"], gn["t"]) - 1, 0, len(imu["t"]) - 1)
    hd = np.degrees(np.abs((out_on[i, 3] - brg + np.pi) % (2 * np.pi) - np.pi))
    sanity = dict(imu_hz=float((len(imu["t"]) - 1) / span), gnss_hz=float((len(gn["t"]) - 1) / (gn["t"][-1] - gn["t"][0])),
                  duration_min=float(span / 60), distance_km=float(np.trapezoid(np.nan_to_num(gn["speed"]), gn["t"]) / 1000),
                  cn0_median=float(np.median(gn["cn0"])), sv_median=float(np.median(gn["sv"])),
                  navic_max=int(np.max(gn["navic"])),
                  heading_vs_course_median_deg=float(np.nanmedian(hd[moving])) if moving.any() else None,
                  gnss_on_err_median_m=float(np.median(err_at_fixes(out_on, imu["t"], gn["t"], ge, gnn))),
                  heading_seed=eng.heading_seed)

    configs = {"shipped loop": base()}
    if head is not None:
        configs["+ fusion head"] = base(head=head)
    if roads is not None:
        configs["+ road snapping (safe)"] = base(roads=roads, map_mode="greedy",
                                                 map_opts=dict(heading=False, unique=True, gate_deg=15))
    drift = {}
    for nm, fac in configs.items():
        drift[nm] = {}
        for D in DURATIONS:
            p = outage_drift(fac, imu, gn, ge, gnn, D)
            drift[nm][D] = dict(median=float(np.median(p)) if p else None, n=len(p))

    stops = []
    for s0, s1 in parked_spans(gn):
        row = dict(t0=float(s0), dur=float(s1 - s0))
        for nm, zu in (("zupt_off", None), ("zupt_strict", "strict")):
            e = EdgeEngine(vehicle=a.vehicle, zupt=zu)
            o = run(e, imu, gn, [(s0 - 5.0, s1)])
            j1 = max(int(np.searchsorted(gn["t"], s1)) - 1, 0)
            row[nm] = float(err_at_fixes(o, imu["t"], gn["t"], ge, gnn)[j1])   # error when the stop ends
        k0, k1 = np.searchsorted(imu["t"], [s0, s1])
        g = imu["gyro"][k0:k1]
        row["gyro_std_dps"] = float(np.degrees(g.std(0).mean())) if len(g) else None
        stops.append(row)
    if stops:
        off = np.median([r["zupt_off"] for r in stops]); st = np.median([r["zupt_strict"] for r in stops])
        zres = dict(n=len(stops), parked_drift_median_m=dict(off=float(off), strict=float(st)))
    else:
        zres = dict(n=0)

    # moving-outage no-harm check for the strict ZUPT
    strict_moving = {D: outage_drift(base(zupt="strict"), imu, gn, ge, gnn, D) for D in (30, 60)}
    strict_moving = {D: float(np.median(p)) if p else None for D, p in strict_moving.items()}
    harm = all(strict_moving[D] is None or drift["shipped loop"][D]["median"] is None
               or strict_moving[D] <= drift["shipped loop"][D]["median"] * 1.05 for D in (30, 60))
    if zres["n"] == 0:
        rec = "no parked span >= 20 s in this drive: park for 1-2 min with the app navigating to answer this"
    elif zres["parked_drift_median_m"]["strict"] < zres["parked_drift_median_m"]["off"] and harm:
        rec = "turn strict ZUPT ON for this phone (less drift while parked, no harm while moving)"
    else:
        rec = "keep ZUPT OFF (strict ZUPT did not reduce parked drift, or it hurt moving outages)"

    res = dict(drive=tilde_home(os.path.abspath(a.dir)), sanity=sanity, drift=drift, stops=stops,
               zupt=dict(**zres, strict_moving_median=strict_moving, recommendation=rec))
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    L = [f"# Phone drive: {name}", "",
         "Live loop (edge_engine = the phone's FusionEngine) replayed on this recording. Truth = the phone's own "
         "GNSS (a few metres of noise), so this scores the phone, not a survey reference.", "",
         "## Sanity", ""] + [f"- {k}: {v:.2f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in sanity.items()]
    L += ["", "## Simulated outages (median drift % of distance; ISRO limit 10 %)", "",
          "| config | " + " | ".join(f"{D} s" for D in DURATIONS) + " |", "|---|" + "--:|" * len(DURATIONS)]
    for nm, r in drift.items():
        L.append(f"| {nm} | " + " | ".join("-" if r[D]["median"] is None else f"{r[D]['median']:.1f} (n={r[D]['n']})"
                                           for D in DURATIONS) + " |")
    L += ["", "## Stops and ZUPT", "", f"parked spans: {zres['n']}"]
    if stops:
        L += ["", "| t0 (s) | parked (s) | drift ZUPT off (m) | drift strict ZUPT (m) | gyro std (deg/s) |", "|--:|--:|--:|--:|--:|"]
        L += [f"| {r['t0']:.0f} | {r['dur']:.0f} | {r['zupt_off']:.1f} | {r['zupt_strict']:.1f} | {r['gyro_std_dps']:.3f} |" for r in stops]
    L += ["", f"strict ZUPT on moving outages: {strict_moving}", "", f"**Recommendation:** {rec}"]
    with open(os.path.join(outdir, "REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L)); print(f"-> {outdir}")


if __name__ == "__main__":
    main()
