"""One-command demo (run.sh calls this): the live loop through GNSS outages.

    PYTHONPATH=. python3 demo.py [--data ~/OffMaps-data/IO-VNBD-sync] [--out ../out/demo]

With the IO-VNBD data (see REALDATA.md): the longest validation drive, one 60 s outage
every 4 min, run through the four ablation stages (ablation.py) and scored against the
survey-grade reference. Writes a summary and a trajectory plot of one outage.

Without it: a synthetic drive (data/synth_rig.py, Hyderabad, MEMS phone IMU at 100 Hz)
through the edge-engine CLI with a 60 s outage. That checks the PLUMBING only (the CSV
path, dead-reckoning, recovery). SpeedNet learned real phone vibration, which the rig
does not reproduce, so no accuracy claim is made from it.
"""
from __future__ import annotations
import argparse, math, os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def real(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from osm_layers import read_roads_bin
    from data.iovnbd_sync import load_sync_dir
    from model.train_real import split_drives
    from model.fusion_head import load_head
    from edge_engine import run
    from phase8_eval import streams
    from ablation import factory, LABEL, STAGES, ROADS
    d = max(split_drives(load_sync_dir(data, verbose=False))["val"], key=len)
    roads = read_roads_bin(ROADS, with_class=True)
    head = load_head(os.path.join(HERE, "model", "fusion_head.pt"))
    imu, gn = streams(d)
    outs = [(a, a + 60.0) for a in np.arange(d.t[0] + 180, d.t[-1] - 70, 240.0)]
    print(f"IO-VNBD validation drive {d.vehicle_id}: {(d.t[-1] - d.t[0]) / 60:.0f} min, "
          f"{len(outs)} GNSS outages of 60 s (the models never trained on this drive)\n")
    tracks, rows = {}, []
    for s in STAGES:
        t0 = time.perf_counter()
        o = run(factory(s, head, roads)(), imu, gn, outs)
        err = np.hypot(o[:, 1] - d.e, o[:, 2] - d.n)
        pct = []
        for a, b in outs:
            i0, i1 = np.searchsorted(d.t, [a, b]); i1 -= 1
            dist = np.trapezoid(d.speed[i0:i1], d.t[i0:i1])
            if dist >= 300:
                pct.append(100 * err[i1] / dist)
        tracks[s] = o
        rows.append((s, np.median(pct), np.mean(np.asarray(pct) < 10), len(pct), time.perf_counter() - t0))
    print(f"{'stage':38s} {'median drift':>12s} {'< 10 %':>7s}")
    for s, med, u10, n, dt in rows:
        print(f"{LABEL[s]:38s} {med:11.1f} % {100 * u10:6.0f} %")
    print(f"\n{rows[0][3]} scored outages on one drive: stage-to-stage differences here are within noise."
          f"\nThe ablation over ~1,400 outages (out/ablation/ablation.md, python ablation.py) is the one to read."
          f"\n~{len(d) / rows[-1][4]:,.0f} samples/s for the full loop in Python.")

    # one outage, zoomed: the most representative (median drift in the shipped stage)
    a, b = outs[len(outs) // 2]
    i0, i1 = np.searchsorted(d.t, [a - 30, b + 20])
    j0, j1 = np.searchsorted(d.t, [a, b])
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(d.e[i0:i1], d.n[i0:i1], color="black", lw=2.2, label="truth (survey GNSS)")
    ax.plot(d.e[j0:j1], d.n[j0:j1], color="black", lw=6, alpha=0.12, label="GNSS denied (60 s)")
    col = {"physics": "#9aa3ad", "speednet": "#5b8def", "head": "#2459c9", "map": "#0f9d74"}
    for s in STAGES:
        o = tracks[s]
        ax.plot(o[i0:i1, 1], o[i0:i1, 2], color=col[s], lw=1.4, label=LABEL[s])
    ax.set_aspect("equal"); ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
    ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("One 60 s outage, IO-VNBD validation drive")
    fig.tight_layout(); p = os.path.join(out, "demo_outage.png"); fig.savefig(p, dpi=150)
    print("plot ->", os.path.relpath(p))


def synthetic(out):
    from data.synth_rig import synth_rig, write_logger_csv
    from edge_engine import main as cli
    import pandas as pd
    rig = synth_rig(600, 100, seed=3, grade="mems")
    d = tempfile.mkdtemp(); write_logger_csv(rig, d)
    track = os.path.join(out, "demo_track.csv")
    print("No IO-VNBD data found: synthetic plumbing check (see REALDATA.md to get the real data).\n")
    cli(["--dir", d, "--out", track, "--outage", "300:360"])
    o = pd.read_csv(track)
    err = np.hypot(o.e.to_numpy() - rig["e"], o.n.to_numpy() - rig["n"])
    hz = 100
    print(f"\nwith GNSS: median error {np.median(err[:300 * hz]):.1f} m | dead-reckoning flag during the outage: "
          f"{o.dead_reckoning[330 * hz]} | 15 s after GNSS returns: {err[375 * hz]:.1f} m")
    print("The synthetic rig checks the pipeline, not accuracy: accuracy numbers come from real drives only.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.environ.get("OFFMAPS_IOVNBD", os.path.expanduser("~/OffMaps-data/IO-VNBD-sync")))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "out", "demo"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if os.path.isdir(a.data):
        real(a.data, a.out)
    else:
        synthetic(a.out)


if __name__ == "__main__":
    main()
