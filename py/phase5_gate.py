"""Phase-5 exit gate: on a straight tunnel corridor, map matching drives
cross-track < 3 m WITHOUT touching along-track (the anisotropy test).
Run: PYTHONPATH=. python3 phase5_gate.py --out ../out"""
from __future__ import annotations
import os, sys, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from data.io_vnbd import synth_drive
from data.outage import make_outages
from eval.metrics import score_outage
import core_bridge  # registers eskf / eskf_map
from core_bridge import MapMatcher, _run, _MAPS
from eval.models import REGISTRY


def _decompose(drive, o, pred):
    """Return (along, cross) end-error components along the road (true heading)."""
    pe, pn, _, _ = pred
    h = drive.heading[o.i1 - 1]
    ee, en = pe[-1] - drive.e[o.i1-1], pn[-1] - drive.n[o.i1-1]
    fwd = np.array([np.sin(h), np.cos(h)]); left = np.array([-np.cos(h), np.sin(h)])
    return abs(ee*fwd[0] + en*fwd[1]), abs(ee*left[0] + en*left[1])


def gate(out):
    drives = [synth_drive(f"tunnel{i}", 600, seed=300+i, straight=True) for i in range(4)]
    nomap_c, map_c, nomap_a, map_a = [], [], [], []
    example = None
    for d in drives:
        mm = MapMatcher(); mm.add_way(d.e[::5], d.n[::5], tunnel=1)   # tunnel=yes -> corridor
        for o in make_outages(d, seed=0):
            if o.duration_s != 60:
                continue
            p_no = _run(d, o, None)
            p_mp = _run(d, o, mm)
            a0, c0 = _decompose(d, o, p_no); a1, c1 = _decompose(d, o, p_mp)
            nomap_a.append(a0); nomap_c.append(c0); map_a.append(a1); map_c.append(c1)
            if example is None: example = (d, o, p_no, p_mp)
    mc, mnc = np.median(map_c), np.median(nomap_c)
    ma, mna = np.median(map_a), np.median(nomap_a)
    print(f"[gate5] cross-track median: no-map={mnc:.1f} m  ->  map={mc:.2f} m  (need <3 m)")
    print(f"[gate5] along-track median: no-map={mna:.1f} m  ->  map={ma:.1f} m  "
          f"(need ~unchanged; not worse)")
    # plot: one example outage, cross & along error vs time, with/without map
    d, o, p_no, p_mp = example
    t = d.t[o.i0:o.i1] - d.t[o.i0]
    def series(p):
        h = d.heading[o.i0:o.i1]
        ee = p[0] - d.e[o.i0:o.i1]; en = p[1] - d.n[o.i0:o.i1]
        along = ee*np.sin(h) + en*np.cos(h); cross = -ee*np.cos(h) + en*np.sin(h)
        return along, cross
    a_no, c_no = series(p_no); a_mp, c_mp = series(p_mp)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(t, np.abs(c_no), label="no map"); ax1.plot(t, np.abs(c_mp), label="+ map")
    ax1.axhline(3, ls="--", c="r"); ax1.set(xlabel="s", ylabel="|cross-track| m", title="cross-track collapses"); ax1.legend()
    ax2.plot(t, a_no, label="no map"); ax2.plot(t, a_mp, label="+ map")
    ax2.set(xlabel="s", ylabel="along-track error m", title="along-track unchanged"); ax2.legend()
    fig.tight_layout(); fig.savefig(f"{out}/gate5_map.png", dpi=110); plt.close(fig)
    assert mc < 3.0, "GATE 5 FAIL: cross-track not < 3 m"
    assert ma <= mna * 1.25 + 2.0, "GATE 5 FAIL: along-track degraded"
    print("[gate5] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../out")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    gate(a.out)
