"""Phase-7b exit gate: at a Y-fork, greedy nearest-road wrong-snaps to the wrong
branch under position noise while the branches are still close; Viterbi/HMM
decodes the whole trajectory and stays on the branch the vehicle actually took.

Scenario: a trunk road that forks into branch A (straight) and branch B
(diverging east). The vehicle takes A. For the first tens of metres past the
fork A and B are only a few metres apart, so GNSS-scale noise makes B the nearest
road on many steps -- a greedy matcher flickers onto the wrong lane there.

Run: PYTHONPATH=. python3 phase7b_gate.py --out ../out
"""
from __future__ import annotations
import os, sys, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


# way indices
TRUNK, A, B = 0, 1, 2


def build_scene(seed=0, sigma_e=4.0, sigma_n=2.0):
    """Ways + a noisy estimate track for a vehicle that drives the trunk then A."""
    ways = [
        (np.array([0.0, 0.0]),   np.array([-80.0, 0.0]), TRUNK),  # trunk, south->fork
        (np.array([0.0, 0.0]),   np.array([0.0, 200.0]), A),      # branch A: straight north
        (np.array([0.0, 30.0]),  np.array([0.0, 200.0]), B),      # branch B: diverges east
    ]
    ds = 2.0
    trunk_y = np.arange(-80, 0, ds)
    a_y = np.arange(0, 200 + ds, ds)
    true_e = np.concatenate([np.zeros(len(trunk_y)), np.zeros(len(a_y))])
    true_n = np.concatenate([trunk_y, a_y])
    true_way = np.concatenate([np.full(len(trunk_y), TRUNK), np.full(len(a_y), A)])
    rng = np.random.default_rng(seed)
    e = true_e + rng.normal(0, sigma_e, len(true_e))
    n = true_n + rng.normal(0, sigma_n, len(true_n))
    psi = np.zeros(len(e)) + rng.normal(0, np.radians(3), len(e))   # heading ~ north
    return ways, e, n, psi, true_e, true_n, true_way


def _build_matcher(cls, ways):
    m = cls()
    for e, n, w in ways:
        m.add_way(e, n)
    m.add_edge(TRUNK, A); m.add_edge(TRUNK, B)     # fork connectivity; A and B NOT adjacent
    return m


def gate(out_dir):
    from map_match_ref import MapMatcherRef
    ways, e, n, psi, te, tn, true_way = build_scene(seed=1)
    m = _build_matcher(MapMatcherRef, ways)

    greedy = m.greedy_seq(e, n, psi)
    vit = m.match_seq(e, n, psi)["way"]

    valid = greedy >= 0                            # steps with any candidate
    g_correct = float(np.mean(greedy[valid] == true_way[valid]))
    v_correct = float(np.mean(vit[valid] == true_way[valid]))
    # wrong-snaps onto branch B in the ambiguous zone (0 <= y <= 40)
    zone = (tn >= 0) & (tn <= 40)
    g_wrongB = int(np.sum(greedy[zone] == B))
    v_wrongB = int(np.sum(vit[zone] == B))

    print(f"[gate7b] Y-fork, {len(e)} steps, sigma_pos=(4,2) m")
    print(f"[gate7b] on-correct-road: greedy={100*g_correct:.0f}%  ->  Viterbi={100*v_correct:.0f}%")
    print(f"[gate7b] wrong-snaps onto branch B near the fork: greedy={g_wrongB}  ->  Viterbi={v_wrongB}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    for (we, wn, wid), col in zip(ways, ("#888", "#1f77b4", "#d62728")):
        ax1.plot(we, wn, color=col, lw=3, alpha=0.5,
                 label=["trunk", "A (taken)", "B"][wid])
    ax1.scatter(e[greedy == B], n[greedy == B], c="red", s=14, label="greedy snap->B")
    ax1.scatter(e, n, c="k", s=3, alpha=0.3)
    ax1.set(xlabel="E (m)", ylabel="N (m)", title="greedy wrong-snaps to B"); ax1.legend(fontsize=8); ax1.axis("equal")
    for (we, wn, wid), col in zip(ways, ("#888", "#1f77b4", "#d62728")):
        ax2.plot(we, wn, color=col, lw=3, alpha=0.5)
    ax2.scatter(e[vit == B], n[vit == B], c="red", s=14, label="Viterbi snap->B")
    ax2.scatter(e, n, c="k", s=3, alpha=0.3)
    ax2.set(xlabel="E (m)", ylabel="N (m)", title=f"Viterbi stays on A ({v_wrongB} wrong)"); ax2.legend(fontsize=8); ax2.axis("equal")
    fig.tight_layout(); fig.savefig(f"{out_dir}/gate7b_viterbi.png", dpi=110); plt.close(fig)

    assert g_wrongB > 0, "scenario should make greedy wrong-snap"
    assert v_wrongB == 0, f"Viterbi should not wrong-snap to B, got {v_wrongB}"
    assert v_correct >= 0.98, f"Viterbi on-road fraction too low: {v_correct:.2f}"
    assert v_correct > g_correct + 0.05, "Viterbi must beat greedy meaningfully"
    print("[gate7b] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../out")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    gate(a.out)
