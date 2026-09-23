"""Phase-7c parity gate: native high-rate front-end (vib) == Python oracle
(VibRef). Streaming RMS/shock DSP must match sample-for-sample: identical
per-window {rms_clean, rms_raw, shock_frac, n_events} and identical per-sample
edge flags over a fixed high-rate sequence with injected potholes.

Run: PYTHONPATH=. python3 test_vib_parity.py
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vib_ref import VibRef
from data.highrate import highrate_accel


def _run(v, hr, window_s=1.0):
    step = int(round(window_s * hr.hz)); N = len(hr.t)
    edges = np.zeros(N, int); wins = []
    for i in range(N):
        edges[i] = v.push(hr.acc[i, 0], hr.acc[i, 1], hr.acc[i, 2], 1.0 / hr.hz)
        if (i + 1) % step == 0:
            wins.append(v.window())
    return edges, np.array(wins)


def vib_parity(hz=250.0, params=None):
    from core_bridge import Vib
    hr = highrate_accel(hz=hz, seed=5)
    ref, cpp = VibRef(hz), Vib(hz)
    if params:
        ref.set_params(**params); cpp.set_params(**params)
    e_ref, w_ref = _run(ref, hr)
    e_cpp, w_cpp = _run(cpp, hr)
    edge_mismatch = int(np.sum(e_ref != e_cpp))
    win_err = float(np.max(np.abs(w_ref - w_cpp)))
    tag = f"{hz:.0f}Hz" + ("" if not params else f" {params}")
    print(f"vib {tag:26} edge mismatches={edge_mismatch}  window max|diff|={win_err:.2e}")
    assert edge_mismatch == 0, "VIB EDGE PARITY FAIL"
    assert win_err < 1e-9, "VIB WINDOW PARITY FAIL"


if __name__ == "__main__":
    vib_parity(250.0)
    vib_parity(400.0)                                       # top of the 200-400 Hz range
    vib_parity(250.0, params=dict(shock_k=10.0, hp_fc=5.0, ema_tau=0.3))
    print("PHASE 7c PARITY GATES PASS: native vibration front-end matches the oracle")
