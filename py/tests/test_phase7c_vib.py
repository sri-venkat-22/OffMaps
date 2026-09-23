"""Phase-7c regressions: the high-rate vibration/pothole front-end.

  1. native vib == oracle VibRef: identical per-sample edges + per-window features
     (bit-exact) across the 200-400 Hz range;
  2. gate: pothole detection recall & precision high, and the pothole-rejected
     RMS is a far cleaner speed cue than the raw RMS on pothole windows.

Skipped cleanly when the native core isn't built. (No torch needed here, but
core_bridge imports it, so guard the same way.)
"""
import numpy as np
import pytest

pytest.importorskip("core_bridge")
from core_bridge import Vib
from vib_ref import VibRef
from data.highrate import highrate_accel

try:
    Vib()
    _BUILT = True
except FileNotFoundError:
    _BUILT = False

pytestmark = pytest.mark.skipif(not _BUILT, reason="native core not built (sh core/build.sh)")


@pytest.mark.parametrize("hz", [250.0, 400.0])
def test_vib_native_matches_oracle(hz):
    from test_vib_parity import _run
    hr = highrate_accel(hz=hz, seed=5)
    e_ref, w_ref = _run(VibRef(hz), hr)
    e_cpp, w_cpp = _run(Vib(hz), hr)
    assert np.array_equal(e_ref, e_cpp), "per-sample shock edges differ"
    assert np.max(np.abs(w_ref - w_cpp)) < 1e-9, "per-window features differ"


def test_pothole_detection_and_rejection():
    from phase7c_gate import run_frontend
    hr = highrate_accel(seed=2)
    events_t, win = run_frontend(hr)
    tc, spd, rms_clean, rms_raw, shock_frac, n_ev = win.T
    inj = hr.pothole_t
    recall = sum(any(abs(e - p) < 0.10 for e in events_t) for p in inj) / len(inj)
    tp = sum(any(abs(e - p) < 0.15 for p in inj) for e in events_t)
    precision = tp / len(events_t)
    assert recall >= 0.9 and precision >= 0.9

    clean_win = n_ev == 0; pot_win = n_ev > 0
    b, a = np.polyfit(rms_clean[clean_win], spd[clean_win], 1)
    err_clean = np.median(np.abs((a + b * rms_clean[pot_win]) - spd[pot_win]))
    err_raw = np.median(np.abs((a + b * rms_raw[pot_win]) - spd[pot_win]))
    assert err_clean < 2.0
    assert err_clean < 0.4 * err_raw, "rejection must remove most of the raw-RMS pothole bias"
