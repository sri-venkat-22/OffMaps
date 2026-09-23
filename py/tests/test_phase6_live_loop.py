"""Phase-6 live-loop regression. Locks the on-device fusion behaviour that
com.offmaps.nav.FusionEngine implements, run off-device against the SAME embedded
libidr core (ctypes) and ONNX SpeedNet path. Mirrors py/phase6_check.py.

Pins the two claims that are pure logic (device-independent):
  1. the live fusion runs and tracks GNSS when it's healthy;
  2. the outage flag is honoured against the LIVE filter -- masked GNSS makes the
     estimate dead-reckon (diverge from truth), and it re-converges once GNSS
     returns. Toggling the flag must change the outcome, not just a replay.

Both shipped profiles run (see test_phase6_shipped_model.py). nn_real on this
SYNTHETIC drive is deliberately off-distribution (trained on real drives; it
barely correlates with the synthetic speed), which makes it the lockout stress
test: before the real-data fix, a net that disagreed with Doppler made the spoof
check reject every fix and GNSS never came back -- with GNSS healthy (282 m median
error here) and again after an outage. The same thresholds must hold for both.

Skipped when the native core isn't built (same guard as the other core tests).
"""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("core_bridge")
from phase6_check import run, cal_apply            # noqa: E402
from data.io_vnbd import synth_drive               # noqa: E402

# Same fixed scenario as phase6_check.main -> deterministic (fixed seed + checkpoint).
DRIVE = synth_drive("phase6", duration=300, seed=11)
OUTAGE = (150.0, 210.0)                            # 60 s masked span
I_END = int(OUTAGE[1] * 10) - 1                    # sample at outage end
I_RET = min(len(DRIVE) - 1, I_END + 100)           # 10 s after GNSS returns


@pytest.fixture(scope="module", params=["nn_real", "nn"])
def runs(request):
    err_on, _, _, _ = run(DRIVE, outage=None, profile=request.param)        # flag OFF: full GNSS fusion
    err_off, fused_off, _, _ = run(DRIVE, outage=OUTAGE, profile=request.param)  # flag ON: mask 60 s
    return err_on, err_off, fused_off


def test_live_fusion_tracks_gnss(runs):
    err_on, _, _ = runs
    assert np.all(np.isfinite(err_on))
    assert np.median(err_on) < 5.0, f"fusion should track GNSS, median err {np.median(err_on):.2f} m"


def test_outage_flag_honoured_against_live_filter(runs):
    err_on, err_off, fused_off = runs
    assert np.all(np.isfinite(fused_off))
    # masked -> the live filter is fed no GNSS and dead-reckons: real drift accrues,
    # and it dwarfs the fusion-ON error at the same instant. This is the flag working
    # against the LIVE filter, not a replay artefact.
    assert err_off[I_END] > 5.0, f"no dead-reckoning drift accrued during outage ({err_off[I_END]:.1f} m)"
    assert err_off[I_END] > 3 * max(err_on[I_END], 1.0), \
        f"outage flag had no effect: off={err_off[I_END]:.1f} m vs on={err_on[I_END]:.1f} m"


def test_reconverges_after_gnss_returns(runs):
    _, err_off, _ = runs
    assert err_off[I_RET] < err_off[I_END], "did not re-converge after GNSS returned"
    assert err_off[I_RET] < 5.0, f"still off {err_off[I_RET]:.1f} m 10 s after GNSS returned"


def test_degenerate_self_cal_fit_is_not_applied():
    """A net that doesn't track speed gives a Deming fit like k=72, c=-824 (measured:
    nn_real on this drive). Applying it would fling the dead-reckoned speed; the loop
    must fall back to the net's own speed instead."""
    assert cal_apply(10.0, 1.3, 0.4) == pytest.approx(13.4)
    assert cal_apply(10.0, 71.8, -824.0) == 10.0
    assert cal_apply(10.0, -10.5, 43.9) == 10.0
