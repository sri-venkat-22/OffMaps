"""Phase-3 exit gate 1 (native ESKF core == Python oracle), locked into the suite.

The parity gate lived only in py/test_eskf_parity.py, a __main__ script outside
`pytest tests` -- so the headline Phase-3 claim ("the object file shipped to
Android/edge IS the researched filter, proven not asserted") was not itself a
regression test, while SpeedCal parity already was (test_speedcal_deming.py).
This pins it over EVERY wired observable, including idr_update_gnss_vel -- which
was in the C ABI and the README's observables table but had no oracle twin and
no bridge method until now, i.e. it shipped unproven.

Skipped cleanly when torch is absent (core_bridge registers the nn-backed eskf
model on import) or when the native core isn't built.
"""
import numpy as np
import pytest

pytest.importorskip("core_bridge")          # imports torch via the eskf registration
from core_bridge import Filter
from eskf_ref import EskfRef
from test_eskf_parity import run_eskf        # the one canonical all-observables sequence

try:
    Filter()                                # forces the ctypes _load(); raises if unbuilt
    _BUILT = True
except FileNotFoundError:
    _BUILT = False

pytestmark = pytest.mark.skipif(not _BUILT, reason="native core not built (sh core/build.sh)")


def test_native_core_matches_oracle():
    """Gate 1. Over a 600-step drive exercising predict + all 7 updates, the
    native filter's final state must match the Python oracle to numeric noise
    (<1e-9; the README quotes ~1e-13). A real divergence means the shipped object
    file no longer matches the researched filter -- change one, change both."""
    a, b = run_eskf(EskfRef()), run_eskf(Filter())
    err = float(np.max(np.abs(a - b)))
    assert err < 1e-9, f"ESKF native vs oracle diverged: max|diff|={err:.2e}"


def test_map_keep_speed_has_parity_and_never_moves_speed():
    """idr_set_map_keep_speed: road updates with the gain's speed row zeroed and a
    Joseph-form covariance. Same all-observables sequence with it ON must match the
    oracle, and a road update alone must leave v exactly where it was."""
    a, b = EskfRef(), Filter()
    a.set_map_keep_speed(True); b.set_map_keep_speed(True)
    err = float(np.max(np.abs(run_eskf(a) - run_eskf(b))))
    assert err < 1e-9, f"keep-speed parity failed: max|diff|={err:.2e}"
    for f in (EskfRef(), Filter()):
        f.set_map_keep_speed(True); f.init(0.0, 0.0, 0.3, 10.0)
        for _ in range(20):
            f.predict(0.1, 0.0)
        v0 = f.state()[3]
        f.update_crosstrack(-1.0, 0.0, 5.0, 1.0); f.update_heading(0.0, np.radians(3))
        assert f.state()[3] == v0
        assert f.state()[0] != 0.0 or f.state()[2] != 0.3     # the lane/heading did move


def test_gnss_vel_update_has_parity():
    """idr_update_gnss_vel in isolation: sequential scalar v then psi. It was the
    one wired observable with no oracle twin, so pin it on its own for a clear
    failure signal, deterministically (no rng)."""
    def seq(f):
        f.init(0.0, 0.0, np.radians(10), 8.0)
        for i in range(200):
            f.predict(0.1, np.radians(2.0))
            if i % 5 == 0:
                f.update_gnss_vel(9.0, np.radians(10) + 0.01 * i, 0.3, np.radians(4))
        return f.state()
    err = float(np.max(np.abs(seq(EskfRef()) - seq(Filter()))))
    assert err < 1e-9, f"gnss_vel parity failed: max|diff|={err:.2e}"
