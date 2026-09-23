"""Phase-7a regressions: the 3D 16-state ESKF core.

Three locks, mirroring the house pattern (parity + honest gate):
  1. native idr3d == Python oracle on a fixed all-updates sequence (~1e-13);
  2. the hand-derived body-velocity Jacobians match finite differences;
  3. on a helical parking ramp the 3D core decisively beats the planar core
     and recovers altitude the planar core has no state for.

Skipped cleanly when torch is absent (core_bridge imports it) or the native
core isn't built.
"""
import numpy as np
import pytest

pytest.importorskip("core_bridge")
from core_bridge import Filter3D
from eskf3d_ref import Idr3dRef
from test_eskf3d_parity import run_eskf3d, jacobian_check

try:
    Filter3D()
    _BUILT = True
except FileNotFoundError:
    _BUILT = False

pytestmark = pytest.mark.skipif(not _BUILT, reason="native core not built (sh core/build.sh)")


def test_eskf3d_native_matches_oracle():
    (sa, ca), (sb, cb) = run_eskf3d(Idr3dRef()), run_eskf3d(Filter3D())
    es = float(np.max(np.abs(sa - sb)))
    ec = float(np.max(np.abs(ca - cb)))
    assert es < 1e-6, f"3D state parity diverged: {es:.2e}"
    assert ec < 1e-6, f"3D cov parity diverged: {ec:.2e}"


def test_eskf3d_jacobians_are_correct():
    jacobian_check()          # asserts internally (finite-diff vs analytic)


def test_eskf3d_beats_planar_on_ramp():
    from phase7a_gate import run_planar, run_3d, _err
    from data.synth3d import helix_ramp
    r = helix_ramp(v=5.0, grade_deg=12.0, radius_h=12.0, duration=45.0, seed=3)
    _, ph_f = _err(r.p, run_planar(r))
    _, e3_f = _err(r.p, run_3d(r))
    assert e3_f[-1] < 6.0, f"3D end error too large: {e3_f[-1]:.2f} m"
    assert ph_f[-1] > 30.0, f"planar should fail on the climb: {ph_f[-1]:.1f} m"
    assert e3_f[-1] < 0.2 * ph_f[-1], "3D must decisively beat planar on the ramp"
