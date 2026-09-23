"""SpeedCal's Doppler self-calibration regresses Doppler on the NN speed. The
NN speed is a noisy regressor, so ordinary LS dilutes the recovered scale k
toward the noise floor (errors-in-variables). Deming regression, fed the NN
sigma as the regressor-noise variance, recovers the true scale. These pin that,
and pin C++/oracle parity on the Deming path.

Skipped when the native core is not built.
"""
import numpy as np
import pytest

core_bridge = pytest.importorskip("core_bridge")
from core_bridge import SpeedCal
from speed_cal_ref import SpeedCalRef


def _profile(seed=0):
    rng = np.random.default_rng(seed)
    v = np.clip(15 + 10 * np.sin(np.arange(600) / 30) + rng.normal(0, 2, 600), 0.1, None)
    return v, rng


def test_ols_is_the_default_and_dilutes():
    """Untouched behaviour: no set_lambda -> OLS -> k under-recovers under noise."""
    v, rng = _profile()
    gap = 0.9
    sc = SpeedCal()
    for vt in v:
        sc.push(vt * gap + rng.normal(0, 1.5), vt)
    k, c, _ = sc.fit()
    assert k * gap < 0.97, "OLS should dilute (attenuation well below 1)"


def test_deming_recovers_the_scale_under_noise():
    v, rng = _profile()
    gap = 0.9
    sig_nn = 1.5
    sc = SpeedCal()
    sc.set_lambda(0.1 ** 2 / sig_nn ** 2)          # Doppler sigma^2 / NN sigma^2
    for vt in v:
        sc.push(vt * gap + rng.normal(0, sig_nn), vt)
    k, c, _ = sc.fit()
    assert 0.98 <= k * gap <= 1.03, f"Deming should recover k=1/gap, got attenuation {k*gap:.3f}"


def test_deming_equals_ols_when_regressor_is_noiseless():
    """lambda -> inf (or no noise) must reduce to OLS -- a safety property."""
    v, rng = _profile()
    gap = 0.9
    a, b = SpeedCal(), SpeedCal()
    b.set_lambda(np.inf)                            # explicit +inf == OLS
    for vt in v:
        x = vt * gap                               # noiseless regressor
        a.push(x, vt); b.push(x, vt)
    ka = a.fit()[0]; kb = b.fit()[0]
    assert abs(ka - kb) < 1e-9
    assert abs(ka - 1 / gap) < 1e-6


@pytest.mark.parametrize("lam", [np.inf, 0.05, 0.0044, 1.0])
def test_cpp_matches_oracle_on_both_paths(lam):
    rng = np.random.default_rng(1)
    cpp, ref = SpeedCal(), SpeedCalRef()
    cpp.set_lambda(lam); ref.set_lambda(lam)
    for _ in range(300):
        v = rng.uniform(3, 25)
        vn = v * 0.9 + rng.normal(0, 0.3); vd = v + rng.normal(0, 0.05)
        cpp.push(vn, vd); ref.push(vn, vd)
    kc, cc, ec = cpp.fit(); kr, cr, er = ref.fit()
    assert ec == er
    assert abs(kc - kr) < 1e-9 and abs(cc - cr) < 1e-9


def test_scale_only_branch_also_debiases():
    """Constant-ish speed (not excited) uses the through-origin fit; Deming must
    still de-bias it and never NaN."""
    rng = np.random.default_rng(3)
    v = np.full(300, 12.0) + rng.normal(0, 0.5, 300)   # low excitation
    gap = 0.9
    sc = SpeedCal(); sc.set_lambda(0.1 ** 2 / 1.5 ** 2)
    for vt in v:
        sc.push(vt * gap + rng.normal(0, 1.5), vt)
    k, c, excited = sc.fit()
    assert np.isfinite(k) and np.isfinite(c)


def test_zero_weight_buffer_keeps_the_fit():
    """Pushes weighted by GNSS trust 0 (the emulator: no satellites) made the fit
    0/0 = NaN, which the app displayed as k=NaN. Both twins must keep their fit."""
    from core_bridge import SpeedCal
    from speed_cal_ref import SpeedCalRef
    for S in (SpeedCal(), SpeedCalRef()):
        for i in range(40):
            S.push(5.0 + 0.1 * i, 6.0 + 0.1 * i, 0.0)
        k, c, _ = S.fit()
        assert (k, c) == (1.0, 0.0), type(S).__name__
