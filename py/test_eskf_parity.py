"""Phase-3/4 parity gate: native core == Python oracle on fixed sequences.
Run: PYTHONPATH=. python3 test_eskf_parity.py"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eskf_ref import EskfRef
from core_bridge import Filter, SpeedCal
from speed_cal_ref import SpeedCalRef


def run_eskf(f):
    """Exercise EVERY wired observable (predict + all 7 updates) so parity covers
    the whole C ABI, not a subset. rng is re-seeded per run, so the oracle and the
    native core see byte-identical inputs; any divergence is pure numeric drift."""
    f.init(10.0, -5.0, np.radians(30), 12.0)
    rng = np.random.default_rng(0)
    for i in range(600):
        gz = np.radians(rng.normal(0, 8))
        f.predict(0.1, gz)
        if i % 10 == 0: f.update_speed(12 + rng.normal(0, 1), 0.8)
        if abs(gz) > np.radians(10): f.update_curvature(0.5 * gz * 12, gz, 2.0)
        if i % 137 == 0: f.update_zupt(gz)
        if i % 50 == 0: f.update_gnss_pos(10 + i * 0.1, -5 + i * 0.05, 3.0)
        if i % 23 == 0: f.update_gnss_vel(12 + rng.normal(0, 0.5), np.radians(30) + 0.02 * i, 0.3, np.radians(5))
        if i % 31 == 0: f.update_crosstrack(-np.cos(0.5), np.sin(0.5), rng.normal(0, 0.3), 0.5)
        if i % 41 == 0: f.update_heading(np.radians(30) + 0.02 * i, np.radians(3))
    return f.state()


def eskf_parity():
    a, b = run_eskf(EskfRef()), run_eskf(Filter())
    err = np.max(np.abs(a - b))
    print(f"eskf     max|diff|={err:.2e}  (gate <1e-3)")
    assert err < 1e-3, "ESKF PARITY FAIL"


def speed_cal_parity(lam=None):
    rng = np.random.default_rng(1); cpp, ref = SpeedCal(), SpeedCalRef()
    if lam is not None:
        cpp.set_lambda(lam); ref.set_lambda(lam)
    for _ in range(300):
        v = rng.uniform(3, 25); vn = v * 0.9 + rng.normal(0, 0.3); vd = v + rng.normal(0, 0.05)
        cpp.push(vn, vd); ref.push(vn, vd)
    kc, cc, _ = cpp.fit(); kr, cr, _ = ref.fit()
    err = max(abs(kc - kr), abs(cc - cr))
    tag = "OLS" if lam is None else f"Deming(lam={lam:g})"
    print(f"speed_cal {tag:16} max|diff|={err:.2e}  (gate <1e-9)")
    assert err < 1e-9, "SPEED_CAL PARITY FAIL"


if __name__ == "__main__":
    eskf_parity()
    speed_cal_parity()                 # OLS path (lambda = +inf)
    speed_cal_parity(lam=0.0044)       # Deming path (errors-in-variables)
    print("PARITY GATES PASS: native core matches Python oracles")
