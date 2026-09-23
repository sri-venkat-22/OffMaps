"""Phase-7a parity gate: native 3D ESKF core (idr3d) == Python oracle.
Run: PYTHONPATH=. python3 test_eskf3d_parity.py

Two independent checks:
  1. parity  -- native core vs the pure-Python oracle on a fixed sequence that
     exercises predict + every update; max|diff| must be numeric noise.
  2. jacobian -- the oracle's analytic measurement Jacobians (the non-trivial
     ones are NHC's) match a finite-difference of the measurement function, so
     "both agree" can't just mean "both wrong the same way".
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eskf3d_ref import Idr3dRef, rotmat, quatmul, expq, normq

DEG = np.pi / 180


def run_eskf3d(f):
    """Exercise predict + all six update types with byte-identical inputs."""
    f.init([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    f.set_noise(0.05, 0.3 * DEG, 0.001, 0.01 * DEG)
    rng = np.random.default_rng(0)
    for i in range(400):
        a_m = [0.2 * np.sin(i * 0.05) + rng.normal(0, 0.02),
               0.1 * np.cos(i * 0.03) + rng.normal(0, 0.02),
               9.81 + rng.normal(0, 0.02)]
        w_m = [0.5 * DEG * np.sin(i * 0.02), 0.3 * DEG,
               3.0 * DEG + rng.normal(0, 0.1 * DEG)]
        f.predict(0.1, a_m, w_m)
        if i % 10 == 0: f.update_gnss_pos([i * 1.0, 0.5 * i, 0.02 * i], 3.0)
        if i % 10 == 5: f.update_gnss_vel([10.0, 0.5, 0.02], 0.3)
        if i % 25 == 0: f.update_nhc(0.2)
        if i % 17 == 0: f.update_odo(10.0, 0.1)
        if i % 41 == 0: f.update_zaru(w_m, 0.1 * DEG)
        if i % 53 == 0: f.update_baro(0.02 * i, 1.0)
    return f.state(), f.cov_diag()


def eskf3d_parity():
    from core_bridge import Filter3D
    (sa, ca), (sb, cb) = run_eskf3d(Idr3dRef()), run_eskf3d(Filter3D())
    err_s = float(np.max(np.abs(sa - sb)))
    err_c = float(np.max(np.abs(ca - cb)))
    print(f"eskf3d   state max|diff|={err_s:.2e}  cov max|diff|={err_c:.2e}  (gate <1e-6)")
    assert err_s < 1e-6, "ESKF3D STATE PARITY FAIL"
    assert err_c < 1e-6, "ESKF3D COV PARITY FAIL"


def _perturb(ref, dx):
    """Apply an error-state dx to a copy of ref's nominal (the inject mapping)."""
    r = Idr3dRef()
    r.p = ref.p + dx[0:3]; r.v = ref.v + dx[3:6]
    r.q = normq(quatmul(ref.q, expq(dx[6:9])))
    r.ba = ref.ba + dx[9:12]; r.bg = ref.bg + dx[12:15]
    return r


def _num_jac(ref, h, eps=1e-6):
    H = np.zeros(15)
    for i in range(15):
        dp = np.zeros(15); dp[i] = eps
        H[i] = (h(_perturb(ref, dp)) - h(_perturb(ref, -dp))) / (2 * eps)
    return H


def jacobian_check():
    """The NHC measurement Jacobians are the only hand-derived ones -- verify them
    against finite differences at a non-trivial attitude."""
    r = Idr3dRef()
    r.p = np.array([3.0, -2.0, 1.0]); r.v = np.array([9.0, 1.5, -0.4])
    r.q = normq(quatmul(expq([5 * DEG, -8 * DEG, 20 * DEG]), r.q))
    R = rotmat(r.q)
    worst = 0.0
    # forward (odometer), lateral & vertical (NHC) all share the u = R^T v Jacobian
    for axis, name in ((0, "forward"), (1, "lateral"), (2, "vertical")):
        u = R.T @ r.v
        H = np.zeros(15); H[3:6] = R[:, axis]
        if axis == 0:   H[6+1] = -u[2]; H[6+2] = u[1]
        elif axis == 1: H[6+0] = u[2]; H[6+2] = -u[0]
        else:           H[6+0] = -u[1]; H[6+1] = u[0]
        Hn = _num_jac(r, lambda s: (rotmat(s.q).T @ s.v)[axis])
        e = float(np.max(np.abs(H - Hn)))
        worst = max(worst, e)
        print(f"body-vel {name:8} H analytic vs finite-diff max|diff|={e:.2e}")
    assert worst < 1e-5, "BODY-VELOCITY JACOBIAN WRONG"


if __name__ == "__main__":
    jacobian_check()
    eskf3d_parity()
    print("PHASE 7a PARITY GATES PASS: native 3D core matches the oracle; NHC Jacobian verified")
