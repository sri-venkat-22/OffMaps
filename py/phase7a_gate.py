"""Phase-7a exit gate: on a genuinely-3D helical parking ramp (a GNSS outage --
you're inside the garage), the full 3D 16-state ESKF tracks position AND altitude
while the planar 5-state core cannot represent the climb at all and over-travels
horizontally by 1/cos(grade). Both filters get the SAME aiding (wheel speed +
gravity-projected yaw); the only difference is that the 3D core knows the geometry.

Run: PYTHONPATH=. python3 phase7a_gate.py --out ../out
"""
from __future__ import annotations
import os, sys, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from data.synth3d import helix_ramp
from core_bridge import Filter, Filter3D

DEG = np.pi / 180


def run_planar(r):
    """Planar 5-state: wheel speed integrated along gravity-projected yaw. No altitude."""
    f = Filter()
    f.set_noise(0.5 * DEG, 0.02 * DEG, 0.5)
    psi0 = np.arctan2(r.v[0, 0], r.v[0, 1])          # course (0=north,+east)
    f.init(r.p[0, 0], r.p[0, 1], psi0, r.speed[0])
    hz = 1.0 / (r.t[1] - r.t[0]); step = int(round(hz))
    N = len(r.t); out = np.zeros((N, 3))
    for i in range(N):
        dt = r.t[i] - r.t[i-1] if i else (r.t[1] - r.t[0])
        f.predict(dt, r.yawrate[i])
        if i % step == 0:
            f.update_speed(r.speed[i], 0.1)
        st = f.state(); psidot = r.yawrate[i] - st[4]
        if abs(psidot) > 10 * DEG:
            f.update_curvature(r.a_lat[i], psidot, 2.0)
        st = f.state(); out[i] = st[0], st[1], 0.0    # planar has no vertical state
    return out


def run_3d(r):
    """Full 3D 16-state: IMU propagation + wheel-speed odometer + NHC."""
    f = Filter3D()
    f.set_noise(0.08, 0.4 * DEG, 0.002, 0.02 * DEG)
    f.init(r.p[0], r.v[0], r.q[0])
    N = len(r.t); out = np.zeros((N, 3))
    for i in range(N):
        dt = r.t[i] - r.t[i-1] if i else (r.t[1] - r.t[0])
        f.predict(dt, r.a_m[i], r.w_m[i])
        f.update_odo(r.speed[i], 0.1)                 # wheel speed = body-forward vel
        f.update_nhc(0.1)                             # lateral & vertical body vel = 0
        st = f.state(); out[i] = st[0], st[1], st[2]
    return out


def _err(true, est):
    d = est - true
    horiz = np.hypot(d[:, 0], d[:, 1])
    full = np.linalg.norm(d, axis=1)
    return horiz, full


def gate(out_dir):
    r = helix_ramp(v=5.0, grade_deg=12.0, radius_h=12.0, duration=45.0, seed=3)
    true = r.p
    planar = run_planar(r)
    eskf3d = run_3d(r)

    ph_h, ph_f = _err(true, planar)
    e3_h, e3_f = _err(true, eskf3d)

    # no-noise reconstruction sanity: clean IMU must integrate back to the path
    r0 = helix_ramp(v=5.0, grade_deg=12.0, radius_h=12.0, duration=45.0, seed=3,
                    accel_bias=0.0, gyro_bias_deg=0.0, accel_noise=0.0, gyro_noise_deg=0.0)
    clean = run_3d(r0)
    _, clean_f = _err(r0.p, clean)

    climb = true[-1, 2]
    alt_err_3d = abs(eskf3d[-1, 2] - climb)
    print(f"[gate7a] helical ramp: grade={r.grade_deg:.0f}deg, climb={climb:.1f} m, "
          f"{r.yawrate[0]*r.t[-1]/(2*np.pi):.1f} turns, {r.t[-1]:.0f} s outage")
    print(f"[gate7a] end 3D error : planar={ph_f[-1]:6.1f} m   3D={e3_f[-1]:5.2f} m")
    print(f"[gate7a] end horiz err: planar={ph_h[-1]:6.1f} m   3D={e3_h[-1]:5.2f} m")
    print(f"[gate7a] end altitude : truth={climb:.1f} m  planar=0.0 m (no state)  "
          f"3D={eskf3d[-1,2]:.1f} m  (3D alt err {alt_err_3d:.2f} m)")
    print(f"[gate7a] clean-IMU reconstruction end error = {clean_f[-1]:.2f} m")

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(true[:, 0], true[:, 1], "k-", lw=2, label="truth")
    ax1.plot(planar[:, 0], planar[:, 1], "r--", label="planar")
    ax1.plot(eskf3d[:, 0], eskf3d[:, 1], "g-", label="3D")
    ax1.set(xlabel="E (m)", ylabel="N (m)", title="horizontal track"); ax1.legend(); ax1.axis("equal")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(r.t, true[:, 2], "k-", lw=2, label="truth")
    ax2.plot(r.t, planar[:, 2], "r--", label="planar (0)")
    ax2.plot(r.t, eskf3d[:, 2], "g-", label="3D")
    ax2.set(xlabel="t (s)", ylabel="altitude (m)", title="altitude vs time"); ax2.legend()
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(r.t, ph_f, "r--", label="planar 3D err")
    ax3.plot(r.t, e3_f, "g-", label="3D err")
    ax3.set(xlabel="t (s)", ylabel="3D position error (m)", title="error grows"); ax3.legend()
    fig.tight_layout(); fig.savefig(f"{out_dir}/gate7a_ramp.png", dpi=110); plt.close(fig)

    assert clean_f[-1] < 2.0, f"clean-IMU 3D reconstruction should be exact-ish, got {clean_f[-1]:.2f} m"
    assert e3_f[-1] < 6.0, f"3D end error too large: {e3_f[-1]:.2f} m"
    assert ph_f[-1] > 30.0, f"planar should fail on the climb: {ph_f[-1]:.1f} m"
    assert e3_f[-1] < 0.2 * ph_f[-1], "3D must decisively beat planar"
    assert alt_err_3d < 3.0, f"3D altitude error too large: {alt_err_3d:.2f} m"
    print("[gate7a] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../out")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    gate(a.out)
