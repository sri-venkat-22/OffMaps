"""Genuinely-3D synthetic drive: a helical parking-garage ramp.

The planar 5-state core assumes all motion is in the horizontal plane and has no
vertical state at all. A spiral ramp is the canonical case that breaks that
assumption: the vehicle climbs while turning, so (a) wheel/Doppler speed is
ALONG the slope -- a planar filter integrating it horizontally over-travels by
1/cos(grade), and (b) the whole climb is invisible to a planar filter. This is
exactly the "multi-level parking" case Phase 3 deferred the 3D core for.

We build the trajectory analytically and DERIVE the 6-axis IMU from it (specific
force = R^T (a_world - g); body rate from dR/dt), so a 3D INS integrating the
IMU from the true initial state reproduces the trajectory. Frames match idr3d:
ENU world (z up), body x-fwd/y-left/z-up, quaternion Hamilton body->world.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

G0 = 9.81


def _rot_to_quat(R):
    """Rotation matrix (body->world) -> Hamilton quaternion [w,x,y,z]."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


@dataclass
class Ramp:
    t: np.ndarray            # (N,)
    p: np.ndarray            # (N,3) ENU world position, m
    v: np.ndarray            # (N,3) ENU world velocity, m/s
    q: np.ndarray            # (N,4) attitude body->world
    a_m: np.ndarray          # (N,3) measured specific force, body
    w_m: np.ndarray          # (N,3) measured angular rate, body
    speed: np.ndarray        # (N,) along-slope (wheel) speed, m/s
    yawrate: np.ndarray      # (N,) world-vertical yaw rate (gravity-projected gyro), rad/s
    a_lat: np.ndarray        # (N,) body lateral specific force, m/s^2
    grade_deg: float


def helix_ramp(v=5.0, grade_deg=12.0, radius_h=12.0, duration=45.0, hz=10.0,
               seed=0, accel_bias=0.02, gyro_bias_deg=0.2,
               accel_noise=0.02, gyro_noise_deg=0.05) -> Ramp:
    """A climbing spiral at constant along-slope speed v and constant grade."""
    rng = np.random.default_rng(seed)
    n = int(duration * hz)
    t = np.arange(n) / hz
    grade = np.radians(grade_deg)
    v_h = v * np.cos(grade)                 # horizontal speed component
    v_up = v * np.sin(grade)                # climb rate
    w = v_h / radius_h                       # horizontal angular rate
    th = w * t
    # horizontal spiral (starts at origin, initial heading = +east), climb linear
    e = radius_h * np.sin(th)
    nn = radius_h * np.cos(th) - radius_h
    up = v_up * t
    p = np.stack([e, nn, up], axis=1)
    # exact velocity
    ve = radius_h * w * np.cos(th)
    vn = -radius_h * w * np.sin(th)
    vu = np.full(n, v_up)
    vel = np.stack([ve, vn, vu], axis=1)

    # body frame from the 3D velocity: x=forward, y=left(horizontal), z=up-ish
    R = np.empty((n, 3, 3))
    zw = np.array([0.0, 0.0, 1.0])
    for i in range(n):
        f = vel[i] / np.linalg.norm(vel[i])
        left = np.cross(zw, f); left /= np.linalg.norm(left)
        up_b = np.cross(f, left)
        R[i] = np.column_stack([f, left, up_b])
    q = np.array([_rot_to_quat(R[i]) for i in range(n)])

    # IMU by differentiation (physically consistent with p and R)
    dt = 1.0 / hz
    a_world = np.gradient(vel, dt, axis=0)              # world acceleration
    gvec = np.array([0.0, 0.0, -G0])
    a_body = np.einsum("nji,nj->ni", R, a_world - gvec)  # R^T (a - g): specific force
    dR = np.gradient(R, dt, axis=0)
    w_body = np.empty((n, 3))
    for i in range(n):
        S = R[i].T @ dR[i]                              # [w]_x = R^T dR/dt
        w_body[i] = np.array([S[2, 1], S[0, 2], S[1, 0]])

    # MEMS imperfections
    ab = rng.normal(0, accel_bias, 3)
    gb = np.radians(rng.normal(0, gyro_bias_deg, 3))
    a_m = a_body + ab + rng.normal(0, accel_noise, (n, 3))
    w_m = w_body + gb + rng.normal(0, np.radians(gyro_noise_deg), (n, 3))

    speed = np.full(n, v)                                # wheel speed = along-slope
    yawrate = np.full(n, w)                              # gravity-projected yaw = world w
    a_lat = a_body[:, 1]                                 # body lateral specific force
    return Ramp(t, p, vel, q, a_m, w_m, speed, yawrate, a_lat, grade_deg)


def _selfcheck():
    r = helix_ramp(seed=1, accel_noise=0.0, gyro_noise_deg=0.0,
                   accel_bias=0.0, gyro_bias_deg=0.0)
    # forward body speed should equal |vel| (NHC exact: no lateral/vertical body vel)
    R0 = None
    from eskf3d_ref import rotmat
    for i in (0, len(r.t)//2, -1):
        u = rotmat(r.q[i]).T @ r.v[i]
        assert abs(u[0] - r.speed[i]) < 1e-6, u
        assert abs(u[1]) < 1e-6 and abs(u[2]) < 1e-6, u
    climb = r.p[-1, 2]
    print(f"synth3d ok: N={len(r.t)} grade={r.grade_deg:.0f}deg climb={climb:.1f}m "
          f"turns={r.yawrate[0]*r.t[-1]/(2*np.pi):.1f} |vel|={np.linalg.norm(r.v[0]):.2f}")


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _selfcheck()
