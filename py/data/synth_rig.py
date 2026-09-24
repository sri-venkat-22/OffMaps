"""Physically consistent device-frame IMU + magnetometer + GNSS at ANY rate.

synth_drive() hands the harness a vehicle-frame IMU at 10 Hz whose yaw gyro is
the compass rate directly -- fine for the speed/filter gates, but it cannot test
anything about attitude: mount rotation, gravity projection, lean, magnetometer.
This rig can. A planar trajectory (speed, compass heading) is lifted to 3-D:

  world (ENU) specific force  f_w = dv/dt * fwd - v*psidot * left + g * up
  body attitude               yaw = heading; roll = 0 (car) or the coordinated-turn
                              lean phi = atan(v*psidot/g) (two-wheeler, into the turn)
  body rates                  w_b = R (psidot_ccw * up + phi_dot * fwd)
  device                      v_dev = M v_b   (M = the phone's mount rotation)

so a gyro projected on the right axis integrates back to the exact heading. The
magnetometer sees the Earth field (Hyderabad-like: 39 uT north, 16 uT down, adjustable
declination) plus a constant hard-iron offset. Grades: "mems" (phone: 0.2 deg/s bias,
0.3 deg/s/sqrt(Hz) noise) or "fog" (fibre-optic gyro: 0.01 deg/h bias, 0.002
deg/sqrt(h) ARW). write_logger_csv() emits the phone logger's imu.csv / gnss.csv
schema (data/phone_log.py), which the edge engine reads.

The vibration speed cue is the same one synth_drive bakes in (std 0.3 + 0.06 v);
SpeedNet's real-data checkpoint is NOT expected to read it -- this rig is for the
attitude/heading/throughput paths, not for speed accuracy.
"""
from __future__ import annotations
import os
import numpy as np

G = 9.81
LAT0, LON0 = 17.3850, 78.4867          # Hyderabad
R_EARTH = 6_371_000.0
B_NORTH, B_DOWN = 39.0, 16.0           # uT

MOUNTS = {
    # rows = device axes expressed in body coords (body: x fwd, y left, z up)
    "flat": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float),        # screen up, top forward
    "dash": np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], float),       # portrait, facing the driver
}


def _rot(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def mount_matrix(kind="dash", yaw_deg=0.0, tilt_deg=0.0):
    """Mount rotation with an extra yaw (about body up) and tilt (about body left)."""
    return MOUNTS[kind] @ _rot([0, 0, 1], np.radians(yaw_deg)).T @ _rot([0, 1, 0], np.radians(tilt_deg)).T


def _smooth(x, n):
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), k, "valid")


def trajectory(duration=600.0, hz=200.0, seed=0, vehicle="car", park_s=0.0):
    """Smooth planar trajectory: t, speed, compass heading, compass yaw rate, dv/dt."""
    rng = np.random.default_rng(seed)
    n = int(duration * hz); t = np.arange(n) / hz
    v = np.zeros(n); r = np.zeros(n)
    vmax = 25.0 if vehicle == "car" else 16.0
    i = 0
    while i < n:
        j = min(i + int(rng.uniform(8, 30) * hz), n)
        kind = rng.choice(["cruise", "stop", "turn"], p=[0.55, 0.15, 0.30])
        if kind == "cruise":
            v[i:j] = rng.uniform(6, vmax)
        elif kind == "stop":
            v[i:j] = 0.0
        else:
            v[i:j] = rng.uniform(6, 12)
            r[i:j] = rng.choice([-1, 1]) * np.radians(rng.uniform(10, 30))
        i = j
    v[:int(park_s * hz)] = 0.0                               # parked start: no GNSS course
    v = _smooth(v, int(3 * hz)); r = _smooth(r, int(1.5 * hz)) * (v > 0.5)
    v[:int(park_s * hz)] = 0.0; r[:int(park_s * hz)] = 0.0
    heading = np.cumsum(r) / hz
    return t, v, heading, r, np.gradient(v, 1.0 / hz)


def synth_rig(duration=600.0, hz=200.0, seed=0, vehicle="car", mount="dash",
              mount_yaw_deg=12.0, mount_tilt_deg=8.0, grade="mems", declination_deg=0.0,
              hard_iron=(4.0, -3.0, 2.0), park_s=0.0, heading0_deg=0.0):
    """-> dict of arrays (device-frame imu/mag at `hz`, truth) + the matrices used."""
    rng = np.random.default_rng(seed + 1000)
    t, v, psi, r, dv = trajectory(duration, hz, seed, vehicle, park_s)
    psi = psi + np.radians(heading0_deg)
    n = len(t)
    fwd = np.stack([np.sin(psi), np.cos(psi), np.zeros(n)], 1)
    left = np.stack([-np.cos(psi), np.sin(psi), np.zeros(n)], 1)
    up = np.array([0.0, 0.0, 1.0])
    f_w = dv[:, None] * fwd - (v * r)[:, None] * left + G * up
    phi = np.arctan(v * r / G) if vehicle == "two_wheeler" else np.zeros(n)
    # body axes in world: x = fwd, z tilted toward the right (-left) by phi, y = z x x
    z_b = np.cos(phi)[:, None] * up - np.sin(phi)[:, None] * left
    y_b = np.cross(z_b, fwd)
    R = np.stack([fwd, y_b, z_b], 1)                         # (n,3,3): world -> body rows
    w_w = (-r)[:, None] * up + np.gradient(phi, 1.0 / hz)[:, None] * fwd
    M = mount_matrix(mount, mount_yaw_deg, mount_tilt_deg)
    to_dev = lambda x_w: np.einsum("ij,njk,nk->ni", M, R, x_w)
    acc = to_dev(f_w)
    gyr = to_dev(w_w)
    D = np.radians(declination_deg)
    b_w = np.array([B_NORTH * np.sin(D), B_NORTH * np.cos(D), -B_DOWN])
    mag = to_dev(np.broadcast_to(b_w, (n, 3))) + np.asarray(hard_iron)

    vib = 0.3 + 0.06 * v
    acc = acc + rng.normal(0, 1, (n, 3)) * vib[:, None]
    if grade == "fog":
        bias = np.radians(0.01 / 3600) * rng.normal(0, 1, 3)
        arw = np.radians(0.002) / 60.0                      # deg/sqrt(h) -> rad/sqrt(s)
    else:
        bias = np.radians(rng.normal(0, 0.2, 3))
        arw = np.radians(0.3) / np.sqrt(10.0)               # 0.3 deg/s per sample at 10 Hz
    gyr = gyr + bias + rng.normal(0, 1, (n, 3)) * arw * np.sqrt(hz)
    mag = mag + rng.normal(0, 0.5, (n, 3))

    dt = 1.0 / hz
    e = np.cumsum(v * np.sin(psi)) * dt; nn = np.cumsum(v * np.cos(psi)) * dt
    return dict(t=t, acc=acc, gyro=gyr, mag=mag, speed=v, heading=psi, yawrate=r, lean=phi,
                e=e, n=nn, mount=M, vehicle=vehicle, hz=hz, grade=grade)


def enu_to_ll(e, n):
    k = np.pi / 180 * R_EARTH
    return LAT0 + n / k, LON0 + e / (np.cos(np.radians(LAT0)) * k)


def write_logger_csv(rig, root, gnss_hz=1.0, pos_sigma=2.0, seed=0, masked=None):
    """imu.csv + gnss.csv in the phone logger schema, + truth.csv (at the IMU rate).
    masked: optional list of (t0, t1) spans written as masked=1 (the Outage Simulator)."""
    import pandas as pd
    rng = np.random.default_rng(seed + 7)
    os.makedirs(root, exist_ok=True)
    t = rig["t"]; tn = np.round(t * 1e9).astype(np.int64)
    a, g, m = rig["acc"], rig["gyro"], rig["mag"]
    pd.DataFrame(dict(t_ns=tn, ax=a[:, 0], ay=a[:, 1], az=a[:, 2], gx=g[:, 0], gy=g[:, 1], gz=g[:, 2],
                      mx=m[:, 0], my=m[:, 1], mz=m[:, 2])).to_csv(os.path.join(root, "imu.csv"), index=False)
    step = int(round(rig["hz"] / gnss_hz))
    idx = np.arange(0, len(t), step)
    e = rig["e"][idx] + rng.normal(0, pos_sigma, len(idx)); n = rig["n"][idx] + rng.normal(0, pos_sigma, len(idx))
    lat, lon = enu_to_ll(e, n)
    v = np.clip(rig["speed"][idx] + rng.normal(0, 0.1, len(idx)), 0, None)
    brg = np.where(v > 1.0, np.degrees(rig["heading"][idx] + rng.normal(0, np.radians(1.0), len(idx))) % 360, np.nan)
    msk = np.zeros(len(idx), int)
    for a0, a1 in masked or []:
        msk[(t[idx] >= a0) & (t[idx] < a1)] = 1
    pd.DataFrame(dict(t_ns=tn[idx], lat=lat, lon=lon, speed=v, bearing=brg, cn0_mean=40.0,
                      sv_used=12, navic_sv=3, masked=msk)).to_csv(os.path.join(root, "gnss.csv"), index=False)
    tl, tlo = enu_to_ll(rig["e"], rig["n"])
    pd.DataFrame(dict(t=t, e=rig["e"], n=rig["n"], lat=tl, lon=tlo, heading=rig["heading"],
                      speed=rig["speed"], lean=rig["lean"])).to_csv(os.path.join(root, "truth.csv"), index=False)
    return root
