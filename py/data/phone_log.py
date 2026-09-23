"""Load a phone-logger recording (imu.csv + gnss.csv) into a Drive.

Schema is the CONTRACT with the Android LoggerActivity -- keep the two in sync.

  imu.csv  : t_ns, ax, ay, az, gx, gy, gz, mx, my, mz, pressure, light
  gnss.csv : t_ns, lat, lon, speed, bearing, cn0_mean, sv_used, navic_sv, masked

Phase-1 mount handling is deliberately minimal (full alignment is Phase 4): we
low-pass accel to get the gravity/up direction and project the gyro onto it to
recover yaw rate about vertical. Speed comes from GNSS Doppler (the `speed`
column). Everything is resampled onto a uniform target_hz grid so it drops
straight into the Phase-0 metrics harness.

The gyro->heading sign depends on device axis handedness; it's a tuning knob
(yaw_sign) to fix against one real log, not something to guess blind.
"""
# ponytail: gravity-projected yaw only; full 3D mount alignment is Phase 4.
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data.io_vnbd import Drive, _lla_to_enu


def _read(path):
    import pandas as pd
    return pd.read_csv(path)


def _gravity_up(acc, hz):
    """EMA low-pass of accel -> unit up vector per sample (specific force ~ +up)."""
    a = 1.0 - np.exp(-1.0 / (0.5 * hz))     # ~0.5 s time constant
    g = np.empty_like(acc); g[0] = acc[0]
    for i in range(1, len(acc)):
        g[i] = g[i - 1] + a * (acc[i] - g[i - 1])
    return g / np.clip(np.linalg.norm(g, axis=1, keepdims=True), 1e-6, None)


def load_phone(root, *, target_hz=10.0, yaw_sign=-1.0) -> Drive:
    imu = _read(os.path.join(root, "imu.csv"))
    gnss = _read(os.path.join(root, "gnss.csv"))
    ti = imu["t_ns"].to_numpy(float) * 1e-9
    tg = gnss["t_ns"].to_numpy(float) * 1e-9
    t0 = min(ti[0], tg[0]); ti -= t0; tg -= t0

    acc = imu[["ax", "ay", "az"]].to_numpy(float)
    gyr = imu[["gx", "gy", "gz"]].to_numpy(float)
    up = _gravity_up(acc, target_hz)
    yaw = yaw_sign * np.einsum("ij,ij->i", gyr, up)     # yaw rate about vertical

    grid = np.arange(0.0, min(ti[-1], tg[-1]), 1.0 / target_hz)
    e_g, n_g = _lla_to_enu(gnss["lat"].to_numpy(float), gnss["lon"].to_numpy(float))
    speed = np.interp(grid, tg, gnss["speed"].to_numpy(float))
    e = np.interp(grid, tg, e_g); n = np.interp(grid, tg, n_g)
    gyro_z = np.interp(grid, ti, yaw)
    accg = np.stack([np.interp(grid, ti, acc[:, k]) for k in range(3)], axis=1)
    gyrg = np.stack([np.interp(grid, ti, gyr[:, k]) for k in range(3)], axis=1)
    bearing = (np.interp(grid, tg, np.radians(gnss["bearing"].to_numpy(float)))
               if "bearing" in gnss else np.arctan2(np.gradient(e), np.gradient(n)))
    meta = {"phone": True, "root": root, "hz": target_hz,
            "navic_sv": int(gnss.get("navic_sv", [0]).max()) if "navic_sv" in gnss else 0}
    return Drive(os.path.basename(os.path.normpath(root)), grid, e, n, speed,
                 bearing, gyro_z=gyro_z, acc=accg, gyro=gyrg, meta=meta)  # acc/gyro device-frame; align in Phase 4


def _selfcheck():
    import tempfile
    from data.io_vnbd import synth_drive, R_EARTH
    d = synth_drive(duration=120, seed=5, hz=50)   # pretend 50 Hz IMU
    root = tempfile.mkdtemp()
    # fake IMU: flat phone (up=+z), gz = true yaw rate; GNSS at 1 Hz from truth
    lat0, lon0 = 17.4, 78.5
    k = np.pi / 180 * R_EARTH
    import csv
    with open(os.path.join(root, "imu.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow("t_ns ax ay az gx gy gz mx my mz pressure light".split())
        for i in range(len(d)):
            w.writerow([int(d.t[i] * 1e9), 0, 0, 9.81, 0, 0, d.gyro_z[i], 0, 0, 0, 101325, 100])
    with open(os.path.join(root, "gnss.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow("t_ns lat lon speed bearing cn0_mean sv_used navic_sv masked".split())
        for i in range(0, len(d), 50):   # 1 Hz
            lat = lat0 + d.n[i] / k
            lon = lon0 + d.e[i] / (np.cos(np.radians(lat0)) * k)
            w.writerow([int(d.t[i] * 1e9), lat, lon, d.speed[i], np.degrees(d.heading[i]),
                        40, 9, 3, 0])
    dp = load_phone(root, target_hz=10, yaw_sign=+1.0)  # +1: our fake gz is already yaw
    assert dp.gyro_z is not None and len(dp) > 100
    # integrated heading from phone gyro should track the true heading trend
    corr = np.corrcoef(np.cumsum(dp.gyro_z), np.interp(dp.t, d.t, np.cumsum(d.gyro_z)))[0, 1]
    assert corr > 0.9, corr
    print(f"phone_log ok: N={len(dp)} navic_sv={dp.meta['navic_sv']} heading_corr={corr:.3f}")


if __name__ == "__main__":
    _selfcheck()
