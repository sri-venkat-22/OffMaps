"""A nav-app recording (imu.csv + gnss.csv) as SpeedNet TRAINING data, fed exactly as the
phone feeds its net.

data/phone_log.load_phone resamples the IMU by interpolation, which is not what the app
does: FusionEngine keeps ONE raw sample per 100 ms (Decimator.kt), levels it with a slow
30 s gravity EMA (Level.kt == model/mount.level_stream) and flips the yaw sign. A net
fine-tuned on interpolated samples would see smoother vibration than it gets on the phone.
Here the 10 Hz grid is the Decimator's own picks, and the label is the phone's GNSS Doppler
speed interpolated onto it (the only speed truth a phone recording has; ~0.1-0.3 m/s).

Seconds without a fix nearby (> 2 s from any GNSS row) are cut out: the drive splits into
contiguous segments there, so no training window straddles a GNSS gap.
"""
from __future__ import annotations
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data.io_vnbd import Drive, _lla_to_enu
from model.mount import level_stream

PERIOD_NS = 100_000_000      # Decimator(100 ms)
MAX_GAP_S = 2.0
MIN_SEG_S = 60.0


def decimate(t_ns):
    """Indices the phone's Decimator keeps: a sample >= 100 ms after the last kept one."""
    keep, last = [], None
    for k, t in enumerate(t_ns):
        if last is None or t - last >= PERIOD_NS:
            keep.append(k); last = t
    return np.asarray(keep)


def load_phone_drive(root, name=None):
    """-> list of Drives (contiguous GNSS-covered segments at the phone's 10 Hz)."""
    import pandas as pd
    imu = pd.read_csv(os.path.join(root, "imu.csv"), on_bad_lines="skip")
    gn = pd.read_csv(os.path.join(root, "gnss.csv"), on_bad_lines="skip")
    imu = imu[np.isfinite(pd.to_numeric(imu["t_ns"], errors="coerce"))]
    ti_ns = imu["t_ns"].to_numpy(np.int64)
    ok = np.r_[True, np.diff(ti_ns) > 0]
    imu = imu[ok]; ti_ns = ti_ns[ok]
    k = decimate(ti_ns)
    acc = imu[["ax", "ay", "az"]].to_numpy(float)[k]
    gyr = imu[["gx", "gy", "gz"]].to_numpy(float)[k]
    t = (ti_ns[k] - ti_ns[0]) * 1e-9
    acc_l, gyr_l = level_stream(acc, gyr)                    # Level.kt, yaw sign included
    gn = gn[np.isfinite(gn["lat"]) & np.isfinite(gn["speed"])]
    tg = (gn["t_ns"].to_numpy(np.int64) - ti_ns[0]) * 1e-9
    speed = np.interp(t, tg, gn["speed"].to_numpy(float))
    lat = np.interp(t, tg, gn["lat"].to_numpy(float)); lon = np.interp(t, tg, gn["lon"].to_numpy(float))
    e, n = _lla_to_enu(lat, lon)
    near = np.abs(tg[np.clip(np.searchsorted(tg, t), 0, len(tg) - 1)] - t)
    near = np.minimum(near, np.abs(tg[np.clip(np.searchsorted(tg, t) - 1, 0, len(tg) - 1)] - t))
    good = (near <= MAX_GAP_S) & (t >= tg[0]) & (t <= tg[-1])
    name = name or os.path.basename(os.path.normpath(root))
    out, i = [], 0
    edges = np.flatnonzero(np.diff(np.r_[0, good.astype(int), 0]))
    for s0, s1 in zip(edges[::2], edges[1::2]):
        if t[s1 - 1] - t[s0] < MIN_SEG_S:
            continue
        sl = slice(s0, s1)
        hd = np.arctan2(np.gradient(e[sl]), np.gradient(n[sl]))
        out.append(Drive(vehicle_id=f"phone/{name}#{i}", t=t[sl] - t[s0], e=e[sl], n=n[sl], speed=speed[sl],
                         heading=hd, gyro_z=gyr_l[sl, 2], acc=acc_l[sl].astype(np.float32),
                         gyro=gyr_l[sl].astype(np.float32), meta=dict(source=root)))
        i += 1
    return out
