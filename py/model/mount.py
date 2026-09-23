"""Level phone IMU into the frame SpeedNet was trained in -- the reference for the app.

The real-data loader (data/iovnbd_sync._level_and_heading) hands SpeedNet
accelerometer in a vehicle-aligned frame with z = UP (+9.81 at rest). The phone
delivers DEVICE-frame samples, and a phone standing in a dash mount has its y
axis up, not z: on val drives that moved nn_real's bias from -4.5 to +0.5 m/s and
its MAE from 5.53 to 6.58 m/s. Measured on the same drives, the net is
insensitive to the HEADING of the horizontal axes (MAE unchanged under any
rotation about up), so leveling with gravity is enough; no forward-axis estimate
is needed. This is exactly what Level.kt / FusionEngine.kt do per 10 Hz step: a SLOW
(30 s) gravity EMA defines up, and each sample is rotated into (h1, h2, up) before
it enters the NN window. (The 0.5 s EMA used for the yaw rate follows the car's
accel/braking; leveling with it cost 0.45 m/s of MAE on val.)

Gyro: the loader's z channel is the yaw rate in the compass sign convention
(gyro_z = YAW_SIGN * g.up, the same value the filter predicts with); x, y are the
two horizontal rates.
"""
from __future__ import annotations
import numpy as np

YAW_SIGN = -1.0
TAU_MED_S = 5.0      # medium gravity a re-mount snaps to (== core/align.cpp REMOUNT_TAU_MED)
LEVEL_TAU_S = 30.0   # gravity EMA for leveling: slow, so vehicle accel/braking cannot tilt "up" (val: 0.5 s cost 0.45 m/s MAE, 30 s = trained frame)


def leveled_basis(up):
    """Rows (h1, h2, u): a right-handed frame with u = unit up (device coords)."""
    u = np.asarray(up, float) / max(np.linalg.norm(up), 1e-6)
    ref = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    h1 = ref - u * (ref @ u); h1 /= np.linalg.norm(h1)
    h2 = np.cross(u, h1)
    return np.stack([h1, h2, u])


def level_stream(acc, gyro, hz=10.0, tau=LEVEL_TAU_S, remounts=()):
    """Streaming leveling, sample by sample, exactly as the app does it (Level.kt).
    acc, gyro: (N,3) device frame at `hz`; remounts: sample indices where the core's
    re-mount detector fired (the slow gravity snaps to the 5 s one BEFORE that
    sample's update, as FusionEngine calls onRemount before step). Returns leveled (acc, gyro)."""
    a_ema = 1.0 - np.exp(-1.0 / (tau * hz))
    a_med = 1.0 - np.exp(-1.0 / (TAU_MED_S * hz))
    remounts = set(remounts)
    g = gm = None
    acc_l = np.empty_like(acc, dtype=float); gyr_l = np.empty_like(gyro, dtype=float)
    for i in range(len(acc)):
        if i in remounts and g is not None:
            g = gm.copy()
        if g is None:
            g = acc[i].astype(float).copy(); gm = g.copy()
        else:
            g = g + a_ema * (acc[i] - g); gm = gm + a_med * (acc[i] - gm)
        R = leveled_basis(g)
        acc_l[i] = R @ acc[i]
        w = R @ gyro[i]
        gyr_l[i] = (w[0], w[1], YAW_SIGN * w[2])
    return acc_l, gyr_l
