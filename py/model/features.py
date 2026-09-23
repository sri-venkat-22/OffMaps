"""Window -> feature tensor. THE contract shared by training and inference:
if these two ever disagree the model silently rots, so there is exactly one
implementation and both sides import it.

Input:  acc (T,3), gyro (T,3) at 10 Hz, vehicle frame.
Output: (9, T) float32 = [ax,ay,az, gx,gy,gz, |a|, |g|, |jerk|], fixed-scaled.
"""
from __future__ import annotations
import numpy as np

C = 9
WIN = 20   # 2 s at 10 Hz
HZ = 10.0
_SCALE = np.array([10, 10, 10, 1, 1, 1, 10, 1, 10], np.float32)[:, None]  # rough unit-ise


def window_features(acc, gyro):
    acc = np.asarray(acc, np.float32); gyro = np.asarray(gyro, np.float32)
    an = np.linalg.norm(acc, axis=1)
    gn = np.linalg.norm(gyro, axis=1)
    jerk = np.linalg.norm(np.diff(acc, axis=0, prepend=acc[:1]), axis=1) * HZ
    feat = np.concatenate([acc.T, gyro.T, an[None], gn[None], jerk[None]], axis=0)
    return (feat / _SCALE).astype(np.float32)
