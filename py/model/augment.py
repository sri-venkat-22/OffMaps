"""Train-time augmentation. The domain gap (IO-VNBD -> phone, vehicle A -> B,
loose mount) is what kills naive models; make the net see it in training.

All ops act on a (T,3) acc + (T,3) gyro window before feature extraction.
"""
from __future__ import annotations
import numpy as np


def _rand_rot(rng, max_deg):
    """Small random SO(3): the phone isn't mounted axis-aligned."""
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax) + 1e-9
    th = np.radians(rng.uniform(0, max_deg))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def augment(acc, gyro, rng, *, max_rot_deg=20.0, gyro_bias_dps=0.5,
            gyro_scale=0.03, noise=0.15):
    R = _rand_rot(rng, max_rot_deg)
    acc = acc @ R.T
    gyro = gyro @ R.T
    gyro = gyro * (1 + rng.normal(0, gyro_scale)) + np.radians(rng.normal(0, gyro_bias_dps, 3))
    acc = acc + rng.normal(0, noise, acc.shape)
    return acc.astype(np.float32), gyro.astype(np.float32)
