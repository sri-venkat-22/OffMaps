"""Turn Drives into contiguous L-second training sequences.

Each 1 s step carries the 2 s IMU window ENDING at that step, the true 1 s
displacement, and a motion class. Augmentation is applied per-window at train
time so every epoch sees a different mount/bias/noise realisation.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from model.features import window_features, WIN, HZ, C
from model.augment import augment

STEP = int(HZ)          # 1 s
L = 30                  # sequence length (s) -- must cover the 30 s horizon


def _motion_class(speed, yawrate):
    if speed < 0.5: return 0                       # stopped
    if abs(np.degrees(yawrate)) < 5: return 1      # straight
    return 2                                       # turning  (3=rough reserved)


def _steps(drive):
    """Yield (win_start, step_lo, step_hi) for every valid 1 s step in a drive.

    The 2 s window ENDS at the step's end, so the last 1 s of the window IS the
    step being predicted -- contemporaneous, matching nn_model.predict_steps
    exactly. It used to end at the step's START (`end - WIN`), i.e. training
    forecast the NEXT second from the prior 2 s while inference read the current
    second; that train/serve skew fed the sigma head an off-by-1 s speed and
    showed up as bias in calibration. For dead reckoning you want the speed
    DURING the interval you integrate, which is what inference already did, so
    training moves to match it.
    """
    n = len(drive)
    for lo in range(WIN - STEP, n - STEP, STEP):   # step [lo, lo+STEP); window ends at lo+STEP
        ws = lo + STEP - WIN                        # >= 0 since lo >= WIN - STEP
        yield ws, lo, lo + STEP


class SeqDataset(Dataset):
    def __init__(self, drives, seed=0, augment_data=True):
        self.aug = augment_data
        self.rng = np.random.default_rng(seed)
        self.items = []              # each: (drive, [(ws, lo, hi), ...L])
        for d in drives:
            steps = list(_steps(d))
            for i in range(0, len(steps) - L, L // 2):     # 50% overlap
                self.items.append((d, steps[i:i + L]))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        d, steps = self.items[idx]
        rng = np.random.default_rng(self.rng.integers(1 << 30))
        X = np.empty((L, C, WIN), np.float32)
        dt = np.empty(L, np.float32); cl = np.empty(L, np.int64)
        for j, (ws, lo, hi) in enumerate(steps):
            acc = d.acc[ws:ws + WIN].copy(); gyro = d.gyro[ws:ws + WIN].copy()
            if self.aug:
                acc, gyro = augment(acc, gyro, rng)
            X[j] = window_features(acc, gyro)
            dt[j] = np.trapezoid(d.speed[lo:hi], d.t[lo:hi])       # true 1 s displacement
            cl[j] = _motion_class(d.speed[lo:hi].mean(), d.gyro_z[lo:hi].mean())
        return torch.from_numpy(X), torch.from_numpy(dt), torch.from_numpy(cl)
