"""Window -> feature tensor. THE contract shared by training and inference:
if these two ever disagree the model silently rots, so there is exactly one
implementation and both sides import it.

Input:  acc (T,3), gyro (T,3) at 10 Hz, leveled frame (z = up, gyro z = yaw rate
        in the compass sign; model/mount.py).

Version 1 (nn_real.pt, the shipped phone net): 2 s window,
  (9, T) float32 = [ax,ay,az, gx,gy,gz, |a|, |g|, |jerk|], fixed-scaled.

Version 2: a 20 s window of physics-shaped channels that do not depend on the
heading of the horizontal axes (the phone levels with gravity but does not know
which horizontal axis is forward, so a v2 net sees the same numbers in the
training frame and on the phone):

  0 |a_h|      horizontal specific force, 0.5 s mean (m/s^2)
  1 a_up       vertical specific force minus its window mean, 0.5 s mean
  2 yaw        yaw rate, 0.5 s mean (rad/s)
  3 |w_h|      pitch/roll rate, 0.5 s mean: holder rocking
  4 vib_a      |a - 0.5 s mean|, 1 s mean: road / engine vibration energy
  5 vib_up     the vertical part of vib_a (tyre and road excitation)
  6 vib_g      |w - 0.5 s mean|, 1 s mean
  7 jerk       |da/dt|, 1 s mean
  8 turn_v     centripetal speed cue: in a turn v = a_lat / yaw_rate, read as
               |a_h| / |yaw| where |yaw| > 0.05 rad/s, clipped to 0..40 m/s
  9 turning    1 where turn_v is defined, else 0

Filters are centred moving averages over the window with the edges held, so a
window's features depend only on that window (the phone recomputes them from
its ring buffer exactly as training does).
"""
from __future__ import annotations
import numpy as np

C = 9
WIN = 20   # 2 s at 10 Hz
HZ = 10.0
_SCALE = np.array([10, 10, 10, 1, 1, 1, 10, 1, 10], np.float32)[:, None]  # rough unit-ise

C2 = 10
WIN2 = 200                 # 20 s at 10 Hz
LP_N, VIB_N = 5, 10        # 0.5 s and 1 s moving means
TURN_MIN = 0.05            # rad/s: below this a_h / yaw is not a speed
TURN_MAX = 40.0            # m/s
_SCALE2 = np.array([1, 1, 0.2, 0.1, 1, 1, 0.1, 10, 10, 1], np.float32)


def window_features(acc, gyro):
    acc = np.asarray(acc, np.float32); gyro = np.asarray(gyro, np.float32)
    an = np.linalg.norm(acc, axis=1)
    gn = np.linalg.norm(gyro, axis=1)
    jerk = np.linalg.norm(np.diff(acc, axis=0, prepend=acc[:1]), axis=1) * HZ
    feat = np.concatenate([acc.T, gyro.T, an[None], gn[None], jerk[None]], axis=0)
    return (feat / _SCALE).astype(np.float32)


def box(x, n):
    """Centred n-sample moving mean along axis -2 of (..., T, k), edges held."""
    if n <= 1:
        return x
    lo, hi = n // 2, n - 1 - n // 2
    xp = np.concatenate([np.repeat(x[..., :1, :], lo, axis=-2), x,
                         np.repeat(x[..., -1:, :], hi, axis=-2)], axis=-2)
    c = np.cumsum(xp, axis=-2, dtype=np.float64)
    c = np.concatenate([np.zeros_like(c[..., :1, :]), c], axis=-2)
    return ((c[..., n:, :] - c[..., :-n, :]) / n).astype(np.float32)


def window_features_v2(acc, gyro):
    """(..., T, 3) acc + gyro -> (..., C2, T) float32. Batched over leading axes."""
    acc = np.asarray(acc, np.float32); gyro = np.asarray(gyro, np.float32)
    a_lp = box(acc, LP_N); w_lp = box(gyro, LP_N)
    ah = np.linalg.norm(a_lp[..., :2], axis=-1)
    au = a_lp[..., 2] - acc[..., 2].mean(axis=-1, keepdims=True)
    yaw = w_lp[..., 2]
    wh = np.linalg.norm(w_lp[..., :2], axis=-1)
    da = acc - a_lp
    vib = np.stack([np.linalg.norm(da, axis=-1), np.abs(da[..., 2]),
                    np.linalg.norm(gyro - w_lp, axis=-1)], axis=-1)
    step = np.diff(acc, axis=-2, prepend=acc[..., :1, :])
    jerk = np.linalg.norm(step, axis=-1, keepdims=True) * np.float32(HZ)
    vib = box(np.concatenate([vib, jerk], axis=-1), VIB_N)
    turning = np.abs(yaw) > TURN_MIN
    turn_v = np.where(turning, np.clip(ah / np.maximum(np.abs(yaw), TURN_MIN), 0.0, TURN_MAX), 0.0)
    feat = np.stack([ah, au, yaw, wh, vib[..., 0], vib[..., 1], vib[..., 2], vib[..., 3],
                     turn_v, turning.astype(np.float32)], axis=-2)
    return (feat / _SCALE2[:, None]).astype(np.float32)


FEATURES = {1: (window_features, C, WIN), 2: (window_features_v2, C2, WIN2)}


def spec(version=1, win=None):
    """(feature fn, channels, window) for a feature version; win overrides the default."""
    fn, c, w = FEATURES[int(version)]
    return fn, c, int(win or w)
