"""High-rate (200-400 Hz) synthetic accelerometer with a speed-dependent
vibration cue and injected pothole shocks.

The learnable speed cue is real: road-excitation vibration amplitude grows with
speed. At 10 Hz that cue aliases (Nyquist 5 Hz); at 250 Hz it is clean. A pothole
is a short, high-amplitude transient that, if not excised, inflates the vibration
RMS and reads as a phantom speed spike. This generator makes both so the Phase-7c
front-end can be gated. (The learned RMS->speed map itself trains on real phone
logs; this only exercises the deterministic front-end.)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

G0 = 9.81


def vib_std(speed):
    """AC vibration std (m/s^2) vs speed -- same model as io_vnbd.synth_drive."""
    return 0.3 + 0.06 * np.asarray(speed)


@dataclass
class HighRate:
    t: np.ndarray            # (N,)
    acc: np.ndarray          # (N,3) tri-axial accel, m/s^2 (incl. gravity on z)
    speed: np.ndarray        # (N,) per-sample speed, m/s
    hz: float
    pothole_t: np.ndarray    # injected pothole onset times, s


def highrate_accel(hz=250.0, seg_s=4.0,
                   speeds=(5, 10, 15, 20, 25, 12, 18, 8),
                   pothole_t=(3.0, 9.0, 15.0, 21.0, 27.0),
                   pothole_amp=12.0, pothole_f=35.0, pothole_tau=0.03,
                   pothole_dur=0.12, seed=0) -> HighRate:
    rng = np.random.default_rng(seed)
    n_seg = len(speeds)
    N = int(n_seg * seg_s * hz)
    t = np.arange(N) / hz
    seg_idx = np.clip((t / seg_s).astype(int), 0, n_seg - 1)
    speed = np.array(speeds, float)[seg_idx]
    vs = vib_std(speed)
    # broadband vibration: strongest vertical, weaker horizontal
    acc = np.empty((N, 3))
    acc[:, 0] = rng.normal(0, 0.4 * vs)
    acc[:, 1] = rng.normal(0, 0.4 * vs)
    acc[:, 2] = G0 + rng.normal(0, vs)
    # inject pothole shocks: a damped oscillation, mostly vertical
    for pt in pothole_t:
        i0 = int(pt * hz); i1 = int((pt + pothole_dur) * hz)
        tau = np.arange(i1 - i0) / hz
        burst = pothole_amp * np.exp(-tau / pothole_tau) * np.sin(2 * np.pi * pothole_f * tau)
        acc[i0:i1, 2] += burst
        acc[i0:i1, 0] += 0.4 * burst
    return HighRate(t, acc, speed, hz, np.array(pothole_t, float))


def _selfcheck():
    hr = highrate_accel(seed=1)
    assert hr.acc.shape[0] == len(hr.t)
    print(f"highrate ok: N={len(hr.t)} hz={hr.hz:.0f} dur={hr.t[-1]:.0f}s "
          f"speeds={sorted(set(hr.speed))} potholes={len(hr.pothole_t)}")


if __name__ == "__main__":
    _selfcheck()
