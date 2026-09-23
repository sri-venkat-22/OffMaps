"""Messy-drive fixtures: what a REAL log looks like, so tests can assert the
harness survives it.

synth_drive() in data/io_vnbd.py is deliberately clean -- uniform 10 Hz, exact
speed and heading, no NaNs, no gaps, never parked. That is the right generator
for "does the pipeline run", and the wrong one for "is the pipeline honest".
These fixtures inject, one knob at a time, the pathologies every real GNSS log
actually has, so a test can name the pathology it is pinning.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

R_EARTH = 6_371_000.0
LAT0, LON0 = 12.9716, 77.5946     # Bengaluru


@dataclass
class MessySpec:
    """Each field is one real-log pathology. All default OFF -> a clean 1 Hz log."""
    n: int = 3600                 # nominal samples
    hz: float = 1.0               # nominal rate (real GNSS is 1 Hz, not 10)
    seed: int = 7
    parked_frac: float = 0.0      # share of time stationary (traffic lights, parking)
    drop_frac: float = 0.0        # randomly dropped samples -> time gaps
    n_dup: int = 0                # duplicated timestamp rows
    n_backwards: int = 0          # rows whose timestamp goes backwards
    nan_speed: int = 0            # NaN cells in the speed column
    gnss_pos_noise_m: float = 0.0 # per-sample position noise (real chip ~3 m)
    ecu_scale: float = 1.04       # wheel-speed optimism vs Doppler truth
    time_unit: str = "s"          # "s" | "ms" | "us" | "ns"
    time_epoch: float = 1.725e9   # 0.0 -> relative/elapsed clock
    speed_kmph: bool = False      # Doppler column in km/h
    ecu_kmph: bool = False        # ECU column in km/h  (independently!)


def make_messy(spec: MessySpec = MessySpec()):
    """Return (DataFrame, truth) for one messy drive.

    truth carries the *pre-corruption* ground truth on the surviving rows, so a
    test can assert what the loader should have recovered.
    """
    import pandas as pd
    rng = np.random.default_rng(spec.seed)
    n, hz = spec.n, spec.hz
    t = np.arange(n, dtype=float) / hz

    speed = np.zeros(n)
    i = 0
    while i < n:
        seg = int(rng.uniform(30, 300) * hz)
        j = min(i + seg, n)
        park = rng.random() < spec.parked_frac
        speed[i:j] = 0.0 if park else rng.uniform(4, 28)
        i = j

    yaw = np.zeros(n)
    for _ in range(max(1, n // 90)):
        k = int(rng.integers(0, max(1, n - 30)))
        yaw[k:k + int(rng.integers(5, 25))] = rng.choice([-1, 1]) * np.radians(rng.uniform(8, 30))
    hdg = np.cumsum(yaw) / hz
    e = np.cumsum(speed * np.sin(hdg)) / hz
    nn = np.cumsum(speed * np.cos(hdg)) / hz

    lat = LAT0 + np.degrees(nn / R_EARTH)
    lon = LON0 + np.degrees(e / (R_EARTH * np.cos(np.radians(LAT0))))
    if spec.gnss_pos_noise_m > 0:
        s = spec.gnss_pos_noise_m
        lat = lat + rng.normal(0, np.degrees(s / R_EARTH), n)
        lon = lon + rng.normal(0, np.degrees(s / (R_EARTH * np.cos(np.radians(LAT0)))), n)

    scale = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}[spec.time_unit]
    df = pd.DataFrame({
        "time": (spec.time_epoch + t) * scale,
        "latitude": lat, "longitude": lon,
        "speed": speed * (3.6 if spec.speed_kmph else 1.0),
        "wheel_speed": speed * spec.ecu_scale * (3.6 if spec.ecu_kmph else 1.0),
    })
    truth = dict(t=t.copy(), speed=speed.copy(), heading=hdg.copy(), e=e.copy(), n=nn.copy(),
                 ecu_scale=spec.ecu_scale, duration_s=float(t[-1] - t[0]), hz=hz)

    keep = np.ones(n, bool)
    if spec.drop_frac > 0:
        keep[rng.choice(n, int(n * spec.drop_frac), replace=False)] = False
        df = df[keep].reset_index(drop=True)
        for k in ("t", "speed", "heading", "e", "n"):
            truth[k] = truth[k][keep]
    if spec.n_dup > 0:
        k = min(spec.n_dup, len(df) - 1)
        df = pd.concat([df.iloc[:100 + k], df.iloc[100:100 + k], df.iloc[100 + k:]]).reset_index(drop=True)
    if spec.n_backwards > 0:
        idx = rng.choice(np.arange(10, len(df) - 10), spec.n_backwards, replace=False)
        df.loc[idx, "time"] = df.loc[idx, "time"] - 5 * scale
    if spec.nan_speed > 0:
        df.loc[rng.choice(len(df), min(spec.nan_speed, len(df)), replace=False), "speed"] = np.nan
    return df, truth


def write_messy(dirpath, spec: MessySpec = MessySpec(), name="drive_A.csv"):
    import os
    os.makedirs(dirpath, exist_ok=True)
    df, truth = make_messy(spec)
    p = os.path.join(dirpath, name)
    df.to_csv(p, index=False)
    return p, truth


REALISTIC = MessySpec(n=3600, hz=1.0, parked_frac=0.25, drop_frac=0.08, n_dup=3,
                      nan_speed=25, gnss_pos_noise_m=3.0, time_unit="ms",
                      speed_kmph=True, ecu_kmph=False)
"""One drive with every pathology on at once -- the integration fixture."""
