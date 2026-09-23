"""Regression for the IO-VNBD synchronised S/V loader (data/iovnbd_sync.py).

Every quirk below was found in the real IO-VNBD files and each one, if handled
wrong, silently corrupts every drift number while the pipeline still runs:

  - the phone logs local time, the vehicle UTC (whole-hour offset), plus a
    sub-second-to-seconds residual skew -> must be recovered;
  - the phone log has gaps the vehicle log does not, so rows can't be paired by
    index -> the post-gap segment must still land on the right vehicle second;
  - the yaw gyro is the column labelled "Pitch";
  - the ECU yaw rate is counter-clockwise positive while the harness heading is
    clockwise positive -> gyro_z must integrate to the compass heading (the bug
    this caught mirrored every turn and made gyro DR lose to a straight line);
  - a loosely mounted phone (gyro uncorrelated with the car) and a flat gyro
    (NaN correlation) must be REJECTED, never scored.

Fixture is synthetic but written in the real files' header layout.
"""
import os

import numpy as np
import pandas as pd

from data.io_vnbd import synth_drive, R_EARTH
from data.iovnbd_sync import load_pair, load_sync_dir

TZ_H = 1.0          # phone local time = vehicle UTC + 1 h (BST, as in the real files)
SKEW = 0.4          # residual phone clock skew, s
V_T0 = 11 * 3600.0  # vehicle time-of-day at start

S_COLS = ["GPS LATITUDE (degrees)", "GPS LONGITUDE (degrees)", "GPS ALTITUDE (m)",
          "GPS SPEED (Kmh)", "GPS ACCURACY (m)", "GPS ORIENTATION (deg)",
          "GPS SATELLITES IN RANGE", "TIME SINCE START (ms)", "DATE (YYYY-MO-DD HH-MI-SS_SSS)",
          "ACCELEROMETER X (m/s2)", "ACCELEROMETER Y (m/s2)", "ACCELEROMETER Z (m/s2)",
          "GYROSCOPE Yaw (rad/s)", "GYROSCOPE Pitch (rad/s)", "GYROSCOPE Roll (rad/s)"]


def _write_pair(folder, *, duration=900, seed=7, gap=(3000, 3400), loose=False, flat=False):
    d = synth_drive("fx", duration=duration, seed=seed, hz=10)
    os.makedirs(folder, exist_ok=True)
    k = np.pi / 180.0 * R_EARTH
    lat = 52.4 + d.n / k
    lon = -1.5 + d.e / (np.cos(np.radians(52.4)) * k)
    ccw_yaw = -d.gyro_z                       # ECU / z-up gyro: counter-clockwise positive
    pd.DataFrame({
        "No of GPS Satellites Available": 11.0,
        "Time Since Start of Day (seconds)": V_T0 + d.t,
        "Latitude (degrees)": lat, "Longitude (degrees)": lon,
        "Velocity (km/hr)": d.speed * 3.6,
        "Heading (degrees)": np.degrees(d.heading) % 360,
        "Yaw Rate (deg/sec)": np.degrees(ccw_yaw),
        "Indicated Vehicle Speed (km/hr)": d.speed_ecu * 3.6,
    }).to_csv(os.path.join(folder, "V-fx.csv"), index=False)

    rng = np.random.default_rng(seed)
    # phone samples the same instants, but its clock reads (vehicle + tz + skew)
    tp = d.t + TZ_H * 3600 + SKEW
    pitch = ccw_yaw + rng.normal(0, 0.01, len(d))       # yaw lives on the "Pitch" column
    if loose:
        pitch = rng.normal(0, 0.2, len(d))
    if flat:
        pitch = np.zeros(len(d))
    keep = np.ones(len(d), bool)
    if gap:
        keep[gap[0]:gap[1]] = False                     # phone-only logging gap
    tod = V_T0 + tp
    stamp = [f"2019-09-08 {int(s//3600):02d}:{int(s%3600//60):02d}:{int(s%60):02d}:"
             f"{int(round((s % 1) * 1000)) % 1000:03d}" for s in tod]
    df = pd.DataFrame({c: 0.0 for c in S_COLS}, index=range(len(d)))
    df[S_COLS[7]] = (d.t * 1000).astype(int)
    df[S_COLS[8]] = stamp
    df[S_COLS[9]], df[S_COLS[10]], df[S_COLS[11]] = d.acc[:, 0], d.acc[:, 1], d.acc[:, 2]
    df[S_COLS[12]] = rng.normal(0, 0.02, len(d))
    df[S_COLS[13]] = pitch
    df[S_COLS[14]] = rng.normal(0, 0.02, len(d))
    df[keep].to_csv(os.path.join(folder, "S-fx.csv"), index=False)
    return d


def test_sync_recovers_clock_axis_sign_and_gap(tmp_path):
    folder = tmp_path / "A" / "fx"
    truth = _write_pair(str(folder))
    drives, rejected = load_pair(str(folder / "S-fx.csv"), str(folder / "V-fx.csv"),
                                 root=str(tmp_path))
    assert len(drives) == 2, rejected                   # the phone gap splits the drive
    for d in drives:
        assert d.meta["tz_h"] == TZ_H
        assert abs(d.meta["skew_s"] - SKEW) < 0.15, d.meta["skew_s"]
        assert d.meta["yaw_col"] == "Pitch"
        assert d.meta["yaw_corr"] > 0.95
        # gyro_z must be d(compass heading)/dt, i.e. the harness convention
        m = d.speed > 2
        hr = np.gradient(np.unwrap(d.heading), d.t)
        slope = np.polyfit(d.gyro_z[m], hr[m], 1)[0]
        assert 0.9 < slope < 1.1, slope
    # the post-gap segment must sit on the right vehicle seconds: its gyro matches
    # the true heading rate at those instants (index pairing would be 40 s off)
    post = drives[1]
    assert np.corrcoef(post.gyro_z, np.gradient(np.unwrap(post.heading), post.t))[0, 1] > 0.9
    assert abs(post.t[0] - 0.0) < 1e-9 and len(post) > 1000


def test_loose_and_flat_phones_are_rejected(tmp_path):
    _write_pair(str(tmp_path / "loose"), loose=True, gap=None)
    _write_pair(str(tmp_path / "flat"), flat=True, gap=None, seed=8)
    _write_pair(str(tmp_path / "good"), gap=None, seed=9)
    drives = load_sync_dir(str(tmp_path), verbose=False)
    ids = [d.vehicle_id for d in drives]
    assert ids == ["good/S-fx#0"], ids
    rej = drives[0].meta["rejected_segments"]
    assert any(r.startswith("loose/") and "corr" in r for r in rej), rej
    assert any(r.startswith("flat/") for r in rej), rej


def test_gyro_dead_reckoning_beats_straight_line(tmp_path):
    """End-to-end: a loader-produced drive runs through outage generation and the
    scored models, and gyro heading (physics) beats constant-velocity. (The sign
    convention itself is pinned by the slope assert in the first test -- this one
    alone does not catch a mirrored gyro on the synthetic fixture.)"""
    from data.outage import make_outages
    from eval import metrics as M
    from eval.models import REGISTRY
    import baseline.physics  # noqa: F401
    _write_pair(str(tmp_path / "g"), gap=None, seed=11, duration=1200)
    d = load_sync_dir(str(tmp_path), verbose=False)[0]
    outs = make_outages(d, (30, 60), 6, 0)
    err = lambda name: np.nanmedian([M.score_outage(d, o, REGISTRY[name](d, o))["end_err"]
                                     for o in outs])
    assert err("physics") < err("cv"), (err("physics"), err("cv"))
