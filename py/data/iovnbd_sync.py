"""IO-VNBD "Synchronised V and S" loader: phone IMU in, vehicle GNSS as truth.

The public IO-VNBD release (github.com/onyekpeu/IO-VNBD) ships each drive as a
pair of 10 Hz CSVs in one folder:

  V-<drive>.csv  vehicle: survey-grade GNSS lat/lon/velocity/heading + ECU yaw
                 rate + indicated (wheel) speed. THIS is the truth.
  S-<drive>.csv  smartphone: accelerometer, gyroscope, (stale ~0.1 Hz) GPS.
                 THIS is the sensor the product actually has.

That is exactly the ISRO problem (phone IMU dead-reckoning, scored against good
GNSS), so the loader pairs them. Three things in the raw files are not what they
say, and each was established against the data, not assumed:

  1. "Synchronised" rows are NOT time-aligned: the phone log has gaps the
     vehicle log does not (S4 drops 312 s), so row i of S and row i of V drift
     apart by tens of seconds. Alignment is therefore on absolute time -- the
     phone DATE column (local time; a whole-hour offset from the vehicle's UTC
     time-of-day) -- refined per continuous phone segment by cross-correlating
     the phone yaw gyro against the ECU yaw rate (residual skew ~+-0.3 s). That
     is a timestamp sync, not a use of the scored truth: no position or speed
     truth enters any model. Segments that do not sync above `min_corr` are
     dropped and recorded -- a phone sliding on the dashboard is not a drive to
     score (most Driver-E drives: corr 0.2-0.7).
  2. The gyro columns are labelled Yaw/Pitch/Roll but the one that tracks the
     ECU yaw rate is "Pitch" (corr 0.94-0.997, slope ~0.98, rad/s). The yaw axis
     and its sign are picked per segment from the data and written to QC.
  3. Sign: the ECU yaw rate is counter-clockwise positive, but the harness
     heading is a compass heading (clockwise positive) and integrates gyro_z
     directly, so gyro_z = -(phone yaw axis synced to the ECU). Getting this
     wrong mirrors every turn and makes gyro dead-reckoning lose to a straight
     line even over 10 s -- which is how it was caught.
  4. The phone "GPS SPEED (Kmh)" column is really m/s; irrelevant here because
     speed truth comes from the vehicle file, but do not use it as km/h.

The phone accelerometer correlates only weakly with the vehicle's longitudinal
acceleration (0.1-0.35 on the Driver-A drives), so accel-driven models should be
expected to do worse on this data than on synthetic -- that is a finding about
the data, surfaced in QC, not something to tune away.
"""
from __future__ import annotations
import glob, os, sys
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.io_vnbd import Drive, DriveQC, _lla_to_enu, sha256_file

HZ = 10.0
MAX_LAG_S = 30.0          # largest residual phone<->vehicle clock skew searched (s)
MIN_SEG_S = 120.0         # shorter synced segments can't hold a 60 s outage + context


def _col(df, *needles):
    """First column whose name contains every needle (case-insensitive)."""
    for c in df.columns:
        lc = c.lower()
        if all(n.lower() in lc for n in needles):
            return c
    raise KeyError(f"no column matching {needles}; have {list(df.columns)[:12]}...")


def _smooth(x, n=int(HZ)):
    return np.convolve(x, np.ones(n) / n, "same")


def _best_lag(g, yr, max_lag):
    """Lag L (rows) maximising |corr(g[i+L], yr[i])|, via FFT. Returns (L, corr)."""
    a = _smooth(g); b = _smooth(yr)
    a = (a - a.mean()) / (a.std() + 1e-12); b = (b - b.mean()) / (b.std() + 1e-12)
    n = len(a) + len(b)
    nfft = 1 << (n - 1).bit_length()
    cc = np.fft.irfft(np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft)), nfft)
    best_L, best_c = 0, 0.0
    for L in range(-max_lag, max_lag + 1):
        m = min(len(a) - max(L, 0), len(b) - max(-L, 0))      # overlap length
        if m < 60 * HZ:
            continue
        c = cc[L % nfft] / m
        if abs(c) > abs(best_c):
            best_L, best_c = L, float(c)
    return best_L, best_c


def _level_and_heading(acc, dvdt):
    """Rotate phone accel into the vehicle frame (x fwd, y left, z up).

    Up = mean specific force (gravity). The forward axis is the horizontal
    direction whose projection best matches the GNSS along-track acceleration --
    a constant mount calibration, fitted once per segment.
    """
    up = acc.mean(0); up /= np.linalg.norm(up)
    ref = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    h1 = ref - up * (ref @ up); h1 /= np.linalg.norm(h1)
    h2 = np.cross(up, h1)
    a1, a2 = _smooth(acc @ h1), _smooth(acc @ h2)
    # least squares dvdt ~ c1*a1 + c2*a2 -> forward direction angle
    c = np.linalg.lstsq(np.c_[a1, a2], _smooth(dvdt), rcond=None)[0]
    th = np.arctan2(c[1], c[0])
    fwd = np.cos(th) * h1 + np.sin(th) * h2
    left = np.cross(up, fwd)
    R = np.stack([fwd, left, up])                 # rows: vehicle axes in phone frame
    corr = float(np.corrcoef(_smooth(acc @ fwd), _smooth(dvdt))[0, 1])
    return R, corr


def _phone_clock(s):
    """Phone wall-clock seconds-of-day from the DATE column ('YYYY-MM-DD HH:MM:SS:mmm').

    TIME SINCE START resets mid-file; DATE does not, and it is what survives the
    phone's own logging gaps (S4 drops 312 s the vehicle log does not).
    """
    d = s[_col(s, "date")].astype(str).str.strip().str.slice(11)
    part = lambda a, b: d.str.slice(a, b).astype(float).to_numpy()
    return part(0, 2) * 3600 + part(3, 5) * 60 + part(6, 8) + part(9, 12) / 1000.0


def load_pair(s_path, v_path, *, root=None, min_corr=0.8, max_lag_s=MAX_LAG_S,
              min_seg_s=MIN_SEG_S):
    """One S/V pair -> list of synced Drives (one per continuous phone segment).

    Alignment is on absolute time: phone DATE minus a whole-hour timezone offset
    (the phone logged local time, the vehicle UTC), refined per segment by the
    yaw cross-correlation (clock skew of a few seconds). Returns (drives, rejected)
    where rejected holds a human-readable reason per dropped segment.
    """
    import pandas as pd
    s = pd.read_csv(s_path, encoding="latin-1"); v = pd.read_csv(v_path)
    s.columns = [c.strip() for c in s.columns]; v.columns = [c.strip() for c in v.columns]

    tv = v[_col(v, "time since start of day")].to_numpy(float)
    lat = v[_col(v, "latitude")].to_numpy(float); lon = v[_col(v, "longitude")].to_numpy(float)
    vel = v[_col(v, "velocity (km")].to_numpy(float) / 3.6
    hdg = np.radians(v[_col(v, "heading")].to_numpy(float))
    yr = np.radians(v[_col(v, "yaw rate")].to_numpy(float))
    ecu = v[_col(v, "indicated vehicle speed")].to_numpy(float) / 3.6

    ps = _phone_clock(s)
    acc = s[[_col(s, "accelerometer", ax) for ax in ("x", "y", "z")]].to_numpy(float)
    gyr = s[[_col(s, "gyroscope", ax) for ax in ("yaw", "pitch", "roll")]].to_numpy(float)
    gnames = ["Yaw", "Pitch", "Roll"]
    tz = round((ps[0] - tv[0]) / 3600.0) * 3600.0          # whole-hour timezone offset
    ps = ps - tz

    dps = np.diff(ps)
    cuts = np.flatnonzero((dps <= 0) | (dps > 1.0)) + 1
    bounds = np.r_[0, cuts, len(ps)]

    base = os.path.splitext(os.path.relpath(s_path, root) if root else os.path.basename(s_path))[0]
    base = base.replace(os.sep, "/")
    sha = sha256_file(s_path)[:16] + "+" + sha256_file(v_path)[:16]
    drives, rejected, short = [], [], 0
    dt = 1.0 / HZ
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        if ps[b - 1] - ps[a] < min_seg_s:
            short += 1; continue
        # phone segment on a uniform grid (its own clock)
        u = np.arange(ps[a], ps[b - 1], dt)
        G = np.stack([np.interp(u, ps[a:b], gyr[a:b, j]) for j in range(3)], axis=1)
        va = int(np.searchsorted(tv, u[0] - max_lag_s)); vb = int(np.searchsorted(tv, u[-1] + max_lag_s))
        if (vb - va) * dt < min_seg_s:
            rejected.append(f"{base}#{k}: no overlapping vehicle rows"); continue
        best = None
        for j in range(3):
            L, c = _best_lag(G[:, j], yr[va:vb], int(max_lag_s * HZ) * 2)
            if best is None or abs(c) > abs(best[1]):
                best = (L, c, j)
        L, c, j = best
        delta = (u[0] - tv[va]) + L * dt          # phone_time = vehicle_time + delta
        vi = np.flatnonzero((tv + delta >= ps[a]) & (tv + delta <= ps[b - 1]))
        if len(vi) * dt < min_seg_s:
            rejected.append(f"{base}#{k}: {len(vi)*dt:.0f}s overlap after sync"); continue
        tq = tv[vi] + delta
        sign = 1.0 if c > 0 else -1.0
        gyr_v = np.stack([np.interp(tq, ps[a:b], gyr[a:b, i]) for i in range(3)], axis=1)
        acc_p = np.stack([np.interp(tq, ps[a:b], acc[a:b, i]) for i in range(3)], axis=1)
        # The ECU yaw rate (and a z-up gyro) is counter-clockwise positive; the
        # harness heading is a compass heading (0=N, +clockwise) and integrates
        # gyro_z directly (synth: gyro_z == d heading/dt). So flip once more.
        # Verified on S3c: slope(d heading/dt vs gyro_z) = -0.98 before this flip.
        gz = -sign * gyr_v[:, j]
        c_al = float(np.corrcoef(_smooth(-gz), _smooth(yr[vi]))[0, 1])
        if not c_al >= min_corr:                   # NaN (flat gyro) must fail too
            rejected.append(f"{base}#{k}: phone-yaw vs ECU-yaw corr {c_al:.2f} < {min_corr} "
                            f"(skew {delta - 0:+.1f}s) -- phone not rigidly mounted / unsyncable")
            continue

        t = tv[vi] - tv[vi][0]
        e, n = _lla_to_enu(lat[vi], lon[vi])
        dvdt = np.gradient(vel[vi], dt)
        R, acc_corr = _level_and_heading(acc_p, dvdt)
        acc_v = acc_p @ R.T
        others = [i for i in range(3) if i != j]
        gyro_v = np.c_[gyr_v[:, others[0]], gyr_v[:, others[1]], gz]

        qc = DriveQC(path=os.path.abspath(s_path), sha256=sha, rows_in_file=len(s),
                     rows_used=int(len(vi)), time_unit="s (vehicle clock)", hz=HZ,
                     duration_s=float(t[-1]),
                     parked_frac=float((vel[vi] <= 0.5).mean()),
                     speed_min=float(vel[vi].min()), speed_max=float(vel[vi].max()),
                     heading_source="vehicle GNSS heading column")
        qc.max_dt = float(np.diff(t).max()) if len(t) > 1 else 0.0
        qc.warnings.append(f"sync: tz {tz/3600:+.0f}h, skew {delta:+.1f}s, yaw axis gyro "
                           f"'{gnames[j]}' sign {sign:+.0f}, yaw corr {c_al:.3f}")
        if acc_corr < 0.5:
            qc.warnings.append(f"phone accel vs GNSS along-track accel corr {acc_corr:.2f} "
                               f"(weak: accel-driven models are handicapped on this drive)")
        drives.append(Drive(
            f"{base}#{k}", t, e, n, vel[vi].copy(), hdg[vi].copy(), speed_ecu=ecu[vi].copy(),
            gyro_z=gz, acc=acc_v, gyro=gyro_v,
            meta={"path": s_path, "vehicle_path": v_path, "iovnbd_sync": True,
                  "tz_h": tz / 3600, "skew_s": delta, "yaw_col": gnames[j], "yaw_sign": sign,
                  "yaw_corr": c_al, "acc_corr": acc_corr,
                  "lat0": float(lat[vi][0]), "lon0": float(lon[vi][0]),   # ENU origin (e, n = 0, 0)
                  "columns": dict(truth="V: lat/lon/velocity/heading",
                                  imu="S: accelerometer + gyroscope"),
                  "units": dict(time="s", speed_kmph=True, ecu_kmph=True)},
            qc=qc))
    if short:
        rejected.append(f"{base}: {short} phone segment(s) shorter than {min_seg_s:.0f}s")
    return drives, rejected


def find_pairs(root):
    """Every folder holding exactly one S-*.csv and one V-*.csv."""
    pairs = []
    for sp in sorted(glob.glob(os.path.join(root, "**", "S-*.csv"), recursive=True)):
        vs = sorted(glob.glob(os.path.join(os.path.dirname(sp), "V-*.csv")))
        if len(vs) == 1:
            pairs.append((sp, vs[0]))
    return pairs


def load_sync_dir(root, *, min_corr=0.8, verbose=True):
    drives, rejected = [], []
    pairs = find_pairs(root)
    if not pairs:
        raise FileNotFoundError(f"no S-*.csv / V-*.csv pairs under {root}")
    for sp, vp in pairs:
        d, r = load_pair(sp, vp, root=root, min_corr=min_corr)
        drives += d; rejected += r
    if verbose:
        print(f"[iovnbd-sync] {len(pairs)} S/V pairs -> {len(drives)} synced segment(s), "
              f"{len(rejected)} rejected")
        for r in rejected:
            print(f"  rejected: {r}")
    if not drives:
        raise ValueError(f"no segment under {root} synced above corr {min_corr}")
    for d in drives:
        d.meta["rejected_segments"] = rejected
    return drives
