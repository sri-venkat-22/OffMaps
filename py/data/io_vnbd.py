"""IO-VNBD loader + synthetic drive generator.

A Drive is the one data structure the whole harness speaks. Everything else
(outage generator, models, metrics) operates on Drive arrays by index.

We don't ship the IO-VNBD files, so load_dir() reads whatever CSVs you point it
at (column names are flags, since the exact headers vary), and synth_drive()
makes a realistic drive so the harness runs end-to-end with zero data.

synth_drive() is uniform, exact, gapless and never parks. Real logs are none of
those, so loading is deliberately strict: every real-log pathology either gets
repaired with a record of what was done (DriveQC) or raises. The one thing this
module will not do is guess -- a silently mis-decoded time axis or a silently
mis-scaled speed column corrupts every number the benchmark reports.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import glob, hashlib, os, sys
import numpy as np

if __package__ in (None, ""):          # allow `python3 data/io_vnbd.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import timeaxis

R_EARTH = 6_371_000.0  # m


@dataclass
class DriveQC:
    """What loading had to repair, and what the drive actually looks like.

    Carried on Drive.qc so the report can print it and the manifest can pin it.
    A benchmark that silently repairs its input is not reproducible.
    """
    path: str = ""
    sha256: str = ""
    rows_in_file: int = 0
    rows_used: int = 0
    time_unit: str = ""
    hz: float = float("nan")
    duration_s: float = 0.0
    n_dropped_nonincreasing: int = 0
    n_nonfinite_time: int = 0
    n_gaps: int = 0
    gap_seconds: float = 0.0
    max_dt: float = 0.0
    n_nan_speed: int = 0
    n_nan_pos: int = 0
    parked_frac: float = 0.0
    speed_min: float = 0.0
    speed_max: float = 0.0
    heading_source: str = ""
    warnings: list = field(default_factory=list)

    def summary(self):
        w = f"  [{len(self.warnings)} warning(s)]" if self.warnings else ""
        return (f"{self.rows_used}/{self.rows_in_file} rows  {self.hz:.2f} Hz  "
                f"{self.duration_s/60:.1f} min  gaps={self.n_gaps} ({self.gap_seconds:.0f}s)  "
                f"speed {self.speed_min:.1f}-{self.speed_max:.1f} m/s  "
                f"parked={self.parked_frac*100:.0f}%  hdg={self.heading_source}{w}")


@dataclass
class Drive:
    vehicle_id: str
    t: np.ndarray          # (N,) seconds, monotonic
    e: np.ndarray          # (N,) local ENU east, m
    n: np.ndarray          # (N,) local ENU north, m
    speed: np.ndarray      # (N,) ground-truth ground speed, m/s
    heading: np.ndarray    # (N,) rad, 0 = north, +east (from velocity if absent)
    speed_ecu: np.ndarray | None = None   # raw wheel/ECU speed if present, m/s
    gyro_z: np.ndarray | None = None       # yaw rate about vertical, rad/s (as a gyro would read it)
    acc: np.ndarray | None = None          # (N,3) specific force, vehicle frame x-fwd y-left z-up, m/s^2
    gyro: np.ndarray | None = None         # (N,3) angular rate, vehicle frame, rad/s
    meta: dict = field(default_factory=dict)
    qc: DriveQC | None = None

    def __len__(self): return len(self.t)

    @property
    def moving(self):
        """Boolean mask of samples where the vehicle is actually moving."""
        return self.speed > 0.5


def _lla_to_enu(lat, lon):
    lat0, lon0 = float(lat[0]), float(lon[0])
    k = np.pi / 180.0 * R_EARTH
    e = (lon - lon0) * np.cos(np.radians(lat0)) * k
    n = (lat - lat0) * k
    return e, n


def _heading_from_xy(e, n, speed=None, *, min_speed=1.0):
    """Course over ground from the ENU track.

    np.gradient on raw ENU is fine while moving and is pure noise while stopped
    -- at standstill the numerator is GNSS jitter, so the heading spins. On a
    real 1 Hz log with 25% parked time that poisons both the turn/straight label
    and the along/cross basis. So: compute where the vehicle moves, and hold the
    last good heading through every stationary run.
    """
    de, dn = np.gradient(e), np.gradient(n)
    h = np.arctan2(de, dn)  # 0=north, +east
    if speed is None:
        return h
    good = np.asarray(speed) > min_speed
    if not good.any():
        return np.zeros_like(h)
    idx = np.flatnonzero(good)
    # hold last good heading forward; back-fill the head with the first good one
    fill = np.searchsorted(idx, np.arange(len(h)), side="right") - 1
    fill = np.clip(fill, 0, len(idx) - 1)
    return h[idx[fill]]


def sha256_file(path, _buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_col(df, name, kind, *, required):
    """Map a requested column onto the frame, refusing near-misses silently.

    A typo'd --lat-col used to fall through to a default or to None. Both are
    worse than stopping: one scores the wrong column, the other silently drops
    a signal the user asked for.
    """
    if name is None:
        return None
    if name in df.columns:
        return name
    lower = {c.lower(): c for c in df.columns}
    if name.lower() in lower:
        return lower[name.lower()]
    near = [c for c in df.columns if name.lower() in c.lower() or c.lower() in name.lower()]
    hint = f" Did you mean {near}?" if near else f" Columns: {list(df.columns)[:20]}"
    if required:
        raise KeyError(f"{kind} column {name!r} not found.{hint}")
    raise KeyError(f"{kind} column {name!r} was requested but not found.{hint} "
                   f"Omit the flag to skip this signal.")


def load_csv(path, *, vehicle_id=None, time_col="time", lat_col="latitude",
             lon_col="longitude", speed_col="speed", heading_col=None,
             ecu_speed_col=None, speed_is_kmph=False, ecu_is_kmph=None,
             time_unit="auto", max_gap_s=None, strict=True) -> Drive:
    """Load one drive from a CSV. Requires pandas (only imported here).

    speed_is_kmph applies to speed_col ONLY. ecu_is_kmph applies to
    ecu_speed_col and defaults to speed_is_kmph for backwards compatibility --
    pass it explicitly when the log mixes units (Doppler m/s + wheel km/h is
    common, and getting it wrong makes ecu_doppler_scale return 1.0, which
    silently removes Phase 4's whole calibration signal).
    """
    import pandas as pd
    df = pd.read_csv(path)
    if ecu_is_kmph is None:
        ecu_is_kmph = speed_is_kmph

    time_col = _resolve_col(df, time_col, "time", required=True)
    lat_col = _resolve_col(df, lat_col, "latitude", required=True)
    lon_col = _resolve_col(df, lon_col, "longitude", required=True)
    speed_col = _resolve_col(df, speed_col, "speed", required=True)
    heading_col = _resolve_col(df, heading_col, "heading", required=False)
    ecu_speed_col = _resolve_col(df, ecu_speed_col, "ecu-speed", required=False)

    qc = DriveQC(path=os.path.abspath(path), sha256=sha256_file(path),
                 rows_in_file=len(df))

    raw_t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
    if not np.isfinite(raw_t).any():                      # ISO-8601 / datetime strings
        raw_t = pd.to_datetime(df[time_col], errors="coerce").astype("int64").to_numpy(float)
        time_unit = "ns" if time_unit == "auto" else time_unit
    t_all, unit = timeaxis.decode(raw_t, time_unit)
    qc.time_unit = unit

    keep, rep = timeaxis.repair(t_all)
    qc.rows_used = rep["n_out"]
    qc.n_dropped_nonincreasing = rep["n_dropped_nonincreasing"]
    qc.n_nonfinite_time = rep["n_nonfinite"]
    qc.hz, qc.duration_s = rep["hz"], rep["duration_s"]
    qc.n_gaps, qc.gap_seconds, qc.max_dt = rep["n_gaps"], rep["gap_seconds"], rep["max_dt"]
    if qc.n_dropped_nonincreasing:
        qc.warnings.append(f"dropped {qc.n_dropped_nonincreasing} non-increasing timestamp(s)")
    if qc.n_gaps:
        qc.warnings.append(f"{qc.n_gaps} gap(s) totalling {qc.gap_seconds:.0f}s, max {qc.max_dt:.1f}s")
    if max_gap_s is not None and qc.max_dt > max_gap_s:
        raise ValueError(f"{path}: {qc.max_dt:.1f}s gap exceeds --max-gap {max_gap_s}s")

    sub = df.iloc[keep]
    t = t_all[keep] - t_all[keep][0]
    lat = pd.to_numeric(sub[lat_col], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(sub[lon_col], errors="coerce").to_numpy(float)
    speed = pd.to_numeric(sub[speed_col], errors="coerce").to_numpy(float)
    qc.n_nan_pos = int((~np.isfinite(lat) | ~np.isfinite(lon)).sum())
    qc.n_nan_speed = int((~np.isfinite(speed)).sum())

    # Position must never be invented -- interpolating it would create truth the
    # benchmark then scores against. Drop those samples outright.
    posok = np.isfinite(lat) & np.isfinite(lon)
    if qc.n_nan_pos:
        qc.warnings.append(f"dropped {qc.n_nan_pos} row(s) with non-finite lat/lon")
        t, lat, lon, speed = t[posok], lat[posok], lon[posok], speed[posok]
        sub = sub.iloc[posok]
    if len(t) < 2:
        raise ValueError(f"{path}: fewer than 2 usable rows after cleaning")

    # Speed gaps ARE interpolated: speed is a smooth physical signal, short
    # dropouts are chip artefacts, and leaving NaN makes every window that
    # touches one score NaN (which nanmedian then hides).
    nan_v = ~np.isfinite(speed)
    if nan_v.any():
        if nan_v.all():
            raise ValueError(f"{path}: speed column {speed_col!r} is entirely non-numeric")
        speed = np.interp(t, t[~nan_v], speed[~nan_v])
        qc.warnings.append(f"interpolated {int(nan_v.sum())} NaN speed sample(s)")
    if speed_is_kmph:
        speed = speed / 3.6
    # 60 m/s = 216 km/h. No ground-vehicle log reaches it, so exceeding it is
    # near-proof the column is km/h being read as m/s. (The robust check is the
    # speed-vs-GNSS-track ratio in eval.report.describe; this is the cheap one.)
    if strict and np.nanmax(speed) > 60.0:
        qc.warnings.append(f"max speed {np.nanmax(speed):.0f} m/s "
                           f"({np.nanmax(speed)*3.6:.0f} km/h) -- is --kmph missing?")

    e, n = _lla_to_enu(lat, lon)
    if heading_col:
        heading = np.radians(pd.to_numeric(sub[heading_col], errors="coerce").to_numpy(float))
        bad = ~np.isfinite(heading)
        if bad.any():
            heading[bad] = np.interp(t[bad], t[~bad], np.unwrap(heading[~bad]))
        qc.heading_source = f"column {heading_col!r}"
    else:
        heading = _heading_from_xy(e, n, speed)
        qc.heading_source = "course-over-ground (speed-gated)"

    ecu = None
    if ecu_speed_col:
        ecu = pd.to_numeric(sub[ecu_speed_col], errors="coerce").to_numpy(float)
        nan_e = ~np.isfinite(ecu)
        if nan_e.any() and not nan_e.all():
            ecu = np.interp(t, t[~nan_e], ecu[~nan_e])
        if ecu_is_kmph:
            ecu = ecu / 3.6

    qc.parked_frac = float((speed <= 0.5).mean())
    qc.speed_min, qc.speed_max = float(speed.min()), float(speed.max())

    return Drive(vehicle_id or os.path.splitext(os.path.basename(path))[0],
                 t, e, n, speed, heading, ecu,
                 meta={"path": path, "columns": dict(
                     time=time_col, lat=lat_col, lon=lon_col, speed=speed_col,
                     heading=heading_col, ecu_speed=ecu_speed_col),
                     "units": dict(time=unit, speed_kmph=speed_is_kmph, ecu_kmph=ecu_is_kmph)},
                 qc=qc)


def drive_id_from_path(path, root):
    """Stable, collision-free id: the path relative to root, minus extension.

    Basenames collide -- real datasets are laid out per-vehicle, so
    vehicleA/drive1.csv and vehicleB/drive1.csv are both 'drive1'. Downstream,
    report.py keys outage windows by this id in a dict, so a collision silently
    drops a drive AND scores the survivor against another vehicle's row indices.
    """
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    return os.path.splitext(rel)[0].replace(os.sep, "/")


def load_dir(root, pattern="*.csv", *, recursive=True, **cols) -> list[Drive]:
    paths = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True)
                   if recursive else glob.glob(os.path.join(root, pattern)))
    if not paths:
        raise FileNotFoundError(f"no {pattern} under {root}")
    drives = []
    for p in paths:
        drives.append(load_csv(p, vehicle_id=drive_id_from_path(p, root), **cols))
    ids = [d.vehicle_id for d in drives]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate drive ids after path-relative naming: {dup}")
    return drives


def vehicle_split(drives, holdout):
    """Split by vehicle_id for cross-vehicle eval (Phase 4 uses this)."""
    hold = {holdout} if isinstance(holdout, str) else set(holdout)
    tr = [d for d in drives if d.vehicle_id not in hold]
    te = [d for d in drives if d.vehicle_id in hold]
    return tr, te


def ecu_doppler_scale(drive, *, min_speed=8.0):
    """Per-vehicle wheel-speed scale vs Doppler ground truth, fit on fast segments.

    Wheel/ECU speed is a few % optimistic (tyre wear/pressure). Train on the
    CORRECTED label speed_ecu/scale, or your 3% along-track budget is spent
    before training starts. Returns scale k such that true ~= ecu / k.
    """
    if drive.speed_ecu is None:
        return 1.0
    m = (drive.speed > min_speed) & (drive.speed_ecu > min_speed)
    m &= np.isfinite(drive.speed) & np.isfinite(drive.speed_ecu)
    if m.sum() < 100:
        return 1.0
    # k minimising |ecu - k*doppler|^2 through origin
    d, ecu = drive.speed[m], drive.speed_ecu[m]
    return float((ecu @ d) / (d @ d))


def synth_drive(vehicle_id="synth", duration=900.0, hz=10.0, seed=0, straight=False) -> Drive:
    """A believable ~15 min urban drive: cruise / stop / turn phases + a spiral.

    Ground-truth speed and heading are exact; ENU is the integral of them. Used
    so the whole Phase-0 harness runs with no dataset. speed_ecu is speed * 1.04
    (a 4% tyre-scale error) so ecu_doppler_scale() has something to recover.
    """
    rng = np.random.default_rng(seed)
    n = int(duration * hz)
    t = np.arange(n) / hz
    speed = np.full(n, 15.0)      # ~54 kmph cruise
    yawrate = np.zeros(n)         # rad/s
    i = 0
    while i < n:
        seg = int(rng.uniform(8, 40) * hz)
        kind = (rng.choice(["cruise", "stop"], p=[0.8, 0.2]) if straight
                else rng.choice(["cruise", "stop", "turn", "spiral"], p=[0.5, 0.2, 0.2, 0.1]))
        j = min(i + seg, n)
        if kind == "cruise":
            speed[i:j] = rng.uniform(8, 25)
        elif kind == "stop":
            ramp = np.linspace(speed[i-1] if i else 15, 0, j - i)
            speed[i:j] = np.clip(ramp, 0, None)
        elif kind == "turn":
            speed[i:j] = rng.uniform(5, 12)
            yawrate[i:j] = rng.choice([-1, 1]) * np.radians(rng.uniform(10, 25))
        else:  # spiral ramp: constant strong yaw, slow
            speed[i:j] = rng.uniform(4, 8)
            yawrate[i:j] = rng.choice([-1, 1]) * np.radians(rng.uniform(20, 35))
        i = j
    heading = np.cumsum(yawrate) / hz
    e = np.cumsum(speed * np.sin(heading)) / hz
    n_ = np.cumsum(speed * np.cos(heading)) / hz

    # --- synthesize a 6-axis IMU (vehicle frame: x fwd, y left, z up) ---
    # The learnable speed cue: road-excitation vibration amplitude grows with
    # speed (real effect). A net can read speed off a 2 s window's vibration
    # statistics -- this is why the whole approach works, so bake it in honestly.
    vib = 0.3 + 0.06 * speed                        # m/s^2 std, speed-dependent
    a_fwd = np.gradient(speed, 1.0 / hz)            # longitudinal specific force
    a_lat = speed * yawrate                         # centripetal (turns encode v)
    acc = np.stack([
        a_fwd + rng.normal(0, vib, n),
        a_lat + rng.normal(0, vib, n),
        9.81 + rng.normal(0, vib, n),               # gravity on z + vertical vibration
    ], axis=1)
    g_bias = np.radians(rng.normal(0, 0.2, 3))      # per-drive constant MEMS bias
    gyro = np.stack([
        rng.normal(0, np.radians(1.0), n) + g_bias[0],  # roll/pitch rate: road jitter
        rng.normal(0, np.radians(1.0), n) + g_bias[1],
        yawrate + g_bias[2] + np.radians(rng.normal(0, 0.05, n)),
    ], axis=1)
    qc = DriveQC(path="<synthetic>", sha256="", rows_in_file=n, rows_used=n,
                 time_unit="s", hz=hz, duration_s=float(t[-1]),
                 parked_frac=float((speed <= 0.5).mean()),
                 speed_min=float(speed.min()), speed_max=float(speed.max()),
                 heading_source="exact (synthetic)")
    return Drive(vehicle_id, t, e, n_, speed, heading, speed_ecu=speed * 1.04,
                 gyro_z=gyro[:, 2], acc=acc, gyro=gyro,
                 meta={"synthetic": True, "hz": hz, "seed": seed}, qc=qc)


def _selfcheck():
    d = synth_drive(duration=120, seed=1)
    assert len(d) == 1200 and d.t[1] - d.t[0] == 0.1
    # ENU displacement magnitude must be <= path length (integral of speed)
    disp = np.hypot(d.e[-1] - d.e[0], d.n[-1] - d.n[0])
    path = np.trapezoid(d.speed, d.t)
    assert disp <= path + 1e-6, (disp, path)
    # ecu scale recovery: injected 1.04, expect ~1.04
    k = ecu_doppler_scale(d)
    assert abs(k - 1.04) < 0.02, k
    print(f"io_vnbd ok: N={len(d)} path={path:.0f}m disp={disp:.0f}m ecu_scale={k:.3f}")


if __name__ == "__main__":
    _selfcheck()
