"""Deterministic GNSS-outage generator + frozen manifest.

An outage is a window [i0,i1) into a Drive where GNSS is hidden. At i0 the model
knows true pos+velocity; it dead-reckons to i1; error at i1 vs truth is scored.

Windows are labelled by mean speed and by heading change (turn vs straight) so
the report can slice, but the headline aggregation is per-duration. Fixed seed
=> the same test set every run. dump/load_manifest freezes it: commit the JSON,
never regenerate for reported numbers.

Windows are cut by TIME, not by sample count. On synth (uniform 10 Hz) the two
are identical; on a real log with dropouts a fixed sample count is not a fixed
duration -- a "10 s" window measured 15 s of real driving, so the per-duration
table's own x-axis was wrong. Windows that straddle a gap, or that cover no
distance (the vehicle was parked), are rejected rather than scored: both make
drift% meaningless, and a rejected window is visible while a NaN one is not.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import json, os, platform, sys
import numpy as np

DURATIONS = (10, 30, 60, 120, 180)  # seconds, per PS
MANIFEST_VERSION = 2


@dataclass
class Outage:
    drive_id: str
    i0: int
    i1: int
    duration_s: float          # nominal label (the table column)
    mean_speed: float
    heading_change_deg: float
    t0: float = 0.0            # actual clock time at i0, seconds into the drive
    t1: float = 0.0            # actual clock time at i1-1
    span_s: float = 0.0        # ACTUAL elapsed seconds -- may differ from duration_s
    dist_m: float = 0.0        # ground distance travelled inside the window
    max_dt: float = 0.0        # largest sample gap inside the window

    @property
    def turn(self): return self.heading_change_deg > 20.0

    @property
    def speed_band(self):
        return "low" if self.mean_speed < 8 else "mid" if self.mean_speed < 18 else "high"


def _hz(d): return 1.0 / np.median(np.diff(d.t))


def make_outages(drive, durations=DURATIONS, n_per_duration=8, seed=0, stride_s=5.0,
                 *, min_dist_m=20.0, max_gap_s=None, span_tol=0.10, allow_overlap=False):
    """Sample candidate windows per duration, fixed rng.

    min_dist_m  drop windows covering less ground than this: drift% = err/dist
                explodes or goes NaN when the vehicle was parked, and those
                windows are trivially easy, so scoring them flatters the model.
    max_gap_s   drop windows containing a sample gap larger than this
                (default: 3x the drive's median sample interval).
    span_tol    drop windows whose ACTUAL elapsed time differs from the nominal
                duration by more than this fraction.
    allow_overlap  windows within one duration are disjoint unless set.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(drive.t, float)
    med_dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    if max_gap_s is None:
        max_gap_s = 3.0 * med_dt if med_dt > 0 else np.inf
    hz = 1.0 / med_dt if med_dt > 0 else float("nan")
    speed = np.asarray(drive.speed, float)
    out, rejected = [], {"span": 0, "gap": 0, "dist": 0, "overlap": 0, "short": 0}

    for dur in durations:
        # window end is the first sample at least `dur` seconds after the start
        i1s = np.searchsorted(t, t + float(dur), side="left")
        stride = max(1, int(round(stride_s * hz))) if hz == hz else 1
        starts = np.arange(0, len(t), stride)
        starts = starts[i1s[starts] < len(t)]
        if len(starts) == 0:
            rejected["short"] += 1
            continue
        starts = starts.copy()
        rng.shuffle(starts)
        taken = []
        for i0 in starts:
            if len(taken) >= n_per_duration:
                break
            i1 = int(i1s[i0]) + 1                       # inclusive of the exit sample
            if i1 - i0 < 2:
                rejected["short"] += 1; continue
            span = float(t[i1 - 1] - t[i0])
            if abs(span - dur) > span_tol * dur:
                rejected["span"] += 1; continue
            seg_dt = np.diff(t[i0:i1])
            mx = float(seg_dt.max()) if len(seg_dt) else 0.0
            if mx > max_gap_s:
                rejected["gap"] += 1; continue
            dist = float(np.trapezoid(speed[i0:i1], t[i0:i1]))
            if dist < min_dist_m:
                rejected["dist"] += 1; continue
            if not allow_overlap and any(i0 < b and a < i1 for a, b in taken):
                rejected["overlap"] += 1; continue
            hdg = np.unwrap(drive.heading[i0:i1])
            taken.append((i0, i1))
            out.append(Outage(
                drive.vehicle_id, int(i0), int(i1), float(dur),
                float(speed[i0:i1].mean()),
                float(np.degrees(np.abs(hdg[-1] - hdg[0]))),
                t0=float(t[i0]), t1=float(t[i1 - 1]), span_s=span,
                dist_m=dist, max_dt=mx,
            ))
    make_outages.last_rejected = rejected
    return out


# ---------------------------------------------------------------- manifest ---

def _env():
    import numpy
    v = {"python": sys.version.split()[0], "numpy": numpy.__version__,
         "platform": platform.platform()}
    try:
        import pandas; v["pandas"] = pandas.__version__
    except Exception:
        pass
    return v


def tilde_home(x):
    """Rewrite the user's home dir as `~` in strings (recursively in lists/dicts).

    Manifests, reports and checkpoint metadata get committed; the absolute
    path adds nothing to reproducibility (sha256 pins the data) but leaks the
    local username.
    """
    home = os.path.expanduser("~")
    if isinstance(x, str):
        return "~" + x[len(home):] if x == home or x.startswith(home + os.sep) else x
    if isinstance(x, dict):
        return {k: tilde_home(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(tilde_home(v) for v in x)
    return x


def _drive_fingerprint(d):
    """Pin the DATA, not just an id string.

    The old manifest stored drive_id + row indices only. Indices into a
    re-exported CSV (one more dropped row, a different loader version) describe
    completely different windows while looking identical, so a "frozen" number
    could not actually be reproduced or audited.
    """
    fp = dict(drive_id=d.vehicle_id, n=len(d),
              t0=float(d.t[0]), t1=float(d.t[-1]),
              dist_m=float(np.trapezoid(d.speed, d.t)))
    if getattr(d, "qc", None) is not None:
        fp.update(sha256=d.qc.sha256, path=d.qc.path, rows_in_file=d.qc.rows_in_file,
                  rows_used=d.qc.rows_used, time_unit=d.qc.time_unit, hz=d.qc.hz)
    if d.meta.get("synthetic"):
        fp.update(synthetic=True, seed=d.meta.get("seed"), hz=d.meta.get("hz"))
    if d.meta.get("columns"):
        fp["columns"] = d.meta["columns"]
    if d.meta.get("units"):
        fp["units"] = d.meta["units"]
    return fp


def dump_manifest(outages, path, *, drives=None, params=None):
    doc = {
        "version": MANIFEST_VERSION,
        "params": params or {},
        "env": _env(),
        "drives": [_drive_fingerprint(d) for d in (drives or [])],
        "windows": [asdict(o) for o in outages],
    }
    with open(path, "w") as f:
        json.dump(tilde_home(doc), f, indent=2)


def load_manifest(path):
    """Return (outages, doc). Accepts the v1 bare-list format too."""
    with open(path) as f:
        doc = json.load(f)
    if isinstance(doc, list):                      # v1: bare list of windows
        return [Outage(**o) for o in doc], {"version": 1, "windows": doc,
                                            "drives": [], "params": {}, "env": {}}
    return [Outage(**o) for o in doc["windows"]], doc


def verify_manifest(doc, drives):
    """Check a loaded manifest still describes THESE drives. Returns problems."""
    problems = []
    by_id = {d.vehicle_id: d for d in drives}
    pinned = {f["drive_id"]: f for f in doc.get("drives", [])}
    if not pinned:
        problems.append("manifest pins no drive fingerprints (v1 format) -- cannot verify")
    for did, fp in pinned.items():
        d = by_id.get(did)
        if d is None:
            problems.append(f"{did}: pinned in manifest but not loaded")
            continue
        if fp.get("n") != len(d):
            problems.append(f"{did}: row count {len(d)} != pinned {fp.get('n')}")
        got = getattr(d, "qc", None)
        if fp.get("sha256") and got is not None and got.sha256 and fp["sha256"] != got.sha256:
            problems.append(f"{did}: file sha256 differs from pinned")
        if fp.get("columns") and d.meta.get("columns") and fp["columns"] != d.meta["columns"]:
            problems.append(f"{did}: column mapping differs from pinned {fp['columns']}")
        if fp.get("units") and d.meta.get("units") and fp["units"] != d.meta["units"]:
            problems.append(f"{did}: unit flags differ from pinned {fp['units']}")
        # content, not just shape: two drives can share an id, a length and a
        # rate and still be different drives (a re-export, or another synth seed)
        for k, tol in (("t0", 1e-6), ("t1", 1e-6), ("dist_m", 1e-3)):
            if fp.get(k) is not None:
                got = {"t0": float(d.t[0]), "t1": float(d.t[-1]),
                       "dist_m": float(np.trapezoid(d.speed, d.t))}[k]
                if abs(got - fp[k]) > tol * max(1.0, abs(fp[k])):
                    problems.append(f"{did}: {k} {got:.6g} != pinned {fp[k]:.6g}")
        if fp.get("synthetic") and fp.get("seed") != d.meta.get("seed"):
            problems.append(f"{did}: synthetic seed {d.meta.get('seed')} != pinned {fp.get('seed')}")
    for did in by_id:
        if pinned and did not in pinned:
            problems.append(f"{did}: loaded but not pinned in manifest")
    return problems


def _selfcheck():
    from data.io_vnbd import synth_drive
    d = synth_drive(duration=600, seed=2)
    outs = make_outages(d, seed=0)
    assert all(o.i1 <= len(d) and o.i0 < o.i1 for o in outs)
    assert make_outages(d, seed=0)[0].i0 == outs[0].i0, "not deterministic"
    durs = {o.duration_s for o in outs}
    assert durs <= set(DURATIONS)
    for o in outs:
        assert abs(o.span_s - o.duration_s) <= 0.10 * o.duration_s, (o.duration_s, o.span_s)
        assert o.dist_m >= 20.0
    for dur in durs:                                   # disjoint within a duration
        g = sorted([o for o in outs if o.duration_s == dur], key=lambda o: o.i0)
        assert all(b.i0 >= a.i1 for a, b in zip(g, g[1:])), f"overlap at {dur}s"
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "m.json")
    dump_manifest(outs, p, drives=[d], params={"seed": 0})
    outs2, doc = load_manifest(p)
    assert len(outs2) == len(outs) and doc["version"] == MANIFEST_VERSION
    assert verify_manifest(doc, [d]) == [], verify_manifest(doc, [d])
    d2 = synth_drive(duration=600, seed=99)
    assert verify_manifest(doc, [d2]), "must detect a swapped drive"
    print(f"outage ok: {len(outs)} windows, durations={sorted(durs)}, "
          f"turns={sum(o.turn for o in outs)}, rejected={make_outages.last_rejected}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _selfcheck()
