"""The scoring the whole project is judged on. Written before any model exists
so nobody can tune the model to a benchmark defined after the fact.

Per outage: end-point error, split into along-/cross-track relative to the true
heading at exit; drift as % of distance travelled (the ISRO metric, <10%).
Across outages: CEP50/95 and per-duration table. Plus sigma calibration: a
heteroscedastic speed head is only fusable if z=(v_pred-v_true)/sigma has unit
variance -- report it as an exit gate, not a nicety.

Real-data notes:
- the exit heading is averaged over a short window, not read from one sample.
  On 1 Hz GNSS a single course-over-ground sample carries tens of degrees of
  noise, and that noise rotates along-track error into the reported cross-track
  (lane) number -- the number Phase 5 is judged on.
- every aggregate reports how many windows actually contributed. A count that
  includes windows whose value was NaN is a count of the wrong thing.
"""
from __future__ import annotations
import numpy as np

SIGMA_Z_VAR_BAND = (0.7, 1.4)
SIGMA_Z_MEAN_MAX = 0.3


def _circmean(h):
    return float(np.arctan2(np.mean(np.sin(h)), np.mean(np.cos(h))))


def _exit_heading(drive, o, win_s=2.0):
    """True heading at window exit, averaged over the last win_s seconds.

    Falls back to net displacement over that same tail when the vehicle is
    stopped at exit -- the tail, not the whole window, because a window that
    contains a turn has a net-displacement heading pointing nowhere the vehicle
    was actually facing.
    """
    t = drive.t
    lo = max(o.i0, int(np.searchsorted(t, t[o.i1 - 1] - win_s, side="left")))
    lo = min(lo, o.i1 - 1)
    seg = slice(lo, o.i1)
    if np.mean(drive.speed[seg]) >= 0.5:
        return _circmean(drive.heading[seg])
    de = drive.e[o.i1 - 1] - drive.e[lo]
    dn = drive.n[o.i1 - 1] - drive.n[lo]
    if np.hypot(de, dn) > 1e-6:
        return float(np.arctan2(de, dn))
    de = drive.e[o.i1 - 1] - drive.e[o.i0]
    dn = drive.n[o.i1 - 1] - drive.n[o.i0]
    if np.hypot(de, dn) > 1e-6:
        return float(np.arctan2(de, dn))
    return float(drive.heading[o.i1 - 1])


def _along_cross(err_e, err_n, drive, o):
    """Decompose end error onto the true exit heading.

    Frame is (east, north) with z up, so a +90 deg (CCW) rotation of the heading
    unit vector is the vehicle's LEFT: fwd=(sin h, cos h) -> (-cos h, sin h).
    Positive cross = error to the vehicle's left.
    """
    h = _exit_heading(drive, o)
    fwd = np.array([np.sin(h), np.cos(h)])          # unit along-track
    left = np.array([-np.cos(h), np.sin(h)])        # unit cross-track (+ = left)
    err = np.array([err_e, err_n])
    return float(err @ fwd), float(err @ left)


def window_distance(drive, o, source="speed"):
    """Ground distance travelled inside the window -- the drift denominator.

    'speed'  integral of the speed column (what ISRO's % is defined against)
    'track'  path length of the GNSS ENU track
    They agree on good data. They disagree when the speed column is wheel speed
    with a scale error, which would rescale every reported drift% without
    touching the position error -- so the report prints both and warns.
    """
    sl = slice(o.i0, o.i1)
    if source == "track":
        return float(np.sum(np.hypot(np.diff(drive.e[sl]), np.diff(drive.n[sl]))))
    return float(np.trapezoid(drive.speed[sl], drive.t[sl]))


def score_outage(drive, o, pred, *, dist_source="speed"):
    pe, pn, vp, sg = pred
    err_e = pe[-1] - drive.e[o.i1 - 1]
    err_n = pn[-1] - drive.n[o.i1 - 1]
    end_err = float(np.hypot(err_e, err_n))
    dist = window_distance(drive, o, dist_source)
    along, cross = _along_cross(err_e, err_n, drive, o)
    r = dict(drive=o.drive_id, dur=o.duration_s, band=o.speed_band, turn=o.turn,
             dist=dist, dist_track=window_distance(drive, o, "track"),
             end_err=end_err, along=along, cross=cross,
             span_s=getattr(o, "span_s", o.duration_s),
             drift_pct=100.0 * end_err / dist if dist > 1e-6 else np.nan)
    if vp is not None:
        vt = drive.speed[o.i0:o.i1]
        vp = np.asarray(vp, float)
        if len(vp) == len(vt):
            r["speed_mae"] = float(np.mean(np.abs(vp - vt)))
            r["speed_bias"] = float(np.mean(vp - vt))
            if sg is not None:
                r["_z"] = (vp - vt) / np.clip(np.asarray(sg, float), 1e-6, None)
    return r


def cep(errs, p):
    """Circular error probable. NaNs are dropped BEFORE the emptiness check --
    a non-empty all-NaN input used to raise IndexError out of np.percentile."""
    errs = np.asarray(errs, float)
    errs = errs[np.isfinite(errs)]
    return float(np.percentile(errs, p)) if errs.size else np.nan


def _med(xs):
    xs = np.asarray(xs, float)
    xs = xs[np.isfinite(xs)]
    return float(np.median(xs)) if xs.size else np.nan


def per_duration_table(rows):
    out = {}
    for dur in sorted({r["dur"] for r in rows}):
        g = [r for r in rows if r["dur"] == dur]
        de = [r["end_err"] for r in g]
        drift = [r["drift_pct"] for r in g]
        n_drift = int(np.isfinite(np.asarray(drift, float)).sum())
        rec = dict(
            n=len(g),                 # windows in this bucket
            n_drift=n_drift,          # windows that actually produced a drift%
            drift_med=_med(drift),
            cep50=cep(de, 50), cep95=cep(de, 95),
            along_med=_med([abs(r["along"]) for r in g]),
            cross_med=_med([abs(r["cross"]) for r in g]),
            span_med=_med([r.get("span_s", dur) for r in g]),
            dist_med=_med([r["dist"] for r in g]),
        )
        mae = [r["speed_mae"] for r in g if "speed_mae" in r]
        if mae:
            rec["speed_mae"] = _med(mae)
        out[dur] = rec
    return out


def sigma_calibration(v_pred, v_true, sigma):
    """A sigma is honest when z=(v_pred-v_true)/sigma is standard normal.

    z_var alone is not the gate: a model with a constant 2 m/s bias and a
    correctly-scaled sigma has z_var = 1.0 and z_mean = 4.0, and a Kalman filter
    fed that speed walks straight off the road. So gate both, and say which
    half failed.
    """
    v_pred, v_true, sigma = (np.asarray(x, float) for x in (v_pred, v_true, sigma))
    ok = np.isfinite(v_pred) & np.isfinite(v_true) & np.isfinite(sigma)
    v_pred, v_true, sigma = v_pred[ok], v_true[ok], sigma[ok]
    if v_pred.size == 0:
        return dict(z_var=np.nan, z_mean=np.nan, reliability=[], n=0,
                    passed=False, why="no finite samples")
    z = (v_pred - v_true) / np.clip(sigma, 1e-6, None)
    z_var, z_mean = float(z.var()), float(z.mean())
    rel = []
    edges = np.unique(np.quantile(sigma, np.linspace(0, 1, 6)))
    if edges.size >= 2:
        for lo, hi in zip(edges[:-1], edges[1:]):
            # closed on the right for the top bin, else the largest sigma is
            # silently excluded from every reliability curve
            m = (sigma >= lo) & (sigma <= hi if hi == edges[-1] else sigma < hi)
            if m.sum() > 5:
                rel.append((float(sigma[m].mean()), float((v_pred[m] - v_true[m]).std())))
    lo, hi = SIGMA_Z_VAR_BAND
    why = []
    if not (lo < z_var < hi):
        why.append(f"z_var {z_var:.2f} outside [{lo},{hi}]")
    if abs(z_mean) > SIGMA_Z_MEAN_MAX:
        why.append(f"|z_mean| {abs(z_mean):.2f} > {SIGMA_Z_MEAN_MAX} (biased speed)")
    return dict(z_var=z_var, z_mean=z_mean, reliability=rel, n=int(v_pred.size),
                passed=not why, why="; ".join(why) or "ok")


def print_table(table, model_name):
    print(f"\n== {model_name} ==")
    has_mae = any("speed_mae" in r for r in table.values())
    hdr = (f"{'dur(s)':>7} {'n':>3} {'nd':>3} {'span':>6} {'drift%':>7} {'CEP50':>7} "
           f"{'CEP95':>8} {'along':>7} {'cross':>7}")
    print(hdr + (f" {'vMAE':>6}" if has_mae else ""))
    for dur, r in table.items():
        line = (f"{dur:>7.0f} {r['n']:>3} {r['n_drift']:>3} {r['span_med']:>6.1f} "
                f"{r['drift_med']:>7.1f} {r['cep50']:>7.1f} {r['cep95']:>8.1f} "
                f"{r['along_med']:>7.1f} {r['cross_med']:>7.1f}")
        if has_mae:
            line += f" {r.get('speed_mae', float('nan')):>6.2f}"
        print(line)
    if any(r["n_drift"] != r["n"] for r in table.values()):
        print("   n = windows scored, nd = windows that produced a finite drift%")


def _selfcheck():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.io_vnbd import synth_drive
    from data.outage import make_outages
    from eval.models import REGISTRY
    d = synth_drive(duration=600, seed=3)
    outs = make_outages(d, seed=0)
    rows = [score_outage(d, o, REGISTRY["truth"](d, o)) for o in outs]
    assert max(r["end_err"] for r in rows) < 1e-6, "oracle should be exact"
    rows0 = [score_outage(d, o, REGISTRY["zero"](d, o)) for o in outs]
    assert np.nanmedian([r["drift_pct"] for r in rows0]) > 50, "zero must drift hard"
    # cep must not explode on an all-NaN column
    assert np.isnan(cep([np.nan, np.nan], 50)) and np.isnan(cep([], 50))
    # sigma calibration: honest sigma -> z_var ~ 1 and passes
    rng = np.random.default_rng(0); vt = rng.normal(15, 3, 5000)
    vp = vt + rng.normal(0, 0.5, 5000); sg = np.full(5000, 0.5)
    c = sigma_calibration(vp, vt, sg)
    assert c["passed"] and 0.7 < c["z_var"] < 1.4, c
    # a 2 m/s biased model with a correctly scaled sigma passes z_var and must
    # still be caught by z_mean
    cb = sigma_calibration(vt + 2.0 + rng.normal(0, 0.5, 5000), vt, sg)
    assert 0.7 < cb["z_var"] < 1.4 and not cb["passed"], cb
    print(f"metrics ok: zero drift_med={np.nanmedian([r['drift_pct'] for r in rows0]):.0f}% "
          f"z_var={c['z_var']:.2f} bias-gate={cb['why']}")


if __name__ == "__main__":
    _selfcheck()
