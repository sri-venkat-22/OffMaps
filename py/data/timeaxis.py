"""Time-axis decoding and repair. Split out of io_vnbd because getting this
wrong is silent and total.

The old heuristic was `t -= t[0]; if t.max() > 1e6: t /= 1000`. It is correct
for unix-seconds and for long millisecond logs, and wrong by 1e3-1e6x for
nanoseconds (which is what our OWN Android logger writes -- t_ns is
elapsedRealtimeNanos), for microseconds, and for any millisecond log shorter
than 16.7 minutes. A 1000x-wrong time axis does not crash: it silently rescales
sample rate, outage-window length and every drift denominator downstream.

So: detect by SAMPLE INTERVAL, not by magnitude, and refuse to guess when the
answer is ambiguous.
"""
from __future__ import annotations
import numpy as np

# multiplier from the named unit to seconds
UNITS = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}

# A vehicle log samples somewhere in here. Outside it we are not looking at a
# drive, we are looking at a misparsed column -- so the band is deliberately
# narrow enough to make unit detection unique.
PLAUSIBLE_HZ = (0.5, 500.0)


class TimeAxisError(ValueError):
    """Raised when the time column cannot be decoded unambiguously."""


def detect_unit(raw, plausible_hz=PLAUSIBLE_HZ) -> str:
    """Infer the time unit from the median sample interval.

    Returns one of UNITS. Raises TimeAxisError with an actionable message when
    zero or several units are consistent -- guessing here is how you lose a
    factor of 1000 without noticing.
    """
    raw = np.asarray(raw, float)
    d = np.abs(np.diff(raw))
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        raise TimeAxisError("time column has no positive increments (all equal or NaN)")
    med = float(np.median(d))
    lo, hi = plausible_hz
    ok = [u for u, s in UNITS.items() if lo <= s / med <= hi]
    if len(ok) == 1:
        return ok[0]
    rates = ", ".join(f"{u}->{UNITS[u]/med:.4g} Hz" for u in UNITS)
    if not ok:
        raise TimeAxisError(
            f"cannot infer time unit: median step {med:g} implies no plausible rate "
            f"in {lo}-{hi} Hz ({rates}). Pass an explicit unit, e.g. --time-unit ms.")
    raise TimeAxisError(
        f"ambiguous time unit: {ok} all plausible ({rates}). Pass --time-unit explicitly.")


def decode(raw, unit="auto"):
    """Raw time column -> seconds since the first sample. Returns (t, unit)."""
    raw = np.asarray(raw, float)
    if raw.size == 0:
        raise TimeAxisError("empty time column")
    u = detect_unit(raw) if unit == "auto" else unit
    if u not in UNITS:
        raise TimeAxisError(f"unknown time unit {u!r}; want one of {sorted(UNITS)} or 'auto'")
    t = (raw - raw[0]) / UNITS[u]
    return t, u


def repair(t, *, gap_factor=3.0):
    """Make a time axis usable: sort, drop non-increasing samples, find gaps.

    Returns (keep, report). `keep` indexes the surviving samples IN ORDER, so
    the caller applies it to every column at once and they stay aligned. We drop
    rather than interpolate: an invented sample is a lie the benchmark would
    then score against.
    """
    t = np.asarray(t, float)
    n0 = len(t)
    finite = np.isfinite(t)
    order = np.argsort(t[finite], kind="stable")
    idx = np.flatnonzero(finite)[order]
    ts = t[idx]
    # keep strictly increasing samples only (kills exact duplicates and, after
    # sorting, the rows that had gone backwards)
    strictly = np.ones(len(ts), bool)
    strictly[1:] = np.diff(ts) > 0
    idx = idx[strictly]
    ts = ts[strictly]
    if len(ts) < 2:
        raise TimeAxisError(f"only {len(ts)} usable timestamps after repair")
    dt = np.diff(ts)
    med = float(np.median(dt))
    gaps = np.flatnonzero(dt > gap_factor * med)
    rep = dict(
        n_in=n0, n_out=len(ts),
        n_nonfinite=int((~finite).sum()),
        n_unsorted=int((np.diff(t[finite]) < 0).sum()) if finite.sum() > 1 else 0,
        n_dropped_nonincreasing=int(len(np.flatnonzero(finite)) - len(ts)),
        hz=1.0 / med if med > 0 else float("nan"),
        median_dt=med,
        max_dt=float(dt.max()),
        n_gaps=int(len(gaps)),
        gap_seconds=float(dt[gaps].sum()) if len(gaps) else 0.0,
        duration_s=float(ts[-1] - ts[0]),
    )
    return idx, rep


def _selfcheck():
    secs = np.arange(3600, dtype=float)
    cases = [("s", 1.725e9 + secs), ("ms", (1.725e9 + secs) * 1e3),
             ("ns", secs * 1e9), ("us", (1.725e9 + secs) * 1e6),
             ("ms", (1.725e9 + np.arange(600.0)) * 1e3)]     # short ms log: old code failed this
    for want, raw in cases:
        t, u = decode(raw)
        assert u == want, (want, u)
        assert abs(t[-1] - (len(raw) - 1)) < 1e-6, (want, t[-1])
    for hz, unit in [(10.0, "ms"), (100.0, "ns"), (400.0, "ns"), (1.0, "s")]:
        raw = np.arange(2000) / hz * UNITS[unit]
        assert detect_unit(raw) == unit, (hz, unit, detect_unit(raw))
    t = np.array([0., 1., 2., 2., 1., 3., 20., 21.])       # dup + backwards + gap
    keep, rep = repair(t)
    assert list(t[keep]) == [0., 1., 2., 3., 20., 21.], t[keep]
    assert rep["n_gaps"] == 1 and rep["n_out"] == 6, rep
    print(f"timeaxis ok: units {[u for u,_ in cases]} decoded; repair {rep['n_in']}->{rep['n_out']} "
          f"gaps={rep['n_gaps']} hz={rep['hz']:.2f}")


if __name__ == "__main__":
    _selfcheck()
