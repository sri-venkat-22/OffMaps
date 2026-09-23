"""The time axis is the one thing a wrong answer never announces: a 1000x error
rescales sample rate, window length and every drift denominator, and the report
still prints a plausible table."""
import numpy as np
import pytest
from data import timeaxis as T


@pytest.mark.parametrize("unit,hz", [("s", 1.0), ("ms", 1.0), ("ms", 10.0),
                                     ("ns", 1.0), ("ns", 100.0), ("ns", 400.0),
                                     ("us", 1.0), ("us", 10.0)])
def test_detects_unit_by_sample_interval(unit, hz):
    raw = 1.725e9 * T.UNITS[unit] + np.arange(2000) / hz * T.UNITS[unit]
    assert T.detect_unit(raw) == unit


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_decode_gives_seconds_from_zero(unit):
    n, hz = 600, 10.0
    raw = 1.725e9 * T.UNITS[unit] + np.arange(n) / hz * T.UNITS[unit]
    t, got = T.decode(raw)
    assert got == unit
    assert t[0] == 0.0
    # abs, not rel: subtracting a ~1.7e9 epoch in float64 leaves ~0.5 us of
    # slack (and ~256 ns for a 1.7e18 ns epoch). Both are far below the ms
    # resolution vehicle navigation needs, but they are not 1e-9 relative.
    assert t[-1] == pytest.approx((n - 1) / hz, abs=1e-4)


def test_short_millisecond_log_is_not_read_as_seconds():
    """The old heuristic keyed on t.max() > 1e6, so any ms log under 16.7 min
    was left in milliseconds -- a 1000x error on exactly the short recordings a
    first real drive produces."""
    raw = (1.725e9 + np.arange(600.0)) * 1e3          # 10 min at 1 Hz, ms
    t, unit = T.decode(raw)
    assert unit == "ms" and t[-1] == pytest.approx(599.0)


def test_nanosecond_log_is_not_read_as_microseconds():
    """Our own Android logger writes elapsedRealtimeNanos."""
    raw = np.arange(3600.0) * 1e9
    t, unit = T.decode(raw)
    assert unit == "ns" and t[-1] == pytest.approx(3599.0)


def test_refuses_to_guess_rather_than_be_wrong():
    with pytest.raises(T.TimeAxisError):
        T.detect_unit(np.arange(100) * 1e12)          # no plausible rate
    with pytest.raises(T.TimeAxisError):
        T.decode(np.zeros(10))                        # no increments


def test_explicit_unit_overrides_detection():
    raw = np.arange(1000.0) * 1e3
    t, unit = T.decode(raw, "us")
    assert unit == "us" and t[-1] == pytest.approx(0.999)


def test_repair_sorts_drops_duplicates_and_finds_gaps():
    t = np.array([0., 1., 2., 2., 1., 3., 20., 21.])
    keep, rep = T.repair(t)
    assert list(t[keep]) == [0., 1., 2., 3., 20., 21.]
    assert np.all(np.diff(t[keep]) > 0)
    assert rep["n_gaps"] == 1
    assert rep["n_in"] == 8 and rep["n_out"] == 6


def test_repair_handles_nonfinite():
    t = np.array([0., 1., np.nan, 3., 4.])
    keep, rep = T.repair(t)
    assert list(t[keep]) == [0., 1., 3., 4.]
    assert rep["n_nonfinite"] == 1
