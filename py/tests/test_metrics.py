"""Scoring. These pin the numbers every phase is judged by, including the ones
that used to be quietly wrong or quietly absent."""
import numpy as np
import pytest
from data.io_vnbd import synth_drive
from data.outage import make_outages, Outage
from eval import metrics as M
from eval.models import REGISTRY


@pytest.fixture(scope="module")
def drive():
    return synth_drive(duration=600, seed=3)


@pytest.fixture(scope="module")
def outs(drive):
    return make_outages(drive, seed=0)


def test_oracle_scores_zero(drive, outs):
    rows = [M.score_outage(drive, o, REGISTRY["truth"](drive, o)) for o in outs]
    assert max(r["end_err"] for r in rows) < 1e-6
    assert max(r["drift_pct"] for r in rows) < 1e-6


def test_do_nothing_floor_drifts_hard(drive, outs):
    rows = [M.score_outage(drive, o, REGISTRY["zero"](drive, o)) for o in outs]
    assert np.nanmedian([r["drift_pct"] for r in rows]) > 50


# ---------------------------------------------------------------- geometry --

def test_cross_track_is_positive_to_the_vehicles_left():
    """(east, north) with z up: +90 deg CCW from fwd=(sin h, cos h) is
    (-cos h, sin h), which facing north is west -- the driver's left."""
    d = synth_drive(duration=60, seed=0)
    d.heading[:] = 0.0                      # due north
    d.speed[:] = 10.0
    d.e[:] = 0.0; d.n[:] = np.arange(len(d)) * 1.0
    o = Outage("x", 0, len(d), 60.0, 10.0, 0.0, span_s=60.0, dist_m=600.0)
    along, cross = M._along_cross(-10.0, 0.0, d, o)    # 10 m due WEST
    assert along == pytest.approx(0.0, abs=1e-9)
    assert cross == pytest.approx(+10.0)              # west == left of north
    along, cross = M._along_cross(+10.0, 0.0, d, o)    # 10 m due EAST
    assert cross == pytest.approx(-10.0)


def test_along_track_is_positive_ahead():
    d = synth_drive(duration=60, seed=0)
    d.heading[:] = 0.0; d.speed[:] = 10.0
    d.e[:] = 0.0; d.n[:] = np.arange(len(d)) * 1.0
    o = Outage("x", 0, len(d), 60.0, 10.0, 0.0, span_s=60.0, dist_m=600.0)
    along, cross = M._along_cross(0.0, 25.0, d, o)     # 25 m due NORTH = ahead
    assert along == pytest.approx(25.0) and cross == pytest.approx(0.0, abs=1e-9)


def test_exit_heading_is_averaged_not_a_single_noisy_sample():
    """One course-over-ground sample at 1 Hz carries tens of degrees of noise,
    which rotates along-track error into the reported cross-track number."""
    d = synth_drive(duration=60, seed=0, hz=1.0)
    d.speed[:] = 20.0
    d.heading[:] = 0.0
    rng = np.random.default_rng(0)
    d.heading[:] += rng.normal(0, np.radians(20), len(d))   # noisy COG
    o = Outage("x", 0, len(d), 60.0, 20.0, 0.0, span_s=60.0, dist_m=1200.0)
    h_avg = M._exit_heading(d, o)
    assert abs(np.degrees(h_avg)) < abs(np.degrees(d.heading[o.i1 - 1]))


def test_stopped_at_exit_uses_the_tail_not_the_whole_window():
    """The old fallback took net displacement across the entire window, which
    on a window containing a turn points nowhere the vehicle ever faced."""
    n = 600
    d = synth_drive(duration=60, seed=0, hz=10.0)
    d.speed[:] = 0.0
    d.e[:] = np.concatenate([np.linspace(0, 300, n // 2), np.full(n - n // 2, 300.0)])
    d.n[:] = np.concatenate([np.zeros(n // 2), np.linspace(0, 300, n - n // 2)])
    o = Outage("x", 0, n, 60.0, 0.0, 0.0, span_s=60.0, dist_m=600.0)
    h = M._exit_heading(d, o)
    # travelled east then north; the tail is northbound, the net is north-east
    assert abs(np.degrees(h)) < 20.0


# ------------------------------------------------------------- aggregation --

def test_cep_handles_all_nan_without_raising():
    """cep() checked len() before filtering NaN, so a non-empty all-NaN column
    raised IndexError out of np.percentile."""
    assert np.isnan(M.cep([np.nan, np.nan, np.nan], 50))
    assert np.isnan(M.cep([], 95))
    assert M.cep([1.0, np.nan, 3.0], 50) == pytest.approx(2.0)


def test_table_reports_how_many_windows_actually_scored():
    """n counted windows; drift_med was computed over a NaN-filtered subset.
    The printed sample size was wrong exactly when the data was bad."""
    rows = [dict(dur=60.0, end_err=10.0, along=1.0, cross=1.0, dist=100.0,
                 span_s=60.0, drift_pct=10.0) for _ in range(5)]
    rows += [dict(dur=60.0, end_err=10.0, along=1.0, cross=1.0, dist=0.0,
                  span_s=60.0, drift_pct=np.nan) for _ in range(3)]
    t = M.per_duration_table(rows)[60.0]
    assert t["n"] == 8 and t["n_drift"] == 5


def test_table_surfaces_speed_mae(drive, outs):
    rows = [M.score_outage(drive, o, REGISTRY["cv"](drive, o)) for o in outs]
    t = M.per_duration_table(rows)
    assert all("speed_mae" in r for r in t.values())


def test_table_carries_actual_span(drive, outs):
    rows = [M.score_outage(drive, o, REGISTRY["cv"](drive, o)) for o in outs]
    t = M.per_duration_table(rows)
    for dur, r in t.items():
        assert r["span_med"] == pytest.approx(dur, rel=0.11)


# ------------------------------------------------------- sigma calibration --

def test_honest_sigma_passes():
    rng = np.random.default_rng(0)
    vt = rng.normal(15, 3, 5000)
    c = M.sigma_calibration(vt + rng.normal(0, .5, 5000), vt, np.full(5000, .5))
    assert c["passed"] and 0.7 < c["z_var"] < 1.4 and abs(c["z_mean"]) < 0.3


def test_biased_speed_fails_even_with_a_perfectly_scaled_sigma():
    """z_var alone is not a gate: a 2 m/s bias with sigma=0.5 gives z_var=1.0
    and z_mean=4.0, and a filter fed that speed drives off the road."""
    rng = np.random.default_rng(0)
    vt = rng.normal(15, 3, 5000)
    c = M.sigma_calibration(vt + 2.0 + rng.normal(0, .5, 5000), vt, np.full(5000, .5))
    assert 0.7 < c["z_var"] < 1.4
    assert not c["passed"] and "z_mean" in c["why"]


def test_overconfident_sigma_fails():
    rng = np.random.default_rng(0)
    vt = rng.normal(15, 3, 5000)
    c = M.sigma_calibration(vt + rng.normal(0, 2.0, 5000), vt, np.full(5000, .5))
    assert not c["passed"] and "z_var" in c["why"]


def test_reliability_bins_include_the_largest_sigma():
    """The top quantile bin was half-open, so the maximum-sigma samples were
    dropped from every reliability curve."""
    rng = np.random.default_rng(0)
    sg = np.linspace(0.2, 2.0, 4000)
    vt = rng.normal(15, 3, 4000)
    c = M.sigma_calibration(vt + rng.normal(0, 1, 4000) * sg, vt, sg)
    assert c["reliability"]
    assert max(s for s, _ in c["reliability"]) > 1.5


def test_constant_sigma_does_not_silently_return_empty_bins():
    rng = np.random.default_rng(0)
    vt = rng.normal(15, 3, 1000)
    c = M.sigma_calibration(vt + rng.normal(0, .5, 1000), vt, np.full(1000, .5))
    assert c["n"] == 1000 and np.isfinite(c["z_var"])


def test_nan_samples_are_dropped():
    v = np.array([1.0, 2.0, np.nan, 4.0])
    c = M.sigma_calibration(v, np.zeros(4), np.ones(4))
    assert c["n"] == 3


# ------------------------------------------------------------- denominator --

def test_drift_denominator_sources_agree_on_clean_data(drive, outs):
    o = outs[0]
    a = M.window_distance(drive, o, "speed")
    b = M.window_distance(drive, o, "track")
    assert a == pytest.approx(b, rel=0.02)


def test_drift_denominator_source_is_selectable(drive, outs):
    o = outs[0]
    r1 = M.score_outage(drive, o, REGISTRY["cv"](drive, o), dist_source="speed")
    r2 = M.score_outage(drive, o, REGISTRY["cv"](drive, o), dist_source="track")
    assert r1["dist"] == pytest.approx(M.window_distance(drive, o, "speed"))
    assert r2["dist"] == pytest.approx(M.window_distance(drive, o, "track"))
