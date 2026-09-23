"""Window sampling and the frozen manifest. The benchmark's credibility rests
on these being what they claim: a 60 s window that measured 70 s of driving is
not a 60 s window, and a 'frozen' manifest that cannot be replayed is a file."""
import numpy as np
import pytest
from data.io_vnbd import load_csv, synth_drive
from data.outage import (make_outages, dump_manifest, load_manifest,
                         verify_manifest, DURATIONS, MANIFEST_VERSION)
from tests.fixtures import MessySpec, REALISTIC, write_messy


@pytest.fixture
def gappy(tmp_path):
    p, _ = write_messy(tmp_path, REALISTIC)
    return load_csv(p, speed_col="speed", ecu_speed_col="wheel_speed",
                    speed_is_kmph=True, ecu_is_kmph=False)


def test_window_span_matches_its_label_on_gappy_data(gappy):
    """Cut by sample count, a '10 s' window spanned up to 15 s of real driving
    because dropouts made the count-to-duration conversion a fiction."""
    for o in make_outages(gappy, seed=0):
        assert abs(o.span_s - o.duration_s) <= 0.10 * o.duration_s, o


def test_windows_do_not_straddle_gaps(gappy):
    med = float(np.median(np.diff(gappy.t)))
    for o in make_outages(gappy, seed=0):
        assert o.max_dt <= 3.0 * med + 1e-9


def test_stationary_windows_are_rejected_not_scored_as_nan(gappy):
    """A parked window has dist ~ 0, so drift% is NaN or enormous; nanmedian
    then hid it while the printed n still counted it."""
    outs = make_outages(gappy, seed=0, min_dist_m=20.0)
    assert outs
    assert all(o.dist_m >= 20.0 for o in outs)


def test_windows_within_a_duration_are_disjoint(gappy):
    outs = make_outages(gappy, seed=0)
    for dur in {o.duration_s for o in outs}:
        g = sorted([o for o in outs if o.duration_s == dur], key=lambda o: o.i0)
        assert all(b.i0 >= a.i1 for a, b in zip(g, g[1:])), f"overlap at {dur}s"


def test_overlap_can_be_re_enabled(gappy):
    a = make_outages(gappy, seed=0, allow_overlap=False)
    b = make_outages(gappy, seed=0, allow_overlap=True)
    assert len(b) >= len(a)


def test_rejections_are_counted_not_silent(gappy):
    make_outages(gappy, seed=0)
    rej = make_outages.last_rejected
    assert set(rej) == {"span", "gap", "dist", "overlap", "short"}
    assert sum(rej.values()) > 0


def test_deterministic_for_a_fixed_seed(gappy):
    a = make_outages(gappy, seed=0)
    b = make_outages(gappy, seed=0)
    assert [(o.i0, o.i1) for o in a] == [(o.i0, o.i1) for o in b]


def test_different_seed_gives_a_different_set(gappy):
    a = make_outages(gappy, seed=0)
    b = make_outages(gappy, seed=1)
    assert [(o.i0, o.i1) for o in a] != [(o.i0, o.i1) for o in b]


def test_windows_carry_their_own_clock_times(gappy):
    for o in make_outages(gappy, seed=0):
        assert o.t0 == pytest.approx(gappy.t[o.i0])
        assert o.t1 == pytest.approx(gappy.t[o.i1 - 1])


def test_uniform_synth_is_unaffected_by_time_based_cutting():
    """Phases 1-5 are graded on synth; on a uniform grid the time cut and the
    old sample-count cut must agree exactly."""
    d = synth_drive(duration=600, seed=2)
    for o in make_outages(d, seed=0):
        assert o.i1 - o.i0 == pytest.approx(o.duration_s * 10 + 1, abs=1)
        assert o.span_s == pytest.approx(o.duration_s, abs=0.11)


# ------------------------------------------------------------------ manifest --

def test_manifest_round_trips(tmp_path):
    d = synth_drive(duration=600, seed=2)
    outs = make_outages(d, seed=0)
    p = tmp_path / "m.json"
    dump_manifest(outs, p, drives=[d], params={"seed": 0})
    back, doc = load_manifest(p)
    assert doc["version"] == MANIFEST_VERSION
    assert [(o.i0, o.i1) for o in back] == [(o.i0, o.i1) for o in outs]


def test_manifest_pins_the_data_not_just_an_id(tmp_path):
    d = synth_drive(duration=600, seed=2)
    p = tmp_path / "m.json"
    dump_manifest(make_outages(d, seed=0), p, drives=[d])
    _, doc = load_manifest(p)
    assert verify_manifest(doc, [d]) == []
    # same id, same length, same rate -- different drive
    other = synth_drive(duration=600, seed=99)
    assert verify_manifest(doc, [other])


def test_manifest_detects_a_changed_csv(tmp_path):
    import pandas as pd
    p, _ = write_messy(tmp_path, MessySpec(n=1200, time_unit="s"))
    d = load_csv(p, speed_col="speed")
    mp = tmp_path / "m.json"
    dump_manifest(make_outages(d, seed=0), mp, drives=[d])
    _, doc = load_manifest(mp)
    assert verify_manifest(doc, [d]) == []
    df = pd.read_csv(p); df.iloc[:-5].to_csv(p, index=False)   # re-export, 5 fewer rows
    d2 = load_csv(p, speed_col="speed")
    problems = verify_manifest(doc, [d2])
    assert problems and any("row count" in x or "sha256" in x for x in problems)


def test_manifest_detects_changed_column_or_unit_flags(tmp_path):
    p, _ = write_messy(tmp_path, MessySpec(n=1800, speed_kmph=True, time_unit="s"))
    d = load_csv(p, speed_col="speed", speed_is_kmph=True)
    mp = tmp_path / "m.json"
    dump_manifest(make_outages(d, seed=0), mp, drives=[d])
    _, doc = load_manifest(mp)
    d_wrong = load_csv(p, speed_col="speed", speed_is_kmph=False)
    assert any("unit" in x for x in verify_manifest(doc, [d_wrong]))


def test_manifest_records_provenance(tmp_path):
    d = synth_drive(duration=600, seed=2)
    p = tmp_path / "m.json"
    dump_manifest(make_outages(d, seed=0), p, drives=[d],
                  params={"seed": 0, "n_per_duration": 8})
    _, doc = load_manifest(p)
    assert doc["params"]["seed"] == 0
    assert "numpy" in doc["env"] and "python" in doc["env"]
    assert doc["drives"][0]["drive_id"] == d.vehicle_id


def test_v1_manifest_still_loads(tmp_path):
    """out/manifest.json committed before this change is a bare list."""
    import json
    p = tmp_path / "v1.json"
    json.dump([dict(drive_id="synth0", i0=10, i1=110, duration_s=10.0,
                    mean_speed=15.0, heading_change_deg=0.0)], open(p, "w"))
    outs, doc = load_manifest(p)
    assert len(outs) == 1 and doc["version"] == 1
    assert verify_manifest(doc, [])          # warns that it cannot verify
