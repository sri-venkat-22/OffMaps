"""Loader behaviour on real-log pathologies. Each test names one thing that
used to pass silently and now must not."""
import numpy as np
import pytest
from data.io_vnbd import (load_csv, load_dir, ecu_doppler_scale, synth_drive,
                          drive_id_from_path, _heading_from_xy)
from tests.fixtures import MessySpec, REALISTIC, write_messy


@pytest.fixture
def realistic(tmp_path):
    p, truth = write_messy(tmp_path / "d", REALISTIC)
    return p, truth


def _load(p, **kw):
    kw.setdefault("speed_col", "speed")
    return load_csv(p, **kw)


def test_mixed_units_do_not_couple(realistic):
    """Doppler in km/h + wheel speed in m/s. One --kmph flag for both made
    ecu_doppler_scale return exactly 1.0, silently deleting the calibration
    signal Phase 4's cross-vehicle gate is built on."""
    p, truth = realistic
    d = _load(p, ecu_speed_col="wheel_speed", speed_is_kmph=True, ecu_is_kmph=False)
    assert ecu_doppler_scale(d) == pytest.approx(truth["ecu_scale"], abs=0.01)


def test_coupled_units_are_still_the_default(tmp_path):
    """Both columns km/h: ecu_is_kmph defaults to speed_is_kmph."""
    p, truth = write_messy(tmp_path, MessySpec(n=1800, speed_kmph=True, ecu_kmph=True))
    d = _load(p, ecu_speed_col="wheel_speed", speed_is_kmph=True)
    assert ecu_doppler_scale(d) == pytest.approx(truth["ecu_scale"], abs=0.01)


def test_time_is_strictly_increasing_after_load(realistic):
    p, _ = realistic
    d = _load(p)
    assert np.all(np.diff(d.t) > 0)
    assert d.t[0] == 0.0


def test_duplicate_and_backwards_timestamps_are_dropped_and_reported(tmp_path):
    p, _ = write_messy(tmp_path, MessySpec(n=1200, n_dup=3, n_backwards=2, time_unit="s"))
    d = _load(p)
    assert np.all(np.diff(d.t) > 0)
    assert d.qc.n_dropped_nonincreasing >= 3
    assert any("non-increasing" in w for w in d.qc.warnings)


def test_nan_speed_is_interpolated_not_left_to_poison_windows(realistic):
    p, _ = realistic
    d = _load(p)
    assert np.isfinite(d.speed).all()
    assert any("NaN speed" in w for w in d.qc.warnings)


def test_nan_position_rows_are_dropped_never_invented(tmp_path):
    import pandas as pd
    p, _ = write_messy(tmp_path, MessySpec(n=1200, time_unit="s"))
    df = pd.read_csv(p); df.loc[[10, 11, 12], "latitude"] = np.nan; df.to_csv(p, index=False)
    d = _load(p)
    assert len(d) == len(df) - 3
    assert d.qc.n_nan_pos == 3


def test_gaps_are_measured_not_hidden(realistic):
    p, _ = realistic
    d = _load(p)
    assert d.qc.n_gaps > 0 and d.qc.max_dt > d.qc.duration_s / len(d)


def test_max_gap_can_reject_a_drive(realistic):
    p, _ = realistic
    with pytest.raises(ValueError, match="gap exceeds"):
        _load(p, max_gap_s=1.5)


def test_heading_does_not_spin_while_parked(realistic):
    """Course over ground from raw np.gradient is GNSS jitter at standstill --
    a median of ~90 deg/sample on a 25%-parked log, which poisons the turn
    label and the along/cross basis."""
    p, _ = realistic
    d = _load(p, speed_is_kmph=True)
    step = np.degrees(np.abs(np.diff(np.unwrap(d.heading))))
    parked = (d.speed < 0.5)[:-1]
    assert parked.sum() > 100
    assert np.median(step[parked]) < 1.0


def test_heading_still_tracks_real_turns(realistic):
    p, _ = realistic
    d = _load(p, speed_is_kmph=True)
    step = np.degrees(np.abs(np.diff(np.unwrap(d.heading))))
    moving = (d.speed > 5.0)[:-1]
    assert np.percentile(step[moving], 95) > 2.0


def test_typo_in_required_column_raises(realistic):
    p, _ = realistic
    with pytest.raises(KeyError, match="lattitude"):
        _load(p, lat_col="lattitude")


def test_typo_in_optional_column_raises_instead_of_silently_skipping(realistic):
    """--heading-col that does not exist used to fall through to derived
    heading, so a typo and an intentional omission looked identical."""
    p, _ = realistic
    with pytest.raises(KeyError, match="heading"):
        _load(p, heading_col="bearng")


def test_case_insensitive_column_match(realistic):
    p, _ = realistic
    d = _load(p, lat_col="LATITUDE", lon_col="Longitude")
    assert len(d) > 0


def test_wrong_units_are_flagged_in_qc(realistic):
    p, _ = realistic
    d = _load(p)                                  # km/h read as m/s
    assert any("kmph" in w for w in d.qc.warnings)


def test_drive_ids_do_not_collide_across_subdirectories(tmp_path):
    """vehicleA/drive1.csv and vehicleB/drive1.csv both became 'drive1'.
    report.py keys windows by this id in a dict, so one drive vanished and the
    survivor was scored against another vehicle's row indices."""
    for v in ("vehicleA", "vehicleB"):
        write_messy(tmp_path / v, MessySpec(n=1200, time_unit="s"), "drive1.csv")
    drives = load_dir(str(tmp_path), speed_col="speed")
    ids = [d.vehicle_id for d in drives]
    assert len(set(ids)) == 2 == len(ids)
    assert ids == ["vehicleA/drive1", "vehicleB/drive1"]


def test_drive_id_from_path_is_relative_and_stable():
    assert drive_id_from_path("/root/a/b/x.csv", "/root") == "a/b/x"


def test_qc_records_provenance(realistic):
    p, _ = realistic
    d = _load(p)
    assert len(d.qc.sha256) == 64
    assert d.qc.rows_in_file >= d.qc.rows_used > 0
    assert d.qc.time_unit == "ms"
    assert d.meta["columns"]["speed"] == "speed"


def test_synth_drive_unchanged(): 
    """Phases 1-5 are graded on synth; the generator must not have moved."""
    d = synth_drive(duration=120, seed=1)
    assert len(d) == 1200
    assert d.t[1] - d.t[0] == pytest.approx(0.1)
    assert ecu_doppler_scale(d) == pytest.approx(1.04, abs=0.02)
    assert d.acc.shape == (1200, 3) and d.gyro.shape == (1200, 3)
