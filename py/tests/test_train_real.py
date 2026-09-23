"""Regression for the real-data SpeedNet trainer (model/train_real.py).

Two properties decide whether its headline number is honest, so both are pinned:

  - the split is BY DRIVE: every synced segment lands in exactly one split, and a
    source drive can never appear in two (window-level leakage flatters the score);
  - the calibration is chosen on VALIDATION error only. On real IO-VNBD the
    Phase-2 affine map (fit on 2 heterogeneous val drives, slope b~0.39) raised
    val MAE 5.53 -> 8.03 m/s and roughly doubled test speed error; the guarded
    rule keeps it only when it lowers val MAE, else variance-only (z_var -> 1).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from data.io_vnbd import synth_drive
from model import train_real as TR
from model.tcn import SpeedNet


def _named(vid, seed, dur=120):
    d = synth_drive(vid, duration=dur, seed=seed)
    d.vehicle_id = vid
    return d


def test_split_is_by_drive_and_disjoint():
    ds = [_named("M (Driver B)/S-M#0", 1), _named("M (Driver B)/S-M#2", 2),
          _named("S (Driver A)/S1/S-S1#0", 3), _named("S (Driver A)/S3b/S-S3b#1", 4),
          _named("S (Driver A)/S3a/S-S3a#0", 5), _named("Y (Driver D)/Y1/S-Y1#3", 6)]
    parts = TR.split_drives(ds)
    ids = {k: [d.vehicle_id for d in v] for k, v in parts.items()}
    assert ids["train"] == ["M (Driver B)/S-M#0", "M (Driver B)/S-M#2", "S (Driver A)/S1/S-S1#0"]
    assert ids["val"] == ["S (Driver A)/S3b/S-S3b#1"]
    assert sorted(ids["test"]) == ["S (Driver A)/S3a/S-S3a#0", "Y (Driver D)/Y1/S-Y1#3"]
    # both segments of one drive stay together (never split across train/test)
    assert all("S-M" not in i for i in ids["val"] + ids["test"])


def test_unassigned_drive_is_refused():
    with pytest.raises(ValueError):
        TR.split_drives([_named("Vta (Driver E)/Vta02/S-Vta2#0", 7)])


def test_calibration_pick_never_raises_val_error():
    torch.manual_seed(0)
    net = SpeedNet().eval()                      # untrained: affine fit is unreliable
    val = [_named("S (Driver A)/S3b/S-S3b#0", 11, dur=200)]
    calib, info = TR.calibrate(net, val)
    maes = {k: info[f"val_{k}"]["mae"] for k in ("affine", "variance_only")}
    assert info["pick"] == min(maes, key=maes.get)
    if info["pick"] == "variance_only":
        assert calib["a"] == 0.0 and calib["b"] == 1.0
        assert abs(info["val_variance_only"]["z_var"] - 1.0) < 0.05   # sigma temperature fitted
