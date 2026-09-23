"""Regression for the checkpoint-carried ESKF fusion settings (core_bridge.eskf_config).

On real IO-VNBD the synthetic fusion settings made the ESKF WORSE than the speed
net it fuses (41.6 % vs 19.1 % at 60 s): NN v < 0.5 fired false ZUPT/ZARU on slow
real driving, and the curvature update trusted a phone accelerometer that barely
tracks the car. model/tune_eskf.py tunes the knobs on VALIDATION drives and stores
them in the checkpoint. Pinned here:

  - ESKF_DEFAULT equals the settings every synthetic gate/parity test was
    validated with, and a checkpoint without "eskf_cfg" (model/nn.pt) gets exactly
    those -- the phone path and old numbers cannot move silently;
  - a checkpoint's eskf_cfg overrides them, and an explicit cfg overrides both;
  - zupt_v=None really disables ZUPT (a slow-but-moving window keeps its speed).
"""
import numpy as np
import pytest

pytest.importorskip("torch")
CB = pytest.importorskip("core_bridge")
import model.nn_model as NNM
from data.io_vnbd import synth_drive
from data.outage import Outage


def test_defaults_are_the_synthetic_validated_settings(monkeypatch):
    assert CB.ESKF_DEFAULT == dict(arw=float(np.radians(0.5)), brw=float(np.radians(0.02)),
                                   srw=3.0, zupt_v=0.5, curv=True, curv_sigma=2.0, sig_scale=1.0,
                                   handover_s=0.0, mm_cross_sigma=1.5, mm_heading_sigma_deg=3.0,
                                   mm_keep_speed=False)
    monkeypatch.setattr(CB, "get_eskf_cfg", lambda: None)          # e.g. model/nn.pt
    assert CB.eskf_config() == CB.ESKF_DEFAULT


def test_checkpoint_cfg_overrides_and_explicit_cfg_wins(monkeypatch):
    monkeypatch.setattr(CB, "get_eskf_cfg", lambda: {"zupt_v": None, "curv": False, "srw": 24.0})
    c = CB.eskf_config()
    assert c["zupt_v"] is None and c["curv"] is False and c["srw"] == 24.0
    assert c["arw"] == CB.ESKF_DEFAULT["arw"]                       # untouched knobs keep defaults
    assert CB.eskf_config({"srw": 1.0})["srw"] == 1.0


def test_zupt_disabled_keeps_slow_speed(monkeypatch):
    """A window where the NN reports 0.3 m/s: default settings clamp v to 0 (ZUPT),
    zupt_v=None must integrate the 0.3 m/s instead."""
    d = synth_drive("slow", duration=60, seed=3)
    d.speed[:] = 0.3
    o = Outage(drive_id="slow", i0=50, i1=350, duration_s=30.0, mean_speed=0.3,
               heading_change_deg=0.0)
    monkeypatch.setattr(CB, "predict_steps",
                        lambda net, drive, i0, i1, calib=None:
                        (np.arange(i0, i1, 10), np.full(len(range(i0, i1, 10)), 0.3),
                         np.full(len(range(i0, i1, 10)), 0.1)))
    monkeypatch.setattr(CB, "get_eskf_cfg", lambda: None)
    _, _, v_def, _ = CB._run(d, o)
    _, _, v_off, _ = CB._run(d, o, cfg={"zupt_v": None, "curv": False})
    assert v_def[-1] < 0.05                  # ZUPT clamped it
    assert abs(v_off[-1] - 0.3) < 0.05       # speed update kept it


def test_handover_holds_entry_speed_then_fuses_nn(monkeypatch):
    """handover_s: for the first seconds of an outage the filter keeps the entry
    (Doppler) speed -- the `physics` baseline, which wins short real outages -- and
    only then fuses the NN. Entry speed 10 m/s, NN says 5 m/s."""
    d = synth_drive("ho", duration=60, seed=4)
    d.speed[:] = 10.0
    o = Outage(drive_id="ho", i0=50, i1=350, duration_s=30.0, mean_speed=10.0,
               heading_change_deg=0.0)
    monkeypatch.setattr(CB, "predict_steps",
                        lambda net, drive, i0, i1, calib=None:
                        (np.arange(i0, i1, 10), np.full(len(range(i0, i1, 10)), 5.0),
                         np.full(len(range(i0, i1, 10)), 0.5)))
    monkeypatch.setattr(CB, "get_eskf_cfg", lambda: None)
    cfg = {"zupt_v": None, "curv": False}
    _, _, v, _ = CB._run(d, o, cfg={**cfg, "handover_s": 10.0})
    assert abs(v[95] - 10.0) < 1e-6          # 9.5 s in: still the held entry speed
    assert abs(v[-1] - 5.0) < 0.5            # after hand-over: the NN speed
    _, _, v0, _ = CB._run(d, o, cfg=cfg)     # default 0 s: NN from the first second
    assert abs(v0[95] - 5.0) < 0.5
