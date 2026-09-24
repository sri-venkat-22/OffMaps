"""Phase 8c/8e: heading aids on physics-exact rigs (phase8_heading_gate.py), locked in.

  - yaw rate about TRUE vertical: the app's old 0.5 s gravity projection ("fast")
    under-reads every turn by cos(atan(v*psidot/g)); "slow" fixes a car, only the
    lean-compensated "coord" fixes a two-wheeler (it leans with the phone);
  - a parked start (no GNSS course) is seeded from the magnetometer within its
    20 deg sigma, instead of an arbitrary heading;
  - the filter is told how unsure that seed is (idr_set_heading_sigma), so the
    first real GNSS course pulls it in.
"""
import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("core_bridge")
import phase8_heading_gate as G   # noqa: E402


def test_yaw_modes_car_and_two_wheeler():
    ok, r = G.gate_a()                       # 3 seeds x 20 min: 18 outages per mode
    assert ok, r


def test_magnetometer_seed():
    ok, r = G.gate_b()
    assert ok, r


def test_unknown_heading_is_corrected_by_first_course():
    from data.synth_rig import synth_rig
    from edge_engine import EdgeEngine, run
    import heading_aids as HA
    rig = synth_rig(90, 50, seed=11, park_s=30, heading0_deg=150.0)
    imu, gn = G._streams(rig)
    o = run(EdgeEngine(use_mag=False, head=None), imu, gn)
    i = int(60 * 50)                                    # 30 s after it started moving
    assert abs(np.degrees(HA.wrap(o[i, 3] - rig["heading"][i]))) < 5.0
