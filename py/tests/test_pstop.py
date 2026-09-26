"""SpeedNet's third output, p(stopped) (model/pstop.py).

  - the stop log-odds is the softmax's class-0 log-odds, and Platt fitting recovers
    known parameters;
  - the shipped profile carries the sidecar's calibration, and the sidecar's
    cross-fitted numbers show calibration HELPED (it is fit on held-out outputs);
  - the phone computes the same probability (Features.pStop: tests/test_kotlin_ports.py);
  - the edge engine reports it: stopped windows read high, driving reads low.
"""
import json, os
import numpy as np
import pytest

from model.pstop import stop_logit, apply_pstop, fit_platt, metrics, sidecar_path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE = os.path.join(ROOT, "android/app/src/main/assets/nn_real.profile.json")
SIDECAR = sidecar_path(os.path.join(HERE, "model", "nn_real.pt"))


def test_stop_logit_is_class0_log_odds():
    rng = np.random.default_rng(0)
    cls = rng.normal(0, 3, (50, 4))
    p = np.exp(cls) / np.exp(cls).sum(1, keepdims=True)
    np.testing.assert_allclose(stop_logit(cls), np.log(p[:, 0] / (1 - p[:, 0])), atol=1e-9)


def test_platt_recovers_parameters():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, 20000)
    y = rng.random(20000) < apply_pstop(z, {"a": 1.7, "b": -0.9})
    cal = fit_platt(z, y)
    assert abs(cal["a"] - 1.7) < 0.1 and abs(cal["b"] + 0.9) < 0.1


def test_profile_carries_the_sidecar_and_calibration_helped():
    with open(SIDECAR) as fh:
        side = json.load(fh)
    with open(PROFILE) as fh:
        prof = json.load(fh)
    assert prof["pstop"] == {"a": side["a"], "b": side["b"]}
    rep = os.path.join(ROOT, "out", "pstop", "pstop.json")
    if os.path.exists(rep):
        r = json.load(open(rep))
        assert r["all_crossfit"]["ece"] < r["all_raw"]["ece"]
        assert r["all_crossfit"]["brier"] <= r["all_raw"]["brier"]
        assert r["all_crossfit"]["auc"] > 0.95


def test_edge_engine_reports_p_stopped():
    pytest.importorskip("onnxruntime")
    import phase6_check as P
    from edge_engine import OnnxSpeedNet
    from model.features import WIN
    net = OnnxSpeedNet(P.load_profile("nn_real"))
    rng = np.random.default_rng(3)
    still_a = np.c_[rng.normal(0, 0.01, (WIN, 2)), 9.81 + rng.normal(0, 0.01, WIN)]
    still_g = rng.normal(0, 0.001, (WIN, 3))
    net.predict(still_a, still_g)
    p_still = net.p_stop
    drive_a = np.c_[rng.normal(0, 0.8, (WIN, 2)), 9.81 + rng.normal(0, 1.2, WIN)]
    drive_g = rng.normal(0, 0.05, (WIN, 3))
    net.predict(drive_a, drive_g)
    assert 0.0 <= net.p_stop <= 1.0 and 0.0 <= p_still <= 1.0
    assert p_still > 0.5 > net.p_stop, (p_still, net.p_stop)
