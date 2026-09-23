"""The nn speed head reads speed off a noisy vibration estimate, so its learned
map suffers regression dilution: v_pred ~= a + b*v_true with b < 1. That shows
up in sigma-calibration as a speed-proportional under-prediction (z_mean < 0),
which a Kalman filter cannot safely weight. These pin the affine recalibration
that fixes it.

Skipped when torch or the trained checkpoint is absent.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from data.io_vnbd import synth_drive
from data.outage import make_outages
from model.dataset import STEP


@pytest.fixture(scope="module")
def net():
    from model.nn_model import load_net, DEFAULT
    import os
    if not os.path.exists(DEFAULT):
        pytest.skip("no trained checkpoint")
    return load_net()


def _gather(net, calib):
    from model.nn_model import _raw_predict, apply_calib
    # held out from BOTH training (seeds 1000+) and calibration fit (1050-54)
    drives = [synth_drive(f"synth{i}", duration=900, seed=i) for i in range(3)]
    vp, vt, sg = [], [], []
    for d in drives:
        for o in make_outages(d, seed=0):
            starts, mu, sig = _raw_predict(net, d, o.i0, o.i1)
            v, s = apply_calib(mu, sig, calib)
            for st, vv, ss in zip(starts, v, s):
                seg = d.speed[st:st + STEP]
                if len(seg):
                    vp.append(vv); vt.append(seg.mean()); sg.append(ss)
    vp, vt, sg = map(np.array, (vp, vt, sg))
    z = (vp - vt) / np.clip(sg, 1e-6, None)
    return z.mean(), z.var(), float(np.mean(vp - vt))


def test_checkpoint_carries_a_calibration():
    from model.nn_model import get_calib
    c = get_calib()
    assert set(c) == {"a", "b", "s"}
    assert 0.7 < c["b"] < 1.0, "b should capture the dilution (scale < 1)"


def test_raw_head_is_biased(net):
    from model.nn_model import IDENTITY_CALIB
    z_mean, _, bias = _gather(net, IDENTITY_CALIB)
    assert z_mean < -0.3, f"expected the documented raw bias, got z_mean={z_mean}"
    assert bias < -0.3, "raw head under-predicts speed"


def test_calibration_removes_the_bias_on_held_out_data(net):
    from model.nn_model import get_calib
    z_mean, z_var, bias = _gather(net, get_calib())
    assert abs(z_mean) <= 0.3, f"z_mean {z_mean} still biased"
    assert 0.7 <= z_var <= 1.4, f"z_var {z_var} out of band"
    assert abs(bias) < 0.5, f"speed bias {bias} m/s too large"


def test_apply_calib_identity_is_a_noop():
    from model.nn_model import apply_calib, IDENTITY_CALIB
    mu = np.array([5.0, 10.0]); sig = np.array([1.0, 2.0])
    v, s = apply_calib(mu, sig, IDENTITY_CALIB)
    assert np.allclose(v, mu) and np.allclose(s, sig)
