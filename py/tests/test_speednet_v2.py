"""SpeedNet feature version 2 (model/features.py): the contract a v2 net is trained on.

- heading-free: rotating the horizontal axes about up leaves every channel unchanged,
  so the phone (which levels with gravity but does not know which way is forward)
  feeds the net the same numbers as the training frame;
- a batched call equals the single-window call (training batches, the phone does one);
- the turn cue reads v = a_lat / yaw on a constant-speed circle;
- checkpoints carry their spec and every loader windows by it; a checkpoint without
  one is version 1 (2 s), as every net before v2 was.
"""
import math
import numpy as np
import torch

from model import features as F
from model.nn_model import build_net, load_net, get_calib
from model.dataset import SeqDataset, STEP


def _drive(T=200, v=12.0, yaw=0.15, seed=0):
    rng = np.random.default_rng(seed)
    acc = np.zeros((T, 3)); gyro = np.zeros((T, 3))
    acc[:, 1] = v * yaw                                  # centripetal (left) on a circle
    acc[:, 2] = 9.81
    gyro[:, 2] = yaw
    acc += rng.normal(0, 0.3, acc.shape); gyro += rng.normal(0, 0.01, gyro.shape)
    return acc, gyro


def _rot_z(x, th):
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return x @ R.T


def test_shape_and_scale():
    acc, gyro = _drive()
    f = F.window_features_v2(acc, gyro)
    assert f.shape == (F.C2, 200) and f.dtype == np.float32
    assert np.all(np.isfinite(f))


def test_invariant_to_heading_of_horizontal_axes():
    acc, gyro = _drive()
    f0 = F.window_features_v2(acc, gyro)
    for th in (0.3, 1.7, -2.9):
        f1 = F.window_features_v2(_rot_z(acc, th), _rot_z(gyro, th))
        np.testing.assert_allclose(f1, f0, atol=1e-5)


def test_batched_equals_single():
    wins = [_drive(seed=s) for s in range(4)]
    A = np.stack([a for a, _ in wins]); G = np.stack([g for _, g in wins])
    fb = F.window_features_v2(A, G)
    for k, (a, g) in enumerate(wins):
        np.testing.assert_allclose(fb[k], F.window_features_v2(a, g), atol=1e-6)


def test_turn_cue_reads_speed_on_a_circle():
    for v in (6.0, 12.0, 20.0):
        f = F.window_features_v2(*_drive(v=v, yaw=0.15))
        turn_v = f[8] * F._SCALE2[8]
        assert f[9].mean() > 0.95                        # turning throughout
        assert abs(np.median(turn_v) - v) < 0.1 * v
    straight = F.window_features_v2(*_drive(yaw=0.0))
    assert straight[9].mean() < 0.2 and np.median(straight[8]) == 0.0


def test_box_is_a_centred_mean_with_held_edges():
    x = np.arange(10, dtype=np.float32)[:, None]
    y = F.box(x, 5)[:, 0]
    np.testing.assert_allclose([y[5], y[0], y[9]], [np.mean(x[3:8]), np.mean([0, 0, 0, 1, 2]),
                                                    np.mean([7, 8, 9, 9, 9])], rtol=1e-6)


def test_checkpoint_spec_round_trip(tmp_path):
    net = build_net({"version": 2, "win": 50})
    assert net.win == 50 and net.inp.in_channels == F.C2
    p = str(tmp_path / "v2.pt")
    torch.save({"state": net.state_dict(), "calib": {"a": 0.0, "b": 1.0, "s": 1.0}, "feat": net.feat}, p)
    got = load_net(p)
    assert got.feat == {"version": 2, "win": 50} and got.feat_fn is F.window_features_v2
    old = str(tmp_path / "v1.pt")                        # a pre-v2 checkpoint: no "feat" key
    torch.save({"state": build_net().state_dict()}, old)
    assert load_net(old).win == F.WIN and get_calib(old) == {"a": 0.0, "b": 1.0, "s": 1.0}


def test_dataset_windows_by_spec():
    class D:                                             # the few Drive fields SeqDataset reads
        pass
    d = D(); n = 1200
    acc, gyro = _drive(T=n)
    d.acc, d.gyro, d.gyro_z = acc.astype(np.float32), gyro.astype(np.float32), gyro[:, 2]
    d.t = np.arange(n) / 10.0; d.speed = np.full(n, 12.0)
    D.__len__ = lambda self: n
    ds = SeqDataset([d], feat=2, win=200, augment_data=False)
    X, dt, cl = ds[0]
    assert tuple(X.shape) == (30, F.C2, 200)
    np.testing.assert_allclose(dt.numpy(), 12.0 * (STEP - 1) / 10.0, rtol=1e-5)   # trapezoid over 10 samples
