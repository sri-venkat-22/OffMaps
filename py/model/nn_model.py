"""Registers the `nn` model: NN speed + gyro heading, dead-reckoned.

Phase 2 has no filter yet (that's Phase 3), so this is the physics baseline
with the NN replacing frozen speed: predict 1 s displacement each second from
the IMU window, lay it along the gyro-integrated heading. Emits per-sample
(v_pred, sigma) so the harness can score sigma calibration.
"""
from __future__ import annotations
import os, numpy as np, torch
from model.tcn import SpeedNet
from model.features import window_features, WIN
from model.dataset import STEP
from eval.models import model

DEFAULT = os.path.join(os.path.dirname(__file__), "nn.pt")
_CACHE = {}
_CALIB = {}
_ESKF = {}          # path -> fusion settings stored with that checkpoint (None = defaults)
IDENTITY_CALIB = {"a": 0.0, "b": 1.0, "s": 1.0}


def load_net(path=None):
    """Load (and cache) a checkpoint. path=None means the module's DEFAULT, looked
    up at CALL time so a caller can repoint every registered model (nn, eskf,
    eskf_map) at another checkpoint, e.g. validate_realdata --nn-ckpt."""
    path = path or DEFAULT
    if path not in _CACHE:
        ckpt = torch.load(path, map_location="cpu")
        net = SpeedNet(); net.load_state_dict(ckpt["state"])
        net.eval(); _CACHE[path] = net
        _CALIB[path] = ckpt.get("calib", IDENTITY_CALIB)
        _ESKF[path] = ckpt.get("eskf_cfg")
    return _CACHE[path]


def get_eskf_cfg(path=None):
    """Fusion settings tuned for this checkpoint's speed/sigma (None -> core_bridge defaults)."""
    path = path or DEFAULT
    load_net(path)
    return _ESKF.get(path)


def get_calib(path=None):
    path = path or DEFAULT
    load_net(path)
    return _CALIB.get(path, IDENTITY_CALIB)


def _raw_predict(net, drive, i0, i1):
    """Uncalibrated per-second displacement + sigma over [i0,i1)."""
    starts = list(range(i0, i1, STEP))
    X = np.empty((len(starts), *window_features(drive.acc[:WIN], drive.gyro[:WIN]).shape), np.float32)
    for k, s in enumerate(starts):
        ws = max(0, s + STEP - WIN)                 # 2 s window ending at the step's end
        acc, gyro = drive.acc[ws:ws + WIN], drive.gyro[ws:ws + WIN]
        if len(acc) < WIN:                          # pad at the very start
            acc = np.pad(acc, ((WIN - len(acc), 0), (0, 0)), mode="edge")
            gyro = np.pad(gyro, ((WIN - len(gyro), 0), (0, 0)), mode="edge")
        X[k] = window_features(acc, gyro)
    with torch.no_grad():
        mu, logvar, _, _ = net(torch.from_numpy(X))
    return np.array(starts), mu.numpy(), np.exp(0.5 * logvar.numpy())


def apply_calib(mu, sigma, calib):
    """Affine mean + variance recalibration -> (v, sigma).

    The net reads speed off a vibration-std estimate computed from a 20-sample
    window; that estimate is noisy, so the learned map suffers regression
    dilution -- v_pred ~= a + b*v_true with b < 1, a speed-proportional
    under-prediction that shows up in sigma-calibration as z_mean < 0 (worse at
    speed). It is not fixable by the loss (it is feature noise, not variance
    weighting), but it is a clean affine error, so we invert it: v = (mu-a)/b,
    and scale sigma to match (sigma/b for the linear map, times a temperature s
    so z-variance -> 1). (a,b,s) are fit ONCE on held-out val drives -- the
    regression analogue of temperature-scaling a classifier -- and stored in the
    checkpoint. Identity calib (a=0,b=1,s=1) reproduces the raw net.
    """
    a, b, s = calib["a"], calib["b"], calib["s"]
    v = (mu - a) / b
    sig = np.clip(sigma / b * s, 1e-6, None)
    return v, sig


def predict_steps(net, drive, i0, i1, calib=None):
    """Per-second displacement + sigma over [i0,i1). Returns (starts, v_step, sig_step)."""
    starts, mu, sigma = _raw_predict(net, drive, i0, i1)
    if calib is None:
        # find which cached path this net belongs to (train.py injects the net
        # into _CACHE and its calib into _CALIB under the same key)
        calib = next((_CALIB[p] for p, m in _CACHE.items() if m is net), None) \
            or get_calib()
    v, sig = apply_calib(mu, sigma, calib)
    return starts, v, sig


@model("nn")
def nn(drive, o):
    net = load_net()
    starts, v_step, sig_step = predict_steps(net, drive, o.i0, o.i1)
    N = o.i1 - o.i0
    t = drive.t[o.i0:o.i1]; dt = np.diff(t, prepend=t[0])
    h = drive.heading[o.i0] + np.cumsum(drive.gyro_z[o.i0:o.i1] * dt)   # gyro heading
    vp = np.empty(N); sg = np.empty(N)
    for s, v, sig in zip(starts, v_step, sig_step):
        lo = s - o.i0; hi = min(lo + STEP, N)
        vp[lo:hi] = v; sg[lo:hi] = sig
    e = drive.e[o.i0] + np.cumsum(vp * np.sin(h) * dt)
    n = drive.n[o.i0] + np.cumsum(vp * np.cos(h) * dt)
    return e, n, vp, sg
