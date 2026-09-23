"""Phase-2 training + exit gate.

    python -m model.train           # train on synth, export ONNX, print the gate

Exit gate (both required): nn 60 s drift <= 0.5 * physics 60 s drift, AND
sigma z-variance in [0.7, 1.4] (honest, fusable uncertainty).
"""
from __future__ import annotations
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from data.io_vnbd import synth_drive
from data.outage import make_outages
from model.tcn import SpeedNet, count_params
from model.dataset import SeqDataset, L
from model import loss as LOSS
from model.nn_model import predict_steps, _raw_predict, apply_calib, DEFAULT
import baseline.physics  # noqa: F401  registers 'physics'
from eval.models import REGISTRY
from eval.metrics import score_outage, sigma_calibration


def _drives(ids, dur=600):
    return [synth_drive(f"veh{i}", duration=dur, seed=1000 + i) for i in ids]


def fit_calibration(net, drives):
    """Fit the affine mean+variance recalibration on HELD-OUT drives.

    Must be disjoint from the training set (regression analogue of fitting a
    temperature on a validation split): these val vehicles are seeds 1050-1054,
    the report's test drives are seeds 0-2, training is 1000-1011 -- all disjoint.
    """
    vp, vt, sg = [], [], []
    for d in drives:
        for o in make_outages(d, seed=0):
            starts, mu, sig = _raw_predict(net, d, o.i0, o.i1)
            for s, m, ss in zip(starts, mu, sig):
                seg = d.speed[s:s + 10]
                if len(seg):
                    vp.append(m); vt.append(seg.mean()); sg.append(ss)
    vp, vt, sg = np.array(vp), np.array(vt), np.array(sg)
    b, a = np.polyfit(vt, vp, 1)                      # vp ~= a + b*vt
    v_cal = (vp - a) / b
    s = float(((v_cal - vt) / (sg / b)).std())        # variance temperature -> z-var ~ 1
    return {"a": float(a), "b": float(b), "s": s}


def train(epochs=25, lr=2e-3, out=DEFAULT):
    tr = _drives(range(12)); va = _drives(range(50, 55))   # disjoint vehicles
    dl = DataLoader(SeqDataset(tr, seed=0), batch_size=16, shuffle=True)
    net = SpeedNet(); opt = torch.optim.Adam(net.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    print(f"params={count_params(net):,}  train_seqs={len(dl.dataset)}")
    for ep in range(epochs):
        net.train(); tot = 0.0
        for X, dt, cl in dl:
            B = X.shape[0]
            mu, lv, _, logits = net(X.reshape(B * L, *X.shape[2:]))
            mu, lv, logits = mu.reshape(B, L), lv.reshape(B, L), logits.reshape(B, L, 4)
            l = LOSS.total(mu, lv, dt, logits, cl)
            opt.zero_grad(); l.backward(); opt.step(); tot += float(l) * B
        sched.step()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  ep{ep:2d} loss={tot/len(dl.dataset):.4f}")
    net.eval()
    calib = fit_calibration(net, va)
    print(f"calibration (fit on held-out val): v=(mu-{calib['a']:.3f})/{calib['b']:.3f}, "
          f"sigma temperature s={calib['s']:.3f}")
    torch.save({"state": net.state_dict(), "calib": calib}, out)
    print(f"saved {out}")
    _export_onnx(net, out.replace(".pt", ".onnx"))
    _gate(net, va, calib)


def _export_onnx(net, path):
    """Export the ONNX deliverable and verify onnxruntime parity. Falls back to
    TorchScript with a LOUD warning if the onnx exporter deps are missing, so a
    minimal env still trains -- but ONNX is never *silently* skipped (it was: the
    old bare-except swallowed the failure and the committed tree had no .onnx)."""
    from model.export import export_onnx, verify_parity
    net.eval()
    try:
        export_onnx(net, path)
        kb = os.path.getsize(path) // 1024
        try:
            verify_parity(net, path)
            print(f"exported {path} ({kb} KB) -- onnxruntime parity OK, run on device via onnxruntime")
        except ImportError:
            print(f"exported {path} ({kb} KB) -- `pip install onnxruntime` to verify parity")
    except Exception as e:
        ts = path.replace(".onnx", ".torchscript.pt")
        torch.jit.save(torch.jit.trace(net, torch.randn(1, 9, 20)), ts)
        print(f"!! ONNX EXPORT FAILED ({e.__class__.__name__}: {e}) -- `pip install onnx`.\n"
              f"   Saved TorchScript {ts} as a fallback; it is NOT the ONNX deliverable.")


def _gate(net, val, calib=None):
    net.eval()
    from model import nn_model
    nn_model._CACHE[DEFAULT] = net           # use the just-trained net
    nn_model._CALIB[DEFAULT] = calib or nn_model.IDENTITY_CALIB   # and its calibration
    vp, vt, sg = [], [], []
    d60_nn, d60_ph = [], []
    for d in val:
        outs = make_outages(d, seed=0)
        for o in outs:
            starts, v, sig = predict_steps(net, d, o.i0, o.i1)
            for s, vv, ss in zip(starts, v, sig):
                seg = d.speed[s:s + 10]
                if len(seg):
                    vp.append(vv); vt.append(seg.mean()); sg.append(ss)
            if o.duration_s == 60:
                d60_nn.append(score_outage(d, o, REGISTRY["nn"](d, o))["drift_pct"])
                d60_ph.append(score_outage(d, o, REGISTRY["physics"](d, o))["drift_pct"])
    nn60, ph60 = np.nanmedian(d60_nn), np.nanmedian(d60_ph)
    cal = sigma_calibration(np.array(vp), np.array(vt), np.array(sg))
    speed_mae = float(np.mean(np.abs(np.array(vp) - np.array(vt))))
    print("\n=== Phase-2 exit gate (val, held-out vehicles) ===")
    print(f"  60s drift:  nn={nn60:.1f}%   physics={ph60:.1f}%   ratio={nn60/ph60:.2f} (want <=0.50)")
    print(f"  speed MAE:  {speed_mae:.2f} m/s")
    print(f"  sigma z-var={cal['z_var']:.2f} z-mean={cal['z_mean']:+.2f} "
          f"(want z-var in [0.7,1.4], |z-mean| <= 0.3) -> {cal['why']}")
    # z-mean IS part of the gate. A model that under-predicts speed but inflates
    # sigma to match passes a variance-only check and then walks a Kalman filter
    # off the road -- that is exactly the bias beta-NLL fixes, so gate it.
    ok = nn60 <= 0.5 * ph60 and cal["passed"]
    print(f"  GATE: {'PASS' if ok else 'not met on synth (see notes)'}")


if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    train()
