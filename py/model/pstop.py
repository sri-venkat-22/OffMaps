"""SpeedNet's third output: p(stopped), calibrated.

    PYTHONPATH=. python3 -m model.pstop                  # nn_real: fit, report, write the sidecar

SpeedNet has always had a motion-class head (stopped / straight / turning, trained with
weight 0.1 next to speed and sigma; model/dataset._motion_class), but nothing read it.
Raw, it ranks well (AUC ~0.98 for "stopped" on the val drives) and is badly calibrated:
the 4-way softmax almost never goes past 0.8, and on val a raw 0.6-0.8 means stopped
~70 % of the time. So p(stopped) is Platt-scaled on the log-odds of that class:

    z = cls[0] - logsumexp(cls[1:])          (log-odds of "stopped" vs the rest)
    p_stop = sigmoid(a * z + b)

(a, b) is fit on HELD-OUT outputs only: the val drives scored by the shipped net, and
every train drive scored by the out-of-fold SpeedNet that never saw it (model/oof/, the
same nets the fusion head was trained on) -- ~10 h instead of val's 1 h. Reported
numbers are cross-fitted by drive (fit on the other five drives, score the sixth).
Label: true speed over the 1 s step < 0.5 m/s, as in training.

Writes model/<ckpt stem>.pstop.json; model/export.py copies it into the phone profile
("pstop"), and SpeedNet.kt / edge_engine read it from there.
"""
from __future__ import annotations
import argparse, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

STOP_V = 0.5            # m/s, the training label's threshold (dataset._motion_class)


def stop_logit(cls):
    """Log-odds of class 0 (stopped) vs the other classes; cls (..., 4) raw logits."""
    cls = np.asarray(cls, float)
    rest = cls[..., 1:]
    m = rest.max(-1)
    return cls[..., 0] - (m + np.log(np.exp(rest - m[..., None]).sum(-1)))


def apply_pstop(z, cal):
    return 1.0 / (1.0 + np.exp(-(cal["a"] * np.asarray(z, float) + cal["b"])))


def fit_platt(z, y, iters=50):
    """Logistic regression y ~ sigmoid(a z + b) by Newton's method (2 parameters)."""
    z = np.asarray(z, float); y = np.asarray(y, float)
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(a * z + b)))
        w = p * (1 - p) + 1e-9
        g = np.array([np.sum((p - y) * z), np.sum(p - y)])
        H = np.array([[np.sum(w * z * z), np.sum(w * z)], [np.sum(w * z), np.sum(w)]]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.max(np.abs(step)) < 1e-10:
            break
    return {"a": float(a), "b": float(b)}


def metrics(p, y):
    """AUC, Brier, expected calibration error (10 bins), and precision/recall at 0.5."""
    p = np.asarray(p, float); y = np.asarray(y, bool)
    order = np.argsort(p); r = np.empty(len(p)); r[order] = np.arange(1, len(p) + 1)
    npos, nneg = y.sum(), (~y).sum()
    auc = (r[y].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[bins == k].mean() - y[bins == k].mean()) * (bins == k).mean() for k in range(10) if (bins == k).any())
    d = p > 0.5
    return dict(n=int(len(p)), stopped=float(y.mean()), auc=float(auc), brier=float(np.mean((p - y) ** 2)),
                ece=float(ece), precision=float(y[d].mean()) if d.any() else float("nan"),
                recall=float(d[y].mean()) if y.any() else float("nan"))


def drive_logits(net, drive):
    """Per 1 s step: (stop log-odds, true stopped) over a whole drive."""
    import torch
    from model.dataset import STEP
    fn, W = net.feat_fn, net.win
    starts = list(range(W - STEP, len(drive) - STEP, STEP))
    if not starts:
        return np.empty(0), np.empty(0, bool)
    A = np.stack([drive.acc[s + STEP - W:s + STEP] for s in starts]).astype(np.float32)
    G = np.stack([drive.gyro[s + STEP - W:s + STEP] for s in starts]).astype(np.float32)
    X = np.stack([fn(a, g) for a, g in zip(A, G)]) if net.feat["version"] == 1 else fn(A, G)
    with torch.no_grad():
        cls = np.concatenate([net(torch.from_numpy(X[i:i + 2048]))[3].numpy() for i in range(0, len(X), 2048)])
    y = np.array([drive.speed[s:s + STEP].mean() < STOP_V for s in starts])
    return stop_logit(cls), y


def sidecar_path(ckpt):
    return os.path.splitext(ckpt)[0] + ".pstop.json"


def main(argv=None):
    from data.iovnbd_sync import load_sync_dir
    from model.train_real import split_drives
    from model.nn_model import load_net
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.path.join(here, "nn_real.pt"))
    ap.add_argument("--data", default=os.path.expanduser("~/OffMaps-data/IO-VNBD-sync"))
    ap.add_argument("--out", default=os.path.join(here, "..", "..", "out", "pstop"))
    a = ap.parse_args(argv)
    from model.fusion_head import FOLDS
    import re
    net = load_net(a.ckpt)
    parts = split_drives(load_sync_dir(a.data, verbose=False))
    oof = os.path.join(here, "oof")
    groups = {}                                   # source drive -> (logits, labels), each from a net that never saw it
    for d in parts["val"]:
        z, y = drive_logits(net, d)
        g = groups.setdefault(d.vehicle_id.split("#")[0], ([], [], "val")); g[0].append(z); g[1].append(y)
    for d in parts["train"]:
        fk = [k for k, rx in FOLDS.items() if re.search(rx, d.vehicle_id.split("#")[0])][0]
        z, y = drive_logits(load_net(os.path.join(oof, f"nn_oof_{fk}.pt")), d)
        g = groups.setdefault(fk, ([], [], "train")); g[0].append(z); g[1].append(y)
    groups = {k: (np.concatenate(v[0]), np.concatenate(v[1]), v[2]) for k, v in groups.items()}
    cf, raw, ys, split = [], [], [], []
    for k, (z, y, sp) in groups.items():             # leave one drive out
        rest = [groups[j] for j in groups if j != k]
        cal = fit_platt(np.concatenate([r[0] for r in rest]), np.concatenate([r[1] for r in rest]))
        cf.append(apply_pstop(z, cal)); raw.append(1 / (1 + np.exp(-z))); ys.append(y); split += [sp] * len(y)
    cf, raw, ys, split = np.concatenate(cf), np.concatenate(raw), np.concatenate(ys), np.array(split)
    cal = fit_platt(np.concatenate([g[0] for g in groups.values()]), np.concatenate([g[1] for g in groups.values()]))
    v = split == "val"
    rep = {
        "checkpoint": os.path.basename(a.ckpt), "calib": cal, "label": f"speed < {STOP_V} m/s over the 1 s step",
        "drives": {k: int(len(g[1])) for k, g in groups.items()},
        "all_raw": metrics(raw, ys), "all_crossfit": metrics(cf, ys),
        "val_raw": metrics(raw[v], ys[v]), "val_crossfit": metrics(cf[v], ys[v]),
        "train_oof_raw": metrics(raw[~v], ys[~v]), "train_oof_crossfit": metrics(cf[~v], ys[~v]),
    }
    with open(sidecar_path(a.ckpt), "w") as fh:
        json.dump({"a": cal["a"], "b": cal["b"],
                   "fit_on": "held-out outputs: val drives (shipped net) + train drives (out-of-fold nets)",
                   "crossfit": rep["all_crossfit"]}, fh, indent=2); fh.write("\n")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "pstop.json"), "w") as fh:
        json.dump(rep, fh, indent=2); fh.write("\n")
    for k in ("all_raw", "all_crossfit", "val_raw", "val_crossfit", "train_oof_raw", "train_oof_crossfit"):
        m = rep[k]
        print(f"{k:19s} n={m['n']:6d} stopped={m['stopped']:.3f} AUC={m['auc']:.3f} Brier={m['brier']:.4f} "
              f"ECE={m['ece']:.3f} P/R@0.5={m['precision']:.2f}/{m['recall']:.2f}")
    print(f"calibration a={cal['a']:.3f} b={cal['b']:.3f} -> {sidecar_path(a.ckpt)}")


if __name__ == "__main__":
    main()
