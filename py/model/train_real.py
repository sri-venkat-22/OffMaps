"""Train SpeedNet on REAL phone IMU (IO-VNBD), labels = vehicle survey-GNSS speed.

    PYTHONPATH=. python3 -m model.train_real --data "<.../Categorised IOVNB Dataset>"

The synthetic-trained net (model/nn.pt) does not transfer to real data: on IO-VNBD
its speed MAE is ~5-6 m/s and its sigma is badly over-confident (z_mean ~ -4.5,
z_var ~ 25), so the ESKF that trusts it does worse than plain physics (REALDATA.md).
This retrains the SAME network, features, loss and calibration on real drives.

Split is BY DRIVE and FIXED here, chosen before any real-data training result
was seen (never tune it toward a number). Consecutive windows of one drive are
near-duplicates, so a window-level split would leak and flatter the score.

  train : M (Driver B), S1, S2, S4 (Driver A)       ~9.5 h
  val   : S3b, S3c (Driver A)                        early stopping + sigma calibration
  test  : Y1 (Driver D -- a driver never trained on), S3a (Driver A)

Output goes to model/nn_real.pt, NOT model/nn.pt: the synthetic gates, the
parity tests and the Android nn.onnx all pin nn.pt. Score the test split with

    PYTHONPATH=. python3 -m validate_realdata --layout iovnbd-sync --data <dir> \\
        --include 'S3a|Y1' --nn-ckpt model/nn_real.pt
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from data.iovnbd_sync import load_sync_dir
from model.tcn import SpeedNet, count_params
from model.dataset import SeqDataset, L, STEP
from model import loss as LOSS
from model.nn_model import _raw_predict, apply_calib, DEFAULT
from model.train import fit_calibration
from data.outage import tilde_home
from eval.metrics import sigma_calibration

SPLIT = {
    "train": r"/S-M$|/S1/|/S2/|/S4/",
    "val":   r"/S3b/|/S3c/",
    "test":  r"/Y1/|/S3a/",
}
OUT = os.path.join(os.path.dirname(__file__), "nn_real.pt")


def split_drives(drives):
    """Assign every synced segment to exactly one split by its SOURCE DRIVE."""
    parts = {k: [] for k in SPLIT}
    for d in drives:
        src = d.vehicle_id.split("#")[0]
        hit = [k for k, rx in SPLIT.items() if re.search(rx, src)]
        if len(hit) != 1:
            raise ValueError(f"drive {src!r} matches splits {hit} -- must be exactly one")
        parts[hit[0]].append(d)
    srcs = {k: {d.vehicle_id.split("#")[0] for d in v} for k, v in parts.items()}
    for a in srcs:
        for b in srcs:
            if a < b:
                assert not (srcs[a] & srcs[b]), f"drive leak {a}/{b}: {srcs[a] & srcs[b]}"
    return parts


def speed_eval(net, drives, calib):
    """Per-second (v_pred, v_true, sigma) over whole drives -> MAE + sigma calibration."""
    vp, vt, sg = [], [], []
    for d in drives:
        starts, mu, sig = _raw_predict(net, d, 0, len(d) - STEP)
        v, s = apply_calib(mu, sig, calib)
        for k, i in enumerate(starts):
            seg = d.speed[i:i + STEP]
            if len(seg):
                vp.append(v[k]); vt.append(seg.mean()); sg.append(s[k])
    vp, vt, sg = map(np.asarray, (vp, vt, sg))
    cal = sigma_calibration(vp, vt, sg)
    return dict(mae=float(np.mean(np.abs(vp - vt))), bias=float(np.mean(vp - vt)),
                z_var=cal["z_var"], z_mean=cal["z_mean"], passed=cal["passed"], n=len(vp))


def calibrate(net, val):
    """Pick the calibration by VALIDATION error only (never test).

    The Phase-2 affine map v=(mu-a)/b fixes regression dilution when the net's
    output tracks true speed tightly. On real IO-VNBD the validation drives are
    few and heterogeneous (S3c's accel barely correlates with the car, 0.03), so
    the fitted slope can come out tiny (b~0.39) and dividing by it amplifies noise:
    val MAE went 5.53 -> 8.03 m/s. Rule: keep the affine mean correction only if it
    lowers val MAE; otherwise identity mean + a sigma temperature s (z_var -> 1)
    fitted on the same val drives.
    """
    ident = {"a": 0.0, "b": 1.0, "s": 1.0}
    affine = fit_calibration(net, val)
    raw = speed_eval(net, val, ident)
    # variance-only: s = std of raw z on val, so z_var ~ 1 with the mean untouched
    vp, vt, sg = [], [], []
    for d in val:
        starts, mu, sig = _raw_predict(net, d, 0, len(d) - STEP)
        for k, i in enumerate(starts):
            seg = d.speed[i:i + STEP]
            if len(seg):
                vp.append(mu[k]); vt.append(seg.mean()); sg.append(sig[k])
    vp, vt, sg = map(np.asarray, (vp, vt, sg))
    var_only = {"a": 0.0, "b": 1.0, "s": float(((vp - vt) / sg).std())}
    cand = {"affine": affine, "variance_only": var_only}
    scores = {k: speed_eval(net, val, c) for k, c in cand.items()}
    pick = "affine" if scores["affine"]["mae"] < scores["variance_only"]["mae"] else "variance_only"
    for k, sc in scores.items():
        print(f"  calib {k:13s}: val MAE {sc['mae']:.2f}  z_var {sc['z_var']:.2f}  "
              f"z_mean {sc['z_mean']:+.2f}{'   <- picked (lower val MAE)' if k == pick else ''}")
    return cand[pick], dict(pick=pick, raw_val=raw, **{f"val_{k}": v for k, v in scores.items()})


def recalibrate(data, ckpt=OUT):
    """Re-run calibrate() on a saved checkpoint's own val split (no retraining)."""
    parts = split_drives(load_sync_dir(data, verbose=False))
    ck = torch.load(ckpt, map_location="cpu")
    net = SpeedNet(); net.load_state_dict(ck["state"]); net.eval()
    calib, info = calibrate(net, parts["val"])
    ck["calib"] = calib; ck.setdefault("meta", {})["calibration"] = info
    ck["meta"] = tilde_home(ck["meta"])
    torch.save(ck, ckpt)
    with open(ckpt.replace(".pt", ".json"), "w") as f:
        json.dump(ck["meta"], f, indent=2, default=float)
    print(f"recalibrated {ckpt}: {info['pick']} {calib}")


EXTRA_TRAIN = r"\(Driver E\)"   # loosely-mounted phones: good vehicle labels, noisy phone yaw


def train(data, *, epochs=20, lr=1e-3, init="synthetic", seed=0, out=OUT, workers=0,
          extra_min_corr=None, patience=None):
    """extra_min_corr: also train on EXTRA_TRAIN drives synced at this looser yaw
    correlation (TRAIN ONLY -- val/test keep the strict 0.8 loader). Their labels
    are the vehicle's GNSS speed (QC 0.994-1.003); only the phone mount is loose.
    patience: stop after this many epochs without a val improvement (the kept
    model is the best-val epoch either way)."""
    torch.manual_seed(seed); np.random.seed(seed)
    parts = split_drives(load_sync_dir(data, verbose=False))
    if extra_min_corr is not None:
        extra = [d for d in load_sync_dir(data, min_corr=extra_min_corr, verbose=False)
                 if re.search(EXTRA_TRAIN, d.vehicle_id)]
        held = {d.vehicle_id.split("#")[0] for k in ("val", "test") for d in parts[k]}
        assert not any(d.vehicle_id.split("#")[0] in held for d in extra), "extra drive in val/test"
        parts["train"] += extra
        print(f"extra train (corr>={extra_min_corr}): {len(extra)} segments, "
              f"{sum(len(d) for d in extra)/10/3600:.2f} h")
    for k, v in parts.items():
        print(f"{k:5s}: {sum(len(d) for d in v)/10/3600:5.2f} h  "
              f"{sorted({d.vehicle_id.split('#')[0] for d in v})}")

    net = SpeedNet()
    if init == "synthetic":
        net.load_state_dict(torch.load(DEFAULT, map_location="cpu")["state"])
    ident = {"a": 0.0, "b": 1.0, "s": 1.0}
    net.eval()
    base = speed_eval(net, parts["val"], ident)
    print(f"init={init}: val MAE {base['mae']:.2f} m/s bias {base['bias']:+.2f}")

    dl = DataLoader(SeqDataset(parts["train"], seed=seed), batch_size=16, shuffle=True,
                    num_workers=workers)
    opt = torch.optim.Adam(net.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    print(f"params={count_params(net):,}  train_seqs={len(dl.dataset)}  epochs={epochs}")
    best, best_state, hist = float("inf"), None, []
    for ep in range(epochs):
        t0 = time.time(); net.train(); tot = 0.0
        for X, dt, cl in dl:
            B = X.shape[0]
            mu, lv, _, logits = net(X.reshape(B * L, *X.shape[2:]))
            mu, lv, logits = mu.reshape(B, L), lv.reshape(B, L), logits.reshape(B, L, 4)
            loss = LOSS.total(mu, lv, dt, logits, cl)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * B
        sched.step(); net.eval()
        ev = speed_eval(net, parts["val"], ident)          # raw (uncalibrated) val speed
        hist.append(dict(epoch=ep, loss=tot / len(dl.dataset), **ev))
        tag = ""
        if ev["mae"] < best:                                 # early stopping on VAL only
            best, best_state, tag = ev["mae"], {k: v.clone() for k, v in net.state_dict().items()}, " *"
        print(f"  ep{ep:2d} loss={tot/len(dl.dataset):.4f}  val MAE {ev['mae']:.2f} "
              f"bias {ev['bias']:+.2f}  ({time.time()-t0:.0f}s){tag}", flush=True)
        if patience and ep - min(range(len(hist)), key=lambda i: hist[i]["mae"]) >= patience:
            print(f"  early stop: no val improvement for {patience} epochs"); break

    net.load_state_dict(best_state); net.eval()
    calib, cinfo = calibrate(net, parts["val"])              # chosen on VAL drives only
    meta = dict(data=tilde_home(os.path.abspath(data)), split=SPLIT, init=init, epochs=epochs, lr=lr,
                extra_train=(EXTRA_TRAIN, extra_min_corr) if extra_min_corr is not None else None,
                seed=seed, best_val_mae=best, history=hist,
                calibration=cinfo, synced_segments={k: [d.vehicle_id for d in v]
                                                     for k, v in parts.items()})
    torch.save({"state": net.state_dict(), "calib": calib, "meta": meta}, out)
    with open(out.replace(".pt", ".json"), "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"saved {out}  (test split untouched -- score it with validate_realdata)")
    return net, calib, parts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="IO-VNBD synchronised dir (S-*/V-* pairs)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init", choices=["synthetic", "scratch"], default="synthetic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--extra-min-corr", type=float, default=None,
                    help="also TRAIN on Driver-E drives synced at this looser yaw corr (e.g. 0.3)")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--recalibrate", action="store_true",
                    help="only re-run the val-chosen calibration on an existing --out checkpoint")
    a = ap.parse_args()
    if a.recalibrate:
        return recalibrate(a.data, a.out)
    train(a.data, epochs=a.epochs, lr=a.lr, init=a.init, seed=a.seed, out=a.out, workers=a.workers,
          extra_min_corr=a.extra_min_corr, patience=a.patience)


if __name__ == "__main__":
    main()
