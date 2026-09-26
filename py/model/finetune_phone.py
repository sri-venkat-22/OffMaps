"""Fine-tune the shipped SpeedNet (nn_real) on the target phone's own drives.

    PYTHONPATH=. python3 -m model.finetune_phone --drives ../drives --hold-out drive_<ms>[,...] --out X.pt

Same network, same 2 s feature contract, so the phone runs the result with no code
change -- only the weights (and the val-chosen sigma calibration) differ. The phone
drives come in exactly as the app feeds its net (data/phone_drive.py); the label is the
phone's GNSS Doppler speed. They are mixed with the IO-VNBD train drives so the net
keeps what it learned there, and repeated PHONE_REPEAT times so ~1 h of phone data
is not drowned by 9.5 h of IO-VNBD.

Settings fixed before any result was seen (1.5 h of phone data is too little to tune
on): init nn_real, lr 3e-4 cosine, EPOCHS epochs, no early stopping; calibration by the
same rule as nn_real (train_real.calibrate on the IO-VNBD val drives). --hold-out names
the phone drives that must not be trained on (the ones it will be scored on).
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.iovnbd_sync import load_sync_dir
from data.phone_drive import load_phone_drive
from data.outage import tilde_home
from model.dataset import SeqDataset, L
from model.nn_model import build_net
from model.train_real import split_drives, calibrate
from model import loss as LOSS

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "nn_real.pt")
PHONE_REPEAT = 5
EPOCHS = 4
LR = 3e-4


def finetune(drives_dir, hold_out, out, *, data=os.path.expanduser("~/OffMaps-data/IO-VNBD-sync"),
             device="cpu", seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    parts = split_drives(load_sync_dir(data, verbose=False))
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(drives_dir, "drive_*")))
    train_names = [n for n in names if n not in hold_out]
    assert not set(hold_out) - set(names), f"unknown hold-out drives {set(hold_out) - set(names)}"
    phone = [s for n in train_names for s in load_phone_drive(os.path.join(drives_dir, n))]
    print(f"IO-VNBD train {sum(len(d) for d in parts['train'])/36000:.2f} h + phone {sum(len(d) for d in phone)/36000:.2f} h "
          f"x{PHONE_REPEAT} ({len(train_names)} drives; held out: {', '.join(hold_out) or 'none'})", flush=True)
    ck = torch.load(BASE, map_location="cpu")
    net = build_net(ck.get("feat")); net.load_state_dict(ck["state"])
    ds = SeqDataset(parts["train"] + phone * PHONE_REPEAT, seed=seed)
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    dev = torch.device(device)
    hist = []
    for ep in range(EPOCHS):
        t0 = time.time(); net.to(dev).train(); tot = 0.0
        for X, dt, cl in dl:
            X, dt, cl = X.to(dev), dt.to(dev), cl.to(dev)
            B = X.shape[0]
            mu, lv, _, logits = net(X.reshape(B * L, *X.shape[2:]))
            loss = LOSS.total(mu.reshape(B, L), lv.reshape(B, L), dt, logits.reshape(B, L, 4), cl)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * B
        sched.step()
        hist.append(dict(epoch=ep, loss=tot / len(ds)))
        print(f"  ep{ep} loss={tot / len(ds):.4f} ({time.time() - t0:.0f}s)", flush=True)
    net.to("cpu").eval()
    calib, cinfo = calibrate(net, parts["val"])
    meta = dict(base=os.path.basename(BASE), phone_train=train_names, phone_held_out=list(hold_out),
                phone_repeat=PHONE_REPEAT, epochs=EPOCHS, lr=LR, seed=seed, history=hist, calibration=cinfo,
                drives=tilde_home(os.path.abspath(drives_dir)), data=tilde_home(os.path.abspath(data)))
    torch.save({"state": net.state_dict(), "calib": calib, "feat": net.feat,
                "eskf_cfg": ck.get("eskf_cfg"), "meta": meta}, out)
    with open(out.replace(".pt", ".json"), "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"saved {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--drives", required=True, help="folder of drive_<ms> recordings")
    ap.add_argument("--hold-out", default="", help="comma-separated drive_<ms> names NOT to train on")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    finetune(a.drives, [h for h in a.hold_out.split(",") if h], a.out, device=a.device)


if __name__ == "__main__":
    main()
