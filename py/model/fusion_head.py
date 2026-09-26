"""Learned fusion head: the speed measurement (and its noise) the ESKF fuses while
dead-reckoning, predicted by a small GRU trained on real IO-VNBD drives.

    PYTHONPATH=. python3 -m model.fusion_head --data ~/OffMaps-data/IO-VNBD-sync [--oof DIR]

Why it exists. Once GNSS drops, the filter's only speed source is SpeedNet, a
2-second window model with no memory: it does not know the speed the car had
when GNSS dropped, whether traffic here is stop-go or cruising, or how wrong it
has been on this drive. Its errors are correlated for tens of seconds, so they
integrate straight into along-track drift (REALDATA.md: heading is fine, speed is
the whole problem). The hand-tuned answer so far was a fixed rule -- hold the
Doppler speed 10 s, then trust SpeedNet with sigma*0.25.

The head replaces that rule with a learned one. It runs once per second during an
outage (causal; state reset when the outage starts) and sees:

  context, frozen at outage start   entry Doppler speed v0; the Doppler self-cal
                                    fit (k, c, used?); the last 120 s of GNSS:
                                    mean speed, stopped fraction, mean/std of
                                    Doppler - calibrated-SpeedNet residual
  each second                       SpeedNet speed (raw + self-cal'd) and sigma;
                                    seconds since outage start; rotation-invariant
                                    IMU statistics of that second (vibration, |a_h|,
                                    a_up, |gyro|, |yaw rate|, jerk)

and outputs (v, sigma). The filter fuses it with update_speed(v, sigma): the net
decides the measurement AND its noise -- the "learned R" of adaptive Kalman
filtering (cf. Brossard et al., AI-IMU dead reckoning), trained end-to-end on the
quantity the benchmark scores: loss = relative cumulative-distance error at
10/30/60/120/180 s + beta-NLL on per-second speed (for an honest sigma).

Split is the repo's fixed by-drive split (model/train_real.SPLIT): trained on the
train drives, early-stopped and chosen on val, the test drives never loaded here.
SpeedNet's outputs on its OWN training drives are in-sample (better than on a new
drive), which would teach the head to over-trust it; --oof uses out-of-fold
SpeedNets (one per held-out train drive) for the training inputs instead. Which
of the two to ship is decided on val.
"""
from __future__ import annotations
import argparse, glob, json, math, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn

from model.dataset import STEP

T_MAX = 180                 # longest outage trained (s); shorter outages are prefixes (causal GRU)
PRE_S = 120                 # context window of GNSS before the outage (s)
HORIZONS = (10, 30, 60, 120, 180)
DOPPLER_SIGMA = 0.1
K_MIN, K_MAX = 1 / 3, 3.0
N_CTX, N_STEP = 9, 12
OUT = os.path.join(os.path.dirname(__file__), "fusion_head.pt")


# ---------------------------------------------------------------- features
def imu_stats(acc, gyro):
    """Rotation-invariant stats of one second of 10 Hz leveled IMU (z = up)."""
    acc = np.asarray(acc, float); gyro = np.asarray(gyro, float)
    an = np.linalg.norm(acc, axis=1); ah = np.linalg.norm(acc[:, :2], axis=1)
    jerk = np.linalg.norm(np.diff(acc, axis=0), axis=1).mean() * 10.0 if len(acc) > 1 else 0.0
    return np.array([an.std(), ah.mean(), acc[:, 2].mean() - 9.81, acc[:, 2].std(),
                     np.linalg.norm(gyro, axis=1).mean(), np.abs(gyro[:, 2]).mean(), jerk])


def context(v0, k, c, pre):
    """pre: list of (doppler, nn_raw) per trusted fix before the outage (oldest first)."""
    cal_ok = K_MIN <= k <= K_MAX
    kk, cc = (k, c) if cal_ok else (1.0, 0.0)
    p = np.asarray(pre[-PRE_S:], float).reshape(-1, 2)
    if len(p):
        dop, vn = p[:, 0], p[:, 1]
        res = dop - np.maximum(kk * vn + cc, 0.0)
        vbar, stop, rm, rs = dop.mean(), (dop < 0.5).mean(), res.mean(), res.std()
    else:
        vbar, stop, rm, rs = v0, 0.0, 0.0, 3.0
    return np.array([v0 / 10, vbar / 10, stop, rm / 5, rs / 5, float(cal_ok), kk, cc / 5,
                     min(len(p), PRE_S) / PRE_S], np.float32)


def step_features(vnn, sig, tau, stats, k, c):
    cal_ok = K_MIN <= k <= K_MAX
    vcal = max(k * vnn + c, 0.0) if cal_ok else vnn
    s = np.asarray(stats, float)
    return np.array([vnn / 10, vcal / 10, math.log(max(sig, 1e-3)), min(tau, 180) / 60, min(tau, 10) / 10,
                     s[0], s[1] / 3, s[2] / 3, s[3], s[4] * 5, s[5] * 5, s[6] / 10], np.float32)


# ---------------------------------------------------------------- model
class FusionHead(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.ctx = nn.Sequential(nn.Linear(N_CTX, hidden), nn.Tanh())
        self.gru = nn.GRU(N_STEP + N_CTX, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 2)

    def forward(self, ctx, x, h=None):
        """ctx (B, N_CTX), x (B, T, N_STEP) -> v (B, T), logvar (B, T), h."""
        if h is None:
            h = self.ctx(ctx)[None]
        z = torch.cat([x, ctx[:, None, :].expand(-1, x.shape[1], -1)], dim=2)
        y, h = self.gru(z, h)
        o = self.out(y)
        v = nn.functional.softplus(o[..., 0] * 5.0)             # m/s >= 0 (x5: scale the init)
        return v, torch.clamp(o[..., 1], -6, 6), h


class HeadRunner:
    """Streaming inference for the edge engine / host mirror: start() at outage start,
    step() once per second. An ensemble of members is averaged: v = mean v_i,
    sigma^2 = mean sigma_i^2 + var v_i (their disagreement is uncertainty too).
    Pure torch on CPU (each member a ~5k-param GRU)."""
    def __init__(self, nets):
        self.nets = [n.eval() for n in (nets if isinstance(nets, (list, tuple)) else [nets])]

    def start(self, ctx_dict):
        c = context(ctx_dict["v0"], ctx_dict["k"], ctx_dict["c"], ctx_dict["pre"])
        return dict(ctx=torch.from_numpy(c)[None], h=[None] * len(self.nets), k=ctx_dict["k"], c=ctx_dict["c"])

    def step(self, st, f):
        x = torch.from_numpy(step_features(f["vnn"], f["sig"], f["tau"], imu_stats(f["acc"], f["gyro"]),
                                           st["k"], st["c"]))[None, None]
        vs, vars_ = [], []
        with torch.no_grad():
            for i, net in enumerate(self.nets):
                v, lv, st["h"][i] = net(st["ctx"], x, st["h"][i])
                vs.append(float(v[0, 0])); vars_.append(math.exp(float(lv[0, 0])))
        return combine(vs, vars_)


def combine(vs, vars_):
    v = sum(vs) / len(vs)
    var = sum(vars_) / len(vars_) + sum((x - v) ** 2 for x in vs) / len(vs)
    return v, math.sqrt(var)


def load_head(path=OUT):
    ck = torch.load(path, map_location="cpu")
    states = ck["states"] if "states" in ck else [ck["state"]]
    nets = []
    for sd in states:
        net = FusionHead(ck.get("hidden", 32)); net.load_state_dict(sd); nets.append(net)
    return HeadRunner(nets)


def export_json(path_pt, path_json):
    """Weights + feature constants for the phone (FusionHead.kt runs the GRU itself)."""
    ck = torch.load(path_pt, map_location="cpu")
    states = ck["states"] if "states" in ck else [ck["state"]]
    mem = [{k: v.numpy().round(8).tolist() for k, v in sd.items()} for sd in states]
    doc = dict(format="offmaps-fusion-head/1", hidden=ck.get("hidden", 32), n_ctx=N_CTX, n_step=N_STEP,
               pre_s=PRE_S, k_min=K_MIN, k_max=K_MAX, members=mem,
               note="GRU gate order r,z,n (torch); v = softplus(5*o0), logvar = clamp(o1,-6,6); h0 = tanh(ctx.W+b)")
    with open(path_json, "w") as f:
        json.dump(doc, f)
    return path_json


# ---------------------------------------------------------------- data
def per_second(drive, net, calib):
    """Causal per-second table of one drive: the NN value available at the END of
    second j (window ending there), its sigma, the true mean speed of second j, the
    Doppler at the start of the second, and the IMU stats of second j."""
    from model.nn_model import predict_steps
    starts, vn, sg = predict_steps(net, drive, 0, len(drive) - STEP, calib=calib)
    vt = np.array([drive.speed[s:s + STEP].mean() for s in starts])
    dop = drive.speed[starts]
    st = np.stack([imu_stats(drive.acc[s:s + STEP], drive.gyro[s:s + STEP]) for s in starts])
    return dict(vn=vn.astype(float), sg=sg.astype(float), vt=vt, dop=dop, st=st)


def selfcal(tab, j0, rng):
    """Doppler self-cal exactly as the live loop runs it (core SpeedCal, Deming lambda
    from the running mean NN variance, 1 push per trusted fix, <= 600 in the buffer)."""
    from core_bridge import SpeedCal
    sc = SpeedCal(); ss = 0.0; n = 0; k, c = 1.0, 0.0; pre = []
    for j in range(max(0, j0 - 600), j0):
        d = tab["dop"][j] + rng.normal(0, DOPPLER_SIGMA)
        vn_prev = tab["vn"][j - 1] if j else 0.0          # the NN value last computed at this fix
        pre.append((d, vn_prev))
        if vn_prev > 0:
            ss += tab["sg"][j - 1] ** 2; n += 1
            sc.set_lambda(DOPPLER_SIGMA ** 2 / max(ss / n, 1e-9)); sc.push(vn_prev, d, 1.0)
            k, c, _ = sc.fit()
    return k, c, pre


def windows(tabs, every, rng, T=T_MAX, min_pre=60):
    """Outage windows -> (ctx, X, vt) arrays. Each window: outage starts at second j0,
    features for seconds j0..j0+T-1."""
    C, X, Y = [], [], []
    for tab in tabs:
        n = len(tab["vt"])
        for j0 in range(min_pre, n - T, every):
            k, c, pre = selfcal(tab, j0, rng)
            v0 = tab["dop"][j0] + rng.normal(0, DOPPLER_SIGMA)
            C.append(context(v0, k, c, pre))
            X.append(np.stack([step_features(tab["vn"][j], tab["sg"][j], j - j0 + 1, tab["st"][j], k, c)
                               for j in range(j0, j0 + T)]))
            Y.append(tab["vt"][j0:j0 + T])
    return np.array(C, np.float32), np.array(X, np.float32), np.array(Y, np.float32)


# ---------------------------------------------------------------- training
def loss_fn(v, lv, y, beta=0.5, w_nll=0.05):
    cp, ct = torch.cumsum(v, 1), torch.cumsum(y, 1)
    dist = 0.0
    for h in HORIZONS:
        if h <= v.shape[1]:
            dist = dist + ((cp[:, h - 1] - ct[:, h - 1]).abs() / torch.clamp(ct[:, h - 1], min=5.0 * h)).mean()
    var = torch.exp(lv)
    nll = (var.detach() ** beta * 0.5 * (lv + (v - y) ** 2 / var)).mean()
    return dist + w_nll * nll


def drift_metric(v, y, durations=(30, 60, 120)):
    """Offline proxy of the benchmark: median |distance error| / distance per duration,
    windows averaging < 5 m/s skipped (as the live evaluation does)."""
    out = {}
    for D in durations:
        dt, dp = y[:, :D].sum(1), v[:, :D].sum(1)
        ok = dt >= 5 * D
        out[D] = float(np.median(100 * np.abs(dp[ok] - dt[ok]) / dt[ok])) if ok.any() else float("nan")
    return out


def fit(Ctr, Xtr, Ytr, Cva, Xva, Yva, epochs=40, lr=3e-3, hidden=32, seed=0, log=print, keep_last=False):
    torch.manual_seed(seed)
    net = FusionHead(hidden)
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-4)
    tr = [torch.from_numpy(a) for a in (Ctr, Xtr, Ytr)]
    va = [torch.from_numpy(a) for a in (Cva, Xva, Yva)]
    best, best_state, hist = float("inf"), None, []
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(tr[0]), generator=g); tot = 0.0
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            v, lv, _ = net(tr[0][b], tr[1][b])
            loss = loss_fn(v, lv, tr[2][b])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            tot += float(loss.detach()) * len(b)
        net.eval()
        with torch.no_grad():
            v, lv, _ = net(va[0], va[1])
        m = drift_metric(v.numpy(), va[2].numpy())
        score = float(np.mean(list(m.values())))
        hist.append(dict(epoch=ep, loss=tot / len(perm), val=m, val_score=score))
        tag = ""
        if score < best:
            best, best_state, tag = score, {k: x.clone() for k, x in net.state_dict().items()}, " *"
        log(f"  ep{ep:2d} loss {tot/len(perm):.4f}  val " + " ".join(f"{D}s {x:5.1f}%" for D, x in m.items())
            + f"  mean {score:5.1f}{tag}")
    if not keep_last:
        net.load_state_dict(best_state)
    return net, best, hist


def build_tables(data, oof_dir=None, speednet=None):
    """Per-second tables for train (in-sample or OOF SpeedNet) and val (the shipped nn_real,
    or the checkpoint `speednet`)."""
    from data.iovnbd_sync import load_sync_dir
    from model.train_real import split_drives
    from model.nn_model import load_net, get_calib
    parts = split_drives(load_sync_dir(data, verbose=False))
    real = speednet or os.path.join(os.path.dirname(__file__), "nn_real.pt")
    net, cal = load_net(real), get_calib(real)
    folds = FOLDS
    tabs = {"train": [], "val": []}
    for d in parts["train"]:
        n_, c_ = net, cal
        if oof_dir:
            src = d.vehicle_id.split("#")[0]
            fk = [k for k, rx in folds.items() if re.search(rx, src)][0]
            p = os.path.join(oof_dir, f"nn_oof_{fk}.pt")
            n_, c_ = load_net(p), get_calib(p)
        tabs["train"].append(per_second(d, n_, c_))
    for d in parts["val"]:
        tabs["val"].append(per_second(d, net, cal))
    return tabs, parts


FOLDS = {"M": r"/S-M$", "S1": r"/S1/", "S2": r"/S2/", "S4": r"/S4/"}


def cross_validate(data, oof_dir, epochs=40, hidden=32, seed=0, variants=("insample", "oof"), speednet=None):
    """Leave-one-TRAIN-drive-out CV to choose the epoch count and the SpeedNet input
    variant WITHOUT touching val or test. The held-out drive's inputs always come from
    its out-of-fold SpeedNet (a net that never saw it), i.e. it plays a new drive; the
    training drives' inputs come from nn_real ("insample") or from their own OOF nets.
    Returns {variant: mean held-out score per epoch}."""
    from data.iovnbd_sync import load_sync_dir
    from model.train_real import split_drives
    from model.nn_model import load_net, get_calib
    train = split_drives(load_sync_dir(data, verbose=False))["train"]
    real = speednet or os.path.join(os.path.dirname(__file__), "nn_real.pt")
    fold_of = lambda d: [k for k, rx in FOLDS.items() if re.search(rx, d.vehicle_id.split("#")[0])][0]
    tabs = {"insample": [], "oof": []}
    for d in train:
        tabs["insample"].append(per_second(d, load_net(real), get_calib(real)))
        p = os.path.join(oof_dir, f"nn_oof_{fold_of(d)}.pt")
        tabs["oof"].append(per_second(d, load_net(p), get_calib(p)))
    folds = [fold_of(d) for d in train]
    curves = {}
    for var in variants:
        per_fold = []
        for fk in FOLDS:
            rng = np.random.default_rng(seed)
            tr = [t for t, f in zip(tabs[var], folds) if f != fk]
            ho = [t for t, f in zip(tabs["oof"], folds) if f == fk]
            Ctr, Xtr, Ytr = windows(tr, 5, rng); Cho, Xho, Yho = windows(ho, 15, rng)
            base = drift_metric(Xho[:, :, 1] * 10, Yho)
            _, _, hist = fit(Ctr, Xtr, Ytr, Cho, Xho, Yho, epochs, hidden=hidden, seed=seed, log=lambda *a: None)
            per_fold.append([h["val_score"] for h in hist])
            print(f"  cv {var:8s} hold-out {fk:3s} (n={len(Yho)}): SpeedNet self-cal {np.mean(list(base.values())):5.1f}%  "
                  f"head best {min(per_fold[-1]):5.1f}% @ep{int(np.argmin(per_fold[-1]))}  last {per_fold[-1][-1]:5.1f}%", flush=True)
        curves[var] = np.mean(per_fold, 0).tolist()
        print(f"cv {var}: mean held-out score by epoch " + " ".join(f"{x:.1f}" for x in curves[var]), flush=True)
    return curves


def train_fold_heads(data, oof_dir, out_dir, epochs=30, ensemble=2, hidden=32, seed=0):
    """One head per held-out TRAIN drive, trained on the other three (OOF SpeedNet inputs),
    for the leave-one-drive-out LIVE evaluation (phase8_eval.py --lodo). Val/test untouched."""
    from data.iovnbd_sync import load_sync_dir
    from model.train_real import split_drives
    from model.nn_model import load_net, get_calib
    train = split_drives(load_sync_dir(data, verbose=False))["train"]
    fold_of = lambda d: [k for k, rx in FOLDS.items() if re.search(rx, d.vehicle_id.split("#")[0])][0]
    tabs = []
    for d in train:
        p = os.path.join(oof_dir, f"nn_oof_{fold_of(d)}.pt")
        tabs.append((fold_of(d), per_second(d, load_net(p), get_calib(p))))
    for fk in FOLDS:
        rng = np.random.default_rng(seed)
        Ctr, Xtr, Ytr = windows([t for f, t in tabs if f != fk], 5, rng)
        Cho, Xho, Yho = windows([t for f, t in tabs if f == fk], 15, rng)
        states = []
        for m in range(ensemble):
            net, _, hist = fit(Ctr, Xtr, Ytr, Cho, Xho, Yho, epochs, hidden=hidden, seed=seed + m,
                               log=lambda *a: None, keep_last=True)
            states.append(net.state_dict())
        path = os.path.join(out_dir, f"fusion_head_oof_{fk}.pt")
        torch.save(dict(states=states, hidden=hidden, meta=dict(held_out=fk, epochs=epochs)), path)
        print(f"fold head without {fk}: {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--oof", help="dir with nn_oof_{M,S1,S2,S4}.pt (out-of-fold SpeedNets for the train inputs)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cv", action="store_true", help="only run leave-one-train-drive-out CV (needs --oof)")
    ap.add_argument("--fixed-epochs", type=int, help="train exactly this many epochs (chosen by --cv); val is then a check, not a selector")
    ap.add_argument("--ensemble", type=int, default=1, help="average this many seeds")
    ap.add_argument("--fold-heads", help="train one head per held-out train drive into this dir (needs --oof)")
    ap.add_argument("--speednet", help="SpeedNet checkpoint for the val (and in-sample) tables (default model/nn_real.pt)")
    a = ap.parse_args()
    if a.fold_heads:
        return train_fold_heads(a.data, a.oof, a.fold_heads, a.fixed_epochs or 30, a.ensemble, a.hidden, a.seed)
    if a.cv:
        curves = cross_validate(a.data, a.oof, a.epochs, a.hidden, a.seed, speednet=a.speednet)
        with open(a.out.replace(".pt", "_cv.json"), "w") as f:
            json.dump(curves, f, indent=2)
        return
    t0 = time.time()
    tabs, parts = build_tables(a.data, a.oof, a.speednet)
    rng = np.random.default_rng(a.seed)
    Ctr, Xtr, Ytr = windows(tabs["train"], 5, rng)
    Cva, Xva, Yva = windows(tabs["val"], 15, rng)
    print(f"windows: train {len(Ytr)}  val {len(Yva)}  ({time.time()-t0:.0f}s)")
    base = {}
    for name, v in [("hold v0", np.repeat(Cva[:, :1] * 10, Yva.shape[1], 1)),
                    ("SpeedNet self-cal'd", Xva[:, :, 1] * 10)]:
        base[name] = drift_metric(v, Yva)
        print(f"  val baseline {name:20s} " + " ".join(f"{D}s {x:5.1f}%" for D, x in base[name].items()))
    states, hists, best = [], [], None
    for m in range(a.ensemble):
        if a.fixed_epochs:            # epochs chosen by --cv on train drives: val is only a check
            net, best_m, hist = fit(Ctr, Xtr, Ytr, Cva, Xva, Yva, a.fixed_epochs, hidden=a.hidden,
                                    seed=a.seed + m, keep_last=True)
        else:
            net, best_m, hist = fit(Ctr, Xtr, Ytr, Cva, Xva, Yva, a.epochs, hidden=a.hidden, seed=a.seed + m)
        states.append(net.state_dict()); hists.append(hist)
        print(f"member {m}: val {hist[-1]['val'] if a.fixed_epochs else best_m}", flush=True)
    runner = HeadRunner([FusionHead(a.hidden) for _ in states])
    for n_, sd in zip(runner.nets, states):
        n_.load_state_dict(sd); n_.eval()
    with torch.no_grad():
        outs = [n_(torch.from_numpy(Cva), torch.from_numpy(Xva))[0].numpy() for n_ in runner.nets]
    ens = drift_metric(np.mean(outs, 0), Yva)
    best = float(np.mean(list(ens.values())))
    print("ensemble val " + " ".join(f"{D}s {x:5.1f}%" for D, x in ens.items()) + f"  mean {best:.1f}")
    hist = hists
    meta = dict(split=("train -> fit for --fixed-epochs (chosen by leave-one-train-drive-out CV); val = check only"
                       if a.fixed_epochs else "train -> fit, val -> early stop") + " (test never loaded)",
                oof=bool(a.oof), fixed_epochs=a.fixed_epochs, ensemble=a.ensemble, val_ensemble=ens,
                train_windows=int(len(Ytr)), val_windows=int(len(Yva)), best_val_score=best,
                val_baselines=base, history=hist, hidden=a.hidden, seed=a.seed,
                train_drives=[d.vehicle_id for d in parts["train"]], val_drives=[d.vehicle_id for d in parts["val"]])
    torch.save(dict(states=states, hidden=a.hidden, meta=meta), a.out)
    with open(a.out.replace(".pt", ".json"), "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"saved {a.out}  best val mean {best:.1f}%")


if __name__ == "__main__":
    main()
