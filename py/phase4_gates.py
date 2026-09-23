"""Phase-4 exit gates (all three required). Run:
    PYTHONPATH=. python3 phase4_gates.py --out ../out
Produces gate plots and asserts the pass criteria.
"""
from __future__ import annotations
import os, sys, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from data.io_vnbd import synth_drive
from data.outage import make_outages
from model.nn_model import predict_steps, load_net
from model.dataset import STEP
from core_bridge import SpeedCal, Align, gq_trust, gq_spoof

DEG = np.pi / 180


def _speed_series(net, d, i0, i1, want_sigma=False):
    """Per-sample NN speed (and sigma) over [i0,i1) (piecewise-constant per second)."""
    starts, v, sig = predict_steps(net, d, i0, i1)
    perv = {s: vv for s, vv in zip(starts, v)}
    pers = {s: ss for s, ss in zip(starts, sig)}
    out = np.empty(i1 - i0); sg = np.empty(i1 - i0); cv, cs = v[0], sig[0]
    for i in range(i1 - i0):
        cv = perv.get(i0 + i, cv); cs = pers.get(i0 + i, cs)
        out[i] = cv; sg[i] = cs
    return (out, sg) if want_sigma else out


DOPPLER_SIGMA = 0.1   # GNSS Doppler ground-speed noise, m/s (chip-typical)


def _along_drift(speed, drive, i0, i1):
    """Integrate speed along TRUE heading (isolates along-track = the scale effect)."""
    t = drive.t[i0:i1]; dt = np.diff(t, prepend=t[0]); h = drive.heading[i0:i1]
    e = drive.e[i0] + np.cumsum(speed * np.sin(h) * dt)
    n = drive.n[i0] + np.cumsum(speed * np.cos(h) * dt)
    err = np.hypot(e[-1] - drive.e[i1-1], n[-1] - drive.n[i1-1])
    dist = np.trapezoid(drive.speed[i0:i1], t)
    return 100 * err / dist


def gate1_calibration(out, gap=0.90):
    """Held-out vehicle: NN reads `gap` low. Doppler self-cal must recover it."""
    net = load_net()
    drives = [synth_drive(f"vehB{i}", 900, seed=200 + i) for i in range(4)]
    base, gapped, cal, ks = [], [], [], []
    for d in drives:
        for o in make_outages(d, seed=0):
            if o.duration_s != 60 or o.i0 < 600:      # need 60 s of pre-outage GNSS
                continue
            v0 = _speed_series(net, d, o.i0, o.i1)
            vB = v0 * gap                              # this vehicle's domain gap
            # calibrate on the 60 s of healthy GNSS before the outage
            sc = SpeedCal()
            pre, pre_sig = _speed_series(net, d, o.i0 - 600, o.i0, want_sigma=True)
            pre = pre * gap
            # errors-in-variables: the NN speed we regress ON is noisy, so plain
            # OLS dilutes k. Feed SpeedCal the NN's own (calibrated, gap-scaled)
            # sigma as the regressor-noise variance so the Deming fit recovers
            # the true scale instead of under-shooting it.
            sig_nn = float(np.mean((pre_sig * gap) ** 2))
            sc.set_lambda(DOPPLER_SIGMA ** 2 / sig_nn)
            ktrace = []
            for j, vn in enumerate(pre):
                sc.push(vn, d.speed[o.i0 - 600 + j])   # Doppler = true speed when healthy
                if j % 10 == 0: ktrace.append(sc.fit()[0])
            k, c, _ = sc.fit(); ks.append(ktrace)
            base.append(_along_drift(v0, d, o.i0, o.i1))
            gapped.append(_along_drift(vB, d, o.i0, o.i1))
            cal.append(_along_drift(k * vB + c, d, o.i0, o.i1))
    b, g, c = np.median(base), np.median(gapped), np.median(cal)
    recovery = (g - c) / (g - b + 1e-9)
    print(f"[gate1] along-track drift  baseline={b:.1f}%  gapped={g:.1f}%  calibrated={c:.1f}%")
    print(f"[gate1] degradation recovered = {100*recovery:.0f}%  (need >50%)")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for tr in ks: a2.plot(np.arange(len(tr)), tr, alpha=.5)
    a2.axhline(1/gap, ls="--", c="k", label=f"true 1/gap={1/gap:.2f}")
    a2.set(xlabel="Doppler samples (×10)", ylabel="estimated k", title="speed-scale k converges"); a2.legend()
    a1.bar(["baseline\n(no gap)", "gapped\n(uncal)", "calibrated"], [b, g, c],
           color=["#888", "#c0392b", "#27ae60"])
    a1.set(ylabel="60 s along-track drift %", title=f"Doppler self-cal recovers {100*recovery:.0f}%")
    fig.tight_layout(); fig.savefig(f"{out}/gate1_calibration.png", dpi=110); plt.close(fig)
    assert recovery > 0.5 and c < 0.6 * g, "GATE 1 FAIL"
    print("[gate1] PASS\n")


def _rot(roll, pitch):
    cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    return Ry @ Rx


def gate2_remount(out, t_remount=40.0):
    """Phone physically re-mounted mid-drive; CUSUM detects, roll/pitch re-level < 5 s."""
    d = synth_drive("remount", 120, seed=7)
    R = _rot(30*DEG, 20*DEG)                           # the new mounting
    g_new = R @ np.array([0, 0, 9.81])
    true_roll, true_pitch = np.arctan2(g_new[1], g_new[2]), np.arctan2(-g_new[0], np.hypot(g_new[1], g_new[2]))
    aln = Align(); T = int(t_remount * 10)
    roll, pitch, changed_at, converged_at = [], [], None, None
    for i in range(len(d)):
        a, g = d.acc[i].copy(), d.gyro[i].copy()
        if i >= T: a, g = R @ a, R @ g                 # apply the re-mount
        aln.update(*a, *g, 0.1)
        r, p, y, ch = aln.get()
        roll.append(r); pitch.append(p)
        if ch and changed_at is None and i >= T: changed_at = i
        if i > T and converged_at is None and abs(r-true_roll) < 5*DEG and abs(p-true_pitch) < 5*DEG:
            converged_at = i
    dt_detect = (changed_at - T) / 10 if changed_at else None
    dt_conv = (converged_at - T) / 10 if converged_at else None
    print(f"[gate2] re-mount at {t_remount}s: CUSUM detect +{dt_detect}s, re-level +{dt_conv}s (need <5s)")
    tt = d.t
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tt, np.degrees(roll), label="roll est"); ax.plot(tt, np.degrees(pitch), label="pitch est")
    ax.axhline(np.degrees(true_roll), ls=":", c="C0"); ax.axhline(np.degrees(true_pitch), ls=":", c="C1")
    ax.axvline(t_remount, c="r", label="re-mount")
    if converged_at: ax.axvline(converged_at/10, c="g", ls="--", label="re-level")
    ax.set(xlabel="time (s)", ylabel="deg", title="Mount re-alignment after physical re-mount"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{out}/gate2_remount.png", dpi=110); plt.close(fig)
    assert changed_at and dt_conv is not None and dt_conv < 5.0, "GATE 2 FAIL"
    print("[gate2] PASS\n")


def gate3_spoof(out, t_spoof=30.0, n=600):
    """GNSS spoofed at t_spoof: C/N0 stays high but INS rejects it (innov spikes)."""
    t = np.arange(n) / 10
    cn0 = np.full(n, 42.0); sv = np.full(n, 9)
    chi2 = np.where(t < t_spoof, np.abs(np.random.default_rng(0).normal(0, 1.0, n)),  # INS agrees
                    40.0)                                                              # INS rejects
    trust, flag = [], []
    for i in range(n):
        trust.append(gq_trust(cn0[i], int(sv[i]), 3, 1.2, chi2[i]))
        flag.append(gq_spoof(cn0[i], int(sv[i]), chi2[i], 0.0))
    flag = np.array(flag)
    first = t[np.argmax(flag)] if flag.any() else None
    print(f"[gate3] spoof at {t_spoof}s: flag first fires at {first}s, "
          f"pre-spoof flags={int(flag[t<t_spoof].sum())}, trust drops {trust[0]:.2f}->{trust[-1]:.3f}")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, trust, label="GNSS trust"); ax.plot(t, flag, label="spoof flag", drawstyle="steps-post")
    ax.axvline(t_spoof, c="r", label="spoof onset"); ax.set(xlabel="time (s)", title="Spoofing detection")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{out}/gate3_spoof.png", dpi=110); plt.close(fig)
    assert flag[t < t_spoof].sum() == 0 and flag[t >= t_spoof].all() and trust[-1] < 0.05, "GATE 3 FAIL"
    print("[gate3] PASS\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../out")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    gate1_calibration(a.out); gate2_remount(a.out); gate3_spoof(a.out)
    print("=== all Phase-4 gates PASS ===")
