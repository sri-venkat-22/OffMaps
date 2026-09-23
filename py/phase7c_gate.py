"""Phase-7c exit gate: the high-rate vibration/pothole front-end (a) detects
injected pothole shocks with high recall AND precision, and (b) its
pothole-rejected RMS is a clean speed cue where the raw RMS is corrupted -- a
pothole read naively spikes the vibration energy and reads as a phantom speed.

Run: PYTHONPATH=. python3 phase7c_gate.py --out ../out
"""
from __future__ import annotations
import os, sys, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from data.highrate import highrate_accel
from core_bridge import Vib


def run_frontend(hr, window_s=1.0):
    v = Vib(hr.hz)
    step = int(round(window_s * hr.hz)); N = len(hr.t)
    events_t, win = [], []
    for i in range(N):
        if v.push(hr.acc[i, 0], hr.acc[i, 1], hr.acc[i, 2], 1.0 / hr.hz):
            events_t.append(hr.t[i])
        if (i + 1) % step == 0:
            o = v.window(); lo = i + 1 - step
            win.append((hr.t[lo:i+1].mean(), hr.speed[lo:i+1].mean(), o[0], o[1], o[2], o[3]))
    return np.array(events_t), np.array(win)


def gate(out_dir):
    hr = highrate_accel(seed=2)
    events_t, win = run_frontend(hr)
    tc, spd, rms_clean, rms_raw, shock_frac, n_ev = win.T

    # --- detection ---
    inj = hr.pothole_t
    matched_inj = sum(any(abs(e - p) < 0.10 for e in events_t) for p in inj)
    tp = sum(any(abs(e - p) < 0.15 for p in inj) for e in events_t)
    recall = matched_inj / len(inj)
    precision = tp / len(events_t) if len(events_t) else 0.0
    print(f"[gate7c] high-rate {hr.hz:.0f} Hz, {len(hr.t)} samples, {len(inj)} potholes")
    print(f"[gate7c] pothole detection: recall={recall:.2f}  precision={precision:.2f}  "
          f"(events fired={len(events_t)})")

    # --- pothole rejection value ---
    clean_win = n_ev == 0                       # pothole-free windows
    pot_win = n_ev > 0                          # windows containing a pothole
    b, a = np.polyfit(rms_clean[clean_win], spd[clean_win], 1)   # rms_clean -> speed map
    corr = np.corrcoef(rms_clean[clean_win], spd[clean_win])[0, 1]
    est_clean = a + b * rms_clean[pot_win]
    est_raw = a + b * rms_raw[pot_win]
    err_clean = float(np.median(np.abs(est_clean - spd[pot_win])))
    err_raw = float(np.median(np.abs(est_raw - spd[pot_win])))
    print(f"[gate7c] clean cue: speed = {a:.2f} + {b:.2f}*rms  (corr={corr:.3f})")
    print(f"[gate7c] speed error on pothole windows: raw RMS={err_raw:.1f} m/s  ->  "
          f"pothole-rejected={err_clean:.1f} m/s")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))
    ax1.plot(hr.t, hr.acc[:, 2], lw=0.4, color="#555")
    for p in inj: ax1.axvline(p, color="orange", ls=":", lw=1)
    for e in events_t: ax1.axvline(e, color="red", ls="-", lw=0.8, alpha=0.7)
    ax1.set(xlabel="t (s)", ylabel="a_z (m/s^2)", title="shocks (orange=injected, red=detected)")
    ax2.scatter(spd[clean_win], rms_clean[clean_win], c="g", s=18, label="clean windows")
    ax2.scatter(spd[pot_win], rms_raw[pot_win], c="r", marker="x", s=40, label="pothole: raw RMS")
    ax2.scatter(spd[pot_win], rms_clean[pot_win], marker="o", s=40, facecolors="none", edgecolors="g", label="pothole: rejected")
    ax2.set(xlabel="true speed (m/s)", ylabel="vibration RMS", title="RMS vs speed"); ax2.legend(fontsize=8)
    ax3.bar(["raw RMS", "pothole-\nrejected"], [err_raw, err_clean], color=["#c0392b", "#27ae60"])
    ax3.set(ylabel="speed error on pothole windows (m/s)", title="rejection removes the phantom spike")
    fig.tight_layout(); fig.savefig(f"{out_dir}/gate7c_vib.png", dpi=110); plt.close(fig)

    assert recall >= 0.9, f"pothole recall too low: {recall:.2f}"
    assert precision >= 0.9, f"too many false pothole alarms: precision {precision:.2f}"
    assert err_clean < 2.0, f"clean speed cue too noisy: {err_clean:.1f} m/s"
    assert err_clean < 0.4 * err_raw, "pothole rejection must remove most of the raw-RMS bias"
    print("[gate7c] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../out")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    gate(a.out)
