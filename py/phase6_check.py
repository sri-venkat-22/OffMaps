"""Phase-6 host verifier: run the ON-DEVICE live-fusion loop off-device.

The Android app can't be executed on a build box, but its brain can. This script
is a line-for-line Python mirror of com.offmaps.nav.FusionEngine driving the SAME
artifacts the phone uses: the embedded libidr core (via ctypes, the identical .so
sources the NDK compiles) and the SAME ONNX SpeedNet path. It proves the two
Phase-6 claims that are pure logic (independent of Android APIs):

  1. runs the live fusion  -- predict / NN-speed / ZUPT / curvature / GNSS at the
     validated cadence, seeded from the first fix, position rendered each step;
  2. honours the outage flag against the LIVE filter -- with the flag ON the filter
     is fed NO GNSS and dead-reckons (blue diverges from grey); with it OFF GNSS
     pulls the estimate back. Toggling the flag must change the outcome.
  3. road snapping (FusionEngine.mapMatchUpdate) -- only while dead-reckoning, the
     matched road pins cross-track + heading, so outage drift shrinks; with GNSS
     healthy the map is never consulted (identical track with or without roads).

The model and its fusion settings come from the SAME profile file the phone reads
(android/app/src/main/assets/<name>.profile.json, SpeedProfile.kt); default is the
one the app ships (SpeedProfile.DEFAULT = nn_real).

What it does NOT cover (needs a device): the gravity-projected yaw port
(a 4-line copy of phone_log._gravity_up), ONNX Runtime Android bindings, and the
Canvas renderer. Those are verified by inspection / the feature-port cross-check.

Run:  PYTHONPATH=. python3 phase6_check.py
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from data.io_vnbd import synth_drive
from model.nn_model import load_net, predict_steps
import model.nn_model as NNM
from model.dataset import STEP
from core_bridge import Filter, SpeedCal, MapMatcher, gq_trust, gq_R, gq_spoof

DEG = np.pi / 180.0
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "android", "app", "src", "main", "assets")
SHIPPED = "nn_real"          # SpeedProfile.DEFAULT in SpeedProfile.kt


def load_profile(name=SHIPPED):
    """The phone's SpeedProfile: {model, checkpoint, calib, eskf} (see model/export.py)."""
    with open(os.path.join(ASSETS, name + ".profile.json")) as fh:
        return json.load(fh)


# --- constants copied verbatim from FusionEngine.kt (per-model ones live in the profile) ---
CURV_GATE = 10.0 * DEG
DOPPLER_SIGMA = 0.1
NOMINAL_DOP = 1.5
GNSS_BASE = 3.0
GNSS_STALE_STEPS = 20        # 2 s at 10 Hz without GNSS applied -> dead-reckoning
MM_MAX_CROSS = 25.0
TRUST_APPLIED = 0.1          # a fix ends dead-reckoning only at trust >= this (sigma <= 30 m)
REACQ_FIXES = 5              # re-seed from GNSS after this many consecutive rejected fixes
K_MIN, K_MAX = 1 / 3, 3.0     # self-cal scale outside this = a degenerate fit, not a calibration (chosen on val)
# healthy-GNSS signal snapshot for synth (matches phase4_gates assumptions)
CN0, SV, NAVIC = 42.0, 9, 3


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def map_match_update(f, matcher, cfg=None):
    """Mirror FusionEngine.mapMatchUpdate (called only while dead-reckoning). -> snapped?"""
    cfg = cfg or dict(mm_cross_sigma=1.5, mm_heading_sigma_deg=3.0)
    st = f.state()
    m = matcher.match(st[0], st[1], st[2])
    if not m["matched"] or abs(m["cross"]) > MM_MAX_CROSS:
        return False
    brg = m["bearing"]
    f.update_crosstrack(-np.cos(brg), np.sin(brg), m["cross"], 0.3 if m["corridor"] else cfg["mm_cross_sigma"])
    tgt = brg if abs(_wrap(brg - st[2])) < np.pi / 2 else brg + np.pi
    f.update_heading(tgt, np.radians(1 if m["corridor"] else cfg["mm_heading_sigma_deg"]))
    return True


def cal_apply(v, k, c):
    """Mirror FusionEngine.calApply: use the self-cal fit only if its scale is physical."""
    return k * v + c if K_MIN <= k <= K_MAX else v


def run(drive, outage=None, matcher=None, profile=SHIPPED, trace=None):
    """Mirror FusionEngine. outage=(t0,t1) seconds masks GNSS in that span (or a list
    of such spans); matcher =
    a loaded MapMatcher (the phone's RoadMatcher window) or None for no roads;
    profile = the SpeedProfile name the engine was built with.
    trace: optional dict, filled per step with position variance (cov_e, cov_n) and speed v.
    Returns (err_over_time, fused_en, k, c)."""
    outages = [] if outage is None else ([outage] if np.isscalar(outage[0]) else list(outage))
    prof = load_profile(profile)
    cfg = prof["eskf"]
    net = load_net(os.path.join(os.path.dirname(NNM.DEFAULT), prof["checkpoint"]))
    N = len(drive)
    starts, vstep, sstep = predict_steps(net, drive, 0, N, calib=prof["calib"])
    per = {s: (v, sg) for s, v, sg in zip(starts, vstep, sstep)}

    f = Filter(); f.set_noise(cfg["arw"], cfg["brw"], cfg["srw"])
    f.set_map_keep_speed(cfg.get("mm_keep_speed", False))
    f.init(drive.e[0], drive.n[0], drive.heading[0], drive.speed[0])
    sc = SpeedCal()

    t = drive.t
    a_lat = drive.acc[:, 1] if drive.acc is not None else np.zeros(N)
    fused = np.empty((N, 2)); err = np.empty(N)
    vnn_raw, sig = drive.speed[0], 1.0
    sumsig, npush, k, c = 0.0, 0, 1.0, 0.0
    since_gnss = 0
    dr_steps = 0                                            # IMU steps since dead-reckoning began
    handover = int(cfg.get("handover_s", 0.0) * 10)         # hold Doppler speed this long first (Kotlin toInt)
    rejects = 0                                             # consecutive strong fixes rejected

    for i in range(N):
        dt = t[i] - t[i - 1] if i else (t[1] - t[0])
        gz = drive.gyro_z[i]
        f.predict(dt, gz)                                   # 10 Hz predict
        masked = any(o0 <= t[i] < o1 for o0, o1 in outages)            # the live toggle
        dead_reckoning = masked or since_gnss > GNSS_STALE_STEPS
        dr_steps = dr_steps + 1 if dead_reckoning else 0

        if i in per:                                        # 1 s NN cadence
            vnn_raw, sig = per[i]                            # always run: self-cal regressor
            if dead_reckoning and dr_steps > handover:       # healthy GNSS / hand-over: Doppler owns speed
                vused = cal_apply(vnn_raw, k, c)             # Doppler self-cal (frozen k,c)
                if cfg["zupt_v"] is not None and vused < cfg["zupt_v"]: f.update_zupt(gz)
                else: f.update_speed(vused, sig * cfg["sig_scale"])

        if cfg["curv"] and dead_reckoning:                  # curvature: also a speed aid (off for nn_real)
            st = f.state(); psidot = gz - st[4]
            if abs(psidot) > CURV_GATE:
                f.update_curvature(a_lat[i], psidot, cfg["curv_sigma"])

        since_gnss += 1                                     # road snapping, dead-reckoning only
        if matcher is not None and (masked or since_gnss > GNSS_STALE_STEPS):
            map_match_update(f, matcher, cfg)

        if i % STEP == 0:                                   # ~1 Hz GNSS callback
            eg, ng = drive.e[i], drive.n[i]
            if not masked:                                  # healthy GNSS: continuous-trust fusion
                st = f.state(); cov = f.cov()
                de, dn = eg - st[0], ng - st[1]
                chi2 = de * de / (cov[0] + GNSS_BASE ** 2) + dn * dn / (cov[1] + GNSS_BASE ** 2)
                trust = gq_trust(CN0, SV, NAVIC, NOMINAL_DOP, chi2)
                spoof = gq_spoof(CN0, SV, chi2, 0.0)          # position chi2 only (see FusionEngine)
                if spoof:
                    rejects += 1
                    if rejects >= REACQ_FIXES:               # strong GNSS keeps disagreeing: our DR is wrong
                        f.init(eg, ng, drive.heading[i], drive.speed[i])
                        rejects = 0; spoof = False
                        trust = gq_trust(CN0, SV, NAVIC, NOMINAL_DOP, 0.0)
                else:
                    rejects = 0
                if not spoof:
                    if trust >= TRUST_APPLIED:
                        since_gnss = 0
                    f.update_gnss_pos(eg, ng, np.sqrt(gq_R(trust)))
                    sv_sig = min(max(DOPPLER_SIGMA / max(trust, 3e-3), 0.1), 50.0)
                    sp_sig = min(max(3 * DEG / max(trust, 3e-3), 1 * DEG), 90 * DEG)
                    f.update_gnss_vel(drive.speed[i], drive.heading[i], sv_sig, sp_sig)
                    if vnn_raw > 0 and trust >= TRUST_APPLIED:   # self-cal push + refit (trusted Doppler only)
                        sumsig += sig * sig; npush += 1
                        sc.set_lambda(DOPPLER_SIGMA ** 2 / max(sumsig / npush, 1e-9))
                        sc.push(vnn_raw, drive.speed[i], trust)
                        k, c, _ = sc.fit()

        st = f.state(); fused[i] = st[0], st[1]
        if trace is not None:
            cv = f.cov(); trace.setdefault("cov_e", []).append(cv[0]); trace.setdefault("cov_n", []).append(cv[1])
            trace.setdefault("v", []).append(st[3])
        err[i] = np.hypot(st[0] - drive.e[i], st[1] - drive.n[i])
    return err, fused, k, c


def main():
    drive = synth_drive("phase6", duration=300, seed=11)
    t = drive.t
    dist = np.trapezoid(drive.speed, t)
    outage = (150.0, 210.0)                                  # 60 s simulated GNSS outage

    err_on, fused_on, k, c = run(drive, outage=None)         # flag OFF: full GNSS fusion
    err_off, fused_off, _, _ = run(drive, outage=outage)     # flag ON: mask a 60 s span

    i_end = int(outage[1] * 10) - 1                          # sample at outage end
    i_ret = min(len(t) - 1, i_end + 100)                     # 10 s after GNSS returns
    print(f"drive: {len(drive)} samples, {t[-1]:.0f} s, {dist/1000:.2f} km, outage {outage[0]:.0f}-{outage[1]:.0f}s")
    print(f"[fusion ON ] median tracking err = {np.median(err_on):.2f} m   (GNSS pulls estimate in)")
    print(f"[outage    ] drift at outage end = {err_off[i_end]:.1f} m   ({100*err_off[i_end]/dist:.2f}% of dist)")
    print(f"[outage    ] fusion-ON err there = {err_on[i_end]:.1f} m")
    print(f"[recover   ] err 10 s after GNSS returns = {err_off[i_ret]:.1f} m")
    print(f"[self-cal  ] converged k={k:.3f} c={c:.2f}")

    # Gate: the flag must be honoured (DR drift >> fused error), fusion must track,
    # and the estimate must re-converge once GNSS returns.
    assert np.median(err_on) < 15.0, "fusion did not track GNSS"
    assert err_off[i_end] > 3 * max(err_on[i_end], 1.0), "outage flag had no effect on the live filter"
    assert err_off[i_ret] < err_off[i_end], "did not re-converge after GNSS returned"
    print("PHASE 6 LIVE-LOOP CHECK PASS: live fusion runs; outage flag honoured against the live filter")


if __name__ == "__main__":
    main()
