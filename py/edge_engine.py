"""Edge engine: the phone's live fusion loop as a streaming CLI, at ANY IMU rate.

    cd py
    # phone-logger folder (imu.csv + gnss.csv), or any pair of CSVs
    PYTHONPATH=. python3 -m edge_engine --imu drive/imu.csv --gnss drive/gnss.csv --out track.csv
    # one merged CSV: IMU columns every row, GNSS columns filled only on fix rows
    PYTHONPATH=. python3 -m edge_engine --csv fog_200hz.csv --out track.csv --outage 300:360

Input columns (names are case-insensitive; see --help for unit flags):
  IMU : t | t_ns | time,  ax ay az (m/s^2),  gx gy gz (rad/s),  [mx my mz (uT)]
  GNSS: t | t_ns | time,  lat lon,  [speed (m/s)] [bearing (deg)] [cn0_mean sv_used navic_sv] [masked]
Output: one row PER IMU SAMPLE (the input rate): t, lat, lon, e, n, heading_deg,
  speed, sigma_e, sigma_n, dead_reckoning, snapped.

What runs, and at what rate (the phone's FusionEngine.kt, validated through its
host mirror phase6_check.run, with the IMU rate made a parameter):

  every IMU sample (any rate)   gravity EMAs -> yaw rate about true vertical
                                (heading_aids.yaw_rate: "slow" for cars, "coord"
                                for two-wheelers) -> idr_predict(dt, yaw). A 200 Hz
                                FOG gets a 200 Hz heading integration, not a
                                decimated one.
  every 100 ms (mean of the     alignment: core aln (re-mount CUSUM) + Level (30 s
  samples in it)                gravity -> leveled NN input) + YawAlign (GNSS-aided
                                forward axis) -> SpeedNet window; map matching.
  every 1 s                     SpeedNet (ONNX, the phone's graph + profile); while
                                dead-reckoning: Doppler self-cal'd speed update,
                                or the learned fusion head (--head) if given.
  every GNSS fix                trust / chi2 spoof gate / pos+vel update / self-cal
                                push; masked fixes (Outage Simulator, --outage)
                                feed nothing.
  first GNSS fix                seeds the filter; with no usable course (parked)
                                the heading comes from the magnetometer (sigma
                                20 deg) or is marked unknown (sigma pi).

Differences from the phone, on purpose: predict runs at the input rate (the phone
decimates to 10 Hz), and each 10 Hz step uses the MEAN of its samples (anti-aliased)
instead of one sample. At 10 Hz input both reduce to the phone's loop.
"""
from __future__ import annotations
import argparse, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import phase6_check as P
import heading_aids as HA
from core_bridge import Filter, SpeedCal, Align, gq_trust, gq_R, gq_spoof
from model.features import window_features, WIN, HZ
from model.mount import leveled_basis, LEVEL_TAU_S, TAU_MED_S

STEP_S = 1.0 / HZ              # 10 Hz alignment / NN grid
NN_EVERY = int(HZ)             # 10 steps = 1 s
TAU_FAST = 0.5                 # gravity EMA for the "fast" yaw projection (phone_log)
TAU_MAG = 1.0                  # magnetometer EMA
STOP_V, STOP_S, STOP_GYRO = 0.3, 3, 0.03   # strict stop: NN speed (m/s), consecutive s, max |gyro| (rad/s)
# map_mode="hmm" preset (Phase 9, README_PHASE9): HMM road matcher without service roads; a road
# update only on a confident match, while the filter sigma <= 25 m, if it passes a chi2 innovation
# gate, and it never moves speed or gyro bias. Train LODO: mean median drift 13.6 -> 11.7 %.
MAP_HMM = dict(exclude=("service",), sigma_max=25.0, keep=3, heading=True, chi2=6.63)


class OnnxSpeedNet:
    """The phone's SpeedNet path: ONNX graph + the profile's calibration (SpeedNet.kt)."""
    def __init__(self, prof):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(os.path.join(P.ASSETS, prof["model"]), so,
                                         providers=["CPUExecutionProvider"])
        c = prof["calib"]; self.a, self.b, self.s = c["a"], c["b"], c["s"]

    def predict(self, acc, gyro):
        x = window_features(acc, gyro)[None]
        mu, logvar = self.sess.run(["mu", "logvar"], {"imu": x})
        mu, sig = float(mu[0]), math.exp(0.5 * float(logvar[0]))
        return (mu - self.a) / self.b, max(sig / self.b * self.s, 1e-6)


class TorchSpeedNet:
    """A SpeedNet checkpoint through torch (same features + calibration as the ONNX path);
    used to run the live loop with an out-of-fold net (phase8_eval.py --lodo)."""
    def __init__(self, ckpt):
        from model.nn_model import load_net, get_calib
        import torch
        self.torch = torch
        self.net = load_net(ckpt); c = get_calib(ckpt)
        self.a, self.b, self.s = c["a"], c["b"], c["s"]

    def predict(self, acc, gyro):
        x = self.torch.from_numpy(window_features(acc, gyro)[None])
        with self.torch.no_grad():
            mu, logvar, _, _ = self.net(x)
        mu, sig = float(mu[0]), math.exp(0.5 * float(logvar[0]))
        return (mu - self.a) / self.b, max(sig / self.b * self.s, 1e-6)


class EdgeEngine:
    def __init__(self, profile=P.SHIPPED, head="profile", roads=None, map_mode="greedy",
                 vehicle="car", yaw_mode=None, declination_deg=0.0, mount_forward=None,
                 use_mag=True, speed_net=None, map_opts=None, zupt=None,
                 head_handover=False):
        self.prof = P.load_profile(profile) if isinstance(profile, str) else profile
        self.cfg = self.prof["eskf"]
        from core_bridge import LIVE_DEFAULT
        live = {**LIVE_DEFAULT, **(self.prof.get("live") or {})}
        self.live = live
        self.net = speed_net or OnnxSpeedNet(self.prof)
        if head == "profile":                  # the profile's learned fusion head, if it ships one
            head = None
            if live["fusion_head"]:
                from model.fusion_head import load_head
                head = load_head(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model",
                                              live["fusion_head"].replace(".json", ".pt")))
        self.head = head                       # learned fusion head (model/fusion_head.py) or None
        self.head_handover = head_handover     # hold the entry speed for handover_s before the head takes over
        self.roads_ll = roads                  # [(lat[], lon[], tunnel, oneway[, highway])] or None
        self.map_mode = map_mode               # "greedy" | "viterbi" | "hmm" | "off"
        self.vehicle = vehicle
        self.yaw_mode = yaw_mode or ("coord" if vehicle == "two_wheeler" else live["yaw_mode"])
        self.decl = math.radians(declination_deg)
        self.mount_forward = mount_forward     # device-frame forward axis override
        self.use_mag = use_mag
        # road-update safeguards (see _map_step); defaults = the phone's Phase-5/6 behaviour
        self.map_opts = {**dict(heading=live["mm_heading"], unique=live["mm_unique"], gate_deg=live["mm_gate_deg"],
                                # Phase 9 safeguards (defaults = off, i.e. the Phase 8 behaviour)
                                exclude=(),           # OSM classes the matcher ignores, e.g. ("service",)
                                sigma_max=None,       # snap only while the filter's position sigma <= this (m)
                                keep=None,            # road-update keep mask (1 speed, 3 speed+gyro bias); None = profile
                                corridor_sigma=0.3,   # greedy: cross-track sigma on an unambiguous road
                                # hmm mode (road_hmm.py)
                                conf=0.9,             # min posterior of the matched road
                                cross_scale=0.6,      # cross-track sigma = this x road half-width (>= 1 m)
                                heading_sigma_deg=6.0,  # road-heading update (heading=True), not near vertices
                                beta=8.0,
                                chi2=None),           # innovation gate on each road update (6.63 = 1 dof, 99 %)
                         **(MAP_HMM if map_mode == "hmm" else {}), **(map_opts or {})}
        # stop detection while dead-reckoning: "profile" = the profile's zupt_v rule (nn_real: off);
        # "strict" = SpeedNet < STOP_V for STOP_S s in a row AND a still gyro (see README_EDGE)
        self.zupt = zupt or ("strict" if live["zupt_strict"] else "profile")
        self.still_s = 0

        self.f = Filter(); self.f.set_noise(self.cfg["arw"], self.cfg["brw"], self.cfg["srw"])
        keep = self.map_opts["keep"]
        self.f.set_map_keep_speed(int(self.cfg.get("mm_keep_speed", False)) if keep is None else keep)
        self.hmm = None
        self.sc = SpeedCal(); self.aln = Align(); self.ya = HA.YawAlign()
        self.handover = int(self.cfg.get("handover_s", 0.0) * HZ)
        self.rw = None
        # streaming state
        self.t_last = None; self.gf = None; self.gs = None; self.lvl_g = None; self.lvl_gm = None
        self.mag = None
        self.bin = [0.0] * 6; self.bin_n = 0; self.t_step = None
        self.ring_a = []; self.ring_g = []
        self.init = False; self.lat0 = self.lon0 = None
        self.steps_nn = 0; self.since_gnss = 0; self.dr_steps = 0; self.rejects = 0
        self.vnn_raw = 0.0; self.sig = 1.0; self.sumsig = 0.0; self.npush = 0
        self.k, self.c = 1.0, 0.0
        self.masked = False; self.snapped = False
        self.hist = []; self.vit_way = -1; self.vit_cor = False
        self.basis = None; self.v_entry = 0.0; self.head_state = None
        self.pre = []                         # (doppler, nn_raw) of trusted fixes, for the head's context
        self.heading_seed = None              # where the initial heading came from

    # ---------------- helpers ----------------
    def dead_reckoning(self):
        return self.masked or self.since_gnss > P.GNSS_STALE_STEPS

    def _forward(self):
        if self.mount_forward is not None:
            return self.mount_forward
        if self.ya.valid and self.basis is not None:
            return self.ya.forward(self.basis)
        return HA.default_forward(self.lvl_g if self.lvl_g is not None else self.gf)

    def _en(self, lat, lon):
        k = math.pi / 180.0 * P_R
        return (lon - self.lon0) * math.cos(math.radians(self.lat0)) * k, (lat - self.lat0) * k

    def _ll(self, e, n):
        k = math.pi / 180.0 * P_R
        return self.lat0 + n / k, self.lon0 + e / (math.cos(math.radians(self.lat0)) * k)

    # ---------------- IMU, any rate ----------------
    def imu(self, t, ax, ay, az, gx, gy, gz, mx=float("nan"), my=float("nan"), mz=float("nan")):
        dt = STEP_S if self.t_last is None else t - self.t_last
        self.t_last = t
        a = (ax, ay, az); w = (gx, gy, gz)
        if self.gf is None:
            self.gf = list(a); self.gs = list(a)
        else:
            af = 1.0 - math.exp(-dt / TAU_FAST); as_ = 1.0 - math.exp(-dt / LEVEL_TAU_S)
            for i in range(3):
                self.gf[i] += af * (a[i] - self.gf[i]); self.gs[i] += as_ * (a[i] - self.gs[i])
        if mx == mx:                                         # not NaN
            if self.mag is None:
                self.mag = [mx, my, mz]
            else:
                am = 1.0 - math.exp(-dt / TAU_MAG)
                self.mag[0] += am * (mx - self.mag[0]); self.mag[1] += am * (my - self.mag[1]); self.mag[2] += am * (mz - self.mag[2])

        if self.init:
            v = self.f.state()[3] if self.yaw_mode == "coord" else 0.0
            fwd = self._forward() if self.yaw_mode == "coord" else None
            yaw = HA.yaw_rate(self.yaw_mode, w, self.gf, self.gs, fwd, v)
            self.f.predict(dt, yaw)
            self.yaw = yaw
        # 10 Hz bin (mean of its samples)
        b = self.bin
        b[0] += ax; b[1] += ay; b[2] += az; b[3] += gx; b[4] += gy; b[5] += gz; self.bin_n += 1
        if self.t_step is None:
            self.t_step = t - STEP_S
        if t - self.t_step >= STEP_S - 1e-3 - 0.25 * min(dt, STEP_S):
            m = [x / self.bin_n for x in b]
            self.bin = [0.0] * 6; self.bin_n = 0
            self._step10(t, t - self.t_step, m[:3], m[3:])
            self.t_step = t

    def _step10(self, t, dt, a, w):
        """The phone's 10 Hz step (FusionEngine.onImu after the decimator)."""
        self.aln.update(a[0], a[1], a[2], w[0], w[1], w[2], dt)
        remount = self.aln.get()[3]
        # Level.kt: slow (30 s) + medium (5 s) gravity, snap on re-mount
        if self.lvl_g is None:
            self.lvl_g = list(a); self.lvl_gm = list(a)
        else:
            if remount:
                self.lvl_g = list(self.lvl_gm); self.ya.reset()
            ae = 1.0 - math.exp(-1.0 / (LEVEL_TAU_S * HZ)); am = 1.0 - math.exp(-1.0 / (TAU_MED_S * HZ))
            for i in range(3):
                self.lvl_g[i] += ae * (a[i] - self.lvl_g[i]); self.lvl_gm[i] += am * (a[i] - self.lvl_gm[i])
        R = leveled_basis(self.lvl_g); self.basis = R
        al = R @ a; wl = R @ w; wl[2] *= HA.YAW_SIGN
        self.ya.add_acc(al[0], al[1])
        self.ring_a.append(al); self.ring_g.append(wl)
        if len(self.ring_a) > WIN:
            self.ring_a.pop(0); self.ring_g.pop(0)
        if not self.init:
            return
        f = self.f
        dr = self.dead_reckoning()
        self.dr_steps = self.dr_steps + 1 if dr else 0
        if self.dr_steps == 1:                               # outage starts: freeze the entry context
            self.v_entry = f.state()[3]
            if self.head is not None:
                self.head_state = self.head.start(self._head_context())
        self.steps_nn += 1
        if self.steps_nn >= NN_EVERY and len(self.ring_a) >= WIN:
            self.steps_nn = 0
            self.vnn_raw, self.sig = self.net.predict(np.array(self.ring_a), np.array(self.ring_g))
            g1 = np.array(self.ring_g[-NN_EVERY:])
            still = self.vnn_raw < STOP_V and float(np.linalg.norm(g1, axis=1).max()) < STOP_GYRO
            self.still_s = self.still_s + 1 if still else 0
            if dr and self.zupt == "strict" and self.still_s >= STOP_S:
                f.update_zupt(float(np.mean(g1[:, 2])))
            elif dr:
                if self.head is not None:
                    v, s = self.head.step(self.head_state, self._head_features())   # always: GRU state
                    if not (self.head_handover and self.dr_steps <= self.handover):
                        f.update_speed(v, s)
                elif self.dr_steps > self.handover:
                    vused = P.cal_apply(self.vnn_raw, self.k, self.c)
                    zv = self.cfg["zupt_v"]
                    if zv is not None and vused < zv:
                        f.update_zupt(getattr(self, "yaw", 0.0))
                    else:
                        f.update_speed(vused, self.sig * self.cfg["sig_scale"])
        if self.cfg["curv"] and dr:
            st = f.state(); psidot = getattr(self, "yaw", 0.0) - st[4]
            if abs(psidot) > P.CURV_GATE:
                left = np.cross(R[2], self._forward())
                f.update_curvature(float(np.dot(a, left)), psidot, self.cfg["curv_sigma"])
        self.since_gnss += 1
        self.snapped = False
        if self.hmm is not None:
            self._map_step_hmm(dt, dr)
        elif self.rw is not None and self.map_mode != "off" and dr:
            self._map_step()
        elif not dr:
            self.hist.clear(); self.vit_way = -1

    def _pos_sigma(self):
        cv = self.f.cov()
        return math.sqrt(max(cv[0] + cv[1], 0.0) / 2)

    def _map_step_hmm(self, dt, dr):
        """1 Hz HMM forward filter, always running (so it is locked on at outage start);
        a road update only while dead-reckoning, on a confident match, while the filter
        is still tight enough for the match to be trusted (DrishtiNav's gating)."""
        st = self.f.state(); o = self.map_opts
        self.hmm.add_travel(abs(st[3]) * dt)
        self.hmm_steps = getattr(self, "hmm_steps", 0) + 1
        if self.hmm_steps % NN_EVERY:
            return
        ps = self._pos_sigma()
        m = self.hmm.update((st[0], st[1]), st[2], abs(st[3]), ps)
        if m is None or not dr or m["conf"] < o["conf"] or abs(st[3]) < 1.0:
            return
        if o["sigma_max"] is not None and ps > o["sigma_max"]:
            return
        brg = m["bearing"]
        nE, nN = -math.cos(brg), math.sin(brg)
        cross = (m["foot"][0] - st[0]) * nE + (m["foot"][1] - st[1]) * nN
        sc = max(1.0, o["cross_scale"] * m["half_width"]); cv = self.f.cov()
        if o["chi2"] is not None and cross * cross / (nE * nE * cv[0] + nN * nN * cv[1] + sc * sc) > o["chi2"]:
            return                                  # the road disagrees with the filter: do not trust it
        self.f.update_crosstrack(nE, nN, cross, sc)
        if o["heading"] and m["seg_len"] >= 25.0 and m["end_dist"] >= 8.0:
            sh = math.radians(o["heading_sigma_deg"]); dpsi = P._wrap(brg - st[2])
            if o["chi2"] is None or dpsi * dpsi / (self.f.cov()[2] + sh * sh) <= o["chi2"]:
                self.f.update_heading(brg, sh)
        self.snapped = True

    def _map_step(self):
        st = self.f.state()
        if self.map_opts["sigma_max"] is not None and self._pos_sigma() > self.map_opts["sigma_max"]:
            return
        if self.map_mode == "viterbi":
            if self.dr_steps % NN_EVERY == 1:                # 1 Hz fixed-lag decode
                self.hist.append((st[0], st[1], st[2]))
                if len(self.hist) > self.rw.LAG:
                    self.hist.pop(0)
                self.vit_way, self.vit_cor = self.rw.decode(self.hist)
            m = self.rw.project(self.vit_way, st[0], st[1], st[2])
            if m["matched"]:
                m["corridor"] = self.vit_cor
        else:
            m = self.rw.match(st[0], st[1], st[2])
        if not m["matched"] or abs(m["cross"]) > P.MM_MAX_CROSS:
            return
        o = self.map_opts
        if o["unique"] and not m["corridor"]:          # ambiguous: another road within 30 m
            return
        brg = m["bearing"]
        db = abs(P._wrap(brg - st[2]))
        if min(db, math.pi - db) > math.radians(o["gate_deg"]):
            return
        self.f.update_crosstrack(-math.cos(brg), math.sin(brg), m["cross"],
                                 o["corridor_sigma"] if m["corridor"] else self.cfg["mm_cross_sigma"])
        if o["heading"]:
            tgt = brg if db < math.pi / 2 else brg + math.pi
            self.f.update_heading(tgt, math.radians(1 if m["corridor"] else self.cfg["mm_heading_sigma_deg"]))
        self.snapped = True

    # ---------------- learned fusion head hooks ----------------
    def _head_context(self):
        return dict(v0=self.v_entry, k=self.k, c=self.c,
                    cal_ok=P.K_MIN <= self.k <= P.K_MAX, pre=list(self.pre[-120:]))

    def _head_features(self):
        return dict(vnn=self.vnn_raw, sig=self.sig, tau=self.dr_steps / HZ,
                    acc=np.array(self.ring_a[-NN_EVERY:]), gyro=np.array(self.ring_g[-NN_EVERY:]))

    # ---------------- GNSS ----------------
    def gnss(self, t, lat, lon, speed=float("nan"), bearing_deg=float("nan"),
             cn0=40.0, sv=10, navic=0, masked=False):
        self.masked = bool(masked)
        if self.lat0 is None:
            self.lat0, self.lon0 = lat, lon
            if self.roads_ll is not None and self.map_mode == "hmm":
                from road_hmm import RoadGraph, HMMMatcher
                g = RoadGraph.cached(self.roads_ll, lat, lon, tuple(self.map_opts["exclude"]))
                self.hmm = HMMMatcher(g, beta=self.map_opts["beta"])
            elif self.roads_ll is not None and self.map_mode != "off":
                from road_window import RoadWindow
                ex = set(self.map_opts["exclude"])
                self.rw = RoadWindow([w[:4] for w in self.roads_ll if (w[4] if len(w) > 4 else None) not in ex],
                                     lat, lon)
        eg, ng = self._en(lat, lon)
        has_v = speed == speed; has_b = bearing_deg == bearing_deg
        brg = math.radians(bearing_deg) if has_b else 0.0
        f = self.f
        if not self.init:
            if masked:
                return
            v0 = speed if has_v else 0.0
            if has_b and (not has_v or speed >= 1.0):      # a real course over ground
                f.init(eg, ng, brg, v0); self.heading_seed = "gnss"
            else:                                           # parked: GNSS bearing is noise
                h = None
                if self.use_mag and self.mag is not None and self.gs is not None:
                    h = HA.mag_heading(self.mag, self.gs, self._forward(), self.decl)
                if h is not None:
                    f.init(eg, ng, h, v0); f.set_heading_sigma(HA.MAG_SIGMA)
                    self.heading_seed = "magnetometer"
                else:
                    f.init(eg, ng, 0.0, v0); f.set_heading_sigma(HA.UNKNOWN_SIGMA)
                    self.heading_seed = "unknown"
            self.init = True; self.since_gnss = 0
            return
        if masked:
            return
        st = f.state(); cov = f.cov()
        de, dn = eg - st[0], ng - st[1]
        chi2 = de * de / (cov[0] + P.GNSS_BASE ** 2) + dn * dn / (cov[1] + P.GNSS_BASE ** 2)
        trust = gq_trust(cn0, sv, navic, P.NOMINAL_DOP, chi2)
        spoof = gq_spoof(cn0, sv, chi2, 0.0)
        if spoof:
            self.rejects += 1
            if self.rejects >= P.REACQ_FIXES:
                f.init(eg, ng, brg if has_b else st[2], speed if has_v else st[3])
                self.rejects = 0; spoof = False
                trust = gq_trust(cn0, sv, navic, P.NOMINAL_DOP, 0.0)
        else:
            self.rejects = 0
        if spoof:
            return
        if trust >= P.TRUST_APPLIED:
            self.since_gnss = 0
        f.update_gnss_pos(eg, ng, math.sqrt(gq_R(trust)))
        if has_v and has_b:
            sv_sig = min(max(P.DOPPLER_SIGMA / max(trust, 3e-3), 0.1), 50.0)
            sp_sig = min(max(3 * P.DEG / max(trust, 3e-3), 1 * P.DEG), 90 * P.DEG)
            f.update_gnss_vel(speed, brg, sv_sig, sp_sig)
        elif has_v:
            f.update_speed(speed, min(max(P.DOPPLER_SIGMA / max(trust, 3e-3), 0.1), 50.0))
        if has_v and trust >= P.TRUST_APPLIED:
            self.ya.on_fix(t, speed)
            self.pre.append((speed, self.vnn_raw))
            if len(self.pre) > 600:
                self.pre.pop(0)
            if self.vnn_raw > 0:
                self.sumsig += self.sig * self.sig; self.npush += 1
                self.sc.set_lambda(P.DOPPLER_SIGMA ** 2 / max(self.sumsig / self.npush, 1e-9))
                self.sc.push(self.vnn_raw, speed, trust)
                self.k, self.c, _ = self.sc.fit()

    def state(self):
        return self.f.state(), self.f.cov()


P_R = 6_371_000.0


# ---------------- CSV front end ----------------
def _col(df, *names, required=True):
    low = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    if required:
        raise KeyError(f"none of {names} in columns {list(df.columns)}")
    return None


def _time(df):
    c = _col(df, "t_ns", "time_ns", "timestamp_ns", required=False)
    if c is not None:
        return df[c].to_numpy(np.float64) * 1e-9
    return df[_col(df, "t", "time", "t_s", "timestamp")].to_numpy(np.float64)


def read_inputs(imu_path=None, gnss_path=None, csv_path=None, gyro_deg=False, acc_g=False, kmph=False):
    """-> (imu dict of arrays, gnss dict of arrays), times in seconds (shared origin)."""
    import pandas as pd
    if csv_path:
        df = pd.read_csv(csv_path)
        gi = df
        latc = _col(df, "lat", "latitude")
        fix = np.isfinite(pd.to_numeric(df[latc], errors="coerce").to_numpy(float))
        g = df[fix]
        # a merged high-rate file often repeats the last fix; keep rows where it changes
        key = g[[latc, _col(g, "lon", "longitude")]].to_numpy(float)
        new = np.r_[True, np.any(np.diff(key, axis=0) != 0, axis=1)]
        gg = g[new]
    else:
        gi = pd.read_csv(imu_path); gg = pd.read_csv(gnss_path)
    ti = _time(gi); tg = _time(gg)
    t0 = min(ti[0], tg[0]) if len(tg) else ti[0]
    acc = gi[[_col(gi, x) for x in ("ax", "ay", "az")]].to_numpy(float) * (9.80665 if acc_g else 1.0)
    gyr = gi[[_col(gi, x) for x in ("gx", "gy", "gz")]].to_numpy(float) * (math.pi / 180 if gyro_deg else 1.0)
    mc = [_col(gi, x, required=False) for x in ("mx", "my", "mz")]
    mag = gi[mc].to_numpy(float) if all(mc) else np.full((len(gi), 3), np.nan)
    get = lambda *n, d=np.nan: (gg[_col(gg, *n, required=False)].to_numpy(float)
                                if _col(gg, *n, required=False) else np.full(len(gg), d))
    gn = dict(t=tg - t0, lat=get("lat", "latitude"), lon=get("lon", "longitude"),
              speed=get("speed", "speed_mps") * (1 / 3.6 if kmph else 1.0), bearing=get("bearing", "course"),
              cn0=get("cn0_mean", "cn0", d=40.0), sv=get("sv_used", "sv", d=10), navic=get("navic_sv", d=0),
              masked=get("masked", d=0))
    return dict(t=ti - t0, acc=acc, gyro=gyr, mag=mag), gn


def run(eng, imu, gn, outages=(), progress=False):
    """Interleave IMU and GNSS by time (a fix at the same instant as a sample is applied
    after it, as in phase6_check.run); returns the per-IMU-sample output arrays."""
    N = len(imu["t"]); out = np.empty((N, 10))
    ti, A, Gy, Mg = imu["t"], imu["acc"], imu["gyro"], imu["mag"]
    tg = gn["t"]; j = 0; ng = len(tg)
    spans = list(outages)
    for i in range(N):
        t = ti[i]
        eng.imu(t, A[i, 0], A[i, 1], A[i, 2], Gy[i, 0], Gy[i, 1], Gy[i, 2], Mg[i, 0], Mg[i, 1], Mg[i, 2])
        while j < ng and tg[j] <= t:
            tj = tg[j]
            msk = bool(gn["masked"][j]) or any(a <= tj < b for a, b in spans)
            eng.gnss(tj, gn["lat"][j], gn["lon"][j], gn["speed"][j], gn["bearing"][j],
                     gn["cn0"][j], int(gn["sv"][j]), int(gn["navic"][j]), msk)
            j += 1
        if eng.init:
            st = eng.f.state(); cv = eng.f.cov()
            out[i] = (t, st[0], st[1], st[2], st[3], cv[0], cv[1], eng.dead_reckoning(), eng.snapped, 1)
        else:
            out[i] = (t, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1, 0, 0)
        if progress and i % 200000 == 0 and i:
            print(f"  {i}/{N} samples", flush=True)
    return out


def write_output(eng, out, path):
    import pandas as pd
    ok = out[:, 9] > 0
    lat = np.full(len(out), np.nan); lon = np.full(len(out), np.nan)
    if eng.lat0 is not None:
        lat[ok], lon[ok] = eng._ll(out[ok, 1], out[ok, 2])
    pd.DataFrame(dict(t=out[:, 0], lat=lat, lon=lon, e=out[:, 1], n=out[:, 2],
                      heading_deg=np.degrees(out[:, 3]) % 360, speed=out[:, 4],
                      sigma_e=np.sqrt(out[:, 5]), sigma_n=np.sqrt(out[:, 6]),
                      dead_reckoning=out[:, 7].astype(int), snapped=out[:, 8].astype(int))
                 ).to_csv(path, index=False, float_format="%.7f")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--imu"); ap.add_argument("--gnss"); ap.add_argument("--csv")
    ap.add_argument("--dir", help="phone-logger folder holding imu.csv + gnss.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default=P.SHIPPED)
    ap.add_argument("--head", help="learned fusion head (.pt from model/fusion_head.py)")
    ap.add_argument("--roads", help="roads.bin (tools/osm_layers.py) for road snapping")
    ap.add_argument("--map-mode", choices=["greedy", "viterbi", "hmm", "off"], default="greedy")
    ap.add_argument("--vehicle", choices=["car", "two_wheeler"], default="car")
    ap.add_argument("--yaw-mode", choices=["fast", "slow", "coord"])
    ap.add_argument("--declination-deg", type=float, default=0.0)
    ap.add_argument("--no-mag", action="store_true")
    ap.add_argument("--outage", action="append", default=[], help="t0:t1 seconds, masks GNSS (repeatable)")
    ap.add_argument("--gyro-deg", action="store_true"); ap.add_argument("--acc-g", action="store_true")
    ap.add_argument("--kmph", action="store_true")
    a = ap.parse_args(argv)
    if a.dir:
        a.imu, a.gnss = os.path.join(a.dir, "imu.csv"), os.path.join(a.dir, "gnss.csv")
    if not (a.csv or (a.imu and a.gnss)):
        ap.error("give --csv, or --imu and --gnss, or --dir")
    t0 = time.perf_counter()
    imu, gn = read_inputs(a.imu, a.gnss, a.csv, a.gyro_deg, a.acc_g, a.kmph)
    t_read = time.perf_counter() - t0
    roads = None
    if a.roads:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
        from osm_layers import read_roads_bin
        roads = read_roads_bin(a.roads, with_class=True)
    head = None
    if a.head:
        from model.fusion_head import load_head
        head = load_head(a.head)
    eng = EdgeEngine(a.profile, head=head, roads=roads, map_mode=a.map_mode, vehicle=a.vehicle,
                     yaw_mode=a.yaw_mode, declination_deg=a.declination_deg, use_mag=not a.no_mag)
    outs = [tuple(float(x) for x in s.split(":")) for s in a.outage]
    t1 = time.perf_counter()
    out = run(eng, imu, gn, outs, progress=True)
    t_run = time.perf_counter() - t1
    write_output(eng, out, a.out)
    N = len(imu["t"]); span = imu["t"][-1] - imu["t"][0]
    rate = (N - 1) / span if span > 0 else float("nan")
    print(f"{N} IMU samples at {rate:.1f} Hz ({span:.0f} s), {len(gn['t'])} GNSS fixes; "
          f"heading seed: {eng.heading_seed}; yaw mode: {eng.yaw_mode}")
    print(f"read {t_read:.2f} s, fuse {t_run:.2f} s = {N / t_run:,.0f} samples/s "
          f"({span / t_run:.0f}x real time) -> {a.out}")


if __name__ == "__main__":
    main()
