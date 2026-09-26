// The phone's live loop (FusionEngine.kt, host twin py/edge_engine.py) in JavaScript.
// Same stages as py/ablation.py:
//   physics   gyro heading + the speed at outage entry, held (no AI while dead-reckoning)
//   speednet  + SpeedNet, Doppler self-calibrated (hold 10 s, then SpeedNet)
//   head      + learned fusion head (speed and its sigma from a GRU)
//   map       + HMM road matching (only when roads are given): the app as it ships
// py/tests/test_web_engine.py runs this file under Node against edge_engine.py.

import { Filter, SpeedCal, Align, gqTrust, gqR, gqSpoof, DEG } from "./core.js";
import { YAW_SIGN, MAG_SIGMA, UNKNOWN_SIGMA, YawAlign, yawRate, magHeading, defaultForward, leveledBasis, mat3vec, wrapPy } from "./aids.js";
import { RoadGraph, HMMMatcher } from "./hmm.js";

export const HZ = 10.0, STEP_S = 0.1, NN_EVERY = 10;
const TAU_FAST = 0.5, TAU_MAG = 1.0, LEVEL_TAU_S = 30.0, TAU_MED_S = 5.0;
const STOP_V = 0.3, STOP_S = 3, STOP_GYRO = 0.03;
// phase6_check.py == FusionEngine.kt constants
const CURV_GATE = 10.0 * DEG, DOPPLER_SIGMA = 0.1, NOMINAL_DOP = 1.5, GNSS_BASE = 3.0, GNSS_STALE_STEPS = 20;
const TRUST_APPLIED = 0.1, REACQ_FIXES = 5, K_MIN = 1 / 3, K_MAX = 3.0;
const R_EARTH = 6371000.0;
export const MAP_HMM = { exclude: ["service"], sigmaMax: 25.0, keep: 3, heading: true, chi2: 6.63,
                         conf: 0.9, crossScale: 0.6, headingSigmaDeg: 6.0, beta: 8.0 };
const LIVE_DEFAULT = { yaw_mode: "fast", fusion_head: null, zupt_strict: false };

const calApply = (v, k, c) => (K_MIN <= k && k <= K_MAX ? k * v + c : v);
const norm3 = (a) => Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** Shares SpeedNet's 1 s predictions between stages: every stage sees the same window at the same step. */
export class SharedNet {
  constructor(net) { this.net = net; this.win = net.win; this.cache = new Map(); }
  predict(acc, gyro, key) {
    if (key !== undefined && this.cache.has(key)) return this.cache.get(key);
    const r = this.net.predict(acc, gyro);
    if (key !== undefined) this.cache.set(key, r);
    return r;
  }
}

export class EdgeEngine {
  /**
   * profile: profile.json; net: SpeedNet or SharedNet; head: FusionHead or null;
   * roads: ways for the HMM (hmm.js) or null; drSpeed false = the no-AI stage.
   */
  constructor({ profile, net, head = null, roads = null, mapMode = "off", vehicle = "car", drSpeed = true,
                useMag = true, declinationDeg = 0.0 }) {
    this.prof = profile; this.cfg = profile.eskf;
    this.live = { ...LIVE_DEFAULT, ...(profile.live || {}) };
    this.net = net; this.head = head; this.drSpeed = drSpeed;
    this.roads = roads; this.mapMode = roads ? mapMode : "off";
    this.vehicle = vehicle;
    this.yawMode = vehicle === "two_wheeler" ? "coord" : this.live.yaw_mode;
    this.decl = declinationDeg * DEG; this.useMag = useMag;
    this.mapOpts = { ...MAP_HMM };
    this.zupt = this.live.zupt_strict ? "strict" : "profile";
    this.stillS = 0;
    this.f = new Filter(); this.f.setNoise(this.cfg.arw, this.cfg.brw, this.cfg.srw);
    this.f.setMapKeepSpeed(this.mapMode === "hmm" ? this.mapOpts.keep : (this.cfg.mm_keep_speed ? 1 : 0));
    this.hmm = null; this.hmmSteps = 0;
    this.sc = new SpeedCal(); this.aln = new Align(); this.ya = new YawAlign();
    this.handover = Math.trunc((this.cfg.handover_s || 0.0) * HZ);
    this.tLast = null; this.gf = null; this.gs = null; this.lvlG = null; this.lvlGm = null; this.mag = null;
    this.bin = [0, 0, 0, 0, 0, 0]; this.binN = 0; this.tStep = null;
    this.ringA = []; this.ringG = [];
    this.init = false; this.lat0 = null; this.lon0 = null;
    this.stepsNn = 0; this.sinceGnss = 0; this.drSteps = 0; this.rejects = 0;
    this.vnnRaw = 0.0; this.sig = 1.0; this.pStop = NaN; this.sumsig = 0.0; this.npush = 0;
    this.k = 1.0; this.c = 0.0; this.masked = false; this.snapped = false;
    this.basis = null; this.vEntry = 0.0; this.headState = null; this.pre = []; this.yaw = 0.0;
    this.headingSeed = null;
  }

  deadReckoning() { return this.masked || this.sinceGnss > GNSS_STALE_STEPS; }

  _forward() {
    if (this.ya.valid && this.basis !== null) return this.ya.forward(this.basis);
    return defaultForward(this.lvlG !== null ? this.lvlG : this.gf);
  }
  _en(lat, lon) {
    const k = (Math.PI / 180) * R_EARTH;
    return [(lon - this.lon0) * Math.cos(this.lat0 * DEG) * k, (lat - this.lat0) * k];
  }
  ll(e, n) {
    const k = (Math.PI / 180) * R_EARTH;
    return [this.lat0 + n / k, this.lon0 + e / (Math.cos(this.lat0 * DEG) * k)];
  }

  imu(t, ax, ay, az, gx, gy, gz, mx = NaN, my = NaN, mz = NaN) {
    const dt = this.tLast === null ? STEP_S : t - this.tLast;
    this.tLast = t;
    const a = [ax, ay, az], w = [gx, gy, gz];
    if (this.gf === null) { this.gf = a.slice(); this.gs = a.slice(); }
    else {
      const af = 1.0 - Math.exp(-dt / TAU_FAST), as = 1.0 - Math.exp(-dt / LEVEL_TAU_S);
      for (let i = 0; i < 3; i++) { this.gf[i] += af * (a[i] - this.gf[i]); this.gs[i] += as * (a[i] - this.gs[i]); }
    }
    if (mx === mx) {
      if (this.mag === null) this.mag = [mx, my, mz];
      else {
        const am = 1.0 - Math.exp(-dt / TAU_MAG);
        this.mag[0] += am * (mx - this.mag[0]); this.mag[1] += am * (my - this.mag[1]); this.mag[2] += am * (mz - this.mag[2]);
      }
    }
    if (this.init) {
      const v = this.yawMode === "coord" ? this.f.x[3] : 0.0;
      const fwd = this.yawMode === "coord" ? this._forward() : null;
      const yaw = yawRate(this.yawMode, w, this.gf, this.gs, fwd, v);
      this.f.predict(dt, yaw);
      this.yaw = yaw;
    }
    const b = this.bin;
    b[0] += ax; b[1] += ay; b[2] += az; b[3] += gx; b[4] += gy; b[5] += gz; this.binN += 1;
    if (this.tStep === null) this.tStep = t - STEP_S;
    if (t - this.tStep >= STEP_S - 1e-3 - 0.25 * Math.min(dt, STEP_S)) {
      const m = b.map((x) => x / this.binN);
      this.bin = [0, 0, 0, 0, 0, 0]; this.binN = 0;
      this._step10(t, t - this.tStep, m.slice(0, 3), m.slice(3), a, w);
      this.tStep = t;
    }
  }

  _step10(t, dt, a, w, nnA, nnW) {
    this.aln.update(a[0], a[1], a[2], w[0], w[1], w[2], dt);
    const remount = this.aln.get()[3];
    if (this.lvlG === null) { this.lvlG = a.slice(); this.lvlGm = a.slice(); }
    else {
      if (remount) { this.lvlG = this.lvlGm.slice(); this.ya.reset(); }
      const ae = 1.0 - Math.exp(-1.0 / (LEVEL_TAU_S * HZ)), am = 1.0 - Math.exp(-1.0 / (TAU_MED_S * HZ));
      for (let i = 0; i < 3; i++) { this.lvlG[i] += ae * (a[i] - this.lvlG[i]); this.lvlGm[i] += am * (a[i] - this.lvlGm[i]); }
    }
    const R = leveledBasis(this.lvlG); this.basis = R;
    let al = mat3vec(R, a), wl = mat3vec(R, w); wl[2] *= YAW_SIGN;
    this.ya.addAcc(al[0], al[1]);
    al = mat3vec(R, nnA); wl = mat3vec(R, nnW); wl[2] *= YAW_SIGN;
    this.ringA.push(al); this.ringG.push(wl);
    if (this.ringA.length > this.net.win) { this.ringA.shift(); this.ringG.shift(); }
    if (!this.init) return;
    const f = this.f, dr = this.deadReckoning();
    this.drSteps = dr ? this.drSteps + 1 : 0;
    if (this.drSteps === 1) {
      this.vEntry = f.x[3];
      if (this.head) this.headState = this.head.start({ v0: this.vEntry, k: this.k, c: this.c, pre: this.pre.slice(-120) });
    }
    this.stepsNn += 1;
    if (this.stepsNn >= NN_EVERY && this.ringA.length >= this.net.win) {
      this.stepsNn = 0;
      [this.vnnRaw, this.sig, this.pStop] = this.net.predict(this.ringA, this.ringG, t);
      const g1 = this.ringG.slice(-NN_EVERY);
      const still = this.vnnRaw < STOP_V && Math.max(...g1.map(norm3)) < STOP_GYRO;
      this.stillS = still ? this.stillS + 1 : 0;
      if (dr && this.zupt === "strict" && this.stillS >= STOP_S) {
        f.updateZupt(g1.reduce((s, g) => s + g[2], 0) / g1.length);
      } else if (dr && this.drSpeed) {
        if (this.head) {
          const [v, s] = this.head.step(this.headState, { vnn: this.vnnRaw, sig: this.sig, tau: this.drSteps / HZ,
                                                         acc: this.ringA.slice(-NN_EVERY), gyro: this.ringG.slice(-NN_EVERY) });
          f.updateSpeed(v, s);
        } else if (this.drSteps > this.handover) {
          const vused = calApply(this.vnnRaw, this.k, this.c), zv = this.cfg.zupt_v;
          if (zv !== null && zv !== undefined && vused < zv) f.updateZupt(this.yaw);
          else f.updateSpeed(vused, this.sig * this.cfg.sig_scale);
        }
      }
    }
    if (this.cfg.curv && dr) {
      const psidot = this.yaw - f.x[4];
      if (Math.abs(psidot) > CURV_GATE) {
        const left = cross(R[2], this._forward());
        f.updateCurvature(a[0] * left[0] + a[1] * left[1] + a[2] * left[2], psidot, this.cfg.curv_sigma);
      }
    }
    this.sinceGnss += 1;
    this.snapped = false;
    if (this.hmm) this._mapStepHmm(dt, dr);
  }

  _posSigma() { const cv = this.f.cov(); return Math.sqrt(Math.max(cv[0] + cv[1], 0.0) / 2); }

  _mapStepHmm(dt, dr) {
    const st = this.f.state(), o = this.mapOpts;
    this.hmm.addTravel(Math.abs(st[3]) * dt);
    this.hmmSteps += 1;
    if (this.hmmSteps % NN_EVERY) return;
    const ps = this._posSigma();
    const m = this.hmm.update([st[0], st[1]], st[2], Math.abs(st[3]), ps);
    if (m === null || !dr || m.conf < o.conf || Math.abs(st[3]) < 1.0) return;
    if (o.sigmaMax !== null && ps > o.sigmaMax) return;
    const brg = m.bearing, nE = -Math.cos(brg), nN = Math.sin(brg);
    const cross2 = (m.foot[0] - st[0]) * nE + (m.foot[1] - st[1]) * nN;
    const sc = Math.max(1.0, o.crossScale * m.halfWidth), cv = this.f.cov();
    if (o.chi2 !== null && (cross2 * cross2) / (nE * nE * cv[0] + nN * nN * cv[1] + sc * sc) > o.chi2) return;
    this.f.updateCrosstrack(nE, nN, cross2, sc);
    if (o.heading && m.segLen >= 25.0 && m.endDist >= 8.0) {
      const sh = o.headingSigmaDeg * DEG, dpsi = wrapPy(brg - st[2]);
      if (o.chi2 === null || (dpsi * dpsi) / (this.f.cov()[2] + sh * sh) <= o.chi2) this.f.updateHeading(brg, sh);
    }
    this.snapped = true;
  }

  gnss(t, lat, lon, speed = NaN, bearingDeg = NaN, cn0 = 40.0, sv = 10, navic = 0, masked = false) {
    this.masked = !!masked;
    if (this.lat0 === null) {
      this.lat0 = lat; this.lon0 = lon;
      if (this.roads && this.mapMode === "hmm") {
        this.hmm = new HMMMatcher(new RoadGraph(this.roads, lat, lon, this.mapOpts.exclude), { beta: this.mapOpts.beta });
      }
    }
    const [eg, ng] = this._en(lat, lon);
    const hasV = speed === speed, hasB = bearingDeg === bearingDeg;
    const brg = hasB ? bearingDeg * DEG : 0.0, f = this.f;
    if (!this.init) {
      if (masked) return;
      const v0 = hasV ? speed : 0.0;
      if (hasB && (!hasV || speed >= 1.0)) { f.init(eg, ng, brg, v0); this.headingSeed = "gnss"; }
      else {
        let h = null;
        if (this.useMag && this.mag !== null && this.gs !== null) h = magHeading(this.mag, this.gs, this._forward(), this.decl);
        if (h !== null) { f.init(eg, ng, h, v0); f.setHeadingSigma(MAG_SIGMA); this.headingSeed = "magnetometer"; }
        else { f.init(eg, ng, 0.0, v0); f.setHeadingSigma(UNKNOWN_SIGMA); this.headingSeed = "unknown"; }
      }
      this.init = true; this.sinceGnss = 0;
      return;
    }
    if (masked) return;
    const st = f.state(), cov = f.cov(), de = eg - st[0], dn = ng - st[1];
    const chi2 = (de * de) / (cov[0] + GNSS_BASE ** 2) + (dn * dn) / (cov[1] + GNSS_BASE ** 2);
    let trust = gqTrust(cn0, sv, navic, NOMINAL_DOP, chi2);
    let spoof = gqSpoof(cn0, sv, chi2, 0.0);
    if (spoof) {
      this.rejects += 1;
      if (this.rejects >= REACQ_FIXES) {
        f.init(eg, ng, hasB ? brg : st[2], hasV ? speed : st[3]);
        this.rejects = 0; spoof = 0;
        trust = gqTrust(cn0, sv, navic, NOMINAL_DOP, 0.0);
      }
    } else this.rejects = 0;
    if (spoof) return;
    if (trust >= TRUST_APPLIED) this.sinceGnss = 0;
    f.updateGnssPos(eg, ng, Math.sqrt(gqR(trust)));
    if (hasV && hasB) {
      const svs = Math.min(Math.max(DOPPLER_SIGMA / Math.max(trust, 3e-3), 0.1), 50.0);
      const sps = Math.min(Math.max((3 * DEG) / Math.max(trust, 3e-3), 1 * DEG), 90 * DEG);
      f.updateGnssVel(speed, brg, svs, sps);
    } else if (hasV) f.updateSpeed(speed, Math.min(Math.max(DOPPLER_SIGMA / Math.max(trust, 3e-3), 0.1), 50.0));
    if (hasV && trust >= TRUST_APPLIED) {
      this.ya.onFix(t, speed);
      this.pre.push([speed, this.vnnRaw]);
      if (this.pre.length > 600) this.pre.shift();
      if (this.vnnRaw > 0) {
        this.sumsig += this.sig * this.sig; this.npush += 1;
        this.sc.setLambda(DOPPLER_SIGMA ** 2 / Math.max(this.sumsig / this.npush, 1e-9));
        this.sc.push(this.vnnRaw, speed, trust);
        [this.k, this.c] = this.sc.fit();
      }
    }
  }
}

export const STAGES = ["physics", "speednet", "head", "map"];
export const LABELS = { physics: "Gyro heading + entry speed (no AI)", speednet: "+ SpeedNet (AI speed)",
                        head: "+ learned fusion head", map: "+ HMM road matching (shipped)" };

/** One engine per stage, as py/ablation.factory builds them. */
export function stageEngine(stage, { profile, net, head, roads, vehicle = "car" }) {
  const base = { profile, net, vehicle };
  switch (stage) {
    case "physics": return new EdgeEngine({ ...base, head: null, drSpeed: false });
    case "speednet": return new EdgeEngine({ ...base, head: null });
    case "head": return new EdgeEngine({ ...base, head });
    case "map": return new EdgeEngine({ ...base, head, roads, mapMode: "hmm" });
    default: throw new Error(`unknown stage ${stage}`);
  }
}

/**
 * Interleave IMU and GNSS by time (a fix at the same instant as a sample goes after it),
 * as edge_engine.run. imu: {t, acc: [[x,y,z]], gyro, mag?}; gn: {t, lat, lon, speed, bearing,
 * cn0, sv, navic, masked}; outages: [[t0, t1]] masked spans. every: record 1 of every N samples.
 * Returns rows [t, e, n, psi, v, P_ee, P_nn, dead_reckoning, snapped, ok, p_stop].
 */
export function run(eng, imu, gn, outages = [], { every = 1, onProgress = null } = {}) {
  const N = imu.t.length, ng = gn.t.length, out = [];
  let j = 0;
  for (let i = 0; i < N; i++) {
    const t = imu.t[i], A = imu.acc[i], Gy = imu.gyro[i], M = imu.mag ? imu.mag[i] : [NaN, NaN, NaN];
    eng.imu(t, A[0], A[1], A[2], Gy[0], Gy[1], Gy[2], M[0], M[1], M[2]);
    while (j < ng && gn.t[j] <= t) {
      const tj = gn.t[j];
      const msk = !!gn.masked[j] || outages.some(([a, b]) => a <= tj && tj < b);
      eng.gnss(tj, gn.lat[j], gn.lon[j], gn.speed[j], gn.bearing[j], gn.cn0[j], gn.sv[j], gn.navic[j], msk);
      j += 1;
    }
    if (i % every === 0 || i === N - 1) {
      if (eng.init) {
        const x = eng.f.x, P = eng.f.P;
        out.push([t, x[0], x[1], x[2], x[3], P[0][0], P[1][1], eng.deadReckoning() ? 1 : 0, eng.snapped ? 1 : 0, 1, eng.pStop]);
      } else out.push([t, NaN, NaN, NaN, NaN, NaN, NaN, 1, 0, 0, eng.pStop]);
    }
    if (onProgress && i % 20000 === 0) onProgress(i / N);
  }
  return out;
}
