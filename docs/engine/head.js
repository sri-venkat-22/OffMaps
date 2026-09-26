// The learned fusion head (model/fusion_head.py, FusionHead.kt) in JavaScript: a small
// GRU ensemble that turns SpeedNet's speed into the speed measurement AND its sigma the
// filter fuses while dead-reckoning. Weights: fusion_head.json, the phone's own file.

const PRE_S = 120, DOPPLER_STOP = 0.5;
const f32 = Math.fround;
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const softplus = (x) => (x > 20 ? x : Math.log1p(Math.exp(x)));
const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
const std = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((x) => (x - m) * (x - m)))); };

export function imuStats(acc, gyro) {          // rotation-invariant stats of one second (10 rows)
  const an = acc.map((a) => Math.hypot(a[0], a[1], a[2])), ah = acc.map((a) => Math.hypot(a[0], a[1]));
  const az = acc.map((a) => a[2]);
  let jerk = 0;
  if (acc.length > 1) {
    const d = [];
    for (let i = 1; i < acc.length; i++) d.push(Math.hypot(acc[i][0] - acc[i - 1][0], acc[i][1] - acc[i - 1][1], acc[i][2] - acc[i - 1][2]));
    jerk = mean(d) * 10.0;
  }
  return [std(an), mean(ah), mean(az) - 9.81, std(az), mean(gyro.map((g) => Math.hypot(g[0], g[1], g[2]))),
          mean(gyro.map((g) => Math.abs(g[2]))), jerk];
}

export function context(v0, k, c, pre, kMin, kMax) {
  const calOk = kMin <= k && k <= kMax;
  const kk = calOk ? k : 1.0, cc = calOk ? c : 0.0;
  const p = pre.slice(-PRE_S);
  let vbar, stop, rm, rs;
  if (p.length) {
    const dop = p.map((x) => x[0]), res = p.map((x) => x[0] - Math.max(kk * x[1] + cc, 0.0));
    vbar = mean(dop); stop = mean(dop.map((d) => (d < DOPPLER_STOP ? 1 : 0))); rm = mean(res); rs = std(res);
  } else { vbar = v0; stop = 0.0; rm = 0.0; rs = 3.0; }
  return [v0 / 10, vbar / 10, stop, rm / 5, rs / 5, calOk ? 1 : 0, kk, cc / 5, Math.min(p.length, PRE_S) / PRE_S].map(f32);
}

export function stepFeatures(vnn, sig, tau, s, k, c, kMin, kMax) {
  const calOk = kMin <= k && k <= kMax;
  const vcal = calOk ? Math.max(k * vnn + c, 0.0) : vnn;
  return [vnn / 10, vcal / 10, Math.log(Math.max(sig, 1e-3)), Math.min(tau, 180) / 60, Math.min(tau, 10) / 10,
          s[0], s[1] / 3, s[2] / 3, s[3], s[4] * 5, s[5] * 5, s[6] / 10].map(f32);
}

class Member {
  constructor(m, hidden) {
    const flat = (a) => Float64Array.from(a.flat(Infinity));
    this.H = hidden;
    this.cW = flat(m["ctx.0.weight"]); this.cB = flat(m["ctx.0.bias"]);
    this.wih = flat(m["gru.weight_ih_l0"]); this.whh = flat(m["gru.weight_hh_l0"]);
    this.bih = flat(m["gru.bias_ih_l0"]); this.bhh = flat(m["gru.bias_hh_l0"]);
    this.oW = flat(m["out.weight"]); this.oB = flat(m["out.bias"]);
    this.nIn = m["gru.weight_ih_l0"][0].length; this.nCtx = m["ctx.0.weight"][0].length;
  }
  h0(ctx) {
    const H = this.H, h = new Float64Array(H);
    for (let r = 0; r < H; r++) { let s = this.cB[r]; for (let c = 0; c < this.nCtx; c++) s += this.cW[r * this.nCtx + c] * ctx[c]; h[r] = Math.tanh(s); }
    return h;
  }
  step(h, z) {                                 // one GRU step on z = [x, ctx] -> [v, var, h']
    const H = this.H, n = this.nIn, gi = new Float64Array(3 * H), gh = new Float64Array(3 * H);
    for (let r = 0; r < 3 * H; r++) {
      let a = this.bih[r], b = this.bhh[r];
      for (let c = 0; c < n; c++) a += this.wih[r * n + c] * z[c];
      for (let c = 0; c < H; c++) b += this.whh[r * H + c] * h[c];
      gi[r] = a; gh[r] = b;
    }
    const nh = new Float64Array(H);
    for (let k = 0; k < H; k++) {
      const r = sigmoid(gi[k] + gh[k]), zz = sigmoid(gi[H + k] + gh[H + k]);
      const nn = Math.tanh(gi[2 * H + k] + r * gh[2 * H + k]);
      nh[k] = (1 - zz) * nn + zz * h[k];
    }
    let o0 = this.oB[0], o1 = this.oB[1];
    for (let c = 0; c < H; c++) { o0 += this.oW[c] * nh[c]; o1 += this.oW[H + c] * nh[c]; }
    return [softplus(o0 * 5.0), Math.exp(Math.min(Math.max(o1, -6), 6)), nh];
  }
}

export class FusionHead {
  constructor(doc) {
    this.members = doc.members.map((m) => new Member(m, doc.hidden));
    this.kMin = doc.k_min; this.kMax = doc.k_max;
  }
  /** ctx: {v0, k, c, pre: [[doppler, nn_raw], ...]} frozen at outage start. */
  start(ctx) {
    const c = context(ctx.v0, ctx.k, ctx.c, ctx.pre, this.kMin, this.kMax);
    return { ctx: c, h: this.members.map((m) => m.h0(c)), k: ctx.k, c: ctx.c };
  }
  /** f: {vnn, sig, tau, acc (10 rows), gyro (10 rows)} -> [v, sigma]. */
  step(st, f) {
    const x = stepFeatures(f.vnn, f.sig, f.tau, imuStats(f.acc, f.gyro), st.k, st.c, this.kMin, this.kMax);
    const z = x.concat(st.ctx), vs = [], vars = [];
    this.members.forEach((m, i) => { const [v, vr, h] = m.step(st.h[i], z); st.h[i] = h; vs.push(v); vars.push(vr); });
    const v = mean(vs), vr = mean(vars) + mean(vs.map((x2) => (x2 - v) * (x2 - v)));
    return [v, Math.sqrt(vr)];
  }
}
