// The C++ core (core/*.cpp) in JavaScript, line for line: the planar 5-state ESKF,
// the Doppler speed self-calibration, the mount/re-mount detector and GNSS trust.
// Checked against the native library by py/tests/test_web_engine.py.

export const DEG = Math.PI / 180;
const N = 5, E = 0, PN = 1, PSI = 2, V = 3, BG = 4;
const CURV_MIN_RATE = 10 * DEG;

export function wrap(a) {                        // -> (-pi, pi]  (eskf.cpp)
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a <= -Math.PI) a += 2 * Math.PI;
  return a;
}

const zeros = () => Array.from({ length: N }, () => new Float64Array(N));

// ---------------------------------------------------------------- eskf.cpp
export class Filter {
  constructor() {
    this.x = new Float64Array(N);
    this.P = zeros();
    this.P[E][E] = this.P[PN][PN] = 1.0;
    this.P[PSI][PSI] = (1 * DEG) * (1 * DEG);
    this.P[V][V] = 0.5 * 0.5;
    this.P[BG][BG] = (0.05 * DEG) * (0.05 * DEG);
    this.mapKeep = 0;
    this.setNoise(0.3 * DEG, 0.01 * DEG, 0.7);
  }
  setNoise(arw, brw, srw) { this.qPsi = arw * arw; this.qBg = brw * brw; this.qV = srw * srw; }
  setMapKeepSpeed(keep) { this.mapKeep = keep & 3; }

  _update(H, innov, R) {
    const P = this.P, x = this.x, PHt = new Float64Array(N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) PHt[i] += P[i][j] * H[j];
    let S = R;
    for (let j = 0; j < N; j++) S += H[j] * PHt[j];
    const K = new Float64Array(N);
    for (let i = 0; i < N; i++) K[i] = PHt[i] / S;
    for (let i = 0; i < N; i++) x[i] += K[i] * innov;
    const HP = new Float64Array(N);
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) HP[j] += H[i] * P[i][j];
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) P[i][j] -= K[i] * HP[j];
  }

  _updateSkip(H, innov, R, skip) {               // gain rows in `skip` forced to 0, Joseph form
    const P = this.P, x = this.x, PHt = new Float64Array(N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) PHt[i] += P[i][j] * H[j];
    let S = R;
    for (let j = 0; j < N; j++) S += H[j] * PHt[j];
    const K = new Float64Array(N);
    for (let i = 0; i < N; i++) K[i] = PHt[i] / S;
    for (let i = 0; i < N; i++) if ((skip >> i) & 1) K[i] = 0.0;
    for (let i = 0; i < N; i++) x[i] += K[i] * innov;
    const A = zeros(), AP = zeros();
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) A[i][j] = (i === j ? 1.0 : 0.0) - K[i] * H[j];
    for (let i = 0; i < N; i++) for (let k = 0; k < N; k++) for (let j = 0; j < N; j++) AP[i][j] += A[i][k] * P[k][j];
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      let s = K[i] * K[j] * R;
      for (let k = 0; k < N; k++) s += AP[i][k] * A[j][k];
      P[i][j] = s;
    }
  }

  _updateMap(H, innov, R) {
    const skip = ((this.mapKeep & 1) ? 1 << V : 0) | ((this.mapKeep & 2) ? 1 << BG : 0);
    if (skip) this._updateSkip(H, innov, R, skip); else this._update(H, innov, R);
  }

  init(e, n, psi, v) { const x = this.x; x[E] = e; x[PN] = n; x[PSI] = psi; x[V] = v; x[BG] = 0.0; }

  predict(dt, gyroZ) {
    const x = this.x, P = this.P;
    const psi = x[PSI], v = x[V], s = Math.sin(psi), c = Math.cos(psi);
    x[E] += v * s * dt;
    x[PN] += v * c * dt;
    x[PSI] = wrap(psi + (gyroZ - x[BG]) * dt);
    const F = zeros();
    for (let i = 0; i < N; i++) F[i][i] = 1.0;
    F[E][PSI] = v * c * dt; F[E][V] = s * dt;
    F[PN][PSI] = -v * s * dt; F[PN][V] = c * dt;
    F[PSI][BG] = -dt;
    const FP = zeros();
    for (let i = 0; i < N; i++) for (let k = 0; k < N; k++) {
      let a = 0; for (let j = 0; j < N; j++) a += F[i][j] * P[j][k]; FP[i][k] = a;
    }
    for (let i = 0; i < N; i++) for (let k = 0; k < N; k++) {
      let a = 0; for (let j = 0; j < N; j++) a += FP[i][j] * F[k][j]; P[i][k] = a;
    }
    P[PSI][PSI] += this.qPsi * dt;
    P[V][V] += this.qV * dt;
    P[BG][BG] += this.qBg * dt;
  }

  updateSpeed(vMeas, sigma) { const H = [0, 0, 0, 1, 0]; this._update(H, vMeas - this.x[V], sigma * sigma); }
  updateZupt(gyroZ) {
    this._update([0, 0, 0, 1, 0], 0.0 - this.x[V], 0.02 * 0.02);
    this._update([0, 0, 0, 0, 1], gyroZ - this.x[BG], (0.02 * DEG) * (0.02 * DEG));
  }
  updateCurvature(aLat, psidot, baseSigma) {
    if (Math.abs(psidot) < CURV_MIN_RATE) return;
    const sigma = baseSigma * CURV_MIN_RATE / Math.abs(psidot);
    this._update([0, 0, 0, 1, 0], aLat / psidot - this.x[V], sigma * sigma);
  }
  updateGnssPos(e, n, sigma) {
    this._update([1, 0, 0, 0, 0], e - this.x[E], sigma * sigma);
    this._update([0, 1, 0, 0, 0], n - this.x[PN], sigma * sigma);
  }
  updateGnssVel(vMeas, bearing, sv, sp) {
    this._update([0, 0, 0, 1, 0], vMeas - this.x[V], sv * sv);
    this._update([0, 0, 1, 0, 0], wrap(bearing - this.x[PSI]), sp * sp);
  }
  updateCrosstrack(ne, nn, crossInnov, sigma) { this._updateMap([ne, nn, 0, 0, 0], crossInnov, sigma * sigma); }
  updateHeading(bearing, sigma) { this._updateMap([0, 0, 1, 0, 0], wrap(bearing - this.x[PSI]), sigma * sigma); }
  setHeadingSigma(sigma) {
    for (let i = 0; i < N; i++) { this.P[PSI][i] = 0.0; this.P[i][PSI] = 0.0; }
    this.P[PSI][PSI] = sigma * sigma;
  }
  state() { return Array.from(this.x); }
  cov() { return [this.P[E][E], this.P[PN][PN], this.P[PSI][PSI]]; }
}

// ---------------------------------------------------------------- speed_cal.cpp
const CAP = 600;
function demingK(Sxx, Syy, Sxy, lam) {
  if (!Number.isFinite(lam)) return Sxy / Sxx;
  if (Math.abs(Sxy) < 1e-12) return Sxy / Sxx;
  const t = Syy - lam * Sxx;
  return (t + Math.sqrt(t * t + 4.0 * lam * Sxy * Sxy)) / (2.0 * Sxy);
}

export class SpeedCal {
  constructor() {
    this.vn = new Float64Array(CAP); this.vd = new Float64Array(CAP); this.w = new Float64Array(CAP);
    this.n = 0; this.head = 0; this.k = 1.0; this.c = 0.0; this.lam = Infinity; this.excited = 0;
  }
  push(vNn, vDop, weight) {
    this.vn[this.head] = vNn; this.vd[this.head] = vDop; this.w[this.head] = weight;
    this.head = (this.head + 1) % CAP;
    if (this.n < CAP) this.n++;
  }
  setLambda(l) { this.lam = l; }
  fit() {
    if (this.n < 30) return [this.k, this.c, 0];
    let sw = 0, swx = 0, swy = 0, swxx = 0, swyy = 0, swxy = 0;
    for (let i = 0; i < this.n; i++) {
      const x = this.vn[i], y = this.vd[i], w = this.w[i];
      sw += w; swx += w * x; swy += w * y; swxx += w * x * x; swyy += w * y * y; swxy += w * x * y;
    }
    if (!(sw > 1e-12)) return [this.k, this.c, 0];
    const mean = swx / sw, vr = swxx / sw - mean * mean;
    this.excited = vr > 4.0 ? 1 : 0;
    if (!Number.isFinite(this.lam)) {
      if (this.excited) {
        const d = sw * swxx - swx * swx;
        this.k = (sw * swxy - swx * swy) / d; this.c = (swxx * swy - swx * swxy) / d;
      } else { this.k = swxy / swxx; this.c = 0.0; }
    } else if (this.excited) {
      const xb = swx / sw, yb = swy / sw;
      const sxx = swxx / sw - xb * xb, syy = swyy / sw - yb * yb, sxy = swxy / sw - xb * yb;
      this.k = demingK(sxx, syy, sxy, this.lam); this.c = yb - this.k * xb;
    } else { this.k = demingK(swxx, swyy, swxy, this.lam); this.c = 0.0; }
    return [this.k, this.c, this.excited];
  }
}

// ---------------------------------------------------------------- align.cpp
const REMOUNT_TAU_MED = 5.0, REMOUNT_TAU_SLOW = 30.0, REMOUNT_SLACK = 10.0 * DEG, REMOUNT_H = 200.0 * DEG;

export class Align {
  constructor() {
    this.g = [0, 0, 9.81]; this.gm = [0, 0, 9.81]; this.gs = [0, 0, 9.81];
    this.cusum = 0; this.Sxx = 0; this.Sxy = 0; this.Syy = 0;
    this.roll = 0; this.pitch = 0; this.yaw = 0; this.haveRef = 0; this.changed = 0;
  }
  update(ax, ay, az, gx, gy, gz, dt) {
    this.changed = 0;
    const beta = 1.0 - Math.exp(-dt / 0.5), g = this.g;
    g[0] += beta * (ax - g[0]); g[1] += beta * (ay - g[1]); g[2] += beta * (az - g[2]);
    const gn = Math.sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) + 1e-9;
    const u = [g[0] / gn, g[1] / gn, g[2] / gn];
    this.roll = Math.atan2(u[1], u[2]);
    this.pitch = Math.atan2(-u[0], Math.sqrt(u[1] * u[1] + u[2] * u[2]));
    const a3 = [ax, ay, az];
    if (!this.haveRef) { this.gm = a3.slice(); this.gs = a3.slice(); this.haveRef = 1; }
    const bm = 1.0 - Math.exp(-dt / REMOUNT_TAU_MED), bs = 1.0 - Math.exp(-dt / REMOUNT_TAU_SLOW);
    for (let i = 0; i < 3; i++) { this.gm[i] += bm * (a3[i] - this.gm[i]); this.gs[i] += bs * (a3[i] - this.gs[i]); }
    const gm = this.gm, gs = this.gs;
    const nm = Math.sqrt(gm[0] * gm[0] + gm[1] * gm[1] + gm[2] * gm[2]) + 1e-9;
    const ns = Math.sqrt(gs[0] * gs[0] + gs[1] * gs[1] + gs[2] * gs[2]) + 1e-9;
    const cosang = (gm[0] * gs[0] + gm[1] * gs[1] + gm[2] * gs[2]) / (nm * ns);
    const dev = Math.acos(cosang > 1 ? 1 : cosang < -1 ? -1 : cosang);
    this.cusum = Math.max(0.0, this.cusum + dev - REMOUNT_SLACK);
    if (this.cusum > REMOUNT_H) {
      this.gs = this.gm.slice(); this.cusum = 0; this.Sxx = this.Sxy = this.Syy = 0; this.changed = 1;
    }
    const ah0 = ax - u[0] * 9.81, ah1 = ay - u[1] * 9.81;
    const amag = Math.sqrt(ah0 * ah0 + ah1 * ah1);
    if (Math.abs(gz) < 5 * DEG && amag > 1.0) {
      const f = 0.98;
      this.Sxx = f * this.Sxx + ah0 * ah0; this.Sxy = f * this.Sxy + ah0 * ah1; this.Syy = f * this.Syy + ah1 * ah1;
      this.yaw = 0.5 * Math.atan2(2 * this.Sxy, this.Sxx - this.Syy);
    }
  }
  get() { return [this.roll, this.pitch, this.yaw, this.changed]; }
}

// ---------------------------------------------------------------- gnss_quality.cpp
const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);

export function gqTrust(cn0, sv, navic, dop, chi2) {
  const sCn0 = clamp01((cn0 - 17.0) / (32.0 - 17.0));
  const sSv = clamp01((sv - 4.0) / (8.0 - 4.0));
  const sDop = clamp01((6.0 - dop) / (6.0 - 1.0));
  const sInnov = Math.exp(-chi2 / 9.0);
  const nav = navic > 0 ? 1.0 + 0.05 * Math.min(navic, 4) : 1.0;
  return clamp01(sCn0 * sSv * sDop * sInnov * nav);
}
export function gqR(trust) { const s = 3.0 / Math.max(trust, 3e-3); return s * s; }
export function gqSpoof(cn0, sv, chi2, dopResid) {
  const strong = cn0 > 35.0 && sv >= 5;
  const rejects = chi2 > 25.0 || Math.abs(dopResid) > 5.0;
  return strong && rejects ? 1 : 0;
}
