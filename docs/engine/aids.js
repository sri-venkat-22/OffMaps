// py/heading_aids.py + model/mount.leveled_basis in JavaScript: the yaw rate about true
// vertical, the magnetometer heading seed, the GNSS-aided forward axis (YawAlign).

export const YAW_SIGN = -1.0;
export const G = 9.81;
export const MAG_SIGMA = (20 * Math.PI) / 180;
export const UNKNOWN_SIGMA = Math.PI;

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
function unit(a) {
  const n = Math.sqrt(dot(a, a));
  return n > 1e-9 ? [a[0] / n, a[1] / n, a[2] / n] : [0, 0, 0];
}
function horiz(a, u) { const d = dot(a, u); return unit([a[0] - d * u[0], a[1] - d * u[1], a[2] - d * u[2]]); }
const pyMod = (x, m) => ((x % m) + m) % m;
export const wrapPy = (a) => pyMod(a + Math.PI, 2 * Math.PI) - Math.PI;   // (a + pi) % 2pi - pi, Python's %

export function defaultForward(up) {
  const u = unit(up);
  return Math.abs(u[2]) >= 0.7 ? horiz([0, 1, 0], u) : horiz([0, 0, -1], u);
}

export function magHeading(mag, up, fwd, declination = 0.0) {
  const u = unit(up), north = horiz(mag, u);
  if (north[0] === 0 && north[1] === 0 && north[2] === 0) return null;
  const east = cross(north, u), f = horiz(fwd, u);
  return wrapPy(Math.atan2(dot(f, east), dot(f, north)) + declination);
}

export function yawRate(mode, gyro, upFast, upSlow, fwd = null, v = 0.0) {
  if (mode === "slow") return YAW_SIGN * dot(gyro, unit(upSlow));
  const z = unit(upFast);
  let r = YAW_SIGN * dot(gyro, z);
  if (mode === "fast" || fwd === null) return r;
  const x = horiz(fwd, z), y = cross(z, x);
  for (let i = 0; i < 3; i++) {
    const phi = Math.atan(v * r / G), c = Math.cos(phi), s = Math.sin(phi);
    r = YAW_SIGN * dot(gyro, [c * z[0] + s * y[0], c * z[1] + s * y[1], c * z[2] + s * y[2]]);
  }
  return r;
}

export class YawAlign {
  constructor() { this.reset(); }
  reset() {
    this.s11 = this.s12 = this.s22 = this.b1 = this.b2 = this.exc = 0.0;
    this.sum1 = this.sum2 = 0.0; this.cnt = 0; this.tPrev = null; this.vPrev = 0.0;
    this.theta = 0.0; this.valid = false;
  }
  addAcc(a1, a2) { this.sum1 += a1; this.sum2 += a2; this.cnt += 1; }
  onFix(t, v) {
    let upd = false;
    if (this.tPrev !== null && this.cnt > 0 && t - this.tPrev >= 0.5 && t - this.tPrev <= 2.5) {
      const dv = (v - this.vPrev) / (t - this.tPrev), a1 = this.sum1 / this.cnt, a2 = this.sum2 / this.cnt, f = 0.995;
      this.s11 = f * this.s11 + a1 * a1; this.s12 = f * this.s12 + a1 * a2; this.s22 = f * this.s22 + a2 * a2;
      this.b1 = f * this.b1 + a1 * dv; this.b2 = f * this.b2 + a2 * dv; this.exc = f * this.exc + dv * dv;
      const det = this.s11 * this.s22 - this.s12 * this.s12;
      if (det > 1e-9) {
        const c1 = (this.s22 * this.b1 - this.s12 * this.b2) / det, c2 = (this.s11 * this.b2 - this.s12 * this.b1) / det;
        const gain = Math.hypot(c1, c2);
        this.theta = Math.atan2(c2, c1);
        this.valid = this.exc >= 4.0 && gain >= 0.2 && gain <= 5.0;
        upd = true;
      }
    }
    this.tPrev = t; this.vPrev = v; this.sum1 = this.sum2 = 0.0; this.cnt = 0;
    return upd;
  }
  forward(basis) {
    const [h1, h2] = basis, c = Math.cos(this.theta), s = Math.sin(this.theta);
    return [c * h1[0] + s * h2[0], c * h1[1] + s * h2[1], c * h1[2] + s * h2[2]];
  }
}

/** Rows (h1, h2, u): a right-handed frame with u = unit up (model/mount.leveled_basis). */
export function leveledBasis(up) {
  const n = Math.max(Math.sqrt(dot(up, up)), 1e-6), u = [up[0] / n, up[1] / n, up[2] / n];
  const ref = Math.abs(u[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const d = dot(ref, u);
  let h1 = [ref[0] - u[0] * d, ref[1] - u[1] * d, ref[2] - u[2] * d];
  const hn = Math.sqrt(dot(h1, h1)); h1 = [h1[0] / hn, h1[1] / hn, h1[2] / hn];
  return [h1, cross(u, h1), u];
}

export const mat3vec = (R, a) => [dot(R[0], a), dot(R[1], a), dot(R[2], a)];
