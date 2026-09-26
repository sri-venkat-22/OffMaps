// SpeedNet in JavaScript: model/features.window_features (version 1) and the TCN + GRU
// forward pass of model/tcn.py, reading the weights web_export.py writes. No runtime,
// no WebAssembly: ~3 M multiply-adds per 1 s step. Returns what the phone's SpeedNet.kt
// returns: calibrated speed, sigma and p(stopped).

const C = 9, WIN = 20, HZ = 10.0;
const SCALE = [10, 10, 10, 1, 1, 1, 10, 1, 10];
const f32 = Math.fround;

/** (9, T) channel-major Float64Array of float32 values, from T rows of leveled acc / gyro. */
export function windowFeatures(acc, gyro) {
  const T = acc.length, out = new Float64Array(C * T);
  let p0 = f32(acc[0][0]), p1 = f32(acc[0][1]), p2 = f32(acc[0][2]);
  for (let t = 0; t < T; t++) {
    const ax = f32(acc[t][0]), ay = f32(acc[t][1]), az = f32(acc[t][2]);
    const gx = f32(gyro[t][0]), gy = f32(gyro[t][1]), gz = f32(gyro[t][2]);
    const an = f32(Math.sqrt(ax * ax + ay * ay + az * az)), gn = f32(Math.sqrt(gx * gx + gy * gy + gz * gz));
    const dx = f32(ax - p0), dy = f32(ay - p1), dz = f32(az - p2);
    const jerk = f32(f32(Math.sqrt(dx * dx + dy * dy + dz * dz)) * HZ);
    p0 = ax; p1 = ay; p2 = az;
    const ch = [ax, ay, az, gx, gy, gz, an, gn, jerk];
    for (let c = 0; c < C; c++) out[c * T + t] = f32(ch[c] / SCALE[c]);
  }
  return out;
}

const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const softplus = (x) => (x > 20 ? x : Math.log1p(Math.exp(x)));   // torch default (beta 1, threshold 20)

export function stopLogit(cls) {                // log-odds of "stopped" vs the rest (model/pstop.py)
  let m = -Infinity;
  for (let i = 1; i < cls.length; i++) m = Math.max(m, cls[i]);
  let s = 0;
  for (let i = 1; i < cls.length; i++) s += Math.exp(cls[i] - m);
  return cls[0] - (m + Math.log(s));
}

export class SpeedNet {
  /** meta: speednet.json; buf: ArrayBuffer of speednet.bin. */
  constructor(meta, buf) {
    const all = new Float32Array(buf);
    this.w = {};
    for (const t of meta.tensors) {
      const n = t.shape.reduce((a, b) => a * b, 1);
      this.w[t.name] = Float64Array.from(all.subarray(t.offset, t.offset + n));
    }
    this.ch = meta.channels; this.dil = meta.dilations; this.win = (meta.feat && meta.feat.win) || WIN;
    this.calib = meta.calib; this.pstop = meta.pstop; this.meta = meta;
  }

  /** Raw forward on a (9, T) feature tensor -> {mu, logvar, cls}. */
  forward(x) {
    const H = this.ch, T = x.length / C, w = this.w;
    let h = new Float64Array(H * T);
    const Wi = w["inp.weight"], bi = w["inp.bias"];
    for (let o = 0; o < H; o++) for (let t = 0; t < T; t++) {
      let s = bi[o];
      for (let c = 0; c < C; c++) s += Wi[o * C + c] * x[c * T + t];
      h[o * T + t] = s;
    }
    const conv = (inp, W, b, d, sc, sh) => {            // Conv1d(k=3, dilation d, pad d) -> ReLU -> BN
      const out = new Float64Array(H * T);
      for (let o = 0; o < H; o++) {
        const ob = o * T;
        for (let t = 0; t < T; t++) out[ob + t] = b[o];
        for (let c = 0; c < H; c++) {
          const wb = (o * H + c) * 3, ib = c * T, w0 = W[wb], w1 = W[wb + 1], w2 = W[wb + 2];
          for (let t = 0; t < T; t++) {
            let s = w1 * inp[ib + t];
            if (t - d >= 0) s += w0 * inp[ib + t - d];
            if (t + d < T) s += w2 * inp[ib + t + d];
            out[ob + t] += s;
          }
        }
        for (let t = 0; t < T; t++) { const v = out[ob + t]; out[ob + t] = (v > 0 ? v : 0) * sc[o] + sh[o]; }
      }
      return out;
    };
    for (let i = 0; i < this.dil.length; i++) {
      const d = this.dil[i], p = `tcn.${i}.`;
      const y1 = conv(h, w[p + "conv0.weight"], w[p + "conv0.bias"], d, w[p + "bn2.scale"], w[p + "bn2.shift"]);
      const y2 = conv(y1, w[p + "conv3.weight"], w[p + "conv3.bias"], d, w[p + "bn5.scale"], w[p + "bn5.shift"]);
      for (let k = 0; k < h.length; k++) h[k] += y2[k];
    }
    // GRU (torch gate order r, z, n), h0 = 0, over t = 0..T-1
    const Wih = w["gru.weight_ih_l0"], Whh = w["gru.weight_hh_l0"], bih = w["gru.bias_ih_l0"], bhh = w["gru.bias_hh_l0"];
    let hs = new Float64Array(H);
    const gi = new Float64Array(3 * H), gh = new Float64Array(3 * H);
    for (let t = 0; t < T; t++) {
      for (let r = 0; r < 3 * H; r++) {
        let a = bih[r], b = bhh[r];
        const rb = r * H;
        for (let c = 0; c < H; c++) { a += Wih[rb + c] * h[c * T + t]; b += Whh[rb + c] * hs[c]; }
        gi[r] = a; gh[r] = b;
      }
      const nh = new Float64Array(H);
      for (let k = 0; k < H; k++) {
        const r = sigmoid(gi[k] + gh[k]), z = sigmoid(gi[H + k] + gh[H + k]);
        const n = Math.tanh(gi[2 * H + k] + r * gh[2 * H + k]);
        nh[k] = (1 - z) * n + z * hs[k];
      }
      hs = nh;
    }
    const lin = (W, b, rows) => Array.from({ length: rows }, (_, r) => {
      let s = b[r]; for (let c = 0; c < H; c++) s += W[r * H + c] * hs[c]; return s;
    });
    const d = lin(w["disp.weight"], w["disp.bias"], 2);
    return { mu: softplus(d[0]), logvar: Math.min(Math.max(d[1], -8), 6), cls: lin(w["cls.weight"], w["cls.bias"], 4) };
  }

  /** acc, gyro: WIN leveled rows -> [v (m/s), sigma (m/s), p(stopped)] (SpeedNet.kt). */
  predict(acc, gyro) {
    const r = this.forward(windowFeatures(acc, gyro));
    const { a, b, s } = this.calib;
    const v = (r.mu - a) / b, sig = Math.max(Math.exp(0.5 * r.logvar) / b * s, 1e-6);
    const p = this.pstop ? sigmoid(this.pstop.a * stopLogit(r.cls) + this.pstop.b) : NaN;
    return [v, sig, p];
  }
}
