// Web worker: parses a drive recording and runs the four stages on one GNSS outage, off the
// page's main thread. Messages in: {type: "load", files, units} | {type: "run", id, outage,
// vehicle, roads}. Out: {type: "loaded" | "progress" | "result" | "error", ...}.

import { SpeedNet } from "./speednet.js";
import { FusionHead } from "./head.js";
import { stageEngine, run, SharedNet, STAGES, LABELS } from "./engine.js";
import { readInputs, isGnss, isImu } from "./csv.js";

const PRE_S = 300, POST_S = 20, SHOW_PRE_S = 60, R = 6371000.0, DEG = Math.PI / 180;
let models = null, drive = null;

async function loadModels() {
  if (models) return models;
  const url = (f) => new URL(`./model/${f}`, import.meta.url);
  const [meta, bin, profile, head] = await Promise.all([
    fetch(url("speednet.json")).then((r) => r.json()), fetch(url("speednet.bin")).then((r) => r.arrayBuffer()),
    fetch(url("profile.json")).then((r) => r.json()), fetch(url("fusion_head.json")).then((r) => r.json())]);
  models = { net: new SpeedNet(meta, bin), profile, head: new FusionHead(head) };
  return models;
}

const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : NaN; };
const lower = (a, x) => { let lo = 0, hi = a.length; while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < x) lo = m + 1; else hi = m; } return lo; };
const thin = (n, max) => Math.max(1, Math.ceil(n / max));

async function load({ files, units }) {
  const texts = await Promise.all(files.map((f) => f.text()));
  let imuText = null, gnssText = null;
  texts.forEach((t) => { if (isImu(t) && isGnss(t)) { imuText = gnssText = t; } else if (isImu(t)) imuText = t; else if (isGnss(t)) gnssText = t; });
  if (!imuText) throw new Error("no IMU columns (ax, ay, az, gx, gy, gz) in the files");
  if (!gnssText) throw new Error("no GNSS columns (lat, lon) in the files");
  let u = { ...units };
  if (u.auto) {                                            // guess units from magnitudes
    const probe = readInputs(imuText.slice(0, 2e6).replace(/\n[^\n]*$/, ""), gnssText === imuText ? imuText.slice(0, 2e6).replace(/\n[^\n]*$/, "") : gnssText, {});
    const an = median(probe.imu.acc.map((a) => Math.hypot(...a)));             // ~9.8 in m/s^2, ~1 in g
    const gs = probe.imu.gyro.map((g) => Math.hypot(...g)).sort((x, y) => x - y);
    const g98 = gs[Math.floor(0.98 * (gs.length - 1))];                           // turning: ~0.2 rad/s, ~10 deg/s
    u = { accG: an < 3, gyroDeg: g98 > 5, kmh: !!units.kmh };
  }
  const { imu, gn } = readInputs(imuText, gnssText, u);
  if (imu.t.length < 100 || gn.t.length < 30) throw new Error("the recording is too short (needs IMU samples and at least 30 GNSS fixes)");
  const T = imu.t[imu.t.length - 1], hz = (imu.t.length - 1) / (T - imu.t[0]);
  drive = { imu, gn, hz };
  const k = thin(gn.t.length, 1500), idx = gn.t.map((_, i) => i).filter((i) => i % k === 0);
  const masked = [];                                        // Outage Simulator spans recorded by the app
  for (let i = 0; i < gn.t.length; i++) if (gn.masked[i]) {
    if (masked.length && i > 0 && gn.masked[i - 1]) masked[masked.length - 1][1] = gn.t[i]; else masked.push([gn.t[i], gn.t[i]]);
  }
  postMessage({ type: "loaded", units: u, duration: T, hz, fixes: gn.t.length,
                t: idx.map((i) => gn.t[i]), speed: idx.map((i) => gn.speed[i]), lat: idx.map((i) => gn.lat[i]), lon: idx.map((i) => gn.lon[i]),
                masked: masked.filter(([a, b]) => b - a >= 5) });
}

async function runOutage({ id, outage: [a, b], vehicle, roads }) {
  const { net, profile, head } = await loadModels();
  const { imu, gn, hz } = drive;
  const i0 = lower(imu.t, a - PRE_S), i1 = lower(imu.t, b + POST_S);
  const j0 = lower(gn.t, a - PRE_S), j1 = lower(gn.t, b + POST_S);
  const cut = (o, lo, hi) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, v && v.slice ? v.slice(lo, hi) : v]));
  const im = cut(imu, i0, i1), g = cut(gn, j0, j1);
  g.masked = g.masked.map(() => 0);                           // the chosen outage is the only one
  const stages = roads && roads.length ? STAGES : STAGES.filter((s) => s !== "map");
  const every = Math.max(1, Math.round(hz / 10)), shared = new SharedNet(net), res = {};
  let lat0 = null, lon0 = null;
  for (let si = 0; si < stages.length; si++) {
    const s = stages[si];
    const eng = stageEngine(s, { profile, net: shared, head, roads, vehicle });
    const rows = run(eng, im, g, [[a, b]], { every, onProgress: (p) => postMessage({ type: "progress", id, p: (si + p) / stages.length, stage: LABELS[s] }) });
    lat0 = eng.lat0; lon0 = eng.lon0;
    res[s] = rows.filter((r) => r[9] > 0 && r[0] >= a - SHOW_PRE_S);
  }
  const k = (Math.PI / 180) * R, cl = Math.cos(lat0 * DEG);
  const truth = g.t.map((t, i) => [t, (g.lon[i] - lon0) * cl * k, (g.lat[i] - lat0) * k]).filter((r) => r[0] >= a - SHOW_PRE_S);
  const inOut = truth.filter((r) => r[0] >= a && r[0] <= b);
  const at = (t) => {                                        // truth interpolated from the fixes
    const j = lower(truth.map((r) => r[0]), t);
    if (j <= 0 || j >= truth.length) return null;
    const p = truth[j - 1], q = truth[j], w = (t - p[0]) / (q[0] - p[0]);
    return q[0] - p[0] > 3 ? null : [p[1] + w * (q[1] - p[1]), p[2] + w * (q[2] - p[2])];
  };
  let dist = 0;
  const path = [at(a), ...inOut.map((r) => [r[1], r[2]]), at(b)].filter(Boolean);
  for (let i = 1; i < path.length; i++) dist += Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]);
  const out = { type: "result", id, outage: [a, b], dist, truth, stages: {}, gaps: inOut.length < (b - a) * 0.5, lat0, lon0 };
  for (const s of stages) {
    const rows = res[s], end = rows.filter((r) => r[0] < b).pop();
    const tp = end ? at(end[0]) : null;
    const err = end && tp ? Math.hypot(end[1] - tp[0], end[2] - tp[1]) : NaN;
    const kk = thin(rows.length, 2500);
    out.stages[s] = { label: LABELS[s], err, drift: dist > 0 ? (100 * err) / dist : NaN,
                      track: rows.filter((_, i) => i % kk === 0 || i === rows.length - 1)
                        .map((r) => [r[0], r[1], r[2], r[3], Math.sqrt(Math.max(r[5] + r[6], 0) / 2), r[8]]) };
  }
  postMessage(out);
}

onmessage = async (e) => {
  try {
    if (e.data.type === "load") await load(e.data);
    else if (e.data.type === "run") await runOutage(e.data);
  } catch (err) {
    postMessage({ type: "error", id: e.data.id, message: err.message || String(err) });
  }
};
