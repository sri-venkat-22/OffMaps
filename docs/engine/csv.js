// CSV front end, as edge_engine.read_inputs: the OffMaps app's recording (imu.csv + gnss.csv,
// DriveRecorder.kt) or any IMU + GNSS CSV pair, or one merged CSV with the GNSS columns
// filled only on fix rows. Column names are case-insensitive.
//   IMU : t | t_ns | time,  ax ay az (m/s^2),  gx gy gz (rad/s),  [mx my mz]
//   GNSS: t | t_ns | time,  lat lon,  [speed (m/s)] [bearing (deg)] [cn0_mean sv_used navic_sv] [masked]

function table(text) {
  const lines = text.split(/\r?\n/);
  const head = lines[0].split(",").map((h) => h.trim().toLowerCase());
  const cols = head.map(() => []);
  for (let i = 1; i < lines.length; i++) {
    const l = lines[i];
    if (!l) continue;
    const v = l.split(",");
    if (v.length < head.length) continue;                // a truncated last row (app killed mid-write)
    for (let c = 0; c < head.length; c++) cols[c].push(v[c] === "" ? NaN : Number(v[c]));
  }
  return { head, cols };
}

const col = (tab, names, required = true) => {
  for (const n of names) { const i = tab.head.indexOf(n); if (i >= 0) return tab.cols[i]; }
  if (required) throw new Error(`none of ${names.join(", ")} in columns ${tab.head.join(", ")}`);
  return null;
};

function times(tab) {
  const ns = col(tab, ["t_ns", "time_ns", "timestamp_ns"], false);
  return ns ? ns.map((x) => x * 1e-9) : col(tab, ["t", "time", "t_s", "timestamp"]);
}

const fill = (a, d) => a.map((x) => (Number.isFinite(x) ? x : d));   // an empty cell -> the default

export function isGnss(text) { const h = text.slice(0, 400).split(/\r?\n/)[0].toLowerCase(); return /(^|,)\s*(lat|latitude)\s*(,|$)/.test(h); }
export function isImu(text) { const h = text.slice(0, 400).split(/\r?\n/)[0].toLowerCase(); return /(^|,)\s*ax\s*(,|$)/.test(h); }

/** -> {imu, gn}: times in seconds from a shared origin. imuText and gnssText may be the same merged file. */
export function readInputs(imuText, gnssText, { gyroDeg = false, accG = false, kmh = false } = {}) {
  const ti = table(imuText);
  let tg = gnssText === imuText ? ti : table(gnssText);
  let gt = times(tg), lat = col(tg, ["lat", "latitude"]), lon = col(tg, ["lon", "longitude"]);
  let rows = gt.map((_, i) => i).filter((i) => Number.isFinite(lat[i]) && Number.isFinite(lon[i]));
  if (tg === ti) rows = rows.filter((i, k) => k === 0 || lat[i] !== lat[rows[k - 1]] || lon[i] !== lon[rows[k - 1]]);
  const pick = (a, d = NaN) => (a ? rows.map((i) => a[i]) : rows.map(() => d));
  const it = times(ti);
  // monotonic IMU clock (the recorder can repeat or reorder a sample)
  const keep = [];
  for (let i = 0; i < it.length; i++) if (Number.isFinite(it[i]) && (keep.length === 0 || it[i] > it[keep[keep.length - 1]])) keep.push(i);
  const t0 = rows.length ? Math.min(it[keep[0]], gt[rows[0]]) : it[keep[0]];
  const a = ["ax", "ay", "az"].map((n) => col(ti, [n])), g = ["gx", "gy", "gz"].map((n) => col(ti, [n]));
  const m = ["mx", "my", "mz"].map((n) => col(ti, [n], false));
  const ka = accG ? 9.80665 : 1.0, kg = gyroDeg ? Math.PI / 180 : 1.0;
  const imu = {
    t: keep.map((i) => it[i] - t0),
    acc: keep.map((i) => [a[0][i] * ka, a[1][i] * ka, a[2][i] * ka]),
    gyro: keep.map((i) => [g[0][i] * kg, g[1][i] * kg, g[2][i] * kg]),
    mag: m.every(Boolean) ? keep.map((i) => [m[0][i], m[1][i], m[2][i]]) : null,
  };
  const spd = col(tg, ["speed", "speed_mps"], false);
  const gn = {
    t: rows.map((i) => gt[i] - t0), lat: pick(lat), lon: pick(lon),
    speed: pick(spd).map((v) => v * (kmh ? 1 / 3.6 : 1.0)), bearing: pick(col(tg, ["bearing", "course"], false)),
    cn0: fill(pick(col(tg, ["cn0_mean", "cn0"], false), 40.0), 40.0), sv: fill(pick(col(tg, ["sv_used", "sv"], false), 10), 10),
    navic: fill(pick(col(tg, ["navic_sv"], false), 0), 0), masked: fill(pick(col(tg, ["masked"], false), 0), 0),
  };
  return { imu, gn };
}
