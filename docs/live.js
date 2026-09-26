// "Run it in your browser": the app's loop (docs/engine/, a JavaScript port checked against
// py/edge_engine.py) on a recording the visitor loads. Everything runs in a worker on this
// device; the only network call is the optional OpenStreetMap road fetch for the map stage.
"use strict";

const NS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);
const STAGE_VAR = { physics: "--s-physics", speednet: "--s-speednet", head: "--s-head", map: "--s-map" };
const ROAD_CLASSES = ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary",
  "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential", "living_street", "service"];
const MIN_MOVING = 5.0;                                    // m/s mean over the outage (the ablation's 5*D metres)
const st = { info: null, a: 0, D: 60, runs: [], busy: false, roads: new Map(), last: null, seed: 1 };
let worker = null;

const fmt = (s) => { s = Math.max(0, Math.round(s)); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(x).padStart(2, "0")}` : `${m}:${String(x).padStart(2, "0")}`; };
const css = (v) => `var(${v})`;
function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function mulberry32(a) { return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

function status(msg, err = false) { const s = $("livestatus"); s.textContent = msg; s.classList.toggle("err", err); }

function ensureWorker() {
  if (worker) return worker;
  worker = new Worker(new URL("engine/worker.js", import.meta.url), { type: "module" });
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "loaded") onLoaded(m);
    else if (m.type === "progress") progress(m.p, m.stage);
    else if (m.type === "result") onResult(m);
    else if (m.type === "error") { st.busy = false; progress(null); status(m.message, true); buttons(); }
  };
  worker.onerror = (e) => { st.busy = false; progress(null); status(`The engine stopped: ${e.message || "worker error"}`, true); buttons(); };
  return worker;
}

// ---------- loading ----------
function onFiles(files) {
  if (!files.length) return;
  st.info = null; st.runs = []; st.roads.clear(); $("livedrive").hidden = true; $("liveout").hidden = true;
  status(`Reading ${[...files].map((f) => f.name).join(" + ")}…`);
  ensureWorker().postMessage({ type: "load", files: [...files], units: { auto: true, kmh: $("kmh").checked } });
}

function onLoaded(m) {
  st.info = m;
  const u = [m.units.accG ? "accel in g" : "accel in m/s²", m.units.gyroDeg ? "gyro in °/s" : "gyro in rad/s"].join(", ");
  $("drivesum").textContent = `${fmt(m.duration)} of driving · IMU at ${Math.round(m.hz)} Hz · ${m.fixes.toLocaleString()} GNSS fixes · ${u}`
    + (m.masked.length ? ` · ${m.masked.length} outage${m.masked.length > 1 ? "s" : ""} simulated in the app` : "");
  $("livedrive").hidden = false;
  status("Pick where GNSS drops on the timeline, or let it pick a random spot.");
  const sim = m.masked[0];
  if (sim) { st.a = sim[0]; st.D = Math.round(sim[1] - sim[0]); }
  else { const v = valid(); st.a = v.length ? v[Math.floor(v.length / 2)] : Math.min(300, m.duration / 2); }
  segButtons(); drawTimeline(); buttons();
}

/** Outage starts (s) where the car keeps moving: mean GNSS speed >= MIN_MOVING over [a, a+D]. */
function valid(D = st.D) {
  const { t, speed, duration } = st.info, out = [];
  for (let a = 60; a + D < duration - 20; a += 5) {
    let s = 0, n = 0;
    for (let i = 0; i < t.length; i++) if (t[i] >= a && t[i] <= a + D && Number.isFinite(speed[i])) { s += speed[i]; n++; }
    if (n >= D / 4 && s / n >= MIN_MOVING) out.push(a);
  }
  return out;
}

// ---------- timeline ----------
function drawTimeline() {
  const c = $("timeline"), { t, speed, duration } = st.info, dpr = devicePixelRatio || 1;
  const W = c.clientWidth, H = c.clientHeight;
  c.width = W * dpr; c.height = H * dpr;
  const g = c.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  const vmax = Math.max(10, ...speed.filter(Number.isFinite)) * 1.1, x = (s) => (s / duration) * W, y = (v) => H - 14 - (v / vmax) * (H - 22);
  g.fillStyle = "#FEEFC3"; g.fillRect(x(st.a), 0, Math.max(2, x(st.a + st.D) - x(st.a)), H - 14);
  g.strokeStyle = "#1A73E8"; g.lineWidth = 1.5; g.beginPath();
  t.forEach((ti, i) => { if (Number.isFinite(speed[i])) (i ? g.lineTo : g.moveTo).call(g, x(ti), y(speed[i])); });
  g.stroke();
  g.fillStyle = "#5F6368"; g.font = "11px Inter, system-ui, sans-serif";
  g.fillText("0:00", 0, H - 2); const e = fmt(duration); g.fillText(e, W - g.measureText(e).width, H - 2);
  const lab = `no GNSS ${fmt(st.a)}–${fmt(st.a + st.D)}`;
  g.fillStyle = "#202124"; g.fillText(lab, Math.min(Math.max(0, x(st.a)), W - g.measureText(lab).width), 11);
}

function segButtons() {
  document.querySelectorAll("#livedur button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.d === st.D)));
}

// ---------- roads (optional: OpenStreetMap via Overpass) ----------
async function roadsFor(a, b) {
  const { t, lat, lon } = st.info;
  const la = [], lo = [];
  t.forEach((ti, i) => { if (ti >= a - 300 && ti <= b + 20) { la.push(lat[i]); lo.push(lon[i]); } });
  if (!la.length) return null;
  const pad = 0.006, s = Math.min(...la) - pad, n = Math.max(...la) + pad, w = Math.min(...lo) - pad, e = Math.max(...lo) + pad;
  if (n - s > 0.25 || e - w > 0.25) throw new Error("the stretch is too long to fetch its roads; try a shorter outage");
  const key = [s, w, n, e].map((x) => x.toFixed(3)).join(",");
  if (st.roads.has(key)) return st.roads.get(key);
  status("Fetching the roads around this stretch from OpenStreetMap…");
  const q = `[out:json][timeout:25];way["highway"~"^(${ROAD_CLASSES.join("|")})$"](${s},${w},${n},${e});out geom;`;
  const r = await fetch("https://overpass-api.de/api/interpreter", { method: "POST", body: "data=" + encodeURIComponent(q),
    headers: { "Content-Type": "application/x-www-form-urlencoded" } });
  if (!r.ok) throw new Error(`OpenStreetMap road fetch failed (${r.status})`);
  const yes = (v) => ["yes", "true", "1"].includes(v);
  const ways = (await r.json()).elements.filter((x) => x.type === "way" && x.geometry && x.geometry.length > 1).map((x) => {
    const tg = x.tags || {}, rev = tg.oneway === "-1", g = rev ? [...x.geometry].reverse() : x.geometry;
    return { lat: g.map((p) => p.lat), lon: g.map((p) => p.lon), tunnel: yes(tg.tunnel) || tg.tunnel === "building_passage",
             oneway: yes(tg.oneway) || rev, highway: tg.highway };
  });
  st.roads.set(key, ways);
  return ways;
}

// ---------- running ----------
async function runAt(a) {
  if (st.busy || !st.info) return;
  st.busy = true; buttons(); st.a = a; drawTimeline();
  let roads = null;
  try { if ($("liveroads").checked) roads = await roadsFor(a, a + st.D); }
  catch (e) { status(`${e.message}. Running without the map stage.`, true); }
  progress(0, "starting"); st.pendingRoads = roads;
  ensureWorker().postMessage({ type: "run", id: st.runs.length + 1, outage: [a, a + st.D], vehicle: $("livevehicle").value, roads });
}

function randomSpot() {
  const all = valid(), done = st.runs.map((r) => r.outage);
  const fresh = all.filter((a) => done.every(([x, y]) => a + st.D <= x || a >= y));   // not overlapping earlier runs
  const v = fresh.length ? fresh : all;
  if (!v.length) { status("No stretch of this drive keeps moving long enough for that outage length.", true); return; }
  const rnd = mulberry32(st.seed++);
  runAt(v[Math.floor(rnd() * v.length)]);
}

function progress(p, stage) {
  const bar = $("liveprog");
  if (p === null) { bar.hidden = true; return; }
  bar.hidden = false; bar.querySelector("i").style.width = `${Math.round(100 * p)}%`;
  bar.querySelector("span").textContent = `Running ${stage}…`;
}

function buttons() {
  $("liverun").disabled = st.busy || !st.info; $("liverand").disabled = st.busy || !st.info;
}

function onResult(m) {
  st.busy = false; progress(null); buttons();
  m.roads = st.pendingRoads;
  st.runs.push(m); st.last = m;
  $("liveout").hidden = false;
  status(m.gaps ? "This recording has gaps in its own GNSS during the outage, so the reference is thin here."
                : `Outage ${fmt(m.outage[0])}–${fmt(m.outage[1])}: ${Math.round(m.dist)} m driven without GNSS.`);
  drawMap(m); drawRuns();
}

// ---------- results ----------
function drawMap(m) {
  const svg = $("livesvg"), [a, b] = m.outage, stages = Object.keys(m.stages);
  svg.replaceChildren();
  const T = m.truth, pts = T.map((r) => [r[1], -r[2]]);
  for (const s of stages) for (const r of m.stages[s].track) if (r[0] >= a - 20) pts.push([r[1], -r[2]]);
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const cw = svg.clientWidth || 600, ch = svg.clientHeight || 420, pad = 0.08;
  let vw = Math.max(x1 - x0, 80) * (1 + 2 * pad), vh = Math.max(y1 - y0, 80) * (1 + 2 * pad);
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  if (vw / vh > cw / ch) vh = (vw * ch) / cw; else vw = (vh * cw) / ch;
  const mpp = vw / cw;
  svg.setAttribute("viewBox", `${cx - vw / 2} ${cy - vh / 2} ${vw} ${vh}`);
  const path = (P) => P.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join("");
  if (m.roads && m.lat0 !== undefined) {
    const k = (Math.PI / 180) * 6371000, cl = Math.cos((m.lat0 * Math.PI) / 180), g = el("g", {}, svg), ds = [];
    for (const w of m.roads) ds.push(path(w.lat.map((la, i) => [(w.lon[i] - m.lon0) * cl * k, -(la - m.lat0) * k])));
    el("path", { d: ds.join(""), class: "rd", stroke: css("--r-casing"), "stroke-width": 6 }, g);
    el("path", { d: ds.join(""), class: "rd", stroke: "#fff", "stroke-width": 4 }, g);
  }
  const seg = (lo, hi) => T.filter((r) => r[0] >= lo && r[0] <= hi).map((r) => [r[1], -r[2]]);
  el("path", { d: path(seg(-Infinity, a)), class: "trail", stroke: css("--truth"), "stroke-width": 2.2 }, svg);
  el("path", { d: path(seg(a, b)), class: "trail", stroke: css("--truth"), "stroke-width": 2.2, "stroke-dasharray": "1 5" }, svg);
  el("path", { d: path(seg(b, Infinity)), class: "trail", stroke: css("--truth"), "stroke-width": 2.2 }, svg);
  const shipped = stages.includes("map") ? "map" : "head";
  for (const s of stages) {
    const tr = m.stages[s].track.filter((r) => r[0] >= a - 20 && r[0] < b);
    el("path", { d: path(tr.map((r) => [r[1], -r[2]])), class: "trail", stroke: css(STAGE_VAR[s]), "stroke-width": s === shipped ? 3 : 2 }, svg);
  }
  const tEnd = T.filter((r) => r[0] < b).pop();
  if (tEnd) el("circle", { cx: tEnd[1], cy: -tEnd[2], r: 5.5 * mpp, fill: "#fff", stroke: css("--truth"), "stroke-width": 2.2, "vector-effect": "non-scaling-stroke" }, svg);
  for (const s of stages) {
    const e = m.stages[s].track.filter((r) => r[0] < b).pop();
    if (!e) continue;
    if (s !== shipped) { el("circle", { cx: e[1], cy: -e[2], r: 4.5 * mpp, fill: css(STAGE_VAR[s]), stroke: "#fff", "stroke-width": 1.8, "vector-effect": "non-scaling-stroke" }, svg); continue; }
    const x = e[1], y = -e[2], R = 46 * mpp, h = (e[3] * 180) / Math.PI, w = (35 * Math.PI) / 180;
    el("circle", { cx: x, cy: y, r: Math.max(e[4], 3 * mpp), fill: css("--puck"), "fill-opacity": 0.15, stroke: css("--puck"), "stroke-opacity": 0.4, "stroke-width": 1, "vector-effect": "non-scaling-stroke" }, svg);
    const gid = "lbeam";
    const defs = el("defs", {}, svg), grad = el("radialGradient", { id: gid, gradientUnits: "userSpaceOnUse", cx: x, cy: y, r: R }, defs);
    el("stop", { offset: 0, "stop-color": "#4285F4", "stop-opacity": 0.55 }, grad); el("stop", { offset: 1, "stop-color": "#4285F4", "stop-opacity": 0 }, grad);
    el("path", { d: `M${x} ${y}L${x - R * Math.sin(w)} ${y - R * Math.cos(w)}A${R} ${R} 0 0 1 ${x + R * Math.sin(w)} ${y - R * Math.cos(w)}Z`, fill: `url(#${gid})`, transform: `rotate(${h.toFixed(1)} ${x} ${y})` }, svg);
    el("circle", { cx: x, cy: y, r: 8.5 * mpp, fill: "#fff", stroke: "rgba(60,64,67,.25)", "stroke-width": 1, "vector-effect": "non-scaling-stroke" }, svg);
    el("circle", { cx: x, cy: y, r: 6 * mpp, fill: css("--puck") }, svg);
  }
  const L = $("livelegend"); L.replaceChildren();
  const add = (color, name, val, cls = "") => {
    const li = document.createElement("li"); li.className = cls; li.style.setProperty("--c", color);
    li.innerHTML = `<span class="sw"></span><span class="nm"></span><span class="val"></span>`;
    li.querySelector(".nm").textContent = name; li.querySelector(".val").textContent = val; L.appendChild(li);
  };
  add(css("--truth"), "Where the phone's own GNSS says the car went", "", "truth");
  for (const s of stages) {
    const r = m.stages[s];
    add(css(STAGE_VAR[s]), r.label, Number.isFinite(r.err) ? `${Math.round(r.err)} m · ${r.drift.toFixed(1)} %` : "–", s === shipped ? "shipped" : "");
  }
}

function drawRuns() {
  const tb = $("liveruns"), stages = ["physics", "speednet", "head", "map"].filter((s) => st.runs.some((r) => r.stages[s]));
  const head = `<tr><th>#</th><th>Outage</th><th>Driven</th>${stages.map((s) => `<th><span class="dot" style="--c:${css(STAGE_VAR[s])}"></span>${
    { physics: "No AI", speednet: "+ SpeedNet", head: "+ head", map: "+ map" }[s]}</th>`).join("")}</tr>`;
  const cell = (r, s) => (r.stages[s] && Number.isFinite(r.stages[s].drift) ? `${r.stages[s].drift.toFixed(1)} %` : "–");
  const rows = st.runs.map((r, i) => `<tr><td>${i + 1}</td><td>${fmt(r.outage[0])}–${fmt(r.outage[1])}</td><td>${Math.round(r.dist)} m</td>${
    stages.map((s) => `<td>${cell(r, s)}</td>`).join("")}</tr>`).join("");
  let med = "";
  if (st.runs.length > 1) {
    const md = (s) => { const v = st.runs.map((r) => r.stages[s] && r.stages[s].drift).filter(Number.isFinite).sort((x, y) => x - y);
      return v.length ? `${v[(v.length - 1) >> 1].toFixed(1)} %` : "–"; };
    med = `<tr class="med"><td></td><td>Median of ${st.runs.length}</td><td></td>${stages.map((s) => `<td>${md(s)}</td>`).join("")}</tr>`;
  }
  tb.innerHTML = `<thead>${head}</thead><tbody>${rows}${med}</tbody>`;
}

/** A sample drive, if the site ships one: data/sample/manifest.json = {label, files: [..]}. */
async function offerSample() {
  try {
    const r = await fetch("data/sample/manifest.json", { cache: "no-cache" });
    if (!r.ok) return;
    const m = await r.json(), b = $("livesample");
    b.textContent = m.label || "Use the sample drive"; b.hidden = false;
    b.addEventListener("click", async () => {
      status("Downloading the sample drive…");
      const files = await Promise.all(m.files.map(async (f) => {
        const x = await fetch(`data/sample/${f}`); if (!x.ok) throw new Error(`sample ${f}: ${x.status}`);
        return new File([await x.blob()], f, { type: "text/csv" });
      }));
      onFiles(files);
    });
  } catch { /* no sample on this site */ }
}

// ---------- wiring ----------
export function initLive() {
  if (!$("live")) return;
  offerSample();
  $("files").addEventListener("change", (e) => onFiles(e.target.files));
  $("livedur").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b || !st.info) return;
    st.D = +b.dataset.d; segButtons();
    if (st.a + st.D > st.info.duration - 20) st.a = Math.max(0, st.info.duration - 20 - st.D);
    drawTimeline();
  });
  $("timeline").addEventListener("click", (e) => {
    if (!st.info || st.busy) return;
    const r = e.currentTarget.getBoundingClientRect(), t = ((e.clientX - r.left) / r.width) * st.info.duration;
    st.a = Math.max(0, Math.min(st.info.duration - 20 - st.D, t - st.D / 2)); drawTimeline();
  });
  $("liverun").addEventListener("click", () => runAt(st.a));
  $("liverand").addEventListener("click", randomSpot);
  new ResizeObserver(() => { if (st.info) drawTimeline(); if (st.last) drawMap(st.last); }).observe($("live"));
  buttons();
}
