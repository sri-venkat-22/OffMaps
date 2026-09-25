// OffMaps replay: validation-drive outages from py/build_site.py, drawn on the real OSM roads.
"use strict";

const NS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const STAGE_VAR = { physics: "--s-physics", speednet: "--s-speednet", head: "--s-head", map: "--s-map" };
const STAGE_W = { physics: 2, speednet: 2, head: 2.2, map: 3.2 };
// roads.bin class codes (tools/osm_layers.py HW_CODE) -> style, drawn minor first
const ROAD_STYLE = [
  { name: "service", codes: [14], fill: "--r-service", w: 1.6, casing: 0 },
  { name: "minor", codes: [0, 11, 12, 13], fill: "--r-minor", w: 3.4, casing: 1.6 },
  { name: "tertiary", codes: [9, 10], fill: "--r-tertiary", w: 4.2, casing: 1.6 },
  { name: "secondary", codes: [7, 8], fill: "--r-secondary", w: 4.8, casing: 1.6 },
  { name: "primary", codes: [5, 6], fill: "--r-primary", w: 5.4, casing: 1.6 },
  { name: "trunk", codes: [3, 4], fill: "--r-trunk", w: 6, casing: 1.6 },
  { name: "motorway", codes: [1, 2], fill: "--r-motorway", w: 6.4, casing: 1.6 },
];

const st = { meta: null, outages: [], roads: {}, abl: null, dur: 60, list: [], pos: 0, cur: null,
             k: 0, t: 0, playing: false, speed: 4, hidden: new Set(), view: null, last: 0 };

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
const css = (v) => `var(${v})`;
const fmtTime = (s) => { s = Math.max(0, Math.round(s)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
const driveName = (id) => (id.match(/\/(S\d\w*)\//) || [, id])[1];

async function load() {
  const [o, r, a] = await Promise.all(["data/outages.json", "data/roads.json", "data/ablation.json"]
    .map((p) => fetch(p).then((x) => { if (!x.ok) throw new Error(`${p}: ${x.status}`); return x.json(); })));
  st.meta = o.meta; st.outages = o.outages; st.abl = a;
  for (const [drive, pieces] of Object.entries(r)) {
    st.roads[drive] = pieces.map((p) => {
      const xy = new Float32Array(p.length - 2);
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (let i = 2; i < p.length; i += 2) {
        const x = p[i] / 10, y = -p[i + 1] / 10;
        xy[i - 2] = x; xy[i - 1] = y;
        x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
      }
      return { c: p[0], xy, box: [x0, y0, x1, y1] };
    });
  }
}

// ---------- outage selection ----------
function selectDuration(d) {
  st.dur = d;
  document.querySelectorAll(".seg button").forEach((b) => b.setAttribute("aria-selected", String(+b.dataset.dur === d)));
  st.list = st.outages.filter((o) => o.dur === d).sort((a, b) => a.stages.map.drift - b.stages.map.drift);
  choose(Math.floor((st.list.length - 1) / 2));
}

function choose(pos) {
  st.pos = Math.max(0, Math.min(st.list.length - 1, pos));
  prepare(st.list[st.pos]);
  const o = st.cur, n = st.list.length, med = Math.floor((n - 1) / 2);
  const tag = st.pos === med ? " (the median one)" : st.pos === 0 ? " (the best one)" : st.pos === n - 1 ? " (the worst one)" : "";
  $("which").textContent = `Outage ${st.pos + 1} of ${n}${tag}, ranked by the shipped app's drift · drive ${driveName(o.drive)} · `
    + `${Math.round(o.dist)} m driven without GNSS · ${fmtTime(o.t0)} into the drive`;
  st.t = 0; st.k = 0;
  $("scrub").max = o.N - 1; $("scrub").value = 0;
  const band = $("deniedband");
  band.style.left = `${(100 * o.d0) / (o.N - 1)}%`;
  band.style.width = `${(100 * (o.d1 - o.d0)) / (o.N - 1)}%`;
  layout(); draw();
  if (!REDUCED) play(true);
}

function prepare(o) {
  if (!o.N) {
    o.N = o.truth[0].length; o.d0 = o.denied[0]; o.d1 = Math.min(o.denied[1], o.N - 1);
    o.T = Array.from({ length: o.N }, (_, i) => [o.truth[0][i] / 10, -o.truth[1][i] / 10]);
    for (const s of st.meta.stages) {
      const g = o.stages[s];
      g.P = Array.from({ length: o.N }, (_, i) => [g.e[i] / 10, -g.n[i] / 10]);
    }
    o.cum = new Float32Array(o.N);            // distance driven since GNSS was lost
    for (let i = o.d0 + 1; i < o.N; i++) o.cum[i] = o.cum[i - 1] + Math.hypot(o.T[i][0] - o.T[i - 1][0], o.T[i][1] - o.T[i - 1][1]);
  }
  st.cur = o;
}

// ---------- geometry ----------
function layout() {
  const o = st.cur, svg = $("svg");
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of o.T) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  // include the estimates, but a wild one may only widen the view by 60 % of the true path's extent
  const w = Math.max(x1 - x0, 120), h = Math.max(y1 - y0, 120);
  const lim = [x0 - 0.6 * w, y0 - 0.6 * h, x1 + 0.6 * w, y1 + 0.6 * h];
  for (const s of st.meta.stages) for (const [x, y] of o.stages[s].P) {
    if (x < lim[0] || x > lim[2] || y < lim[1] || y > lim[3]) continue;
    x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  }
  const cw = svg.clientWidth || 800, ch = svg.clientHeight || 550, pad = 0.09;
  let vw = (x1 - x0) * (1 + 2 * pad), vh = (y1 - y0) * (1 + 2 * pad);
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  if (vw / vh > cw / ch) vh = vw * ch / cw; else vw = vh * cw / ch;
  st.view = { x: cx - vw / 2, y: cy - vh / 2, w: vw, h: vh, mpp: vw / cw };
  svg.setAttribute("viewBox", `${st.view.x} ${st.view.y} ${vw} ${vh}`);
  drawRoads(); drawTruth(); drawScale();
}

function drawRoads() {
  const g = $("roads"); g.replaceChildren();
  const v = st.view, pieces = st.roads[st.cur.drive] || [];
  const inView = pieces.filter((p) => p.box[2] >= v.x && p.box[0] <= v.x + v.w && p.box[3] >= v.y && p.box[1] <= v.y + v.h);
  const paths = ROAD_STYLE.map((s) => ({ s, d: [] }));
  for (const p of inView) {
    const k = ROAD_STYLE.findIndex((s) => s.codes.includes(p.c));
    let d = `M${p.xy[0].toFixed(1)} ${p.xy[1].toFixed(1)}`;
    for (let i = 2; i < p.xy.length; i += 2) d += `L${p.xy[i].toFixed(1)} ${p.xy[i + 1].toFixed(1)}`;
    paths[k < 0 ? 1 : k].d.push(d);
  }
  for (const { s, d } of paths) if (d.length && s.casing)
    el("path", { d: d.join(""), class: "rd", stroke: css("--r-casing"), "stroke-width": s.w + s.casing }, g);
  for (const { s, d } of paths) if (d.length)
    el("path", { d: d.join(""), class: "rd", stroke: css(s.fill), "stroke-width": s.w }, g);
}

const pathOf = (P, a, b) => P.slice(a, b + 1).map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join("");

function drawTruth() {
  const o = st.cur, g = $("truth"); g.replaceChildren();
  el("path", { d: pathOf(o.T, 0, o.d0), class: "trail", stroke: css("--truth"), "stroke-width": 2.4 }, g);
  el("path", { d: pathOf(o.T, o.d0, o.d1), class: "trail", stroke: css("--truth"), "stroke-width": 2.4, "stroke-dasharray": "1 5" }, g);
  el("path", { d: pathOf(o.T, o.d1, o.N - 1), class: "trail", stroke: css("--truth"), "stroke-width": 2.4 }, g);
}

function drawScale() {
  const mpp = st.view.mpp, target = 110 * mpp;
  const nice = [10, 20, 50, 100, 200, 500, 1000, 2000].find((x) => x >= target * 0.6) || 5000;
  $("scalebar").style.width = `${nice / mpp}px`;
  $("scaletext").textContent = nice >= 1000 ? `${nice / 1000} km` : `${nice} m`;
}

// ---------- frame ----------
function draw() {
  const o = st.cur, k = st.k, mpp = st.view.mpp;
  const out = k >= o.d0 && k < o.d1, after = k >= o.d1;
  $("map").classList.toggle("dark", out);
  $("skytext").textContent = out ? "No GNSS" : after ? "GNSS back" : "GNSS";

  const trails = $("trails"), dots = $("dots"), ring = $("ring"), snaps = $("snaps");
  trails.replaceChildren(); dots.replaceChildren(); ring.replaceChildren(); snaps.replaceChildren();
  for (const s of st.meta.stages) {
    if (st.hidden.has(s)) continue;
    const P = o.stages[s].P;
    el("path", { d: pathOf(P, 0, k), class: "trail", stroke: css(STAGE_VAR[s]), "stroke-width": STAGE_W[s], "stroke-opacity": s === "map" ? 1 : 0.9 }, trails);
  }
  if (!st.hidden.has("map")) {
    const m = o.stages.map;
    for (let i = o.d0; i <= Math.min(k, o.d1); i++) if (m.snap[i])
      el("circle", { cx: m.P[i][0], cy: m.P[i][1], r: 2.6 * mpp, fill: css("--paper"), stroke: css("--s-map"), "stroke-width": 1.4, "vector-effect": "non-scaling-stroke" }, snaps);
    const [x, y] = m.P[k];
    el("circle", { cx: x, cy: y, r: Math.max(m.sigma[k], 3 * mpp), fill: css("--s-map"), "fill-opacity": 0.08,
                   stroke: css("--s-map"), "stroke-width": 1.4, "stroke-dasharray": "4 3", "vector-effect": "non-scaling-stroke" }, ring);
  }
  if (!st.hidden.has("truth")) {
    const [tx, ty] = o.T[k];
    el("circle", { cx: tx, cy: ty, r: 5.5 * mpp, fill: css("--paper"), stroke: css("--truth"), "stroke-width": 2.2, "vector-effect": "non-scaling-stroke" }, dots);
  }
  for (const s of st.meta.stages) {
    if (st.hidden.has(s)) continue;
    const [x, y] = o.stages[s].P[k];
    el("circle", { cx: x, cy: y, r: (s === "map" ? 5.5 : 4.2) * mpp, fill: css(STAGE_VAR[s]), stroke: css("--paper"), "stroke-width": 1.6, "vector-effect": "non-scaling-stroke" }, dots);
  }

  // clock
  const hz = o.hz;
  if (k < o.d0) { $("clocklabel").textContent = "GNSS lost in"; $("clock").textContent = fmtTime((o.d0 - k) / hz); $("clocksub").textContent = "on GNSS: every estimate is corrected each second"; }
  else if (out) { $("clocklabel").textContent = "Without GNSS"; $("clock").textContent = fmtTime((k - o.d0) / hz); $("clocksub").textContent = `${Math.round(o.cum[k])} m driven blind`; }
  else { $("clocklabel").textContent = "GNSS back after"; $("clock").textContent = fmtTime(o.dur); $("clocksub").textContent = `${Math.round(o.dist)} m driven blind`; }

  // legend values
  for (const s of st.meta.stages) {
    const row = document.querySelector(`.legend li[data-s="${s}"]`);
    if (!row) continue;
    const g = o.stages[s];
    let txt, pct = 0;
    if (k < o.d0) { txt = `${Math.hypot(g.P[k][0] - o.T[k][0], g.P[k][1] - o.T[k][1]).toFixed(0)} m`; }
    else if (out) {
      const err = Math.hypot(g.P[k][0] - o.T[k][0], g.P[k][1] - o.T[k][1]);
      pct = o.cum[k] > 40 ? (100 * err) / o.cum[k] : 0;
      txt = o.cum[k] > 40 ? `${pct.toFixed(1)} %` : `${err.toFixed(0)} m`;
    } else { pct = g.drift; txt = `${g.drift.toFixed(1)} %`; }
    row.querySelector(".val").textContent = txt;
    row.querySelector(".bar i").style.width = `${Math.min(100, (pct / 40) * 100)}%`;
  }
  $("scrub").value = k;
}

// ---------- playback ----------
function play(on) {
  st.playing = on;
  $("playicon").setAttribute("d", on ? "M7 5h4v14H7zm6 0h4v14h-4z" : "M7 5v14l12-7z");
  $("play").setAttribute("aria-label", on ? "Pause" : "Play");
  if (on) { if (st.k >= st.cur.N - 1) { st.k = 0; st.t = 0; } st.last = performance.now(); requestAnimationFrame(tick); }
}
function tick(now) {
  if (!st.playing) return;
  st.t += ((now - st.last) / 1000) * st.speed; st.last = now;
  const k = Math.min(st.cur.N - 1, Math.floor(st.t * st.cur.hz));
  if (k !== st.k) { st.k = k; draw(); }
  if (k >= st.cur.N - 1) { play(false); return; }
  requestAnimationFrame(tick);
}

// ---------- legend + ablation ----------
function buildLegend() {
  const ul = $("legend"); ul.replaceChildren();
  const rows = [["truth", "Where the car really went"], ...st.meta.stages.map((s) => [s, st.meta.labels[s]])];
  for (const [s, label] of rows) {
    const li = document.createElement("li");
    li.dataset.s = s;
    li.style.setProperty("--c", css(STAGE_VAR[s] || "--truth"));
    if (s === "truth") li.className = "truth";
    if (s === "map") li.classList.add("shipped");
    const pretty = label.replace(/^\+ /, "+ ");
    li.innerHTML = `<button aria-pressed="true"><span class="sw"></span><span class="nm"></span><span class="val"></span>`
      + (s === "truth" ? "" : `<span class="bar"><i></i><b style="left:25%"></b></span>`) + `</button>`;
    li.querySelector(".nm").textContent = pretty;
    li.querySelector("button").addEventListener("click", (e) => {
      const on = st.hidden.has(s);
      on ? st.hidden.delete(s) : st.hidden.add(s);
      li.classList.toggle("off", !on); e.currentTarget.setAttribute("aria-pressed", String(on));
      draw();
    });
    ul.appendChild(li);
  }
}

function buildBars() {
  const host = $("bars"), MAX = 30;
  const sets = [["lodo", "Train drives · models never saw them · ~9.5 h"], ["val", "Validation drives · ~1 h"]];
  for (const [key, title] of sets) {
    const S = st.abl[key].summary, panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML = `<h3></h3>`; panel.querySelector("h3").textContent = title;
    for (const s of st.meta.stages) {
      const durs = Object.values(S[s]); const mean = durs.reduce((a, d) => a + d.median, 0) / durs.length;
      const row = document.createElement("div"); row.className = "brow";
      row.style.setProperty("--c", css(STAGE_VAR[s]));
      row.innerHTML = `<span class="lab"></span><span class="rail"><span class="fill" style="width:${(mean / MAX) * 100}%"></span>`
        + `<span class="tgt" style="left:${(10 / MAX) * 100}%"></span></span><span class="num">${mean.toFixed(1)} %</span>`;
      row.querySelector(".lab").textContent = st.meta.labels[s];
      panel.appendChild(row);
    }
    const cap = document.createElement("p"); cap.className = "cap";
    cap.textContent = `Bars run from 0 to ${MAX} %. The dashed line is the PS target of 10 %.`;
    panel.appendChild(cap);
    host.appendChild(panel);
  }
}

// ---------- wiring ----------
function wire() {
  document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => selectDuration(+b.dataset.dur)));
  document.querySelectorAll(".walk button").forEach((b) => b.addEventListener("click", () => {
    const n = st.list.length, j = b.dataset.jump;
    choose(j === "first" ? 0 : j === "last" ? n - 1 : j === "median" ? Math.floor((n - 1) / 2) : st.pos + (j === "next" ? 1 : -1));
  }));
  document.querySelectorAll(".speed button").forEach((b) => b.addEventListener("click", () => {
    st.speed = +b.dataset.speed;
    document.querySelectorAll(".speed button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  }));
  $("play").addEventListener("click", () => play(!st.playing));
  $("scrub").addEventListener("input", (e) => { play(false); st.k = +e.target.value; st.t = st.k / st.cur.hz; draw(); });
  new ResizeObserver(() => { if (st.cur) { layout(); draw(); } }).observe($("map"));
}

load().then(() => { buildLegend(); buildBars(); wire(); selectDuration(60); })
  .catch((e) => { $("which").textContent = `Could not load the replay data (${e.message}). Serve docs/ over HTTP, e.g. python3 -m http.server -d docs.`; });
