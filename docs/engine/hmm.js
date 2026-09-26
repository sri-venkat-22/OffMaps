// py/road_hmm.py in JavaScript: the online HMM road matcher (Newson & Krumm 2009 as a
// causal forward filter) on a directed segment graph of OSM ways.
//
// ways: [{lat: [...], lon: [...], tunnel: bool, oneway: bool, highway: "primary" | ...}]

const R_EARTH = 6371000.0, CELL_M = 100.0;
const HALF_WIDTH = { motorway: 7.0, trunk: 7.0, primary: 6.0, secondary: 5.0, tertiary: 4.5,
                     unclassified: 3.5, residential: 3.5, living_street: 3.0, service: 3.0 };
const pyMod = (x, m) => ((x % m) + m) % m;
const wrapArr = (a) => pyMod(a + Math.PI, 2 * Math.PI) - Math.PI;
const roundHalfEven = (x) => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r; };

class Heap {                                    // min-heap of [d, node], ties by node (Python tuples)
  constructor() { this.a = []; }
  get size() { return this.a.length; }
  _lt(i, j) { const x = this.a[i], y = this.a[j]; return x[0] < y[0] || (x[0] === y[0] && x[1] < y[1]); }
  push(v) {
    const a = this.a; a.push(v); let i = a.length - 1;
    while (i > 0) { const p = (i - 1) >> 1; if (!this._lt(i, p)) break; [a[i], a[p]] = [a[p], a[i]]; i = p; }
  }
  pop() {
    const a = this.a, top = a[0], last = a.pop();
    if (a.length) {
      a[0] = last; let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1; let m = i;
        if (l < a.length && this._lt(l, m)) m = l;
        if (r < a.length && this._lt(r, m)) m = r;
        if (m === i) break;
        [a[i], a[m]] = [a[m], a[i]]; i = m;
      }
    }
    return top;
  }
}

export class RoadGraph {
  constructor(ways, lat0, lon0, exclude = []) {
    const k = (Math.PI / 180) * R_EARTH, cl = Math.cos((lat0 * Math.PI) / 180);
    const kid = [];
    ways.forEach((w, i) => { if (!exclude.includes(w.highway)) kid.push(i); });
    // node ids: unique (round(lat*1e7), round(lon*1e7)), numbered in sorted order as np.unique does
    const pts = [];
    kid.forEach((wi, j) => {
      const w = ways[wi];
      for (let v = 0; v < w.lat.length; v++)
        pts.push({ j, v, ka: roundHalfEven(w.lat[v] * 1e7), ko: roundHalfEven(w.lon[v] * 1e7),
                   e: (w.lon[v] - lon0) * cl * k, n: (w.lat[v] - lat0) * k });
    });
    const keys = [...new Set(pts.map((p) => p.ka + "," + p.ko))]
      .map((s) => s.split(",").map(Number)).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const id = new Map(keys.map((kk, i) => [kk[0] + "," + kk[1], i]));
    for (const p of pts) p.node = id.get(p.ka + "," + p.ko);
    const A = [], B = [], P = [], D = [], L = [], BRG = [], WAY = [], VTX = [], ONE = [], TUN = [], HW = [];
    for (let s = 0; s + 1 < pts.length; s++) {
      const p = pts[s], q = pts[s + 1];
      if (p.j !== q.j || (q.e === p.e && q.n === p.n)) continue;
      const w = ways[kid[p.j]], de = q.e - p.e, dn = q.n - p.n;
      A.push(p.node); B.push(q.node); P.push([p.e, p.n]); D.push([de, dn]);
      L.push(Math.hypot(de, dn)); BRG.push(Math.atan2(de, dn));
      WAY.push(kid[p.j]); VTX.push(p.v); ONE.push(!!w.oneway); TUN.push(!!w.tunnel);
      HW.push(HALF_WIDTH[(w.highway || "").replace("_link", "")] ?? 3.5);
    }
    Object.assign(this, { a: A, b: B, p: P, d: D, len: L, brg: BRG, way: WAY, vtx: VTX, oneway: ONE, tunnel: TUN, halfWidth: HW });
    this.out = new Map();
    const add = (u, x) => { if (!this.out.has(u)) this.out.set(u, []); this.out.get(u).push(x); };
    for (let i = 0; i < A.length; i++) { add(A[i], [B[i], i, 1]); if (!ONE[i]) add(B[i], [A[i], i, -1]); }
    this.grid = new Map();
    for (let i = 0; i < A.length; i++) {
      const x0 = Math.floor(Math.min(P[i][0], P[i][0] + D[i][0]) / CELL_M), x1 = Math.floor(Math.max(P[i][0], P[i][0] + D[i][0]) / CELL_M);
      const y0 = Math.floor(Math.min(P[i][1], P[i][1] + D[i][1]) / CELL_M), y1 = Math.floor(Math.max(P[i][1], P[i][1] + D[i][1]) / CELL_M);
      for (let cx = x0; cx <= x1; cx++) for (let cy = y0; cy <= y1; cy++) {
        const key = cx + "," + cy;
        if (!this.grid.has(key)) this.grid.set(key, []);
        this.grid.get(key).push(i);
      }
    }
  }

  candidates(xy, r) {
    const c0 = [Math.floor((xy[0] - r) / CELL_M), Math.floor((xy[1] - r) / CELL_M)];
    const c1 = [Math.floor((xy[0] + r) / CELL_M), Math.floor((xy[1] + r) / CELL_M)];
    const set = new Set();
    for (let cx = c0[0]; cx <= c1[0]; cx++) for (let cy = c0[1]; cy <= c1[1]; cy++) {
      const g = this.grid.get(cx + "," + cy); if (g) for (const i of g) set.add(i);
    }
    const ids = [...set].sort((a, b) => a - b), out = [];
    for (const i of ids) {
      const p = this.p[i], d = this.d[i], L = this.len[i];
      const t = Math.min(Math.max(((xy[0] - p[0]) * d[0] + (xy[1] - p[1]) * d[1]) / (L * L), 0.0), 1.0);
      const foot = [p[0] + t * d[0], p[1] + t * d[1]], dist = Math.hypot(foot[0] - xy[0], foot[1] - xy[1]);
      if (dist <= r) out.push({ seg: i, foot, dist, t });
    }
    return out;
  }

  reach(src, maxD) {                            // bounded Dijkstra: node -> driving distance
    const dist = new Map([[src, 0.0]]), pq = new Heap();
    pq.push([0.0, src]);
    while (pq.size) {
      const [dd, u] = pq.pop();
      if (dd > (dist.has(u) ? dist.get(u) : Infinity) || dd > maxD) continue;
      for (const [v, s] of this.out.get(u) || []) {
        const nd = dd + this.len[s];
        if (nd < (dist.has(v) ? dist.get(v) : Infinity) && nd <= maxD) { dist.set(v, nd); pq.push([nd, v]); }
      }
    }
    return dist;
  }
}

export class HMMMatcher {
  constructor(graph, { sigmaMin = 4.0, beta = 8.0, radius = 45.0, headingSigmaDeg = 25.0, maxCands = 10, radiusMax = 150.0 } = {}) {
    Object.assign(this, { g: graph, sigmaMin, beta, radius, radiusMax, maxCands, hs: (headingSigmaDeg * Math.PI) / 180 });
    this.reset();
  }
  reset() { this.prev = null; this.logp = null; this.travel = 0.0; this.last = null; }
  addTravel(d) { this.travel += d; }

  _cands(xy, psi, moving, r) {
    const c = this.g.candidates(xy, r);
    if (!c.length) return null;
    for (const x of c) {
      x.dirn = 1; x.head = this.g.brg[x.seg];
      if (moving && !this.g.oneway[x.seg] && Math.abs(wrapArr(psi - x.head)) > Math.PI / 2) { x.dirn = -1; x.head = wrapArr(x.head + Math.PI); }
    }
    c.sort((a, b) => a.dist - b.dist || a.seg - b.seg);
    return c.slice(0, this.maxCands);
  }

  _route(a, b, cache) {
    const g = this.g, L0 = g.len[a.seg], L1 = g.len[b.seg];
    if (a.seg === b.seg && a.dirn === b.dirn) {
      const along = (b.t - a.t) * L0 * a.dirn;
      return along >= -3.0 ? along : Infinity;
    }
    const exit0 = a.dirn > 0 ? g.b[a.seg] : g.a[a.seg], key = a.seg + "," + a.dirn;
    if (!cache.has(key)) cache.set(key, g.reach(exit0, this.travel + 60.0));
    const entry1 = b.dirn > 0 ? g.a[b.seg] : g.b[b.seg], sp = cache.get(key).get(entry1);
    if (sp === undefined) return Infinity;
    return (a.dirn > 0 ? (1 - a.t) * L0 : a.t * L0) + sp + (b.dirn > 0 ? b.t * L1 : (1 - b.t) * L1);
  }

  update(xy, psi, speed, posSigma) {
    const moving = speed > 2.0, sig = Math.max(this.sigmaMin, posSigma);
    const cu = this._cands(xy, psi, moving, Math.min(Math.max(this.radius, 3 * sig), this.radiusMax));
    if (cu === null) { this.reset(); return null; }
    const em = cu.map((c) => {
      let e = -0.5 * (c.dist / sig) ** 2;
      if (moving) e -= 0.5 * (wrapArr(psi - c.head) / this.hs) ** 2;
      return e;
    });
    let lp;
    if (this.prev === null) lp = em.slice();
    else {
      const pv = this.prev, plp = this.logp, cache = new Map();
      lp = em.map((e, j) => {
        let best = -Infinity;
        for (let i = 0; i < plp.length; i++) {
          if (plp[i] === -Infinity) continue;
          const rd = this._route(pv[i], cu[j], cache);
          if (rd < Infinity) best = Math.max(best, plp[i] - Math.abs(rd - this.travel) / this.beta);
        }
        return best + e;
      });
      if (!lp.some(Number.isFinite)) lp = em.slice();
    }
    const mx = Math.max(...lp);
    lp = lp.map((x) => x - mx);
    let post = lp.map(Math.exp); const tot = post.reduce((s, x) => s + x, 0); post = post.map((x) => x / tot);
    this.prev = cu; this.logp = post.map((x) => Math.log(Math.max(x, 1e-300))); this.travel = 0.0;
    let k = 0; for (let i = 1; i < post.length; i++) if (post[i] > post[k]) k = i;
    const best = cu[k], s = best.seg, L = this.g.len[s];
    let conf = 0;
    for (let i = 0; i < cu.length; i++) {
      const df = Math.hypot(cu[i].foot[0] - best.foot[0], cu[i].foot[1] - best.foot[1]);
      if (df < 5.0 && Math.abs(wrapArr(cu[i].head - best.head)) < (30 * Math.PI) / 180) conf += post[i];
    }
    this.last = { seg: s, way: this.g.way[s], vtx: this.g.vtx[s], foot: best.foot, bearing: best.head, dist: best.dist,
                  conf, segConf: post[k], segLen: L, endDist: Math.min(best.t, 1 - best.t) * L,
                  halfWidth: this.g.halfWidth[s], tunnel: this.g.tunnel[s] };
    return this.last;
  }
}
