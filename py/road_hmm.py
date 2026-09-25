"""Online HMM map matching (Newson & Krumm 2009, run as a causal forward filter).

Phase 9 alternative to the greedy / fixed-lag matchers in road_window.py. States are
directed road segments (consecutive OSM vertices) near the filter's position:

  emission    N(distance; 0, sigma) * N(heading error; 0, 25 deg)   sigma = max(4 m, filter sigma)
  transition  exp(-|route distance - distance driven| / beta)        route distance by Dijkstra
                                                                     on the directed road graph

It runs at 1 Hz in GNSS mode too, so it is already locked onto the right road when
an outage starts. The posterior of the best road is the match confidence; the
engine only uses a match that is confident AND made while the filter itself is
still tight (edge_engine._map_step_hmm). Roads can be restricted by OSM class:
service roads (car parks, driveways) are half of the IO-VNBD-area network and are
the parallel "roads" a drifting estimate snaps onto.
"""
from __future__ import annotations
import heapq, math
from collections import defaultdict
import numpy as np

R_EARTH = 6_371_000.0
CELL_M = 100.0
# carriageway half-width (m) by OSM highway class; the cross-track sigma is 0.6 x this
HALF_WIDTH = {"motorway": 7.0, "trunk": 7.0, "primary": 6.0, "secondary": 5.0, "tertiary": 4.5,
              "unclassified": 3.5, "residential": 3.5, "living_street": 3.0, "service": 3.0}
_GRAPHS = {}


class RoadGraph:
    """Directed segment graph of roads.bin ways in an ENU frame at (lat0, lon0)."""

    def __init__(self, ways, lat0, lon0, exclude=()):
        k = math.pi / 180 * R_EARTH; cl = math.cos(math.radians(lat0))
        keep = [w for w in ways if (w[4] if len(w) > 4 else None) not in exclude]
        la = np.concatenate([w[0] for w in keep]); lo = np.concatenate([w[1] for w in keep])
        wid = np.repeat(np.arange(len(keep)), [len(w[0]) for w in keep])
        key = np.stack([np.round(la * 1e7), np.round(lo * 1e7)], 1).astype(np.int64)
        _, node = np.unique(key, axis=0, return_inverse=True)
        node = node.ravel()
        e = (lo - lon0) * cl * k; n = (la - lat0) * k
        s = np.nonzero(wid[:-1] == wid[1:])[0]                  # point s -> s+1 is a segment
        s = s[(e[s + 1] != e[s]) | (n[s + 1] != n[s])]
        self.a, self.b = node[s], node[s + 1]
        self.p = np.stack([e[s], n[s]], 1)
        self.d = np.stack([e[s + 1] - e[s], n[s + 1] - n[s]], 1)
        self.len = np.hypot(self.d[:, 0], self.d[:, 1])
        self.brg = np.arctan2(self.d[:, 0], self.d[:, 1])       # compass bearing, geometry order
        w = wid[s]
        self.oneway = np.array([keep[i][3] for i in w], bool)
        self.tunnel = np.array([keep[i][2] for i in w], bool)
        cls = [keep[i][4] if len(keep[i]) > 4 else None for i in w]
        hw = {c: HALF_WIDTH.get((c or "").replace("_link", ""), 3.5) for c in set(cls)}
        self.half_width = np.array([hw[c] for c in cls])
        # directed adjacency: node -> [(next node, seg, dirn)]
        self.out = defaultdict(list)
        nbrs = defaultdict(set)
        for i, (u, v) in enumerate(zip(self.a.tolist(), self.b.tolist())):
            self.out[u].append((v, i, 1))
            if not self.oneway[i]:
                self.out[v].append((u, i, -1))
            nbrs[u].add(v); nbrs[v].add(u)
        self.degree = defaultdict(int, {u: len(x) for u, x in nbrs.items()})
        # uniform grid over segment bounding boxes
        lo_c = np.floor(np.minimum(self.p, self.p + self.d) / CELL_M).astype(int)
        hi_c = np.floor(np.maximum(self.p, self.p + self.d) / CELL_M).astype(int)
        grid = defaultdict(list)
        for i in range(len(self.len)):
            for cx in range(lo_c[i, 0], hi_c[i, 0] + 1):
                for cy in range(lo_c[i, 1], hi_c[i, 1] + 1):
                    grid[(cx, cy)].append(i)
        self.grid = {c: np.array(v) for c, v in grid.items()}

    @classmethod
    def cached(cls, ways, lat0, lon0, exclude=()):
        key = (id(ways), round(lat0, 7), round(lon0, 7), tuple(sorted(exclude)))
        if key not in _GRAPHS:
            _GRAPHS[key] = cls(ways, lat0, lon0, exclude)
        return _GRAPHS[key]

    def candidates(self, xy, r):
        """-> (seg ids, foot points, distance, t along segment) within r of xy."""
        c0 = np.floor((np.asarray(xy) - r) / CELL_M).astype(int)
        c1 = np.floor((np.asarray(xy) + r) / CELL_M).astype(int)
        ids = [self.grid[(cx, cy)] for cx in range(c0[0], c1[0] + 1) for cy in range(c0[1], c1[1] + 1)
               if (cx, cy) in self.grid]
        if not ids:
            return (np.empty(0, int),) * 4
        ids = np.unique(np.concatenate(ids))
        p, d, L = self.p[ids], self.d[ids], self.len[ids]
        t = np.clip(((xy[0] - p[:, 0]) * d[:, 0] + (xy[1] - p[:, 1]) * d[:, 1]) / (L * L), 0.0, 1.0)
        foot = p + t[:, None] * d
        dist = np.hypot(foot[:, 0] - xy[0], foot[:, 1] - xy[1])
        m = dist <= r
        return ids[m], foot[m], dist[m], t[m]

    def reach(self, src, max_d):
        """Bounded Dijkstra on the directed graph: node -> driving distance from src."""
        dist = {src: 0.0}; pq = [(0.0, src)]
        while pq:
            dd, u = heapq.heappop(pq)
            if dd > dist.get(u, math.inf) or dd > max_d:
                continue
            for v, s, _ in self.out.get(u, ()):
                nd = dd + self.len[s]
                if nd < dist.get(v, math.inf) and nd <= max_d:
                    dist[v] = nd; heapq.heappush(pq, (nd, v))
        return dist


class HMMMatcher:
    def __init__(self, graph, sigma_min=4.0, beta=8.0, radius=45.0, heading_sigma_deg=25.0,
                 max_cands=10):
        self.g = graph
        self.sigma_min, self.beta, self.radius = sigma_min, beta, radius
        self.hs = math.radians(heading_sigma_deg)
        self.max_cands = max_cands
        self.reset()

    def reset(self):
        self.prev = None; self.logp = None; self.travel = 0.0; self.last = None

    def _cands(self, xy, psi, moving, r):
        ids, foot, dist, t = self.g.candidates(xy, r)
        if len(ids) == 0:
            return None
        dirn = np.ones(len(ids), int)
        head = self.g.brg[ids].copy()
        if moving:
            rev = ~self.g.oneway[ids] & (np.abs(_wrap_arr(psi - head)) > math.pi / 2)
            dirn[rev] = -1; head[rev] = _wrap_arr(head[rev] + math.pi)
        o = np.argsort(dist)[: self.max_cands]
        return dict(seg=ids[o], dirn=dirn[o], foot=foot[o], dist=dist[o], t=t[o], head=head[o])

    def _route(self, pv, i, cu, j, cache):
        s0, d0, t0 = int(pv["seg"][i]), int(pv["dirn"][i]), pv["t"][i]
        s1, d1, t1 = int(cu["seg"][j]), int(cu["dirn"][j]), cu["t"][j]
        L0, L1 = self.g.len[s0], self.g.len[s1]
        if s0 == s1 and d0 == d1:
            along = (t1 - t0) * L0 * d0
            return along if along >= -3.0 else math.inf
        exit0 = int(self.g.b[s0] if d0 > 0 else self.g.a[s0])
        if (s0, d0) not in cache:
            cache[(s0, d0)] = self.g.reach(exit0, self.travel + 60.0)
        entry1 = int(self.g.a[s1] if d1 > 0 else self.g.b[s1])
        sp = cache[(s0, d0)].get(entry1)
        if sp is None:
            return math.inf
        return ((1 - t0) * L0 if d0 > 0 else t0 * L0) + sp + (t1 * L1 if d1 > 0 else (1 - t1) * L1)

    def update(self, xy, psi, speed, pos_sigma):
        """One 1 Hz step at the filter's (e, n, psi). Call add_travel() in between."""
        moving = speed > 2.0
        sig = max(self.sigma_min, pos_sigma)
        cu = self._cands(np.asarray(xy, float), psi, moving, max(self.radius, 3 * sig))
        if cu is None:
            self.reset(); return None
        em = -0.5 * (cu["dist"] / sig) ** 2
        if moving:
            em = em - 0.5 * (_wrap_arr(psi - cu["head"]) / self.hs) ** 2
        if self.prev is None:
            lp = em
        else:
            pv, plp, cache = self.prev, self.logp, {}
            lp = np.full(len(em), -np.inf)
            for j in range(len(em)):
                best = -np.inf
                for i in range(len(plp)):
                    if plp[i] == -np.inf:
                        continue
                    rd = self._route(pv, i, cu, j, cache)
                    if rd < math.inf:
                        best = max(best, plp[i] - abs(rd - self.travel) / self.beta)
                lp[j] = best + em[j]
            if not np.isfinite(lp).any():               # broken chain (off-map, big jump): restart
                lp = em
        lp = lp - lp.max()
        post = np.exp(lp); post /= post.sum()
        self.prev, self.logp, self.travel = cu, np.log(np.maximum(post, 1e-300)), 0.0
        k = int(np.argmax(post)); s = int(cu["seg"][k])
        L = self.g.len[s]; tk = cu["t"][k]
        # Confidence in the ROAD, not the segment: OSM draws a road as many short
        # vertex-to-vertex segments, and near a vertex two of them are equally likely.
        # Sum the posterior of candidates on the same line (feet within 5 m, same
        # direction within 30 deg). A parallel street or the opposite carriageway is
        # further than 5 m away, so it still counts against the match.
        df = np.hypot(*(cu["foot"] - cu["foot"][k]).T)
        same = (df < 5.0) & (np.abs(_wrap_arr(cu["head"] - cu["head"][k])) < math.radians(30))
        self.last = dict(seg=s, foot=cu["foot"][k], bearing=float(cu["head"][k]), dist=float(cu["dist"][k]),
                         conf=float(post[same].sum()), seg_conf=float(post[k]),
                         seg_len=float(L), end_dist=float(min(tk, 1 - tk) * L),
                         half_width=float(self.g.half_width[s]), tunnel=bool(self.g.tunnel[s]))
        return self.last

    def add_travel(self, d):
        self.travel += d


def _wrap_arr(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
