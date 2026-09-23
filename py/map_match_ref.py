"""Pure-Python twin of the Viterbi/HMM map matcher in core/map_match.cpp.

The ORACLE for the Phase-7b parity check. Mirrors mm_match_seq exactly: same
candidate gate, same emission/transition costs, same constants, same strict
tie-break (lowest way index wins), so the decoded way-index sequence is
bit-identical to the native core. It also exposes greedy_seq (per-step nearest,
what mm_match does) so the gate can show what Viterbi fixes at a junction.

Change one, change both, re-run parity.
"""
from __future__ import annotations
import numpy as np

SEQ_GATE2 = 2500.0     # candidate if within 50 m of a way
SWITCH_PEN = 1.0       # cost to change way across a junction (adjacent)
FAR_PEN = 50.0         # extra cost to hop between non-adjacent ways


def _wrap(a):
    while a > np.pi: a -= 2 * np.pi
    while a <= -np.pi: a += 2 * np.pi
    return a


class Way:
    def __init__(self, e, n, tunnel, oneway):
        self.e = np.asarray(e, float); self.n = np.asarray(n, float)
        self.tunnel = int(tunnel); self.oneway = int(oneway)
        self.cum = np.zeros(len(self.e))
        self.cum[1:] = np.cumsum(np.hypot(np.diff(self.e), np.diff(self.n)))


def _proj_way(w, px, py):
    """Best foot + tangent bearing + dist^2 + arc length to foot (matches C++)."""
    best = (1e18, px, py, 0.0, 0.0)     # d2, fe, fn, brg, arclen
    for i in range(len(w.e) - 1):
        dx = w.e[i+1] - w.e[i]; dy = w.n[i+1] - w.n[i]
        seg = np.hypot(dx, dy); L2 = dx*dx + dy*dy + 1e-12
        t = ((px - w.e[i]) * dx + (py - w.n[i]) * dy) / L2
        t = 0.0 if t < 0 else 1.0 if t > 1 else t
        fx = w.e[i] + t*dx; fy = w.n[i] + t*dy
        dd = (px-fx)**2 + (py-fy)**2
        if dd < best[0]:
            best = (dd, fx, fy, np.arctan2(dx, dy), w.cum[i] + t*seg)
    return best


class Cand:
    __slots__ = ("way", "fe", "fn", "brg", "cross", "arclen", "corridor")
    def __init__(self, way, fe, fn, brg, cross, arclen, corridor):
        self.way, self.fe, self.fn, self.brg = way, fe, fn, brg
        self.cross, self.arclen, self.corridor = cross, arclen, corridor


class MapMatcherRef:
    def __init__(self):
        self.ways = []
        self.bearing_gate = np.pi / 4
        self.adj = None
        self.has_edges = 0
        self.sigma_emit = 8.0
        self.sigma_trans = 12.0
        self.beta_bearing = 4.0

    def add_way(self, e, n, tunnel=0, oneway=0):
        self.ways.append(Way(e, n, tunnel, oneway))

    def add_edge(self, a, b):
        W = len(self.ways)
        if not (0 <= a < W and 0 <= b < W): return
        if self.adj is None or len(self.adj) != W:
            self.adj = np.zeros((W, W), np.int8)
        self.adj[a, b] = self.adj[b, a] = 1
        self.has_edges = 1

    def set_hmm(self, sigma_emit=None, sigma_trans=None, beta_bearing=None):
        if sigma_emit: self.sigma_emit = sigma_emit
        if sigma_trans: self.sigma_trans = sigma_trans
        if beta_bearing is not None and beta_bearing >= 0: self.beta_bearing = beta_bearing

    def _adjacent(self, a, b):
        if a == b: return True
        if not self.has_edges: return True
        if a < 0 or b < 0: return False
        return bool(self.adj[a, b])

    def _cands(self, e, n, psi):
        cs = []
        for j, w in enumerate(self.ways):
            d2, fe, fn, brg, arclen = _proj_way(w, e, n)
            if d2 > SEQ_GATE2: continue
            db = abs(_wrap(brg - psi))
            if db > self.bearing_gate and db < np.pi - self.bearing_gate: continue
            nx, ny = -np.cos(brg), np.sin(brg)
            cross = (fe - e) * nx + (fn - n) * ny
            cs.append(Cand(j, fe, fn, brg, cross, arclen, w.tunnel))
        return cs

    def _emit(self, c, psi):
        db = _wrap(c.brg - psi)
        align = np.cos(db) if self.ways[c.way].oneway else abs(np.cos(db))
        dcross = c.cross / self.sigma_emit
        return 0.5 * dcross * dcross + self.beta_bearing * (1.0 - align)

    def match_seq(self, e, n, psi):
        """Viterbi decode. Returns dict(way, foot_e, foot_n, bearing, corridor)."""
        e = np.asarray(e, float); n = np.asarray(n, float); psi = np.asarray(psi, float)
        ns = len(e)
        if ns == 0:
            return dict(way=np.array([], int), foot_e=np.array([]), foot_n=np.array([]),
                        bearing=np.array([]), corridor=np.array([], int))
        NULL_EMIT = 0.5 * (SEQ_GATE2 / (self.sigma_emit ** 2)) + self.beta_bearing
        cand, cost, back = [], [], []
        for i in range(ns):
            cs = self._cands(e[i], n[i], psi[i])
            cs.append(Cand(-1, e[i], n[i], psi[i], 0.0, 0.0, 0))   # null (off-road)
            cand.append(cs)
            C = len(cs)
            row = np.zeros(C); bp = [-1] * C
            for c in range(C):
                em = NULL_EMIT if cs[c].way < 0 else self._emit(cs[c], psi[i])
                if i == 0:
                    row[c] = em; continue
                dgps = np.hypot(e[i]-e[i-1], n[i]-n[i-1])
                best = 1e300; bestp = -1
                for p in range(len(cand[i-1])):
                    P = cand[i-1][p]; Cc = cs[c]
                    if P.way < 0 or Cc.way < 0:
                        base, droute = SWITCH_PEN, dgps
                    elif P.way == Cc.way:
                        base, droute = 0.0, abs(Cc.arclen - P.arclen)
                    elif self._adjacent(P.way, Cc.way):
                        base, droute = SWITCH_PEN, np.hypot(Cc.fe-P.fe, Cc.fn-P.fn)
                    else:
                        base, droute = SWITCH_PEN + FAR_PEN, np.hypot(Cc.fe-P.fe, Cc.fn-P.fn)
                    dc = (dgps - droute) / self.sigma_trans
                    tot = cost[i-1][p] + base + 0.5 * dc * dc
                    if tot < best:
                        best = tot; bestp = p
                row[c] = em + best; bp[c] = bestp
            cost.append(row); back.append(bp)
        # backtrack
        last = cost[ns-1]; c = int(np.argmin(last))   # argmin -> first min (matches strict <)
        chosen = [0] * ns
        for i in range(ns-1, -1, -1):
            chosen[i] = c
            if i: c = back[i][c]
        way = np.empty(ns, int); fe = np.empty(ns); fn = np.empty(ns)
        brg = np.empty(ns); cor = np.empty(ns, int)
        for i in range(ns):
            s = cand[i][chosen[i]]; ncand = len(cand[i]) - 1
            way[i] = s.way; fe[i] = s.fe; fn[i] = s.fn; brg[i] = s.brg
            cor[i] = 1 if (s.way >= 0 and (s.corridor or ncand <= 1)) else 0
        return dict(way=way, foot_e=fe, foot_n=fn, bearing=brg, corridor=cor)

    def greedy_seq(self, e, n, psi):
        """Per-step nearest-road (what mm_match does): min emission, no transition.
        The baseline Viterbi improves on -- prone to wrong-snaps at junctions."""
        e = np.asarray(e, float); n = np.asarray(n, float); psi = np.asarray(psi, float)
        ns = len(e); way = np.full(ns, -1, int)
        for i in range(ns):
            cs = self._cands(e[i], n[i], psi[i])
            if not cs: continue
            costs = [self._emit(c, psi[i]) for c in cs]
            way[i] = cs[int(np.argmin(costs))].way
        return way
