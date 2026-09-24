"""A moving window of offline roads around the car + LIVE (fixed-lag) Viterbi matching.

Python twin of RoadMatcher.kt (phone) for the edge engine and the host checks.

Greedy (Phase 5, what the phone ran): each 10 Hz step snaps to the nearest road
whose bearing agrees (core mm_match). Near a junction or a parallel street a
dead-reckoned position that has drifted a few metres can pick the wrong road,
and the cross-track update then pulls the car onto it.

Viterbi live (Phase 7b's decoder, run online): every second while
dead-reckoning, the last LAG one-second positions are decoded with mm_match_seq
(HMM: cross-track + bearing emission, road-adjacency + distance-consistency
transition), and the way chosen for the NEWEST step is the road used for the
next second's 10 Hz cross-track/heading updates. The decode is fixed-lag, so it
is causal: it only re-reads the past, never waits for the future. Adjacency
comes from the map itself: roads.bin splits OSM ways into <= 32-vertex pieces
that share endpoint coordinates, and ways meeting at a junction share that
node's coordinates, so "shares a vertex" == "connected".
"""
from __future__ import annotations
import math
from collections import defaultdict
import numpy as np
from core_bridge import MapMatcher

R_EARTH = 6_371_000.0
CELL_DEG = 0.005


class RoadWindow:
    WINDOW_M = 1000.0
    RECENTER_M = 400.0
    LAG = 20                       # seconds of history the live decoder re-reads
    MAX_CROSS = 25.0               # == FusionEngine MM_MAX_CROSS
    BEARING_GATE = math.pi / 4     # == core/map_match.cpp bearing_gate

    def __init__(self, ways_ll, lat0, lon0, hmm=None):
        """ways_ll: [(lat[], lon[], tunnel, oneway)] (tools/osm_layers.read_roads_bin);
        (lat0, lon0) is the live ENU origin (the first GNSS fix)."""
        self.ways = ways_ll
        self.lat0, self.lon0 = lat0, lon0
        self.k = math.pi / 180.0 * R_EARTH
        self.coslat = math.cos(math.radians(lat0))
        self.hmm = hmm
        cells = defaultdict(list)
        for w, (la, lo, _, _) in enumerate(ways_ll):
            for iy in range(int(math.floor(la.min() / CELL_DEG)), int(math.floor(la.max() / CELL_DEG)) + 1):
                for ix in range(int(math.floor(lo.min() / CELL_DEG)), int(math.floor(lo.max() / CELL_DEG)) + 1):
                    cells[(iy, ix)].append(w)
        self.cells = cells
        self.mm = None; self.cE = self.cN = 0.0
        self.geo = []              # window way -> (e[], n[])
        self.window_ways = 0

    # --- geometry ---
    def to_en(self, lat, lon):
        return (np.asarray(lon) - self.lon0) * self.coslat * self.k, (np.asarray(lat) - self.lat0) * self.k

    def to_ll(self, e, n):
        return self.lat0 + n / self.k, self.lon0 + e / (self.coslat * self.k)

    def _near(self, lat, lon, r):
        dla = r / self.k; dlo = r / (self.k * max(math.cos(math.radians(lat)), 0.1))
        out = set()
        for iy in range(int(math.floor((lat - dla) / CELL_DEG)), int(math.floor((lat + dla) / CELL_DEG)) + 1):
            for ix in range(int(math.floor((lon - dlo) / CELL_DEG)), int(math.floor((lon + dlo) / CELL_DEG)) + 1):
                out.update(self.cells.get((iy, ix), ()))
        return sorted(out)

    def _ensure(self, e, n):
        if self.mm is not None and math.hypot(e - self.cE, n - self.cN) <= self.RECENTER_M:
            return False
        lat, lon = self.to_ll(e, n)
        ids = self._near(lat, lon, self.WINDOW_M)
        m = MapMatcher(); geo = []; nodes = defaultdict(set)
        for j, w in enumerate(ids):
            la, lo, tun, one = self.ways[w]
            we, wn = self.to_en(la, lo)
            m.add_way(we, wn, tun, one); geo.append((np.asarray(we), np.asarray(wn)))
            for a, b in zip(np.round(la * 1e7).astype(np.int64), np.round(lo * 1e7).astype(np.int64)):
                nodes[(int(a), int(b))].add(j)
        for s in nodes.values():
            if len(s) > 1:
                s = sorted(s)
                for i in range(len(s)):
                    for j in range(i + 1, len(s)):
                        m.add_edge(s[i], s[j])
        if self.hmm:
            m.set_hmm(*self.hmm)
        self.mm, self.geo, self.cE, self.cN, self.window_ways = m, geo, e, n, len(ids)
        return True

    # --- greedy (Phase 5) ---
    def match(self, e, n, psi):
        self._ensure(e, n)
        return self.mm.match(e, n, psi)

    # --- live Viterbi ---
    def decode(self, hist):
        """hist: list of (e, n, psi) at 1 Hz, oldest first. -> window way index for the
        newest step (-1 = off-road) and its corridor flag."""
        e, n, psi = map(np.asarray, zip(*hist))
        if self._ensure(e[-1], n[-1]):
            pass                                   # new window: indices refer to it
        r = self.mm.match_seq(e, n, psi)
        return int(r["way"][-1]), bool(r["corridor"][-1])

    def project(self, way, e, n, psi):
        """Snap (e, n) onto one window way -> dict like MapMatcher.match (matched=False if
        the way is gone, too far, or its bearing disagrees with psi)."""
        if way < 0 or way >= len(self.geo):
            return dict(matched=False)
        we, wn = self.geo[way]
        dx, dy = np.diff(we), np.diff(wn)
        L2 = dx * dx + dy * dy + 1e-12
        tt = np.clip(((e - we[:-1]) * dx + (n - wn[:-1]) * dy) / L2, 0.0, 1.0)
        fx, fy = we[:-1] + tt * dx, wn[:-1] + tt * dy
        d2 = (e - fx) ** 2 + (n - fy) ** 2
        i = int(np.argmin(d2))
        brg = math.atan2(dx[i], dy[i])
        db = abs((brg - psi + math.pi) % (2 * math.pi) - math.pi)
        if self.BEARING_GATE < db < math.pi - self.BEARING_GATE:
            return dict(matched=False)
        cross = (fx[i] - e) * -math.cos(brg) + (fy[i] - n) * math.sin(brg)
        return dict(matched=True, foot=(fx[i], fy[i]), bearing=brg, cross=cross, corridor=False)
