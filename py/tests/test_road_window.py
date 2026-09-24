"""Phase 8d: road snapping on REAL roads -- safeguards + the live Viterbi.

The oracle map (each drive's own GNSS track) said road snapping helps. On the real
OpenStreetMap roads of the IO-VNBD area it made outages far WORSE (val 11.3 % ->
31.7 %, train 17.2 % -> 33.8 % mean drift): at junctions and parallel streets the
greedy matcher snapped to the wrong road and its heading update dragged the
heading up to 180 deg. Pinned here, on a synthetic junction (no data needed):

  - RoadWindow builds junction adjacency from shared vertices, and the live decode
    returns a way index the projection can use;
  - the safeguards: with a side road inside the gate, "unique" refuses to snap and
    "heading=False" never moves the heading, while the default does both.
"""
import math
import numpy as np
import pytest

pytest.importorskip("core_bridge")
from road_window import RoadWindow   # noqa: E402

LAT0, LON0 = 17.4, 78.5
K = math.pi / 180 * 6_371_000.0


def _ll(e, n):
    e, n = np.asarray(e, float), np.asarray(n, float)
    return LAT0 + n / K, LON0 + e / (math.cos(math.radians(LAT0)) * K)


def _ways():
    # a north-going trunk, and at n=500 a branch leaving at 20 deg (a Y junction)
    t = np.linspace(0, 500, 11); b = np.linspace(0, 400, 9)
    trunk = _ll(np.zeros_like(t), t)
    cont = _ll(np.zeros_like(t), 500 + t)
    branch = _ll(b * math.sin(math.radians(20)), 500 + b * math.cos(math.radians(20)))
    return [(la, lo, 0, 0) for la, lo in (trunk, cont, branch)]


def test_adjacency_and_live_decode():
    rw = RoadWindow(_ways(), LAT0, LON0)
    hist = [(0.5, float(n), 0.0) for n in range(300, 320)]
    way, _ = rw.decode(hist)
    assert way == 0                                             # on the trunk
    m = rw.project(way, 3.0, 310.0, 0.0)
    assert m["matched"] and abs(abs(m["cross"]) - 3.0) < 1e-6
    assert not rw.project(way, 3.0, 310.0, math.radians(90))["matched"]   # bearing gate


def test_safeguards_refuse_ambiguous_snaps():
    pytest.importorskip("onnxruntime")
    from edge_engine import EdgeEngine
    old = dict(heading=True, unique=False, gate_deg=45.0)            # the pre-Phase-8 road update
    for opts, may_snap, may_turn in ((old, True, True), ({**old, "unique": True}, False, False),
                                     ({**old, "heading": False}, True, False)):
        eng = EdgeEngine(head=None, roads=_ways(), map_mode="greedy", map_opts=opts)
        eng.gnss(0.0, LAT0, LON0, 10.0, 0.0)
        eng.f.init(6.0, 540.0, math.radians(8), 10.0)        # just past the fork, between the roads
        eng.masked = True; eng.since_gnss = 99; eng.dr_steps = 5
        h0 = eng.f.state()[2]
        eng._map_step()
        assert eng.snapped == may_snap, opts
        assert (abs(eng.f.state()[2] - h0) > 1e-9) == may_turn, opts
