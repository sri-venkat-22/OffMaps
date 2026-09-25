"""Phase 9: the HMM road matcher (road_hmm.py) on a synthetic Y junction, no data needed.

  - class filtering removes service roads from the graph;
  - route distance respects one-ways and the road graph (a parallel street that is
    not connected cannot be reached in one second);
  - confidence is per ROAD: many short OSM segments of one road do not split it;
  - the engine's road update never moves speed or gyro bias with keep=3.
"""
import math, os
import numpy as np
import pytest

from road_hmm import RoadGraph, HMMMatcher

LAT0, LON0 = 17.4, 78.5
K = math.pi / 180 * 6_371_000.0


def _ll(e, n):
    e, n = np.asarray(e, float), np.asarray(n, float)
    return LAT0 + n / K, LON0 + e / (math.cos(math.radians(LAT0)) * K)


def _ways():
    t = np.linspace(0, 1000, 101)                        # a north trunk drawn every 10 m
    trunk = _ll(np.zeros_like(t), t)
    par = _ll(np.full_like(t, 12.0), t)                  # an unconnected parallel street 12 m east
    svc = _ll(np.full_like(t, -6.0), t)                  # a car-park aisle 6 m west
    return [(*trunk, 0, 0, "primary"), (*par, 0, 0, "residential"), (*svc, 0, 0, "service")]


def test_service_roads_are_excluded():
    assert len(RoadGraph(_ways(), LAT0, LON0).len) == 300
    g = RoadGraph(_ways(), LAT0, LON0, exclude=("service",))
    assert len(g.len) == 200 and set(np.round(g.p[:, 0]).astype(int)) == {0, 12}


def test_follows_the_connected_road_with_road_level_confidence():
    g = RoadGraph(_ways(), LAT0, LON0, exclude=("service",))
    hmm = HMMMatcher(g)
    rng = np.random.default_rng(1)
    for s in range(40):                                  # 10 m/s north, 3 m noise, 4 m off the trunk
        if s:
            hmm.add_travel(10.0)
        m = hmm.update((4.0 + rng.normal(0, 1.0), 100.0 + 10 * s), 0.0, 10.0, 5.0)
    assert abs(m["foot"][0]) < 1e-6                      # the trunk, not the street 8 m away
    assert m["conf"] > 0.9 and m["conf"] >= m["seg_conf"]
    assert abs(m["bearing"]) < 1e-6 and m["half_width"] == 6.0


def test_one_way_is_not_matched_against_its_direction():
    w = _ways()
    w[0] = (*w[0][:2], 0, 1, "primary")                  # the trunk is one-way northbound
    g = RoadGraph(w, LAT0, LON0, exclude=("service",))
    hmm = HMMMatcher(g)
    for s in range(10):                                  # driving SOUTH 4 m off the trunk
        if s:
            hmm.add_travel(10.0)
        m = hmm.update((4.0, 800.0 - 10 * s), math.pi, 10.0, 5.0)
    assert m["foot"][0] == pytest.approx(12.0)           # the two-way street, not the one-way


def test_engine_road_update_keeps_speed_and_bias():
    pytest.importorskip("onnxruntime")
    from edge_engine import EdgeEngine
    eng = EdgeEngine(head=None, roads=_ways(), map_mode="hmm",
                     map_opts=dict(exclude=("service",), sigma_max=None, keep=3, heading=True))
    eng.gnss(0.0, LAT0, LON0, 10.0, 0.0)
    eng.f.init(4.0, 300.0, math.radians(3), 10.0)
    eng.masked = True; eng.since_gnss = 99
    for _ in range(3):
        eng.hmm.add_travel(10.0); eng.hmm.update((4.0, 300.0), math.radians(3), 10.0, 5.0)
    v0, bg0 = eng.f.state()[3], eng.f.state()[4]
    eng.hmm_steps = 9
    eng._map_step_hmm(0.1, True)
    st = eng.f.state()
    assert eng.snapped and st[0] < 4.0                   # pulled toward the trunk
    assert st[3] == pytest.approx(v0 + 0.0) and st[4] == bg0


def test_chi2_gate_refuses_a_road_the_filter_disagrees_with():
    """The preset gates each road update on its innovation (DrishtiNav does the same):
    a confident match 18 m (the nearest non-service road) from a tight filter is a wrong
    road, not a correction.
    On train LODO this cut the worst single outage from +287 to +83 points."""
    pytest.importorskip("onnxruntime")
    from edge_engine import EdgeEngine, MAP_HMM
    moved = {}
    for chi2 in (None, MAP_HMM["chi2"]):
        eng = EdgeEngine(head=None, roads=_ways(), map_mode="hmm", map_opts=dict(chi2=chi2, sigma_max=None))
        eng.gnss(0.0, LAT0, LON0, 10.0, 0.0)
        eng.f.init(30.0, 300.0, 0.0, 10.0)               # tight filter, 30 m east of the only road left
        eng.masked = True; eng.since_gnss = 99
        for _ in range(3):
            eng.hmm.add_travel(10.0); eng.hmm.update((30.0, 300.0), 0.0, 10.0, 5.0)
        eng.hmm_steps = 9
        eng._map_step_hmm(0.1, True)
        moved[chi2] = 30.0 - eng.f.state()[0]
    assert moved[None] > 1.0 and moved[MAP_HMM["chi2"]] == 0.0


DATA = os.environ.get("OFFMAPS_IOVNBD", os.path.expanduser("~/OffMaps-data/IO-VNBD-sync"))
ROADS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "map", "coventry", "roads.bin")


@pytest.mark.skipif(not (os.path.isdir(DATA) and os.path.exists(ROADS)), reason="IO-VNBD data / Coventry roads not found")
def test_hmm_preset_on_real_roads_is_not_worse_than_no_map_on_val():
    """Phase 9 gate, real val drives on the real OSM network (README_PHASE9): the
    MAP_HMM preset must not raise the median drift, and must help more outages than
    it hurts. (Val is small: on it the gain is ~0.5 point; train LODO shows -1.9.)"""
    pytest.importorskip("onnxruntime")
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(ROADS), "..", "..", "tools"))
    from osm_layers import read_roads_bin
    from data.iovnbd_sync import load_sync_dir
    from model.train_real import split_drives
    from model.fusion_head import load_head
    from edge_engine import EdgeEngine
    from phase8_eval import live
    roads = read_roads_bin(ROADS, with_class=True)
    val = split_drives(load_sync_dir(DATA, verbose=False))["val"]
    head = load_head(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "fusion_head.pt"))
    off = live(val, lambda: EdgeEngine(head=head, map_mode="off"), durations=(30, 60), raw=True)
    hmm = live(val, lambda: EdgeEngine(head=head, roads=roads, map_mode="hmm"), durations=(30, 60), raw=True)
    d = np.concatenate([np.asarray(hmm[D]) - np.asarray(off[D]) for D in (30, 60)])
    assert len(d) >= 60
    assert np.mean([np.median(hmm[D]) for D in (30, 60)]) <= np.mean([np.median(off[D]) for D in (30, 60)])
    assert (d < -0.5).sum() > (d > 0.5).sum()
