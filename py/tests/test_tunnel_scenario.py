"""tunnel_scenario.py: the Mindspace Underpass route transplant is faithful.

  - each route really drives through its own underpass carriageway (and not the other);
  - with GNSS on everywhere the transplanted reference and the fixes agree (< 1 m median);
  - the gyro transplant has the right sign and frame: after a 1 km outage the no-AI stage
    (entry speed + gyro) ends within 15 deg of the route heading, while the same run with
    the route's turn rate put in with the wrong sign ends far off.
Needs the Hyderabad roads.bin and the IO-VNBD data (skipped without them).
"""
import math, os
import numpy as np
import pytest

pytest.importorskip("onnxruntime")
import tunnel_scenario as T                                  # noqa: E402

DATA = os.environ.get("OFFMAPS_IOVNBD", os.path.expanduser("~/OffMaps-data/IO-VNBD-sync"))
pytestmark = pytest.mark.skipif(not (os.path.isdir(DATA) and os.path.exists(T.HYD)),
                                reason="IO-VNBD data / Hyderabad roads.bin not found")


@pytest.fixture(scope="module")
def setup():
    from data.iovnbd_sync import load_sync_dir
    from model.train_real import split_drives
    ways = T.load_ways(); adj, xy = T.graph(ways)
    routes = {k: T.Route(*T.build_route(k, adj, xy)) for k in T.UNDERPASS}
    d = max(split_drives(load_sync_dir(DATA, verbose=False))["val"], key=len)
    return routes, d


def test_routes_go_through_their_own_carriageway(setup):
    routes, _ = setup
    for k, r in routes.items():
        assert 340 < r.s_out - r.s_in < 380, (k, r.s_out - r.s_in)            # one carriageway, ~360 m
        e, n, h, _ = r.at(np.linspace(r.s_in + 20, r.s_out - 20, 20))
        (la0, lo0), (la1, lo1) = T.UNDERPASS[k]
        (e0, n0), (e1, n1) = T.en(la0, lo0), T.en(la1, lo1)
        want = math.atan2(e1 - e0, n1 - n0)
        assert np.all(np.abs((h - want + np.pi) % (2 * np.pi) - np.pi) < math.radians(30)), k   # driven the right way
        assert r.s_in > 4000 and r.s[-1] - r.s_out > 1000, k                   # room for warm-up and exit


def test_transplant_is_consistent_and_signed_right(setup):
    from edge_engine import run
    from ablation import factory
    routes, d = setup
    r = routes["northbound"]
    c0 = r.s_in + (r.s_out - r.s_in) / 2 - T.CORRIDOR_M / 2; c1 = c0 + T.CORRIDOR_M
    idx, cum = T.windows(d, r, c0, c1)
    assert idx, "no cruising stretch found"
    imu, gn, tr = T.transplant(d, r, cum, idx[0], c0)
    a = float(np.interp(c0, tr["s"], tr["t"])); b = float(np.interp(c1, tr["s"], tr["t"]))
    j = int(np.searchsorted(tr["t"], b)) - 1

    def err(o):
        la, lo = T.ll_engine(o, gn); e, n = T.en(la, lo)
        return np.hypot(e - tr["e"], n - tr["n"])

    def herr(o):
        h = np.interp(tr["s"][j], r.g, r.h)
        return abs((o[j, 3] - h + np.pi) % (2 * np.pi) - np.pi)

    on = run(factory("physics", None, None)(), imu, gn, [])
    assert np.median(err(on)[300:]) < 1.0
    ok = run(factory("physics", None, None)(), imu, gn, [(a, b)])
    bad_imu = dict(imu); bad_imu["gyro"] = imu["gyro"].copy()
    kap = np.interp(tr["s"], r.g, r.k) * tr["v"]
    bad_imu["gyro"][:, 2] += 2 * kap                          # the route's turns with the wrong sign
    bad = run(factory("physics", None, None)(), bad_imu, gn, [(a, b)])
    assert herr(ok) < math.radians(15) and herr(bad) > math.radians(45), (math.degrees(herr(ok)), math.degrees(herr(bad)))
