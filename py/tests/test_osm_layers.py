"""tools/osm_layers.py: the OSM -> app reshaper. Locks the roads.bin byte layout that
RoadNetwork.kt memory-maps (read_roads_bin is its Python twin), the way splitting
that feeds the phone's grid index, and the layer/label rules the style depends on.
"""
import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "osm_layers", Path(__file__).resolve().parents[2] / "tools" / "osm_layers.py")
ol = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ol)


def _feat(gtype, coords, **tags):
    return {"type": "Feature", "geometry": {"type": gtype, "coordinates": coords}, "properties": tags}


def test_split_pieces_share_endpoints_and_cover_the_line():
    lon = np.linspace(78.40, 78.50, 100); lat = np.linspace(17.30, 17.40, 100)
    pieces = list(ol.split_piece(lon, lat, n=32))
    assert all(len(p[0]) <= 32 for p in pieces)
    for (a, _), (b, _) in zip(pieces, pieces[1:]):
        assert a[-1] == b[0]                       # consecutive pieces join -> same geometry
    joined = np.concatenate([pieces[0][0]] + [p[0][1:] for p in pieces[1:]])
    assert np.array_equal(joined, lon)


def test_roads_bin_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    ways = [(78.4 + rng.random(k) * 0.1, 17.3 + rng.random(k) * 0.1, t, o)
            for k, t, o in [(2, 0, 0), (70, 1, 0), (5, 0, 1)]]   # 70 pts -> split into 3 pieces
    nw, npts = ol.write_roads_bin(tmp_path / "roads.bin", ways)
    back = ol.read_roads_bin(tmp_path / "roads.bin")
    assert len(back) == nw == 1 + 3 + 1
    assert sum(len(w[0]) for w in back) == npts
    assert [w[2:] for w in back] == [(0, 0), (1, 0), (1, 0), (1, 0), (0, 1)]
    # 1e-7 deg quantisation ~ 1 cm
    assert np.allclose(back[0][0], ways[0][1], atol=1e-7) and np.allclose(back[0][1], ways[0][0], atol=1e-7)
    assert np.allclose(np.concatenate([back[1][1], back[2][1][1:], back[3][1][1:]]), ways[1][0], atol=1e-7)


def test_roads_bin_header_padding(tmp_path):
    # nways not a multiple of 4 -> flags padded so the i32 arrays stay aligned
    ol.write_roads_bin(tmp_path / "r.bin", [(np.array([78.0, 78.1]), np.array([17.0, 17.1]), 0, 0)])
    b = (tmp_path / "r.bin").read_bytes()
    assert b[:4] == b"OMRD" and len(b) == 16 + 4 * 2 + 4 + 4 * 2 * 2


def test_classify_layers_and_labels():
    road = ol.classify(_feat("LineString", [[0, 0], [1, 1]], highway="primary", name="Road No. 12"))
    assert road == ("roads", {"class": "main", "name": "Road No. 12"}, 10)
    telugu_only = ol.classify(_feat("LineString", [[0, 0], [1, 1]], highway="residential", name="రోడ్"))
    assert telugu_only == ("roads", {"class": "street"}, 13)          # no Indic shaping in MapLibre
    en = ol.classify(_feat("LineString", [[0, 0], [1, 1]], highway="trunk", name="రోడ్", **{"name:en": "NH 44"}))
    assert en[1]["name"] == "NH 44"
    assert ol.classify(_feat("Polygon", [[[0, 0], [1, 0], [1, 1], [0, 0]]], natural="water"))[0] == "water"
    assert ol.classify(_feat("Polygon", [[[0, 0], [1, 0], [1, 1], [0, 0]]], leisure="park"))[0] == "green"
    assert ol.classify(_feat("LineString", [[0, 0], [1, 1]], railway="subway"))[:2] == ("rail", {"class": "subway"})
    # closed highway ways also come out of osmium as polygons: those must not become roads
    assert ol.classify(_feat("Polygon", [[[0, 0], [1, 0], [1, 1], [0, 0]]], highway="primary")) is None
    assert ol.classify(_feat("LineString", [[0, 0], [1, 1]], highway="footway")) is None
