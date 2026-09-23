"""Offline road geometry for the matcher.

geojson_roads(): load OSM roads exported as GeoJSON LineStrings (highway=* ways;
tunnel=yes -> corridor). map_from_path(): build a centerline map from a known
trajectory -- used to validate the matcher (the vehicle WAS on a road, so its
true path is that road's centerline).
"""
from __future__ import annotations
import json, numpy as np
from data.io_vnbd import _lla_to_enu


def geojson_roads(path, lat0=None, lon0=None):
    """Yield (e, n, tunnel, oneway) polylines from a GeoJSON FeatureCollection."""
    with open(path) as f:
        gj = json.load(f)
    out = []
    for ft in gj.get("features", []):
        geom = ft.get("geometry", {})
        if geom.get("type") != "LineString":
            continue
        lon = np.array([c[0] for c in geom["coordinates"]])
        lat = np.array([c[1] for c in geom["coordinates"]])
        e, n = _lla_to_enu(lat, lon)          # NB: uses first vertex as origin per-way
        tags = ft.get("properties", {})
        out.append((e, n, int(tags.get("tunnel") in ("yes", 1, True)),
                    int(tags.get("oneway") in ("yes", 1, True))))
    return out


def map_from_path(e, n, tunnel=0, decimate=5):
    """Centerline polyline from a trajectory (decimated). One way."""
    return [(np.ascontiguousarray(e[::decimate]), np.ascontiguousarray(n[::decimate]), tunnel, 1)]
