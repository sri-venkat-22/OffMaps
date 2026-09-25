#!/usr/bin/env python3
"""OSM GeoJSON exports -> the two things the app ships (called by build_map.sh).

  1. <layer>.geojsonseq  slim per-layer features for tippecanoe (the basemap):
       roads  {class, name}      water {} (polygons)  waterway {}  green {}  rail {}
     Each feature carries a tippecanoe minzoom so side streets only appear zoomed in.
  2. roads.bin            the matcher's road network, a flat little-endian file the
     phone reads with one ByteBuffer (RoadNetwork.kt). Long ways are split into
     pieces of <= PIECE_PTS vertices so the phone's grid index can hand the native
     matcher only the roads near the car (consecutive pieces share an endpoint, so
     the geometry the matcher projects onto is unchanged).

roads.bin layout (all little-endian):
  char[4] "OMRD" | u32 version=1 | u32 nways | u32 npts
  u32 start[nways+1]    point offset of each way (start[nways] == npts)
  u8  flags[nways]      bit0 tunnel, bit1 oneway (in geometry order), bits2-5 highway
                        class code (HW_CODE; 0 = unknown, files built before it)
  pad to 4 bytes
  i32 lat[npts]         degrees * 1e7
  i32 lon[npts]         degrees * 1e7

ponytail: no custom tiler or PBF parser -- osmium does the OSM side, tippecanoe the
tiles; this only reshapes GeoJSON, so it stays stdlib + numpy.

Usage: osm_layers.py <osmium_export.geojsonseq> <out_dir>
"""
from __future__ import annotations
import json, struct, sys
from pathlib import Path
import numpy as np

MAGIC, VERSION = b"OMRD", 1
PIECE_PTS = 32
SCALE = 1e7

# highway=* -> (style class, tile minzoom). Matcher keeps every class listed here.
ROAD_CLASS = {
    "motorway": ("major", 10), "motorway_link": ("major", 12),
    "trunk": ("major", 10), "trunk_link": ("major", 12),
    "primary": ("main", 10), "primary_link": ("main", 13),
    "secondary": ("main", 11), "secondary_link": ("main", 13),
    "tertiary": ("minor", 12), "tertiary_link": ("minor", 14),
    "unclassified": ("street", 13), "residential": ("street", 13),
    "living_street": ("street", 14), "service": ("service", 15),
}
# highway=* -> flag code (bits 2-5 of the roads.bin flags; 0 = unknown). Append only.
HW_CODE = {h: i + 1 for i, h in enumerate(ROAD_CLASS)}
HW_NAME = {v: k for k, v in HW_CODE.items()}
WATER_AREA = {("natural", "water"), ("landuse", "reservoir"), ("waterway", "riverbank")}
WATERWAY_LINE = {"river", "canal", "stream"}
GREEN = {("leisure", "park"), ("leisure", "garden"), ("landuse", "forest"), ("natural", "wood"),
         ("landuse", "grass"), ("landuse", "recreation_ground"), ("leisure", "golf_course"),
         ("landuse", "cemetery")}
RAIL = {"rail", "light_rail", "subway"}


def _yes(v) -> bool:
    return v in ("yes", "true", "1", 1, True)


def label(tags: dict) -> str | None:
    """Road label. MapLibre cannot shape Indic scripts, so only Latin names are drawn."""
    for k in ("name:en", "name"):
        v = tags.get(k)
        if v and all(ord(ch) < 0x250 for ch in v):
            return v
    return None


def classify(feat: dict):
    """-> (layer, props, minzoom) or None. Pure function of one osmium feature."""
    tags = feat.get("properties") or {}
    gtype = (feat.get("geometry") or {}).get("type")
    lines = gtype in ("LineString", "MultiLineString")
    areas = gtype in ("Polygon", "MultiPolygon")
    hw = tags.get("highway")
    if lines and hw in ROAD_CLASS:
        cls, z = ROAD_CLASS[hw]
        props = {"class": cls}
        name = label(tags)
        if name:
            props["name"] = name
        return "roads", props, z
    if areas and any((k, tags.get(k)) in WATER_AREA for k in ("natural", "landuse", "waterway")):
        return "water", {}, 10
    if lines and tags.get("waterway") in WATERWAY_LINE:
        return "waterway", {}, 11
    if areas and any((k, tags.get(k)) in GREEN for k in ("leisure", "landuse", "natural")):
        return "green", {}, 11
    if lines and tags.get("railway") in RAIL:
        return "rail", {"class": tags["railway"]}, 10
    return None


def road_lines(feat: dict):
    """Yield (lon[], lat[], tunnel, oneway, hw_code) for a road feature (LineString or Multi).
    oneway=-1 (travel against the drawing direction) is reversed, so a oneway way is
    always driven in geometry order."""
    tags = feat.get("properties") or {}
    geom = feat["geometry"]
    parts = [geom["coordinates"]] if geom["type"] == "LineString" else geom["coordinates"]
    tunnel = int(_yes(tags.get("tunnel")) or tags.get("tunnel") == "building_passage")
    rev = tags.get("oneway") == "-1"
    oneway = int(_yes(tags.get("oneway")) or rev)
    code = HW_CODE.get(tags.get("highway"), 0)
    for coords in parts:
        if len(coords) >= 2:
            c = np.asarray(coords, float)
            if rev:
                c = c[::-1]
            yield c[:, 0], c[:, 1], tunnel, oneway, code


def split_piece(lon, lat, n=PIECE_PTS):
    """Split a polyline into consecutive pieces of <= n vertices sharing endpoints."""
    step = n - 1
    for s in range(0, len(lon) - 1, step):
        yield lon[s:s + n], lat[s:s + n]


def write_roads_bin(path, ways):
    """ways: iterable of (lon[], lat[], tunnel, oneway[, hw_code]). Returns (nways, npts)."""
    starts, flags, lats, lons, off = [0], [], [], [], 0
    for lon, lat, tunnel, oneway, *code in ways:
        code = code[0] if code else 0
        for plon, plat in split_piece(lon, lat):
            lats.append(np.round(plat * SCALE).astype("<i4"))
            lons.append(np.round(plon * SCALE).astype("<i4"))
            off += len(plon); starts.append(off)
            flags.append(tunnel | (oneway << 1) | ((code & 15) << 2))
    nw, npts = len(flags), off
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<III", VERSION, nw, npts))
        f.write(np.asarray(starts, "<u4").tobytes())
        f.write(np.asarray(flags, "u1").tobytes())
        f.write(b"\0" * (-nw % 4))
        f.write((np.concatenate(lats) if lats else np.empty(0, "<i4")).tobytes())
        f.write((np.concatenate(lons) if lons else np.empty(0, "<i4")).tobytes())
    return nw, npts


def read_roads_bin(path, with_class=False):
    """Python twin of RoadNetwork.kt's reader: -> list of (lat[], lon[], tunnel, oneway),
    or (lat[], lon[], tunnel, oneway, highway) with with_class (highway None if unknown)."""
    b = Path(path).read_bytes()
    assert b[:4] == MAGIC, "not a roads.bin"
    ver, nw, npts = struct.unpack_from("<III", b, 4)
    assert ver == VERSION
    o = 16
    starts = np.frombuffer(b, "<u4", nw + 1, o); o += 4 * (nw + 1)
    flags = np.frombuffer(b, "u1", nw, o); o += nw + (-nw % 4)
    lat = np.frombuffer(b, "<i4", npts, o) / SCALE; o += 4 * npts
    lon = np.frombuffer(b, "<i4", npts, o) / SCALE
    ways = [(lat[starts[i]:starts[i + 1]], lon[starts[i]:starts[i + 1]],
             int(flags[i] & 1), int(flags[i] >> 1 & 1)) for i in range(nw)]
    if with_class:
        ways = [w + (HW_NAME.get(int(flags[i]) >> 2 & 15),) for i, w in enumerate(ways)]
    return ways


def main(src, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    sinks = {k: open(out / f"{k}.geojsonseq", "w") for k in ("roads", "water", "waterway", "green", "rail")}
    counts = dict.fromkeys(sinks, 0)
    ways = []
    with open(src) as f:
        for line in f:
            line = line.strip().lstrip("\x1e")          # RFC 8142 record separator
            if not line:
                continue
            feat = json.loads(line)
            hit = classify(feat)
            if hit is None:
                continue
            layer, props, z = hit
            sinks[layer].write(json.dumps({"type": "Feature", "geometry": feat["geometry"],
                                           "properties": props, "tippecanoe": {"minzoom": z}},
                                          separators=(",", ":")) + "\n")
            counts[layer] += 1
            if layer == "roads":
                ways.extend(road_lines(feat))
    for s in sinks.values():
        s.close()
    nw, npts = write_roads_bin(out / "roads.bin", ways)
    print("features:", counts)
    print(f"roads.bin: {nw} pieces, {npts} points, {(out / 'roads.bin').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
