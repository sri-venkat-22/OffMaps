#!/bin/sh
# Offline map pipeline for a target area. Produces, in out_dir:
#   tiles.mbtiles  -> offline vector basemap for the app (MapLibre): roads, water, green, rail
#   roads.bin      -> the matcher's road network (RoadNetwork.kt / tools/osm_layers.py)
#   *.geojsonseq   -> the slim per-layer GeoJSON both were built from
#
# ponytail: no reinvented tile generator -- this is the standard osmium+tippecanoe
# pipeline. Install once: brew install osmium-tool tippecanoe   (or apt equivalents)
#
# Usage: tools/build_map.sh region.osm.pbf out_dir [minlon,minlat,maxlon,maxlat]
#   Hyderabad (ORR + margin):
#   tools/build_map.sh southern-zone-latest.osm.pbf map/hyderabad 78.20,17.15,78.75,17.65
#   then ship both in the app:
#   cp map/hyderabad/tiles.mbtiles map/hyderabad/roads.bin android/app/src/main/assets/map/
set -e
PBF="$1"; OUT="${2:-map}"; BBOX="$3"
[ -f "$PBF" ] || { echo "usage: build_map.sh region.osm.pbf [out_dir] [bbox]"; exit 1; }
for t in osmium tippecanoe python3; do command -v $t >/dev/null || { echo "missing $t (brew install osmium-tool tippecanoe)"; exit 1; }; done
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"

# 0) clip to the area (smart strategy keeps multipolygons like lakes complete)
SRC="$PBF"
if [ -n "$BBOX" ]; then
  osmium extract -b "$BBOX" -s smart "$PBF" -o "$OUT/area.osm.pbf" --overwrite
  SRC="$OUT/area.osm.pbf"
  echo "clipped $BBOX -> $SRC"
fi

# 1) only what the basemap + matcher use (referenced nodes/members come along)
osmium tags-filter "$SRC" \
  w/highway=motorway,motorway_link,trunk,trunk_link,primary,primary_link,secondary,secondary_link,tertiary,tertiary_link,unclassified,residential,living_street,service \
  wr/natural=water,wood wr/landuse=reservoir,forest,grass,recreation_ground,cemetery \
  wr/waterway=riverbank w/waterway=river,canal,stream \
  wr/leisure=park,garden,golf_course w/railway=rail,light_rail,subway \
  -o "$OUT/layers.osm.pbf" --overwrite
osmium export "$OUT/layers.osm.pbf" -f geojsonseq -o "$OUT/layers.geojsonseq" --overwrite

# 2) reshape: slim per-layer features (+ tile minzoom) and the matcher's roads.bin
python3 "$HERE/osm_layers.py" "$OUT/layers.geojsonseq" "$OUT"

# 3) offline basemap tiles (zoom 10-16 is plenty for road-level nav; MapLibre overzooms past 16)
tippecanoe -o "$OUT/tiles.mbtiles" -Z10 -z16 -n "OffMaps" --force \
  --attribution "© OpenStreetMap contributors" \
  --drop-densest-as-needed --extend-zooms-if-still-dropping \
  -L roads:"$OUT/roads.geojsonseq" -L water:"$OUT/water.geojsonseq" \
  -L waterway:"$OUT/waterway.geojsonseq" -L green:"$OUT/green.geojsonseq" \
  -L rail:"$OUT/rail.geojsonseq"
ls -lh "$OUT/tiles.mbtiles" "$OUT/roads.bin"
