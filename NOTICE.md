# Third-party data

The code in this repository is MIT-licensed (see `LICENSE`). The data below is not
covered by that licence and keeps its own terms.

## OpenStreetMap

The offline maps in this repository (Hyderabad, and the Coventry/Warwickshire road
network of the IO-VNBD area) contain map data
© [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
available under the [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/).

Files derived from OpenStreetMap data:

- `map/hyderabad/tiles.mbtiles`, `map/hyderabad/roads.bin`
- `map/coventry/roads.bin` (built by `tools/build_map.sh` from the Geofabrik
  `west-midlands` + `warwickshire` extracts)
- `android/app/src/main/assets/map/tiles.mbtiles`, `android/app/src/main/assets/map/roads.bin`

The Hyderabad files were built by `tools/build_map.sh` from the Geofabrik extract
`india/southern-zone`, clipped to the bounding box 78.20,17.15,78.75,17.65.
The tiles carry the attribution "© OpenStreetMap contributors" in their
metadata, and the app's map shows it through MapLibre's attribution control.

## IO-VNBD

The real-drive results use the IO-VNBD dataset (U. Onyekpe et al., Data in Brief,
2021; github.com/onyekpeu/IO-VNBD), under its own terms. The dataset itself is not
in this repository; REALDATA.md describes how to obtain and resynchronise it.
