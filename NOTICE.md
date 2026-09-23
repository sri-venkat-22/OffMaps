# Third-party data

## OpenStreetMap

The offline Hyderabad map in this repository contains map data
© [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
available under the [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/).

Files derived from OpenStreetMap data:

- `map/hyderabad/tiles.mbtiles`, `map/hyderabad/roads.bin`
- `android/app/src/main/assets/map/tiles.mbtiles`, `android/app/src/main/assets/map/roads.bin`

They were built by `tools/build_map.sh` from the Geofabrik extract
`india/southern-zone`, clipped to the bounding box 78.20,17.15,78.75,17.65.
The tiles carry the attribution "© OpenStreetMap contributors" in their
metadata, and the app's map shows it through MapLibre's attribution control.
