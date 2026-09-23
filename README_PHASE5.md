# Phase 5 — Map matching & corridor mode

Roads are polylines (offline OSM). A road pins the LANE (cross-track), never the
distance travelled (along-track) — so in a tunnel the 2-D problem collapses to
1-D progress and the whole error becomes speed accuracy.

## The lazy core insight
The plan's anisotropic pseudo-measurement (σ_cross=1.5, σ_along=50) reduces to a
**single cross-track scalar update**: the perpendicular foot has zero along-track
residual by construction, so σ_along=50 is a no-op. One `idr_update_crosstrack`
+ one `idr_update_heading`. No 2-D matrix, no Viterbi (yet).

## Modules
| file | what |
|---|---|
| `core/map_match.cpp` | candidate search + bearing gate + perpendicular projection + corridor flag |
| `core/eskf.cpp`      | `idr_update_crosstrack` (road-normal only), `idr_update_heading` |
| `py/mapdata.py`      | `geojson_roads()` (OSM export) + `map_from_path()` (centerline for validation) |
| `py/core_bridge.py`  | `MapMatcher`; `eskf_map` model = eskf + matching |
| `tools/build_map.sh` | osmium+tippecanoe → `roads.geojson` (matcher) + `tiles.mbtiles` (app basemap) |

## Corridor mode
`tunnel=yes`, or the sole candidate with no junction in the horizon → 1-D:
cross-track σ 1.5 m → **0.3 m** and heading locked to the road bearing (1° vs 3°).

## Gate (run: `PYTHONPATH=. python3 phase5_gate.py --out ../out`)
On a straight tunnel corridor, 60 s outages:
- cross-track median **24.7 m → 0.00 m** (< 3 m). ✅
- along-track median **111.4 → 111.2 m** — statistically unchanged. ✅

That second line is the real test: a team snapping isotropically would *degrade*
along-track. We don't touch it — the drift that remains in a tunnel is pure
speed accuracy, which is what Phase 4's calibration attacks.

In the mixed-curviness harness: at 60 s `eskf_map` cross-track = 1.9 m vs
`eskf` 39.1 m, along-track unchanged (90.0 vs 93.9 m).

## ponytail notes
- Validation map = the road's true centerline (the vehicle was on the road).
  Real OSM ingestion is `tools/build_map.sh`; the matcher math is identical.
- No HMM/Viterbi. Nearest-road + bearing gate + corridor lock is ~90% of the
  value; add Viterbi only if replays show wrong-road snaps at junctions.
  **Built in Phase 7b** — `mm_match_seq` is an offline Viterbi/HMM decode of the
  whole trajectory (emission + road-adjacency + distance-consistency); on a
  Y-fork it cuts junction wrong-snaps from 4 to 0 (greedy 78% → Viterbi 99% on
  the correct road). The greedy matcher stays the live path. See
  [README_PHASE7.md](README_PHASE7.md).
- `build_map.sh` uses standard tools rather than a reinvented tiler.
