# Phase 9: map matching on real roads, with DrishtiNav's gating

Phase 8d found that road snapping on the real OpenStreetMap network makes outages worse,
so it shipped off. DrishtiNav (github.com/Abhishek222983101/drishtinav, the same PS)
reports the opposite on the same dataset: its HMM map matching cut their 60 s median
from 15.6 to 10.9 %. This phase ports the parts of their matcher that differ from ours
and measures each one in our live loop (the edge engine = the phone's loop, shipped
profile + fusion head). **The test split was not touched.**

| # | what differed (theirs vs ours) | built here |
|---|---|---|
| 1 | matcher road classes: drivable only vs **all incl. `service`** (half of our pieces: 19,603 / 38,754) | highway class in `roads.bin` flag bits 2–5; `exclude=("service",)` |
| 2 | snap only while the **filter's position σ ≤ 25 m** vs any match within 25 m | `sigma_max` |
| 3 | HMM forward filter, **route-distance transitions** (Dijkstra, one-way aware), posterior ≥ 0.9 vs nearest road / fixed-lag Viterbi | `py/road_hmm.py`, `map_mode="hmm"` |
| 4 | cross-track σ = 0.6 × road half-width (by class), heading 6° and not within 8 m of a vertex or on segments < 25 m, vs 0.3 m in a "corridor" / 1–3° | preset values |
| 5 | road updates move position + heading only vs speed frozen but **gyro bias free** | core `idr_set_map_keep_speed` is now a mask: 1 speed, 2 gyro bias, 3 both |
| 6 | **χ² innovation gate** on every road update vs none | `chi2=6.63` (1 dof, 99 %) |

Two things had to change to make their matcher work on our data:
- **Confidence per road, not per segment.** OSM draws a road as many short
  vertex-to-vertex segments. Near a vertex two segments of the same road split the
  posterior, so their "posterior ≥ 0.9" was met in only 27 % of GNSS-aided seconds. Summing
  candidates on the same line (feet within 5 m, direction within 30°) gives 48 %. A
  parallel street is further than 5 m away, so it still counts against the match.
- **Their 25 m σ gate is only right by accident.** Our filter's GNSS-aided σ is about 9 m
  (theirs about 2.5 m). The sweep below shows 25 m is still the best gate for us, and that
  removing it is harmful.

Also fixed along the way: `oneway=-1` ways were stored as ordinary one-ways in drawing
order, i.e. backwards. They are now reversed when `roads.bin` is built.

## Results

Same protocol and metric as Phase 8 (outages 10/30/60/120 s with 3 min of GNSS before
each, median drift % of distance). Two sets:
- **val**: 2 drives, about 1 h, 33–50 outages per duration.
- **train LODO**: every train drive run with a SpeedNet and fusion head that never saw
  it. About 9.5 h, 294–392 outages per duration.

Every outage window is identical across configs, so the configs are compared outage by
outage (`phase9_map_eval.py --paired`).

**Train LODO** (`out/phase9/lodo.json`), median drift %:

| config | 10 s | 30 s | 60 s | 120 s | mean | < 10 % at 30 / 60 s |
|---|--:|--:|--:|--:|--:|--:|
| no map (shipped) | 11.7 | 12.5 | 14.1 | 15.9 | 13.6 | 33 / 35 % |
| greedy, as the app's toggle | 13.2 | 13.0 | 14.3 | 15.7 | 14.0 | 33 / 33 % |
| greedy + no service + σ ≤ 25 m | 13.5 | 13.3 | 14.3 | 15.4 | 14.1 | 32 / 36 % |
| HMM, all road classes | 11.1 | 9.9 | 11.5 | 14.8 | 11.8 | 51 / 46 % |
| HMM, no service | 11.3 | 10.1 | 11.5 | 14.0 | 11.7 | 49 / 46 % |
| HMM, no service, keep speed + bias | 11.3 | 10.1 | 11.6 | 14.0 | 11.8 | 49 / 47 % |
| **HMM preset (+ χ² gate) = `MAP_HMM`** | **11.2** | **10.1** | **11.5** | **14.0** | **11.7** | **49 / 47 %** |

Paired against no map, preset, per-outage median difference with a 90 % bootstrap interval:
- 10 s: +0.0 [−0.0, +0.1]
- 30 s: −2.3 [−2.5, −1.6], 72 % of outages better
- 60 s: −1.1 [−1.6, −0.8], 66 % better
- 120 s: −0.5 [−0.8, −0.3], 64 % better
- Mean of the medians: −1.8 points.

The χ² gate adds nothing to the median, but it cuts the tail:

| | worst single outage vs no map | worst at 120 s | outages > 20 points worse |
|---|--:|--:|--:|
| without the gate | +287 points (10 s: 5.6 → 293 %) | +73 | 2.4 % |
| with the gate | +83 | +27 | 1.7 % |

**Val** (`out/phase9/val.json`): the same direction, too small to resolve. No map 15.0 %;
preset 14.2 % (60 s: 16.4 → 13.5). Paired medians differ by 0.0–0.2 points and every
interval includes 0. The HMM helps 53–64 % of outages; greedy is worse more often than it
is better. The σ-gate sweep on val (15 / 20 / 25 / 40 / 60 / none: 15.1 / 15.2 / 14.3 /
16.3 / 16.7 / 16.7) is noisy below 25 m and clearly harmful above it.

**What did the work:** the HMM with its gating (posterior + filter σ + χ²). Removing
service roads and freezing the gyro bias are neutral on this data. They stay in the preset
because they cost nothing and they close a failure mode (car parks; a false bias that keeps
turning the car after a wrong snap). The greedy matcher cannot be rescued by the same
safeguards (14.1 %).

**Why the gain is capped:** on val, at outage end, the median along-track error is 10.5–14.1 %
of distance, while cross-track is only 3.7–5.8 %. The HMM cuts cross-track (30 s: 3.7 → 2.0 %;
60 s: 4.3 → 3.1 %) but cannot see along-track error. By 120 s the filter σ has passed
25 m and the gate is closed. Speed (along-track) is still the lever, as in Phase 8.

## Status

- **Edge engine / host:** `--map-mode hmm` (preset `edge_engine.MAP_HMM`) is available. The
  CLI default is still `greedy` when `--roads` is given, the phone's twin.
- **Phone: not ported.** The app still has the greedy matcher, off by default, which
  measures as harmful (14.0 vs 13.6 %) on train. Shipping this means porting `road_hmm.py`
  (≈200 lines: graph from `roads.bin`, bounded Dijkstra, forward filter) to Kotlin, calling
  `setMapKeepSpeed(3)` through JNI (`idrSetMapKeepSpeed` takes a boolean today), and
  **rebuilding `map/hyderabad/roads.bin`** so it carries the road class.
- **Next lever from the map:** along-track fixes at turns. When the gyro sees a turn and
  the HMM is confident, the turn's position on the road pins the along-track error. It is
  not built.

## Reproduce

    cd py
    PYTHONPATH=. python3 phase9_map_eval.py --split val --out ../out/phase9/val.json
    PYTHONPATH=. python3 phase9_map_eval.py --lodo --configs off,hmm_all,hmm,hmm_keep,hmm_gate --out ../out/phase9/lodo.json
    PYTHONPATH=. python3 phase9_map_eval.py --paired ../out/phase9/lodo.json
    python3 ../tools/osm_layers.py ../map/coventry/layers.geojsonseq ../map/coventry   # roads.bin with road classes

Pinned by:
- `tests/test_road_hmm.py`: class filter; road-level confidence; one-way; speed and bias
  untouched; χ² refusal; on real val (skipped without data), the preset is not worse than
  no map and helps more outages than it hurts.
- `tests/test_eskf_core_parity.py`: keep mask 3, C core == oracle.
- `tests/test_osm_layers.py`: class bits round-trip, old readers unaffected; `oneway=-1`.
