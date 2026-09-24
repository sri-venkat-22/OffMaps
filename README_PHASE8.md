# Phase 8: learned fusion, edge engine, heading aids, real-road map check

This phase covers the pending list, minus the parts that need the Redmi on the road.
Every number is on real IO-VNBD drives or on physics-exact synthetic rigs, and each
table says which. The Redmi items (3, 6, and the phone half of 1 and 8) are built to
run with one command after a drive (§8g).

| # | item | status |
|---|---|---|
| 1 | drift under 10 % | **not reached**; see §8b and the held-out table |
| 2 | a real AI fusion part | **built**: learned fusion head (GRU, speed + noise), §8b |
| 3 | score a real Redmi drive | **ready**: the app records every session; `score_drive.py`, §8g |
| 4 | edge engine CLI + 200 Hz benchmark | **built**: `edge_engine.py`, ~280× real time at 200 Hz, §8a |
| 5 | yaw alignment + magnetometer in the live loop | **built**, §8c |
| 6 | stops / ZUPT decision | **decided on IO-VNBD, confirmed per phone by `score_drive`**, §8e |
| 7 | Viterbi map matcher live | **built and measured**: it hurts on real roads, so it is off by default, §8d |
| 8 | two-wheelers | **heading solved (lean compensation)**; speed untested without a recording, §8f |

## Where item 1 stands (the pass/fail number)

**Not under 10 %.** Real IO-VNBD drives, live loop (the edge engine, which is the phone's
loop), outages 3 min apart, median drift % of distance:

| held-out TEST drives (S3a + Y1), scored once | 10 s | 30 s | 60 s | 120 s | mean |
|---|--:|--:|--:|--:|--:|
| loop before this phase | 14.5 | 17.3 | 17.5 | **16.6** | 16.5 |
| + learned fusion head (shipped) | **13.5** | **15.3** | **17.1** | 19.6 | **16.4** |

On the test drives the head is a **tie on the mean**. It is better at 10–60 s and on the
bad tail (p75 at 30 s: 24.7 vs 31.3 %; at 60 s: 23.4 vs 27.3 %) and worse at 120 s. The
larger held-out estimate (below) predicted a clear win that these two drives did not
show. This is the test split's fifth look overall (four in REALDATA.md /
README_PHASE6), and nothing was chosen from it. `out/phase8/test.json`.

What is between here and 10 %, in order of expected size:
1. **Speed on the target phone.** The IO-VNBD phone's accelerometer barely tracks the car
   (corr 0.02–0.5). The Redmi at 400 Hz with a rigid mount is a different sensor. Record,
   score, then fine-tune SpeedNet and the head on it (§8g). That is the lever left.
2. **More drivers.** Every model here learned from 2 drivers in 4 train drives. The
   between-drive spread (val drives 11 %, train drives 17 %, test 16 %) is as large as
   any method difference measured.
3. Maps and ZUPT were measured and are **not** levers on this data (§8d, §8e).

## 8b: learned fusion head (item 2: the AI fusion part)

`py/model/fusion_head.py`, `nav/FusionHead.kt`. The ESKF's speed measurement **and its noise**
during an outage come from a small GRU (a 5-seed ensemble of 32 units each, about 5k
parameters per member):
- **Context, frozen at outage start:** entry Doppler speed, the Doppler self-cal fit, and
  the last 120 s of GNSS (mean speed, stopped fraction, SpeedNet's residual mean and spread).
- **Each second:** SpeedNet speed (raw and self-cal'd) and its σ, time since outage start,
  and rotation-invariant IMU statistics.
- **Output:** (v, σ), fed to `update_speed(v, σ)`. It replaces the hand rule "hold Doppler
  10 s, then SpeedNet × σ·0.25". This is a learned-R adaptive Kalman update.
- **Loss:** relative cumulative-distance error at 10/30/60/120/180 s (the metric itself),
  plus β-NLL for an honest σ. The ensemble's σ includes member disagreement.

**How it was chosen, without touching val or test:**
- SpeedNet's outputs on its own training drives are in-sample. So four **out-of-fold
  SpeedNets** were trained, one per held-out train drive (`model/oof/`).
- Leave-one-train-drive-out **CV** chose the inputs and the epoch count. Mean held-out
  distance error: SpeedNet 14.8 %; head fed in-sample ≈ 10.5 %; head fed out-of-fold
  ≈ 9.5 % (`model/fusion_head_cv.json`). Chosen: OOF inputs, 30 epochs, 5 seeds.
- The **leave-one-drive-out live loop** is each train drive run through the edge engine
  with a SpeedNet and a head that never saw it. It covers about 9.5 h and 1,400 outages:

| live loop, held-out train drives | 10 s | 30 s | 60 s | 120 s | mean | p75 60 s |
|---|--:|--:|--:|--:|--:|--:|
| before | 13.8 | 20.2 | 20.3 | 19.9 | 18.5 | 34.0 |
| **fusion head** | **11.8** | **13.2** | **15.8** | **17.2** | **14.5** | **24.6** |
| hold 10 s, then head | 13.8 | 14.0 | 16.0 | 17.5 | 15.3 | 25.9 |

- **Val (2 drives, 170 outages), the check:** before 10.1 / 19.5 / 14.7 / 12.0 (mean 14.1),
  head 14.1 / 13.0 / 16.9 / 16.5 (mean 15.1). The medians **disagreed**; the tail improved
  (p75 21–23 % vs 27–35 %).
- **Decision, before test:** ship the head, on the larger held-out evidence. Test then came
  out as the tie shown above.

On the phone: `assets/fusion_head.json` (the same weights, `tests/test_fusion_head.py`) is run
by `FusionHead.kt`, a plain-Kotlin GRU. `tests/test_kotlin_ports.py` checks it against the
torch head to 1e-5 over a 40 s outage. The profile's `live.fusion_head` switches it on.
Remove that key and the app falls back to the old rule.

Reproduce:

    PYTHONPATH=. python3 -m model.fusion_head --data ~/OffMaps-data/IO-VNBD-sync --oof model/oof --cv
    PYTHONPATH=. python3 -m model.fusion_head --data ~/OffMaps-data/IO-VNBD-sync --oof model/oof --fixed-epochs 30 --ensemble 5
    PYTHONPATH=. python3 -m model.fusion_head --data ~/OffMaps-data/IO-VNBD-sync --oof model/oof --fold-heads model/oof --fixed-epochs 30 --ensemble 2
    PYTHONPATH=. python3 phase8_eval.py --lodo --configs shipped,head,ho_head
    PYTHONPATH=. python3 phase8_eval.py --split val --configs shipped,head


## 8a: edge engine (`py/edge_engine.py`, `py/edge_bench.py`)

The phone's live loop (FusionEngine.kt) as a streaming CLI that reads raw IMU + GNSS
CSV at **any rate**:

    cd py
    PYTHONPATH=. python3 -m edge_engine --dir drive/ --out track.csv                 # imu.csv + gnss.csv
    PYTHONPATH=. python3 -m edge_engine --csv fog_200hz.csv --out track.csv --outage 300:360
    PYTHONPATH=. python3 edge_bench.py                                               # throughput

- **Every IMU sample:** gravity EMAs, then the yaw rate about true vertical (§8c), then
  `idr_predict` at the input rate. A 200 Hz FOG gets 200 Hz heading integration, not a
  decimated one.
- **Every 100 ms** (the mean of that bin's samples, anti-aliased):
  - core `aln` re-mount CUSUM
  - Level (30 s gravity) into the leveled SpeedNet window
  - YawAlign
  - map matching
- **Every 1 s:** SpeedNet (the phone's ONNX graph + profile), self-cal, and the learned head
  or the hand-over rule while dead-reckoning.
- **Every fix:** trust, χ² spoof gate, position + velocity update. Masked fixes (`masked`
  column or `--outage`) feed nothing.
- **Output:** one row **per IMU sample**: lat/lon, e/n, heading, speed, σ_e/σ_n, dead-reckoning flag.

Input: the phone-logger schema (`t_ns, ax..gz, [mx my mz]` + `t_ns, lat, lon, speed, bearing,
…`), any `t`/`time` column, or one merged CSV whose GNSS columns are filled only on fix rows.
There are unit flags (`--gyro-deg`, `--acc-g`, `--kmph`).

**Throughput** (`out/edge_bench.json`: 30 min synthetic drives, one thread, CSV parse +
fusion + CSV write, measured with other jobs running, so conservative):

| IMU rate | samples | end-to-end | real-time factor |
|--:|--:|--:|--:|
| 10 Hz | 18 k | 13 k/s | 1316× |
| 100 Hz | 180 k | 48 k/s | 483× |
| **200 Hz (FOG)** | 360 k | **55 k/s** | **277×** |
| 400 Hz | 720 k | 77 k/s | 194× |

**Same loop as the phone:** at 10 Hz on the real val drives the engine gives 10.8 % median
60 s drift, against 12.9 % for the validated mirror `phase6_check.run`, on the same windows.
The gap comes from the engine's NN window being causal. The mirror's window reaches
1 s into the future, which the phone's cannot.

Pinned by `tests/test_edge_engine.py`:
- a 200 Hz FOG drive tracks, dead-reckons and recovers;
- the CLI runs at ≥ 20× a 200 Hz stream;
- the merged CSV gives the same result as two files;
- the 10 Hz run on real val stays within 4 points of the mirror.

## 8c: heading aids (`py/heading_aids.py`, `nav/HeadingAids.kt`)

**Yaw rate about true vertical.** In any sustained turn the specific force tilts toward
the turn centre by φ = atan(v·ψ̇/g). The app projected the gyro on a 0.5 s gravity EMA,
which follows that tilt, so every turn was under-read by cos φ: 4 % at 0.3 g. The host
validation never saw this, because it used the loader's fixed gyro axis. There are
three modes:
- `slow` (30 s gravity, the leveling frame's up) is exact for a car and needs nothing else.
- `coord` tilts the fast up back by φ, using the filter speed and the forward axis. A
  two-wheeler leans with the phone, so it needs this mode.
- `fast` is the old behaviour.

**GNSS-aided yaw alignment** (`YawAlign`): a forgetting least-squares fit of Doppler dv/dt
against the leveled horizontal accel gives the vehicle's forward axis, sign included. It
is the causal version of the loader's per-segment fit. It feeds the `coord` yaw, the
magnetometer seed and the curvature update's lateral accel. It resets on a re-mount.

**Magnetometer seed.** When the first fix has no usable course (parked: GNSS bearing is
noise below 1 m/s), the heading is seeded from the tilt-compensated magnetometer. It uses
the forward axis from YawAlign, or a mount guess, and declination from Android's
`GeomagneticField`. The seed carries σ = 20°. Without a magnetometer the heading is marked
unknown (σ = π). New core call **`idr_set_heading_sigma`**, with its oracle twin, ctypes
and JNI bindings, and parity at 1e-12. Before it, a missing course was seeded as 0° with
the filter believing it to 1°, and GNSS course could hardly correct it.

Gate (`phase8_heading_gate.py`, physics-exact rigs: dash mount rotated 12°/8°, MEMS gyro,
100 Hz, 18 outages of 60 s per mode):

| median heading error at outage end | car | two-wheeler (lean ≤ 31°) |
|---|--:|--:|
| `fast` (old app) | 19.9° | 18.9° |
| `slow` | **2.5°** | 14.3° |
| `coord` | 2.8° | **5.4°** |

Parked start at 8 random headings: magnetometer seed median **17°** (max 34°), against
82° with no seed. The synthetic turns reach 0.6 g, which is harsher than typical urban
driving, so `fast` is overstated here. On IO-VNBD, `fast` and `slow` scored the same;
that loader's accelerometer and gyro frames are not jointly rotated, so it cannot show
the effect. `coord` needs a sane speed: with a badly wrong dead-reckoned speed it
mis-estimates the lean.

Pinned by `tests/test_phase8_heading.py`, `tests/test_eskf_core_parity.py`, and
`tests/test_kotlin_ports.py` (HeadingAids.kt == heading_aids.py to 1e-12).

## 8d: road snapping on REAL roads, and the live Viterbi

Every earlier map number used an **oracle map**: each drive's own GNSS track. This phase
built the real OpenStreetMap network of the IO-VNBD area. That needed the Geofabrik
West Midlands and Warwickshire extracts merged: the "west-midlands" file stops at
Coventry, and Rugby is in Warwickshire.

    osmium merge west-midlands-latest.osm.pbf warwickshire-latest.osm.pbf -o merged.osm.pbf
    tools/build_map.sh merged.osm.pbf map/coventry -1.63,52.34,-1.20,52.58

38,754 ways. The drives' GNSS tracks lie a median 2 m (p90 5 m) from the nearest road.
Live loop, mean of the median drift over 30/60/120 s outages:

| road snapping | val (2 drives) | train (8 segments) |
|---|--:|--:|
| none | **11.3 %** | **17.2 %** |
| greedy + heading update (the app until now) | 31.7 % | 33.8 % |
| greedy, cross-track only, unambiguous only, 15° gate | 11.2 % | 19.9 % |
| **live Viterbi** (fixed-lag, 20 s), same safeguards | 13.6 % | 26.1 % |

**On real roads, snapping makes outages worse.** Near junctions and parallel streets a
dead-reckoned position that is tens of metres off along-track snaps to the wrong road. The
heading update then drags the heading, by up to 180° in the traces. With the safeguards
it becomes neutral at best. It still cannot fix the along-track error that dominates. The
live Viterbi (`road_window.py` / `RoadMatcher.decode`: `mm_match_seq` over the last 20 one-second
positions, road adjacency from shared vertices) is built and runs in the loop. It was
worse than greedy on both sets, because the decoder commits to a road sequence from
positions that are already off along-track.

Shipped decision (profile `live` block): road snapping is **off by default**. When switched
on it uses the safe settings: no heading update, unambiguous matches only, 15° gate.
Viterbi stays selectable (`mm_viterbi`). Pinned by `tests/test_road_window.py`.

## 8e: stops and ZUPT

SpeedNet sees 95–97 % of real stops (< 0.2 m/s). At 10 Hz it also calls "stopped" in
31–48 % of slow-traffic seconds (1–4 m/s) and in 1–6 % of moving seconds. The accelerometer
and gyro statistics of a stopped car and of slow traffic overlap on this data. A ZUPT also
takes the yaw rate as gyro bias (ZARU), so a false stop corrupts heading as well as speed.

| live loop, mean drift 30/60/120 s | val | train |
|---|--:|--:|
| ZUPT off (shipped) | 11.3 % | 17.2 % |
| naive ZUPT: SpeedNet < 0.5 m/s | 33.6 % | 35.9 % |
| **strict**: SpeedNet < 0.3 m/s for 3 s and max \|gyro\| < 0.03 rad/s | 11.9 % | 16.5 % |

**Decision:** the naive rule must never ship. The strict detector is harmless on IO-VNBD
and built (`zupt_strict` in the profile, FusionEngine + edge engine). It stays **off**
until a stationary test on the Redmi shows it reduces drift while parked. `score_drive.py`
measures exactly that (§8g) and prints the recommendation. The IO-VNBD phone is not the
Redmi, and idle vibration differs by phone and car.

## 8f: two-wheelers

Nothing was trained or tested on a two-wheeler before, and there is still no two-wheeler
recording. What changes physically, and what is now handled:

- **Lean.** The phone leans with the bike: 31° at 0.6 g in the rig. Gravity-projected yaw
  under-reads by cos(lean), even with slow gravity. The `coord` yaw rate cuts the outage-end
  heading error from 14–19° to 5.4° (§8c). The app has a Car / Two-wheeler toggle, and the
  edge engine and scorer take `--vehicle two_wheeler`.
- **Speed.** SpeedNet and the fusion head are car models. The Doppler self-calibration
  corrects scale and offset online, but the vibration signature of a scooter is different.
  **No accuracy claim is made for two-wheeler speed** until a recording exists:
  `score_drive.py --vehicle two_wheeler` on a Redmi scooter ride answers it.

## 8g: the Redmi drive (items 3 and 6, and the phone half of 1 and 8)

The nav app now **records every navigation session** (`DriveRecorder.kt`): raw IMU, gyro and
magnetometer at the sensor rate, and every GNSS fix, in the logger schema, with the Outage
Simulator state. One command replays it through the same loop on the host:

    adb pull /sdcard/Android/data/com.offmaps/files/drive_<ms> drives/
    cd py && PYTHONPATH=. python3 -m score_drive --dir ../drives/drive_<ms> [--roads ../map/hyderabad/roads.bin] [--vehicle two_wheeler]

It writes `out/phone/<drive>/REPORT.md` with three sections:
1. **Sanity:** IMU and GNSS rates, C/N0, satellites, heading against GNSS course (a mirrored
   yaw shows up here), and the heading seed source.
2. **Drift on this phone:** simulated 10/30/60/120 s outages, 3 min apart, for the shipped
   loop, the fusion head and road snapping, scored against the phone's own GNSS.
3. **Stops:** every parked span of 20 s or more, replayed with ZUPT off and with the strict
   detector, a moving-outage no-harm check, and an **on/off recommendation**.

Protocol for the first drive (about 40 min):
1. Mount the phone rigidly (dash or vent mount), open OffMaps, press Start navigation, and
   wait for GNSS LOCK.
2. Drive normally: city and arterial roads, with turns and some traffic stops.
3. Park with the engine running for 2 minutes, twice. This is the ZUPT test.
4. Leave the Outage Simulator alone; the scorer simulates the outages.
5. Stop navigation, then `adb pull`.

For two-wheelers, repeat with the phone on a handlebar mount and the Two-wheeler toggle on.
If the drive shows the NN is off-distribution, fine-tune on it: `model/train_real.py` takes
phone drives through `data/phone_log.py`, and the fusion head retrains in minutes.

## Files
| file | what |
|---|---|
| `py/edge_engine.py`, `py/edge_bench.py` | edge CLI + throughput benchmark |
| `py/model/fusion_head.py` | learned fusion head: features, GRU, CV, fold heads, ensemble, JSON export |
| `py/model/oof/` | out-of-fold SpeedNets + fold heads (training inputs and leave-one-drive-out eval) |
| `py/phase8_eval.py` | live-loop evaluation (val / test / leave-one-drive-out) |
| `py/heading_aids.py`, `py/phase8_heading_gate.py` | yaw modes, YawAlign, magnetometer seed; their gate |
| `py/data/synth_rig.py` | physics-exact device-frame IMU + magnetometer rig (car / two-wheeler, any rate) |
| `py/road_window.py` | road window + live fixed-lag Viterbi (host twin of RoadMatcher.kt) |
| `py/score_drive.py` | one-command scorer for a phone recording |
| `core/eskf.cpp` `idr_set_heading_sigma` | honest heading prior after a course-less seed |
| `nav/FusionHead.kt`, `FusionHeadAsset.kt` | the head on the phone (pure Kotlin GRU, checked against Python) |
| `nav/HeadingAids.kt` | yaw modes, magnetometer heading, YawAlign |
| `nav/RoadMatcher.kt` | + live Viterbi (decode/project, junction adjacency) |
| `nav/DriveRecorder.kt` | records every session for `score_drive.py` |
| `map/coventry/` | OSM roads of the IO-VNBD area (build inputs are git-ignored) |
