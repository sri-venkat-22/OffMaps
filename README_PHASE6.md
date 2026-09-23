# Phase 6 — the real on-device nav app

Everything before this ran in the harness; nothing ran the live fusion on a
phone. The Phase-1 app was a **logger** — it recorded CSVs and did no on-device
filtering. Phase 6 is the unbuilt piece, now built: **embed libidr via NDK/JNI,
run the ESKF live, honour the outage flag against the live filter, render
position.**

## The one design decision
`libidr` stays the single source of truth. The JNI layer
(`android/app/src/main/cpp/idr_jni.cpp`) is a **thin marshaller** — it exposes the
exact C ABI that `py/core_bridge.py` reaches through ctypes and `tools/replay.cpp`
links directly, compiled from the **unmodified `core/*.cpp`** (CMake pulls the repo
sources; no fork, no copy). The **live fusion loop lives in Kotlin**
(`FusionEngine`), not C++, because the loop that was *validated* is
`core_bridge.py::_run` — orchestration in the binding layer, primitives in the
core. So Phase 6 is: the same core object + the same loop shape, now fed by live
Android sensors instead of a replayed CSV.

```
Android sensors ─▶ FusionEngine (Kotlin)         ┌─ nn_real.onnx + .profile.json (ONNX Runtime)
  accel/gyro 100+Hz  ├ predict / ZUPT / curvature ┤
  GNSS ~1Hz          └ GNSS gated by outage flag ─┴─ IdrNative (JNI) ─▶ libidr core/*.cpp
                                                                          (idr/spc/aln/gq/mm)
```

## Cadence (the validated `eskf` model while dead-reckoning; see 6c)
| when | action |
|---|---|
| every 10 Hz IMU step | `idr_predict(dt, gyro_z)`; if the profile has `curv` and we are dead-reckoning, curvature update when \|ψ̇\|>10°/s |
| every 1 s | SpeedNet → (v, σ), always (self-cal regressor). **Dead-reckoning only:** self-cal applied if 1/3 ≤ k ≤ 3; v < `zupt_v` → ZUPT, else speed update with σ·`sig_scale` |
| every GNSS fix, **unmasked** | `gq_trust`→R; spoof = position χ² only; if not spoofed: `idr_update_gnss_pos`/`_vel`, `spc_push`+`spc_fit`; 5 rejected in a row → re-seed from GNSS |
| every GNSS fix, **masked (outage)** | **nothing fed to the filter → pure dead-reckoning** |

"Dead-reckoning" = Outage Simulator on, or no GNSS applied for 2 s. Noise, `zupt_v`,
`curv`, `sig_scale` and the speed calibration come from the model's profile.

`gyro_z` is the gravity-projected yaw rate and `a_lat` the device-y accel — the
same inputs `phone_log.py` feeds the validated model. Mount alignment (`aln_*`)
runs alongside for re-mount detection/status but does not yet rotate the fusion
inputs (a wired, not-yet-defaulted improvement).

## Honouring the outage flag (the demo)
The **Outage Simulator** toggle sets `masked`. GNSS keeps arriving and is drawn as
grey ground truth, but while masked the live filter is fed no GNSS and
dead-reckons — the blue track visibly peels away from grey (orange segments), and
the readout shows the metres of drift. This is exactly what `tools/replay.cpp`
does with `mask==0`, now against the *live* filter, as promised in
[android/README.md](android/README.md) and `LoggerActivity.kt:22`.

## Files
| file | what |
|---|---|
| `android/app/src/main/cpp/idr_jni.cpp` | JNI bridge → the 32 libidr entry points |
| `android/app/src/main/cpp/CMakeLists.txt` | compiles `core/*.cpp` + bridge → `libidrjni.so` |
| `android/…/nav/IdrNative.kt` | `external fun`s + Filter/SpeedCal/Align/Gq handles (mirror `core_bridge.py`) |
| `android/…/nav/Features.kt` | `window_features` + affine calib, ported verbatim from `py/model/features.py` |
| `android/…/nav/SpeedProfile.kt` | reads `assets/<name>.profile.json`: ONNX file, calib, fusion settings; `DEFAULT = "nn_real"` |
| `android/…/nav/SpeedNet.kt` | ONNX Runtime session over the profile's graph |
| `android/…/nav/Enu.kt` | local ENU projection (== `io_vnbd._lla_to_enu`) |
| `android/…/nav/FusionEngine.kt` | **the live loop** (mirror of `core_bridge._run` + GNSS gating) |
| `android/…/nav/NavActivity.kt` | UI: Start/Stop, Outage toggle, status readout |
| `android/…/nav/TrackView.kt` | original blank-canvas render (superseded by `NavMap.kt`, see 6b) |
| `android/app/src/main/assets/nn_real.onnx` + `.profile.json` | the shipped SpeedNet (real-data retrain, 620 KB, opset 17) |
| `android/app/src/main/assets/nn.onnx` + `.profile.json` | the synthetic net, kept as a one-line fallback |
| `py/phase6_check.py` | host verifier: runs the live loop off-device against the real core+net |

## Build
No Gradle scaffolding was committed before (Phase 1 note); it is now. With the
Android SDK + NDK + CMake present:

    cd android
    # first checkout only: let Studio/Gradle generate the wrapper jar
    gradle wrapper --gradle-version 8.7        # or open the folder in Android Studio
    ./gradlew assembleDebug                    # -> app/build/outputs/apk/debug/app-debug.apk
    ./gradlew installDebug                     # onto a connected phone (sensors/GNSS need real HW)

CMake compiles the repo's `core/*.cpp` in place, so `android/` must stay inside
the OffMaps tree. `minSdk 26` (NavIC `CONSTELLATION_IRNSS` + `GnssStatus.Callback`).

## Exit gate
`libidr` gained one **pure accessor**, `idr_get_cov` (position covariance for the
live GNSS χ²) — added to the header, `eskf.cpp`, ctypes and JNI. It changes no
state, so every prior gate still holds:

- **ESKF/SpeedCal parity** unchanged: `test_eskf_parity.py` max\|diff\| **1.4e-13**;
  full `pytest` **106 passed** (incl. the live-loop regression below); now **163
  passed** with Phase 7's cores, the IO-VNBD real-data loader, the real-data
  SpeedNet trainer, the checkpoint-tuned ESKF settings, the offline map and the
  real-data model on the phone added (see README_PHASE7.md, REALDATA.md, 6b, 6c).
- **Native embed** builds and links against the core; **all 32** JNI symbols export
  and match the Kotlin `external fun`s 1:1 (host-compiled against the JDK `jni.h`).
- **Feature port** matches `features.py` to **6e-8**; the on-device ONNX speed path
  matches the reference to **1e-6 m/s**.
- **Live loop** (`phase6_check.py`, the on-device loop run off-device against the
  real core+net): fusion tracks GNSS to **0.18 m** median; a **60 s** outage
  produces **19.7 m** dead-reckoning drift (0.44% of distance) versus **0.0 m**
  with GNSS, and re-converges when GNSS returns. The outage flag demonstrably
  changes the live filter's output — the Phase-6 claim, proven. Locked as a
  regression in `tests/test_phase6_live_loop.py` (3 tests: tracks / flag honoured /
  re-converges), now run for both shipped profiles. (Current numbers after 6c:
  `nn` 0.15 m / 16.0 m / 0.02 m; `nn_real`, off-distribution on synthetic data,
  0.03 m / 334 m / 0.07 m.)

Not executable on a build box (needs a device): the Canvas renderer, ONNX Runtime
*Android* bindings, and live sensor delivery. Those are verified by inspection and
the host cross-checks above; the fusion logic itself is proven end-to-end.

## ponytail notes
- Fusion loop in Kotlin, not JNI: the validated loop was always in the binding
  layer (`_run`); pushing it into C++ would fork the thing we spent five phases
  proving. JNI stays a marshaller.
- One background thread owns every native call, so the core needs no locking;
  the engine posts immutable `NavState` snapshots to the UI thread.
- Map/corridor mode (`mm_*`) is now live: see 6b below.
- DOP isn't in the logging contract, so `gq_trust` gets a nominal DOP; wire real
  HDOP through `GnssStatus`/`meas.csv` when a replay shows it matters.

## 6b — offline map + road snapping (Hyderabad)

The app now draws everything on an offline map and uses the road network while
dead-reckoning. No network, no API key.

| Piece | What |
|---|---|
| `tools/build_map.sh` | `.osm.pbf` (+ bbox) → osmium clip/filter/export → `osm_layers.py` → tippecanoe |
| `tools/osm_layers.py` | slim per-layer GeoJSON (roads/water/waterway/green/rail) + `roads.bin` (format in its docstring) |
| `assets/map/tiles.mbtiles` | Hyderabad basemap, z10–16, 21 MB (`78.20,17.15,78.75,17.65`: ORR + margin) |
| `assets/map/roads.bin` | 208k OSM roads → 211k ≤32-vertex pieces, 9.7 MB, memory-mapped |
| `assets/map/style.json`, `assets/glyphs/NotoSansRegular/` | map style; Latin road-name glyphs (OFL) |
| `nav/NavMap.kt` | MapLibre view: basemap + grey/blue/orange tracks + position/heading, follow camera |
| `nav/RoadNetwork.kt` | reads `roads.bin`, ~550 m grid index → `waysNear()` |
| `nav/RoadMatcher.kt` | keeps a 1 km window of roads in the native `mm` (rebuilt after 400 m) |
| `nav/FusionEngine.kt::mapMatchUpdate` | the Phase-5 `eskf_map` step: cross-track + road-heading update |

**When the road is used.** Only while dead-reckoning: Outage Simulator on, or no GNSS
applied for 2 s (real outage / spoof rejection). A centreline isn't your lane, so with
healthy GNSS it is never consulted. Matches farther than 25 m are ignored. The
"Road snapping" toggle turns it off to compare the two outage tracks.

**What it buys (host proof, `tests/test_phase6_map_aid.py`).** Same 60 s outage as the
Phase-6 gate: cross-track at outage end 10.1 m → 0.0 m; along-track unchanged
(~17 m). (After 6c the loop learns gyro bias well enough that this outage barely
drifts sideways, so the test now adds a 0.05°/s gyro-bias step at outage start:
cross-track 19.8 m → 0.0 m with `nn_real`.) A road pins the lane, never the distance travelled — that is still the
speed model's job. With GNSS healthy the fused track is bit-identical with/without roads.

**Checked without a device:** APK builds (JDK 17); style passes `gl-style-validate`;
the style + mbtiles were rendered headless with MapLibre Native (same core as the
Android SDK); the matcher on real Hyderabad windows is ~1k ways, ~0.07 ms/match on a
laptop. **Needs a device:** the MapView itself, the camera follow, live GNSS/IMU.

Rebuild / another city:

    brew install osmium-tool tippecanoe
    tools/build_map.sh southern-zone-latest.osm.pbf map/hyderabad 78.20,17.15,78.75,17.65
    cp map/hyderabad/tiles.mbtiles map/hyderabad/roads.bin android/app/src/main/assets/map/

(`map/hyderabad/area.osm.pbf` is the kept 34 MB clip: `tools/build_map.sh map/hyderabad/area.osm.pbf map/hyderabad`
refreshes styling/layers without re-downloading the 531 MB Southern Zone extract.)

Build with JDK 17 (Gradle 8.7 does not run on the system JDK 22):
`JAVA_HOME=~/Library/Java/JavaVirtualMachines/jdk-17.0.20.1+1/Contents/Home ./gradlew assembleDebug`.
Map data © OpenStreetMap contributors (ODbL), shown in the map's attribution.

## 6c — the real-data SpeedNet on the phone (2026-09-23)

The app now runs `nn_real` (REALDATA.md) with the ESKF settings tuned for it on
validation drives. Nothing is hand-copied into Kotlin any more:

    cd py
    python -m model.export --ckpt model/nn_real.pt --onnx ../android/app/src/main/assets/nn_real.onnx --profile

writes the graph (parity-checked against torch) and `nn_real.profile.json`: the
checkpoint's calibration and its resolved fusion settings (`ESKF_DEFAULT` + the
checkpoint's `eskf_cfg`). `SpeedProfile.kt` and `phase6_check.py` both read that
file. `tests/test_phase6_shipped_model.py` checks each shipped graph and profile
against its checkpoint. To fall back, change `SpeedProfile.DEFAULT` to `"nn"`.

**Porting it exposed a live-loop bug that the old app already had.** On the real
validation drives (S3b, S3c; never the test split), the loop as built rejected
~99 % of real GNSS fixes as "spoofed" with the old `nn` profile. Median error was
0.65 km and 12.6 km **with GNSS on**. The chain:

1. The loop fused the NN speed while GNSS was healthy. On real roads the net is
   metres/second off, so just before each fix it pulled the filter's speed away from Doppler.
2. `gq_spoof`'s fixed 5 m/s Doppler-residual test then fired. Phase 4's spoof gate
   only ever validated the position-χ² test (it passes a residual of 0), so this
   part had never been checked.
3. Rejected fixes never reach the Doppler self-calibration that is meant to remove
   the NN's bias, and after a long outage the χ² test kept rejecting a correct
   GNSS against an over-confident dead-reckoned position. GNSS never came back.

Fix, in `FusionEngine.kt` and its mirror:
- **Speed aids only while dead-reckoning.** With GNSS healthy, Doppler (~0.1 m/s)
  owns speed. The NN still runs every second as the self-cal regressor.
- **Spoof check = position χ² only**, the configuration Phase 4 validated.
- **Re-acquire:** 5 strong fixes rejected in a row → re-seed the filter from GNSS.
  Trade-off: a spoof sustained for more than 5 s is then followed; the spoof flag shows until then.
- **Self-cal guard:** the fitted scale is used only if 1/3 ≤ k ≤ 3. A net that
  doesn't track speed produces fits like k = 72 (`nn_real` on synthetic data).

Real validation drives, live loop, 60 s outage every 4 min (16 windows, `nn_real`):

| self-cal | median 60 s drift | p75 | 10 s after GNSS returns (max) |
|---|--:|--:|--:|
| off (k = 1) | 30.6 % | 49.7 % | 1.8 m |
| guard 0.5–2 | 19.1 % | 50.7 % | 1.7 m |
| no guard | 15.7 % | 45.5 % | 1.7 m |
| **guard 1/3–3 (shipped)** | **11.3 %** | **24.5 %** | **1.7 m** |

The guard was chosen from these four settings on the same 16 val windows, so
11.3 % carries some selection optimism. The test split was not used. For
comparison, the old `nn` profile under the fixed loop drifts 82.8 % (its fits are
rejected) but no longer locks out (≤ 2.4 m after return). Doppler self-cal matters
because the harness `eskf` numbers in REALDATA.md start each outage cold. The
live app has minutes of GNSS before an outage to learn the net's bias.

Pinned by `tests/test_phase6_realval.py` (tracks with GNSS on, re-admits GNSS after
every outage; skipped where the local IO-VNBD copy is absent),
`tests/test_phase6_live_loop.py` (both profiles; `nn_real` on synthetic is the
lockout stress test) and `tests/test_phase6_shipped_model.py`.

**Still needs a device.** None of this has run on a phone. The IO-VNBD phone and
car are not a Hyderabad phone and car, so expect `nn_real` to be somewhat
off-distribution there too. The loop now degrades safely (GNSS still tracks and
recovers), but the dead-reckoning accuracy on this hardware is unknown until a
real drive is logged and scored.

## 6d — accuracy work on real data (2026-09-23)

All choices below were made on the **validation** drives (S3b, S3c) in the live
loop (`phase6_check.run`), each by a script that writes its choice and full sweep
into `model/nn_real.pt` / `nn_real.json`. The test split was scored once, at the end.

**Physics→NN hand-over** (`model/tune_handover.py`, `handover_s`). For the first
N s of dead-reckoning the filter keeps the last Doppler speed (that *is* the
`physics` baseline, which wins short outages), then fuses the NN. Val, median drift
at 10/30/60/120 s: NN from the start 24.0/11.3/11.3/10.4 %; physics only
8.8/16.0/20.0/20.9 %; **hand-over at 10 s 8.8/12.8/12.9/11.2 %** (the best mean of 11 settings).

**Calibration (k-fold) — not needed on the phone.** The offline σ temperature was
fit on only two val drives, and k-fold by drive was meant to steady it. But the
phone learns the NN's affine bias online (Doppler self-cal), and what σ still
decides is the covariance. That is already honest on val: at outage end the
position error is inside the filter's 95 % ellipse for 90–100 % of outages
(mean z² 0.3–2.5; 2 is ideal). Pinned by `tests/test_phase6_realval.py` instead of
retraining six fold models.

**Road snapping made real outages worse; fixed** (`model/tune_map.py`, new core
switch `idr_set_map_keep_speed`). With `srw=24`, a cross-track innovation leaked
into *speed* through the heading/position correlation, so 60 s drift went from
12.9 % to 19.2 % with the map on. The new switch zeroes the gain's speed row for
road updates (Joseph-form covariance, exact for that gain). With it off, the old
update is unchanged bit for bit. Native vs oracle parity is 2.6e-13 in both modes,
and the dead-reckoned speed is now bit-identical with or without roads
(`tests/test_phase6_map_aid.py`). Val, oracle map, 30/60/120 s:

| setting | 30 s | 60 s | 120 s | mean |
|---|--:|--:|--:|--:|
| no map | 12.9 % | 12.9 % | 11.3 % | 12.4 % |
| old road update (σ 1.5 m, 3°) | 16.6 % | 19.2 % | 29.1 % | 21.6 % |
| **keep speed, σ 5 m, 3° (shipped)** | **8.6 %** | **8.7 %** | **10.8 %** | **9.4 %** |

The map here is each drive's own GNSS track, so this is an upper bound on what the
real Hyderabad OSM roads will give.

## 6e — the NN sees the frame it was trained in (2026-09-23)

`nn_real` was trained on IMU rotated into a z-up vehicle frame (the loader's
`_level_and_heading`). The app fed it raw **device-frame** samples. On val, a
portrait dash mount moved its bias from −4.5 to +0.5 m/s and its MAE from 5.53 to
6.56 m/s. The net turned out insensitive to the horizontal heading (MAE unchanged
under any rotation about up), so leveling is enough.
`Level.kt` (reference `py/model/mount.py`) rotates each 10 Hz sample into (h1,
h2, up) using a **30 s** gravity average. The 0.5 s one follows braking and
cost 0.45 m/s of MAE. With it, every mount tested gives the trained-frame MAE (5.53–5.55).

**Re-mount detector rebuilt** (`core/align.cpp`). The old CUSUM compared the 0.5 s
gravity with a snapshot at 2° slack. On real rigid-mount drives it fired
~1,800×/hour, and the app would have shown "↻ re-mount detected" almost constantly.
Phase 4's gate only ever checked that it fires after a re-mount. Now: CUSUM on
the angle between the 5 s and 30 s gravity (slack 10°, threshold 200°), tuned on
train+val. About 1 false alarm per hour; a 30° re-mount is found in 3–8 s, and
Phase 4 gate 2 still passes (detect +3.0 s, re-level +0.4 s). On an alarm the
leveling snaps to the new mount (`Level.onRemount`).

`tests/test_kotlin_ports.py` compiles `Level.kt` + `Features.kt` on the host (the
Kotlin compiler Gradle caches, JDK 17) and checks them against `mount.py` /
`features.py` (1e-9 / float32).

## 6f — Phase 7 on the phone (2026-09-23)

The NDK build now compiles the whole core (`vib.cpp`, `eskf3d.cpp` were missing),
and the JNI bridge binds every function in `idr.h`: 55 symbols, 1:1 with
`IdrNative.kt` (`tests/test_jni_bridge.py` compiles the bridge against the JDK's
`jni.h` and checks the exports).

- **7c potholes/bumps — live.** Every raw accelerometer sample (full sensor rate,
  measured over the first 2 s) goes through `Vib`. Each shock is placed at the
  fused position, counted in the status line and drawn as an orange dot on the map.
- **7a 3D filter, 7b Viterbi decode — bound, not in the live loop** (`Filter3D`,
  `MapMatcher.matchSeq`). The planar filter is the one validated on real drives.
  7a's gain is for multi-level ramps, and 7b needs road topology that `roads.bin`
  does not carry (no `mm_add_edge` data). Phase 7's own rule applies: wire them in
  once a device replay shows the need.

## Held-out result (test split, scored once after every choice above was fixed)

Test = S3a + Y1 (Driver D, never trained on), 2.5 h. This is the test set's
**fourth** look overall (three in REALDATA.md); nothing was chosen from it.
Live loop, median drift:

| | 10 s | 30 s | 60 s | 120 s |
|---|--:|--:|--:|--:|
| **shipped (no map)** | **11.2 %** | **17.1 %** | **17.0 %** | **17.3 %** |
| physics only (hold Doppler) | 11.2 % | 17.2 % | 23.7 % | 29.5 % |
| shipped + road snapping (oracle map) | 11.9 % | 15.9 % | 14.7 % | 17.8 % |

Validation was more optimistic (mean ~11 % vs ~16 % here), which is expected when
every choice is made on two drives. Still short of the ISRO 10 % target.

## 6g — first run: Android emulator (2026-09-23)

The app had never executed. On an API 34 arm64 emulator (Pixel 6) it installs,
launches, renders the offline Hyderabad map (street level, 210,939 road pieces
loaded), loads ONNX Runtime + `libidrjni` + the model profile, and survives
start/stop cycles and the outage toggle. Road snapping works live. Driven with
`adb emu geo fix` at 1 Hz. Three bugs only a real run could find:

1. **The IMU path never ran, on any device (since Phase 6).** The 10 Hz decimation
   did `t - Long.MIN_VALUE < 100 ms`. That overflows to a negative Long for every
   positive timestamp, so every sample was dropped: no predict, no NN, position
   frozen while GNSS speed read 10 m/s. Now `Decimator.kt`, checked on the host by
   `tests/test_kotlin_ports.py` at real phone timestamps. The host mirror never
   had the bug, which is why no gate saw it.
2. **Weak GNSS switched the aids off.** Any unrejected fix counted as "GNSS
   healthy", which disabled NN speed and road snapping while a trust-≈0 fix
   (few satellites, low C/N0) contributed nothing. Now only fixes with trust ≥ 0.1
   (σ ≤ 30 m) end dead-reckoning; weaker fixes are still fused with their large R.
   Val numbers are unchanged, because those drives always had strong trust.
3. **`k=NaN`.** Self-cal pushes weighted by trust 0 made the fit 0/0. `spc_fit`
   now keeps its previous fit when the total weight is 0 (C++ and oracle, pinned in
   `tests/test_speedcal_deming.py`), and only trusted fixes are pushed.

**What the emulator cannot show:** it reports no satellites (`GnssStatus` is
empty, so trust is always 0 and the healthy-GNSS path never runs there), its IMU is
static (the NN sees a parked car), and it gives every fix bearing 0°. Accuracy,
sensor rates, potholes and the trusted-GNSS path still need a real phone.

## 6h — first run on a real phone (Redmi Note 9 Pro, Android 12, 2026-09-23)

Installed over USB, location allowed by the owner. On a desk by a window:
no crash; offline map, road snapping, ONNX model and JNI all live. Accel and gyro
at **400 Hz** (the pothole detector measures and uses the real rate). 197 real
location fixes delivered, 11 satellites used, but C/N0 was only 22 dB-Hz indoors.
That is below `gq_trust`'s 25 dB-Hz floor, so the app correctly treated GNSS as
untrusted and dead-reckoned.

Fixed from it: the pothole counter showed 20 "bumps" from handling the phone, so
shocks now count only above 3 m/s. A stationary self-cal fit (k = 0; Doppler ≈ 0)
is now shown as "(not used)", which it already was.

Not answerable on a desk: dead-reckoning accuracy, whether stops (ZUPT is off in
the `nn_real` profile) hold still in a car, the trusted-GNSS path, real potholes.
Those need a drive. Log one and score it with `validate_realdata.py --phone`
(REALDATA.md).
