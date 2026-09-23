# Android — logger (Phase 1) + nav app (Phase 6)

This module now holds **two** activities:

- **`nav/NavActivity`** — the launcher. The **real on-device nav app** (Phase 6):
  it embeds `libidr` via NDK/JNI, runs the ESKF **live**, honours the outage flag
  against the live filter, and renders position. See [../README_PHASE6.md](../README_PHASE6.md).
- **`logger/LoggerActivity`** — the Phase-1 logger below. Its whole job: record
  raw IMU + GNSS to CSV so a real drive replays through the Python harness. It
  does no fusion; it's the data-capture tool, not the nav app.

## Build

Gradle + NDK/CMake scaffolding is now committed (`settings.gradle`,
`build.gradle`, `app/build.gradle`, `app/src/main/cpp/CMakeLists.txt`). With the
Android SDK + NDK + CMake installed:

    cd android
    gradle wrapper --gradle-version 8.7   # first checkout only (generates the wrapper jar)
    ./gradlew assembleDebug               # or just open android/ in Android Studio
    ./gradlew installDebug                # onto a physical phone (sensors/GNSS need real HW)

minSdk 26: `GnssMeasurementsEvent.Callback` and `CONSTELLATION_IRNSS` (NavIC)
are API 24+; 26 is a safe floor. CMake compiles the repo's `core/*.cpp` in place,
so keep `android/` inside the OffMaps tree.

## The logger (Phase 1)

## Using it

- **Start logging** — writes to `Android/data/com.offmaps.logger/files/drive_<ts>/`.
- **Outage Simulator** — toggles `masked=1` on GNSS rows. GNSS keeps logging as
  ground truth; replay treats masked spans as outages. (The Phase-6 nav app's own
  Outage Simulator now honours this flag against the *live* filter — it feeds the
  ESKF no GNSS while masked, so the fused track dead-reckons.)
- **Stop logging** — flushes the CSVs. `adb pull` the drive folder to your machine.

## Output — the contract with `py/data/phone_log.py`

| file | columns |
|---|---|
| `imu.csv`  | `t_ns,ax,ay,az,gx,gy,gz,mx,my,mz,pressure,light` (a row per accel sample at `SENSOR_DELAY_FASTEST`, ~100–400 Hz; other sensors are latest-value) |
| `gnss.csv` | `t_ns,lat,lon,speed,bearing,cn0_mean,sv_used,navic_sv,masked` (speed/bearing are chip Doppler) |
| `meas.csv` | `t_ns,svid,constellation,prr_mps,cn0` — raw pseudorange-rate, logged now, used for proper GNSS velocity in Phase 4 |

`t_ns` is `elapsedRealtimeNanos` (monotonic) everywhere, so IMU and GNSS share
one clock — the Phase-0 spline resample assumes this.

## Replay a recorded drive

    adb pull /sdcard/Android/data/com.offmaps.logger/files/drive_<ts> ./mydrive
    cd ../py
    python -m eval.report --phone ../mydrive --model physics --out ../out/mydrive

`--yaw-sign` (default −1) is the gyro→heading sign knob; flip it once against
your first real log if the dead-reckoned track mirrors the turns.
