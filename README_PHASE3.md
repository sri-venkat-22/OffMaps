# Phase 3 — ESKF core in C++

One C++17 core (`core/libidr`), C ABI, no external deps. The Android app (NDK),
the edge build, `tools/replay`, and the Python harness (ctypes) all drive the
**same object file** — that's the dual-deliverable claim, proven, not asserted.

## What it is (and honestly isn't)
A **planar 5-state EKF**: `[e, n, ψ, v, b_g]`. Position integrates SPEED along
HEADING — never the accelerometer — so drift stays linear in distance. In the
unicycle model NHC (lateral v = 0) is structural, so there is no separate NHC
update. All updates are sequential scalar → no matrix inverse, no Eigen.

Not the full 16-state 3D error-state formulation from the plan. That (δθ(3),
b_a(3), quaternion attitude, vertical) is the documented extension for the
genuinely-3D case — multi-level parking. The planar core is exact for the
road/corridor case that dominates ISRO's scoring. Build the 3D core when
parking replays demand it.

> **Built in Phase 7a** — `core/eskf3d.cpp` is the full 16-state error-state KF
> (δθ/b_a/quaternion/vertical, NHC + odometer + baro). It is still not the
> default; the planar core stays exact and cheaper for the road case. On a
> helical parking ramp it holds 1.8 m where the planar core is 46.8 m off.
> See [README_PHASE7.md](README_PHASE7.md).

## Observables wired
| update | when | note |
|---|---|---|
| `idr_update_speed`     | 1 Hz (NN) / GNSS Doppler | primary along-track aid |
| `idr_update_zupt`      | stopped (v<0.5) | v=0 **and** ZARU re-observes gyro bias |
| `idr_update_curvature` | \|ψ̇\|>10°/s | v=a_lat/ψ̇, R∝1/ψ̇; independent of the NN |
| `idr_update_gnss_pos`  | GNSS healthy | time-varying R = the "seamless" transition (Phase 4 drives R) |
| `idr_update_gnss_vel`  | GNSS Doppler | observes v and ψ |

## Build & run
    sh core/build.sh                      # -> core/libidr.dylib + tools/replay
    cd py
    PYTHONPATH=. python3 test_eskf_parity.py         # gate 1
    PYTHONPATH=. python3 -m eval.report --model eskf,nn,physics,cv --out ../out

## Files
| file | what |
|---|---|
| `core/idr.h`        | C ABI |
| `core/eskf.cpp`     | the filter (predict + all updates) |
| `core/build.sh`     | shared lib + replay binary |
| `tools/replay.cpp`  | standalone C++ replay (no Python) |
| `py/eskf_ref.py`    | pure-Python twin — the parity oracle |
| `py/core_bridge.py` | ctypes wrapper + registers the `eskf` model |
| `py/test_eskf_parity.py` | C++ vs oracle cross-check — standalone script, all 7 updates |
| `py/tests/test_eskf_core_parity.py` | the same gate as a `pytest` regression |

## Exit gates
1. **Native core == Python oracle** on a fixed sequence exercising **all 7
   updates**: **max|diff| = 1.4e-13** (< 1e-3). ✅ Now a suite regression, not
   just a `__main__` script: `tests/test_eskf_core_parity.py` (SpeedCal parity
   was already pinned in `test_speedcal_deming.py`). `idr_update_gnss_vel` was in
   the C ABI + the observables table but had **no oracle twin and no Python
   bridge method** — it shipped unproven; it now has both and is in the gate.
2. **Not worse than Phase-2 naive integration:** on synth, `eskf` ties `nn` at
   60 s (8.8% vs 8.4%, within n=24 noise) and **beats it at 120/180 s (10.1% vs
   21.7%, 12.7% vs 16.7%) and on CEP95** (615 m vs 647 m). The filter's real edge
   — bias-corrected speed (Phase 4), partial-GNSS fusion, map/road-bearing
   (Phase 5) — is exactly what synthetic data can't contain, so a decisive win
   here would be a tuning artefact, not a result. ⚠️→ separates in Phase 4/5.

## ponytail notes
- Observables live in `eskf.cpp`, not a separate `observables.cpp` — 5 update
  functions don't need their own translation unit. Split when they grow limbs.
- Speed process-noise is tuned at the call site (`core_bridge.py`) — it's the
  hardware/vehicle knob; a real vehicle's speed dynamics differ from synth.
