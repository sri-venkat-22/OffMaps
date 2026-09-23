# Phase 4 — Calibration, alignment, GNSS trust

The phase the competition turns on. Three on-device estimators, same `libidr`
core, feeding the EKF. Uncalibrated we're borderline; **Doppler self-calibration
is what buys the margin.**

## Modules (C++, C ABI)
| file | what | key idea |
|---|---|---|
| `core/speed_cal.cpp`    | 60 s ring buffer, weighted-LS `v_true = k·v̂ + c`, excitation-gated | recovers the domain gap online, **no retraining** |
| `core/align.cpp`        | gravity→roll/pitch (EMA), PCA yaw on longitudinal events, CUSUM re-mount detect | auto re-align in <1 s |
| `core/gnss_quality.cpp` | continuous trust from C/N₀·SV·DOP·χ², NavIC-weighted; `R` 3 m→1e6 | **no mode switch**; spoof = 2nd threshold |

Bindings: `py/core_bridge.py` (`SpeedCal`, `Align`, `gq_*`). Oracle:
`py/speed_cal_ref.py`.

## Gates (run: `PYTHONPATH=. python3 phase4_gates.py --out ../out`)
1. **Cross-vehicle** — an artificial 10% scale gap on a held-out vehicle;
   Doppler self-cal on 60 s of pre-outage GNSS recovers it: baseline **3.1%**,
   gapped **8.0%**, calibrated **3.6%** along-track drift at 60 s — **91%
   recovered**, and calibrated lands back at baseline. `k` converges to `1/gap`. ✅

   *History (Phase-2 update):* this gate once reported *152%* on the same gap
   "because it also removes the model's own bias" — it was leaning on the NN's
   scale error as free headroom. When Phase 2's head was made honest
   (σ-calibrated), that crutch vanished and the gate exposed a real
   errors-in-variables flaw: `SpeedCal`'s **OLS** fit regresses Doppler on the
   *noisy* NN speed, so its slope `k` diluted toward the noise floor (recovery
   fell to 38%). Fixed properly (option (b)): `core/speed_cal.cpp` now supports a
   **Deming** fit (`spc_set_lambda`) that treats the NN speed as noisy, using the
   calibrated NN σ as the regressor-noise variance
   (`λ = σ_doppler² / σ_nn²`). It recovers `k = 1/gap` under noise; `λ = +∞`
   reproduces the old OLS fit bit-for-bit, so parity and every other caller are
   untouched (parity: OLS 5e-14, Deming 9e-14).
2. **Re-mount** — phone physically re-rotated mid-drive: CUSUM detects instantly,
   roll/pitch re-level in **0.4 s** (<5 s). ✅
3. **Spoofing** — GNSS position spoofed while C/N₀ stays high: INS rejects it,
   flag fires at onset with zero false positives, trust 1.00 → 0.013. ✅

Parity (in `test_eskf_parity.py`): `speed_cal` C++ vs oracle max|diff| < 1e-9.

## ponytail notes
- Calibration is a rolling-buffer LS, **not** filter states — `k,c` are
  correlated at constant speed, so the excitation test (`std(v̂)>2 m/s`) decides
  2-param vs scale-only. This is simpler AND more robust than two random-walk
  states in the EKF.
- `gq_*` are stateless pure functions; the χ² input comes from the filter's own
  innovation in the live loop.
- PCA yaw accumulates only when `|ψ̇|<5°/s` and `|a_horiz|>1` — turns give
  lateral accel (90°-wrong axis), so they're excluded. The forward/backward
  180° sign is resolved by GNSS course when healthy (live-loop detail).
