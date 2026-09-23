# Phase 7 — the deferred extensions, built

Three items were deliberately deferred in earlier phases as "build when replays
demand it," not gaps. This phase builds all three to the house standard: a C++
core module in `libidr`, a bit-exact Python **oracle twin**, a **parity gate**
(native == oracle to numeric noise), an **honest exit gate** on synthetic data
that shows what the feature actually buys, and a **pytest regression**. The
planar 5-state core and every prior gate are untouched — the new cores are
additive (`idr3d_*`, `mm_match_seq`, `vib_*`), so Phases 3–6 still hold.

| deferred item (where) | now |
|---|---|
| 3D 16-state ESKF (README_PHASE3.md) | `core/eskf3d.cpp` — full δθ/b_a/quaternion/vertical error-state KF |
| Viterbi/HMM map matching (README_PHASE5.md) | `core/map_match.cpp::mm_match_seq` — global sequence decode |
| high-rate 200–400 Hz vibration/pothole filtering (the plan) | `core/vib.cpp` — streaming shock detector + pothole-rejected RMS |

---

## 7a — full 3D 16-state error-state KF (`idr3d`)

**What it is.** Nominal state `[p(3), v(3), q(4), b_a(3), b_g(3)]` (16); error
covariance is 15×15 `[δp, δv, δθ, δb_a, δb_g]`. ENU world (z up), body
x-fwd/y-left/z-up, Hamilton quaternion body→world, local (right) attitude error.
Gravity is modelled, so pitch/grade no longer leak into horizontal position the
way the planar core's "all speed is horizontal" assumption does. Propagation is
the classic first-order INS error model; updates are sequential scalar with
immediate inject-and-reset (G≈I) — the same discipline as the planar core, so no
Eigen and no matrix inverse. Updates wired: GNSS pos/vel, ZUPT, ZARU, **NHC**
(lateral+vertical body velocity = 0 — a real update in 3D, where it was structural
in the planar unicycle), **odometer** (body-forward velocity = wheel speed), baro.

**What it honestly isn't.** Still not the default. The planar core remains exact
and cheaper for the road/corridor case that dominates scoring; `idr3d` is for
genuinely-3D motion (multi-level parking, steep ramps). It is host/edge-ready and, since
README_PHASE6 §6f, bound on the JNI path (`Filter3D`). It is not in the live loop:
wire it in when a device replay needs it (same rationale Phase 6 gave for dormant map mode).

**Gate** (`py/phase7a_gate.py`): a helical parking ramp (12° grade, 2.9 turns,
46.7 m climb) driven as a GNSS outage. Both filters get the **same** aiding
(wheel speed + gravity-projected yaw); only the geometry model differs.
- planar end 3D error **46.8 m** (it has no state for the climb) → 3D **1.84 m**;
- 3D recovers altitude to **1.4 m** (planar: none);
- clean-IMU reconstruction closes to **0.66 m** (validates the synthesizer+integrator).

**Proof.** `py/test_eskf3d_parity.py`: native vs oracle max|diff| **7e-13** over
predict + all updates; the hand-derived body-velocity (NHC/odometer) Jacobians
match finite differences to **1e-9** (so "both agree" can't mean "both wrong the
same way"). Locked in `tests/test_phase7a_eskf3d.py`.

---

## 7b — Viterbi/HMM map matching (`mm_match_seq`)

**What it is.** `mm_match` is greedy nearest-road; at a junction where two roads
are momentarily close it can wrong-snap under position noise. `mm_match_seq`
decodes the WHOLE trajectory with an HMM: states are the road ways, **emission** =
cross-track distance + bearing agreement, **transition** = road-network adjacency
(`mm_add_edge`) + a GPS-vs-route distance-consistency term (Newson–Krumm style).
Viterbi returns the globally most-consistent way per step. `mm_set_hmm` tunes the
emission/transition/bearing weights.

**What it honestly isn't.** Not online — it's a batch decode over a stored
trajectory (map matching was always the offline aid). The greedy per-step matcher
stays the live path; Viterbi is the offline refinement, exactly the "~90% of the
value from nearest-road; add Viterbi if junctions wrong-snap" trade Phase 5 named.

**Gate** (`py/phase7b_gate.py`): a Y-fork (trunk → branch A straight, branch B
diverging) with GNSS-scale noise; the vehicle takes A.
- greedy on the correct road **78%**, with **4** wrong-snaps onto B near the fork;
- Viterbi **99%**, with **0** wrong-snaps.

**Proof.** `py/test_mapmatch_seq_parity.py`: native decode vs oracle is
**bit-identical** — 0 way-index mismatches, feet to **7e-15** — a real test for a
discrete argmin, where any FP divergence in the cost DP would flip a snap. Locked
in `tests/test_phase7b_viterbi.py`.

---

## 7c — high-rate (200–400 Hz) vibration / pothole front-end (`vib`)

**What it is.** The 10 Hz feature path (`model/features.py`) can't see road
vibration above its 5 Hz Nyquist — it aliases — and a pothole read at 10 Hz looks
like a huge acceleration that corrupts the speed cue. `vib` is a streaming, O(1)
DSP on the raw high-rate accelerometer: a 1st-order high-pass isolates the
vibration band; an adaptive threshold (EMA baseline + k·scale, frozen during a
shock and warmed up first, so a pothole can't poison its own baseline) flags
shock transients; per-window RMS is accumulated twice — `rms_raw` over all
samples and `rms_clean` with shocks excised. `rms_clean` is the pothole-robust
speed cue; the event edges are pothole detections.

**What it honestly isn't.** The **learned** high-rate RMS→speed map still trains
on real phone logs — this is the deterministic front-end it consumes, not a
trained model. The gate proves the front-end's DSP, not a shipped speed net.

**Slip head.** The TCN's lateral-slip head (`model/tcn.py`) was wired but
unsupervised — dead weight, because on synthetic data lateral accel *is*
`v·yawrate` by construction, so the slip residual is zero. Phase 7c wires the
supervision (`model/loss.py::slip_loss`, folded into `total()` only when a target
is supplied, so existing training is byte-unchanged) and proves the path learns
an injected slip target end-to-end (`tests/test_slip_head.py`: loss drops,
prediction correlates >0.9). Real-slip learning still needs phone logs — the note
stands; the wiring is no longer dead.

**Gate** (`py/phase7c_gate.py`): 250 Hz synthetic accel, speed-dependent
vibration + 5 injected potholes.
- detection **recall 1.00, precision 1.00** (exactly 5 events for 5 potholes);
- `rms_clean` is a clean speed cue (corr **0.994**);
- on pothole windows, naive raw-RMS speed is **6.0 m/s** off → pothole-rejected
  **0.5 m/s**.

**Proof.** `py/test_vib_parity.py`: native vs oracle is **bit-exact** — 0
per-sample edge mismatches, per-window features to **1e-16** — at 250 and 400 Hz
and with custom params. Locked in `tests/test_phase7c_vib.py`.

---

## Files
| file | what |
|---|---|
| `core/eskf3d.cpp` | 3D 16-state ESKF (predict + 8 updates) |
| `core/map_match.cpp` | `mm_match_seq` Viterbi + `mm_add_edge`/`mm_set_hmm` |
| `core/vib.cpp` | high-rate vibration/pothole front-end |
| `py/eskf3d_ref.py` · `py/map_match_ref.py` · `py/vib_ref.py` | oracle twins |
| `py/data/synth3d.py` | helical-ramp 3D IMU synthesizer |
| `py/data/highrate.py` | 200–400 Hz accel + pothole synthesizer |
| `py/phase7{a,b,c}_gate.py` | exit gates (+ plots in `out/`) |
| `py/test_{eskf3d,mapmatch_seq,vib}_parity.py` | standalone parity scripts |
| `py/tests/test_phase7{a,b,c}_*.py`, `test_slip_head.py` | pytest regressions |

## Build & run
    sh core/build.sh                                   # libidr + replay, now incl. eskf3d/vib
    cd py
    PYTHONPATH=. python3 test_eskf3d_parity.py         # 7a parity + Jacobians
    PYTHONPATH=. python3 test_mapmatch_seq_parity.py   # 7b parity
    PYTHONPATH=. python3 test_vib_parity.py            # 7c parity
    PYTHONPATH=. python3 phase7a_gate.py --out ../out  # + phase7b_gate.py, phase7c_gate.py
    PYTHONPATH=. python3 -m pytest tests -q            # 163 passed

## Exit gate
All three cores match their oracles to numeric noise (7a state 7e-13; 7b 0
mismatches; 7c 1e-16), each honest gate passes, and the full suite is **121
passed** (was 110) with every prior gate (Phases 3–6) still green — the new cores
are additive, so nothing that shipped was disturbed.

## ponytail notes
- Three separate cores, not one refactor: `idr3d` does not replace `idr`. Forking
  the validated planar filter to bolt on a z-axis would risk five phases of proof;
  the 3D core stands beside it and is selected per replay.
- Every new capability got an oracle twin *and* an independent correctness check
  (finite-diff Jacobians for 7a, discrete-decode parity for 7b, per-sample edge
  parity for 7c) — "native matches oracle" alone can hide a shared bug.
- The slip head is the one place synth can't supply truth; it's wired and proven
  learnable, then left honest — the deferral is about data, not code.
