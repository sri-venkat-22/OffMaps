# Phase 2 — AI Speed & Vibration Filter (displacement head)

A dilated TCN + GRU that regresses **1 s displacement with a log-variance**,
trained to minimise the drift metric ISRO scores — not instantaneous MAE.
Registers an `nn` model into the same harness, so the drift table just gains a
column.

## Train + gate (synthetic; no dataset needed)

    cd py && python -m model.train

Trains on synthetic drives (held-out vehicles for val), exports ONNX, prints
the exit gate.

## Why this beats a plain speed regressor
- **Gaussian NLL** → an honest sigma. A point estimate can't be weighted by a
  Kalman filter; a calibrated sigma can. The gate checks `z=(v̂−v)/σ` has unit
  variance.
- **Displacement-consistency loss** (`model/loss.py`) over 1/5/10/30 s horizons
  penalises the *integral* error = the drift benchmark. Suppresses biased
  error, tolerates zero-mean noise.
- **Augmentation** (`model/augment.py`): random SO(3) mount rotation + gyro
  bias/scale + noise every epoch → survives the loose-mount / vehicle-change gap.

## Files
| file | what |
|---|---|
| `model/features.py` | window → (9,20) tensor; the one contract shared by train & inference |
| `model/tcn.py`      | `SpeedNet`: 5 dilated TCN blocks (64ch) + GRU + 3 heads, ~151K params |
| `model/loss.py`     | NLL + displacement-consistency + class CE |
| `model/augment.py`  | mount/bias/noise augmentation |
| `model/dataset.py`  | Drives → contiguous 30 s sequences |
| `model/train.py`    | training loop, exit gate; calls `model/export.py` for ONNX |
| `model/export.py`   | checkpoint → `nn.onnx` (dynamic batch) + onnxruntime↔torch parity check |
| `model/nn_model.py` | registers `nn` (NN speed + gyro heading), emits (v̂, σ) |

## Run in the harness

    python -m eval.report --model nn,physics,cv --out ../out

## On-device export (ONNX)

The deployment artifact is `model/nn.onnx`, regenerable from any checkpoint
WITHOUT retraining:

    python -m model.export           # model/nn.pt -> model/nn.onnx (+ parity check)

For the phone, add `--profile` to also write `<name>.profile.json` (calib + fusion
settings) next to the graph; see the Phase 6 write-up §6c (`git show ec89099:README_PHASE6.md`).

It exports a **dynamic batch axis** (`nn_model` runs a whole outage of 1 s steps
in one call, not batch-1) and then loads the graph under **onnxruntime** to
confirm every head reproduces the torch net to <1e-4 — so the deployed net is
the gated net. `torch.onnx.export` needs the `onnx` package; before it was
installed the exporter's bare-`except` silently shipped a `nn.torchscript.pt`
fallback and no `.onnx`. That failure is now loud (`train.py`) and the ONNX
round-trip is a regression test (`tests/test_phase2_gate.py`).

## Honest sigma (mean + variance)
The displacement head reads speed off a vibration-std estimate computed from a
20-sample window. That estimate is noisy, so the learned map suffers regression
dilution — `v_pred ≈ 0.32 + 0.86·v_true` — a speed-proportional under-prediction
that surfaces in σ-calibration as `z_mean ≈ −0.8`. It is **not** a loss-weighting
problem (β-NLL fixes the variance shape but leaves the mean untouched), so it is
corrected in two honest steps:
- **β-NLL** (`model/loss.py`, Seitzer et al. 2022): weight each sample's NLL by
  `detach(σ^{2β})` so heteroscedastic down-weighting stops reverting the mean.
- **Affine recalibration** fit ONCE on held-out val drives (seeds 1050–54,
  disjoint from training 1000–11 and the report's test set 0–2), stored in the
  checkpoint as `(a, b, s)`: `v = (μ−a)/b`, `σ = (σ/b)·s`. The regression
  analogue of temperature-scaling a classifier.

On held-out drives this takes the head from `z_mean −0.78, z_var 1.23` to
`z_mean −0.08, z_var 1.13` — and it flows through to `eskf` (z_mean −0.12, PASS).

## Exit gate (both required)
1. `nn` 60 s drift ≤ 0.5 × `physics` 60 s drift.
2. σ-calibration passes **both** `z_var ∈ [0.7, 1.4]` AND `|z_mean| ≤ 0.3`.
   The `z_mean` half is not optional: a model that under-predicts speed but
   inflates σ to match passes a variance-only check and then walks a Kalman
   filter off the road. `model/train.py` gates it; so does `eval/report.py`.

Both halves are now pinned by the suite: gate 1 (drift ratio) and the ONNX
round-trip in `tests/test_phase2_gate.py`, gate 2 (σ-calibration) in
`tests/test_nn_calibration.py`. On the committed checkpoint: nn 60 s drift
**8.4%** vs physics **33.2%** (ratio 0.25 ≤ 0.50), σ z_var **1.28** / z_mean
**−0.08** → PASS. `python -m pytest tests` — 163 passed.

**On synth these are pipeline-validation numbers, not competition claims** — the
synthetic speed cue (vibration ∝ speed) is clean by construction. The real
numbers come from IO-VNBD + your own recordings; the code, loss, and gate are
what transfer. This is stated in the proposal as the IO-VNBD-10Hz / high-rate
architecture split.

## Deliberate scope
- High-rate (200–400 Hz) vibration/pothole filtering is NOT learned here —
  IO-VNBD's phone stream is 10 Hz. That branch trains on your own phone logs.
- The slip head is wired but unsupervised on synth (car ≈ no slip); it gets a
  real target from Doppler cross-track residual on real data.
