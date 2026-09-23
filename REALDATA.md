# Real-data validation

Every headline number in the phase READMEs is on **synthetic** data (the first
real-data result is below). The READMEs are honest about it: the synth speed cue is clean by construction, so those are
*pipeline-validation* numbers, not competition claims. This doc is the
push-button path from an **actual drive** to a *reportable* number.

The `--data` (IO-VNBD) and `--phone` loaders already exist and are tested; the
only missing ingredient is a real drive. Drop one in and run one command.

## One command

    cd py

    # IO-VNBD export (a dir of drive CSVs)
    PYTHONPATH=. python3 -m validate_realdata --data /path/to/iovnbd_drives \
        --preset io-vnbd --kmph          # --kmph if the speed column is km/h

    # A phone log recorded by the app (imu.csv + gnss.csv from a drive_<ts> folder)
    PYTHONPATH=. python3 -m validate_realdata --phone /path/to/drive_1699 --yaw-sign -1

Output lands in `out/realdata/<name>/`:

| file | what |
|---|---|
| `REPORT.md` | the human-readable validation report (verdict, provenance, tables) |
| `results.json` | machine-readable summary (QC, provenance, per-duration tables) |
| `manifest.json` | the frozen outage windows, pinned to the data's sha256 |
| `drift_vs_duration.png` | median drift % vs outage length (10 % ISRO line) |
| `examples_<model>.png` | dead-reckoned track vs truth for the longest windows |

## The reportable stamp (what makes a number honest)

`validate_realdata` prints **REPORTABLE** or **NOT REPORTABLE** and writes the
same verdict at the top of `REPORT.md`. It refuses the stamp — with the reason —
when any of these fail. It never fabricates:

1. **Not synthetic / not `--smoke`.** Synthetic input (or an explicit `--smoke`
   dev run) is always stamped *pipeline smoke, not a competition claim*.
2. **QC passes.** The speed column and the GNSS ENU track must agree (ratio in
   [0.85, 1.15]). A mismatch means wrong units or a wheel-speed scale error, and
   every drift % would be silently off — so it blocks the stamp (`--kmph` /
   `--speed-col` fix it). This is the check that catches the classic "km/h read
   as m/s" mistake before it becomes a plausible-looking table.
3. **Enough windows.** At least `MIN_WINDOWS` (5) usable outage windows survive
   the distance/gap/span filters; below that a "number" is noise.
4. **Manifest consistent.** When reproducing with `--manifest`, the data's
   sha256/columns/units must still match what was frozen, or the number can't be
   reproduced and isn't reportable.

## Provenance (so a number can be audited)

`REPORT.md` records the sha256 of every input file, the resolved column/unit
mapping, the environment (python/numpy/pandas/platform), the seed, and the exact
command to reproduce. The manifest additionally pins each drive's fingerprint, so
re-running with `--manifest out/realdata/<name>/manifest.json` recomputes the
*same* windows and refuses if the data underneath changed.

## First real result — IO-VNBD, phone IMU vs vehicle GNSS (2026-09-23)

**This is the repo's first number on real data.** Report: `py/out/realdata/io-vnbd-sync/`
(REPORT.md, results.json, manifest.json pinned to the input hashes, plots).

    cd py
    PYTHONPATH=. python3 -m validate_realdata --layout iovnbd-sync \
        --data "/Volumes/Big Data/miscellaneous/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset" \
        --out out/realdata/io-vnbd-sync

Data: the public IO-VNBD release (github.com/onyekpeu/IO-VNBD, 2.2 GB, Git-LFS)
on the external SSD. Each drive folder pairs `S-*.csv` (smartphone accel + gyro,
10 Hz) with `V-*.csv` (vehicle survey GNSS + ECU, 10 Hz). `data/iovnbd_sync.py`
scores the **phone** sensors against the **vehicle** GNSS — the product's real
situation. Four things in the raw files are not what the headers say; each was
measured, handled, and pinned by `tests/test_iovnbd_sync.py`:

| quirk | fix |
|---|---|
| S/V rows are not time-aligned (phone log has gaps the vehicle log lacks; S4 drops 312 s) | align on absolute time (phone `DATE` − whole-hour TZ), refine by yaw cross-correlation → residual skew ≈ ±0.3 s |
| the yaw gyro is the column labelled **"Pitch"** | yaw axis + sign picked per segment from data, recorded in QC |
| ECU yaw rate is CCW-positive, harness heading is compass (CW-positive) | `gyro_z = −yaw`; the unflipped version mirrored every turn (caught because gyro DR lost to a straight line at 10 s) |
| phone "GPS SPEED (Kmh)" is actually m/s | unused — speed truth is the vehicle file |

Segments whose phone yaw tracks the car's ECU yaw below corr 0.8 are **rejected,
not scored** (listed in the report): most Driver-E drives (corr 0.2–0.7 — phone not
rigidly mounted). Kept: 12 segments, ~14 h, Drivers A/B/D, yaw corr 0.96–1.00,
QC speed/track 0.998–1.004. **435 outage windows. Stamp: REPORTABLE.**

| model | 10 s | 30 s | 60 s | 120 s | 180 s | CEP50 |
|---|--:|--:|--:|--:|--:|--:|
| `physics` (entry speed + phone gyro) | **10.6 %** | **20.5 %** | **31.3 %** | **34.6 %** | **37.8 %** | **113 m** |
| `nn` (SpeedNet, synthetic-trained) | 65.4 % | 61.2 % | 48.8 % | 44.7 % | 42.0 % | 190 m |
| `eskf` | 78.2 % | 71.6 % | 70.7 % | 70.3 % | 66.6 % | 275 m |
| `eskf_map` | 78.5 % | 74.3 % | 67.6 % | 59.2 % | 56.2 % | 255 m |
| `cv` (straight line) | 27.5 % | 51.4 % | 82.1 % | 85.2 % | 79.5 % | 257 m |

(median drift % of distance travelled; ISRO limit 10 %)

**What it says, honestly:**

1. **Nothing meets the 10 % limit on real data.** The synthetic headline numbers
   do not transfer.
2. **Heading is solved; speed is the whole problem.** Phone-gyro heading follows
   every turn (physics' cross-track error is ~3× below cv's); the remaining error
   is along-track, from speed.
3. **SpeedNet does not transfer from synthetic to real.** Speed MAE ≈ 5–6 m/s and
   σ-calibration fails hard (z_mean ≈ −4.5, z_var ≈ 25: it under-reads and is
   over-confident). Because the ESKF trusts that σ, `eskf` is *worse* than the
   plain physics baseline. The phone accelerometer here also correlates only
   weakly (0.0–0.44) with the vehicle's true along-track acceleration.
4. `eskf_map` builds its road map from the drive's own true track (an oracle
   map), so it is an upper bound for map mode, not a deployable number.

**Next:** train/fine-tune SpeedNet on real phone IMU with vehicle-speed labels
(IO-VNBD Drivers A/B/D, **held out by drive** — never window-split), and re-fit
its σ-calibration on real validation drives so the ESKF stops trusting a biased
speed. Until that happens, `physics` is the honest real-data baseline to beat.

## SpeedNet retrained on real data (2026-09-23)

`model/train_real.py` retrains the **same** SpeedNet (features, β-NLL +
consistency loss) on real phone IMU with vehicle-GNSS speed labels, starting from
the synthetic weights. Output: `model/nn_real.pt` (+ `nn_real.json` with the
split, per-epoch history and the calibration decision). `model/nn.pt`, the
synthetic gates and the Android `nn.onnx` are untouched.

**Split: by drive, fixed before any result was seen.** Train M, S1, S2, S4 (9.5 h);
val S3b, S3c (1.2 h, early stopping + calibration); **test Y1 (Driver D, never
trained on) + S3a** (2.5 h). Early stopping picked epoch 3; after that train loss
kept falling while val error did not, so more epochs of this data overfit.

    cd py
    PYTHONPATH=. python3 -m model.train_real --data ~/OffMaps-data/IO-VNBD-sync
    PYTHONPATH=. python3 -m validate_realdata --layout iovnbd-sync --data ~/OffMaps-data/IO-VNBD-sync \
        --include '/Y1/|/S3a/' --n-per-duration 30 \
        --manifest out/realdata/io-vnbd-test-synthnn/manifest.json --nn-ckpt model/nn_real.pt \
        --out out/realdata/io-vnbd-test-realnn

(`~/OffMaps-data/IO-VNBD-sync` is a local copy of the SSD's `Categorised IOVNB
Dataset`, 216/216 files, CSV sha256 identical — the SSD kept self-ejecting.)

**Held-out test drives, same 256 frozen windows** (`--manifest` replay; median drift %):

| model | 10 s | 30 s | 60 s | 120 s | 180 s | CEP50 |
|---|--:|--:|--:|--:|--:|--:|
| `physics` | **16.8** | **22.0** | 28.8 | 27.5 | 25.2 | 93 m |
| `nn` synthetic (`nn.pt`) | 85.6 | 73.1 | 67.9 | 48.8 | 43.3 | 218 m |
| **`nn` real (`nn_real.pt`)** | 22.7 | 25.1 | **19.1** | **17.3** | **17.2** | **73 m** |
| `eskf` synthetic | 87.6 | 78.5 | 74.8 | 66.2 | 59.9 | 268 m |
| `eskf` real | 33.5 | 40.0 | 41.6 | 46.2 | 52.5 | 150 m |

Reports: `out/realdata/io-vnbd-test-synthnn/` (before), `out/realdata/io-vnbd-test-realnn/` (after).

**Honest reading:**

1. **The real-trained net beats physics on every outage ≥ 60 s** (19 % vs 29 % at
   60 s, CEP50 73 m vs 93 m) — the first model in this repo to beat the physics
   floor on real data. Physics still wins ≤ 30 s: frozen entry speed is accurate
   for a few seconds. Still **not** at the ISRO 10 % limit.
2. **Calibration was chosen on val, and the test set was scored twice.** The
   Phase-2 affine calibration raised val MAE 5.53 → 8.03 m/s (visible in the
   training log *before* any test scoring); scored anyway it gave nn 40 % at
   60 s (`out/realdata/io-vnbd-test-realnn-affine/`). The rule "keep affine only if
   it lowers val MAE, else variance-only σ" (`train_real.calibrate`, pinned by
   `tests/test_train_real.py`) picks variance-only from val alone. Both test runs
   are kept on disk; treat the after-number as having one look of test exposure.
3. **The ESKF is now the weak link, not the net:** `eskf` (41.6 % at 60 s) is
   worse than the `nn` it fuses (19.1 %). Its noise settings were tuned on
   synthetic data (`core_bridge._run`: "tune vs real data") and its σ is biased
   (z_mean −0.68). σ on test is also over-cautious for `nn` (z_var 0.48): the val
   set is only two drives, one of them (S3c) an outlier with near-zero accel
   correlation.

**Next:** (a) re-tune the ESKF process/measurement noise on the **val** drives;
(b) more/cleaner val drives (k-fold by drive) for a steadier calibration;
(c) a physics→nn hand-over for short outages, designed on val, not test.

## Driver-E data and ESKF tuning (2026-09-23)

**More training data did not help.** Adding the 64 Driver-E drives (12.0 h synced
at the looser yaw corr ≥ 0.3, *train only*; `--extra-min-corr 0.3`) doubled
training to 21.5 h but gave best val MAE 5.58 m/s vs 5.53 without it. The rule
fixed before training was "adopt only if val improves", so `model/nn_real.pt`
stays; `model/nn_real_e.pt` is kept for reference. The other 27.9 h of IO-VNBD
(A1–A13, I, T1–T11) has no vehicle file, hence no speed truth, and is unused.

**ESKF tuned on val** (`model/tune_eskf.py`; coordinate descent, 2 passes, 172
val windows, test drives never loaded). Settings now live in the checkpoint
(`eskf_cfg`); `nn.pt` has none and keeps the synthetic defaults
(`core_bridge.ESKF_DEFAULT`, pinned by `tests/test_eskf_config.py`).

| val (mean of per-duration median drift) | |
|---|--:|
| synthetic defaults | 50.9 % |
| ZUPT off (`zupt_v=None`) — the big one | 28.1 % |
| + curvature update off, trust NN speed (`srw=24`, `sig_scale=0.25`) | **27.4 %** |
| NN alone, for reference | 26.7 % |

Why: with a real, slightly under-reading net, "NN v < 0.5 → ZUPT" fired on
slow-but-moving traffic, clamping v = 0 **and** taking the yaw rate as gyro bias
(ZARU), which then rotated the heading. The curvature update (v = a_lat / ψ̇)
trusted a phone accelerometer that barely tracks the car here.

**Test (same 256 frozen windows; third look at the test set):**

| model | 10 s | 30 s | 60 s | 120 s | 180 s | CEP50 |
|---|--:|--:|--:|--:|--:|--:|
| `eskf` synthetic settings | 33.5 | 40.0 | 41.6 | 46.2 | 52.5 | 150 m |
| **`eskf` tuned** | 22.0 | 24.3 | **20.2** | **17.1** | **17.3** | **72 m** |
| `nn` alone | 22.7 | 25.1 | 19.1 | 17.3 | 17.2 | 73 m |
| `physics` | **16.8** | **22.0** | 28.8 | 27.5 | 25.2 | 93 m |

The tuned filter essentially follows the NN speed — correct when that is its only
speed source during an outage; its value is honest covariance and folding in
GNSS/map aiding when present. Still not at ISRO 10 %. `eskf_map` (oracle map)
got *worse* than `eskf` with these settings: its corridor/heading updates were
also tuned on synthetic and are untested on real maps.

**On the phone (2026-09-23):** the app now runs `nn_real` with these settings,
read from `assets/nn_real.profile.json`. Porting it exposed a live-loop GNSS lockout
(the old app rejected ~99 % of real fixes). Fix and live-loop numbers on val
drives are in README_PHASE6.md §6c: 11.3 % median 60 s drift with Doppler self-cal
warmed up by the preceding GNSS, versus 19–20 % for the cold-start harness above.
Later the same day: physics→NN hand-over, keep-speed road snapping, and leveling
of the NN inputs (README_PHASE6 §6d–6e). Final held-out live-loop result (4th look
at test): 11.2 / 17.1 / 17.0 / 17.3 % at 10/30/60/120 s, vs physics 11.2 / 17.2 /
23.7 / 29.5 %.

## Getting a real drive

- **IO-VNBD** (Onyekpe et al., WMG Warwick). For the synchronised release use
  `--layout iovnbd-sync` (above). For a generic one-CSV-per-drive export, point
  `--data` at the CSVs with `--preset io-vnbd` and fix columns/units with the
  `--*-col` / `--kmph` flags. `--describe` (in `eval.report`) parses + QCs without
  scoring, so a column/unit mistake surfaces in a second.
- **A phone recording** — drive a route with the Phase-1 logger or the Phase-6 nav
  app, then `adb pull` the `drive_<ts>` folder and pass it to `--phone`. Flip
  `--yaw-sign` once if the dead-reckoned track mirrors the turns.

The scaffold itself is verified by `tests/test_realdata_scaffold.py` (artifacts +
provenance always written; smoke and QC-failure both blocked; clean input earns
the stamp) on synthetic fixtures that are explicitly **never** treated as
reportable; the IO-VNBD loader by `tests/test_iovnbd_sync.py`.
