# Phase 0 — Benchmark before anything else

The harness every later phase is graded by. **No model, no filter yet** — on
purpose. A benchmark written after the model always flatters the model.

Hardened for real logs: `synth_drive()` is uniform 10 Hz, exact, gapless and
never parks, so a harness that only ever sees it bakes in assumptions no real
drive satisfies. Everything below exists because a real CSV broke it.

## Run (no dataset needed)

    cd py
    pip install -r requirements.txt
    python -m eval.report --model cv,zero,truth --out ../out

Runs on a synthetic drive. `truth` must score 0 (oracle), `zero`/`cv` are the
floors to beat. Table + plots + frozen `manifest.json` land in `../out/`.

## Run on real IO-VNBD (or your phone logs)

**Parse first, score second.** `--describe` loads, QCs and prints the drive
without scoring anything, so a wrong column or unit shows up in a second
instead of as a plausible-looking table:

    python -m eval.report --data /path/to/csvs --describe \
      --time-col time --lat-col latitude --lon-col longitude \
      --speed-col speed --ecu-speed-col wheel_speed --kmph

    drive_A
      3300/3303 rows  1.00 Hz  60.0 min  gaps=3 (12s)  speed 0.0-26.7 m/s
      parked=22%  hdg=course-over-ground (speed-gated)  [3 warning(s)]
        ! dropped 3 non-increasing timestamp(s)
        ! 3 gap(s) totalling 12s, max 4.0s
        ! interpolated 25 NaN speed sample(s)
      speed vs GNSS track: ratio 1.032 over 2412 fast samples

That last line is the load-bearing one: it cross-checks the drift denominator
(the speed column) against its numerator (the GNSS track). Read km/h as m/s and
it reads 0.289 and the run exits non-zero. It compares per-sample steps over
fast samples, not total path lengths — position noise inflates a total by ~12%
on a clean 1 Hz log and would false-positive.

Then score:

    python -m eval.report --data /path/to/csvs --model cv --kmph --out ../out

Column flags because IO-VNBD / phone headers vary. `--preset io-vnbd` and
`--preset phone` fill in a starting set; explicit flags still win.

### Units are per-column
`--kmph` applies to `--speed-col` **only**. `--ecu-kmph` applies to
`--ecu-speed-col` and defaults to `--kmph`. A log with Doppler in m/s and wheel
speed in km/h is normal; one coupled flag made `ecu_doppler_scale()` return
exactly 1.0, which silently deletes the signal Phase 4's cross-vehicle gate is
built on.

### Time units are detected by sample interval, never by magnitude
`--time-unit {auto,s,ms,us,ns}`. Auto picks the unit whose implied rate lands in
0.5–500 Hz and **raises rather than guesses** when zero or several fit. The old
`if t.max() > 1e6: t /= 1000` rule was wrong by 1e3–1e6× for nanoseconds (what
our own Android logger writes), for microseconds, and for any millisecond log
shorter than 16.7 minutes — i.e. exactly a first real recording. A wrong time
axis does not crash; it rescales sample rate, window length and every drift
denominator, and still prints a plausible table.

## Files

| file | what |
|---|---|
| `data/timeaxis.py` | time-unit detection + axis repair (sort, drop non-increasing, find gaps) |
| `data/io_vnbd.py`  | `Drive`, `DriveQC`, strict CSV loader, `synth_drive()`, per-vehicle `ecu_doppler_scale()` |
| `data/outage.py`   | time-based outage windows, rejection accounting, manifest v2 (`dump`/`load`/`verify`) |
| `eval/models.py`   | DR baselines (`zero`, `cv`, `truth`) + registry — add real models here |
| `eval/metrics.py`  | drift %, CEP50/95, along/cross split, per-duration table, σ-calibration |
| `eval/report.py`   | the exit-gate CLI (`--describe`, `--manifest`, presets) |
| `tests/`           | 163 regression tests; `tests/fixtures.py` generates messy drives |

## What loading refuses to do silently

| real-log input | before | now |
|---|---|---|
| duplicate / backwards timestamps | kept; `dt` went negative | sorted, dropped, counted in QC |
| sampling gaps | invisible | measured; windows straddling one are rejected |
| NaN speed | window scored NaN, hidden by `nanmedian` | interpolated (speed is smooth); counted |
| NaN lat/lon | propagated into the error | row dropped — position is never invented |
| a typo'd `--lat-col` | fell through to a default | `KeyError` with the column list |
| a typo'd `--heading-col` | silently fell back to derived heading | `KeyError` — a typo and an omission are now different |
| `vehicleA/drive1.csv` + `vehicleB/drive1.csv` | both `drive1`; one drive vanished from the results dict and the other was scored against its row indices | ids are path-relative (`vehicleA/drive1`); duplicates raise |
| stopped vehicle | heading from `np.gradient` spun ~90°/sample, poisoning the turn label and the along/cross basis | course over ground is speed-gated and held through stops |

## Windows are cut by time, not by sample count

On the uniform synthetic grid the two are identical. On a real log with
dropouts they are not: a "10 s" window measured up to **15 s** of real driving,
so the per-duration table's own x-axis was wrong. Windows now carry `t0`, `t1`,
`span_s`, `dist_m` and `max_dt`, and a window is rejected if it

- spans more than ±10% off its nominal duration (`span_tol`),
- contains a sample gap over 3× the median interval (`--max-gap`),
- covers less than `--min-dist` (20 m) of ground — a parked window's drift% is
  NaN or enormous, and it is trivially easy, so scoring it flatters the model,
- overlaps a window already taken at that duration (windows are now disjoint,
  so CEP95 is computed over independent samples).

Rejections are printed, never silent:

    3 drive(s), 99/120 frozen outage windows (seed=0)
      rejected candidates: dist=1, overlap=888

The table prints both `n` (windows scored) and `nd` (windows that produced a
finite drift%). They used to be conflated, so on messy data the report showed
`n=8` while `drift_med` was a median of 3.

## The frozen test set is now actually freezable

`manifest.json` is v2 and pins the **data**, not just row indices: per-drive
file path + sha256 + row counts + time unit + rate + the column/unit flags used,
plus the sampler params, the argv, and numpy/pandas/python versions.

    python -m eval.report --data DIR --manifest ../out/manifest.json --model cv

replays the committed windows and **exits non-zero if the data underneath
changed** (`--allow-manifest-drift` overrides; results are then not reportable).
Before, `load_manifest` had no caller and `report.py` overwrote the file on
every run — the README told you to commit a file that nothing could read back.
v1 bare-list manifests still load.

## Exit gate (met)
`python -m eval.report --model cv,zero,truth` runs end to end and emits the
table + plots. `truth` scores 0 at every duration; `zero` and `cv` drift hard.
`python -m pytest tests` — **163 passed**.

## Register a real model later
Add to `eval/models.py`:

    @model("nn_cv")
    def nn_cv(drive, o):
        ...
        return pred_e, pred_n, v_pred, sigma   # sigma -> σ-calibration scored

Then: `python -m eval.report --model nn_cv,cv`.

σ-calibration is now actually scored by the report (it was documented here but
only ever called from `model/train.py`). It gates on **both** halves:

- `z_var ∈ [0.7, 1.4]` — an over/under-confident σ
- `|z_mean| ≤ 0.3` — a **biased** speed

The second is not optional. A model with a constant 2 m/s bias and a correctly
scaled σ has `z_var = 1.00` and passes a variance-only gate, while `z_mean = 4.0`
and a Kalman filter fed that speed drives off the road.

## Numbers moved — regenerate before quoting
The window set changed (time-based cutting, gap/parked/overlap rejection), so
per-duration numbers are **not comparable to those in READMEs 1–5**. All phase
gates still pass on the new set (Phase 3 parity 8e-15; Phase 4 recovery 152%;
Phase 5 cross-track 33.9 m → 0.00 m), but the tables in those READMEs are stale.
