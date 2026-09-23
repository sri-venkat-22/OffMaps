# Phase 1 — Physics baseline + logger app

Two coupled deliverables. The Python side gives the **number every later phase
must beat**; the Android side is how you get real drives to feed it.

## Physics baseline (`baseline/physics.py`)

The competition-deciding rule, in code: **never double-integrate accel.** Freeze
forward speed at the entry Doppler value, integrate it along a gyro-dead-reckoned
heading. Drift stays ~linear in distance instead of ~t². Run it against the floors:

    cd py && python -m eval.report --model physics,cv,zero --out ../out

`physics` sits far below `cv` (straight-line) and `zero` (frozen). On the noisy
synthetic drive it's ~30–45% at 60 s because the synth turns hard and speed is
frozen — a smoke test, not a headline. Real IO-VNBD / highway drives land in the
single digits to low-teens; that's the Phase-1 bar the NN head (Phase 2) beats.

## Logger app (`../android/`)

Records raw IMU + GNSS to CSV. Its output schema is the exact input schema of
`data/phone_log.py`, so a recorded drive replays with:

    python -m eval.report --phone ./mydrive --model physics --out ../out/mydrive

Includes the **Outage Simulator** toggle (marks `masked=1`, keeps GNSS as truth).
See `../android/README.md`.

## New this phase
- `data/io_vnbd.py`  — `Drive` gained an optional `gyro_z` (yaw rate); synth emits a realistic biased+noisy gyro.
- `baseline/physics.py` — the `physics` model (registered on import).
- `data/phone_log.py` — loads `imu.csv`+`gnss.csv`, gravity-projects gyro to yaw, resamples to a 10 Hz grid.
- `eval/report.py`   — `--phone DIR`, `--hz`, `--yaw-sign` flags.

## Exit gate (met)
- `physics` scores below both floors on the frozen set. ✅
- A recorded phone log replays through the identical harness (`--phone`). ✅

Now enforced, not just asserted: `tests/test_physics_baseline.py` pins the gate
(physics beats `cv` and `zero` on the frozen outage set, across drive seeds and
end-to-end through the exit-gate command) so it can't silently regress the way
every later phase's gate is already pinned. `python -m pytest tests` — 163 passed.
