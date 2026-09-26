# What we got wrong, and how we found it

A dead-reckoning result is only as good as the checks behind it. These are the mistakes we
made in OffMaps: our own bugs, results that flattered us, and features that did not work
on real data. Each one names what exposed it and what pins the fix, so it cannot quietly
come back.

## Bugs in our own pipeline

**1. SpeedNet was trained to predict the wrong second.** Training windows ended at the
*start* of the 1 s step being predicted, so the net learned to forecast the next second.
At inference the window ends at the step's *end*. This off-by-one-second train/serve skew
fed the σ head a speed from the wrong second, and it showed up as a bias in the
calibration. *Found by:* the σ-calibration check on validation drives.
*Fixed in:* `py/model/dataset.py` (`_steps`, the docstring tells the story).

**2. The app locked itself out of GNSS.** On real validation drives the first phone loop
rejected about 99 % of real fixes as "spoofed". Its median error was 12.6 km *with GNSS
on*. The chain:
- the NN speed was fused while GNSS was healthy;
- the spoof test compared the Doppler residual with a fixed 5 m/s, a threshold our spoof
  gate had never validated;
- there was no path back.

*Found by:* porting the live loop to the host and replaying real drives through it.
*Fixed:* speed aids only while dead-reckoning, the spoof test uses position χ² only, and
the filter re-seeds after 5 rejected fixes. *Pinned by:* `py/tests/test_phase6_shipped_model.py`
(lockout stress test). Write-up: `git show ec89099:README_PHASE6.md`.

**3. The phone's IMU path never ran, on any device.** The 10 Hz decimator computed
`t - Long.MIN_VALUE`, which overflows, so every sample was dropped. The same first emulator
run also found two more bugs:
- a fix at trust 0 counted as "healthy GNSS";
- self-calibration pushes weighted by trust 0 made the fit `k = NaN`.

*Found by:* the first run on an Android emulator. *Pinned by:* `Decimator.kt` and
`py/tests/test_kotlin_ports.py`, which compiles the Kotlin on the host and checks it against Python.

**4. The phone fed SpeedNet the wrong frame.** `nn_real` was trained on z-up (levelled)
accelerometer data; the phone gave it the raw device frame. On validation drives that moved
the net's bias from −4.5 to +0.5 m/s and its MAE from 5.53 to 6.58 m/s. *Fixed:* `Level.kt`
levels every sample with a slow gravity estimate, twin of `py/model/mount.py`.

**5. The re-mount detector cried wolf about 1,800 times an hour.** The first version compared
a 0.5 s gravity estimate with a snapshot at 2° slack, so ordinary driving set it off.
*Rebuilt* as a CUSUM on the angle between 5 s and 30 s gravity estimates: about one
false alarm per hour, and a 30° re-mount is found in 3 to 8 s (`core/align.cpp`).

**6. The pothole detector marked a "pothole" every 37 m.** Replayed on the first Redmi Note 9
Pro drives in Hyderabad (8 drives, 37 km), the adaptive shock test fired 27.3 times per km.
It reacted to ordinary road texture. *Fixed:* a mark now needs all three of:
- a vertical jolt of at least 1 g;
- at least 1 s since the previous mark;
- no earlier mark within 30 m.

That gives 1.7 marks per km, and a 2 g pothole is still found every time.
*Pinned by:* `py/tests/test_pothole_filter.py`, which replays a real drive.

**7. Turning the phone ended the drive.** The navigation screen did not handle rotation, so
Android re-created it, and its `onDestroy` stopped navigation and closed the drive
recording. A phone on a car mount is often sideways, so one rotation mid-drive would end
the session, and landscape also hid the map behind the panel. *Found by:* installing the
app on the Redmi, which was lying sideways. *Fixed:* the activity now handles rotation
itself (`configChanges` in the manifest). In landscape the panel becomes a side card, as
in Google Maps. Checked on the emulator: navigation keeps running across a rotation.

## Results that flattered us

**8. Oracle maps.** Every early road-snapping number used each drive's own GNSS track as the
"map". On the real OpenStreetMap network, greedy snapping made outages much worse: validation
drift went from 11.3 % to 31.7 %. A dead-reckoned position that is tens of metres off
along-track snaps to the wrong street at junctions. Snapping stayed off until the gated HMM
matcher, which does help: 13.6 → 11.7 % on train drives, and the paired bootstrap intervals
exclude zero at 30, 60 and 120 s. See `git show ec89099:README_PHASE8.md` (§8d) and `py/phase9_map_eval.py`.

**9. Zero-velocity updates.** A naive rule (ZUPT whenever SpeedNet reads under 0.5 m/s)
took validation drift from 11.3 % to 33.6 %. At 10 Hz, slow traffic looks like a stop, and
a false ZUPT also corrupts the gyro bias. It never shipped. A strict detector is built and
off until a stationary test on the target phone.

**10. Selection optimism we cannot fully remove.** The train-drive numbers for the fusion
head and the map carry some selection optimism, and the README says so wherever they
appear:
- each drive is scored by models that never saw it;
- but the head's inputs and epoch count were chosen by cross-validation on those drives;
- and the map's χ² gate was added after we saw their worst outages.

The held-out test split has been looked at five times in total, and nothing was ever
chosen on it. The validation set, about 1 h, is too small to separate the last stages.

## Things that do not work (yet)

**11. Drift under 10 %.** It is not reached. The shipped loop is at 11.7 % on train drives
(leave-one-drive-out) and 16.4 % on the test drives.

**12. The tunnel benchmark.** The PS example asks for under 100 m over 1 km. It is not met:
in the Hyderabad Mindspace Underpass scenario the shipped app ends a median 163 m off. In
steady cruising, simply holding the entry speed beats the AI speed there (`py/tunnel_scenario.py`,
`out/tunnel/summary.md`).

**13. An output nobody read.** SpeedNet has had a motion-class head (stopped / straight /
turning) since it was first trained, but nothing used it. Its raw probabilities were
mis-calibrated: a raw 0.6–0.8 meant "stopped" about 70 % of the time on validation, and it
almost never went past 0.8. It is now Platt-scaled on held-out outputs and ships as
p(stopped). Cross-fitted calibration error went from 0.020 to 0.012, and on the validation
drives from 0.058 to 0.035 (`py/model/pstop.py`, `out/pstop/pstop.json`).

## How we check ourselves

- **One engine, several implementations, each checked against a reference.**
  - The C++ core against a Python oracle to 1e-9.
  - The Kotlin ports, compiled on the host, against Python (`tests/test_kotlin_ports.py`).
  - The browser engine against the Python edge engine (`tests/test_web_engine.py`).
- **Every drive is scored by models that never saw it.** Out-of-fold SpeedNets and heads
  are used for the train drives (`py/ablation.py`).
- **Paired comparisons.** Every stage runs on the identical outages, and each step is
  reported with a paired bootstrap interval, not just a median.
- **Fixed windows.** The outage windows follow a fixed rule (after at least 3 min of GNSS,
  a fixed spacing) and are never hand-picked. The demo page opens on the median outage.
