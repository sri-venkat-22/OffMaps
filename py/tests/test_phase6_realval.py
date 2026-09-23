"""The nav app's live loop on REAL IO-VNBD validation drives (never the test split).

Run through phase6_check's mirror of FusionEngine with the shipped profile. Before
the fix, the app's loop fused the NN speed while GNSS was healthy and used a fixed
5 m/s Doppler spoof test: on these drives ~99% of real fixes were rejected as
"spoofed" and the median error WITH GNSS on was 0.65-12.6 km. Pinned here:

  - GNSS on: the loop tracks it (no lockout);
  - 60 s outages (every 4 min): GNSS is re-admitted every time -- 10 s after it
    returns the estimate is back within a few metres;
  - road snapping (oracle map) makes outages better, not worse;
  - the dead-reckoned covariance is honest-or-cautious at outage end (position error
    inside the filter's 95 % ellipse for >= 80 % of outages, mean z^2 near 2). This
    is what an offline sigma calibration exists to deliver; the live loop gets it
    from the Doppler self-cal, so no k-fold recalibration of the net is needed.

Needs the local IO-VNBD sync copy (REALDATA.md); skipped where it isn't present.
Set OFFMAPS_IOVNBD to point elsewhere.
"""
import os
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("core_bridge")
DATA = os.environ.get("OFFMAPS_IOVNBD", os.path.expanduser("~/OffMaps-data/IO-VNBD-sync"))
if not os.path.isdir(DATA):
    pytest.skip(f"IO-VNBD sync data not found at {DATA}", allow_module_level=True)

from phase6_check import run                       # noqa: E402
from data.iovnbd_sync import load_sync_dir         # noqa: E402
from model.train_real import split_drives          # noqa: E402

VAL = split_drives(load_sync_dir(DATA, verbose=False))["val"]


@pytest.mark.parametrize("i", range(len(VAL)))
def test_real_val_drive_tracks_and_recovers(i):
    d = VAL[i]
    err_on = run(d)[0]
    assert np.median(err_on) < 2.0, f"GNSS on, median err {np.median(err_on):.1f} m (lockout?)"

    t0s = np.arange(d.t[0] + 180, d.t[-1] - 70, 240.0)
    outs = [(a, a + 60) for a in t0s]
    tr = {}
    err, fused, _, _ = run(d, outage=outs, trace=tr)
    ce, cn = np.asarray(tr["cov_e"]), np.asarray(tr["cov_n"])
    z2 = []
    for a, b in outs:
        i_end = int(np.searchsorted(d.t, b)) - 1
        back = err[min(len(d) - 1, i_end + 100)]
        assert back < 5.0, f"outage {a - d.t[0]:.0f}s: still {back:.1f} m off 10 s after GNSS returned"
        de, dn = fused[i_end, 0] - d.e[i_end], fused[i_end, 1] - d.n[i_end]
        z2.append(de * de / ce[i_end] + dn * dn / cn[i_end])
    z2 = np.asarray(z2)
    assert np.mean(z2 < 5.99) >= 0.8, f"over-confident: only {np.mean(z2 < 5.99):.0%} inside the 95% ellipse"
    assert np.mean(z2) < 6.0, f"over-confident: mean z^2 {np.mean(z2):.1f} (honest = 2)"


def test_road_snapping_helps_on_real_val():
    """With the profile's map settings (road updates may not change speed), snapping
    to the road must not make real outages worse -- before model/tune_map.py it
    did (60 s median 12.9 % -> 19.2 %). Map = each drive's own track (oracle)."""
    from model.tune_map import live_map_drift
    from phase6_check import load_profile
    prof = load_profile()
    off = live_map_drift(VAL, prof, use_map=False, durations=[60])[60]["median"]
    on = live_map_drift(VAL, prof, use_map=True, durations=[60])[60]["median"]
    assert on < off, f"road snapping made 60 s outages worse: {off:.1f}% -> {on:.1f}%"


def test_remount_detector_quiet_on_rigid_mounts():
    """core/align.cpp's re-mount detector on the real rigid-mount val drives (in a
    portrait dash-mount frame). The first version fired ~1800x/hour here and would
    have flashed "re-mount detected" non-stop; tuned on train+val: ~1/hour."""
    from core_bridge import Align
    portrait = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], float)
    n, steps = 0, 0
    for d in VAL:
        a = d.acc @ portrait.T; g = d.gyro @ portrait.T
        al = Align()
        for i in range(len(a)):
            al.update(*a[i], *g[i], 0.1); n += al.get()[3]
        steps += len(a)
    per_hour = n / (steps / 36000)
    assert per_hour <= 3.0, f"{per_hour:.1f} false re-mounts per hour"
