"""The phone's pothole markers: the road-hazard filter on top of the Phase-7c detector.

The adaptive detector alone (shock_k 8 over an EMA baseline) is relative to the
road's own texture, so on real roads it fires on ordinary roughness: replayed on
the first Redmi drives (Hyderabad, 2026-09-25, ~37 km) it marked ~27 "potholes"
per km, one every ~37 m. The app now also requires:
  - a vertical jolt >= SHOCK_MIN_PEAK (m/s^2, along gravity) within the shock,
  - >= SHOCK_GAP_S since the previous event (one pothole = one mark, both axles),
  - no earlier mark within SHOCK_MERGE_M metres (FusionEngine),
which leaves ~2 per km. Pinned here:
  1. native == oracle with the filter on (edges + windows);
  2. with the filter off, the detector is unchanged (tests/test_phase7c_vib.py);
  3. synthetic: a 2 g pothole is still found every time, road-texture jolts are not;
  4. real phone drives (skipped without drives/): <= 3 per km with the app's
     settings, where the unfiltered detector gives > 15.
"""
import glob, math, os, re
import numpy as np
import pytest

pytest.importorskip("core_bridge")
from core_bridge import Vib
from vib_ref import VibRef
from data.highrate import highrate_accel
from test_vib_parity import _run

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FE = os.path.join(ROOT, "android/app/src/main/java/com/offmaps/nav/FusionEngine.kt")
DRIVES = sorted(glob.glob(os.path.join(ROOT, "drives", "drive_*")))


def _const(name):
    m = re.search(rf"val {name} = ([0-9.]+)", open(FE).read())
    assert m, f"{name} not found in FusionEngine.kt"
    return float(m.group(1))


MIN_PEAK, GAP_S, MERGE_M, MIN_SPEED = (_const(n) for n in
                                       ("SHOCK_MIN_PEAK", "SHOCK_GAP_S", "SHOCK_MERGE_M", "SHOCK_MIN_SPEED"))

try:
    Vib()
    _BUILT = True
except FileNotFoundError:
    _BUILT = False
pytestmark = pytest.mark.skipif(not _BUILT, reason="native core not built (sh core/build.sh)")


@pytest.mark.parametrize("hz", [250.0, 400.0])
def test_filter_native_matches_oracle(hz):
    hr = highrate_accel(hz=hz, seed=5)
    ref, cpp = VibRef(hz), Vib(hz)
    for v in (ref, cpp):
        v.set_shock_filter(MIN_PEAK, GAP_S)
    e_ref, w_ref = _run(ref, hr)
    e_cpp, w_cpp = _run(cpp, hr)
    assert np.array_equal(e_ref, e_cpp)
    assert np.max(np.abs(w_ref - w_cpp)) < 1e-9


def _events(amp, hz=400.0):
    hr = highrate_accel(hz=hz, seed=2, pothole_amp=amp)
    v = Vib(hz); v.set_shock_filter(MIN_PEAK, GAP_S)
    e, _ = _run(v, hr)
    return hr, hr.t[e.astype(bool)]


def test_real_pothole_still_found_texture_jolt_not():
    hr, ev = _events(20.0)                       # a 2 g hit
    assert all(any(abs(e - p) < 0.2 for e in ev) for p in hr.pothole_t)
    assert len(ev) == len(hr.pothole_t)
    _, ev = _events(5.0)                         # the adaptive test alone flags these
    assert len(ev) == 0


def test_filter_off_is_the_old_detector():
    hr = highrate_accel(hz=400.0, seed=5)
    a, b = Vib(400.0), Vib(400.0)
    b.set_shock_filter(0.0, 0.0)
    ea, wa = _run(a, hr); eb, wb = _run(b, hr)
    assert np.array_equal(ea, eb) and np.array_equal(wa, wb)


def _drive_rate(root, filt):
    import pandas as pd
    imu = pd.read_csv(os.path.join(root, "imu.csv"), on_bad_lines="skip")
    gn = pd.read_csv(os.path.join(root, "gnss.csv"), on_bad_lines="skip")
    t = imu.t_ns.to_numpy(np.int64); a = imu[["ax", "ay", "az"]].to_numpy(float)
    ok = np.r_[True, np.diff(t) > 0]; t = t[ok]; a = a[ok]
    gn = gn[np.isfinite(gn.speed) & np.isfinite(gn.lat)]
    tg = gn.t_ns.to_numpy(np.int64); sp = gn.speed.to_numpy(float)
    spd = np.interp(t, tg, sp)
    lat = np.interp(t, tg, gn.lat.to_numpy(float)); lon = np.interp(t, tg, gn.lon.to_numpy(float))
    km = float(np.sum(sp[1:] * np.diff(tg) * 1e-9) / 1000)
    v = Vib((len(t) - 1) / ((t[-1] - t[0]) * 1e-9))
    if filt:
        v.set_shock_filter(MIN_PEAK, GAP_S)
    marks = []
    for k in range(len(t)):
        if v.push(a[k, 0], a[k, 1], a[k, 2], 0.0) and spd[k] >= MIN_SPEED:
            e = math.radians(lon[k]) * 6371000 * math.cos(math.radians(lat[k]))
            n = math.radians(lat[k]) * 6371000
            if filt and any(math.hypot(e - pe, n - pn) < MERGE_M for pe, pn in marks):
                continue
            marks.append((e, n))
    return len(marks), km


@pytest.mark.skipif(not DRIVES, reason="no phone drives in drives/")
def test_real_phone_drives_rate():
    # the longest drive carries half the distance; enough to pin the rate
    root = max(DRIVES, key=lambda d: os.path.getsize(os.path.join(d, "imu.csv")))
    n_old, km = _drive_rate(root, filt=False)
    n_new, _ = _drive_rate(root, filt=True)
    assert km > 5
    assert n_old / km > 15, (n_old, km)
    assert n_new / km <= 3.0, (n_new, km)
