"""Phase 8a: the edge engine (edge_engine.py) -- the phone's live loop at ANY IMU rate.

Pinned:
  - a 200 Hz FOG drive (data/synth_rig.py) read from the phone-logger CSVs: one
    output row per IMU sample, tracks GNSS, a simulated outage makes it dead-reckon
    (error grows) and GNSS pulls it back;
  - throughput: the whole CLI path (parse -> fuse -> write) keeps up with 200 Hz
    by a wide margin (the benchmark, edge_bench.py, measures ~280x real time);
  - the merged single-CSV input (GNSS columns only on fix rows) gives the same track
    as the two-file input;
  - at 10 Hz on the REAL IO-VNBD val drives the engine reproduces the validated
    live loop (phase6_check.run) statistically (skipped without the local data).
"""
import os, time
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("core_bridge")
from data.synth_rig import synth_rig, write_logger_csv     # noqa: E402
from edge_engine import EdgeEngine, read_inputs, run, write_output, main as cli   # noqa: E402


@pytest.fixture(scope="module")
def fog200(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("fog200"))
    rig = synth_rig(240, 200, seed=2, grade="fog")
    write_logger_csv(rig, root)
    return rig, root


def test_fog_200hz_tracks_dead_reckons_and_recovers(fog200):
    rig, root = fog200
    imu, gn = read_inputs(os.path.join(root, "imu.csv"), os.path.join(root, "gnss.csv"))
    out = run(EdgeEngine(), imu, gn, [(120.0, 150.0)])
    assert len(out) == len(rig["t"])                                  # positions at the input rate
    err = np.hypot(out[:, 1] - rig["e"], out[:, 2] - rig["n"])
    assert np.median(err[:120 * 200]) < 5.0                           # GNSS on: tracks
    assert out[149 * 200, 7] == 1 and out[100 * 200, 7] == 0          # dead-reckoning flag follows the mask
    assert err[150 * 200 - 1] > 2 * err[120 * 200 - 1]                # outage: the fix is gone, error grows
    assert err[165 * 200] < err[150 * 200 - 1]                        # GNSS back: re-converges


def test_cli_keeps_up_with_200hz(fog200, tmp_path):
    _, root = fog200
    out = str(tmp_path / "track.csv")
    t0 = time.perf_counter()
    cli(["--dir", root, "--out", out, "--outage", "120:150"])
    dt = time.perf_counter() - t0
    n = len(pd.read_csv(out))
    assert n == 240 * 200
    assert n / dt > 20 * 200, f"{n / dt:.0f} samples/s: under 20x a 200 Hz stream"


def test_merged_csv_equals_two_files(fog200, tmp_path):
    rig, root = fog200
    imu = pd.read_csv(os.path.join(root, "imu.csv")); gn = pd.read_csv(os.path.join(root, "gnss.csv"))
    m = imu.merge(gn, on="t_ns", how="left")
    p = str(tmp_path / "merged.csv"); m.to_csv(p, index=False)
    a = run(EdgeEngine(), *read_inputs(os.path.join(root, "imu.csv"), os.path.join(root, "gnss.csv")))
    b = run(EdgeEngine(), *read_inputs(csv_path=p))
    assert np.nanmax(np.abs(a[:, 1:5] - b[:, 1:5])) < 1e-9


DATA = os.environ.get("OFFMAPS_IOVNBD", os.path.expanduser("~/OffMaps-data/IO-VNBD-sync"))


@pytest.mark.skipif(not os.path.isdir(DATA), reason="IO-VNBD sync data not found")
def test_10hz_reproduces_the_validated_live_loop_on_real_val():
    import math
    import phase6_check as P
    from data.iovnbd_sync import load_sync_dir
    from model.train_real import split_drives
    val = split_drives(load_sync_dir(DATA, verbose=False))["val"]
    A, B = [], []
    for d in val:
        k = math.pi / 180 * 6_371_000.0
        lat = d.meta["lat0"] + d.n / k; lon = d.meta["lon0"] + d.e / (math.cos(math.radians(d.meta["lat0"])) * k)
        g = d.gyro.copy(); g[:, 2] = -g[:, 2]                         # compass yaw -> device CCW
        imu = dict(t=d.t, acc=d.acc, gyro=g, mag=np.full((len(d), 3), np.nan))
        i = np.arange(0, len(d), 10); n = len(i)
        gn = dict(t=d.t[i], lat=lat[i], lon=lon[i], speed=d.speed[i], bearing=np.degrees(d.heading[i]) % 360,
                  cn0=np.full(n, 42.0), sv=np.full(n, 9), navic=np.full(n, 3), masked=np.zeros(n))
        outs = [(a, a + 60) for a in np.arange(d.t[0] + 180, d.t[-1] - 70, 240.0)]
        o = run(EdgeEngine(head=None), imu, gn, outs)
        ea = np.hypot(o[:, 1] - d.e, o[:, 2] - d.n); eb = P.run(d, outage=outs)[0]
        for a, b in outs:
            i0, i1 = np.searchsorted(d.t, [a, b]); i1 -= 1
            dist = np.trapezoid(d.speed[i0:i1], d.t[i0:i1])
            if dist >= 300:
                A.append(100 * ea[i1] / dist); B.append(100 * eb[i1] / dist)
    assert len(A) >= 10
    assert abs(np.median(A) - np.median(B)) < 4.0, (np.median(A), np.median(B))


def test_no_ai_stage_holds_the_entry_speed(fog200):
    """ablation.py's no-AI stage (dr_speed=False): while dead-reckoning nothing updates the
    speed, so the filter coasts on the speed it had at outage entry; the default loop
    does update it (SpeedNet / head). GNSS still recovers both."""
    rig, root = fog200
    imu, gn = read_inputs(os.path.join(root, "imu.csv"), os.path.join(root, "gnss.csv"))
    held = run(EdgeEngine(head=None, dr_speed=False), imu, gn, [(120.0, 150.0)])
    live = run(EdgeEngine(head=None), imu, gn, [(120.0, 150.0)])
    i0, i1 = 121 * 200, 149 * 200
    assert np.ptp(held[i0:i1, 4]) < 1e-9                              # speed frozen in the outage
    assert np.ptp(live[i0:i1, 4]) > 1e-3                              # the default loop moves it
    err = np.hypot(held[:, 1] - rig["e"], held[:, 2] - rig["n"])
    assert err[170 * 200] < 10.0                                      # GNSS pulls it back
