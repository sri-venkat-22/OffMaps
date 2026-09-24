"""Throughput benchmark for the edge engine: can it keep up with a 200 Hz FOG (and faster)?

    PYTHONPATH=. python3 edge_bench.py [--minutes 30] [--rates 10,100,200,400] [--out ../out/edge_bench.json]

For each IMU rate: synthesize a drive with the physically consistent rig
(data/synth_rig.py; "fog" gyro grade at >= 100 Hz, "mems" below), write it as the
phone-logger CSV pair, then time the edge engine end to end exactly as the CLI
runs it: CSV parse -> fusion (predict at the input rate, 10 Hz alignment,
1 Hz SpeedNet ONNX, GNSS updates, a 60 s simulated outage) -> CSV write, one row
per IMU sample. Reports samples/s and the real-time factor (drive seconds per
wall-clock second); the requirement is >= 1x at 200 Hz, i.e. >= 200 samples/s.
Single core, single thread (ONNX Runtime intra-op threads = 1).
"""
from __future__ import annotations
import argparse, json, os, platform, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from data.synth_rig import synth_rig, write_logger_csv
from edge_engine import EdgeEngine, read_inputs, run, write_output


def bench(hz, minutes=30.0, seed=0, roads=None):
    dur = minutes * 60.0
    rig = synth_rig(dur, hz, seed=seed, grade="fog" if hz >= 100 else "mems")
    root = tempfile.mkdtemp(prefix=f"edge{int(hz)}_")
    write_logger_csv(rig, root)
    t0 = time.perf_counter()
    imu, gn = read_inputs(os.path.join(root, "imu.csv"), os.path.join(root, "gnss.csv"))
    t1 = time.perf_counter()
    eng = EdgeEngine(roads=roads)
    out = run(eng, imu, gn, [(dur / 2, dur / 2 + 60.0)])
    t2 = time.perf_counter()
    write_output(eng, out, os.path.join(root, "track.csv"))
    t3 = time.perf_counter()
    n = len(imu["t"])
    e = np.hypot(out[:, 1] - rig["e"], out[:, 2] - rig["n"])
    return dict(hz=hz, samples=n, drive_s=dur, read_s=t1 - t0, fuse_s=t2 - t1, write_s=t3 - t2,
                fuse_samples_per_s=n / (t2 - t1), end_to_end_samples_per_s=n / (t3 - t0),
                realtime_factor=dur / (t3 - t0), rows_out=int(len(out)),
                median_err_gnss_m=float(np.median(e[: int(dur / 2 * hz)])))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--rates", default="10,100,200,400")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "edge_bench.json"))
    a = ap.parse_args()
    res = []
    for hz in [float(x) for x in a.rates.split(",")]:
        r = bench(hz, a.minutes)
        res.append(r)
        print(f"{hz:5.0f} Hz: {r['samples']:>8,} samples  fuse {r['fuse_samples_per_s']:>9,.0f}/s  "
              f"end-to-end {r['end_to_end_samples_per_s']:>9,.0f}/s = {r['realtime_factor']:6.0f}x real time "
              f"(read {r['read_s']:.1f}s fuse {r['fuse_s']:.1f}s write {r['write_s']:.1f}s)  "
              f"GNSS-on err {r['median_err_gnss_m']:.1f} m", flush=True)
        assert r["end_to_end_samples_per_s"] >= hz, f"cannot keep up with {hz} Hz"
    meta = dict(machine=platform.platform(), processor=platform.processor(), python=platform.python_version(),
                single_thread=True, minutes=a.minutes, results=res)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
