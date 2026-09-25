"""Phase 8 heading gate: magnetometer seed + turn-tilt yaw rate, car AND two-wheeler.

    PYTHONPATH=. python3 phase8_heading_gate.py [--out ../out]

Physics-exact synthetic rigs (data/synth_rig.py: device-frame IMU + magnetometer,
dash/handlebar mount rotated 12 deg yaw / 8 deg tilt, MEMS gyro, 100 Hz) driven
through the edge engine (the phone's loop). Only HEADING is scored here: the speed
model was trained on cars (IO-VNBD) and nothing about two-wheeler speed can be
claimed without a two-wheeler recording -- see the Phase 8 write-up §8f (git show ec89099:README_PHASE8.md).

Gate A (yaw rate about true vertical). 60 s GNSS outages every 3 min; heading
error at each outage end, per yaw mode:
  car          "fast" (the app's old 0.5 s gravity projection) under-reads every
               turn by cos(phi), phi = atan(v*psidot/g); "slow" and "coord" do not.
  two-wheeler  the phone leans with the bike, so "slow" is wrong too; only "coord"
               (tilt back by the coordinated-turn lean) holds heading.
  PASS: car slow/coord and two-wheeler coord beat "fast" by >= 2x in median outage-end
  heading error, and two-wheeler coord beats two-wheeler slow by >= 2x.

Gate B (magnetometer seed). Drives that START PARKED (no GNSS course) at 8 random
headings: the first-fix heading with the magnetometer vs without.
  PASS: median seed error < 20 deg (the sigma it is fused with) and < 1/3 of the
  no-magnetometer error.
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from data.synth_rig import synth_rig
from edge_engine import EdgeEngine, run
import heading_aids as HA


def _streams(rig, gnss_hz=1.0, seed=0):
    from data.synth_rig import enu_to_ll
    rng = np.random.default_rng(seed)
    step = int(round(rig["hz"] / gnss_hz)); idx = np.arange(0, len(rig["t"]), step)
    lat, lon = enu_to_ll(rig["e"][idx] + rng.normal(0, 2, len(idx)), rig["n"][idx] + rng.normal(0, 2, len(idx)))
    v = np.clip(rig["speed"][idx] + rng.normal(0, 0.1, len(idx)), 0, None)
    brg = np.where(v > 1.0, np.degrees(rig["heading"][idx]) % 360, np.nan)
    n = len(idx)
    gn = dict(t=rig["t"][idx], lat=lat, lon=lon, speed=v, bearing=brg, cn0=np.full(n, 40.0),
              sv=np.full(n, 12), navic=np.full(n, 3), masked=np.zeros(n))
    return dict(t=rig["t"], acc=rig["acc"], gyro=rig["gyro"], mag=rig["mag"]), gn


def gate_a(seeds=(0, 1, 2), minutes=20, hz=100.0):
    res = {}
    for veh in ("car", "two_wheeler"):
        for mode in ("fast", "slow", "coord"):
            errs = []
            for s in seeds:
                rig = synth_rig(minutes * 60, hz, seed=s, vehicle=veh)
                imu, gn = _streams(rig, seed=s)
                outs = [(a, a + 60.0) for a in np.arange(120.0, minutes * 60 - 70, 180.0)]
                o = run(EdgeEngine(vehicle=veh, yaw_mode=mode, head=None), imu, gn, outs)   # heading only
                for a, b in outs:
                    i = int(b * hz) - 1
                    errs.append(abs(np.degrees(HA.wrap(o[i, 3] - rig["heading"][i]))))
            res[(veh, mode)] = float(np.median(errs))
            print(f"  {veh:12s} yaw {mode:5s}: median heading error at 60 s outage end {res[(veh, mode)]:5.2f} deg "
                  f"(n={len(errs)}, max lean {np.degrees(np.abs(rig['lean']).max()):.0f} deg)", flush=True)
    ok = (res[("car", "slow")] * 2 <= res[("car", "fast")] and res[("car", "coord")] * 2 <= res[("car", "fast")]
          and res[("two_wheeler", "coord")] * 2 <= res[("two_wheeler", "fast")]
          and res[("two_wheeler", "coord")] * 2 <= res[("two_wheeler", "slow")])
    return ok, {f"{v}/{m}": x for (v, m), x in res.items()}


def gate_b(n=8, hz=50.0):
    rng = np.random.default_rng(3)
    e_mag, e_none = [], []
    for k in range(n):
        h0 = float(rng.uniform(-180, 180)); mount = "dash" if k % 2 else "flat"
        rig = synth_rig(90, hz, seed=10 + k, park_s=30, heading0_deg=h0, mount=mount,
                        hard_iron=tuple(rng.normal(0, 4, 3)))
        imu, gn = _streams(rig, seed=k)
        i = int(25 * hz)
        for use, acc in ((True, e_mag), (False, e_none)):
            o = run(EdgeEngine(use_mag=use, head=None), imu, gn)
            acc.append(abs(np.degrees(HA.wrap(o[i, 3] - rig["heading"][i]))))
    m, z = float(np.median(e_mag)), float(np.median(e_none))
    print(f"  parked start, first-fix heading error: magnetometer {m:.1f} deg (max {max(e_mag):.1f}), "
          f"none {z:.1f} deg (n={n})")
    return (m < 20.0 and m * 3 < z), dict(mag_median=m, mag_max=float(max(e_mag)), none_median=z)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out"))
    a = ap.parse_args()
    print("Gate A: yaw rate about true vertical (60 s outages)")
    ok_a, ra = gate_a()
    print("Gate B: magnetometer heading seed")
    ok_b, rb = gate_b()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "phase8_heading_gate.json"), "w") as f:
        json.dump(dict(gate_a=ra, gate_a_pass=ok_a, gate_b=rb, gate_b_pass=ok_b), f, indent=2)
    print(f"gate A {'PASS' if ok_a else 'FAIL'}   gate B {'PASS' if ok_b else 'FAIL'}")
    assert ok_a and ok_b


if __name__ == "__main__":
    main()
