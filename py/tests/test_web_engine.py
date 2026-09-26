"""The browser engine (docs/engine/*.js) against the Python it was ported from, under Node.

  1. core.js == the native core (libidr): ESKF, Doppler self-cal, re-mount detector, GNSS
     trust, on a random op sequence, to 1e-9;
  2. speednet.js == torch nn_real (mu, logvar, motion logits) on random windows;
  3. the whole loop, all four ablation stages, == edge_engine.py on a 200 Hz synthetic drive
     with an outage and a road network (a road along the drive + a parallel decoy);
  4. the same on a real IO-VNBD val drive with the real Coventry roads (skipped without
     the local data).

The JS engine runs SpeedNet in float64 from the float32 weights, the phone and Python run
it in float32, so the loop is compared with a tolerance, not bit for bit. Skipped without
Node.
"""
import json, math, os, shutil, subprocess
import numpy as np
import pytest

pytest.importorskip("core_bridge")
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not found")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNNER = os.path.join(HERE, "web_engine_runner.mjs")
MODEL = os.path.join(ROOT, "docs", "engine", "model")


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def node(job, tmp_path):
    src, dst = tmp_path / "job.json", tmp_path / "out.json"
    src.write_text(json.dumps(_clean(job)))
    subprocess.run([NODE, RUNNER, str(src), str(dst)], check=True, capture_output=True, timeout=1200)
    return json.loads(dst.read_text())


def test_exported_model_is_the_shipped_one():
    meta = json.load(open(os.path.join(MODEL, "speednet.json")))
    prof = json.load(open(os.path.join(ROOT, "android/app/src/main/assets/nn_real.profile.json")))
    assert meta["onnx_sha256"] == prof["onnx_sha256"] and meta["calib"] == prof["calib"]
    assert meta["pstop"] == prof["pstop"]
    assert json.load(open(os.path.join(MODEL, "profile.json"))) == prof


def test_core_matches_native(tmp_path):
    from core_bridge import Filter, SpeedCal, Align, gq_trust, gq_R, gq_spoof
    rng = np.random.default_rng(0)
    ops = [["noise", 0.01, 0.0004, 24.0], ["init", 3.0, -2.0, 0.4, 11.0]]
    for _ in range(400):
        k = rng.integers(0, 10)
        ops.append([["predict", 0.1, rng.normal(0, 0.1)], ["speed", rng.uniform(0, 20), rng.uniform(0.2, 3)],
                    ["gpos", rng.normal(0, 50), rng.normal(0, 50), rng.uniform(2, 30)],
                    ["gvel", rng.uniform(0, 20), rng.uniform(-3, 3), 0.3, 0.05],
                    ["cross", *(lambda b: [-math.cos(b), math.sin(b)])(rng.uniform(-3, 3)), rng.normal(0, 3), 2.0],
                    ["heading", rng.uniform(-3, 3), 0.1], ["zupt", rng.normal(0, 0.01)],
                    ["curv", rng.normal(0, 2), rng.normal(0, 0.4), 2.0], ["keep", int(rng.integers(0, 4))],
                    ["predict", 0.0025, rng.normal(0, 0.1)]][k])
    ops.insert(50, ["hsig", 0.3])
    f, rows = Filter(), []
    for op, *a in ops:
        {"noise": f.set_noise, "init": f.init, "predict": f.predict, "speed": f.update_speed, "zupt": f.update_zupt,
         "curv": f.update_curvature, "gpos": f.update_gnss_pos, "gvel": f.update_gnss_vel,
         "cross": f.update_crosstrack, "heading": f.update_heading, "keep": f.set_map_keep_speed,
         "hsig": f.set_heading_sigma}[op](*a)
        rows.append(list(f.state()) + list(f.cov()))
    cal = [[rng.uniform(0, 25), rng.uniform(0, 25), rng.uniform(0, 1), None if i < 300 else 0.02] for i in range(900)]
    sc, calr = SpeedCal(), []
    for vn, vd, w, lam in cal:
        sc.set_lambda(math.inf if lam is None else lam); sc.push(vn, vd, w); calr.append(list(sc.fit()))
    aln = [[*rng.normal([0, 0, 9.8], 1.5), *rng.normal(0, 0.1, 3), 0.1] for _ in range(3000)]
    for i in range(1500, 3000):                                   # a re-mount half way
        aln[i][0], aln[i][2] = aln[i][2], aln[i][0]
    al, alr = Align(), []
    for a in aln:
        al.update(*a); alr.append(list(al.get()))
    gq = [[rng.uniform(10, 45), int(rng.integers(0, 20)), int(rng.integers(0, 5)), rng.uniform(0.5, 8), rng.uniform(0, 40)]
          for _ in range(200)]
    gqr = [[gq_trust(*g), gq_R(gq_trust(*g)), gq_spoof(g[0], g[1], g[4], 0.0)] for g in gq]
    js = node(dict(mode="core", ops=ops, cal=cal, aln=aln, gq=gq), tmp_path)
    assert np.max(np.abs(np.array(js["filter"]) - np.array(rows))) < 1e-9
    assert np.max(np.abs(np.array(js["cal"]) - np.array(calr))) < 1e-9
    assert np.max(np.abs(np.array(js["aln"]) - np.array(alr))) < 1e-9
    assert sum(r[3] for r in alr) >= 1                              # the re-mount was seen
    assert np.max(np.abs(np.array(js["gq"]) - np.array(gqr))) < 1e-9


def test_speednet_matches_torch(tmp_path):
    import torch
    from model.nn_model import load_net
    net = load_net(os.path.join(ROOT, "py", "model", "nn_real.pt"))
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (12, 9, 20)).astype(np.float32)
    with torch.no_grad():
        mu, lv, _, cls = net(torch.from_numpy(X))
    js = node(dict(mode="speednet", x=X), tmp_path)
    assert np.max(np.abs(np.array([r["mu"] for r in js]) - mu.numpy())) < 1e-4
    assert np.max(np.abs(np.array([r["logvar"] for r in js]) - lv.numpy())) < 1e-4
    assert np.max(np.abs(np.array([r["cls"] for r in js]) - cls.numpy())) < 1e-4


STAGES = ["physics", "speednet", "head", "map"]


def _python_stages(imu, gn, outages, roads, stages=STAGES):
    from edge_engine import EdgeEngine, run
    from model.fusion_head import load_head
    head = load_head(os.path.join(ROOT, "py", "model", "fusion_head.pt"))
    fac = {"physics": lambda: EdgeEngine(head=None, dr_speed=False),
           "speednet": lambda: EdgeEngine(head=None),
           "head": lambda: EdgeEngine(head=head),
           "map": lambda: EdgeEngine(head=head, roads=roads, map_mode="hmm")}
    return {s: run(fac[s](), imu, gn, outages) for s in stages}


def _compare(py, js, imu, outages, tol_m):
    for s in py:
        a, b = py[s], np.array(js[s], float)
        assert a.shape == b.shape, s
        ok = a[:, 9] > 0
        d = np.hypot(a[ok, 1] - b[ok, 1], a[ok, 2] - b[ok, 2])
        assert np.array_equal(a[:, 7], b[:, 7]), f"{s}: dead-reckoning flags differ"
        assert d.max() < tol_m, f"{s}: max position difference {d.max():.4f} m"
        for t0, t1 in outages:                                       # the scored quantity: error at outage end
            i = np.searchsorted(imu["t"], t1) - 1
            assert abs(a[i, 1] - b[i, 1]) < tol_m and abs(a[i, 2] - b[i, 2]) < tol_m, s


def _job(imu, gn, outages, roads, every=1, stages=STAGES):
    return dict(mode="loop", stages=stages, every=every, outages=[list(o) for o in outages],
                imu=dict(t=imu["t"], acc=imu["acc"], gyro=imu["gyro"], mag=imu["mag"]),
                gnss={k: gn[k] for k in ("t", "lat", "lon", "speed", "bearing", "cn0", "sv", "navic", "masked")},
                roads=[dict(lat=w[0], lon=w[1], tunnel=bool(w[2]), oneway=bool(w[3]), highway=w[4]) for w in roads])


def test_loop_matches_edge_engine_on_a_synthetic_drive(tmp_path):
    pytest.importorskip("onnxruntime")
    from data.synth_rig import synth_rig, write_logger_csv
    from edge_engine import read_inputs
    root = tmp_path / "rig"; root.mkdir()
    write_logger_csv(synth_rig(240, 200, seed=4, grade="phone"), str(root))
    imu, gn = read_inputs(str(root / "imu.csv"), str(root / "gnss.csv"))
    # roads: one along the drive (every 5th fix), a parallel decoy 40 m to the side, a service road
    lat, lon = gn["lat"][::5], gn["lon"][::5]
    k = math.pi / 180 * 6_371_000.0
    roads = [(lat, lon, 0, 0, "primary"), (lat + 40 / k, lon, 0, 0, "residential"), (lat - 20 / k, lon, 0, 1, "service")]
    outages = [(100.0, 160.0)]
    py = _python_stages(imu, gn, outages, roads)
    js = node(_job(imu, gn, outages, roads), tmp_path)
    assert py["map"][:, 8].any()                                      # the map stage did snap
    _compare(py, js, imu, outages, tol_m=0.01)


DATA = os.environ.get("OFFMAPS_IOVNBD", os.path.expanduser("~/OffMaps-data/IO-VNBD-sync"))
COVENTRY = os.path.join(ROOT, "map", "coventry", "roads.bin")


@pytest.mark.skipif(not (os.path.isdir(DATA) and os.path.exists(COVENTRY)), reason="IO-VNBD data / Coventry roads not found")
def test_loop_matches_edge_engine_on_a_real_val_drive(tmp_path):
    pytest.importorskip("onnxruntime")
    import sys
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from osm_layers import read_roads_bin
    from data.iovnbd_sync import load_sync_dir
    from model.train_real import split_drives
    from phase8_eval import streams
    d = split_drives(load_sync_dir(DATA, verbose=False))["val"][0]
    imu, gn = streams(d)
    n = min(len(imu["t"]), 9000)                                       # 15 min at 10 Hz
    imu = {k: v[:n] for k, v in imu.items()}
    m = gn["t"] <= imu["t"][-1]
    gn = {k: v[m] for k, v in gn.items()}
    lat, lon = gn["lat"], gn["lon"]
    ways = read_roads_bin(COVENTRY, with_class=True)
    box = (lat.min() - 0.01, lat.max() + 0.01, lon.min() - 0.015, lon.max() + 0.015)
    roads = [w for w in ways if (w[0].max() >= box[0]) & (w[0].min() <= box[1]) & (w[1].max() >= box[2]) & (w[1].min() <= box[3])]
    t0 = imu["t"][0]
    outages = [(t0 + 300.0, t0 + 360.0), (t0 + 600.0, t0 + 720.0)]
    py = _python_stages(imu, gn, outages, roads)
    js = node(_job(imu, gn, outages, roads), tmp_path)
    _compare(py, js, imu, outages, tol_m=0.01)
