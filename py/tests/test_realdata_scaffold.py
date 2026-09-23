"""Regression for the real-data validation scaffold (validate_realdata.py).

The scaffold's job is to turn a real drive into a committed, auditable report --
and, crucially, to never let a non-real or unvalidated run wear a "reportable"
stamp. These pin that gate:

  - artifacts + provenance (file hashes, env, reproduce cmd) are always written;
  - --smoke input is never reportable (the dev/pipeline path);
  - a QC failure (speed in km/h read as m/s) blocks the stamp with a reason;
  - clean, QC-passing input with enough windows DOES earn the stamp.

Baselines only (physics, cv) so it needs neither torch nor the native core.
"""
import csv
import os

import numpy as np

import validate_realdata as V
from data.io_vnbd import synth_drive, R_EARTH


def _write_phone_log(root, *, duration=200, seed=4, kmph_speed=False):
    """A phone-log fixture from a synthetic drive: real GNSS track+speed, flat IMU.
    Consistent lat/lon vs speed => QC passes, unless kmph_speed inflates speed x3.6
    without the --kmph flag (the classic units-mistake the QC gate must catch)."""
    d = synth_drive("fixture", duration=duration, seed=seed, hz=10)
    os.makedirs(root, exist_ok=True)
    k = np.pi / 180.0 * R_EARTH
    lat0, lon0 = 17.4, 78.5
    with open(os.path.join(root, "imu.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow("t_ns ax ay az gx gy gz mx my mz pressure light".split())
        for i in range(len(d)):
            w.writerow([int(d.t[i] * 1e9), 0, 0, 9.81, 0, 0, d.gyro_z[i], 0, 0, 0, 101325, 100])
    scale = 3.6 if kmph_speed else 1.0
    with open(os.path.join(root, "gnss.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow("t_ns lat lon speed bearing cn0_mean sv_used navic_sv masked".split())
        for i in range(0, len(d), 10):   # 1 Hz
            lat = lat0 + d.n[i] / k
            lon = lon0 + d.e[i] / (np.cos(np.radians(lat0)) * k)
            w.writerow([int(d.t[i] * 1e9), lat, lon, d.speed[i] * scale,
                        np.degrees(d.heading[i]), 40, 9, 3, 0])
    return root


def _run(root, out, extra=None):
    ap = V._build_argparser()
    args = ap.parse_args(["--phone", root, "--models", "physics,cv", "--out", out,
                          "--yaw-sign", "1.0", *(extra or [])])
    V._apply_preset(args, ap)
    return V.run(args)


def test_artifacts_and_provenance(tmp_path):
    root = _write_phone_log(str(tmp_path / "drive"))
    out = str(tmp_path / "out")
    s = _run(root, out, extra=["--smoke"])
    for f in ("REPORT.md", "results.json", "manifest.json"):
        assert os.path.exists(os.path.join(out, f)), f"missing {f}"
    # provenance pins the exact bytes that fed the run
    assert len(s["input_files"]) == 2
    for entry in s["input_files"]:
        assert len(entry["sha256"]) == 64
    assert s["env"].get("python") and s["env"].get("numpy")
    assert set(s["models"]) == {"physics", "cv"}


def test_smoke_is_never_reportable(tmp_path):
    root = _write_phone_log(str(tmp_path / "drive"))
    s = _run(root, str(tmp_path / "out"), extra=["--smoke"])
    assert s["reportable"] is False
    assert any("smoke" in r for r in s["reasons"])


def test_qc_failure_blocks_the_stamp(tmp_path):
    # speed written in km/h but NOT flagged --kmph -> loader reads it as m/s ->
    # speed/track ratio ~1/3.6 -> SUSPECT -> not reportable.
    root = _write_phone_log(str(tmp_path / "drive"), kmph_speed=True)
    s = _run(root, str(tmp_path / "out"))
    assert s["qc_ok"] is False
    assert s["reportable"] is False
    assert any("QC failed" in r for r in s["reasons"])


def test_clean_input_earns_the_stamp(tmp_path):
    # QC passes, enough windows, no --smoke, not synthetic-meta -> reportable.
    root = _write_phone_log(str(tmp_path / "drive"))
    s = _run(root, str(tmp_path / "out"))
    assert s["qc_ok"] is True
    assert s["n_windows"] >= V.MIN_WINDOWS
    assert s["reportable"] is True, s["reasons"]
