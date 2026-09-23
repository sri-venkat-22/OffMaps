"""End-to-end CLI. A real dataset arriving for the first time should either
score correctly or fail loudly -- never produce a plausible wrong table."""
import json, subprocess, sys, os
import numpy as np
import pytest
from tests.fixtures import MessySpec, REALISTIC, write_messy

PY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(*args, expect=0):
    r = subprocess.run([sys.executable, "-m", "eval.report", *map(str, args)],
                       cwd=PY_DIR, capture_output=True, text=True)
    assert r.returncode == expect, f"rc={r.returncode}\n{r.stdout}\n{r.stderr}"
    return r.stdout + r.stderr


def test_synthetic_gate_still_holds(tmp_path):
    """The Phase 0 exit gate: truth scores 0, the floors drift hard."""
    out = run("--model", "cv,zero,truth", "--out", tmp_path)
    tables = json.load(open(tmp_path / "results.json"))["tables"]
    for r in tables["truth"].values():
        assert r["drift_med"] == pytest.approx(0.0, abs=1e-6)
        assert r["cep95"] == pytest.approx(0.0, abs=1e-6)
    assert np.median([r["drift_med"] for r in tables["zero"].values()]) > 30
    assert np.median([r["drift_med"] for r in tables["cv"].values()]) > 10
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "drift_vs_duration.png").exists()


def test_describe_only_parses(tmp_path):
    p, _ = write_messy(tmp_path / "d", REALISTIC)
    out = run("--data", tmp_path / "d", "--describe", "--speed-col", "speed", "--kmph")
    assert "nothing scored" in out
    assert "speed vs GNSS track" in out
    assert not (tmp_path / "results.json").exists()


def test_describe_flags_wrong_units(tmp_path):
    p, _ = write_messy(tmp_path / "d", REALISTIC)
    out = run("--data", tmp_path / "d", "--describe", "--speed-col", "speed", expect=1)
    assert "SUSPECT" in out


def test_scores_a_messy_real_like_drive(tmp_path):
    p, _ = write_messy(tmp_path / "d", REALISTIC)
    out = run("--data", tmp_path / "d", "--model", "cv,truth", "--speed-col", "speed",
              "--ecu-speed-col", "wheel_speed", "--kmph", "--out", tmp_path / "o")
    tables = json.load(open(tmp_path / "o" / "results.json"))["tables"]
    for r in tables["truth"].values():
        assert r["drift_med"] == pytest.approx(0.0, abs=1e-6)
    for dur, r in tables["cv"].items():
        assert r["n"] == r["n_drift"], "every scored window must yield a drift%"
        assert r["span_med"] == pytest.approx(float(dur), rel=0.11)


def test_unknown_model_exits_cleanly(tmp_path):
    out = run("--model", "nope", "--out", tmp_path, expect=1)
    assert "unknown model" in out


def test_typo_in_column_flag_fails_loudly(tmp_path):
    p, _ = write_messy(tmp_path / "d", REALISTIC)
    r = subprocess.run([sys.executable, "-m", "eval.report", "--data", str(tmp_path / "d"),
                        "--describe", "--lat-col", "lattitude"],
                       cwd=PY_DIR, capture_output=True, text=True)
    assert r.returncode != 0 and "lattitude" in r.stderr


def test_manifest_replay_is_reproducible(tmp_path):
    p, _ = write_messy(tmp_path / "d", MessySpec(n=3600, hz=1.0, time_unit="s"))
    args = ["--data", tmp_path / "d", "--speed-col", "speed", "--model", "cv"]
    run(*args, "--out", tmp_path / "a")
    run(*args, "--manifest", tmp_path / "a" / "manifest.json", "--out", tmp_path / "b")
    ta = json.load(open(tmp_path / "a" / "results.json"))["tables"]["cv"]
    tb = json.load(open(tmp_path / "b" / "results.json"))["tables"]["cv"]
    assert ta == tb


def test_manifest_refuses_mismatched_data(tmp_path):
    import pandas as pd
    p, _ = write_messy(tmp_path / "d", MessySpec(n=3600, hz=1.0, time_unit="s"))
    run("--data", tmp_path / "d", "--speed-col", "speed", "--model", "cv",
        "--out", tmp_path / "a")
    df = pd.read_csv(p); df.iloc[:-50].to_csv(p, index=False)      # data changed
    out = run("--data", tmp_path / "d", "--speed-col", "speed", "--model", "cv",
              "--manifest", tmp_path / "a" / "manifest.json", "--out", tmp_path / "c",
              expect=1)
    assert "does not match" in out
    out = run("--data", tmp_path / "d", "--speed-col", "speed", "--model", "cv",
              "--manifest", tmp_path / "a" / "manifest.json", "--out", tmp_path / "c",
              "--allow-manifest-drift")
    assert "WARNING" in out and "NOT reportable" not in out.split("WARNING")[0]


def test_manifest_records_the_invocation(tmp_path):
    run("--model", "cv", "--out", tmp_path, "--seed", "5", "--n-per-duration", "4")
    doc = json.load(open(tmp_path / "manifest.json"))
    assert doc["version"] == 2
    assert doc["params"]["seed"] == 5 and doc["params"]["n_per_duration"] == 4
    assert doc["drives"] and "numpy" in doc["env"]
    assert "--seed" in doc["params"]["argv"]


def test_short_drive_without_60s_windows_still_plots(tmp_path):
    """_plot_examples hardcoded 60 s and would call subplots(1, 0)."""
    p, _ = write_messy(tmp_path / "d", MessySpec(n=45, hz=1.0, time_unit="s"))
    out = run("--data", tmp_path / "d", "--speed-col", "speed", "--model", "cv",
              "--out", tmp_path / "o")
    assert (tmp_path / "o" / "examples_cv.png").exists()


def test_no_usable_windows_exits_cleanly(tmp_path):
    p, _ = write_messy(tmp_path / "d", MessySpec(n=8, hz=1.0, time_unit="s"))
    out = run("--data", tmp_path / "d", "--speed-col", "speed", "--model", "cv",
              "--out", tmp_path / "o", expect=1)
    assert "no usable outage windows" in out
