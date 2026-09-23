"""Phase-1 exit gate, locked into the suite.

Phase 1's headline deliverable is the `physics` baseline -- "THE number every
later phase must beat" -- yet it was the one headline model with no regression
test (only a __main__ self-check in baseline/physics.py, which CI never runs).
Every later phase pins its gate (test_nn_calibration, test_speedcal_deming); this
does the same for Phase 1 so the baseline cannot silently regress above the floors.

The gate, verbatim from py/README_PHASE1.md:
  - `physics` scores below both floors (`cv`, `zero`) on the frozen set.
  - the documented exit-gate command runs end to end.
"""
import json, subprocess, sys, os
import numpy as np
import pytest

from data.io_vnbd import synth_drive
from data.outage import make_outages
from eval.metrics import score_outage
from eval.models import REGISTRY
import baseline.physics  # noqa: F401  (import registers the 'physics' model)

PY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _drift_med(drive, outs, name):
    rows = [score_outage(drive, o, REGISTRY[name](drive, o)) for o in outs]
    return np.nanmedian([r["drift_pct"] for r in rows])


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_physics_beats_both_floors(seed):
    """The competition-deciding claim: freezing Doppler speed and dead-reckoning
    heading off the gyro drifts far less than coasting straight (`cv`) or staying
    put (`zero`). Pinned on the frozen outage set across several drive seeds."""
    d = synth_drive(duration=600, seed=seed)
    outs = make_outages(d, seed=0)                 # frozen window set
    physics = _drift_med(d, outs, "physics")
    cv = _drift_med(d, outs, "cv")
    zero = _drift_med(d, outs, "zero")
    assert physics < cv, f"physics {physics:.1f}% should beat cv floor {cv:.1f}%"
    assert physics < zero, f"physics {physics:.1f}% should beat zero floor {zero:.1f}%"


def test_physics_without_gyro_reduces_to_cv():
    """Documented fallback (baseline/physics.py): with no gyro, heading is held
    constant, so the physics track must equal `cv` exactly -- freezing the same
    speed along the same frozen heading. A silent divergence here would mean the
    gyro dead-reckoning path had leaked into the no-gyro branch."""
    d = synth_drive(duration=600, seed=1)
    d.gyro_z = None                                # knock out the yaw-rate channel
    o = make_outages(d, seed=0)[0]
    pe, pn, pv, _ = REGISTRY["physics"](d, o)
    ce, cn, cvv, _ = REGISTRY["cv"](d, o)
    assert np.allclose(pe, ce, atol=1e-9) and np.allclose(pn, cn, atol=1e-9)
    assert np.allclose(pv, cvv, atol=1e-9)


def test_physics_exit_gate_command_runs(tmp_path):
    """The exact command README_PHASE1 tells you to run must succeed and, in the
    frozen results, physics must beat both floors at the aggregate -- the Phase-0
    gate test's twin for Phase 1."""
    r = subprocess.run(
        [sys.executable, "-m", "eval.report", "--model", "physics,cv,zero", "--out", str(tmp_path)],
        cwd=PY_DIR, capture_output=True, text=True)
    assert r.returncode == 0, f"rc={r.returncode}\n{r.stdout}\n{r.stderr}"
    tables = json.load(open(tmp_path / "results.json"))["tables"]
    med = {m: np.median([row["drift_med"] for row in tables[m].values()])
           for m in ("physics", "cv", "zero")}
    assert med["physics"] < med["cv"], med
    assert med["physics"] < med["zero"], med
    # and it beats the straight-line floor at every single duration it was scored
    for dur, row in tables["physics"].items():
        assert row["drift_med"] < tables["cv"][dur]["drift_med"], (dur, med)
