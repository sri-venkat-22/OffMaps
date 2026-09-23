"""Phase-2 exit gate + ONNX deliverable, locked into the suite.

Gate 2 (sigma-calibration: z_var in [0.7,1.4] AND |z_mean|<=0.3) is already
pinned in test_nn_calibration.py. Two Phase-2 claims were NOT pinned:

  - Gate 1: `nn` 60 s drift <= 0.5 * `physics` 60 s drift -- the headline
    "the NN head halves the physics baseline" claim, untested until now.
  - The ONNX deliverable: README says training "exports ONNX ... run on device
    via onnxruntime", but the exporter's bare-except used to swallow a missing
    `onnx` and ship only TorchScript. These pin that the graph exports AND that
    onnxruntime reproduces the torch net -- i.e. the deployed net is the gated net.

All skip cleanly when torch / the checkpoint / the onnx tooling is absent.
"""
import os
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from data.io_vnbd import synth_drive
from data.outage import make_outages
from eval.metrics import score_outage
from eval.models import REGISTRY
import baseline.physics  # noqa: F401  registers 'physics'
from model.nn_model import DEFAULT

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT), reason="no trained checkpoint")


@pytest.fixture(scope="module")
def net():
    from model.nn_model import load_net
    return load_net()


def test_nn_halves_physics_drift_at_60s(net):
    """Gate 1. On held-out drives (seeds 0-2, the report's own test set) the NN
    head's 60 s drift must be at most half the physics baseline's -- the whole
    reason Phase 2 exists. Registering `nn` requires the checkpoint, loaded above."""
    import model.nn_model  # noqa: F401  registers 'nn' (import has the side effect)
    nn_d, ph_d = [], []
    for i in range(3):
        d = synth_drive(f"synth{i}", duration=900, seed=i)
        for o in make_outages(d, seed=0):
            if o.duration_s == 60:
                nn_d.append(score_outage(d, o, REGISTRY["nn"](d, o))["drift_pct"])
                ph_d.append(score_outage(d, o, REGISTRY["physics"](d, o))["drift_pct"])
    nn60, ph60 = np.nanmedian(nn_d), np.nanmedian(ph_d)
    assert nn60 <= 0.5 * ph60, f"gate 1: nn {nn60:.1f}% must be <= 0.5*physics {ph60:.1f}%"


def test_onnx_export_roundtrips_through_onnxruntime(net, tmp_path):
    """The ONNX deliverable, from the committed checkpoint: it must export and
    onnxruntime must reproduce every head (mu/logvar/slip/cls) to <1e-4, across
    batch sizes -- nn_model feeds the net a whole outage of 1 s steps at once, so
    the dynamic batch axis has to hold, not just batch=1."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from model.export import export_onnx, verify_parity
    p = str(tmp_path / "nn.onnx")
    export_onnx(net, p)
    worst = verify_parity(net, p, batches=(1, 8, 37))   # raises if any head > 1e-4
    assert max(worst.values()) < 1e-4, worst


def test_shipped_onnx_is_the_shipped_checkpoint(net):
    """If model/nn.onnx is committed alongside nn.pt, it must BE nn.pt -- a stale
    export would silently deploy an un-gated net. Skips if the artifact is absent
    (it is a build product, regenerable via `python -m model.export`)."""
    onnx_path = os.path.splitext(DEFAULT)[0] + ".onnx"
    if not os.path.exists(onnx_path):
        pytest.skip("model/nn.onnx not built (run: python -m model.export)")
    pytest.importorskip("onnxruntime")
    from model.export import verify_parity
    worst = verify_parity(net, onnx_path, batches=(1, 8, 37))
    assert max(worst.values()) < 1e-4, worst
