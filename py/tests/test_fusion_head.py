"""Phase 8b: the learned fusion head (model/fusion_head.py).

  - causal: its output for second t of an outage never depends on later seconds, so
    one trained 180 s sequence serves every shorter outage (and the phone's streaming
    step-by-step run equals the batch run);
  - the ensemble's sigma includes the members' disagreement;
  - the phone ships exactly this checkpoint: assets/fusion_head.json holds the same
    weights (FusionHead.kt runs them; its port is checked in test_kotlin_ports.py) and
    the nn_real profile points at it.
"""
import json, os
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from model.fusion_head import FusionHead, HeadRunner, combine, load_head, N_CTX, N_STEP, OUT   # noqa: E402
import phase6_check as P                                                                         # noqa: E402


def test_causal_and_streaming_equals_batch():
    torch.manual_seed(0)
    net = FusionHead(16).eval()
    c = torch.randn(3, N_CTX); x = torch.randn(3, 50, N_STEP)
    with torch.no_grad():
        v_full, lv_full, _ = net(c, x)
        v_pre, _, _ = net(c, x[:, :20])
        h = None; vs = []
        for t in range(50):
            v, _, h = net(c, x[:, t:t + 1], h); vs.append(v[:, 0])
    assert torch.allclose(v_full[:, :20], v_pre, atol=1e-6)
    assert torch.allclose(v_full, torch.stack(vs, 1), atol=1e-5)


def test_ensemble_sigma_includes_disagreement():
    v, s = combine([10.0, 14.0], [1.0, 1.0])
    assert v == 12.0 and abs(s - np.sqrt(1.0 + 4.0)) < 1e-12


@pytest.mark.skipif(not os.path.exists(OUT), reason="no trained head")
def test_phone_ships_this_head():
    prof = P.load_profile()
    assert prof["live"]["fusion_head"] == "fusion_head.json"
    with open(os.path.join(P.ASSETS, "fusion_head.json")) as f:
        doc = json.load(f)
    ck = torch.load(OUT, map_location="cpu")
    assert doc["format"] == "offmaps-fusion-head/1" and len(doc["members"]) == len(ck["states"])
    for mem, sd in zip(doc["members"], ck["states"]):
        for k, v in sd.items():
            assert np.max(np.abs(np.asarray(mem[k]) - v.numpy())) < 1e-6, k          # float32 rounded to 8 decimals
    r = load_head()
    st = r.start(dict(v0=10.0, k=1.0, c=0.0, pre=[(10.0, 8.0)] * 60))
    v, s = r.step(st, dict(vnn=8.0, sig=2.0, tau=1.0, acc=np.tile([0, 0, 9.81], (10, 1)),
                           gyro=np.zeros((10, 3))))
    assert 0.0 <= v < 40.0 and s > 0.0
