"""Phase-7c: the slip head is wired but was unsupervised (dead) -- the loss now
has a slip term (model/loss.slip_loss). This proves the supervision PATH is live:
given a learnable lateral-slip residual target, the head + loss learn it end to
end (loss drops, prediction correlates with target). It does NOT claim synthetic
data contains real slip -- on synth the residual is zero by construction, so the
real-slip model still trains on phone logs. This is the learnability gate for the
wiring, run on an injected target.

Skipped cleanly when torch is absent.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from model.tcn import SpeedNet
from model.features import C, WIN
from model import loss as LOSS


def test_slip_loss_is_wired_into_total():
    """total() must actually add the slip term when a target is supplied."""
    torch.manual_seed(0)
    mu = torch.rand(2, 3); lv = torch.zeros(2, 3); dt = torch.rand(2, 3)
    logits = torch.randn(2, 3, 4); cl = torch.zeros(2, 3, dtype=torch.long)
    base = LOSS.total(mu, lv, dt, logits, cl)
    sp = torch.zeros(2, requires_grad=True); st = torch.ones(2)
    with_slip = LOSS.total(mu, lv, dt, logits, cl, slip_pred=sp, slip_true=st)
    assert float(with_slip.detach()) > float(base.detach()), "slip term did not contribute"
    with_slip.backward()
    assert sp.grad is not None and float(sp.grad.abs().sum()) > 0, "no gradient into the slip head"


def test_slip_head_learns_injected_slip():
    """Overfit a small fixed set whose slip target is a learnable function of the
    input window; the head must recover it (loss down, correlation up)."""
    torch.manual_seed(0)
    net = SpeedNet()
    B = 64
    X = torch.randn(B, C, WIN)
    # a deterministic, learnable "slip residual": a linear readout of two channels
    slip_true = 2.0 * X[:, 1, :].mean(dim=1) + 0.5 * X[:, 4, :].mean(dim=1)
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    first = None
    for it in range(200):
        _, _, slip, _ = net(X)
        l = LOSS.slip_loss(slip, slip_true)
        opt.zero_grad(); l.backward(); opt.step()
        if it == 0:
            first = float(l.detach())
    last = float(l.detach())
    net.eval()
    with torch.no_grad():
        _, _, slip, _ = net(X)
    r = float(np.corrcoef(slip.numpy(), slip_true.numpy())[0, 1])
    assert last < 0.5 * first, f"slip loss did not drop: {first:.3f} -> {last:.3f}"
    assert r > 0.9, f"slip head did not learn the target (corr={r:.2f})"
