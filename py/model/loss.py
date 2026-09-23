"""The paper-worthy bit: optimise the metric ISRO scores, not instantaneous MAE.

- NLL on 1 s displacement: teaches an HONEST sigma (this is what makes the
  output fusable; a point estimate a Kalman filter can't weight).
- Displacement-consistency over 1/5/10/30 s horizons: penalise the *integral*
  error, which is the drift the benchmark measures. Suppresses biased error,
  tolerates zero-mean noise.
- Motion class CE.

Batches are contiguous L-second sequences: mu/dtrue are (B, L).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

HORIZONS = (1, 5, 10, 30)   # seconds
BETA = 0.5                  # beta-NLL mean-fitting weight (Seitzer et al. 2022)


def nll(mu, logvar, dtrue, beta=BETA):
    """beta-NLL: Gaussian NLL with each sample re-weighted by detach(sigma^(2*beta)).

    Plain Gaussian NLL (beta=0) divides the squared error by sigma^2, so wherever
    the model can inflate sigma -- exactly the high-speed windows whose vibration
    cue is noisiest -- the mean term is down-weighted and mu reverts toward the
    dataset mean. That shows up in sigma-calibration as a speed-proportional
    under-prediction (z_mean < 0): the model explains error with variance instead
    of fixing the mean. Multiplying the per-sample loss by stop_grad(sigma^(2*beta))
    cancels that 1/sigma^2 down-weighting in the mu gradient (beta=1 fully, so mu
    trains like MSE) while leaving the logvar gradient free to stay calibrated.
    beta=0.5 is the paper's accuracy/calibration compromise.
    """
    var = torch.exp(logvar)
    per = 0.5 * (logvar + (mu - dtrue) ** 2 / var)
    w = var.detach() ** beta
    return (w * per).mean()


def consistency(mu, dtrue, horizons=HORIZONS):
    """Windowed h-second displacement error over the whole sequence."""
    cp = torch.cumsum(mu, dim=1); ct = torch.cumsum(dtrue, dim=1)
    loss = mu.new_zeros(())
    L = mu.shape[1]
    for h in horizons:
        if h >= L:
            continue
        wp = cp[:, h:] - cp[:, :-h]      # predicted displacement over each h-window
        wt = ct[:, h:] - ct[:, :-h]
        loss = loss + (wp - wt).abs().mean()
    return loss


def slip_loss(slip_pred, slip_true):
    """Lateral-slip residual head: robust (Huber) regression to the measured
    lateral-slip residual = a_lat - v*yawrate.

    On synthetic data a_lat IS v*yawrate by construction, so the residual is zero
    and this term teaches nothing -- the head only becomes useful on real phone
    logs, where tyre slip makes the residual non-zero. It is wired here (and its
    learnability is proved in tests/test_slip_head.py) so the path is live and
    ready for real data, not dead weight in the checkpoint.
    """
    return F.smooth_l1_loss(slip_pred, slip_true)


def total(mu, logvar, dtrue, cls_logits, cls_true, w_cons=1.0, w_cls=0.1,
          slip_pred=None, slip_true=None, w_slip=0.2):
    loss = (nll(mu, logvar, dtrue)
            + w_cons * consistency(mu, dtrue)
            + w_cls * F.cross_entropy(cls_logits.reshape(-1, 4), cls_true.reshape(-1)))
    if slip_pred is not None and slip_true is not None:   # supervised only when a target exists
        loss = loss + w_slip * slip_loss(slip_pred, slip_true)
    return loss
