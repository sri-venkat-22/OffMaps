"""Dead-reckoning baselines. A model maps (drive, outage) -> predicted track.

Phase 0 only needs the do-nothing floor ('zero') and the honest physics floor
('cv' = freeze GNSS velocity at outage entry, the number Phase 1's baseline and
every later phase must beat). Register real models here later with the same
signature; the report iterates the registry.

Return: (pred_e, pred_n) over the window, plus optional (v_pred, sigma) per
sample for sigma-calibration scoring. Trajectory is anchored at true entry pose.
"""
from __future__ import annotations
import numpy as np

REGISTRY = {}


def model(name):
    def deco(fn): REGISTRY[name] = fn; return fn
    return deco


@model("zero")
def zero(drive, o):
    """Stay put. The floor: drift ~= 100% of distance travelled."""
    N = o.i1 - o.i0
    e = np.full(N, drive.e[o.i0]); n = np.full(N, drive.n[o.i0])
    return e, n, None, None


@model("cv")
def constant_velocity(drive, o):
    """Freeze velocity at entry, coast straight. Honest physics floor."""
    t = drive.t[o.i0:o.i1] - drive.t[o.i0]
    v = drive.speed[o.i0]; h = drive.heading[o.i0]
    e = drive.e[o.i0] + v * np.sin(h) * t
    n = drive.n[o.i0] + v * np.cos(h) * t
    vp = np.full(len(t), v)
    return e, n, vp, None


@model("truth")
def truth(drive, o):
    """Oracle (sanity: must score ~0 drift). Not a real model."""
    return drive.e[o.i0:o.i1].copy(), drive.n[o.i0:o.i1].copy(), \
           drive.speed[o.i0:o.i1].copy(), None
