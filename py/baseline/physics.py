"""Phase-1 physics baseline. THE number every later phase must beat.

The one competition-deciding rule, made concrete: do NOT double-integrate
accelerometer. Freeze forward speed at the last GNSS Doppler value on outage
entry, and integrate that speed along a heading dead-reckoned from the gyro.
Drift is then linear in distance (speed-scale error) + a gyro-bias term in
heading -- exactly the error budget, and orders of magnitude below accel
double-integration.

No NN, no filter. Importing this module registers the `physics` model.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from eval.models import model


@model("physics")
def physics(drive, o):
    t = drive.t[o.i0:o.i1]
    dt = np.diff(t, prepend=t[0])
    v = drive.speed[o.i0]                      # frozen Doppler speed at entry
    h0 = drive.heading[o.i0]                   # known heading at entry
    if drive.gyro_z is not None:
        h = h0 + np.cumsum(drive.gyro_z[o.i0:o.i1] * dt)   # dead-reckon heading
    else:
        h = np.full(len(t), h0)                # no gyro -> straight line (== cv)
    e = drive.e[o.i0] + np.cumsum(v * np.sin(h) * dt)
    n = drive.n[o.i0] + np.cumsum(v * np.cos(h) * dt)
    vp = np.full(len(t), v)
    return e, n, vp, None


def _selfcheck():
    from data.io_vnbd import synth_drive
    from data.outage import make_outages
    from eval.metrics import score_outage
    from eval.models import REGISTRY
    d = synth_drive(duration=600, seed=3)
    outs = make_outages(d, seed=0)
    rows = [score_outage(d, o, physics(d, o)) for o in outs]
    med = np.nanmedian([r["drift_pct"] for r in rows])
    cv = np.nanmedian([score_outage(d, o, REGISTRY["cv"](d, o))["drift_pct"] for o in outs])
    assert med < cv, (med, cv)
    print(f"physics ok: drift_med={med:.1f}% (cv straight-line floor={cv:.1f}%)")


if __name__ == "__main__":
    _selfcheck()
