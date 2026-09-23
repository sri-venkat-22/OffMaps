"""Pure-Python twin of core/speed_cal.cpp LS/Deming -- parity oracle."""
from __future__ import annotations
import numpy as np


def _deming_k(Sxx, Syy, Sxy, lam):
    """Deming slope for y=k*x; lam = var(err_y)/var(err_x). lam=inf -> OLS."""
    if not np.isfinite(lam):
        return Sxy / Sxx
    if abs(Sxy) < 1e-12:
        return Sxy / Sxx
    t = Syy - lam * Sxx
    return (t + np.sqrt(t * t + 4.0 * lam * Sxy * Sxy)) / (2.0 * Sxy)


class SpeedCalRef:
    CAP = 600
    def __init__(self):
        self.vn, self.vd, self.w = [], [], []
        self.k, self.c, self.excited = 1.0, 0.0, 0
        self.lam = np.inf                       # +inf = OLS, finite = Deming
    def push(self, v_nn, v_dop, w=1.0):
        for buf, val in ((self.vn, v_nn), (self.vd, v_dop), (self.w, w)):
            buf.append(val)
            if len(buf) > self.CAP: buf.pop(0)
    def set_lambda(self, lam): self.lam = lam
    def fit(self):
        n = len(self.vn)
        if n < 30: return self.k, self.c, 0
        x, y, w = np.array(self.vn), np.array(self.vd), np.array(self.w)
        sw = w.sum(); swx = (w*x).sum(); swy = (w*y).sum()
        swxx = (w*x*x).sum(); swyy = (w*y*y).sum(); swxy = (w*x*y).sum()
        if not sw > 1e-12: return self.k, self.c, 0      # all weights 0: keep the fit (mirrors C++)
        mean = swx/sw; var = swxx/sw - mean*mean
        self.excited = int(var > 4.0)
        if not np.isfinite(self.lam):            # OLS (default)
            if self.excited:
                det = sw*swxx - swx*swx
                self.k = (sw*swxy - swx*swy)/det
                self.c = (swxx*swy - swx*swxy)/det
            else:
                self.k = swxy/swxx; self.c = 0.0
        else:                                    # Deming (errors-in-variables)
            if self.excited:
                xb = swx/sw; yb = swy/sw
                sxx = swxx/sw - xb*xb; syy = swyy/sw - yb*yb; sxy = swxy/sw - xb*yb
                self.k = _deming_k(sxx, syy, sxy, self.lam)
                self.c = yb - self.k*xb
            else:
                self.k = _deming_k(swxx, swyy, swxy, self.lam); self.c = 0.0
        return self.k, self.c, self.excited
