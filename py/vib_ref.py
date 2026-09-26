"""Pure-Python twin of core/vib.cpp -- the ORACLE for the Phase-7c parity check.

Mirrors the streaming high-pass + adaptive shock detector + windowed clean/raw
RMS exactly (same recurrences, same integer refractory countdown, same EMA
freeze during a shock), so per-window features and event counts are identical to
the native front-end. Change one, change both, re-run parity.
"""
from __future__ import annotations
import numpy as np


class VibRef:
    def __init__(self, hz=250.0):
        self.hz = hz if hz > 0 else 250.0
        self.hp_fc = 3.0
        self.ema_tau = 0.5
        self.shock_k = 8.0
        self._recompute()
        self.px = [0.0, 0.0, 0.0]; self.py = [0.0, 0.0, 0.0]; self.have_prev = 0
        self.m = 0.0; self.s = 0.0; self.have_base = 0
        self.refractory = 0
        self.clean_ss = self.raw_ss = 0.0
        self.clean_cnt = self.cnt = self.shock_cnt = 0
        self.events = 0
        self.g = [0.0, 0.0, 0.0]; self.min_peak = 0.0; self.gap_set = self.gap = 0; self.armed = 0

    def _recompute(self):
        dt = 1.0 / self.hz
        tau_hp = 1.0 / (2 * np.pi * self.hp_fc)
        self.alpha = tau_hp / (tau_hp + dt)
        self.beta = 1.0 - np.exp(-dt / self.ema_tau)
        self.refractory_set = int(round(0.15 * self.hz))
        self.warm = int(round(0.4 * self.hz))
        self.beta_g = 1.0 - np.exp(-dt / 1.0)

    def set_params(self, shock_k=0.0, refractory_s=0.0, hp_fc=0.0, ema_tau=0.0):
        if shock_k > 0: self.shock_k = shock_k
        if hp_fc > 0: self.hp_fc = hp_fc
        if ema_tau > 0: self.ema_tau = ema_tau
        self._recompute()
        if refractory_s > 0: self.refractory_set = int(round(refractory_s * self.hz))

    def set_shock_filter(self, min_peak=0.0, gap_s=0.0):
        self.min_peak = min_peak if min_peak > 0 else 0.0
        self.gap_set = int(round(gap_s * self.hz)) if gap_s > 0 else 0

    def push(self, ax, ay, az, dt=0.0):
        x = [ax, ay, az]; y = [0.0, 0.0, 0.0]
        if not self.have_prev:
            for i in range(3): self.px[i] = x[i]; self.py[i] = 0.0; y[i] = 0.0; self.g[i] = x[i]
            self.have_prev = 1
        else:
            for i in range(3):
                y[i] = self.alpha * (self.py[i] + x[i] - self.px[i])
                self.px[i] = x[i]; self.py[i] = y[i]
                self.g[i] += self.beta_g * (x[i] - self.g[i])
        if self.gap > 0: self.gap -= 1
        mag = np.sqrt(y[0]*y[0] + y[1]*y[1] + y[2]*y[2])
        if not self.have_base:
            self.m = mag; self.s = 0.0; self.have_base = 1

        if self.warm > 0:                       # warmup: build the baseline, never fire
            self.warm -= 1
            dev = abs(mag - self.m)
            self.m += self.beta * (mag - self.m)
            self.s += self.beta * (dev - self.s)
            self.raw_ss += mag * mag; self.cnt += 1
            self.clean_ss += mag * mag; self.clean_cnt += 1
            return 0

        s_floor = self.s if self.s > 0.05 else 0.05
        thresh = self.m + self.shock_k * s_floor
        over = mag > thresh
        edge = 0
        if self.refractory > 0:
            shock_sample = 1; self.refractory -= 1
        elif over:
            shock_sample = 1; self.armed = 1; self.refractory = self.refractory_set
        else:
            shock_sample = 0
            dev = abs(mag - self.m)
            self.m += self.beta * (mag - self.m)
            self.s += self.beta * (dev - self.s)
        if not shock_sample:
            self.armed = 0
        elif self.armed and self.gap == 0:
            g = self.g
            gn = np.sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2])
            vert = abs(y[0]*g[0] + y[1]*g[1] + y[2]*g[2]) / gn if gn > 0 else mag
            if vert >= self.min_peak:
                edge = 1; self.armed = 0; self.events += 1; self.gap = self.gap_set
        self.raw_ss += mag * mag; self.cnt += 1
        if shock_sample: self.shock_cnt += 1
        else: self.clean_ss += mag * mag; self.clean_cnt += 1
        return edge

    def window(self):
        out = [
            np.sqrt(self.clean_ss / self.clean_cnt) if self.clean_cnt > 0 else 0.0,
            np.sqrt(self.raw_ss / self.cnt) if self.cnt > 0 else 0.0,
            self.shock_cnt / self.cnt if self.cnt > 0 else 0.0,
            float(self.events),
        ]
        self.clean_ss = self.raw_ss = 0.0
        self.clean_cnt = self.cnt = self.shock_cnt = 0
        self.events = 0
        return np.array(out)
