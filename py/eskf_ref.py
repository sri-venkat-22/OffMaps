"""Pure-Python twin of core/eskf.cpp -- the ORACLE for the C++ parity check.

Yes this duplicates the C++ math. That duplication IS the deliverable: it's the
CI check that the code shipped to Android/edge matches the researched filter.
Keep the two in lockstep; if you change one, change both and re-run parity.
"""
from __future__ import annotations
import numpy as np

E, PN, PSI, V, BG = range(5)
DEG = np.pi / 180
CURV_MIN_RATE = 10 * DEG


def _wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


class EskfRef:
    def __init__(self):
        self.x = np.zeros(5)
        self.P = np.diag([1.0, 1.0, (1 * DEG) ** 2, 0.5 ** 2, (0.05 * DEG) ** 2])
        self.set_noise(0.3 * DEG, 0.01 * DEG, 0.7)
        self.map_keep_v = False

    def set_noise(self, arw, brw, srw):
        self.q_psi, self.q_bg, self.q_v = arw * arw, brw * brw, srw * srw

    def init(self, e, n, psi, v): self.x[:] = [e, n, psi, v, 0.0]

    def _update(self, H, innov, R):
        H = np.asarray(H, float)
        PHt = self.P @ H
        S = R + H @ PHt
        K = PHt / S
        self.x += K * innov
        self.P -= np.outer(K, H @ self.P)

    def _update_skip(self, H, innov, R, skip):
        """Gain row `skip` forced to 0; Joseph-form P (valid for a sub-optimal gain)."""
        H = np.asarray(H, float)
        PHt = self.P @ H
        K = PHt / (R + H @ PHt)
        K[skip] = 0.0
        self.x += K * innov
        A = np.eye(5) - np.outer(K, H)
        self.P = A @ self.P @ A.T + np.outer(K, K) * R

    def _update_map(self, H, innov, R):
        if self.map_keep_v: self._update_skip(H, innov, R, V)
        else: self._update(H, innov, R)

    def set_map_keep_speed(self, keep): self.map_keep_v = bool(keep)

    def set_heading_sigma(self, sigma):
        self.P[PSI, :] = 0.0; self.P[:, PSI] = 0.0; self.P[PSI, PSI] = sigma * sigma

    def predict(self, dt, gyro_z):
        x = self.x; psi, v = x[PSI], x[V]
        s, c = np.sin(psi), np.cos(psi)
        x[E] += v * s * dt; x[PN] += v * c * dt
        x[PSI] = _wrap(psi + (gyro_z - x[BG]) * dt)
        F = np.eye(5)
        F[E, PSI] = v * c * dt; F[E, V] = s * dt
        F[PN, PSI] = -v * s * dt; F[PN, V] = c * dt
        F[PSI, BG] = -dt
        self.P = F @ self.P @ F.T
        self.P[PSI, PSI] += self.q_psi * dt
        self.P[V, V] += self.q_v * dt
        self.P[BG, BG] += self.q_bg * dt

    def update_speed(self, v_meas, sigma):
        H = np.zeros(5); H[V] = 1; self._update(H, v_meas - self.x[V], sigma ** 2)

    def update_zupt(self, gyro_z):
        H = np.zeros(5); H[V] = 1; self._update(H, -self.x[V], 0.02 ** 2)
        H = np.zeros(5); H[BG] = 1; self._update(H, gyro_z - self.x[BG], (0.02 * DEG) ** 2)

    def update_curvature(self, a_lat, psidot, base_sigma):
        if abs(psidot) < CURV_MIN_RATE: return
        v_curv = a_lat / psidot
        sigma = base_sigma * CURV_MIN_RATE / abs(psidot)
        H = np.zeros(5); H[V] = 1; self._update(H, v_curv - self.x[V], sigma ** 2)

    def update_gnss_pos(self, e, n, sigma):
        H = np.zeros(5); H[E] = 1; self._update(H, e - self.x[E], sigma ** 2)
        H = np.zeros(5); H[PN] = 1; self._update(H, n - self.x[PN], sigma ** 2)

    def update_gnss_vel(self, v_meas, bearing, sigma_v, sigma_psi):
        # mirrors idr_update_gnss_vel: sequential scalar v then psi (same order)
        H = np.zeros(5); H[V] = 1; self._update(H, v_meas - self.x[V], sigma_v ** 2)
        H = np.zeros(5); H[PSI] = 1; self._update(H, _wrap(bearing - self.x[PSI]), sigma_psi ** 2)

    def update_crosstrack(self, ne, nn, cross_innov, sigma):
        H = np.zeros(5); H[E] = ne; H[PN] = nn; self._update_map(H, cross_innov, sigma ** 2)

    def update_heading(self, bearing, sigma):
        H = np.zeros(5); H[PSI] = 1; self._update_map(H, _wrap(bearing - self.x[PSI]), sigma ** 2)

    def state(self): return self.x.copy()
