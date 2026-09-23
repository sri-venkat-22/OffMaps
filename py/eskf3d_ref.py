"""Pure-Python twin of core/eskf3d.cpp -- the ORACLE for the 3D ESKF parity check.

Same contract as eskf_ref.py: this duplicates the C++ math on purpose. It is the
CI proof that the 16-state error-state filter shipped to edge/Android matches the
researched filter. Change one, change both, re-run parity.

Nominal x = [p(3), v(3), q(4 w,x,y,z), b_a(3), b_g(3)].
Error dx  = [dp(3), dv(3), dtheta(3), db_a(3), db_g(3)]; P is 15x15.
q = Hamilton, body->world; local (right) attitude error q_true = q (x) Exp(dtheta).
"""
from __future__ import annotations
import numpy as np

N = 15
G0 = 9.81
DEG = np.pi / 180


def rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def quatmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def normq(q):
    n = np.sqrt(np.dot(q, q))
    return q / n if n > 0 else q


def expq(phi):
    th = np.sqrt(np.dot(phi, phi))
    if th < 1e-12:
        return normq(np.array([1.0, 0.5*phi[0], 0.5*phi[1], 0.5*phi[2]]))
    s = np.sin(0.5*th) / th
    return np.array([np.cos(0.5*th), s*phi[0], s*phi[1], s*phi[2]])


def _skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


class Idr3dRef:
    def __init__(self):
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0, 0, 0])
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)
        self.P = np.zeros((N, N))
        self.P[0:3, 0:3] = np.eye(3) * 4.0
        self.P[3:6, 3:6] = np.eye(3) * 0.25
        self.P[6:9, 6:9] = np.eye(3) * (2*DEG)**2
        self.P[9:12, 9:12] = np.eye(3) * 0.1**2
        self.P[12:15, 12:15] = np.eye(3) * (0.2*DEG)**2
        self.set_noise(0.05, 0.3*DEG, 0.001, 0.01*DEG)

    def set_noise(self, vrw, arw, barw, bgrw):
        self.q_a, self.q_g, self.q_ba, self.q_bg = vrw*vrw, arw*arw, barw*barw, bgrw*bgrw

    def init(self, p, v, q):
        self.p = np.asarray(p, float).copy()
        self.v = np.asarray(v, float).copy()
        self.q = normq(np.asarray(q, float).copy())
        self.ba = np.zeros(3); self.bg = np.zeros(3)

    def _inject(self, dx):
        self.p += dx[0:3]; self.v += dx[3:6]
        self.q = normq(quatmul(self.q, expq(dx[6:9])))
        self.ba += dx[9:12]; self.bg += dx[12:15]

    def _update(self, H, innov, R):
        H = np.asarray(H, float)
        PHt = self.P @ H
        S = R + H @ PHt
        K = PHt / S
        self._inject(K * innov)
        self.P -= np.outer(K, H @ self.P)

    def predict(self, dt, a_m, w_m):
        a_m = np.asarray(a_m, float); w_m = np.asarray(w_m, float)
        R = rotmat(self.q)
        ab = a_m - self.ba
        wb = w_m - self.bg
        aw = R @ ab + np.array([0, 0, -G0])
        self.p = self.p + self.v*dt + 0.5*aw*dt*dt
        self.v = self.v + aw*dt
        phi = wb*dt
        self.q = normq(quatmul(self.q, expq(phi)))
        # error-state transition
        F = np.eye(N)
        F[0:3, 3:6] = np.eye(3)*dt
        F[3:6, 6:9] = -(R @ _skew(ab))*dt
        F[3:6, 9:12] = -R*dt
        F[6:9, 6:9] = np.eye(3) - _skew(phi)
        F[6:9, 12:15] = -np.eye(3)*dt
        self.P = F @ self.P @ F.T
        for i in range(3):
            self.P[3+i, 3+i] += self.q_a*dt
            self.P[6+i, 6+i] += self.q_g*dt
            self.P[9+i, 9+i] += self.q_ba*dt
            self.P[12+i, 12+i] += self.q_bg*dt

    def update_gnss_pos(self, p, sigma):
        for k in range(3):
            H = np.zeros(N); H[k] = 1.0
            self._update(H, p[k] - self.p[k], sigma*sigma)

    def update_gnss_vel(self, v, sigma):
        for k in range(3):
            H = np.zeros(N); H[3+k] = 1.0
            self._update(H, v[k] - self.v[k], sigma*sigma)

    def update_zupt(self, sigma):
        for k in range(3):
            H = np.zeros(N); H[3+k] = 1.0
            self._update(H, 0.0 - self.v[k], sigma*sigma)

    def update_zaru(self, w_m, sigma):
        for k in range(3):
            H = np.zeros(N); H[12+k] = 1.0
            self._update(H, w_m[k] - self.bg[k], sigma*sigma)

    def update_nhc(self, sigma):
        # body velocity u = R^T v; constrain lateral (y) and vertical (z).
        R = rotmat(self.q); u = R.T @ self.v
        H = np.zeros(N); H[3:6] = R[:, 1]; H[6+0] = u[2]; H[6+2] = -u[0]
        self._update(H, 0.0 - u[1], sigma*sigma)
        R = rotmat(self.q); u = R.T @ self.v
        H = np.zeros(N); H[3:6] = R[:, 2]; H[6+0] = -u[1]; H[6+1] = u[0]
        self._update(H, 0.0 - u[2], sigma*sigma)

    def update_odo(self, v_fwd, sigma):
        # body forward velocity u_x = (R^T v)_x = wheel speed.
        R = rotmat(self.q); u = R.T @ self.v
        H = np.zeros(N); H[3:6] = R[:, 0]; H[6+1] = -u[2]; H[6+2] = u[1]
        self._update(H, v_fwd - u[0], sigma*sigma)

    def update_baro(self, up, sigma):
        H = np.zeros(N); H[2] = 1.0
        self._update(H, up - self.p[2], sigma*sigma)

    def state(self):
        return np.concatenate([self.p, self.v, self.q, self.ba, self.bg])

    def cov_diag(self):
        return np.diag(self.P).copy()
