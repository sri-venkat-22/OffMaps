"""Heading aids for the live loop: yaw alignment, magnetometer seed, turn-tilt yaw rate.

Reference implementation for the edge engine (edge_engine.py) and the phone
(HeadingAids.kt, checked against this file by tests/test_kotlin_ports.py).

1. YawAlign -- the vehicle's forward axis in the LEVELED frame, from GNSS Doppler.
   Between two trusted fixes the along-track acceleration is known (dv/dt of the
   Doppler speed); a forgetting least-squares fit dv/dt ~ c1*a_h1 + c2*a_h2 of the
   leveled horizontal accel gives the forward direction theta = atan2(c2, c1), sign
   included (PCA on accel, core/align.cpp, leaves a 180 deg ambiguity). It is the
   causal version of what the real-data loader does per segment
   (data/iovnbd_sync._level_and_heading). Frozen while GNSS is out; reset on a re-mount.

2. mag_heading -- tilt-compensated magnetometer heading of the vehicle's forward
   axis. Used ONCE: to seed the filter's heading at the first GNSS fix when that fix
   has no usable course (parked: GNSS bearing is noise below ~1 m/s), with a
   sigma of MAG_SIGMA instead of the filter's default 1 deg. A car's steel body
   and electronics distort the field, so it is never used as a running update.
   The forward axis comes from YawAlign when it has converged (a previous drive in
   the same mount), else from default_forward's mount guess.

3. yaw_rate -- the yaw rate about TRUE vertical, three ways:
     "fast"  gyro . (0.5 s gravity EMA)      -- what the app did before
     "slow"  gyro . (30 s gravity EMA)       -- the leveling frame's up
     "coord" gyro . (fast up tilted back by the coordinated-turn angle)
   In a sustained turn the specific force leans toward the turn centre by
   phi = atan(v*psidot/g). A car's body barely rolls, so the fast EMA (which follows
   the specific force) tilts by phi while the gyro still turns about true vertical:
   gyro . u_fast = psidot*cos(phi), a 4 % under-read at 0.3 g. A two-wheeler LEANS by
   phi, so the device itself tilts and even the slow EMA (the bike's upright axis) is
   off by cos(phi) -- 13 % at a 30 deg lean. "coord" undoes it: true up =
   cos(phi)*u_fast + sin(phi)*left (left = vehicle y), with phi from the filter's
   speed and the yaw rate itself (3 fixed-point passes). It needs the
   forward axis (YawAlign / mount guess) and is the two-wheeler default; "slow" is
   exact for a car and needs nothing, so it is the car default.
"""
from __future__ import annotations
import math

YAW_SIGN = -1.0          # compass yaw = YAW_SIGN * (gyro . up)  (phone_log / Level.kt)
G = 9.81
MAG_SIGMA = math.radians(20.0)   # heading sigma of a magnetometer seed in a car
UNKNOWN_SIGMA = math.pi          # no course and no magnetometer: heading unknown


def _dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _unit(a):
    n = math.sqrt(_dot(a, a))
    return (a[0] / n, a[1] / n, a[2] / n) if n > 1e-9 else (0.0, 0.0, 0.0)


def _horiz(a, u):
    d = _dot(a, u)
    return _unit((a[0] - d * u[0], a[1] - d * u[1], a[2] - d * u[2]))


def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi


def default_forward(up):
    """Mount guess for the vehicle's forward axis (device coords), used until YawAlign
    converges. A phone lying flat (screen up) has its top (+y) forward; a phone
    standing in a dash/handlebar mount faces the driver, so its back (-z) is forward."""
    u = _unit(up)
    return _horiz((0.0, 1.0, 0.0), u) if abs(u[2]) >= 0.7 else _horiz((0.0, 0.0, -1.0), u)


def mag_heading(mag, up, fwd, declination=0.0):
    """Compass heading (0 = north, clockwise) of `fwd` from a device-frame magnetometer
    reading. Tilt-compensated with `up`; `declination` (rad, east +) turns magnetic
    north into true north. Returns None if the field has no horizontal component."""
    u = _unit(up)
    north = _horiz(mag, u)
    if north == (0.0, 0.0, 0.0):
        return None
    east = _cross(north, u)                       # ENU: E = N x U
    f = _horiz(fwd, u)
    return wrap(math.atan2(_dot(f, east), _dot(f, north)) + declination)


def yaw_rate(mode, gyro, up_fast, up_slow, fwd=None, v=0.0):
    """Compass yaw rate (rad/s) about true vertical; see the module doc for the modes."""
    if mode == "slow":
        return YAW_SIGN * _dot(gyro, _unit(up_slow))
    z = _unit(up_fast)
    r = YAW_SIGN * _dot(gyro, z)
    if mode == "fast" or fwd is None:
        return r
    x = _horiz(fwd, z)
    y = _cross(z, x)                              # vehicle left
    for _ in range(3):                            # fixed point: cos(phi) ~ 1, converges fast
        phi = math.atan(v * r / G)                # right turn (r > 0) tilts toward the right
        c, s = math.cos(phi), math.sin(phi)
        r = YAW_SIGN * _dot(gyro, (c * z[0] + s * y[0], c * z[1] + s * y[1], c * z[2] + s * y[2]))
    return r


class YawAlign:
    """GNSS-aided forward axis in the leveled frame (see module doc, 1)."""
    FORGET = 0.995           # per trusted fix: ~200 s memory
    MIN_EXC = 4.0            # sum of (dv/dt)^2 (m/s^2)^2 before the fit is used
    GAIN_OK = (0.2, 5.0)     # |c|: accel-to-dv/dt gain of a real fit (1 = ideal)

    def __init__(self):
        self.reset()

    def reset(self):
        self.s11 = self.s12 = self.s22 = self.b1 = self.b2 = self.exc = 0.0
        self.sum1 = self.sum2 = 0.0; self.cnt = 0
        self.t_prev = None; self.v_prev = 0.0
        self.theta = 0.0; self.valid = False

    def add_acc(self, a1, a2):
        """One leveled horizontal accel sample (h1, h2 components), any rate."""
        self.sum1 += a1; self.sum2 += a2; self.cnt += 1

    def on_fix(self, t, v):
        """A trusted Doppler speed at time t (s). Returns True if the fit was updated."""
        upd = False
        if self.t_prev is not None and self.cnt > 0 and 0.5 <= t - self.t_prev <= 2.5:
            dv = (v - self.v_prev) / (t - self.t_prev)
            a1, a2 = self.sum1 / self.cnt, self.sum2 / self.cnt
            f = self.FORGET
            self.s11 = f * self.s11 + a1 * a1; self.s12 = f * self.s12 + a1 * a2
            self.s22 = f * self.s22 + a2 * a2
            self.b1 = f * self.b1 + a1 * dv; self.b2 = f * self.b2 + a2 * dv
            self.exc = f * self.exc + dv * dv
            det = self.s11 * self.s22 - self.s12 * self.s12
            if det > 1e-9:
                c1 = (self.s22 * self.b1 - self.s12 * self.b2) / det
                c2 = (self.s11 * self.b2 - self.s12 * self.b1) / det
                gain = math.hypot(c1, c2)
                self.theta = math.atan2(c2, c1)
                self.valid = self.exc >= self.MIN_EXC and self.GAIN_OK[0] <= gain <= self.GAIN_OK[1]
                upd = True
        self.t_prev, self.v_prev = t, v
        self.sum1 = self.sum2 = 0.0; self.cnt = 0
        return upd

    def forward(self, basis):
        """Forward axis in device coords, given the leveled basis rows (h1, h2, up)."""
        h1, h2 = basis[0], basis[1]
        c, s = math.cos(self.theta), math.sin(self.theta)
        return (c * h1[0] + s * h2[0], c * h1[1] + s * h2[1], c * h1[2] + s * h2[2])
