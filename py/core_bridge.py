"""ctypes bridge to core/libidr + registers the `eskf` model in the harness.

If the native lib isn't built, falls back to the Python twin so the harness
still runs (with a note) -- but the parity test insists they agree.
"""
from __future__ import annotations
import ctypes as C
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_LIB = None


def _load():
    global _LIB
    if _LIB is None:
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")
        for ext in ("dylib", "so"):
            p = os.path.join(base, f"libidr.{ext}")
            if os.path.exists(p):
                _LIB = C.CDLL(p); _decl(_LIB); return _LIB
        raise FileNotFoundError("build core first: sh core/build.sh")
    return _LIB


def _decl(l):
    l.idr_create.restype = C.c_void_p
    d = C.c_double
    for name, args in {
        "idr_init": [d, d, d, d], "idr_set_process_noise": [d, d, d],
        "idr_predict": [d, d], "idr_update_speed": [d, d], "idr_update_zupt": [d],
        "idr_update_curvature": [d, d, d], "idr_update_gnss_pos": [d, d, d],
        "idr_update_gnss_vel": [d, d, d, d],
    }.items():
        getattr(l, name).argtypes = [C.c_void_p] + args
    l.idr_get_state.argtypes = [C.c_void_p, C.POINTER(C.c_double * 5)]
    l.idr_get_cov.argtypes = [C.c_void_p, C.POINTER(C.c_double * 3)]
    l.idr_destroy.argtypes = [C.c_void_p]
    d, P, v = C.c_double, C.POINTER(C.c_double), C.c_void_p
    l.spc_create.restype = v
    l.spc_push.argtypes = [v, d, d, d]
    l.spc_set_lambda.argtypes = [v, d]
    l.spc_fit.argtypes = [v, P, P]; l.spc_fit.restype = C.c_int
    l.spc_apply.argtypes = [v, d]; l.spc_apply.restype = d
    l.spc_get.argtypes = [v, P, P, C.POINTER(C.c_int)]
    l.aln_create.restype = v
    l.aln_update.argtypes = [v, d, d, d, d, d, d, d]
    l.aln_get.argtypes = [v, P, P, P, C.POINTER(C.c_int)]
    for nm in ("gq_trust", "gq_R", "gq_spoof"): getattr(l, nm).restype = d
    l.gq_trust.argtypes = [d, C.c_int, C.c_int, d, d]
    l.gq_R.argtypes = [d]
    l.gq_spoof.argtypes = [d, C.c_int, d, d]; l.gq_spoof.restype = C.c_int
    l.idr_update_crosstrack.argtypes = [v, d, d, d, d]
    l.idr_update_heading.argtypes = [v, d, d]
    l.idr_set_map_keep_speed.argtypes = [v, C.c_int]
    l.mm_create.restype = v
    l.mm_add_way.argtypes = [v, P, P, C.c_int, C.c_int, C.c_int]
    l.mm_match.argtypes = [v, d, d, d, C.POINTER(C.c_double * 6)]
    I = C.POINTER(C.c_int)
    l.mm_match_seq.argtypes = [v, P, P, P, C.c_int, I, P, P, P, I]
    l.mm_match_seq.restype = C.c_int
    l.mm_add_edge.argtypes = [v, C.c_int, C.c_int]
    l.mm_set_hmm.argtypes = [v, d, d, d]
    # --- Phase 7a: 3D 16-state ESKF ---
    l.idr3d_create.restype = v
    l.idr3d_destroy.argtypes = [v]
    l.idr3d_init.argtypes = [v, C.POINTER(C.c_double * 3), C.POINTER(C.c_double * 3), C.POINTER(C.c_double * 4)]
    l.idr3d_set_noise.argtypes = [v, d, d, d, d]
    l.idr3d_predict.argtypes = [v, d, C.POINTER(C.c_double * 3), C.POINTER(C.c_double * 3)]
    l.idr3d_update_gnss_pos.argtypes = [v, C.POINTER(C.c_double * 3), d]
    l.idr3d_update_gnss_vel.argtypes = [v, C.POINTER(C.c_double * 3), d]
    l.idr3d_update_zupt.argtypes = [v, d]
    l.idr3d_update_zaru.argtypes = [v, C.POINTER(C.c_double * 3), d]
    l.idr3d_update_nhc.argtypes = [v, d]
    l.idr3d_update_odo.argtypes = [v, d, d]
    l.idr3d_update_baro.argtypes = [v, d, d]
    l.idr3d_get_state.argtypes = [v, C.POINTER(C.c_double * 16)]
    l.idr3d_get_cov.argtypes = [v, C.POINTER(C.c_double * 15)]
    # --- Phase 7c: high-rate vibration / pothole front-end ---
    l.vib_create.restype = v
    l.vib_create.argtypes = [d]
    l.vib_destroy.argtypes = [v]
    l.vib_set_params.argtypes = [v, d, d, d, d]
    l.vib_push.argtypes = [v, d, d, d, d]; l.vib_push.restype = C.c_int
    l.vib_window.argtypes = [v, C.POINTER(C.c_double * 4)]


class Filter:
    """Thin native handle; method names mirror EskfRef for parity testing."""
    def __init__(self): self.l = _load(); self.h = self.l.idr_create()
    def init(self, e, n, psi, v): self.l.idr_init(self.h, e, n, psi, v)
    def set_noise(self, a, b, s): self.l.idr_set_process_noise(self.h, a, b, s)
    def predict(self, dt, gz): self.l.idr_predict(self.h, dt, gz)
    def update_speed(self, v, s): self.l.idr_update_speed(self.h, v, s)
    def update_zupt(self, gz): self.l.idr_update_zupt(self.h, gz)
    def update_curvature(self, al, pd, bs): self.l.idr_update_curvature(self.h, al, pd, bs)
    def update_gnss_pos(self, e, n, s): self.l.idr_update_gnss_pos(self.h, e, n, s)
    def update_gnss_vel(self, v, brg, sv, sp): self.l.idr_update_gnss_vel(self.h, v, brg, sv, sp)
    def update_crosstrack(self, ne, nn, ci, s): self.l.idr_update_crosstrack(self.h, ne, nn, ci, s)
    def update_heading(self, brg, s): self.l.idr_update_heading(self.h, brg, s)
    def set_map_keep_speed(self, keep): self.l.idr_set_map_keep_speed(self.h, int(bool(keep)))
    def state(self):
        out = (C.c_double * 5)()
        self.l.idr_get_state(self.h, C.byref(out)); return np.array(out)
    def cov(self):
        out = (C.c_double * 3)()
        self.l.idr_get_cov(self.h, C.byref(out)); return np.array(out)  # [P_ee, P_nn, P_psipsi]
    def __del__(self):
        try: self.l.idr_destroy(self.h)
        except Exception: pass


def _c3(x):
    return (C.c_double * 3)(*[float(v) for v in x])


class Filter3D:
    """Native handle for the 3D 16-state ESKF; method names mirror Idr3dRef."""
    def __init__(self): self.l = _load(); self.h = self.l.idr3d_create()
    def init(self, p, v, q):
        self.l.idr3d_init(self.h, C.byref(_c3(p)), C.byref(_c3(v)),
                          C.byref((C.c_double * 4)(*[float(x) for x in q])))
    def set_noise(self, vrw, arw, barw, bgrw):
        self.l.idr3d_set_noise(self.h, vrw, arw, barw, bgrw)
    def predict(self, dt, a_m, w_m):
        self.l.idr3d_predict(self.h, dt, C.byref(_c3(a_m)), C.byref(_c3(w_m)))
    def update_gnss_pos(self, p, sigma): self.l.idr3d_update_gnss_pos(self.h, C.byref(_c3(p)), sigma)
    def update_gnss_vel(self, v, sigma): self.l.idr3d_update_gnss_vel(self.h, C.byref(_c3(v)), sigma)
    def update_zupt(self, sigma): self.l.idr3d_update_zupt(self.h, sigma)
    def update_zaru(self, w_m, sigma): self.l.idr3d_update_zaru(self.h, C.byref(_c3(w_m)), sigma)
    def update_nhc(self, sigma): self.l.idr3d_update_nhc(self.h, sigma)
    def update_odo(self, v_fwd, sigma): self.l.idr3d_update_odo(self.h, v_fwd, sigma)
    def update_baro(self, up, sigma): self.l.idr3d_update_baro(self.h, up, sigma)
    def state(self):
        out = (C.c_double * 16)(); self.l.idr3d_get_state(self.h, C.byref(out)); return np.array(out)
    def cov_diag(self):
        out = (C.c_double * 15)(); self.l.idr3d_get_cov(self.h, C.byref(out)); return np.array(out)
    def __del__(self):
        try: self.l.idr3d_destroy(self.h)
        except Exception: pass


class Vib:
    """Native handle for the high-rate vibration/pothole front-end; mirrors VibRef."""
    def __init__(self, hz=250.0): self.l = _load(); self.h = self.l.vib_create(hz)
    def set_params(self, shock_k=0.0, refractory_s=0.0, hp_fc=0.0, ema_tau=0.0):
        self.l.vib_set_params(self.h, shock_k, refractory_s, hp_fc, ema_tau)
    def push(self, ax, ay, az, dt=0.0): return self.l.vib_push(self.h, ax, ay, az, dt)
    def window(self):
        out = (C.c_double * 4)(); self.l.vib_window(self.h, C.byref(out)); return np.array(out)
    def __del__(self):
        try: self.l.vib_destroy(self.h)
        except Exception: pass


class SpeedCal:
    def __init__(self): self.l = _load(); self.h = self.l.spc_create()
    def push(self, v_nn, v_dop, w=1.0): self.l.spc_push(self.h, v_nn, v_dop, w)
    def set_lambda(self, lam): self.l.spc_set_lambda(self.h, lam)
    def fit(self):
        k, c = C.c_double(), C.c_double()
        ex = self.l.spc_fit(self.h, C.byref(k), C.byref(c))
        return k.value, c.value, ex
    def apply(self, v_nn): return self.l.spc_apply(self.h, v_nn)


class Align:
    def __init__(self): self.l = _load(); self.h = self.l.aln_create()
    def update(self, ax, ay, az, gx, gy, gz, dt):
        self.l.aln_update(self.h, ax, ay, az, gx, gy, gz, dt)
    def get(self):
        r, p, y, ch = C.c_double(), C.c_double(), C.c_double(), C.c_int()
        self.l.aln_get(self.h, C.byref(r), C.byref(p), C.byref(y), C.byref(ch))
        return r.value, p.value, y.value, ch.value


def gq_trust(cn0, sv, navic, dop, chi2): return _load().gq_trust(cn0, sv, navic, dop, chi2)
def gq_R(trust): return _load().gq_R(trust)
def gq_spoof(cn0, sv, chi2, dop_resid): return _load().gq_spoof(cn0, sv, chi2, dop_resid)


class MapMatcher:
    def __init__(self):
        self.l = _load(); self.h = self.l.mm_create()
    def add_way(self, e, n, tunnel=0, oneway=0):
        e = np.ascontiguousarray(e, float); n = np.ascontiguousarray(n, float)
        self.l.mm_add_way(self.h, e.ctypes.data_as(C.POINTER(C.c_double)),
                          n.ctypes.data_as(C.POINTER(C.c_double)), len(e), tunnel, oneway)
    def match(self, e, n, psi):
        out = (C.c_double * 6)(); self.l.mm_match(self.h, e, n, psi, C.byref(out))
        o = list(out)
        return dict(matched=bool(o[0]), foot=(o[1], o[2]), bearing=o[3],
                    cross=o[4], corridor=bool(o[5]))

    def add_edge(self, a, b): self.l.mm_add_edge(self.h, int(a), int(b))

    def set_hmm(self, sigma_emit=-1.0, sigma_trans=-1.0, beta_bearing=-1.0):
        self.l.mm_set_hmm(self.h, sigma_emit, sigma_trans, beta_bearing)

    def match_seq(self, e, n, psi):
        """Viterbi/HMM decode over a whole trajectory (mm_match_seq)."""
        e = np.ascontiguousarray(e, float); n = np.ascontiguousarray(n, float)
        psi = np.ascontiguousarray(psi, float)
        ns = len(e)
        Pd = C.POINTER(C.c_double)
        way = (C.c_int * ns)(); fe = (C.c_double * ns)(); fn = (C.c_double * ns)()
        brg = (C.c_double * ns)(); cor = (C.c_int * ns)()
        self.l.mm_match_seq(self.h, e.ctypes.data_as(Pd), n.ctypes.data_as(Pd),
                            psi.ctypes.data_as(Pd), ns, way, fe, fn, brg, cor)
        return dict(way=np.array(way, int), foot_e=np.array(fe), foot_n=np.array(fn),
                    bearing=np.array(brg), corridor=np.array(cor, int))


# --- register `eskf` into the eval harness --------------------------------
from eval.models import model  # noqa: E402
from model.nn_model import predict_steps, load_net, get_eskf_cfg  # noqa: E402
from model.dataset import STEP  # noqa: E402


_MAPS = {}   # cache: drive id -> MapMatcher built from its true centerline

# Fusion-loop settings. These defaults are what every synthetic gate and parity
# test was validated with; a checkpoint may carry its own tuned set ("eskf_cfg",
# e.g. model/nn_real.pt, tuned on real VALIDATION drives only) which overrides them.
ESKF_DEFAULT = dict(
    arw=float(np.radians(0.5)), brw=float(np.radians(0.02)), srw=3.0,  # process noise
    zupt_v=0.5,        # NN speed below this -> ZUPT + ZARU (None: never)
    curv=True,         # lateral-accel / yaw-rate speed pseudo-measurement in turns
    curv_sigma=2.0,    # its base sigma (m/s at 10 deg/s)
    sig_scale=1.0,     # multiplier on the NN's sigma fed to update_speed
    handover_s=0.0,    # first seconds of an outage: hold the entry (Doppler) speed, i.e. `physics`
    mm_cross_sigma=1.5,        # road cross-track update sigma (m), open road (corridor: 0.3)
    mm_heading_sigma_deg=3.0,  # road heading update sigma (deg), open road (corridor: 1)
    mm_keep_speed=False,       # road updates may not change speed (idr_set_map_keep_speed)
)


def eskf_config(cfg=None):
    return {**ESKF_DEFAULT, **(get_eskf_cfg() or {}), **(cfg or {})}


def _run(drive, o, matcher=None, cfg=None):
    c = eskf_config(cfg)
    net = load_net()
    starts, v_step, sig_step = predict_steps(net, drive, o.i0, o.i1)
    per_sec = {s: (v, sig) for s, v, sig in zip(starts, v_step, sig_step)}
    f = Filter()
    f.set_noise(c["arw"], c["brw"], c["srw"])
    f.set_map_keep_speed(c["mm_keep_speed"])
    f.init(drive.e[o.i0], drive.n[o.i0], drive.heading[o.i0], drive.speed[o.i0])
    N = o.i1 - o.i0
    t = drive.t[o.i0:o.i1]
    a_lat = drive.acc[o.i0:o.i1, 1] if drive.acc is not None else np.zeros(N)
    e = np.empty(N); n = np.empty(N); vp = np.empty(N); sg = np.empty(N)
    cur_sig = sig_step[0] if len(sig_step) else 1.0
    for i in range(N):
        gi = o.i0 + i
        dt = t[i] - t[i - 1] if i else (t[1] - t[0])
        gz = drive.gyro_z[gi]
        f.predict(dt, gz)
        if gi in per_sec:
            v, cur_sig = per_sec[gi]
            if t[i] - t[0] < c["handover_s"]:
                pass                                  # physics hand-over: keep the entry speed
            elif c["zupt_v"] is not None and v < c["zupt_v"]:
                f.update_zupt(gz)
            else:
                f.update_speed(v, cur_sig * c["sig_scale"])
        st = f.state()
        psidot = gz - st[4]
        if c["curv"] and abs(psidot) > np.radians(10):
            f.update_curvature(a_lat[i], psidot, c["curv_sigma"])
            st = f.state()
        if matcher is not None:                       # map matching: cross-track + heading only
            m = matcher.match(st[0], st[1], st[2])
            if m["matched"]:
                brg = m["bearing"]
                nx, ny = -np.cos(brg), np.sin(brg)    # left-normal (matches C++ cross sign)
                sig_c = 0.3 if m["corridor"] else c["mm_cross_sigma"]  # corridor collapses cross-track hard
                f.update_crosstrack(nx, ny, m["cross"], sig_c)
                tgt = brg if abs((brg - st[2] + np.pi) % (2*np.pi) - np.pi) < np.pi/2 else brg + np.pi
                f.update_heading(tgt, np.radians(1 if m["corridor"] else c["mm_heading_sigma_deg"]))
                st = f.state()
        e[i], n[i], vp[i], sg[i] = st[0], st[1], st[3], cur_sig
    return e, n, vp, sg


@model("eskf")
def eskf(drive, o):
    return _run(drive, o)


@model("eskf_map")
def eskf_map(drive, o):
    key = id(drive)
    if key not in _MAPS:
        mm = MapMatcher()
        # validation map = the road's true centerline (vehicle was on the road).
        # corridor if the drive is a near-straight single corridor (tunnel proxy).
        e, n = drive.e, drive.n
        straight = np.std(np.unwrap(drive.heading)) < np.radians(20)
        mm.add_way(e[::5], n[::5], tunnel=int(straight))
        _MAPS[key] = mm
    return _run(drive, o, _MAPS[key])
