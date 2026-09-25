"""The app's pure-Kotlin math, compiled on the host and checked against its Python
reference -- so a port is proven, not eyeballed.

  - nav/Level.kt (leveling the NN inputs into the z-up frame SpeedNet was trained
    in, incl. the re-mount snap) == model/mount.level_stream to 1e-9;
  - nav/Features.kt (window -> (9,20) feature tensor) == model/features.py;
  - nav/FusionHead.kt (the learned fusion head's GRU ensemble + features)
    == model/fusion_head.HeadRunner to 1e-9 on a whole outage;
  - nav/HeadingAids.kt (yaw-rate modes, magnetometer heading, mount guess,
    YawAlign) == heading_aids.py to 1e-12;
  - nav/Decimator.kt keeps the 10 Hz grid from a 200 Hz stream at real phone
    timestamps (the first version overflowed and dropped every sample);
  - nav/RoadHmm.kt (the Phase 9 HMM road matcher, 1 km window of roads) == road_hmm.py
    (whole network) step for step, on a synthetic 3 km city and on the real
    Coventry OSM network along a real val drive.

Compiled with the Kotlin compiler Gradle already caches plus JDK 17, through the
harness tools/kthost/Main.kt. Skipped where either is missing.
Input: a synthetic drive rotated into a portrait dash mount, re-mounted by 30/20
deg half way; the re-mount flags come from the native core's detector (aln), as
FusionEngine feeds them.
"""
import glob, os, subprocess
import numpy as np
import pytest

pytest.importorskip("core_bridge")
from core_bridge import Align                           # noqa: E402
from data.io_vnbd import synth_drive                    # noqa: E402
from model.mount import level_stream, YAW_SIGN          # noqa: E402
from model import features as F                         # noqa: E402
from phase4_gates import _rot                           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GC = os.path.expanduser("~/.gradle/caches/modules-2/files-2.1")
JDKS = sorted(glob.glob(os.path.expanduser("~/Library/Java/JavaVirtualMachines/jdk-17*/Contents/Home")))


def _jar(pattern):
    hits = sorted(glob.glob(os.path.join(GC, pattern)))
    return hits[-1] if hits else None


JARS = [_jar(p) for p in (
    "org.jetbrains.kotlin/kotlin-compiler-embeddable/1.9.24/*/*.jar",
    "org.jetbrains.kotlin/kotlin-stdlib/1.9.24/*/kotlin-stdlib-1.9.24.jar",
    "org.jetbrains.kotlin/kotlin-script-runtime/1.9.24/*/*.jar",
    "org.jetbrains.kotlin/kotlin-reflect/1.9.*/*/*.jar",
    "org.jetbrains.intellij.deps/trove4j/*/*/*.jar",
    "org.jetbrains/annotations/13.0/*/*.jar",
    "org.jetbrains.kotlin/kotlin-daemon-embeddable/1.9.24/*/*.jar",
    "org.jetbrains.kotlinx/kotlinx-coroutines-core-jvm/1.6.4/*/*.jar")]
pytestmark = pytest.mark.skipif(not JDKS or None in JARS, reason="JDK 17 / cached Kotlin compiler not found")


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    java = os.path.join(JDKS[-1], "bin", "java")
    out = str(tmp_path_factory.mktemp("kthost") / "kthost.jar")
    nav = os.path.join(ROOT, "android/app/src/main/java/com/offmaps/nav")
    subprocess.run([java, "-cp", ":".join(JARS), "org.jetbrains.kotlin.cli.jvm.K2JVMCompiler",
                    "-no-stdlib", "-no-reflect", "-classpath", JARS[1], "-jvm-target", "17", "-d", out,
                    f"{nav}/Level.kt", f"{nav}/Features.kt", f"{nav}/Decimator.kt",
                    f"{nav}/FusionHead.kt", f"{nav}/HeadingAids.kt", f"{nav}/RoadHmm.kt", f"{nav}/Enu.kt",
                    os.path.join(ROOT, "tools/kthost/Main.kt")],
                   check=True, capture_output=True)
    return lambda stdin, *args: subprocess.run([java, "-cp", f"{out}:{JARS[1]}", "com.offmaps.nav.MainKt", *args],
                                               input=stdin, check=True, capture_output=True, text=True).stdout


@pytest.fixture(scope="module")
def stream():
    d = synth_drive("kport", 90, seed=5)
    portrait = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], float)
    g = np.c_[d.gyro[:, :2], YAW_SIGN * d.gyro[:, 2]]      # physical rate about up
    acc, gyr = d.acc @ portrait.T, g @ portrait.T
    T = len(acc) // 2
    R = _rot(np.radians(30), np.radians(20))                # the re-mount
    acc[T:], gyr[T:] = acc[T:] @ R.T, gyr[T:] @ R.T
    aln, rem = Align(), []
    for i in range(len(acc)):                                # FusionEngine order: aln, then level
        aln.update(*acc[i], *gyr[i], 0.1)
        if aln.get()[3]:
            rem.append(i)
    return acc, gyr, rem, T


def test_remount_seen_after_the_remount_only(stream):
    _, _, rem, T = stream
    assert rem and all(i >= T for i in rem), rem


def test_level_kt_matches_python(harness, stream):
    acc, gyr, rem, _ = stream
    lines = "\n".join(" ".join(repr(float(x)) for x in (*acc[i], *gyr[i], float(i in rem)))
                      for i in range(len(acc)))
    out = harness(lines + "\n").strip().split("\n")
    kt = np.array([[float(x) for x in l.split()] for l in out[:-1]])
    al, gl = level_stream(acc, gyr, remounts=rem)
    assert np.max(np.abs(kt - np.c_[al, gl])) < 1e-9
    # and the leveled frame really is z-up after the re-mount settles
    assert abs(al[-1, 2] - 9.81) < 1.5 and abs(kt[-1, 2] - 9.81) < 1.5

    feat_kt = np.array([float(x) for x in out[-1].split()])
    feat_py = F.window_features(al[-F.WIN:], gl[-F.WIN:]).reshape(-1)
    assert np.max(np.abs(feat_kt - feat_py)) < 1e-5          # Kotlin emits float32


def test_decimator_keeps_10hz_at_phone_timestamps(harness):
    """elapsedRealtimeNanos on a phone up for a day: 200 Hz for 10 s -> 100 kept.
    The old `t - Long.MIN_VALUE` check overflowed and kept 0."""
    t0 = 86_400 * 10**9
    ts = "\n".join(str(t0 + i * 5_000_000) for i in range(2000))
    assert int(harness(ts + "\n", "decim").strip()) == 100


def _head_weights(path, runner):
    nets = runner.nets
    H = nets[0].gru.hidden_size
    out = [f"{len(nets)} {H}"]
    for net in nets:
        sd = net.state_dict()
        for k in ("ctx.0.weight", "ctx.0.bias", "gru.weight_ih_l0", "gru.weight_hh_l0",
                  "gru.bias_ih_l0", "gru.bias_hh_l0", "out.weight", "out.bias"):
            a = sd[k].double().numpy()
            a = a.reshape(a.shape[0], -1)
            out.append(f"{a.shape[0]} {a.shape[1]} " + " ".join(repr(float(x)) for x in a.reshape(-1)))
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def test_fusion_head_kt_matches_python(harness, tmp_path):
    import torch
    from model.fusion_head import FusionHead, HeadRunner
    torch.manual_seed(0)
    runner = HeadRunner([FusionHead(16), FusionHead(16)])       # random weights: exercises every path
    for n in runner.nets:
        for p in n.parameters():
            p.data.normal_(0, 0.4)
    wp = str(tmp_path / "w.txt"); _head_weights(wp, runner)
    rng = np.random.default_rng(1)
    pre = [(float(rng.uniform(0, 15)), float(rng.uniform(0, 12))) for _ in range(150)]
    v0, k, c = 11.3, 1.4, -0.7
    st = runner.start(dict(v0=v0, k=k, c=c, pre=pre))
    lines = [f"{v0} {k} {c} {len(pre)}"] + [f"{a} {b}" for a, b in pre]
    ref = []
    for tau in range(1, 41):
        acc = rng.normal([0, 0, 9.81], 0.5, (10, 3)); gyr = rng.normal(0, 0.05, (10, 3))
        vnn, sig = float(rng.uniform(0, 15)), float(rng.uniform(0.3, 4))
        ref.append(runner.step(st, dict(vnn=vnn, sig=sig, tau=float(tau), acc=acc, gyro=gyr)))
        lines.append(" ".join(repr(float(x)) for x in [vnn, sig, float(tau), *acc.reshape(-1), *gyr.reshape(-1)]))
    kt = np.array([[float(x) for x in l.split()] for l in harness("\n".join(lines) + "\n", "head", wp).split("\n") if l])
    assert np.max(np.abs(kt - np.array(ref))) < 1e-5            # torch float32 vs Kotlin double


def test_heading_aids_kt_matches_python(harness):
    import heading_aids as HA
    rng = np.random.default_rng(2)
    lines, ref = [], []
    for _ in range(60):
        g, uf, us, fw = rng.normal(0, 1, 3), rng.normal([0, 0, 9.8], 2, 3), rng.normal([0, 0, 9.8], 1, 3), rng.normal(0, 1, 3)
        v = float(rng.uniform(0, 25))
        for mode in ("fast", "slow", "coord"):
            lines.append(f"yaw {mode} " + " ".join(repr(float(x)) for x in [*g, *uf, *us, *fw, v]))
            ref.append([HA.yaw_rate(mode, tuple(g), tuple(uf), tuple(us), tuple(fw), v)])
        m, decl = rng.normal(0, 30, 3), float(rng.uniform(-0.2, 0.2))
        lines.append("mag " + " ".join(repr(float(x)) for x in [*m, *uf, *fw, decl]))
        ref.append([HA.mag_heading(tuple(m), tuple(uf), tuple(fw), decl)])
        lines.append("fwd " + " ".join(repr(float(x)) for x in uf))
        ref.append(list(HA.default_forward(tuple(uf))))
    ya = HA.YawAlign(); t = 0.0
    for _ in range(80):
        for _ in range(10):
            a1, a2 = (float(x) for x in rng.normal(0, 1, 2))
            lines.append(f"acc {a1!r} {a2!r}"); ya.add_acc(a1, a2)
        t += 1.0; v = float(rng.uniform(0, 20))
        lines.append(f"fix {t!r} {v!r}"); ya.on_fix(t, v); ref.append([ya.theta, float(ya.valid)])
    kt = [[float(x) for x in l.split()] for l in harness("\n".join(lines) + "\n", "aids").split("\n") if l]
    assert len(kt) == len(ref)
    assert max(float(np.max(np.abs(np.array(a) - np.array(b)))) for a, b in zip(kt, ref)) < 1e-12


# ---------------- RoadHmm.kt == road_hmm.py ----------------
K_M = np.pi / 180 * 6_371_000.0


def _to7(e, n, lat0, lon0):
    lat = lat0 + np.asarray(n) / K_M
    lon = lon0 + np.asarray(e) / (np.cos(np.radians(lat0)) * K_M)
    return np.round(lat * 1e7).astype(np.int64), np.round(lon * 1e7).astype(np.int64)


def _pieces(la7, lo7, n=32):
    for s in range(0, len(la7) - 1, n - 1):
        yield la7[s:s + n], lo7[s:s + n]


def _hmm_parity(harness, tmp_path, ways7, lat0, lon0, steps):
    """ways7: [(lat7[], lon7[], tunnel, oneway, hw_code)]; steps: [(travel, e, n, psi, v, sigma)]."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from osm_layers import HW_NAME
    from road_hmm import RoadGraph, HMMMatcher
    rp = tmp_path / "roads.txt"
    with open(rp, "w") as f:
        f.write(f"{lat0!r} {lon0!r} {len(ways7)}\n")
        for la, lo, t, o, c in ways7:
            f.write(f"{t} {o} {c} {len(la)} " + " ".join(f"{a} {b}" for a, b in zip(la, lo)) + "\n")
    ways = [(np.asarray(la) / 1e7, np.asarray(lo) / 1e7, t, o, HW_NAME.get(c)) for la, lo, t, o, c in ways7]
    hmm = HMMMatcher(RoadGraph(ways, lat0, lon0, exclude=("service",)))
    ref = []
    for tr, e, n, psi, v, sg in steps:
        hmm.add_travel(tr)
        m = hmm.update((e, n), psi, v, sg)
        ref.append(None if m is None else [m["way"], m["vtx"], *m["foot"], m["bearing"], m["dist"], m["conf"],
                                           m["seg_conf"], m["seg_len"], m["end_dist"], m["half_width"], float(m["tunnel"])])
    stdin = "\n".join(" ".join(repr(float(x)) for x in s) for s in steps) + "\n"
    kt = [None if l.strip() == "none" else [float(x) for x in l.split()]
          for l in harness(stdin, "hmm", str(rp)).split("\n") if l.strip()]
    assert len(kt) == len(ref)
    matched = 0
    for i, (a, b) in enumerate(zip(ref, kt)):
        assert (a is None) == (b is None), f"step {i}: python {a} kotlin {b}"
        if a is None:
            continue
        assert a[:2] == b[:2], f"step {i}: segment python {a[:2]} kotlin {b[:2]}"   # same road segment
        assert np.max(np.abs(np.array(a[2:]) - np.array(b[2:]))) < 1e-9, f"step {i}"
        matched += 1
    return matched, ref


def _drive_steps(te, tn, th, rng, v=11.0):
    """1 Hz filter inputs along a truth track: 60 s GNSS-tight, then 90 s of drift (sigma up to
    140 m, past the 150 m candidate cap), repeated; estimate = truth + a random-walk error."""
    steps, drift = [], np.zeros(2)
    for i in range(1, len(te)):
        ph = i % 150
        if ph < 60:
            drift *= 0.5; sg = 5.0
        else:
            drift += rng.normal(0, 1.2, 2); sg = 5.0 + 1.5 * (ph - 60)
        tr = float(np.hypot(te[i] - te[i - 1], tn[i] - tn[i - 1]) * rng.uniform(0.9, 1.1))
        steps.append((tr, te[i] + drift[0] + rng.normal(0, 1.5), tn[i] + drift[1] + rng.normal(0, 1.5),
                      th[i] + rng.normal(0, 0.05), v * rng.uniform(0.8, 1.2), sg))
    return steps


def test_road_hmm_kt_matches_python_on_a_synthetic_city(harness, tmp_path):
    lat0, lon0 = 17.40, 78.45
    rng = np.random.default_rng(3)
    ways7, B, L = [], 150.0, 3000.0
    grid = np.arange(0.0, L + 1, B)
    for k, c in enumerate(grid):                          # N-S and E-W streets, a vertex every 25 m
        pts = np.arange(0.0, L + 1, 25.0)
        for ns in (True, False):
            e, n = (np.full_like(pts, c), pts) if ns else (pts, np.full_like(pts, c))
            la, lo = _to7(e, n, lat0, lon0)
            one = int(k % 5 == 2)                         # every fifth street one-way
            code = 5 if k % 4 == 0 else 12                # primary / residential
            for pl, po in _pieces(la, lo):
                ways7.append((pl, po, 0, one, code))
    for c in grid[1:-1:3]:                                # service lanes 6 m off, and an unconnected street 12 m off
        pts = np.arange(20.0, L - 20, 25.0)
        for off, code in ((6.0, 14), (12.0, 12)):
            la, lo = _to7(np.full_like(pts, c + off), pts, lat0, lon0)
            for pl, po in _pieces(la, lo):
                ways7.append((pl, po, 0, 0, code))
    # truth: a route along the grid with turns, 11 m/s, sampled at 1 Hz, ~2.6 km of driving
    wp = [(0.0, 0.0), (0.0, 900.0), (900.0, 900.0), (900.0, 1800.0), (1650.0, 1800.0), (1650.0, 2400.0)]
    te, tn, th = [], [], []
    for (e0, n0), (e1, n1) in zip(wp, wp[1:]):
        d = np.hypot(e1 - e0, n1 - n0); k = int(d / 11.0)
        f = np.arange(k) / k
        te += list(e0 + f * (e1 - e0)); tn += list(n0 + f * (n1 - n0)); th += [np.arctan2(e1 - e0, n1 - n0)] * k
    steps = _drive_steps(np.array(te), np.array(tn), np.array(th), rng)
    matched, ref = _hmm_parity(harness, tmp_path, ways7, lat0, lon0, steps)
    assert matched > 0.8 * len(steps)
    assert any(r is not None and r[6] < 0.9 for r in ref) and any(r is not None and r[6] > 0.99 for r in ref)


DATA = os.environ.get("OFFMAPS_IOVNBD", os.path.expanduser("~/OffMaps-data/IO-VNBD-sync"))
COV = os.path.join(ROOT, "map", "coventry", "roads.bin")


@pytest.mark.skipif(not (os.path.isdir(DATA) and os.path.exists(COV)), reason="IO-VNBD data / Coventry roads not found")
def test_road_hmm_kt_matches_python_on_real_roads(harness, tmp_path):
    """The real OSM network of the IO-VNBD area (38,754 pieces, service roads included in
    the file), with a noisy trajectory along 25 min of a real val drive."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from osm_layers import read_roads_bin, HW_CODE
    from data.iovnbd_sync import load_sync_dir
    from model.train_real import split_drives
    d = max(split_drives(load_sync_dir(DATA, verbose=False))["val"], key=len)
    lat0, lon0 = d.meta["lat0"], d.meta["lon0"]
    ways7 = [(np.round(la * 1e7).astype(np.int64), np.round(lo * 1e7).astype(np.int64), t, o, HW_CODE.get(c, 0))
             for la, lo, t, o, c in read_roads_bin(COV, with_class=True)]
    i = np.arange(0, min(len(d), 25 * 600), 10)                # 1 Hz
    steps = _drive_steps(d.e[i], d.n[i], d.heading[i], np.random.default_rng(4), v=12.0)
    matched, _ = _hmm_parity(harness, tmp_path, ways7, lat0, lon0, steps)
    assert matched > 0.8 * len(steps)
