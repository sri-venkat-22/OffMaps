"""The app's pure-Kotlin math, compiled on the host and checked against its Python
reference -- so a port is proven, not eyeballed.

  - nav/Level.kt (leveling the NN inputs into the z-up frame SpeedNet was trained
    in, incl. the re-mount snap) == model/mount.level_stream to 1e-9;
  - nav/Features.kt (window -> (9,20) feature tensor) == model/features.py;
  - nav/Decimator.kt keeps the 10 Hz grid from a 200 Hz stream at real phone
    timestamps (the first version overflowed and dropped every sample).

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
