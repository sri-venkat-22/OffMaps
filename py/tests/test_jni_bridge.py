"""The phone links the whole core and binds all of it, 1:1 with Kotlin.

Phase 6 checked this once by hand ("32 JNI symbols match the Kotlin externals").
Since then the core grew (idr_get_cov, idr_set_map_keep_speed, Phase 7's vib /
idr3d / mm_match_seq) and Phase 7 never reached the phone: CMake did not even
compile vib.cpp / eskf3d.cpp. Pinned here, host-only:

  - every Kotlin `external fun` in IdrNative.kt has a JNI export and vice versa;
  - every function declared in core/idr.h is called from idr_jni.cpp;
  - every core source the host library builds (core/build.sh) is in the NDK
    CMakeLists, so nothing links on the host that the phone lacks;
  - idr_jni.cpp + the core compile against the JDK's jni.h and export exactly
    the Kotlin set (skipped without a JDK).
"""
import glob, os, re, subprocess
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KT = os.path.join(ROOT, "android/app/src/main/java/com/offmaps/nav/IdrNative.kt")
JNI = os.path.join(ROOT, "android/app/src/main/cpp/idr_jni.cpp")
CMAKE = os.path.join(ROOT, "android/app/src/main/cpp/CMakeLists.txt")
HDR = os.path.join(ROOT, "core/idr.h")
BUILD = os.path.join(ROOT, "core/build.sh")


def _read(p):
    with open(p) as fh:
        return fh.read()


KT_EXT = set(re.findall(r"external fun (\w+)\(", _read(KT)))
JNI_EXP = set(re.findall(r"Java_com_offmaps_nav_IdrNative_(\w+)\(", _read(JNI)))


def test_kotlin_externals_match_jni_exports():
    assert KT_EXT == JNI_EXP, dict(kotlin_only=KT_EXT - JNI_EXP, jni_only=JNI_EXP - KT_EXT)


def test_every_core_function_is_bound():
    hdr = re.sub(r"/\*.*?\*/", "", _read(HDR), flags=re.S)
    decls = set(re.findall(r"\b((?:idr3d|idr|spc|aln|gq|mm|vib)_\w+)\s*\(", hdr))
    called = set(re.findall(r"\b((?:idr3d|idr|spc|aln|gq|mm|vib)_\w+)\s*\(", _read(JNI)))
    assert decls and decls <= called, sorted(decls - called)


def test_phone_compiles_every_core_source():
    host = set(re.findall(r"core/(\w+\.cpp)", _read(BUILD)))
    phone = set(re.findall(r"\$\{CORE_DIR\}/(\w+\.cpp)", _read(CMAKE)))
    assert host and host <= phone, sorted(host - phone)


JDK = sorted(glob.glob(os.path.expanduser("~/Library/Java/JavaVirtualMachines/jdk-17*/Contents/Home")))


@pytest.mark.skipif(not JDK, reason="no JDK (jni.h) found")
def test_bridge_compiles_and_exports_the_kotlin_set(tmp_path):
    inc = os.path.join(JDK[-1], "include")
    plat = [d for d in glob.glob(os.path.join(inc, "*")) if os.path.isdir(d)]
    srcs = [JNI] + [os.path.join(ROOT, "core", f) for f in
                    sorted(set(re.findall(r"\$\{CORE_DIR\}/(\w+\.cpp)", _read(CMAKE))))]
    lib = str(tmp_path / "libidrjni_host.so")
    subprocess.run(["clang++", "-std=c++17", "-O1", "-fPIC", "-shared", "-I", inc,
                    *sum((["-I", p] for p in plat), []), "-I", os.path.join(ROOT, "core"),
                    *srcs, "-o", lib], check=True, capture_output=True)
    nm = subprocess.run(["nm", "-gU", lib], check=True, capture_output=True, text=True).stdout
    exported = set(re.findall(r"_?Java_com_offmaps_nav_IdrNative_(\w+)", nm))
    assert exported == KT_EXT, dict(missing=KT_EXT - exported, extra=exported - KT_EXT)
