"""The phone ships the model it claims to, with the settings it was validated with.

android/app/src/main/assets/<name>.onnx + <name>.profile.json are written by
`python -m model.export --ckpt model/<name>.pt --onnx <assets>/<name>.onnx --profile`.
FusionEngine/SpeedNet read calibration and fusion settings from the profile instead
of hard-coding them (the old app hard-coded nn.pt's calib and the synthetic ESKF
settings, which is how "retrained on real data" could exist on host but not on the
phone). Pinned here:

  - each shipped graph IS its checkpoint (onnxruntime vs torch < 1e-4) and its
    sha256 matches the profile, so a stale export can't ship;
  - the profile's calib and eskf equal what the host harness gives that checkpoint
    (checkpoint calib; core_bridge.ESKF_DEFAULT overridden by its eskf_cfg);
  - the app's default profile and the loop constants the host mirror copies from
    FusionEngine.kt are the ones phase6_check.py actually uses.
"""
import hashlib, json, os, re
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
CB = pytest.importorskip("core_bridge")
import phase6_check as P                                   # noqa: E402
from model.export import load_checkpoint, verify_parity    # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
KT = os.path.join(P.ASSETS, "..", "java", "com", "offmaps", "nav")
PROFILES = sorted(f[:-len(".profile.json")] for f in os.listdir(P.ASSETS) if f.endswith(".profile.json"))


def _kt(name):
    with open(os.path.join(KT, name)) as fh:
        return fh.read()


def test_both_profiles_ship():
    assert {"nn", "nn_real"} <= set(PROFILES), PROFILES


@pytest.mark.parametrize("name", PROFILES)
def test_shipped_graph_is_the_checkpoint(name):
    prof = P.load_profile(name)
    onnx_path = os.path.join(P.ASSETS, prof["model"])
    with open(onnx_path, "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == prof["onnx_sha256"], "profile sha256 is stale"
    net = load_checkpoint(os.path.join(MODEL_DIR, prof["checkpoint"]))
    worst = verify_parity(net, onnx_path, batches=(1, 8))
    assert max(worst.values()) < 1e-4, worst


@pytest.mark.parametrize("name", PROFILES)
def test_profile_matches_host_settings(name):
    prof = P.load_profile(name)
    ckpt = torch.load(os.path.join(MODEL_DIR, prof["checkpoint"]), map_location="cpu")
    assert prof["calib"] == pytest.approx(ckpt["calib"])
    assert prof["eskf"] == {**CB.ESKF_DEFAULT, **(ckpt.get("eskf_cfg") or {})}


def test_app_default_is_the_real_data_model_and_mirror_agrees():
    m = re.search(r'const val DEFAULT = "(\w+)"', _kt("SpeedProfile.kt"))
    assert m and m.group(1) == P.SHIPPED == "nn_real"


def test_mirror_constants_match_fusion_engine():
    src = _kt("FusionEngine.kt")
    assert re.search(r"REACQ_FIXES = (\d+)", src).group(1) == str(P.REACQ_FIXES)
    assert "K_MIN = 1.0 / 3.0" in src and P.K_MIN == pytest.approx(1 / 3)
    assert re.search(r"K_MAX = ([\d.]+)", src).group(1) == "3.0" and P.K_MAX == 3.0
    assert "Gq.spoof(cn0, svUsed, chi2, 0.0)" in src          # position chi2 only, as mirrored
    assert "drSteps > (p.handoverS * Features.HZ).toInt()" in src   # hand-over, as mirrored
    assert re.search(r"TRUST_APPLIED = ([\d.]+)", src).group(1) == str(P.TRUST_APPLIED)
    assert "if (trust >= TRUST_APPLIED) stepsSinceGnss = 0" in src
    assert "if (st[3] < SHOCK_MIN_SPEED) return" in src          # potholes only while driving
    assert 'optDouble("handover_s", 0.0)' in _kt("SpeedProfile.kt")
    assert "it.setMapKeepSpeed(p.mmKeepSpeed)" in src                 # map settings from the profile
    assert "else p.mmCrossSigma" in src and "else p.mmHeadingSigmaDeg" in src
