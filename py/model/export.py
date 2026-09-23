"""Export a trained SpeedNet checkpoint to ONNX for on-device inference.

    python -m model.export                 # model/nn.pt -> model/nn.onnx (+ parity)
    python -m model.export --ckpt A --onnx B
    python -m model.export --ckpt model/nn_real.pt \
        --onnx ../android/app/src/main/assets/nn_real.onnx --profile   # + phone profile

Phase 2's deliverable is "run on device via onnxruntime". This produces that
artifact from a committed checkpoint WITHOUT retraining, exports a dynamic batch
axis (nn_model feeds the net one row per 1 s step, a whole outage at a time, so
the graph must accept any batch size), and verifies the ONNX graph reproduces
the torch outputs (mu/logvar/slip/cls) to <1e-4 -- so the deployed net IS the
gated net, not a lookalike. train.py calls export_onnx() at the end of training.

--profile also writes <onnx stem>.profile.json beside the graph: the checkpoint's
speed calibration (a,b,s) and its fully resolved fusion settings (core_bridge.
ESKF_DEFAULT overridden by the checkpoint's eskf_cfg -- exactly what
core_bridge.eskf_config gives the harness). The phone (SpeedNet.kt/FusionEngine.kt)
and its host mirror (phase6_check.py) both read that file, so the calibration and
the val-tuned filter settings are never hand-copied into Kotlin.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from model.tcn import SpeedNet
from model.nn_model import DEFAULT

OUT_NAMES = ["mu", "logvar", "slip", "cls"]


def export_onnx(net, path, opset=17):
    """Export with batch as a dynamic axis so any number of 1 s steps runs."""
    net.eval()
    dummy = torch.randn(1, 9, 20)
    axes = {n: {0: "batch"} for n in ["imu", *OUT_NAMES]}
    torch.onnx.export(net, dummy, path, input_names=["imu"], output_names=OUT_NAMES,
                      opset_version=opset, dynamo=False, dynamic_axes=axes)
    return path


def verify_parity(net, path, batches=(1, 8, 37), atol=1e-4, seed=0):
    """Run the ONNX graph under onnxruntime and confirm it matches torch. Returns
    the worst max|Δ| per output; raises AssertionError if any exceeds atol."""
    import onnxruntime as ort
    net.eval()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    g = torch.Generator().manual_seed(seed)
    worst = {n: 0.0 for n in OUT_NAMES}
    for b in batches:
        x = torch.randn(b, 9, 20, generator=g)
        with torch.no_grad():
            ref = [t.numpy() for t in net(x)]
        got = sess.run(None, {"imu": x.numpy()})
        for n, a, c in zip(OUT_NAMES, ref, got):
            worst[n] = max(worst[n], float(np.max(np.abs(a - c))))
    bad = {n: d for n, d in worst.items() if d > atol}
    assert not bad, f"ONNX/torch parity exceeded {atol}: {bad}"
    return worst


def load_checkpoint(path=DEFAULT):
    ckpt = torch.load(path, map_location="cpu")
    net = SpeedNet(); net.load_state_dict(ckpt["state"]); net.eval()
    return net


def profile_path(onnx_path):
    return os.path.splitext(onnx_path)[0] + ".profile.json"


def build_profile(ckpt_path, onnx_path):
    """The phone-side contract for one checkpoint: which graph, its calibration,
    and the fusion settings it was validated with (defaults + checkpoint override)."""
    from core_bridge import ESKF_DEFAULT          # needs the native core; only for --profile
    ckpt = torch.load(ckpt_path, map_location="cpu")
    calib = ckpt.get("calib") or {"a": 0.0, "b": 1.0, "s": 1.0}
    with open(onnx_path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return {
        "model": os.path.basename(onnx_path),
        "checkpoint": os.path.basename(ckpt_path),
        "onnx_sha256": sha,
        "calib": {k: float(calib[k]) for k in ("a", "b", "s")},
        "eskf": {**ESKF_DEFAULT, **(ckpt.get("eskf_cfg") or {})},
    }


def write_profile(ckpt_path, onnx_path):
    path = profile_path(onnx_path)
    with open(path, "w") as fh:
        json.dump(build_profile(ckpt_path, onnx_path), fh, indent=2)
        fh.write("\n")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=DEFAULT, help="checkpoint .pt (default model/nn.pt)")
    ap.add_argument("--onnx", default=None, help="output .onnx (default: <ckpt>.onnx)")
    ap.add_argument("--profile", action="store_true",
                    help="also write <onnx stem>.profile.json (calib + fusion settings) for the phone")
    args = ap.parse_args()
    onnx_path = args.onnx or os.path.splitext(args.ckpt)[0] + ".onnx"
    net = load_checkpoint(args.ckpt)
    export_onnx(net, onnx_path)
    print(f"exported {onnx_path} ({os.path.getsize(onnx_path) // 1024} KB) -- on-device via onnxruntime")
    worst = verify_parity(net, onnx_path)
    print("onnxruntime vs torch max|Δ|: " + ", ".join(f"{n}={d:.1e}" for n, d in worst.items()))
    print("PARITY OK (<1e-4)")
    if args.profile:
        print(f"wrote {write_profile(args.ckpt, onnx_path)}")


if __name__ == "__main__":
    main()
