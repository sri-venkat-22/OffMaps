"""Export the shipped models for the browser engine (docs/engine/, the web twin of the app).

    cd py && PYTHONPATH=. python3 web_export.py

Writes docs/engine/model/:
  speednet.bin    float32 little-endian weights of the shipped SpeedNet (nn_real.pt), in the
                  order listed in speednet.json; BatchNorm folded into per-channel
                  scale/shift (eval mode, eps 1e-5), since it sits after the ReLU
  speednet.json   tensor table + the profile's calibration (a, b, s) and p(stopped) (a, b)
  profile.json    the phone's profile (nn_real.profile.json): the ESKF + live-loop settings
  fusion_head.json  the phone's fusion-head weights (FusionHead.kt reads the same file)

tests/test_web_engine.py checks the JS forward pass against torch and the whole JS loop
against edge_engine.py.
"""
from __future__ import annotations
import json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "android", "app", "src", "main", "assets")
OUT = os.path.join(ROOT, "docs", "engine", "model")
BN_EPS = 1e-5


def speednet_tensors(ckpt):
    """[(name, array)] in the order the JS forward pass reads them."""
    import torch
    sd = torch.load(ckpt, map_location="cpu")["state"]
    g = lambda k: sd[k].numpy().astype(np.float64)
    out = [("inp.weight", g("inp.weight")[:, :, 0]), ("inp.bias", g("inp.bias"))]
    for i in range(5):
        for conv, bn in (("0", "2"), ("3", "5")):
            p = f"tcn.{i}.net."
            scale = g(p + bn + ".weight") / np.sqrt(g(p + bn + ".running_var") + BN_EPS)
            shift = g(p + bn + ".bias") - g(p + bn + ".running_mean") * scale
            out += [(f"tcn.{i}.conv{conv}.weight", g(p + conv + ".weight")), (f"tcn.{i}.conv{conv}.bias", g(p + conv + ".bias")),
                    (f"tcn.{i}.bn{bn}.scale", scale), (f"tcn.{i}.bn{bn}.shift", shift)]
    for k in ("gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0",
              "disp.weight", "disp.bias", "cls.weight", "cls.bias"):
        out.append((k, g(k)))
    return out


def main():
    prof_path = os.path.join(ASSETS, "nn_real.profile.json")
    prof = json.load(open(prof_path))
    ckpt = os.path.join(HERE, "model", prof["checkpoint"])
    os.makedirs(OUT, exist_ok=True)
    table, blobs, off = [], [], 0
    for name, arr in speednet_tensors(ckpt):
        a = np.ascontiguousarray(arr, dtype="<f4")
        table.append(dict(name=name, shape=list(a.shape), offset=off))
        blobs.append(a.tobytes()); off += a.size
    with open(os.path.join(OUT, "speednet.bin"), "wb") as fh:
        fh.write(b"".join(blobs))
    feat = prof.get("feat") or {"version": 1, "win": 20}
    meta = dict(format="offmaps-speednet/1", checkpoint=prof["checkpoint"], onnx_sha256=prof["onnx_sha256"],
                channels=64, kernel=3, dilations=[1, 2, 4, 8, 16], feat=feat,
                calib=prof["calib"], pstop=prof.get("pstop"), floats=off, tensors=table)
    with open(os.path.join(OUT, "speednet.json"), "w") as fh:
        json.dump(meta, fh, indent=1); fh.write("\n")
    shutil.copyfile(prof_path, os.path.join(OUT, "profile.json"))
    shutil.copyfile(os.path.join(ASSETS, "fusion_head.json"), os.path.join(OUT, "fusion_head.json"))
    print(f"wrote {OUT}: speednet.bin ({4 * off // 1024} KB, {len(table)} tensors), speednet.json, profile.json, fusion_head.json")


if __name__ == "__main__":
    main()
