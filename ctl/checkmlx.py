#!/usr/bin/env python3
"""Prove the deployable directory holds the numbers we measured.

dwq.py writes two directories from the same trained parameters: one packed the way the app reads it,
one dequantized the way the evaluator reads it. Everything we say about accuracy is measured on the
second, so the claim only transfers to the phone if unpacking the first reproduces the second
exactly. This unpacks every quantized tensor and compares.

  python3 checkmlx.py /root/dwq_mlx4 /root/dwq_hf
"""
import json, sys
import numpy as np
import torch
from safetensors.numpy import load_file
from safetensors.torch import load_file as load_torch

MLX, HF = sys.argv[1], sys.argv[2]
cfg = json.load(open(MLX + "/config.json"))
bits = cfg["quantization"]["bits"]; group = cfg["quantization"]["group_size"]
q = load_file(MLX + "/model.safetensors")
h = {k: v.float().numpy() for k, v in load_torch(HF + "/model.safetensors").items()}   # bf16 is not a numpy dtype
stems = sorted(k[: -len(".scales")] for k in q if k.endswith(".scales"))
print(f"[check] {len(stems)} quantized tensors, {bits} bits, group {group}")

bad = 0
for s in stems:
    packed = q[s + ".weight"]                       # uint32, [rows, in*bits/32]
    sc = q[s + ".scales"].astype(np.float32)        # [rows, in/group]
    bi = q[s + ".biases"].astype(np.float32)
    per_int = 32 // bits
    codes = np.right_shift(packed[:, :, None].astype(np.uint32),
                           (np.arange(per_int, dtype=np.uint32) * bits)[None, None, :])
    codes = (codes & ((1 << bits) - 1)).reshape(packed.shape[0], -1)
    w = (codes.reshape(codes.shape[0], -1, group).astype(np.float32) * sc[:, :, None]
         + bi[:, :, None]).reshape(codes.shape)
    ref = h[s + ".weight"]
    if w.shape != ref.shape:
        print(f"  SHAPE {s}: {w.shape} vs {ref.shape}"); bad += 1; continue
    d = np.abs(w - ref).max()
    if d > 0:
        print(f"  DIFF  {s}: max |unpacked - dequantized| = {d:.3e}"); bad += 1

# the tensors the conversion does not touch must still be present and finite
plain = [k for k in q if not k.endswith((".scales", ".biases")) and k[: -len(".weight")] not in stems]
for k in plain:
    if not np.isfinite(q[k].astype(np.float32)).all():
        print(f"  NONFINITE {k}"); bad += 1
print(f"[check] {len(plain)} tensors left unquantized (norms and attention biases)")
print("MLX_CHECK_OK" if bad == 0 else f"MLX_CHECK_FAILED ({bad} tensors)")
