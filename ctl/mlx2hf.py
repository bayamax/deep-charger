#!/usr/bin/env python3
"""The app's MLX 4-bit directory -> the dequantized Hugging Face directory the evaluator loads (what gptq.py --out-hf
writes): every packed weight unpacked (uint32, 8 codes a word, low bits first) and rebuilt as codes x scale + bias per
group of 64, in bfloat16; everything else as it is. checkmlx.py proved the two forms agree in the other direction.

  usage: python3 mlx2hf.py --mlx /root/hfdl/release/g14-4bit-gptq-trained --out /root/g14q_hf
"""
import argparse, json, os, shutil
import numpy as np
import torch
from safetensors.numpy import load_file
from safetensors.torch import save_file
ap = argparse.ArgumentParser(); ap.add_argument("--mlx", required=True); ap.add_argument("--out", required=True)
A = ap.parse_args()
cfg = json.load(open(os.path.join(A.mlx, "config.json")))
qc = cfg.get("quantization") or cfg.get("quantization_config") or {}
G, B = int(qc.get("group_size", 64)), int(qc.get("bits", 4))
sd = load_file(os.path.join(A.mlx, "model.safetensors")); out = {}
shifts = np.arange(0, 32, B, dtype=np.uint32); mask = np.uint32((1 << B) - 1); nq = 0
for k, v in sd.items():
    if k.endswith(".scales") or k.endswith(".biases"):
        continue
    base = k[:-len(".weight")] if k.endswith(".weight") else None
    if base and base + ".scales" in sd:
        codes = ((v[..., None] >> shifts) & mask).reshape(v.shape[0], -1).astype(np.float32)
        s = np.repeat(sd[base + ".scales"].astype(np.float32), G, axis=1); b = np.repeat(sd[base + ".biases"].astype(np.float32), G, axis=1)
        out[k] = torch.from_numpy(codes * s + b).to(torch.bfloat16); nq += 1
    else:
        out[k] = torch.from_numpy(v.astype(np.float32)).to(torch.bfloat16)
os.makedirs(A.out, exist_ok=True)
save_file({k: v.contiguous() for k, v in out.items()}, os.path.join(A.out, "model.safetensors"), metadata={"format": "pt"})
for key in ("quantization", "quantization_config"): cfg.pop(key, None)
cfg["torch_dtype"] = "bfloat16"
json.dump(cfg, open(os.path.join(A.out, "config.json"), "w"), indent=2)
for f in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
    if os.path.exists(os.path.join(A.mlx, f)): shutil.copy(os.path.join(A.mlx, f), os.path.join(A.out, f))
print(f"MLX2HF_DONE {len(out)} tensors, {nq} dequantized (group {G}, {B} bits) -> {A.out}", flush=True)
