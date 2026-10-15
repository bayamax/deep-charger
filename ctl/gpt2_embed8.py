#!/usr/bin/env python3
"""Finish gpt2_gptq.py's packed directory: the tied token table back at 8 bits (round-to-nearest from the original, groups
of 64) - at 4 bits it carried almost all of the loss (WikiText-2 perplexity 38.1 against 27.5 with it in float, 27.8 at 8
bits; the blocks stay GPTQ 4-bit) - and every non-quantized tensor (scales, biases, norms, position table) in fp32, so MLX
runs the stream in fp32: GPT-2's residual carries outliers in the thousands, and in fp16 the 4-bit model's logits drift by
up to 3.4 from its own dequantized weights (0.1 in fp32). The 8-bit table is declared per layer in config.json, which
mlx-lm's loader reads.

  python3 gpt2_embed8.py gpt2-4bit-gptq gpt2-4bit-gptq-e8
"""
import json, os, shutil, sys

import numpy as np
import torch
from safetensors.numpy import load_file, save_file
from transformers import GPT2LMHeadModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q4  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]; GROUP, BITS = 64, 8
os.makedirs(dst, exist_ok=True)
w = load_file(os.path.join(src, "model.safetensors"))
w = {k: (v.astype(np.float32) if v.dtype == np.float16 else v) for k, v in w.items()}
E = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32).transformer.wte.weight.detach()
q, s, b = q4.affine_params(E, GROUP, BITS)
codes = q.reshape(E.shape[0], -1).to(torch.int64); shifts = torch.arange(0, 32, BITS, dtype=torch.int64)
w["model.wte.weight"] = ((codes.reshape(E.shape[0], -1, 32 // BITS) << shifts).sum(-1) & 0xFFFFFFFF).numpy().astype("uint32")
w["model.wte.scales"] = s.reshape(E.shape[0], -1).float().numpy(); w["model.wte.biases"] = b.reshape(E.shape[0], -1).float().numpy()
save_file(w, os.path.join(dst, "model.safetensors"))
for f in ("tokenizer.json", "tokenizer_config.json", "scores.json"):
    if os.path.exists(os.path.join(src, f)): shutil.copy(os.path.join(src, f), os.path.join(dst, f))
cfg = json.load(open(os.path.join(src, "config.json")))
cfg["quantization"]["model.wte"] = {"group_size": GROUP, "bits": BITS, "mode": "affine"}; cfg["torch_dtype"] = "float32"
cfg["quantization_config"] = cfg["quantization"]
json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
print(f"[out] {dst}: blocks GPTQ 4-bit, token table 8-bit, the rest fp32 ({os.path.getsize(os.path.join(dst, 'model.safetensors'))/1e6:.0f} MB)")
