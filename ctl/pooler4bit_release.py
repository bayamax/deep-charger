#!/usr/bin/env python3
"""The shipped 4-bit pooler: the MLX-packed GPTQ file with its float parts (query, layernorms, biases, out_scale)
taken in float32 from the dequantized file that was evaluated, so the release reproduces the measured weights
exactly. Checks that before writing.

  usage: python3 pooler4bit_release.py <pooler_gptq_mix_mlx.safetensors> <pooler_gptq_mix_dq.safetensors> <out>
"""
import sys, hashlib
import numpy as np
from safetensors.numpy import load_file, save_file
m, d = load_file(sys.argv[1]), load_file(sys.argv[2])
out = {}
sh = np.arange(0, 32, 4, dtype=np.uint32); nq = 0
for k in d:
    if m[k].dtype == np.uint32:
        w = m[k]; rows = w.shape[0]
        c = ((w[..., None] >> sh) & 15).reshape(rows, -1).astype(np.float32)
        r = c * np.repeat(m[k + ".scales"].astype(np.float32), 64, 1) + np.repeat(m[k + ".biases"].astype(np.float32), 64, 1)
        assert np.array_equal(r, d[k]), k
        out[k] = w; out[k + ".scales"] = m[k + ".scales"]; out[k + ".biases"] = m[k + ".biases"]; nq += 1
    else:
        out[k] = np.ascontiguousarray(d[k].astype(np.float32))
assert nq == 18 and len(out) == 100, (nq, len(out))
save_file(out, sys.argv[3], metadata={"format": "mlx-affine", "bits": "4", "group_size": "64"})
print(f"POOLER4_RELEASE_OK {nq} quantized + {len(out) - 3 * nq} float32 tensors, sha256 {hashlib.sha256(open(sys.argv[3], 'rb').read()).hexdigest()}")
