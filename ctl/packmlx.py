#!/usr/bin/env python3
"""Pack a jointfit run into the directory the app loads.

jointfit.py writes the dequantized directory the evaluator measures and saves the trained scales and
biases in its state file, but not the packed directory. This rebuilds it: the codes are the ones the
run used (affine_params of the base weights, no clip search), the scales and biases come from the
state, and everything unquantized is copied from the dequantized directory in fp16. checkmlx.py then
proves the result unpacks to what was measured.

  python3 packmlx.py --base /root/eval_hf200 --hf /root/sft_hf_s1 --state /root/sft/s1.pt --out /root/sft_mlx4_s1
"""
import argparse, json, os, shutil
import numpy as np
import torch
from safetensors.torch import load_file
from safetensors.numpy import save_file as save_np
import q4

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="/root/eval_hf200"); ap.add_argument("--hf", required=True)
ap.add_argument("--state", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--group", type=int, default=64); ap.add_argument("--bits", type=int, default=4)
A = ap.parse_args()


def plain_name(n):
    return n.replace("base_model.model.", "").replace(".base_layer", "") + ".weight"


base = load_file(os.path.join(A.base, "model.safetensors"))
sd = load_file(os.path.join(A.hf, "model.safetensors"))
st = torch.load(A.state, map_location="cpu")
print(f"[pack] state from step {st.get('step')} val {st.get('val'):.4f}: {len(st['q'])} quantized tensors")
mx, quant = {}, set()
shifts = torch.arange(0, 32, A.bits, dtype=torch.int64)
with torch.no_grad():
    for name, (s_, b_) in st["q"].items():
        k = plain_name(name)
        w = base[k].float()
        q, _, _ = q4.affine_params(w, A.group, A.bits)
        rows = w.shape[0]
        c = q.to(torch.int64).reshape(rows, -1)
        packed = (c.reshape(rows, -1, 32 // A.bits) << shifts).sum(-1)
        s_ = s_.float().reshape(rows, -1); b_ = b_.float().reshape(rows, -1)
        # what the runtime will compute, checked against what the evaluator measured
        deq = (q.reshape(rows, -1, A.group) * s_[:, :, None] + b_[:, :, None]).reshape(w.shape)
        err = (deq.to(sd[k].dtype).float() - sd[k].float()).abs().max().item()
        assert err == 0.0, f"{k}: dequantized weight differs from the measured directory by {err}"
        mx[k] = (packed & 0xFFFFFFFF).numpy().astype("uint32")
        mx[k[: -len(".weight")] + ".scales"] = s_.to(torch.float16).numpy()
        mx[k[: -len(".weight")] + ".biases"] = b_.to(torch.float16).numpy()
        quant.add(k)
for k, v in sd.items():
    if k in quant:
        continue
    mx[k] = v.to(torch.float16).numpy()
os.makedirs(A.out, exist_ok=True)
save_np(mx, os.path.join(A.out, "model.safetensors"))
cfg = json.load(open(os.path.join(A.hf, "config.json")))
cfg["quantization"] = cfg["quantization_config"] = {"group_size": A.group, "bits": A.bits, "mode": "affine"}
cfg["torch_dtype"] = "float16"
json.dump(cfg, open(os.path.join(A.out, "config.json"), "w"), indent=2)
for f in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
          "vocab.json", "merges.txt", "added_tokens.json"):
    p = os.path.join(A.hf, f)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(A.out, f))
print(f"[out] {A.out}: {len(mx)} tensors, {len(quant)} quantized")
print("PACKMLX_DONE")
