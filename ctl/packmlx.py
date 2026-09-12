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


DEV = "cuda" if torch.cuda.is_available() else "cpu"
base = load_file(os.path.join(A.base, "model.safetensors"))
sd = load_file(os.path.join(A.hf, "model.safetensors"))
st = torch.load(A.state, map_location="cpu")
print(f"[pack] state from step {st.get('step')} val {st.get('val'):.4f}: {len(st['q'])} quantized tensors")
mx, quant = {}, set()
shifts = torch.arange(0, 32, A.bits, dtype=torch.int64)
n_bins = float((1 << A.bits) - 1)
moved = 0
with torch.no_grad():
    for name, (s_, b_) in st["q"].items():
        k = plain_name(name)
        w = base[k].float().to(DEV)
        # the run computed its codes on the card; do the same, then settle any element that a
        # rounding tie put one step away from the measured value
        q, _, _ = q4.affine_params(w, A.group, A.bits)
        rows = w.shape[0]
        # the run rounded the trained parameters to fp16 before writing the measured directory
        s_ = s_.to(DEV).to(torch.float16).float().reshape(rows, -1)
        b_ = b_.to(DEV).to(torch.float16).float().reshape(rows, -1)
        tgt = sd[k].to(DEV)
        q = q.reshape(rows, -1, A.group)
        def err_of(qq):
            deq = (qq * s_[:, :, None] + b_[:, :, None]).reshape(w.shape)
            return (deq.to(tgt.dtype).float() - tgt.float()).abs().reshape(rows, -1, A.group)
        e0 = err_of(q)
        for d in (-1.0, 1.0):
            qd = torch.clamp(q + d, 0.0, n_bins)
            ed = err_of(qd)
            better = ed < e0
            moved += int(better.sum().item())
            q = torch.where(better, qd, q); e0 = torch.where(better, ed, e0)
        err = e0.max().item()
        assert err == 0.0, f"{k}: dequantized weight differs from the measured directory by {err}"
        c = q.to(torch.int64).reshape(rows, -1).cpu()
        packed = (c.reshape(rows, -1, 32 // A.bits) << shifts).sum(-1)
        s_ = s_.cpu(); b_ = b_.cpu()
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
print(f"[out] {A.out}: {len(mx)} tensors, {len(quant)} quantized, {moved} codes settled by a rounding tie")
print("PACKMLX_DONE")
