#!/usr/bin/env python3
"""Rebuild the dequantized model directory a jointfit run measures, from its state file and its base.

jointfit writes that directory last, after the state; when the write fails (the disk filled up after a
4000-step run) the training is not lost: the codes are the affine grid of the base weights (no clip search,
as the quant mode runs it) and the scales and biases are in the state. This writes the same numbers packmlx
expects to find there - the trained parameters rounded to fp16, as the run rounded them.

  python3 dequant_state.py --base /root/reeval_hf_g14m --state /root/sft/q14b.pt --out /root/sft_hf_q14b
"""
import argparse, os, shutil, sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q4  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--state", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--group", type=int, default=64); ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--codes-from", default="", help="a gptq.py state: the run started from these codes, not from rounding the base")
A = ap.parse_args()


def plain_name(n):
    return n.replace("base_model.model.", "").replace(".base_layer", "") + ".weight"


DEV = "cuda" if torch.cuda.is_available() else "cpu"
sd = load_file(os.path.join(A.base, "model.safetensors"))
st = torch.load(A.state, map_location="cpu")
CODES = torch.load(A.codes_from, map_location="cpu")["q"] if A.codes_from else None
print(f"[dequant] state from step {st.get('step')} val {st.get('val'):.4f}: {len(st['q'])} quantized tensors; codes {'from ' + A.codes_from if CODES else 'by rounding the base'}", flush=True)
with torch.no_grad():
    for name, (s_, b_) in st["q"].items():
        k = plain_name(name)
        assert k in sd, f"{k} is not in {A.base}"
        w = sd[k].float().to(DEV)
        rows = w.shape[0]
        if CODES is not None and k in CODES:
            q = CODES[k][0].reshape(rows, -1, A.group).float().to(DEV)
        else:
            q, _, _ = q4.affine_params(w, A.group, A.bits)
        s_ = s_.to(DEV).to(torch.float16).float().reshape(rows, -1)
        b_ = b_.to(DEV).to(torch.float16).float().reshape(rows, -1)
        deq = (q.reshape(rows, -1, A.group) * s_[:, :, None] + b_[:, :, None]).reshape(w.shape)
        sd[k] = deq.to(sd[k].dtype).cpu().contiguous()
os.makedirs(A.out, exist_ok=True)
save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(A.out, "model.safetensors"), metadata={"format": "pt"})
for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
    if os.path.exists(os.path.join(A.base, f)):
        shutil.copy(os.path.join(A.base, f), os.path.join(A.out, f))
print(f"[out] {A.out}: {len(sd)} tensors, {len(st['q'])} replaced", flush=True)
print("JOINTFIT_DONE (rebuilt from the state)", flush=True)
