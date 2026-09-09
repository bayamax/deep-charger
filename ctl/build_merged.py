#!/usr/bin/env python3
"""Fold the SFT checkpoint (full weights + rank-128 LoRA, peft-named) into a plain HF model.

Needed before changing the LoRA rank/layers: peft renames a LoRA'd module's weight to
"...base_layer.weight", so a checkpoint saved under rank-128-on-all-layers cannot be loaded into a
model whose LoRA covers different layers -- the untouched layers would silently fall back to the
stock base model. Merging removes the LoRA wrapper entirely, so a fresh LoRA of any shape can be
mounted on top of exactly the SFT weights.

  out: <outdir>/            HF model dir (merged weights + tokenizer)
       <pooler_out>         pooler tensors only, for --pooler-init
usage: build_merged.py <sft.safetensors> <outdir> <pooler_out.safetensors>
"""
import os, sys
SFT, OUT, POUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ.setdefault("SP_BASE", "/root/fft_hf")
os.environ["SP_RANK"] = "128"                    # the shape the SFT checkpoint was saved under
os.environ["SP_NOSYS"] = "1"; os.environ["SP_EPISODIC"] = "1"; os.environ["SP_TRAIN_POOLER"] = "0"
import torch
from safetensors.torch import load_file, save_file
F = "/root/work/grpo_e2e_torch.py"
src = open(F).read().split("\n")
cut = next(i for i, l in enumerate(src) if l.startswith("if MULTI > 1:"))
sys.argv = ["grpo_e2e_torch.py", "0", "6", "0", "1500"]
sys.path.insert(0, "/root/work")
ns = {"__name__": "build_merged", "__file__": F}
exec(compile("\n".join(src[:cut]), F, "exec"), ns)
model, tok = ns["model"], ns["tok"]
sd = load_file(SFT)
md = {k: v for k, v in sd.items() if not k.startswith("pooler.")}
pl = {k[len("pooler."):]: v for k, v in sd.items() if k.startswith("pooler.")}
r = model.load_state_dict(md, strict=False)
print(f"[merge] {len(md)} model tensors, unexpected {len(r.unexpected_keys)}, missing {len(r.missing_keys)}", flush=True)
if r.unexpected_keys:
    print("  sample unexpected:", r.unexpected_keys[:3], flush=True)
assert len(r.unexpected_keys) == 0, "the SFT checkpoint does not match the rank-128 wrapping"
merged = model.merge_and_unload()
merged.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
save_file({k: v.contiguous() for k, v in pl.items()}, POUT)
print(f"[merge] wrote {OUT} ({len(list(merged.parameters()))} params) and {POUT} ({len(pl)} pooler tensors)", flush=True)
print("MERGE_DONE", flush=True)
