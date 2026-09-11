#!/usr/bin/env python3
"""Fold the SFT checkpoint (full weights + rank-128 LoRA, peft-named) into a plain HF model.

Needed before changing the LoRA rank/layers: peft renames a LoRA'd module's weight to
"...base_layer.weight", so a checkpoint saved under rank-128-on-all-layers cannot be loaded into a
model whose LoRA covers different layers -- the untouched layers would silently fall back to the
stock base model. Merging removes the LoRA wrapper entirely, so a fresh LoRA of any shape can be
mounted on top of exactly the SFT weights.

  out: <outdir>/            HF model dir (merged weights + tokenizer)
       <pooler_out>         pooler tensors only, for --pooler-init
usage: build_merged.py <ckpt.safetensors> <outdir> <pooler_out.safetensors> [rank] [layers]
       rank/layers default to the SFT shape (128, all layers); a GRPO checkpoint from the v6 run
       needs "16 20-27", the shape it was trained under.
"""
import os, sys
SFT, OUT, POUT = sys.argv[1], sys.argv[2], sys.argv[3]
RANK = sys.argv[4] if len(sys.argv) > 4 else "128"
LAYERS = sys.argv[5] if len(sys.argv) > 5 else "all"
os.environ.setdefault("SP_BASE", "/root/fft_hf")
os.environ["SP_RANK"] = RANK                     # must be the shape the checkpoint was saved under
os.environ["SP_NOSYS"] = "1"; os.environ["SP_EPISODIC"] = "1"; os.environ["SP_TRAIN_POOLER"] = "0"
import torch
from safetensors.torch import load_file, save_file
F = "/root/work/grpo_e2e_torch.py"
src = open(F).read().split("\n")
cut = next(i for i, l in enumerate(src) if l.startswith("if MULTI > 1:"))
sys.argv = ["grpo_e2e_torch.py", "0", "6", "0", "1500"]
sys.path.insert(0, "/root/work")
_L = None if LAYERS == "all" else list(range(int(LAYERS.split("-")[0]), int(LAYERS.split("-")[1]) + 1))
ns = {"__name__": "build_merged", "__file__": F, "_SP_LAYERS": _L}
prefix = "\n".join(src[:cut]).replace(
    'target_modules=TARGETS, bias="none", task_type="CAUSAL_LM")',
    'target_modules=TARGETS, bias="none", task_type="CAUSAL_LM", layers_to_transform=_SP_LAYERS)', 1)
exec(compile(prefix, F, "exec"), ns)
model, tok = ns["model"], ns["tok"]
sd = load_file(SFT)
md = {k: v for k, v in sd.items() if not k.startswith("pooler.")}
pl = {k[len("pooler."):]: v for k, v in sd.items() if k.startswith("pooler.")}
r = model.load_state_dict(md, strict=False)
print(f"[merge] rank={RANK} layers={LAYERS}: {len(md)} model tensors, unexpected {len(r.unexpected_keys)}, missing {len(r.missing_keys)}", flush=True)
if r.unexpected_keys:
    print("  sample unexpected:", r.unexpected_keys[:3], flush=True)
assert len(r.unexpected_keys) == 0, f"the checkpoint does not match rank={RANK} layers={LAYERS}"
merged = model.merge_and_unload()
merged.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
save_file({k: v.contiguous() for k, v in pl.items()}, POUT)
print(f"[merge] wrote {OUT} ({len(list(merged.parameters()))} params) and {POUT} ({len(pl)} pooler tensors)", flush=True)
print("MERGE_DONE", flush=True)
