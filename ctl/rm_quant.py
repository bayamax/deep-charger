#!/usr/bin/env python3
"""Save a reward model quantised to 4 bits under /root/rm_q and drop its bf16 download from the hub cache (2026-10-15:
two 8B judges' downloads filled the box's 60 GB disk).  PYTHONPATH=/root/rmenv python3 rm_quant.py <hub name>"""
import os, shutil, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rm_load import load_rm, local_dir
from transformers import AutoTokenizer
name = sys.argv[1]; d = local_dir(name)
if not os.path.exists(os.path.join(d, "config.json")):
    tok, m = load_rm(name + "@4bit")
    probe = tok.apply_chat_template([{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4."}], tokenize=False)
    with torch.no_grad(): a = float(m(**tok(probe, return_tensors="pt", add_special_tokens=False).to(m.device)).logits[0, 0])
    os.makedirs(d, exist_ok=True); m.save_pretrained(d); tok.save_pretrained(d); del m; torch.cuda.empty_cache()
    tok2, m2 = load_rm(name + "@4bit")
    with torch.no_grad(): b = float(m2(**tok2(probe, return_tensors="pt", add_special_tokens=False).to(m2.device)).logits[0, 0])
    print(f"[rm_quant] {name}: saved 4-bit to {d}; probe {a:.3f} -> reloaded {b:.3f}", flush=True)
    if abs(a - b) > 0.05: shutil.rmtree(d); raise SystemExit(f"[rm_quant] {name}: the reloaded model scores differently - kept the download")
cache = os.path.expanduser("~/.cache/huggingface/hub/models--" + name.replace("/", "--"))
if os.path.exists(cache): shutil.rmtree(cache); print(f"[rm_quant] {name}: bf16 download removed ({cache})", flush=True)
print("RM_QUANT_DONE", name, flush=True)
