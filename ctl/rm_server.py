#!/usr/bin/env python3
"""The reward model as its own process (2026-10-14): the GRPO loop is pinned to an older transformers than the
Skywork-Reward-V2 (Qwen3) models need, so the reward model runs here, under its own library path. One JSON line in
({"convs": [[{role, content}, ...], ...]}), one JSON line out ({"scores": [...]}); "READY" once loaded.
  PYTHONPATH=/root/rmenv python3 rm_server.py <model> <device>
"""
import json, sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from rm_load import load_rm
name, dev = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "cuda")

MODELS = [load_rm(n_, dev) for n_ in name.split("+")]   # an ensemble: every member scores, the loop combines
print("READY", flush=True)
for line in sys.stdin:
    try:
        convs = json.loads(line)["convs"]
        cols = []
        for tok, model in MODELS:
            texts = [tok.apply_chat_template(c, tokenize=False) for c in convs]; col = []
            with torch.no_grad():
                for i in range(0, len(texts), 6):
                    enc = tok(texts[i:i + 6], return_tensors="pt", padding=True, truncation=True, max_length=2048, add_special_tokens=False).to(model.device)
                    col += model(**enc).logits[:, 0].float().cpu().tolist()
            cols.append(col)
        out = cols[0] if len(cols) == 1 else [list(x) for x in zip(*cols)]
        if dev.startswith("cuda"): torch.cuda.empty_cache()
        print(json.dumps({"scores": out}), flush=True)
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {str(e)[:200]}"}), flush=True)
