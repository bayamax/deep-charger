#!/usr/bin/env python3
"""The reward model as its own process (2026-10-14): the GRPO loop is pinned to an older transformers than the
Skywork-Reward-V2 (Qwen3) models need, so the reward model runs here, under its own library path. One JSON line in
({"convs": [[{role, content}, ...], ...]}), one JSON line out ({"scores": [...]}); "READY" once loaded.
  PYTHONPATH=/root/rmenv python3 rm_server.py <model> <device>
"""
import json, sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
name, dev = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "cuda")
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.bfloat16 if dev.startswith("cuda") else torch.float32, num_labels=1).to(dev).eval()
print("READY", flush=True)
for line in sys.stdin:
    try:
        convs = json.loads(line)["convs"]
        texts = [tok.apply_chat_template(c, tokenize=False) for c in convs]
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 6):
                enc = tok(texts[i:i + 6], return_tensors="pt", padding=True, truncation=True, max_length=2048, add_special_tokens=False).to(dev)
                out += model(**enc).logits[:, 0].float().cpu().tolist()
        if dev.startswith("cuda"): torch.cuda.empty_cache()
        print(json.dumps({"scores": out}), flush=True)
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {str(e)[:200]}"}), flush=True)
