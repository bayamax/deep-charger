#!/usr/bin/env python3
"""Reply quality by the reward model, for a screen's output (pool_eval jsonl: q, reply, correct; multi-turn rows carry
their history in "hist" when present). Run under the reward model's library path. Prints one line:
  RMQ <tag> n <rows> correct <k> rm_mean_correct <x> rm_mean_all <y> len_mean_correct <z>
  PYTHONPATH=/root/rmenv python3 rm_quality.py <model> <screen.jsonl> <tag>
"""
import json, sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
name, path, tag = sys.argv[1], sys.argv[2], sys.argv[3]
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32, num_labels=1).to(dev).eval()
rows = [json.loads(l) for l in open(path) if l.strip()]
rows = [r for r in rows if (r.get("reply") or "").strip()]
convs = []
for r in rows:
    h = [{"role": m["role"], "content": m["content"]} for m in (r.get("hist") or []) if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    q = r.get("ran") or r.get("q") or ""
    convs.append(h + [{"role": "user", "content": q}, {"role": "assistant", "content": r["reply"]}])
texts = [tok.apply_chat_template(c, tokenize=False) for c in convs]
sc = []
with torch.no_grad():
    for i in range(0, len(texts), 8):
        enc = tok(texts[i:i + 8], return_tensors="pt", padding=True, truncation=True, max_length=2048, add_special_tokens=False).to(dev)
        sc += model(**enc).logits[:, 0].float().cpu().tolist()
cor = [s for s, r in zip(sc, rows) if r.get("correct")]; lc = [len(r["reply"]) for r in rows if r.get("correct")]
m = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
print(f"RMQ {tag} n {len(rows)} correct {len(cor)} rm_mean_correct {m(cor):.3f} rm_mean_all {m(sc):.3f} len_mean_correct {m(lc):.0f}", flush=True)
