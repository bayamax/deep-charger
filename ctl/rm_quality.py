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

def load_rm(spec, dev):
    """<hub name>[@8bit|@4bit]: a quantised reward model (bitsandbytes) when asked, bf16 on the card otherwise"""
    name, _, q = spec.partition("@")
    kw = {"num_labels": 1}
    if q in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True) if q == "8bit" else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}; kw["torch_dtype"] = torch.bfloat16
        m = AutoModelForSequenceClassification.from_pretrained(name, **kw)
    else:
        m = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.bfloat16 if str(dev).startswith("cuda") else torch.float32, **kw).to(dev)
    return AutoTokenizer.from_pretrained(name), m.eval()

tok, model = load_rm(name, dev)
rows = [json.loads(l) for l in open(path) if l.strip()]
for r in rows: r["reply"] = r.get("reply") or r.get("answer") or ""   # the single-turn screens call it "answer"
rows = [r for r in rows if r["reply"].strip()]
convs = []
for r in rows:
    h = [{"role": m["role"], "content": m["content"]} for m in (r.get("hist") or []) if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    q = r.get("ran") or r.get("q") or ""
    convs.append(h + [{"role": "user", "content": q}, {"role": "assistant", "content": r["reply"]}])
texts = [tok.apply_chat_template(c, tokenize=False) for c in convs]
sc = []
with torch.no_grad():
    for i in range(0, len(texts), 8):
        enc = tok(texts[i:i + 8], return_tensors="pt", padding=True, truncation=True, max_length=2048, add_special_tokens=False).to(model.device)
        sc += model(**enc).logits[:, 0].float().cpu().tolist()
cor = [s for s, r in zip(sc, rows) if r.get("correct")]; lc = [len(r["reply"]) for r in rows if r.get("correct")]
m = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
print(f"RMQ {tag} n {len(rows)} correct {len(cor)} rm_mean_correct {m(cor):.3f} rm_mean_all {m(sc):.3f} len_mean_correct {m(lc):.0f}", flush=True)
