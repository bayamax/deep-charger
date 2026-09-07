#!/usr/bin/env python3
"""usage: gold_pooled.py rows.jsonl [RW=768] [label]
Did the model answer correctly while the gold sat ONLY in the pooled region (outside the raw window)?
For every row (pool_eval rows or GRPO rollouts: text/gold/correct) locate each gold occurrence before </think>,
inside <information> blocks (served) and outside them (the model's own think), and measure the token distance
from the occurrence to the answer position.
  A  gold-in-window     : a served gold occurrence within the last RW tokens before the answer
  B  self-carried       : served gold is beyond RW+C (pooled), but the model re-wrote the gold in its own think within RW
  C  pooled-only        : no gold occurrence of any kind within the last RW+C tokens -> the fact had to come through the pooler (or memory)
  Z  never served       : gold was never shown; a correct answer here is parametric memory / luck
Rows between RW and RW+C are ambiguous (the raw window holds between RW and RW+C tokens) and reported separately."""
import json, sys, re
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/root/fft_hf")
P = sys.argv[1]; RW = int(sys.argv[2]) if len(sys.argv) > 2 else 768; LAB = sys.argv[3] if len(sys.argv) > 3 else P.split("/")[-1]
C = 128
def ntok(s): return len(tok.encode(s, add_special_tokens=False)) if s else 0
def gold_re(g):
    parts = [re.escape(w) for w in re.sub(r"[^A-Za-z0-9 ]", " ", g).split()]
    return re.compile(r"[^A-Za-z0-9]*".join(parts), re.I) if parts else None
rows = []
for l in open(P):
    try: rows.append(json.loads(l))
    except Exception: pass
seen = set(); R = []
for r in rows:
    k = (r.get("q"), r.get("text")[:200] if r.get("text") else None)
    if k in seen: continue
    seen.add(k); R.append(r)
cat = {"A": [], "B": [], "C": [], "amb": [], "Z": []}
nocorr = 0
for r in R:
    if not r.get("correct"): nocorr += 1; continue
    text, gold = r.get("text") or "", r.get("gold") or ""
    if "</think>" not in text: continue
    think = text.split("</think>")[0]; ans_tok = ntok(think)
    rx = gold_re(gold)
    if rx is None: continue
    info_spans = [(m.start(), m.end()) for m in re.finditer(r"<information>(.*?)</information>", think, re.S)]
    served = [m for m in rx.finditer(think) if any(a <= m.start() < b for a, b in info_spans)]
    own = [m for m in rx.finditer(think) if not any(a <= m.start() < b for a, b in info_spans)]
    if not served:
        cat["Z"].append((r, None, None)); continue
    d_info = min(ans_tok - ntok(think[:m.end()]) for m in served)
    d_any = min([d_info] + [ans_tok - ntok(think[:m.end()]) for m in own])
    if d_info <= RW: cat["A"].append((r, d_info, d_any))
    elif d_any <= RW: cat["B"].append((r, d_info, d_any))
    elif d_any > RW + C: cat["C"].append((r, d_info, d_any))
    else: cat["amb"].append((r, d_info, d_any))
nc = len(R) - nocorr
print(f"=== GOLD_POOLED {LAB} rows={len(R)} correct={nc} RW={RW}")
print(f"  A gold-in-window {len(cat['A'])}   B self-carried {len(cat['B'])}   C pooled-only {len(cat['C'])}   ambiguous {len(cat['amb'])}   Z never-served {len(cat['Z'])}")
for k in ("C", "B"):
    for r, di, da in cat[k][:6]:
        print(f"  [{k}] d_info={di} d_any={da} q={r['q'][:70]!r} gold={r['gold']!r} ans={r.get('answer','')[:50]!r}")
print("=== GOLD_POOLED END")
