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
cat = {"A": [], "B": [], "C": [], "amb": [], "Q": [], "Z": []}
nocorr = 0
for r in R:
    if not r.get("correct"): nocorr += 1; continue
    text, gold, q = r.get("text") or "", r.get("gold") or "", r.get("q") or ""
    if "</think>" not in text: continue
    rx = gold_re(gold)
    if rx is None: continue
    if rx.search(q):
        cat["Q"].append((r, "gold is in the question")); continue
    think = text.split("</think>")[0]; ans_tok = ntok(think)
    info_spans = [(m.start(), m.end()) for m in re.finditer(r"<information>(.*?)</information>", think, re.S)]
    occ = sorted((ntok(think[:m.end()]), "served" if any(a <= m.start() < b for a, b in info_spans) else "own") for m in rx.finditer(think))
    if not any(k == "served" for _, k in occ):
        cat["Z"].append((r, "never served")); continue
    # walk the occurrences: an OWN occurrence is supported if some earlier occurrence (any kind) lies within RW tokens before it;
    # an unsupported own occurrence (gap > RW+C) means the model produced the gold while it existed only in the pooled region
    first_served = next(p for p, k in occ if k == "served")
    gaps = [(p - max([pp for pp, _ in occ if pp < p] or [-10**9]), p) for p, k in occ if k == "own" and p > first_served]
    worst = max([g for g, _ in gaps] or [0])
    d_last = ans_tok - occ[-1][0]; d_served = ans_tok - max(p for p, k in occ if k == "served")
    desc = f"served@{[p for p,k in occ if k=='served'][:4]} own@{[p for p,k in occ if k=='own'][:6]} answer@{ans_tok} d_served={d_served} d_last={d_last} worst_gap={worst}"
    if worst > RW + C or d_last > RW + C: cat["C"].append((r, desc))
    elif worst > RW or d_last > RW: cat["amb"].append((r, desc))
    elif d_served > RW: cat["B"].append((r, desc))
    else: cat["A"].append((r, desc))
nc = len(R) - nocorr
print(f"=== GOLD_POOLED {LAB} rows={len(R)} correct={nc} RW={RW}")
print(f"  A gold-in-window {len(cat['A'])}   B self-carried(chain) {len(cat['B'])}   C recalled-from-pooled {len(cat['C'])}   ambiguous {len(cat['amb'])}   Q gold-in-question {len(cat['Q'])}   Z never-served {len(cat['Z'])}")
for k in ("C", "amb", "B"):
    for r, d in cat[k][:6]:
        print(f"  [{k}] {d} q={r['q'][:60]!r} gold={r['gold']!r}")
print("=== GOLD_POOLED END")
