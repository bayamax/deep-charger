#!/usr/bin/env python3
"""Paired comparison on the same held-out questions: teacher (ckpt_mus200, no compression, 2 rolls/q)
vs a pool_eval output (compressed). usage: paired.py teacher600.jsonl pooleval_X.jsonl [label]"""
import json, sys
def rows(p):
    out = []
    for l in open(p):
        try: out.append(json.loads(l))
        except Exception: pass
    return out
T = {}
for r in rows(sys.argv[1]):
    T.setdefault(r["q"], []).append(r)
S = rows(sys.argv[2]); lab = sys.argv[3] if len(sys.argv) > 3 else "post"
S = [r for r in S if r["q"] in T]
n = len(S)
if n == 0:
    print(f"[paired] no rows yet"); sys.exit()
tc = sum(sum(x["correct"] for x in T[r["q"]]) / len(T[r["q"]]) for r in S)
tg = sum(sum(x["grounded"] for x in T[r["q"]]) / len(T[r["q"]]) for r in S)
ts = sum(sum(x["ns"] for x in T[r["q"]]) / len(T[r["q"]]) for r in S)
sc = sum(bool(r["correct"]) for r in S); sg = sum(bool(r["grounded"]) for r in S); ss = sum(r.get("ns", 0) for r in S)
both = sum(1 for r in S if r["correct"] and all(x["correct"] for x in T[r["q"]]))
only_s = sum(1 for r in S if r["correct"] and not any(x["correct"] for x in T[r["q"]]))
only_t = sum(1 for r in S if not r["correct"] and all(x["correct"] for x in T[r["q"]]))
print(f"[paired n={n}] teacher correct={100*tc/n:.1f}% grounded={100*tg/n:.1f}% srch={ts/n:.1f} | "
      f"{lab} correct={100*sc/n:.1f}% grounded={100*sg/n:.1f}% srch={ss/n:.1f} | diff={100*(sc-tc)/n:+.1f}pt")
print(f"[paired n={n}] both-right={both} {lab}-only={only_s} teacher-only(2/2)={only_t}")
