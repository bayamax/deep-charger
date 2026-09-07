#!/usr/bin/env python3
"""usage: paired.py teacher600.jsonl pooleval_X.jsonl label -> 3 short lines: n solved, student acc, teacher acc on the same questions."""
import json, sys
def rows(p):
    out = []
    for l in open(p):
        try: out.append(json.loads(l))
        except Exception: pass
    return out
T = {}
for r in rows(sys.argv[1]):
    T.setdefault(r["q"].strip(), []).append(r)
lab = sys.argv[3] if len(sys.argv) > 3 else "post"
S = [r for r in rows(sys.argv[2]) if r["q"].strip() in T]
n = len(S)
if n == 0:
    print(f"{lab}: 0/300"); sys.exit()
sc = sum(bool(r["correct"]) for r in S)
tc = sum(sum(x["correct"] for x in T[r["q"].strip()]) / len(T[r["q"].strip()]) for r in S)
print(f"{lab} {n}/300")
print(f"  {lab:<8}{100*sc/n:5.1f}%")
print(f"  teacher {100*tc/n:5.1f}%  ({100*(sc-tc)/n:+.1f})")
