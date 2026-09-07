#!/usr/bin/env python3
"""usage: cnc.py teacher600.jsonl pooleval_all_c.jsonl pooleval_all_nc.jsonl
Compressed vs non-compressed vs teacher on the questions BOTH student runs have solved."""
import json, sys
def rows(p):
    out = {}
    for l in open(p):
        try: r = json.loads(l); out.setdefault(r["q"].strip(), r)
        except Exception: pass
    return out
T = {}
for l in open(sys.argv[1]):
    try: r = json.loads(l); T.setdefault(r["q"].strip(), []).append(r)
    except Exception: pass
C, N = rows(sys.argv[2]), rows(sys.argv[3])
Q = [q for q in C if q in N and q in T]
n = len(Q)
if not n: print("common 0"); sys.exit()
c = sum(bool(C[q]["correct"]) for q in Q); m = sum(bool(N[q]["correct"]) for q in Q)
t = sum(sum(x["correct"] for x in T[q]) / len(T[q]) for q in Q)
cw = sum(bool(C[q]["correct"]) and not N[q]["correct"] for q in Q)
nw = sum(bool(N[q]["correct"]) and not C[q]["correct"] for q in Q)
print(f"common {n} q")
print(f"  comp    {100*c/n:5.1f}%")
print(f"  nocomp  {100*m/n:5.1f}%  (comp-nocomp {100*(c-m)/n:+.1f})")
print(f"  teacher {100*t/n:5.1f}%")
print(f"  only-comp-right {cw}  only-nocomp-right {nw}")
