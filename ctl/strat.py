#!/usr/bin/env python3
"""usage: strat.py teacher600.jsonl pooleval_all_c.jsonl pooleval_all_nc.jsonl [RW]
Compressed vs non-compressed vs teacher, restricted to rows where compression actually engaged
(trajectory longer than RW), plus length buckets."""
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/root/fft_hf")
RW = int(sys.argv[4]) if len(sys.argv) > 4 else 768
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
L = lambda r: len(tok.encode(r["text"], add_special_tokens=False))
lc = {q: L(C[q]) for q in Q}; ln = {q: L(N[q]) for q in Q}
lt = {q: sum(L(x) for x in T[q]) / len(T[q]) for q in Q}
def rep(name, qs):
    n = len(qs)
    if not n: print(f"{name}: 0"); return
    c = sum(bool(C[q]["correct"]) for q in qs); m = sum(bool(N[q]["correct"]) for q in qs)
    t = sum(sum(x["correct"] for x in T[q]) / len(T[q]) for q in qs)
    oc = sum(bool(C[q]["correct"]) and not N[q]["correct"] for q in qs); on = sum(bool(N[q]["correct"]) and not C[q]["correct"] for q in qs)
    print(f"{name:<34} n={n:3d}  comp {100*c/n:5.1f}  nocomp {100*m/n:5.1f}  teacher {100*t/n:5.1f}  | only-comp {oc} only-nocomp {on}")
print(f"=== STRAT common={len(Q)} RW={RW}")
rep("all", Q)
rep(f"comp traj > {RW} (engaged)", [q for q in Q if lc[q] > RW])
rep(f"comp traj <= {RW} (not engaged)", [q for q in Q if lc[q] <= RW])
rep(f"comp traj > {RW+512}", [q for q in Q if lc[q] > RW + 512])
rep(f"either student traj > {RW}", [q for q in Q if lc[q] > RW or ln[q] > RW])
rep(f"teacher mean traj > {RW}", [q for q in Q if lt[q] > RW])
rep(f"teacher mean traj <= {RW}", [q for q in Q if lt[q] <= RW])
eng = [q for q in Q if lc[q] > RW]
print(f"engaged rows: pooled tokens mean {sum(lc[q]-RW for q in eng)/max(1,len(eng)):.0f}, searches mean {sum(C[q]['ns'] for q in eng)/max(1,len(eng)):.2f} vs nocomp on same q {sum(N[q]['ns'] for q in eng)/max(1,len(eng)):.2f}")
print("=== STRAT END")
