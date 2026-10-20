#!/usr/bin/env python3
"""Chat ability kept or lost: bits per assistant character of each model on held-out conversations, by source file.

  python3 chat_bits.py --models npc4.pt,agent3/model_final.pt --sets soda_val.jsonl,help_val.jsonl,npc_val.jsonl --n 200
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import charlm_run as CR  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); ap.add_argument("--sets", required=True); ap.add_argument("--n", type=int, default=200)
A = ap.parse_args()
for mp in A.models.split(","):
    m = CR.Model(mp); line = []
    for sp in A.sets.split(","):
        b = n = 0
        for l in list(open(sp))[:A.n]:
            x, k = m.bits(json.loads(l)["messages"]); b += x; n += k
        line.append(f"{os.path.basename(sp).replace('.jsonl', '')} {b / max(n, 1):.3f}")
    print(f"[chat] {mp}: " + ", ".join(line) + " (bits per assistant character; lower is better)", flush=True)
