#!/usr/bin/env python3
"""SODA (allenai/soda, CC BY 4.0: everyday social conversations, each with a short narrative of who and what) as
{"messages": [...]} for charlm_sft.py: the second speaker is the character, the first the player; the narrative becomes
the card ("You are <name>. <narrative> You are talking with <name>.") in most, none in the rest (plain chat).

  python3 soda_prep.py --parquet 0.parquet --n 40000 --val 300 --out soda.jsonl --val-out soda_val.jsonl
"""
import argparse, json, random, re

import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", required=True); ap.add_argument("--n", type=int, default=40000); ap.add_argument("--val", type=int, default=300)
ap.add_argument("--out", required=True); ap.add_argument("--val-out", required=True); ap.add_argument("--no-card", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
R = random.Random(A.seed)
AI = re.compile(r"\b(as an ai|language model|virtual assistant|ai assistant)\b", re.I)


def conv(row):
    d, sp = row["dialogue"], row["speakers"]
    if len(d) < 4 or len(sp) < len(d): return None
    a, b = sp[0], sp[1]
    if a == b or any(sp[i] != (a if i % 2 == 0 else b) for i in range(len(d))): return None   # two speakers taking turns
    if any(AI.search(t) or len(t) > 500 for t in d): return None
    msgs = []
    if R.random() >= A.no_card:
        msgs.append({"role": "system", "content": f"You are {b}. {row['narrative'].strip()} You are talking with {a}."})
    for i, t in enumerate(d): msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": t.strip()})
    if msgs[-1]["role"] == "user": msgs.pop()
    return {"messages": msgs, "src": "soda"}


t = pq.read_table(A.parquet, columns=["dialogue", "speakers", "narrative"]).to_pylist(); R.shuffle(t)
rows = []
for r in t:
    c = conv(r)
    if c: rows.append(c)
    if len(rows) >= A.n + A.val: break
with open(A.val_out, "w") as f:
    for c in rows[:A.val]: f.write(json.dumps(c) + "\n")
with open(A.out, "w") as f:
    for c in rows[A.val:]: f.write(json.dumps(c) + "\n")
print(f"[soda] {len(t)} rows read; {len(rows) - A.val} conversations -> {A.out}, {min(A.val, len(rows))} held out -> {A.val_out}", flush=True)
