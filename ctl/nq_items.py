#!/usr/bin/env python3
"""Fresh search questions for GRPO from Natural Questions (nq_open, the train split: real search queries, short answers).

The user (2026-10-07): train on questions the model has never seen or has barely learned, or has not solved. The home-made
set (selfq_all, 1905) is nearly used up; nq_open has ~88k. Kept: questions whose shortest answer is 1-4 words (the reward
is the gold string in the reply), not already in any evaluation set or earlier training set (--exclude, by normalised
question text), shuffled with a fixed seed; the first --n written as GRPO items {q, gold, hist: []}, and every kept
question with its answer to --pool for later rounds (so the next round continues where this one stopped).

  python3 nq_items.py --n 600 --out /root/work/nq_items_1.jsonl --pool /root/work/nq_pool.jsonl --exclude /root/work/eval300.jsonl ...
"""
import argparse, json, os, random, re
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=600); ap.add_argument("--out", required=True); ap.add_argument("--pool", default="")
ap.add_argument("--skip", type=int, default=0, help="skip this many of the shuffled kept questions first (the next round takes the next block)")
ap.add_argument("--exclude", nargs="*", default=[]); ap.add_argument("--seed", type=int, default=11)
A = ap.parse_args()
from huggingface_hub import hf_hub_download  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


def norm(s): return " ".join(re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split())


seen = set()
for f in A.exclude:
    if not os.path.exists(f): continue
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        for q in [r.get("q")] + [t.get("q") for t in (r.get("turns") or [])]:
            if q: seen.add(norm(q))
p = hf_hub_download("google-research-datasets/nq_open", "nq_open/train-00000-of-00001.parquet", repo_type="dataset")
t = pq.read_table(p).to_pydict()
kept, dup, long_ = [], 0, 0
for q, ans in zip(t["question"], t["answer"]):
    q = (q or "").strip()
    if not q or norm(q) in seen: dup += 1; continue
    cands = sorted({a.strip() for a in (ans or []) if a and 1 <= len(a.split()) <= 4 and len(a.strip()) >= 2}, key=len)
    if not cands: long_ += 1; continue
    gold = cands[0]
    if re.fullmatch(r"\d{1,2}", gold): long_ += 1; continue        # a one- or two-digit number appears everywhere
    qt = q[0].upper() + q[1:] + ("" if q.endswith("?") else "?")
    kept.append({"q": qt, "gold": gold, "hist": [], "answers": cands})
random.Random(A.seed).shuffle(kept)
if A.pool: open(A.pool, "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
sel = kept[A.skip:A.skip + A.n]
open(A.out, "w").write("".join(json.dumps({k: r[k] for k in ("q", "gold", "hist")}, ensure_ascii=False) + "\n" for r in sel))
print(f"NQ_ITEMS_DONE {len(sel)} items (of {len(kept)} kept; {dup} already in an evaluation or training set, {long_} without a short answer) -> {A.out}", flush=True)
