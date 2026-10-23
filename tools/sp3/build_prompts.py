"""Prompt pool for the big single-pass pooler training: real first turns (WildChat-1M, ODC-BY, English, non-toxic,
not redacted) + MATH train (levels 1-5) + GSM8K train, deduplicated, minus every prompt already used (the use-case bench
and pool, the 1200 traces of archive/kvpool/traces.jsonl - its first 60 are the held-out set).
  python build_prompts.py --used traces.jsonl uc_bench.jsonl uc_pool.jsonl --out prompts_big.jsonl
"""
import argparse, json, random, re
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--used", nargs="+", default=[])
ap.add_argument("--n-chat", type=int, default=8000); ap.add_argument("--n-math", type=int, default=3000); ap.add_argument("--n-gsm", type=int, default=2000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
key = lambda s: re.sub(r"\s+", " ", s.lower())[:200]
SUF = "\nPlease reason step by step, and put your final answer within \\boxed{}."
seen = set()
for f in a.used:
    for l in open(f):
        r = json.loads(l); q = r.get("q", "")
        seen.add(key(q)); seen.add(key(q.replace(SUF, "")))
out = []
n = 0
for r in load_dataset("allenai/WildChat-1M", split="train", streaming=True):
    if r["language"] != "English" or r["toxic"] or r["redacted"]: continue
    conv = r["conversation"]
    if not conv or conv[0]["role"] != "user": continue
    q = conv[0]["content"].strip()
    if not (20 <= len(q) <= 2000): continue
    k = key(q)
    if k in seen: continue
    seen.add(k); out.append(("chat", q)); n += 1
    if n >= a.n_chat: break
math = []
for sub in ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]:
    math += [r["problem"] for r in load_dataset("EleutherAI/hendrycks_math", sub, split="train")]
random.Random(1).shuffle(math)
m = 0
for q in math:
    if key(q) in seen: continue
    seen.add(key(q)); out.append(("math", q + SUF)); m += 1
    if m >= a.n_math: break
gsm = [r["question"] for r in load_dataset("openai/gsm8k", "main", split="train")]
random.Random(2).shuffle(gsm)
g = 0
for q in gsm:
    if key(q) in seen: continue
    seen.add(key(q)); out.append(("gsm8k", q + SUF)); g += 1
    if g >= a.n_gsm: break
random.Random(3).shuffle(out)
with open(a.out, "w") as f:
    for src, q in out: f.write(json.dumps({"src": src, "q": q}) + "\n")
print(f"[prompts] {len(out)}: chat {n}, math {m}, gsm8k {g} -> {a.out}")
