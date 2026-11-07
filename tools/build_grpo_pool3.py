"""The weak areas only (the user: train where the memory is clearly worse, leave what it already does for later).

  python -I build_grpo_pool3.py SRC_DIR POOL2.jsonl USED_IDS.txt OUT.jsonl USED.jsonl [USED2.jsonl ...]
On the 60 problems the memory trails the raw window on competition problems (-1.6 per 12), MATH level 4 (-1.7) and
level 5 (-1.2) - each time with more answers that never close - and is level on MATH level 3 and MMLU-Pro. This pool:
the competition and MATH 4-5 problems of POOL2 not drawn yet (USED_IDS: every problem id in the hub's group files), plus
fresh MATH level 4-5 problems from SRC_DIR (tools/build_grpo_prompts.py's parquet files) outside every USED file, POOL2
and MATH-500. Rows as ctl/dpo_pool.jsonl.
"""
import glob, json, os, random, re, sys

import pandas as pd

SRC, POOL2, USED_IDS, OUT, USED = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
SUF = "\nPlease reason step by step, and put your final answer within \\boxed{}."
N_FRESH = 700   # per level
rng = random.Random(3)
key = lambda s: re.sub(r"\s+", "", s or "")
used = set(open(USED_IDS).read().split())
ban = {key(p) for p in pd.read_parquet(os.path.join(SRC, "m500.parquet"))["problem"]}
seen = set()
for f in USED + [POOL2]:
    for l in open(f):
        r = json.loads(l); seen.add(r["id"])
        t = r.get("q") or r["prompt"].split("<|im_start|>user\n", 1)[-1].split("<|im_end|>", 1)[0]
        ban.add(key(t.rsplit("\nPlease reason step by step", 1)[0]))

rows = [json.loads(l) for l in open(POOL2)]
rows = [r for r in rows if not r["src"].startswith("mc") and r["src"] != "math-3" and r["id"] not in used]
n_left = len(rows)


def last_boxed(t):
    i = t.rfind("\\boxed{")
    if i < 0: return None
    j = i + 7; d = 1; out = []
    while j < len(t) and d:
        c = t[j]; d += (c == "{") - (c == "}")
        if d: out.append(c)
        j += 1
    return "".join(out) if not d else None


m = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(SRC, "math_*.parquet")))], ignore_index=True)
by_lv = {}
for i, r in m.iterrows():   # ids as in build_grpo_prompts.py
    g = last_boxed(r["solution"])
    if r["level"] in ("Level 4", "Level 5") and g and len(g) <= 40 and f"math{i}" not in seen and key(r["problem"]) not in ban:
        by_lv.setdefault(r["level"], []).append({"id": f"math{i}", "src": "math-" + r["level"][-1], "gold": g,
                                                 "prompt": "<|im_start|>user\n" + r["problem"] + SUF + "<|im_end|>\n<|im_start|>assistant\n"})
for lv in sorted(by_lv): rng.shuffle(by_lv[lv]); rows += by_lv[lv][:N_FRESH]
rng.shuffle(rows)
with open(OUT, "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
cnt = {}
for r in rows: cnt[r["src"]] = cnt.get(r["src"], 0) + 1
print(f"{len(rows)} rows ({n_left} left from pool 2, {len(rows) - n_left} fresh MATH)", cnt)
