"""Fresh problems for the GRPO rollout workers (the user: problems once each, rather than a second pass over the same pool).

  python -I build_grpo_pool2.py SRC_DIR USED.jsonl [USED2.jsonl ...] OUT.jsonl
SRC_DIR: the parquet files of tools/build_grpo_prompts.py. USED: every problem file already drawn from (ctl/grpo_prompts.jsonl,
ctl/rl_eval120.jsonl) - their ids and texts are left out, as is MATH-500.
The mix follows what gave informative groups (some draws right, some wrong) in GRPO run 2: MATH level 3 was mostly all right
(12% mixed) and is left out; levels 4-5, the competition problems (DeepScaleR, short answers, the shorter half of the
reference solutions) and MMLU-Pro (every category equally) were 26-29% mixed.
Rows as ctl/dpo_pool.jsonl: {"id", "src", "gold", "prompt"} with the chat template rendered.
"""
import glob, json, os, random, re, sys

import pandas as pd

SRC, USED, OUT = sys.argv[1], sys.argv[2:-1], sys.argv[-1]
SUF = "\nPlease reason step by step, and put your final answer within \\boxed{}."
SUF_MC = "\nPlease reason step by step, and put the letter of your final answer within \\boxed{}."
N_MATH, N_DSR, N_MC = 1200, 1800, 1500
rng = random.Random(1)
key = lambda s: re.sub(r"\s+", "", s or "")
ban = {key(p) for p in pd.read_parquet(os.path.join(SRC, "m500.parquet"))["problem"]}
used_ids = set()
for f in USED:
    for l in open(f):
        r = json.loads(l); used_ids.add(r["id"])
        t = r.get("q") or r["prompt"].split("<|im_start|>user\n", 1)[-1].split("<|im_end|>", 1)[0]
        ban.add(key(t.rsplit("\nPlease reason step by step", 1)[0]))
ok = lambda i, q: i not in used_ids and key(q) not in ban


def last_boxed(t):
    i = t.rfind("\\boxed{")
    if i < 0: return None
    j = i + 7; d = 1; out = []
    while j < len(t) and d:
        c = t[j]; d += (c == "{") - (c == "}")
        if d: out.append(c)
        j += 1
    return "".join(out) if not d else None


rows = []
m = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(SRC, "math_*.parquet")))], ignore_index=True)
by_lv = {}
for i, r in m.iterrows():   # ids as in build_grpo_prompts.py (the row index of the same concatenation)
    g = last_boxed(r["solution"])
    if r["level"] in ("Level 4", "Level 5") and g and len(g) <= 40 and ok(f"math{i}", r["problem"]):
        by_lv.setdefault(r["level"], []).append({"src": "math-" + r["level"][-1], "id": f"math{i}", "gold": g, "q": r["problem"] + SUF})
for lv in sorted(by_lv): rng.shuffle(by_lv[lv]); rows += by_lv[lv][:N_MATH // 2]
d = pd.read_parquet(os.path.join(SRC, "dsr.parquet"))
d = d[d["answer"].str.len().between(1, 20)]
d = d[d["solution"].str.len() <= d["solution"].str.len().median()]
d = d[[ok(f"dsr{i}", p) for i, p in zip(d.index, d["problem"])]].drop_duplicates("problem")
for i, r in d.sample(N_DSR, random_state=1).iterrows():
    rows.append({"src": "comp", "id": f"dsr{i}", "gold": r["answer"], "q": r["problem"] + SUF})
p = pd.read_parquet(os.path.join(SRC, "mmlupro.parquet"))
p = p[[ok(f"mmlupro{i}", q) for i, q in zip(p["question_id"], p["question"])]]
cats = sorted(p["category"].unique()); per = N_MC // len(cats)
for c in cats:
    pc = p[p["category"] == c]
    for _, r in pc.sample(min(per, len(pc)), random_state=1).iterrows():
        opts = "\n".join(f"{chr(65 + k)}. {o}" for k, o in enumerate(r["options"]))
        rows.append({"src": "mc-" + c.replace(" ", "_"), "id": f"mmlupro{r['question_id']}", "gold": r["answer"],
                     "q": r["question"] + "\n\nOptions:\n" + opts + SUF_MC})
rng.shuffle(rows)
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps({"id": r["id"], "src": r["src"], "gold": r["gold"],
                            "prompt": "<|im_start|>user\n" + r["q"] + "<|im_end|>\n<|im_start|>assistant\n"}) + "\n")
cnt = {}
for r in rows: cnt[r["src"].split("-")[0] + ("-" + r["src"][-1] if r["src"].startswith("math") else "")] = cnt.get(r["src"].split("-")[0] + ("-" + r["src"][-1] if r["src"].startswith("math") else ""), 0) + 1
print(len(rows), "rows", cnt)
