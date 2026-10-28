"""Problems for GRPO under the deployment loop: long-reasoning questions with a checkable answer, mixed like real use.

  python -I build_grpo_prompts.py SRC_DIR OUT.jsonl
SRC_DIR holds the parquet files: math_<subject>.parquet (EleutherAI/hendrycks_math train), dsr.parquet
(agentica-org/DeepScaleR-Preview-Dataset: AIME/AMC/Omni-MATH/STILL), mmlupro.parquet (TIGER-Lab/MMLU-Pro test),
m500.parquet (HuggingFaceH4/MATH-500, only to drop overlaps with the 96-problem check).
Rows: {"src", "id", "gold", "q"} - q is the user turn (sp_qwen3 renders the chat template), gold the answer
(a LaTeX expression, or a letter for the multiple-choice rows).
"""
import glob, json, os, random, re, sys

import pandas as pd

SRC, OUT = sys.argv[1], sys.argv[2]
SUF = "\nPlease reason step by step, and put your final answer within \\boxed{}."
SUF_MC = "\nPlease reason step by step, and put the letter of your final answer within \\boxed{}."
N_MATH, N_DSR, N_MC = 1200, 900, 900
rng = random.Random(0)
key = lambda s: re.sub(r"\s+", "", s or "")
ban = {key(p) for p in pd.read_parquet(os.path.join(SRC, "m500.parquet"))["problem"]}


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
# MATH train, levels 3-5 (the 96-problem check is MATH-500 levels 2-4): an equal share per level
m = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(SRC, "math_*.parquet")))], ignore_index=True)
by_lv = {}
for i, r in m.iterrows():
    g = last_boxed(r["solution"])
    if r["level"] in ("Level 3", "Level 4", "Level 5") and g and len(g) <= 40 and key(r["problem"]) not in ban:
        by_lv.setdefault(r["level"], []).append({"src": "math-" + r["level"][-1], "id": f"math{i}", "gold": g, "q": r["problem"] + SUF})
for lv in sorted(by_lv): rng.shuffle(by_lv[lv]); rows += by_lv[lv][:N_MATH // 3]
# competition problems (DeepScaleR): short answers, and the shorter half of the reference solutions (the longest ones are
# mostly far past what a 1.7B model solves within the length cap - all-wrong groups teach nothing)
d = pd.read_parquet(os.path.join(SRC, "dsr.parquet"))
d = d[d["answer"].str.len().between(1, 20) & ~d["problem"].map(key).isin(ban)]
d = d[d["solution"].str.len() <= d["solution"].str.len().median()]
for i, r in d.sample(N_DSR, random_state=0).iterrows():
    rows.append({"src": "comp", "id": f"dsr{i}", "gold": r["answer"], "q": r["problem"] + SUF})
# MMLU-Pro: every category equally (law, history, health, ... beside the sciences) - the varied side of real use
p = pd.read_parquet(os.path.join(SRC, "mmlupro.parquet")); cats = sorted(p["category"].unique()); per = N_MC // len(cats)
for c in cats:
    for _, r in p[p["category"] == c].sample(per, random_state=0).iterrows():
        opts = "\n".join(f"{chr(65 + k)}. {o}" for k, o in enumerate(r["options"]))
        rows.append({"src": "mc-" + c.replace(" ", "_"), "id": f"mmlupro{r['question_id']}", "gold": r["answer"],
                     "q": r["question"] + "\n\nOptions:\n" + opts + SUF_MC})
rng.shuffle(rows)
with open(OUT, "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
cnt = {}
for r in rows: cnt[r["src"].split("-")[0]] = cnt.get(r["src"].split("-")[0], 0) + 1
print(len(rows), "rows", cnt)
