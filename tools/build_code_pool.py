"""Coding problems for GRPO, graded by running the program (tools/sp_codecheck.py).

  python -I build_code_pool.py OUT.jsonl N SRC.parquet [SRC2.parquet ...]
SRC: agentica-org/DeepCoder-Preview-Dataset primeintellect parquet files (problem, solutions, tests). A problem is kept when
its tests are all stdin/stdout, it has at least 5, and one of its first 3 reference solutions passes the 12 tests kept (the
first 3 - usually the statement's examples - and 9 drawn at random, each under 8000 characters): that drops Python 2
references, broken tests and problems with several right outputs (no special judge here). The reference runs sandboxed as
the model's programs do. Rows as ctl/dpo_pool.jsonl: {"id", "src": "code", "gold": {"tests", "tl"}, "prompt"}.
"""
import json, os, random, re, sys
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sp_codecheck import run_tests

OUT, N, SRCS = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
SUF = "\n\nPut your final solution in a single ```python code block."


def one(item):
    pid, problem, tests, sols = item
    rng = random.Random(pid)
    t = [x for x in json.loads(tests) if len(x["input"]) + len(x["output"]) < 8000]
    if len(t) < 5 or any(x.get("type") != "stdin_stdout" for x in t): return None
    sel = t[:3] + rng.sample(t[3:], min(9, len(t) - 3))
    sel = [{"input": x["input"], "output": x["output"]} for x in sel]
    for s in list(sols)[:3]:
        m = re.search(r"```(?:python3?|py)?\s*\n(.*?)```", s, re.S)
        if run_tests(m.group(1) if m else s, sel, 4):
            return {"id": pid, "src": "code", "gold": {"tests": sel, "tl": 4},
                    "prompt": "<|im_start|>user\n" + problem.strip() + SUF + "<|im_end|>\n<|im_start|>assistant\n"}
    return None


if __name__ == "__main__":
    items = []
    for f in SRCS:
        d = pd.read_parquet(f); tag = os.path.basename(os.path.dirname(f))[:2] + re.search(r"(\d+)-of", f).group(1)
        for i, r in d.iterrows():
            if len(r["problem"]) <= 5000 and len(r["solutions"]): items.append((f"{tag}_{i}", r["problem"], r["tests"], r["solutions"]))
    random.Random(0).shuffle(items); items = items[:N]
    with Pool(4) as p: rows = [r for r in p.imap(one, items, chunksize=4) if r]
    with open(OUT, "w") as fo:
        for r in rows: fo.write(json.dumps(r) + "\n")
    print(f"{len(rows)} of {len(items)} kept")
