#!/usr/bin/env python3
"""Bundles of K independent GSM8K questions as one task (the multi-part problems that trip a small model's bookkeeping),
and, when an API key is given, each bundle split into sub-questions by a large model (the "big planner" arm of
par_sp.py's decomp arm). The key is read from DEEPSEEK_API_KEY (or OPENAI_API_KEY with --api openai), never printed.

  DEEPSEEK_API_KEY=... python3 par_bundles.py --gsm gsm8k_test.parquet --parts 3,5 --per 24 --out bundles.jsonl
"""
import argparse, json, os, re, sys, time, urllib.request

import numpy as np
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--gsm", required=True); ap.add_argument("--parts", default="3,5"); ap.add_argument("--per", type=int, default=24)
ap.add_argument("--out", required=True); ap.add_argument("--api", default="deepseek", choices=["deepseek", "openai", "none"]); ap.add_argument("--skip", type=int, default=100)
A = ap.parse_args()
rows = pq.read_table(A.gsm).to_pylist(); order = np.random.RandomState(1).permutation(len(rows))[A.skip:]
gold = lambda r: r["answer"].split("####")[-1].strip().replace(",", "")
bundles = []; p = 0
for K in [int(x) for x in A.parts.split(",")]:
    for i in range(A.per):
        rs = [rows[j] for j in order[p:p + K]]; p += K
        bundles.append({"id": f"k{K}_{i}", "k": K, "qs": [r["question"] for r in rs], "golds": [gold(r) for r in rs]})


def task_text(b): return "Solve this task:\n" + "\n".join(f"({j + 1}) {q}" for j, q in enumerate(b["qs"]))


PLAN = ("Split the task below into independent sub-questions that can each be solved on their own, in the task's order. "
        "Each sub-question must be self-contained: copy every number and name it needs. Do not solve anything.\n"
        'Return only JSON: {"subquestions": [str, ...]}\n\n')


def api_split(text):
    key = os.environ.get("DEEPSEEK_API_KEY" if A.api == "deepseek" else "OPENAI_API_KEY", "")
    url = "https://api.deepseek.com/chat/completions" if A.api == "deepseek" else "https://api.openai.com/v1/chat/completions"
    model = "deepseek-chat" if A.api == "deepseek" else "gpt-4o-mini"
    body = {"model": model, "messages": [{"role": "user", "content": PLAN + text}], "temperature": 0, "max_tokens": 1200}
    if A.api == "deepseek": body["response_format"] = {"type": "json_object"}
    for t in range(3):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
            d = json.load(urllib.request.urlopen(req, timeout=120)); c = d["choices"][0]["message"]["content"]
            a, e = c.find("{"), c.rfind("}"); j = json.loads(c[a:e + 1]); subs = [str(x).strip() for x in j.get("subquestions", []) if str(x).strip()]
            return subs, d.get("usage", {})
        except Exception as ex:
            err = ex; time.sleep(2 * (t + 1))
    print(f"[bundles] api failed: {str(err)[:200]}", file=sys.stderr); return None, {}


tot_in = tot_out = 0; ok = 0
with open(A.out, "w") as f:
    for b in bundles:
        if A.api != "none":
            subs, u = api_split(task_text(b)); b["api_sub"] = subs; tot_in += u.get("prompt_tokens", 0); tot_out += u.get("completion_tokens", 0)
            ok += bool(subs) and len(subs) == b["k"]
        f.write(json.dumps(b) + "\n")
print(f"[bundles] {len(bundles)} bundles -> {A.out}" + (f"; api splits with the right count {ok}/{len(bundles)}; tokens in {tot_in}, out {tot_out}" if A.api != "none" else ""))
