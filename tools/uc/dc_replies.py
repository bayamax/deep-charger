#!/usr/bin/env python3
"""deep-charger's rollouts on the use-case bench (pool_eval.py --stop eos rows) -> {"id", "q", "kind", "reply"} rows for
uc_judge.py: the reply is what the app shows, the text after </think> with the end token and any stray tool tags removed
(an unfinished rollout - no </think> - shows nothing). Matched to the bench by the prompt text.

  python3 dc_replies.py --bench uc_bench.jsonl --rollouts uc_dc210.jsonl --out dc210.jsonl --model 2.1.0
"""
import argparse, json, re

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True); ap.add_argument("--rollouts", required=True); ap.add_argument("--out", required=True); ap.add_argument("--model", default="dc")
A = ap.parse_args()
bench = {json.loads(l)["q"].strip(): json.loads(l) for l in open(A.bench) if l.strip()}
n = empty = 0
with open(A.out, "w") as f:
    for l in open(A.rollouts):
        try: r = json.loads(l)
        except Exception: continue
        b = bench.get((r.get("q") or "").strip())
        if not b: continue
        t = r.get("text") or ""
        rep = t.split("</think>")[-1] if "</think>" in t else ""
        rep = re.sub(r"<｜end▁of▁sentence｜>.*", "", rep, flags=re.S)
        rep = re.sub(r"<search>.*?(</search>|$)|<information>.*?(</information>|$)|<more\s*/?>", "", rep, flags=re.S).strip()
        empty += not rep; n += 1
        f.write(json.dumps({"id": b["id"], "q": b["q"], "kind": b.get("kind"), "model": A.model, "reply": rep, "searches": r.get("ns")}, ensure_ascii=False) + "\n")
print(f"[dc] {n} replies ({empty} empty: no </think>) -> {A.out}")
