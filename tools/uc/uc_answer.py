#!/usr/bin/env python3
"""A hosted model's replies to the use-case prompts, as its own product answers them (one user message, no system
prompt beyond the model's defaults): the opponent's side of the comparison. Resumable.

  OPENAI_API_KEY=... python3 -I uc_answer.py --inp uc_bench.jsonl --out ans_4omini.jsonl --model gpt-4o-mini
"""
import argparse, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True); ap.add_argument("--model", default="gpt-4o-mini")
ap.add_argument("--api", default="openai", choices=["openai", "deepseek"]); ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--max-tokens", type=int, default=900); ap.add_argument("--threads", type=int, default=12)
A = ap.parse_args()
URL = "https://api.openai.com/v1/chat/completions" if A.api == "openai" else "https://api.deepseek.com/chat/completions"
KEY = os.environ.get("OPENAI_API_KEY" if A.api == "openai" else "DEEPSEEK_API_KEY", "")


def answer(r):
    body = {"model": A.model, "messages": [{"role": "user", "content": r["q"]}], "temperature": A.temp, "max_tokens": A.max_tokens}
    for t in range(4):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
            d = json.load(urllib.request.urlopen(req, timeout=180)); return d["choices"][0]["message"]["content"] or "", d.get("usage", {})
        except Exception as ex:
            err = ex; time.sleep(3 * (t + 1))
    return "", {"error": str(err)[:120]}


rows = [json.loads(l) for l in open(A.inp) if l.strip()]
done = set()
if os.path.exists(A.out):
    for l in open(A.out):
        try: done.add(json.loads(l)["id"])
        except Exception: pass
todo = [r for r in rows if r["id"] not in done]; tin = tout = 0
with open(A.out, "a") as fh, ThreadPoolExecutor(A.threads) as ex:
    for r, (rep, u) in zip(todo, ex.map(answer, todo)):
        fh.write(json.dumps({"id": r["id"], "kind": r.get("kind"), "q": r["q"], "model": A.model, "reply": rep}, ensure_ascii=False) + "\n"); fh.flush()
        tin += u.get("prompt_tokens", 0); tout += u.get("completion_tokens", 0)
print(f"[answer] {A.model}: {len(todo)} replies -> {A.out} (tokens in {tin} out {tout})", flush=True)
