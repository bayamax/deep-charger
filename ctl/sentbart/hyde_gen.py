#!/usr/bin/env python3
"""HyDE-style queries for the article-search test: each of the app model's search queries rewritten as ONE sentence
the way the sought Wikipedia article would state it (a hypothetical document sentence), by a small teacher (nano).
The sentence-sequence BART's page vector was trained to be found from a sentence of its article; if the searching model
wrote sentence-like queries instead of keyword lists, would the page vector find the article? This answers it before
anything is trained. The teacher knows more than the app model, so this is an optimistic reading.

  python3 hyde_gen.py --queries dcq.jsonl --out dcq_hyde.jsonl     (adds h_api / h_good next to q_api / q_good)
"""
import argparse, json, time, urllib.request
import concurrent.futures as cf
ap = argparse.ArgumentParser()
ap.add_argument("--queries", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model", default="gpt-5-nano"); ap.add_argument("--workers", type=int, default=8)
A = ap.parse_args()
KEY = open("/root/.oai").read().strip(); URL = "https://api.openai.com/v1/chat/completions"
SYS = ("You turn a Wikipedia search query into ONE sentence written the way the English Wikipedia article being searched "
       "for would state it - encyclopedic, declarative, 15-35 words, naming the subject. Write what the article most likely "
       "says; if unsure of a detail, keep the sentence about the topic rather than inventing specifics. Reply as JSON: "
       '{"sentence": "..."}')
ERR = [0]


def chat(q, tries=4):
    body = {"model": A.model, "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": "Query: " + q}],
            "response_format": {"type": "json_object"}, "max_completion_tokens": 4000}
    for t in range(tries):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(),
                headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}), timeout=120))
            return (json.loads(d["choices"][0]["message"]["content"]).get("sentence") or "").strip()
        except Exception as e:
            if ERR[0] < 3: ERR[0] += 1; print(f"[hyde] api error: {type(e).__name__} {str(e)[:120]}", flush=True)
            time.sleep(3 * (t + 1))
    return ""


rows = [json.loads(l) for l in open(A.queries) if l.strip()]
jobs = [(i, k) for i, r in enumerate(rows) for k in ("q_api", "q_good") if r.get(k)]
with cf.ThreadPoolExecutor(A.workers) as ex:
    for (i, k), s in zip(jobs, ex.map(lambda j: chat(rows[j[0]][j[1]]), jobs)):
        if s: rows[i]["h" + k[1:]] = s
open(A.out, "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
n = sum(1 for r in rows for k in ("h_api", "h_good") if r.get(k))
print(f"HYDE_DONE {n} of {len(jobs)} queries rewritten; e.g. {rows[0].get('q_good') or rows[0].get('q_api')!r} -> {rows[0].get('h_good') or rows[0].get('h_api')!r}", flush=True)
