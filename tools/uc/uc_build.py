#!/usr/bin/env python3
"""The use-case set: what people actually ask a chat assistant, taken from WildChat (real first turns of real
conversations, ODC-BY), English, non-toxic, one message that stands alone (no pasted document, no "continue"), sorted by
a hosted classifier into the kinds of request, then a held-out bench (stratified, --bench per kind at most) and a
training pool (the rest; never the bench's). Every prompt keeps its source id.

  OPENAI_API_KEY=... python3 -I uc_build.py --parquet shard0.parquet [shard1 ...] --out-dir uc --bench 300
"""
import argparse, collections, json, os, random, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", nargs="+", required=True); ap.add_argument("--out-dir", required=True)
ap.add_argument("--bench", type=int, default=300); ap.add_argument("--pool", type=int, default=6000, help="prompts classified (bench + training pool)")
ap.add_argument("--model", default="gpt-5.4-nano"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=16)
A = ap.parse_args()
import pyarrow.parquet as pq  # noqa: E402

KINDS = {
    "fact": "a factual question with a short answer (who/what/when/where, a number, a name)",
    "explain": "asks to explain a concept, how something works, or why something happens",
    "howto": "practical how-to or advice for the person's own situation (health, money, travel, cooking, devices, study, work)",
    "recommend": "asks for suggestions, ideas, a plan, a list of options, or a recommendation",
    "compare": "compare options, pros and cons, or an opinion / judgement on something",
    "writing": "write, rewrite, summarise or translate a text (email, post, story, poem, letter)",
    "code": "programming: write, fix or explain code",
    "math": "maths, logic or a worked problem",
    "chat": "small talk, feelings, a personal conversation, or a role-play opener",
}
SYS = ("You classify one message a person sent to a chat assistant. Reply with JSON only: "
       '{"kind": one of ' + json.dumps(list(KINDS)) + ', "standalone": true/false, "phone_fit": true/false}. Kinds: '
       + "; ".join(f"{k} = {v}" for k, v in KINDS.items()) +
       ". standalone = the message can be answered as it is (it does not refer to an attachment, earlier messages, a pasted long "
       "document, or say 'continue'). phone_fit = a helpful reply fits in a phone chat (under about 300 words) and needs no files, "
       "images or live data past 2023.")
KEY = os.environ.get("OPENAI_API_KEY", "")


def classify(q):
    body = {"model": A.model, "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": q[:1500]}], "max_completion_tokens": 400,
            "response_format": {"type": "json_object"}}
    for t in range(4):
        try:
            req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
            d = json.load(urllib.request.urlopen(req, timeout=120)); j = json.loads(d["choices"][0]["message"]["content"])
            return j, d.get("usage", {})
        except Exception as ex:
            err = ex; time.sleep(2 * (t + 1))
    return {"error": str(err)[:120]}, {}


rows, seen = [], set()
for p in A.parquet:
    for r in pq.read_table(p, columns=["conversation_hash", "conversation", "language", "toxic"]).to_pylist():
        if r["language"] != "English" or r["toxic"] or not r["conversation"]: continue
        q = (r["conversation"][0].get("content") or "").strip()
        if not (12 <= len(q) <= 800) or q.count("\n") > 4: continue
        k = re.sub(r"\W+", " ", q.lower()).strip()[:120]
        if k in seen: continue
        seen.add(k); rows.append({"id": r["conversation_hash"], "q": q})
random.Random(A.seed).shuffle(rows); rows = rows[:A.pool]
print(f"[uc] {len(rows)} candidate prompts to classify with {A.model}", flush=True)
os.makedirs(A.out_dir, exist_ok=True); CF = os.path.join(A.out_dir, "classified.jsonl")
done = {}
if os.path.exists(CF):
    for l in open(CF):
        try: d = json.loads(l); done[d["id"]] = d
        except Exception: pass
todo = [r for r in rows if r["id"] not in done]; usage = collections.Counter()
with open(CF, "a") as fh, ThreadPoolExecutor(A.threads) as ex:
    for i, (r, (j, u)) in enumerate(zip(todo, ex.map(lambda r: classify(r["q"]), todo))):
        r = dict(r, **{k: j.get(k) for k in ("kind", "standalone", "phone_fit", "error") if k in j}); done[r["id"]] = r
        fh.write(json.dumps(r, ensure_ascii=False) + "\n"); usage["in"] += u.get("prompt_tokens", 0); usage["out"] += u.get("completion_tokens", 0)
        if (i + 1) % 500 == 0: print(f"[uc] classified {i + 1}/{len(todo)} (tokens in {usage['in']} out {usage['out']})", flush=True)
C = [done[r["id"]] for r in rows if r["id"] in done]
ok = [r for r in C if r.get("kind") in KINDS and r.get("standalone") and r.get("phone_fit")]
print(f"[uc] {len(C)} classified, {len(ok)} standalone and phone-fit; kinds {dict(collections.Counter(r['kind'] for r in ok).most_common())}", flush=True)
# the bench: the real mix of kinds, but every kind with at least 12 prompts so each can be read on its own
by = collections.defaultdict(list)
for r in ok: by[r["kind"]].append(r)
share = {k: len(v) / len(ok) for k, v in by.items()}
want = {k: max(12, round(A.bench * s)) for k, s in share.items()}
scale = A.bench / sum(want.values()); want = {k: max(12, int(round(v * scale))) for k, v in want.items()}
bench = []
for k, v in by.items(): bench += v[:want[k]]
bids = {r["id"] for r in bench}; pool = [r for r in ok if r["id"] not in bids]
with open(os.path.join(A.out_dir, "uc_bench.jsonl"), "w") as f:
    for r in bench: f.write(json.dumps({"id": r["id"], "q": r["q"], "kind": r["kind"], "gold": ""}, ensure_ascii=False) + "\n")
with open(os.path.join(A.out_dir, "uc_pool.jsonl"), "w") as f:
    for r in pool: f.write(json.dumps({"id": r["id"], "q": r["q"], "kind": r["kind"], "gold": ""}, ensure_ascii=False) + "\n")
print(f"[uc] bench {len(bench)} ({dict(collections.Counter(r['kind'] for r in bench).most_common())}); training pool {len(pool)}", flush=True)
