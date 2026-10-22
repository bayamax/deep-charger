#!/usr/bin/env python3
"""An API model (gpt-4o-mini by default) on the search held-out under the same protocol as pool_eval.py: the question
alone, <search>keywords</search> answered with the local store's page (its first 256 tokens of the model's tokenizer),
<more> for the next 256, at most --maxs searches, and the reply's first sentence must carry the gold ("the answer is X").
The target number for the on-device model: what a hosted model reaches with the same search.

  OPENAI_API_KEY=... python3 api_search_eval.py --store wiki_store --bge bge-small --tok tok --q eval300.jsonl --out o.jsonl
  --search 0 runs the same model closed-book (no tool) for the gap the search is worth.
The key is read from the environment only; nothing here prints it.
"""
import argparse, collections, json, os, re, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--store", required=True); ap.add_argument("--bge", required=True); ap.add_argument("--tok", required=True)
ap.add_argument("--q", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=300)
ap.add_argument("--g", type=int, default=2, help="rollouts per distinct question"); ap.add_argument("--model", default="gpt-4o-mini")
ap.add_argument("--api", default="openai", choices=["openai", "deepseek"]); ap.add_argument("--temp", type=float, default=0.6)
ap.add_argument("--maxs", type=int, default=5); ap.add_argument("--maxm", type=int, default=8); ap.add_argument("--search", type=int, default=1)
ap.add_argument("--style", default="search", choices=["search", "verify"], help="verify: answer from knowledge, the search only to check when unsure")
ap.add_argument("--localsearch", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ctl", "localsearch"))
A = ap.parse_args()
PAGE_STEP = 256
NOTICE = "(no searches left - answer from what you have read)"; NOMORE = "(no more of this page - search again or answer)"

sys.path.insert(0, A.localsearch)
from search import LocalSearch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
tok = Tokenizer.from_file(os.path.join(A.tok, "tokenizer.json"))
LS = LocalSearch(A.store, A.bge) if A.search else None
cache = {}; CF = A.out + ".pages"
if os.path.exists(CF):
    for l in open(CF):
        try: d = json.loads(l); cache[d["kw"]] = d["page"]
        except Exception: pass
cfh = open(CF, "a")


def get_page(kw):
    if kw not in cache:
        cache[kw] = LS.fetch(kw); cfh.write(json.dumps({"kw": kw, "page": cache[kw]}, ensure_ascii=False) + "\n"); cfh.flush()
    return cache[kw]


def norm(s): return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()
def has(t, g): return (" " + norm(g) + " ") in (" " + norm(t) + " ")
ABBR = {"st", "mr", "mrs", "ms", "dr", "jr", "sr", "mt", "vs", "no", "inc", "ltd", "co"}


def head_sentence(a):
    h = (a or "").strip().split("\n")[0]
    for m in re.finditer(r"[.!?](?=\s|$)", h):
        t = re.split(r"[\s(\"]", h[:m.start()])[-1]
        if len(t) == 1 or t.lower() in ABBR: continue
        return h[:m.end()][:120]
    return h[:120]


SYS_SEARCH = ("You answer questions with the help of a Wikipedia search tool. To search, write exactly <search>keywords</search> and stop; "
              "the first part of the best matching page is then shown to you inside <information>...</information>. Write <more> and stop to read "
              f"the next part of the same page. You may search at most {A.maxs} times. When you know the answer, reply with ONE line that begins "
              "'The answer is X.' (X as short as possible, as it would appear in Wikipedia), followed at most by one sentence of justification.")
SYS_VERIFY = ("You answer questions from your own knowledge, and you have a Wikipedia search tool to check facts you are unsure of. To search, write exactly <search>keywords</search> and stop; "
              "the first part of the best matching page is then shown to you inside <information>...</information>. Write <more> and stop to read the next part. "
              f"At most {A.maxs} searches; do not search when you already know the answer. If a page contradicts your belief, weigh which is more reliable. When ready, reply with ONE line that begins "
              "'The answer is X.' (X as short as possible, as it would appear in Wikipedia), followed at most by one sentence of justification.")
SYS_PLAIN = "Answer the question from your own knowledge. Reply with ONE line that begins 'The answer is X.' (X as short as possible, as it would appear in Wikipedia)."
URL = "https://api.openai.com/v1/chat/completions" if A.api == "openai" else "https://api.deepseek.com/chat/completions"
KEY = os.environ.get("OPENAI_API_KEY" if A.api == "openai" else "DEEPSEEK_API_KEY", "")
usage = collections.Counter()


def chat(msgs, stop):
    body = {"model": A.model, "messages": msgs, "temperature": A.temp, "max_tokens": 400, "stop": stop}
    for t in range(4):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
            d = json.load(urllib.request.urlopen(req, timeout=120)); u = d.get("usage", {})
            usage["in"] += u.get("prompt_tokens", 0); usage["out"] += u.get("completion_tokens", 0)
            return d["choices"][0]["message"]["content"] or "", d["choices"][0].get("finish_reason")
        except Exception as ex:
            err = ex; time.sleep(3 * (t + 1))
    print(f"[api] failed: {str(err)[:200]}", file=sys.stderr); return "", "error"


def rollout(q):
    msgs = [{"role": "system", "content": (SYS_VERIFY if A.style == "verify" else SYS_SEARCH) if A.search else SYS_PLAIN}, {"role": "user", "content": q}]
    ns = nm = 0; served, queries, page_ids, off = [], [], [], 0; txt = ""
    for step in range(A.maxs + A.maxm + 3):
        out, fin = chat(msgs, ["</search>", "<more>"] if A.search else None)
        txt += out
        m = re.search(r"<search>(.*)$", out, re.S) if A.search else None
        if m and fin == "stop" and "answer is" not in out.lower():
            ns += 1; kw = m.group(1).strip().replace("</search>", ""); queries.append(kw)
            if ns > A.maxs: blk = NOTICE
            else:
                pg = get_page(kw) if kw else ""
                if not pg: blk = "(no results)"; page_ids, off = [], 0
                else: page_ids = tok.encode(pg, add_special_tokens=False).ids; blk = tok.decode(page_ids[:PAGE_STEP]); off = PAGE_STEP; served.append(blk)
            msgs.append({"role": "assistant", "content": out + "</search>"}); msgs.append({"role": "user", "content": f"<information>\n{blk}\n</information>"}); continue
        if A.search and fin == "stop" and out.strip().endswith("<more") or (A.search and "<more>" in out and "answer is" not in out.lower()):
            nxt = page_ids[off:off + PAGE_STEP] if nm < A.maxm else []
            if nxt: nm += 1; off += len(nxt); blk = tok.decode(nxt); served.append(blk)
            else: blk = NOMORE
            msgs.append({"role": "assistant", "content": out + "<more>"}); msgs.append({"role": "user", "content": f"<information>\n{blk}\n</information>"}); continue
        msgs.append({"role": "assistant", "content": out}); break
    reply = msgs[-1]["content"] if msgs[-1]["role"] == "assistant" else ""
    ans = head_sentence(reply)
    return {"answer": ans, "reply": reply[:600], "ns": ns, "more": nm, "queries": queries, "served_n": len(served),
            "grounded": any(has(s, GOLD) for s in served), "landed": bool(ans.strip())}


rows = [json.loads(l) for l in open(A.q) if l.strip()][:A.n]
qs, seen = [], set()
for r in rows:
    if r["q"] not in seen: seen.add(r["q"]); qs.append((r["q"], r["gold"]))
done = collections.Counter()
if os.path.exists(A.out):
    for l in open(A.out):
        try: done[json.loads(l)["q"]] += 1
        except Exception: pass
stat = collections.Counter(); t0 = time.time()
with open(A.out, "a") as fh:
    for i, (q, GOLD) in enumerate(qs):
        for j in range(A.g):
            if done[q] > j: continue
            r = rollout(q); r.update(q=q, gold=GOLD, model=A.model, search=A.search, style=A.style)
            r["correct"] = r["landed"] and has(r["answer"], GOLD)
            fh.write(json.dumps(r, ensure_ascii=False) + "\n"); fh.flush()
            stat["n"] += 1; stat["c"] += r["correct"]; stat["g"] += r["grounded"]; stat["s"] += r["ns"]
        if (i + 1) % 10 == 0:
            n = max(stat["n"], 1)
            print(f"[api] {i + 1}/{len(qs)} questions: correct {100 * stat['c'] / n:.1f}% grounded {100 * stat['g'] / n:.1f}% searches {stat['s'] / n:.1f} "
                  f"[{(time.time() - t0) / 60:.1f} min, tokens in {usage['in']} out {usage['out']}]", flush=True)
# the whole file: per rollout and per question (either / both of the g rollouts right)
R = [json.loads(l) for l in open(A.out)]; byq = collections.defaultdict(list)
for r in R: byq[r["q"]].append(r["correct"])
n = len(R); print(f"[api] {A.model} search={A.search}: {n} rollouts, correct {100 * sum(r['correct'] for r in R) / n:.1f}%, grounded {100 * sum(r['grounded'] for r in R) / n:.1f}%, "
                  f"searches {sum(r['ns'] for r in R) / n:.2f}; per question any {100 * sum(any(v) for v in byq.values()) / len(byq):.1f}% all {100 * sum(all(v) for v in byq.values()) / len(byq):.1f}% "
                  f"({len(byq)} questions); tokens in {usage['in']} out {usage['out']}", flush=True)
