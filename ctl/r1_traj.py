#!/usr/bin/env python3
"""Teacher trajectories from R1 for the items the model never solves (probe pass 0 of G).

R1 (deepseek-reasoner) plays the policy in the student's own environment: the same Wikipedia lookups as the
rollouts (online_loop's fetch, the shared page cache), the same 256-token page chunks (a repeated page serves its
next chunk), at most --maxsrch searches. Each step R1 sees the conversation, its trajectory so far and the last
result, and answers JSON: {"thought": ..., "search": ...} or {"thought": ..., "answer": ...}. The trajectory is written
in the student's format - <search>q</search, the injected <information> block, a short thought, ... </think>, reply -
and kept only when the reply holds the gold and a served chunk does (grounded), as the rollouts are scored.

  python3 r1_traj.py --probe /root/work/probe_s100.jsonl --out /root/work/r1_traj.jsonl --tok /root/gptq_hf_gq14
Rows: {q, gold, hist, traj, reply, ns}
"""
import argparse, json, os, re, ssl, threading, time, urllib.parse, urllib.request, urllib.error
import concurrent.futures as cf
ap = argparse.ArgumentParser()
ap.add_argument("--probe", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tok", default="/root/gptq_hf_gq14")
ap.add_argument("--model", default="deepseek-reasoner"); ap.add_argument("--maxsrch", type=int, default=7)
ap.add_argument("--workers", type=int, default=6); ap.add_argument("--n", type=int, default=0)
ap.add_argument("--max-pass", type=int, default=0, help="items whose probe passed at most this many samples")
A = ap.parse_args()
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(A.tok)
DSK = open("/root/.dsk").read().strip()
PAGE_STEP = 256

# ---- the rollouts' Wikipedia lookups (online_loop.fetch), read-through cache shared with them ----
WAPI = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "deep-charger-grpo-ep/1.0 (research; bayamax@icloud.com)"}
CTX = ssl.create_default_context()
try:
    import certifi; CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    pass
cache = {}
for f in ("/root/work/pool_eval_cache.jsonl", "/root/work/r1_page_cache.jsonl"):
    if os.path.exists(f):
        for line in open(f):
            try: d = json.loads(line); cache[d["kw"]] = d["page"]
            except Exception: pass
cache_fh = open("/root/work/r1_page_cache.jsonl", "a"); lock = threading.Lock()


def api(params, tries=3):
    url = WAPI + "?" + urllib.parse.urlencode({**params, "maxlag": 5, "format": "json"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20, context=CTX) as resp:
                out = json.loads(resp.read().decode())
            if isinstance(out, dict) and out.get("error", {}).get("code") == "maxlag":
                time.sleep(5 * (i + 1)); continue
            time.sleep(0.6); return out
        except Exception:
            time.sleep(3)
    return {}


def fetch(kw):
    sd_ = api({"action": "query", "list": "search", "srsearch": kw, "srlimit": 3})
    hits = [h["title"] for h in sd_.get("query", {}).get("search", [])][:3]
    if not hits: return ""
    d = api({"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": "max", "redirects": 1, "titles": "|".join(hits)})
    pages = {}
    for p in d.get("query", {}).get("pages", {}).values():
        t, ex = p.get("title", ""), (p.get("extract", "") or "")
        if ex: pages[t] = ex[:40000]
    for t in hits:
        if t in pages:
            full = api({"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "titles": t})
            for p in full.get("query", {}).get("pages", {}).values():
                if p.get("extract"): return f"{t}: {p['extract'][:40000]}"
            return f"{t}: {pages[t]}"
    return ""


def get_page(kw):
    if kw in cache: return cache[kw]
    page = fetch(kw)
    with lock:
        cache[kw] = page; cache_fh.write(json.dumps({"kw": kw, "page": page}, ensure_ascii=False) + "\n"); cache_fh.flush()
    return page


def norm(s): return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())
def has(t, g): return (" " + norm(g) + " ") in (" " + norm(t) + " ")


SYS = """You are answering a user's question in a chat by searching Wikipedia step by step. Each step, reply with JSON only:
{"thought": "...", "search": "..."}  to look something up (a short keyword query, the way you would type it into Wikipedia's search box), or
{"thought": "...", "answer": "..."}  when the results you have read support the answer.
thought: one or two plain sentences on what the last result showed and what to look up next (the first step may leave it empty).
answer: two or three conversational sentences that answer the user's latest message, stating the answer plainly and only what the results support.
Each search returns about 250 tokens of the best-matching page; searching the same query again returns the next part of that page.
If the latest message refers to something earlier in the conversation ("that film", "he", "it"), search for the thing it refers to by name.
Search the way someone who does NOT know the answer would: build each query only from the conversation and from what the results so far have shown. Never put the answer you expect into a query - find it.
Keep thoughts short and about the results ("The page names X but not Y; search Y."), not about the user."""


def r1(messages, tries=3):
    body = {"model": A.model, "messages": messages, "max_tokens": 6000}
    for t in range(tries):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request("https://api.deepseek.com/chat/completions", data=json.dumps(body).encode(),
                headers={"Authorization": "Bearer " + DSK, "Content-Type": "application/json"}), timeout=300))
            c = d["choices"][0]["message"].get("content") or ""
            return json.loads(c[c.find("{"): c.rfind("}") + 1])
        except urllib.error.HTTPError as e:
            if e.code in (401, 402): print(f"[r1] HTTP {e.code}: stopping", flush=True); raise SystemExit(1)
            time.sleep(5 * (t + 1))
        except Exception:
            time.sleep(5 * (t + 1))
    return None


def solve(it):
    q, gold, hist = it["q"], it["gold"], it.get("hist") or []
    conv = "\n".join(("USER: " if m["role"] == "user" else "ASSISTANT: ") + m["content"][:1500] for m in hist) + ("\n" if hist else "") + "USER: " + q
    steps, traj, served, seen, ns = [], "", [], {}, 0
    for _ in range(A.maxsrch + 2):
        log = "".join(f"\nSTEP {i+1}: thought={s['thought']!r} search={s['search']!r}\nRESULT: {s['result']}" for i, s in enumerate(steps))
        v = r1([{"role": "system", "content": SYS}, {"role": "user", "content": f"CONVERSATION:\n{conv}\n\nYOUR SEARCHES SO FAR:{log or ' none'}\n\nNext step (JSON):"}])
        if not v: return None
        th = (v.get("thought") or "").strip()
        if v.get("answer"):
            reply = v["answer"].strip()
            traj += (th + "\n" if th else "") + "</think>\n\n" + reply
            ok = has(reply, gold) and any(has(c, gold) for c in served)
            return {"q": q, "gold": gold, "hist": hist, "traj": traj, "reply": reply, "ns": ns} if ok else None
        kw = (v.get("search") or "").strip()
        if not kw or ns >= A.maxsrch: return None
        if has(kw, gold) and not has(conv, gold) and not any(has(c, gold) for c in served): return None   # searched for the answer before finding it: not a demonstration
        ns += 1
        pg = get_page(kw)
        if not pg: chunk = "(no results)"
        else:
            ids = tok.encode(pg, add_special_tokens=False); key = pg[:120]; off = seen.get(key, 0)
            nxt = ids[off:off + PAGE_STEP]; chunk = tok.decode(nxt) if nxt else "(this page is used up - search a different query or answer)"
            seen[key] = off + len(nxt)
            if nxt: served.append(chunk)
        traj += (th + "\n" if th else "") + f"<search>{kw}</search" + f"\n<information>\n{chunk}\n[READER] (no extraction)\n</information>\n"
        steps.append({"thought": th, "search": kw, "result": chunk[:1500]})
    return None


items = [json.loads(l) for l in open(A.probe) if l.strip()]
items = [it for it in items if it.get("pass", 0) <= A.max_pass]
if A.n: items = items[:A.n]
done = set()
if os.path.exists(A.out):
    done = set(json.loads(l)["q"] for l in open(A.out) if l.strip())
todo = [it for it in items if it["q"] not in done]
print(f"[r1] {len(items)} items the model never solved, {len(todo)} to go, {A.workers} at a time", flush=True)
ok = 0; t0 = time.time()
with open(A.out, "a") as fo, cf.ThreadPoolExecutor(A.workers) as ex:
    for k, r in enumerate(ex.map(solve, todo)):
        if r: ok += 1; fo.write(json.dumps(r, ensure_ascii=False) + "\n"); fo.flush()
        if (k + 1) % 20 == 0: print(f"[r1] {k+1}/{len(todo)}: {ok} verified ({(time.time()-t0)/60:.0f} min)", flush=True)
print(f"R1_TRAJ_DONE {ok} verified trajectories of {len(todo)}", flush=True)
