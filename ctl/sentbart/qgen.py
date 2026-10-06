#!/usr/bin/env python3
"""Queries for the sentence BART's query -> article training: for each document of pick_docs.py's file, from its
passage, (1) a search query the way the app model types one (a short keyword query, 3-8 words) and (2) a HyDE
sentence - one sentence written the way the article would state the fact. By a small teacher (nano). Resumable.
Output rows: {id, s0, s1, query, hyde}.

  python3 qgen.py --docs docs_for_qgen.jsonl --out queries.jsonl --workers 8
"""
import argparse, json, os, time, urllib.error, urllib.request
import concurrent.futures as cf
ap = argparse.ArgumentParser()
ap.add_argument("--docs", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model", default="gpt-5-nano"); ap.add_argument("--workers", type=int, default=8)
A = ap.parse_args()
KEY = open("/root/.oai").read().strip(); URL = "https://api.openai.com/v1/chat/completions"
SYS = ("You read a passage of an English Wikipedia article and write two things someone looking for the fact in it might use. "
       "1) query: a search query as typed into a search box - 3 to 8 words, keywords only, no question mark, naming the subject when the passage does. "
       "2) hyde: ONE sentence, 15-30 words, written the way the article itself states the fact - encyclopedic, declarative, naming the subject; "
       "it must not copy the passage's sentence word for word. Reply as JSON: {\"query\": \"...\", \"hyde\": \"...\"}")
ERR = [0]


def ask(r, tries=4):
    body = {"model": A.model, "messages": [{"role": "system", "content": SYS},
            {"role": "user", "content": f"Article title: {r['title']}\nPassage: {r['passage'][:1200]}"}],
            "response_format": {"type": "json_object"}, "max_completion_tokens": 3000}
    for t in range(tries):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(),
                headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}), timeout=120))
            ch = d["choices"][0]; c = ch["message"].get("content") or ""
            if not c.strip():                       # the reasoning used the whole budget: no content to parse
                ERR[0] += 1
                if ERR[0] % 50 == 1: print(f"[qgen] empty content (finish {ch.get('finish_reason')}, usage {d.get('usage', {}).get('completion_tokens_details')}) x{ERR[0]}", flush=True)
                body["max_completion_tokens"] = 6000; continue
            v = json.loads(c[c.find("{"): c.rfind("}") + 1])
            q, h = (v.get("query") or "").strip(), (v.get("hyde") or "").strip()
            if q and h: return {"id": r["id"], "s0": r["s0"], "s1": r["s1"], "query": q, "hyde": h}
            return None
        except urllib.error.HTTPError as e:
            msg = e.read()[:200].decode(errors="replace") if hasattr(e, "read") else ""
            ERR[0] += 1
            if ERR[0] % 50 == 1 or ERR[0] <= 5: print(f"[qgen] http {e.code} x{ERR[0]}: {msg}", flush=True)
            if e.code in (401, 402) or "insufficient_quota" in msg: raise SystemExit(f"QGEN_ABORT http {e.code} {msg[:100]}")
            time.sleep(20 if e.code == 429 else 3 * (t + 1))
        except Exception as e:
            ERR[0] += 1
            if ERR[0] % 50 == 1 or ERR[0] <= 5: print(f"[qgen] error x{ERR[0]}: {type(e).__name__} {str(e)[:120]}", flush=True)
            time.sleep(3 * (t + 1))
    return None


rows = [json.loads(l) for l in open(A.docs) if l.strip()]
done = set()
if os.path.exists(A.out):
    done = {json.loads(l)["id"] for l in open(A.out) if l.strip()}
todo = [r for r in rows if r["id"] not in done]
print(f"[qgen] {len(rows)} documents, {len(todo)} to go, {A.workers} at a time", flush=True)
ok = 0; t0 = time.time()
with open(A.out, "a") as fo, cf.ThreadPoolExecutor(A.workers) as ex:
    for k, r in enumerate(ex.map(ask, todo)):
        if r: ok += 1; fo.write(json.dumps(r, ensure_ascii=False) + "\n"); fo.flush()
        if (k + 1) % 500 == 0: print(f"[qgen] {k+1}/{len(todo)}: {ok} written ({(time.time()-t0)/60:.0f} min)", flush=True)
ex_ = rows[0] if rows else {}
print(f"QGEN_DONE {ok} of {len(todo)} written ({len(done)} were there); e.g. {ex_.get('title','')!r} -> " +
      (open(A.out).readline().strip()[:200] if os.path.exists(A.out) else ""), flush=True)
