#!/usr/bin/env python3
"""usage: head_scan.py pooleval_X.jsonl [pool_eval_cache.jsonl]
For student-wrong rows where the gold string never appeared in the served text: look at the FULL cached page of
each query and find the first token offset of the gold. Table: how many of those failures would have had gold served
with head = 256/512/768/1024/1536/2048/anywhere, and how the teacher did on the same rows."""
import json, sys, re, os
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/root/fft_hf")
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()
def has(t, g): return (" " + norm(g) + " ") in (" " + norm(t) + " ")
cache = {}
for l in open(sys.argv[2] if len(sys.argv) > 2 else "/root/work/pool_eval_cache.jsonl"):
    try: d = json.loads(l); cache[d["kw"]] = d["page"]
    except Exception: pass
R = [json.loads(l) for l in open(sys.argv[1])]
T = {}
if os.path.exists("/root/work/teacher600.jsonl"):
    for l in open("/root/work/teacher600.jsonl"):
        try: r = json.loads(l); T.setdefault(r["q"].strip(), []).append(r)
        except Exception: pass
HEADS = [256, 512, 768, 1024, 1536, 2048]
wrong = [r for r in R if not r["correct"]]
never = [r for r in wrong if not any(has(m, r["gold"]) for m in re.findall(r"<information>(.*?)</information>", r["text"], re.S))]
best = []   # per row: min token offset of gold over its queries' full pages (None = not on any fetched page)
nq0 = 0; miss_cache = 0
for r in never:
    qs = [q for q in r.get("queries", []) if q]
    if not qs: nq0 += 1; best.append(None); continue
    off = None
    for q in qs:
        pg = cache.get(q)
        if pg is None: miss_cache += 1; continue
        if not has(pg, r["gold"]): continue
        ids = tok.encode(pg, add_special_tokens=False)
        lo, hi = 0, len(ids)          # smallest prefix length whose decoded text contains gold
        while lo < hi:
            mid = (lo + hi) // 2
            if has(tok.decode(ids[:mid]), r["gold"]): hi = mid
            else: lo = mid + 1
        off = lo if off is None else min(off, lo)
    best.append(off)
n = len(never)
print(f"=== HEAD_SCAN rows={len(R)} wrong={len(wrong)} gold-never-served={n} (no-query rows {nq0}, cache misses {miss_cache})")
on_page = sum(o is not None for o in best)
print(f"gold is somewhere on a page the student fetched: {on_page}/{n}  (not on any fetched page: {n - on_page})")
for h in HEADS:
    print(f"  head={h:<5} would have served gold in {sum(o is not None and o <= h for o in best):3d}/{n}")
tk = sum(any(x['correct'] for x in T.get(r['q'].strip(), [])) for r in never)
print(f"teacher got at least one roll right on these never-served rows: {tk}/{n}")
if T:
    tserved = sum(any(has(m, r['gold']) for x in T.get(r['q'].strip(), []) for m in re.findall(r"<information>(.*?)</information>", x['text'], re.S)) for r in never)
    print(f"teacher had gold served (any roll) on these rows: {tserved}/{n}")
# same scan restricted to rows where gold is on a fetched page but beyond 256
deep = [(r, o) for r, o in zip(never, best) if o is not None and o > 256]
print(f"rows with gold on a fetched page beyond the 256 head: {len(deep)}; teacher right on {sum(any(x['correct'] for x in T.get(r['q'].strip(), [])) for r, _ in deep)} of them")
for r, o in deep[:8]:
    print(f"  off={o:5d} q={r['q'][:80]!r} gold={r['gold']!r} queries={[q[:40] for q in r['queries'][:4]]}")
print("=== HEAD_SCAN END")
