#!/usr/bin/env python3
"""A term index over the WHOLE text of every article: which articles mention a word anywhere in their body.
This is the reach Wikipedia's own search has and the opening-only store lacks. Exact, not learned.

Two passes over the dump (streamed one parquet file at a time, nothing kept):
  1. document frequency of every token (letters/digits, 3+ characters, not a stopword);
  2. the vocabulary = tokens in 2..--max-df articles (the very common ones carry no reach and would be the bulk
     of the index); each article's distinct vocabulary tokens are scattered into a postings array laid out by
     term (counting sort by the pass-1 counts: no sort, article ids come out increasing).
Files: terms.txt (one term a line, in id order), term_offsets.npy (V+1,) int64, postings.bin uint32 article ids,
term_df.npy (V,) int32. About 8 bytes a (term, article) pair before any packing; the sizes are printed.

  python3 build_terms.py --out /tmp/wterms --store /tmp/wstore6 [--files 0-40]
"""
import argparse, os, re, sys, time, json, collections
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search import WORD, STOP
from huggingface_hub import hf_hub_download
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--store", required=True, help="the store whose article order the ids follow (titles.txt)")
ap.add_argument("--files", default="0-40"); ap.add_argument("--max-df", type=int, default=200_000); ap.add_argument("--min-len", type=int, default=3)
ap.add_argument("--keep-dump", type=int, default=0)
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
a, b = map(int, A.files.split("-")); FILES = list(range(a, b + 1))
# the store's cleaning, so the same articles are kept (stubs dropped) and ids line up
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_store.py")).read()
i = src.index("def clean("); j = src.index("\n\n", src.index("def cut(")); exec("\n".join(l for l in src[:i].splitlines() if re.match(r"^[A-Z_]+ = re\.compile", l)) + "\n" + src[i:j])
tid = {}
for k, t in enumerate(open(os.path.join(A.store, "titles.txt"), encoding="utf-8")): tid.setdefault(t.rstrip("\n"), k)
N = len(tid); print(f"[terms] {N} articles in the store", flush=True)
def dump(k):
    return hf_hub_download("wikimedia/wikipedia", f"20231101.en/train-{k:05d}-of-00041.parquet", repo_type="dataset", local_dir="/tmp/wdump")
def articles(k):
    f = dump(k); pf = pq.ParquetFile(f)
    for batch in pf.iter_batches(batch_size=2048, columns=["title", "text"]):
        for title, text in zip(batch.column("title").to_pylist(), batch.column("text").to_pylist()):
            title = (title or "").replace("\n", " ").strip(); body = clean(text or "")
            if not title or len(body) < 80 or title not in tid: continue
            yield tid[title], title, body
    if not A.keep_dump: os.remove(f)
def toks(title, body):
    return set(w for w in (x.lower() for x in WORD.findall(title + " " + body)) if len(w) >= A.min_len and w not in STOP)
t0 = time.time(); df = collections.Counter(); n = 0
for k in FILES:
    for i_, title, body in articles(k):
        df.update(toks(title, body)); n += 1
    print(f"[terms] pass 1 file {k}: {n} articles, {len(df)} distinct tokens, {time.time()-t0:.0f}s", flush=True)
vocab = sorted(t for t, c in df.items() if 2 <= c <= A.max_df)
V = len(vocab); term_id = {t: i for i, t in enumerate(vocab)}
counts = np.array([df[t] for t in vocab], dtype=np.int64); offsets = np.concatenate([[0], np.cumsum(counts)])
P = int(offsets[-1]); print(f"[terms] vocabulary {V} of {len(df)} tokens (2..{A.max_df} articles); {P} postings = {P*4/1e9:.2f} GB as uint32", flush=True)
del df
with open(os.path.join(A.out, "terms.txt"), "w", encoding="utf-8") as fh:
    for t in vocab: fh.write(t + "\n")
np.save(os.path.join(A.out, "term_offsets.npy"), offsets); np.save(os.path.join(A.out, "term_df.npy"), counts.astype(np.int32))
post = np.memmap(os.path.join(A.out, "postings.bin"), dtype=np.uint32, mode="w+", shape=(P,))
cur = offsets[:-1].copy(); n = 0; t0 = time.time()
for k in FILES:
    for i_, title, body in articles(k):
        ids = [term_id[w] for w in toks(title, body) if w in term_id]
        if ids:
            ids = np.array(ids, dtype=np.int64); pos = cur[ids]; post[pos] = i_; cur[ids] = pos + 1
        n += 1
    print(f"[terms] pass 2 file {k}: {n} articles, {time.time()-t0:.0f}s", flush=True)
post.flush()
short = int((cur != offsets[1:]).sum()); print(f"[terms] terms whose postings came out short (article dropped between passes): {short}")
json.dump({"n_docs": N, "vocab": V, "postings": P, "max_df": A.max_df, "min_len": A.min_len}, open(os.path.join(A.out, "terms_meta.json"), "w"))
print(f"TERMS_DONE {V} terms {P} postings {os.path.getsize(os.path.join(A.out, 'postings.bin'))/1e9:.2f} GB", flush=True)
