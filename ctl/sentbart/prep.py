#!/usr/bin/env python3
"""Wikipedia -> documents as sentence lists, for the sentence-sequence BART.

Reads the English Wikipedia dump on the hub (wikimedia/wikipedia, 20231101.en, parquet shards), splits each
article into sentences (blingfire), drops the tail sections that are lists rather than prose (References, See also,
External links, ...), and writes shards of {"id", "title", "sents": [...]} as jsonl.

  usage: python3 prep.py --out /root/sb/data/docs --shards 0-9 [--min-sents 8] [--max-sents 256]
"""
import argparse, json, os, re, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--shards", default="0-9", help="parquet shard range of the 41 in 20231101.en, e.g. 0-9")
ap.add_argument("--min-sents", type=int, default=8)
ap.add_argument("--max-sents", type=int, default=256)
ap.add_argument("--min-chars", type=int, default=20, help="shorter pieces (headings, stray tokens) are not sentences")
ap.add_argument("--max-chars", type=int, default=600)
ap.add_argument("--ids-only", type=int, default=0, help="1: write only ids_XXX.npy - the page ids of the documents docs_XXX.jsonl holds, in the same order (same filter), and delete the downloaded shard")
A = ap.parse_args()
import pyarrow.parquet as pq
from blingfire import text_to_sentences
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "wikimedia/wikipedia"
files = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.startswith("20231101.en/") and f.endswith(".parquet"))
a, b = (int(x) for x in A.shards.split("-"))
os.makedirs(A.out, exist_ok=True)
TAIL = re.compile(r"\n(References|See also|External links|Further reading|Notes|Bibliography|Sources|Citations|Footnotes)\s*\n", re.I)
print(f"[prep] {len(files)} shards in the dump, taking {a}-{b}", flush=True)
for si in range(a, b + 1):
    out = os.path.join(A.out, f"ids_{si:03d}.npy" if A.ids_only else f"docs_{si:03d}.jsonl")
    if os.path.exists(out):
        print(f"[prep] shard {si} already written", flush=True); continue
    t0 = time.time()
    path = hf_hub_download(REPO, files[si], repo_type="dataset")
    tbl = pq.read_table(path, columns=["id", "title", "text"])
    nd = ns = 0
    if A.ids_only:
        import numpy as np
        keep = []
        for rid, text in zip(tbl["id"].to_pylist(), tbl["text"].to_pylist()):
            m = TAIL.search(text)
            if m: text = text[:m.start()]
            n = 0
            for para in text.split("\n"):
                para = para.strip()
                if len(para) < A.min_chars: continue
                for s_ in text_to_sentences(para).split("\n"):
                    if A.min_chars <= len(s_.strip()) <= A.max_chars: n += 1
            if n >= A.min_sents: keep.append(int(rid))
        np.save(out, np.array(keep, dtype=np.int64)); os.remove(os.path.realpath(path))
        print(f"[prep] shard {si}: {len(keep)} document ids in {time.time() - t0:.0f}s", flush=True); continue
    with open(out + ".part", "w") as fh:
        for rid, title, text in zip(tbl["id"].to_pylist(), tbl["title"].to_pylist(), tbl["text"].to_pylist()):
            m = TAIL.search(text)
            if m: text = text[:m.start()]
            sents = []
            for para in text.split("\n"):
                para = para.strip()
                if len(para) < A.min_chars: continue          # headings and blank lines
                for s in text_to_sentences(para).split("\n"):
                    s = s.strip()
                    if A.min_chars <= len(s) <= A.max_chars: sents.append(s)
            if len(sents) < A.min_sents: continue
            fh.write(json.dumps({"id": rid, "title": title, "sents": sents[:A.max_sents]}, ensure_ascii=False) + "\n")
            nd += 1; ns += min(len(sents), A.max_sents)
    os.replace(out + ".part", out)
    print(f"[prep] shard {si}: {nd} documents, {ns} sentences ({ns / max(nd, 1):.0f} per document) in {time.time() - t0:.0f}s", flush=True)
print("PREP_DONE", flush=True)
