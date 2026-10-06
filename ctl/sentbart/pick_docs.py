#!/usr/bin/env python3
"""The documents the query writer (qgen.py) gets: a passage from each, with the article's title.

Two sets, one file: --n training documents drawn evenly from the training shards' text (docs_XXX.jsonl, not 001), and
the 2000 held-out documents train.py evaluates on (shard 001; the same draw: documents of >= 8 sentences, shuffled by
random.Random(1), the first 2000 - from off_001.npy, no torch needed), so the query -> article retrieval can be
measured on the pool as well. Each row: {id: "<shard>:<line>", title, s0, s1, passage, nsents} - the passage is up
to three consecutive sentences starting anywhere in the article (deep facts as often as the lead).

  python3 pick_docs.py --vec /root/sb/data/docs --text /root/sb/data/e2e --eval-text /root/sb/data/docs/docs_001.jsonl --n 20000 --out /root/sb/e2e2/docs_for_qgen.jsonl
"""
import argparse, glob, json, os, random
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--vec", required=True); ap.add_argument("--text", required=True); ap.add_argument("--eval-text", required=True)
ap.add_argument("--n", type=int, default=20000); ap.add_argument("--out", required=True)
ap.add_argument("--min-sents", type=int, default=8); ap.add_argument("--seed", type=int, default=3)
A = ap.parse_args()
os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True)
rng = random.Random(A.seed)


def passage(sents):
    s0 = rng.randint(0, len(sents) - 1); s1 = min(len(sents), s0 + 3)
    return s0, s1, " ".join(sents[s0:s1])


rows = []
# the 2000 evaluation documents, as train.py draws them
eoff = np.load(os.path.join(A.vec, "off_001.npy"))
edocs = [(int(eoff[d]), int(eoff[d + 1]), d) for d in range(len(eoff) - 1) if eoff[d + 1] - eoff[d] >= A.min_sents]
random.Random(1).shuffle(edocs); edocs = edocs[:2000]
want = {d for _, _, d in edocs}
for ln, line in enumerate(open(A.eval_text)):
    if ln in want:
        d = json.loads(line); s0, s1, p = passage(d["sents"])
        rows.append({"id": f"001:{ln}", "title": d.get("title", ""), "s0": s0, "s1": s1, "passage": p, "nsents": len(d["sents"])})
assert len(rows) == len(edocs), (len(rows), len(edocs))
ne = len(rows)
# training documents, evenly over the shards
files = [p for p in sorted(glob.glob(os.path.join(A.text, "docs_*.jsonl"))) if not p.endswith("docs_001.jsonl")]
per = A.n // max(1, len(files))
for p in files:
    key = os.path.basename(p)[5:-6]
    n_lines = sum(1 for _ in open(p, "rb")); pick = set(rng.sample(range(n_lines), min(per * 2, n_lines))); got = 0
    for ln, line in enumerate(open(p)):
        if ln not in pick: continue
        d = json.loads(line)
        if len(d["sents"]) < A.min_sents: continue
        s0, s1, ps = passage(d["sents"])
        rows.append({"id": f"{key}:{ln}", "title": d.get("title", ""), "s0": s0, "s1": s1, "passage": ps, "nsents": len(d["sents"])}); got += 1
        if got >= per: break
open(A.out, "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
print(f"PICK_DONE {ne} evaluation documents + {len(rows) - ne} training documents -> {A.out}", flush=True)
