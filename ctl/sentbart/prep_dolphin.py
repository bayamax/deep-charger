#!/usr/bin/env python3
"""Dolphin conversations as sentence-sequence documents for the context BART (the user, 2026-10-07: Dolphin's context
sequences, the question included as context but kept out of the learning; e2e from the BART before bge was touched).

Source: dolphin-r1's deepseek reasoning set (question -> reasoning -> answer), streamed from the hub. One document per
conversation: the user's question split into sentences (nq of them), then the reasoning's sentences, then the answer's.
"nq" marks the prefix train.py never hides or predicts. Excluded: the questions of dolphin_v1 (the GRPO set, its held-out
hundred inside). Written like prep.py: docs_000.jsonl (training) and docs_001.jsonl (held-out: the 2000 evaluated and
the pool); train_e2e.py runs as on Wikipedia - no stored vectors, off_XXX.npy carries the layout and the text is embedded in the step.

  python3 prep_dolphin.py --out /root/sb/data/dolphin --n-train 50000 --n-eval 22000 --exclude /root/sb/dl/dolphin_v1.jsonl
"""
import argparse, json, os, random, re, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--n-train", type=int, default=50000); ap.add_argument("--n-eval", type=int, default=22000)
ap.add_argument("--exclude", nargs="*", default=[]); ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--min-sents", type=int, default=10, help="reasoning + answer sentences at least"); ap.add_argument("--max-sents", type=int, default=160)
ap.add_argument("--max-q", type=int, default=12, help="question sentences at most (longer questions are skipped)")
ap.add_argument("--url", default="https://huggingface.co/datasets/cognitivecomputations/dolphin-r1/resolve/main/dolphin-r1-reasoning-deepseek.jsonl")
ap.add_argument("--jsonl", default="", help="a local copy instead of streaming --url")
A = ap.parse_args()
from blingfire import text_to_sentences  # noqa: E402


def norm(s): return " ".join(re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split())


def split(text):
    out = []
    for para in re.split(r"\n+", text or ""):   # reasoning text breaks lines at thoughts, list items and headings: each line is a unit
        para = " ".join(para.split())
        if not para: continue
        for s_ in text_to_sentences(para).split("\n"):
            s_ = s_.strip()
            if len(s_) >= 3: out.append(s_[:600])
    return out


seen = set()
for f in A.exclude:
    if not os.path.exists(f): continue
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        if r.get("q"): seen.add(norm(r["q"])[:300])
os.makedirs(A.out, exist_ok=True)
src = open(A.jsonl, "rb") if A.jsonl else urllib.request.urlopen(urllib.request.Request(A.url, headers={"User-Agent": "deep-charger/1.0"}), timeout=120)
want = A.n_train + A.n_eval; rng = random.Random(A.seed)
docs = []; n_read = n_dup = n_short = n_longq = 0
for raw in src:
    n_read += 1
    try: r = json.loads(raw)
    except Exception: continue
    msgs = r.get("messages") or []
    user = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
    if not user or norm(user)[:300] in seen: n_dup += 1; continue
    qs = split(user)
    if not qs or len(qs) > A.max_q: n_longq += 1; continue
    body = split(r.get("reasoning", "")) + split(r.get("answer", ""))
    if len(body) < A.min_sents: n_short += 1; continue
    sents = (qs + body)[:A.max_sents]
    docs.append({"id": f"dr1:{n_read}", "title": "", "sents": sents, "nq": len(qs)})
    if len(docs) % 20000 == 0: print(f"[prep_dolphin] {n_read} read, {len(docs)} kept", flush=True)
    if len(docs) >= want * 2: break          # read past the need, then draw at random for an even mix of the set
rng.shuffle(docs); docs = docs[:want]
ev, tr = docs[:A.n_eval], docs[A.n_eval:]
import numpy as np  # noqa: E402
for name, part in (("000", tr), ("001", ev)):
    with open(os.path.join(A.out, f"docs_{name}.jsonl"), "w") as f:
        for d in part: f.write(json.dumps(d, ensure_ascii=False) + "\n")
    np.save(os.path.join(A.out, f"off_{name}.npy"), np.cumsum([0] + [len(d["sents"]) for d in part]).astype(np.int64))   # the layout train.py reads; no vectors (e2e embeds the text itself)
ls = sorted(len(d["sents"]) for d in docs); lq = sorted(d["nq"] for d in docs)
print(f"PREP_DOLPHIN_DONE {len(tr)} training + {len(ev)} held-out conversations of {n_read} read ({n_dup} excluded, {n_longq} long questions, {n_short} short); "
      f"sentences median {ls[len(ls)//2]} (90% {ls[int(len(ls)*0.9)]}), question sentences median {lq[len(lq)//2]}", flush=True)
