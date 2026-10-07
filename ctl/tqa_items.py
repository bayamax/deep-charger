#!/usr/bin/env python3
"""Fresh search questions for GRPO from TriviaQA (the Wikipedia-verified subset, train split): trivia questions whose
answer is a Wikipedia entity, with the answer's canonical form and its Wikipedia title as accepted forms.

The user (2026-10-07): a good-quality corpus - nq_open's golds were often wrong or stale (a quarter of the all-miss
questions), so a right answer was scored 0. TriviaQA's answers are curated entities with aliases; kept here are the
canonical value and the matched Wikipedia entity name (the long alias lists hold junk like "The weather in York"), each
1-4 words. Items: {q, gold (shown to the teachers), golds (any of these in the reply counts), hist: []}; questions already
in an evaluation or training set (--exclude, by normalised text) are dropped; shuffled with a fixed seed; --skip/--n pick
the round's block; --pool keeps every kept question for later rounds.

  python3 tqa_items.py --n 600 --out /root/work/tqa_items_1.jsonl --pool /root/work/tqa_pool.jsonl --exclude /root/work/eval300.jsonl ...
"""
import argparse, json, os, random, re
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=600); ap.add_argument("--out", required=True); ap.add_argument("--pool", default="")
ap.add_argument("--skip", type=int, default=0); ap.add_argument("--exclude", nargs="*", default=[]); ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--parquet", default="", help="a local copy of rc.wikipedia.nocontext/train-00000-of-00001.parquet (else downloaded)")
A = ap.parse_args()
import pyarrow.parquet as pq  # noqa: E402


def norm(s): return " ".join(re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split())


def short(a):
    a = (a or "").strip()
    return a if a and 1 <= len(a.split()) <= 4 and len(a) >= 2 and not re.fullmatch(r"\d{1,2}", a) else ""


seen = set()
for f in A.exclude:
    if not os.path.exists(f): continue
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        for q in [r.get("q")] + [t.get("q") for t in (r.get("turns") or [])]:
            if q: seen.add(norm(q))
p = A.parquet
if not p:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext/train-00000-of-00001.parquet", repo_type="dataset")
t = pq.read_table(p, columns=["question", "answer"]).to_pydict()
kept, dup, long_ = [], 0, 0
for q, ans in zip(t["question"], t["answer"]):
    q = (q or "").strip()
    if not q or norm(q) in seen: dup += 1; continue
    golds = []
    v = (ans.get("value") or "").strip()
    cands = [v, ans.get("matched_wiki_entity_name") or ""]
    if "(" in v:   # "(Robert E.) LEE", "Argentina (1978)": the bracketed part spelled out, and left out
        cands = [re.sub(r"[()]", "", v), re.sub(r"\s*\([^)]*\)", "", v), ans.get("matched_wiki_entity_name") or ""]
    for a in cands:
        a = short(" ".join(a.split()))
        if a and norm(a) not in {norm(x) for x in golds}: golds.append(a)
    if not golds: long_ += 1; continue
    kept.append({"q": q, "gold": golds[0], "golds": golds, "hist": []})
random.Random(A.seed).shuffle(kept)
if A.pool: open(A.pool, "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
sel = kept[A.skip:A.skip + A.n]
open(A.out, "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sel))
print(f"TQA_ITEMS_DONE {len(sel)} items (of {len(kept)} kept; {dup} already in an evaluation or training set, {long_} without a short answer) -> {A.out}", flush=True)
