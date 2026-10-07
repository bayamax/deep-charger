#!/usr/bin/env python3
"""Two-hop search questions for GRPO from MuSiQue (the answerable split, train): compositional questions whose answer takes
two Wikipedia lookups - the bridge entity first, then the fact about it ("When was the institute that owned The Collegian
founded?" -> Houston Baptist University -> 1960). The user (2026-10-07) asked whether MuSiQue would serve better than
TriviaQA: it trains the step-by-step search the single-hop sets never need, at the price of stilted questions, so it is
mixed in rather than used alone. Kept: 2-hop, answerable, answer 1-4 words; golds = the answer and its short aliases.
Items {q, gold, golds, hist: [], src: "musique"}; --exclude, --skip/--n, --pool as nq_items.py / tqa_items.py.

  python3 musique_items.py --n 200 --out /root/work/musique_items_1.jsonl --pool /root/work/musique_pool.jsonl --exclude ...
"""
import argparse, json, os, random, re
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200); ap.add_argument("--out", required=True); ap.add_argument("--pool", default="")
ap.add_argument("--skip", type=int, default=0); ap.add_argument("--exclude", nargs="*", default=[]); ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--hops", default="2hop", help="id prefixes kept, comma-separated (2hop, 3hop1, ...)")
ap.add_argument("--jsonl", default="", help="a local copy of musique_ans_v1.0_train.jsonl (else downloaded)")
A = ap.parse_args()


def norm(s): return " ".join(re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split())


def short(a):
    a = " ".join((a or "").split())
    return a if a and 1 <= len(a.split()) <= 4 and len(a) >= 2 and not re.fullmatch(r"\d{1,2}", a) else ""


seen = set()
for f in A.exclude:
    if not os.path.exists(f): continue
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        for q in [r.get("q")] + [t.get("q") for t in (r.get("turns") or [])]:
            if q: seen.add(norm(q))
p = A.jsonl
if not p:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("dgslibisey/MuSiQue", "musique_ans_v1.0_train.jsonl", repo_type="dataset")
hops = set(A.hops.split(","))
kept, dup, long_ = [], 0, 0
for l in open(p):
    if not l.strip(): continue
    r = json.loads(l)
    if r["id"].split("__")[0] not in hops or not r.get("answerable", True): continue
    q = " ".join((r.get("question") or "").split())
    if not q or norm(q) in seen: dup += 1; continue
    golds = []
    for a in [r.get("answer")] + list(r.get("answer_aliases") or []):
        a = short(a)
        if a and norm(a) not in {norm(x) for x in golds}: golds.append(a)
    if not golds or not short(r.get("answer")): long_ += 1; continue
    kept.append({"q": q[0].upper() + q[1:], "gold": golds[0], "golds": golds, "hist": [], "src": "musique"})
random.Random(A.seed).shuffle(kept)
if A.pool: open(A.pool, "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in kept))
sel = kept[A.skip:A.skip + A.n]
open(A.out, "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in sel))
print(f"MUSIQUE_ITEMS_DONE {len(sel)} items (of {len(kept)} kept; {dup} already in an evaluation or training set, {long_} without a short answer) -> {A.out}", flush=True)
