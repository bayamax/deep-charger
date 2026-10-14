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
ap.add_argument("--keep-eval", default="", help="an existing docs_001.jsonl: kept as the held-out set unchanged (same 2000 evaluated, same pool), its conversations excluded from the training set - the numbers stay comparable")
ap.add_argument("--rewrite-eval", default="", help="an existing docs_001.jsonl: the same conversations, in the same order, written again whole up to --max-sents as docs_001.jsonl (+ off_001.npy) in --out - the held-out set at full length, the same 2000 evaluated (2026-10-12: how the long-window training reads on long contexts)")
ap.add_argument("--id-prefix", default="dr1", help="ids are <prefix>:<line of the source file>; another corpus takes another prefix (fl1 for the flash set)")
ap.add_argument("--shard", default="000", help="--all writes docs_<shard>.jsonl / off_<shard>.npy (another corpus takes another shard, e.g. 002)")
ap.add_argument("--exclude-questions", nargs="*", default=[], help="docs_*.jsonl files whose conversations' questions (the first nq sentences) are kept out, by text - the held-out set against another corpus that may share prompts")
ap.add_argument("--schema", default="dolphin", choices=["dolphin", "am"], help="am: AM-DeepSeek-R1-Distilled rows (messages[user].content, messages[assistant].info.think_content / answer_content); rows with CJK text are skipped")
ap.add_argument("--skip-rows", type=int, default=0, help="--all: rows of the source skipped before any is taken (a held-out set from the part of a corpus a training set did not reach)")
ap.add_argument("--stride", type=int, default=1, help="--all: every k-th qualifying conversation is taken (a held-out spread over a corpus whose rows come grouped by source)")
ap.add_argument("--max-docs", type=int, default=0, help="--all: stop after this many kept conversations (0: the whole corpus)")
ap.add_argument("--all", type=int, default=0, help="1: every qualifying conversation goes to the training set (2026-10-11, the user: the whole corpus, one epoch; 50k had been gone round ~45 times)")
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


def rows(path_or_url):
    """the corpus lines, a .zst file decompressed as it is read"""
    if path_or_url and path_or_url.endswith(".zst"):
        import io, zstandard
        return io.BufferedReader(zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(open(path_or_url, "rb")))
    return open(path_or_url, "rb") if A.jsonl else urllib.request.urlopen(urllib.request.Request(path_or_url, headers={"User-Agent": "deep-charger/1.0"}), timeout=120)


CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


SRC = {}   # --schema am: the source of each kept conversation, counted


def parts(r):
    """(question, reasoning, answer) of one row under --schema; None when the row is not usable"""
    msgs = r.get("messages") or []
    if A.schema == "am":
        u = next((m for m in msgs if m.get("role") == "user"), None); a_ = next((m for m in msgs if m.get("role") == "assistant"), None)
        if not u or not a_: return None
        info = a_.get("info") or {}; q = u.get("content", ""); th = info.get("think_content") or ""; an = info.get("answer_content") or ""
        if not th or CJK.search(q) or CJK.search(th[:3000]): return None
        SRC["_last"] = (u.get("info") or {}).get("source") or "none"
        return q, th, an
    user = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
    return user, r.get("reasoning", ""), r.get("answer", "")


seen = set()
for f in A.exclude:
    if not os.path.exists(f): continue
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        if r.get("q"): seen.add(norm(r["q"])[:300])
os.makedirs(A.out, exist_ok=True)
import numpy as np  # noqa: E402
if A.rewrite_eval:
    order = [json.loads(l)["id"] for l in open(A.rewrite_eval) if l.strip()]; want_ids = set(order); got = {}
    src = rows(A.jsonl or A.url)
    n_read = 0
    for raw in src:
        n_read += 1
        if f"{A.id_prefix}:{n_read}" not in want_ids: continue
        r = json.loads(raw); user, reasoning, answer = parts(r) or ("", "", "")
        qs = split(user); body = split(reasoning) + split(answer)
        got[f"{A.id_prefix}:{n_read}"] = {"id": f"{A.id_prefix}:{n_read}", "title": "", "sents": (qs + body)[:A.max_sents], "nq": len(qs)}
        if len(got) == len(want_ids): break
    miss = [i for i in order if i not in got]
    if miss: raise SystemExit(f"PREP_DOLPHIN_FAILED {len(miss)} held-out conversations not found in the corpus (first {miss[:3]})")
    with open(os.path.join(A.out, "docs_001.jsonl"), "w") as f:
        for i in order: f.write(json.dumps(got[i], ensure_ascii=False) + "\n")
    np.save(os.path.join(A.out, "off_001.npy"), np.cumsum([0] + [len(got[i]["sents"]) for i in order]).astype(np.int64))
    ls = sorted(len(got[i]["sents"]) for i in order)
    print(f"PREP_DOLPHIN_DONE held-out rewritten whole: {len(order)} conversations, sentences median {ls[len(ls)//2]} (90% {ls[int(len(ls)*0.9)]}, max {ls[-1]}) -> {A.out}", flush=True)
    raise SystemExit(0)
keep_ids = set()
for f in A.exclude_questions:
    for l in open(f):
        if not l.strip(): continue
        d = json.loads(l); nq_ = int(d.get("nq", 0) or 0)
        if nq_: seen.add(norm(" ".join(d["sents"][:nq_]))[:300])
if A.keep_eval:   # the held-out set stays: its ids (dr1:<line of the source>) and questions are out of the training set
    for l in open(A.keep_eval):
        if l.strip(): d = json.loads(l); keep_ids.add(d["id"]); seen.add(norm(d["sents"][0] if d["sents"] else "")[:300])
src = rows(A.jsonl or A.url)
want = A.n_train + A.n_eval; rng = random.Random(A.seed)
docs = []; n_read = n_dup = n_short = n_longq = n_kept = n_unus = 0; lens = []; nqs = []
out_all = open(os.path.join(A.out, f"docs_{A.shard}.jsonl"), "w") if A.all else None   # --all: written as read (no shuffle in memory; train_e2e --once shuffles)
for raw in src:
    n_read += 1
    if n_read <= A.skip_rows: continue
    try: r = json.loads(raw)
    except Exception: continue
    pr = parts(r)
    if pr is None: n_unus += 1; continue
    user, reasoning, answer = pr
    if f"{A.id_prefix}:{n_read}" in keep_ids: continue
    if not user or norm(user)[:300] in seen: n_dup += 1; continue
    qs = split(user)
    if not qs or len(qs) > A.max_q: n_longq += 1; continue
    body = split(reasoning) + split(answer)
    if len(body) < A.min_sents: n_short += 1; continue
    n_qual = globals().get("n_qual", 0) + 1; globals()["n_qual"] = n_qual
    if A.stride > 1 and n_qual % A.stride: continue
    sents = (qs + body)[:A.max_sents]
    d = {"id": f"{A.id_prefix}:{n_read}", "title": "", "sents": sents, "nq": len(qs)}
    if A.schema == "am": SRC[SRC["_last"]] = SRC.get(SRC["_last"], 0) + 1
    if out_all is not None:
        out_all.write(json.dumps(d, ensure_ascii=False) + "\n"); lens.append(len(sents)); nqs.append(len(qs)); n_kept += 1
        if n_kept % 20000 == 0: print(f"[prep_dolphin] {n_read} read, {n_kept} kept", flush=True)
        if A.max_docs and n_kept >= A.max_docs: break
        continue
    docs.append(d)
    if len(docs) % 20000 == 0: print(f"[prep_dolphin] {n_read} read, {len(docs)} kept", flush=True)
    if len(docs) >= want * 2: break          # read past the need, then draw at random for an even mix of the set
if out_all is not None:
    out_all.close()
    np.save(os.path.join(A.out, f"off_{A.shard}.npy"), np.cumsum([0] + lens).astype(np.int64))
    if not A.keep_eval: raise SystemExit("--all needs --keep-eval (the held-out set is not drawn here)")
    ls = sorted(lens); lq = sorted(nqs)
    print(f"PREP_DOLPHIN_DONE {n_kept} training conversations ({'the first ' + str(A.max_docs) + ' kept' if A.max_docs and n_kept >= A.max_docs else 'all of the corpus'}) of {n_read} read ({n_dup} excluded, {n_longq} long questions, {n_short} short, {n_unus} unusable, {len(keep_ids)} held out); "
          f"sentences median {ls[len(ls)//2]} (90% {ls[int(len(ls)*0.9)]}), question sentences median {lq[len(lq)//2]}", flush=True)
    if A.schema == "am": print("PREP_DOLPHIN_SOURCES " + ", ".join(f"{k} {v}" for k, v in sorted(((k, v) for k, v in SRC.items() if k != "_last"), key=lambda kv: -kv[1])), flush=True)
    raise SystemExit(0)
rng.shuffle(docs); docs = docs[:want]
ev, tr = docs[:A.n_eval], docs[A.n_eval:]
for name, part in (("000", tr), ("001", ev)):
    with open(os.path.join(A.out, f"docs_{name}.jsonl"), "w") as f:
        for d in part: f.write(json.dumps(d, ensure_ascii=False) + "\n")
    np.save(os.path.join(A.out, f"off_{name}.npy"), np.cumsum([0] + [len(d["sents"]) for d in part]).astype(np.int64))   # the layout train.py reads; no vectors (e2e embeds the text itself)
ls = sorted(len(d["sents"]) for d in docs); lq = sorted(d["nq"] for d in docs)
print(f"PREP_DOLPHIN_DONE {len(tr)} training + {len(ev)} held-out conversations of {n_read} read ({n_dup} excluded, {n_longq} long questions, {n_short} short); "
      f"sentences median {ls[len(ls)//2]} (90% {ls[int(len(ls)*0.9)]}), question sentences median {lq[len(lq)//2]}", flush=True)
