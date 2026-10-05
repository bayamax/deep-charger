#!/usr/bin/env python3
"""Article retrieval from one vector per article: can the LM's state at the end of an article index it?

Questions with a known gold Wikipedia article (BEIR's Natural Questions test set: 3452 questions, each tied to
passages of named articles) against a pool of articles from our prepared shards (the gold ones that are in them plus
random others). For each question the gold article's rank in the pool; top-1/10/100.

  last-L    the LM reads [BOS] + article (first --art-tok tokens); the state of its last token at layer L.
            The question the same way.
  eol-L     the same with a closing prompt after the text (PromptEOL): '...\nThis article in one word: "' and
            '...\nThis question in one word: "' - the next token has to name the subject
  mean-L    the mean of every token's state at layer L
  bge-first bge-small on the article's first 512 tokens (title + opening), the question with bge's query instruction
  bge-max   bge-small on every 256-token chunk of the article, the article scored by its best chunk

LM vectors are centred (articles by the articles' mean, questions by the questions' mean - the raw states share one
dominant direction) and L2-normalised; cosine ranking. L = the final layer and the middle one.

  python3 artret.py --docs /root/lz/docs --shards 000,001,002,003,004,005,006 --pool 30000 --out /root/lz/artret
"""
import argparse, gzip, json, os, random, time
import numpy as np
import torch
ap = argparse.ArgumentParser()
ap.add_argument("--docs", default="/root/lz/docs"); ap.add_argument("--shards", default="000,001,002,003,004,005,006")
ap.add_argument("--pool", type=int, default=30000); ap.add_argument("--out", default="/root/lz/artret")
ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M"); ap.add_argument("--bge", default="BAAI/bge-small-en-v1.5")
ap.add_argument("--art-tok", type=int, default=2048); ap.add_argument("--tok-budget", type=int, default=16384)
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True); DEV = "cuda"
from huggingface_hub import hf_hub_download  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

# ---- the questions and their gold titles ----
POOLF = os.path.join(A.out, "pool.jsonl"); QF = os.path.join(A.out, "queries.jsonl")
if not (os.path.exists(POOLF) and os.path.exists(QF)):
    qr = hf_hub_download("BeIR/nq-qrels", "test.tsv", repo_type="dataset")
    qs = hf_hub_download("BeIR/nq", "queries/queries-00000-of-00001.parquet", repo_type="dataset")
    cp = hf_hub_download("BeIR/nq", "corpus/corpus-00000-of-00001.parquet", repo_type="dataset")
    rel = {}
    for line in open(qr).read().splitlines()[1:]:
        q, c, s = line.split("\t")
        if int(s) > 0: rel.setdefault(q, set()).add(c)
    need = set(c for v in rel.values() for c in v)
    t = pq.read_table(cp, columns=["_id", "title"]); ptitle = {}
    for i, ti in zip(t["_id"].to_pylist(), t["title"].to_pylist()):
        if i in need: ptitle[i] = ti
    del t
    t = pq.read_table(qs); qtext = dict(zip(t["_id"].to_pylist(), t["text"].to_pylist()))
    gold = {q: sorted(set(ptitle[c] for c in cs if c in ptitle)) for q, cs in rel.items()}
    gtitles = set(x for v in gold.values() for x in v)
    # every article of the shards, the gold ones kept, a random sample of the rest
    found, other = {}, []
    for k in A.shards.split(","):
        for line in open(os.path.join(A.docs, f"docs_{k}.jsonl")):
            d = json.loads(line)
            if d["title"] in gtitles: found[d["title"]] = d
            else: other.append(line)
    random.Random(0).shuffle(other)
    pool = list(found.values()) + [json.loads(l) for l in other[:max(0, A.pool - len(found))]]
    del other
    with open(POOLF, "w") as f:
        for d in pool: f.write(json.dumps({"title": d["title"], "text": d["title"] + "\n" + " ".join(d["sents"])}, ensure_ascii=False) + "\n")
    with open(QF, "w") as f:
        for q, g in gold.items():
            g = [x for x in g if x in found]
            if g and q in qtext: f.write(json.dumps({"q": qtext[q], "gold": g}, ensure_ascii=False) + "\n")
pool = [json.loads(l) for l in open(POOLF)]; queries = [json.loads(l) for l in open(QF)]
tidx = {d["title"]: i for i, d in enumerate(pool)}
gold = [[tidx[g] for g in q["gold"]] for q in queries]
print(f"[artret] {len(queries)} questions with a gold article among {len(pool)} pool articles", flush=True)


def scores(name, Q, D, agg=None):
    """rank of the best-ranked gold article per question; agg: chunk -> article index for max-over-chunks"""
    Q = torch.tensor(Q, device=DEV); D = torch.tensor(D, device=DEV); r1 = r10 = r100 = 0
    for i in range(0, len(Q), 256):
        s = Q[i:i + 256] @ D.T
        if agg is not None:
            a = torch.full((s.shape[0], len(pool)), -9.0, device=DEV)
            s = a.scatter_reduce(1, torch.tensor(agg, device=DEV).expand(s.shape[0], -1), s, reduce="amax")
        for j, row in enumerate(s):
            g = gold[i + j]; best = row[g].max()
            rank = int((row > best).sum())
            r1 += rank < 1; r10 += rank < 10; r100 += rank < 100
    n = len(Q); line = f"[artret] {name:10s} top1 {r1/n*100:5.1f}  top10 {r10/n*100:5.1f}  top100 {r100/n*100:5.1f}"
    print(line, flush=True); return line


def norm(v): return v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-8)


res = []
# ---- the LM: last / prompted last / mean, final and middle layer ----
tok = AutoTokenizer.from_pretrained(A.model)
lm = AutoModel.from_pretrained(A.model, torch_dtype=torch.bfloat16).to(DEV).eval()
NL = lm.config.num_hidden_layers; LAYERS = [NL, NL // 2]; BOS = tok.bos_token_id if tok.bos_token_id is not None else 0


@torch.no_grad()
def lm_states(texts, maxtok, suffix):
    suf = tok.encode(suffix, add_special_tokens=False) if suffix else []
    enc = [[BOS] + tok.encode(t, add_special_tokens=False)[:maxtok - 1 - len(suf)] + suf for t in texts]
    order = np.argsort([-len(e) for e in enc]); H = len(LAYERS)
    last = np.zeros((H, len(enc), lm.config.hidden_size), np.float32); mean = np.zeros_like(last)
    i = 0
    while i < len(order):
        L = len(enc[order[i]]); b = max(1, A.tok_budget // L); idx = order[i:i + b]; i += b
        x = torch.zeros((len(idx), L), dtype=torch.long); m = torch.zeros((len(idx), L), dtype=torch.long)
        for r, j in enumerate(idx): x[r, :len(enc[j])] = torch.tensor(enc[j]); m[r, :len(enc[j])] = 1
        hs = lm(input_ids=x.to(DEV), attention_mask=m.to(DEV), output_hidden_states=True).hidden_states
        mm = m.to(DEV).unsqueeze(-1).float(); ln = m.sum(1).to(DEV) - 1
        for h, l in enumerate(LAYERS):
            s = hs[l].float()
            last[h, idx] = s[torch.arange(len(idx), device=DEV), ln].cpu().numpy()
            mean[h, idx] = ((s * mm).sum(1) / mm.sum(1)).cpu().numpy()
    return last, mean


t0 = time.time()
qs_ = [q["q"] for q in queries]; ds_ = [d["text"] for d in pool]
qa, qm = lm_states(qs_, 128, ""); da, dm = lm_states(ds_, A.art_tok, "")
qe, _ = lm_states(qs_, 128, '\nThis question in one word: "'); de, _ = lm_states(ds_, A.art_tok, '\nThis article in one word: "')
print(f"[artret] LM states {(time.time()-t0)/60:.1f} min", flush=True)
for h, l in enumerate(LAYERS):
    for nm, Q, D in (("last", qa, da), ("eol", qe, de), ("mean", qm, dm)):
        res.append(scores(f"{nm}-{l}", norm(Q[h] - Q[h].mean(0)), norm(D[h] - D[h].mean(0))))
        if nm == "eol" and h == 0: res.append(scores(f"eol-{l}raw", norm(Q[h]), norm(D[h])))
del lm; torch.cuda.empty_cache()

# ---- bge-small ----
bt = AutoTokenizer.from_pretrained(A.bge); bm = AutoModel.from_pretrained(A.bge, torch_dtype=torch.float16).to(DEV).eval()


@torch.no_grad()
def bge(texts):
    out = []
    for i in range(0, len(texts), 128):
        e = bt(texts[i:i + 128], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
        out.append(torch.nn.functional.normalize(bm(**e).last_hidden_state[:, 0].float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


t0 = time.time()
bq = bge(["Represent this sentence for searching relevant passages: " + q for q in qs_])
res.append(scores("bge-first", bq, bge(ds_)))
chunks, owner = [], []
for i, d in enumerate(pool):
    ids = bt.encode(d["text"], add_special_tokens=False)
    for s in range(0, max(1, len(ids)), 256):
        chunks.append((d["title"] + "\n" if s else "") + bt.decode(ids[s:s + 256])); owner.append(i)
res.append(scores("bge-max", bq, bge(chunks), agg=owner))
print(f"[artret] bge {(time.time()-t0)/60:.1f} min, {len(chunks)} chunks", flush=True)
open(os.path.join(A.out, "result.txt"), "w").write(f"{len(queries)} questions, {len(pool)} articles\n" + "\n".join(res) + "\n")
print("ARTRET_DONE", flush=True)
