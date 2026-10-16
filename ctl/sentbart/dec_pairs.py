#!/usr/bin/env python3
"""The context BART's predictions, paired with the sentences they stand for - the decoder's training data.

For each conversation the question (its first nq sentences) is read and never hidden; every sentence after it is hidden
and the BART decoder predicts it from the true sentences before it (teacher-forced, as in training and in sb_eval.py).
Each answer sentence t gives one pair:
    pred  - the BART's vector for sentence t (what the next sentence should be, from the question and sentences < t)
    true  - sentence t's own vector, from the BART checkpoint's own (trained) bge
    text  - sentence t;  prev - sentence t-1 (the raw text a decoder may also be shown)
    ctx   - with --ctx-chars N: the last N characters of everything before sentence t (question included), the raw
            window a decoder may be shown
Written as <out>/pairs_<name>.npz (pred, true: float16 [N, 384]) and pairs_<name>.jsonl (doc, t, nq, text, prev).

  python3 dec_pairs.py --ckpt once7/model_best.pt --docs data/am_dec/docs_007.jsonl --name train --out dec --max-docs 30000
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--docs", required=True); ap.add_argument("--out", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--max-docs", type=int, default=0); ap.add_argument("--skip-docs", type=int, default=0); ap.add_argument("--max-t", type=int, default=256, help="answer sentences per conversation at most")
ap.add_argument("--ctx-chars", type=int, default=0); ap.add_argument("--min-sents", type=int, default=4); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--bge", default="BAAI/bge-small-en-v1.5")
X = ap.parse_args()
DEV = os.environ.get("SB_DEV", "cuda" if torch.cuda.is_available() else "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "train.py")).read().split("\n")
a0 = next(i for i, l in enumerate(src) if l.startswith("ap = argparse")); a1 = next(i for i, l in enumerate(src) if l.startswith("A = ap.parse_args()"))
c0 = next(i for i, l in enumerate(src) if l.startswith("class SentBART")); c1 = next(i for i, l in enumerate(src) if l.startswith("def nce("))


def build(layers, seq):
    ns = {"argparse": argparse, "torch": torch, "nn": nn, "F": F, "np": np, "math": math, "DIM": 384}
    argv = sys.argv; sys.argv = ["train.py", "--data", "x", "--out", "x", "--seq", str(seq), "--d", "512", "--layers", str(layers), "--heads", "8", "--ffn", "2048", "--page", "1", "--page-input", "one"]
    exec("\n".join(src[a0:a1 + 1]), ns); sys.argv = argv
    exec("\n".join(src[c0:c1]), ns)
    return ns["SentBART"]()


def grow_bge(m, n):
    import copy
    L = m.encoder.layer
    while len(L) < n:
        j = len(L) // 2; L.insert(j + 1, copy.deepcopy(L[j]))
    m.config.num_hidden_layers = len(L)


btok = AutoTokenizer.from_pretrained(X.bge)


@torch.no_grad()
def embed(enc, sents, chunk=256):
    order = sorted(range(len(sents)), key=lambda i: len(sents[i])); parts = []
    for i in range(0, len(order), chunk):
        b = btok([sents[j] for j in order[i:i + chunk]], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEV)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            parts.append(F.normalize(enc(**b).last_hidden_state[:, 0].float(), dim=-1))
    out = torch.cat(parts); inv = torch.empty(len(order), dtype=torch.long, device=DEV); inv[torch.tensor(order, device=DEV)] = torch.arange(len(order), device=DEV)
    return out[inv]


CK = torch.load(X.ckpt, map_location="cpu", weights_only=False)
rows = CK["model"]["pos.weight"].shape[0]; seq = rows - 1
model = build(CK["layers"], seq); model.load_state_dict(CK["model"]); model.to(DEV).eval()
enc = AutoModel.from_pretrained(X.bge); grow_bge(enc, CK.get("enc_layers", 12)); enc.load_state_dict(CK["enc"]); enc.to(DEV).eval()

docs = []
with open(X.docs) as f:
    for i, l in enumerate(f):
        if i < X.skip_docs or not l.strip(): continue
        d = json.loads(l); d["sents"] = d["sents"][:seq]
        if len(d["sents"]) >= X.min_sents and int(d.get("nq", 0) or 0) < len(d["sents"]) - 1: docs.append(d)
        if X.max_docs and len(docs) >= X.max_docs: break
print(f"[pairs] {X.name}: {len(docs)} conversations from {X.docs}; BART {CK['layers']}+{CK['layers']} step {CK.get('step')}, table {rows}, {DEV}", flush=True)

os.makedirs(X.out, exist_ok=True); PRED, TRUE = [], []; meta = open(os.path.join(X.out, f"pairs_{X.name}.jsonl"), "w"); t0 = time.time(); cs = []
for i in range(0, len(docs), X.batch):
    part = docs[i:i + X.batch]
    flat = [s for d in part for s in d["sents"]]; V = embed(enc, flat); off = np.cumsum([0] + [len(d["sents"]) for d in part])
    Lm = max(len(d["sents"]) for d in part)
    x = torch.zeros(len(part), Lm, 384, device=DEV); valid = torch.zeros(len(part), Lm, dtype=torch.bool, device=DEV); masked = torch.zeros_like(valid); ks = []
    for b, d in enumerate(part):
        n = len(d["sents"]); x[b, :n] = V[off[b]:off[b + 1]]; valid[b, :n] = True
        k = max(1, int(d.get("nq", 0) or 0)); masked[b, k:n] = True; ks.append(k)   # the question read, everything after it hidden
    with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
        _, pd = model(x, valid, masked)
    pd = F.normalize(pd.float(), dim=-1)
    for b, d in enumerate(part):
        n = len(d["sents"]); k = ks[b]; e = min(n, k + X.max_t)
        P = pd[b, k:e]; T = V[off[b] + k:off[b] + e]; cs.append(float((P * T).sum(-1).mean()))
        PRED.append(P.half().cpu().numpy()); TRUE.append(T.half().cpu().numpy())
        for t in range(k, e):
            m_ = {"doc": d.get("id", i + b), "t": t, "nq": k, "text": d["sents"][t], "prev": d["sents"][t - 1]}
            if X.ctx_chars: m_["ctx"] = " ".join(d["sents"][max(0, t - 64):t])[-X.ctx_chars:]
            meta.write(json.dumps(m_) + "\n")
    if (i // X.batch) % 100 == 0: print(f"[pairs] {X.name}: {min(i + X.batch, len(docs))}/{len(docs)} conversations, cos(pred, true) so far {np.mean(cs):.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
meta.close()
PRED, TRUE = np.concatenate(PRED), np.concatenate(TRUE)
np.savez(os.path.join(X.out, f"pairs_{X.name}.npz"), pred=PRED, true=TRUE)
print(f"[pairs] {X.name}: {len(PRED)} pairs, mean cos(pred, true) {float((PRED.astype(np.float32) * TRUE.astype(np.float32)).sum(-1).mean()):.3f} [{(time.time() - t0) / 60:.1f} min]", flush=True)
print("DEC_PAIRS_DONE", flush=True)
