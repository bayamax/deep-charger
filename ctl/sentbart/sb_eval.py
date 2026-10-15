#!/usr/bin/env python3
"""The context BART measured on a fixed yardstick (2026-10-13, the user: the trouble is the evaluation, not the training).
train.py's evaluation names each hidden sentence among every sentence of its evaluation batch, so the candidate count
moves with the documents' lengths (a whole-length batch has more of them), and it only ever sees a document's first 128
sentences. Here, for every checkpoint alike:
  - the held-out documents whole (up to the checkpoint's position table), the tail from a fixed cut hidden, the decoder
    predicting each hidden sentence from the true ones before it (teacher-forced continuation, as dec_top1);
  - every prediction scored against the SAME fixed candidate set: --cands sentences drawn once (seed) from the held-out
    pool, plus the true sentence - so top-1 means the same thing for every model and every document length;
  - the cosine of the prediction to the true vector, which needs no candidates at all;
  - all of it by position band (1-128, 129-256, 257-512) and for the first hidden sentence alone ("next").
Writes <out>.json and prints one table per checkpoint.

  python3 sb_eval.py --docs /root/sb/data/dolphin512full/docs_001.jsonl --ckpt run1=/root/sb/dolphin_once/state_e2e.pt run2b=... --out /root/sb_eval.json
"""
import argparse, json, math, os, random, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", nargs="+", required=True, help="name=path of a model_*.pt / state_e2e.pt (model, enc, enc_layers, layers)")
ap.add_argument("--docs", required=True, help="the held-out shard's text (docs_001.jsonl), whole")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=2000, help="documents evaluated (train.py's draw: the same 2000)"); ap.add_argument("--min-sents", type=int, default=8)
ap.add_argument("--cands", type=int, default=3223, help="fixed distractor candidates, drawn once from the pool (the other held-out documents)")
ap.add_argument("--bands", default="128,256,512"); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--bge", default="BAAI/bge-small-en-v1.5"); ap.add_argument("--seed", type=int, default=5)
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


# ---- the documents: train.py's draw of the 2000 (the layout's order, min_sents, Random(1)), the pool = the rest ----
docs = [json.loads(l) for l in open(X.docs) if l.strip()]
idx = [k for k, d in enumerate(docs) if len(d["sents"]) >= X.min_sents]
random.Random(1).shuffle(idx); ev = idx[:X.n]; pool = [k for k in range(len(docs)) if k not in set(ev)]
rng = random.Random(X.seed)
cand_src = [(k, j) for k in pool for j in range(len(docs[k]["sents"]))]; rng.shuffle(cand_src); cand_src = cand_src[:X.cands]
cand_text = [docs[k]["sents"][j] for k, j in cand_src]
bands = [int(b) for b in X.bands.split(",")]
print(f"[sb_eval] {len(ev)} documents (sentences median {sorted(len(docs[k]['sents']) for k in ev)[len(ev)//2]}, max {max(len(docs[k]['sents']) for k in ev)}), {len(cand_text)} fixed candidates from {len(pool)} pool documents, bands {bands}, device {DEV}", flush=True)

res = {"args": vars(X), "runs": {}}
for spec in X.ckpt:
    name, path = spec.split("=", 1); t0 = time.time()
    CK = torch.load(path, map_location="cpu", weights_only=False)
    rows = CK["model"]["pos.weight"].shape[0]; seq = rows - 1
    model = build(CK["layers"], seq); model.load_state_dict(CK["model"]); model.to(DEV).eval()
    enc = AutoModel.from_pretrained(X.bge); grow_bge(enc, CK.get("enc_layers", 12)); enc.load_state_dict(CK["enc"]); enc.to(DEV).eval()
    C = embed(enc, cand_text)                                              # [N, 384], this checkpoint's own bge
    stat = {}   # band -> [n, cos_sum, top1, top10, z_sum, cand_cos_sum]
    def add(key, cos, r, z=0.0, cm=0.0):
        s = stat.setdefault(key, [0, 0.0, 0, 0, 0.0, 0.0]); s[0] += 1; s[1] += cos; s[2] += int(r == 0); s[3] += int(r < 10); s[4] += z; s[5] += cm
    for i in range(0, len(ev), X.batch):
        part = [docs[k] for k in ev[i:i + X.batch]]
        part = [{"sents": d["sents"][:seq], "nq": int(d.get("nq", 0) or 0)} for d in part]   # whole, up to the checkpoint's table
        flat = [s for d in part for s in d["sents"]]; V = embed(enc, flat); off = np.cumsum([0] + [len(d["sents"]) for d in part])
        Lm = max(len(d["sents"]) for d in part)
        x = torch.zeros(len(part), Lm, 384, device=DEV); valid = torch.zeros(len(part), Lm, dtype=torch.bool, device=DEV); masked = torch.zeros_like(valid); cuts = []
        for b, d in enumerate(part):
            n = len(d["sents"]); x[b, :n] = V[off[b]:off[b + 1]]; valid[b, :n] = True
            k = max(d["nq"] + 1, n // 4, 1); k = min(k, n - 1); masked[b, k:n] = True; cuts.append(k)   # the tail from a fixed cut: a quarter in, past the question
        with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            _, pd = model(x, valid, masked)
        pd = pd.float()
        for b, d in enumerate(part):
            n = len(d["sents"]); k = cuts[b]
            P = F.normalize(pd[b, k:n], dim=-1); T = V[off[b] + k:off[b] + n]          # predictions and truths for the hidden tail
            cos = (P * T).sum(-1)                                                       # [n-k]
            s_true = cos; s_c = P @ C.T                                                 # scores: the truth and the fixed candidates
            rank = (s_c > s_true[:, None]).sum(-1)                                      # how many candidates beat the truth (0 = top-1)
            cm_ = s_c.mean(-1); zz = (s_true - cm_) / s_c.std(-1).clamp_min(1e-6)       # the truth's cosine in standard deviations above the candidates' (comparable with a token LM's)
            for t in range(k, n):
                band = next(bn for bn in bands if t + 1 <= bn) if t + 1 <= bands[-1] else bands[-1]
                a_ = (float(cos[t - k]), int(rank[t - k]), float(zz[t - k]), float(cm_[t - k]))
                add(f"pos<={band}", *a_); add("all", *a_)
                if t == k: add("next", float(cos[0]), int(rank[0]), float(zz[0]), float(cm_[0]))
        if (i // X.batch) % 25 == 0: print(f"[sb_eval] {name}: {min(i + X.batch, len(ev))}/{len(ev)} documents ({(time.time() - t0) / 60:.1f} min)", flush=True)
    run = {k: {"n": v[0], "cos": round(v[1] / v[0], 4), "top1": round(v[2] / v[0], 4), "top10": round(v[3] / v[0], 4), "z": round(v[4] / v[0], 3), "cand_cos": round(v[5] / v[0], 4)} for k, v in stat.items()}
    run["layers"] = CK["layers"]; run["enc_layers"] = CK.get("enc_layers", 12); run["pos_rows"] = rows; run["step"] = CK.get("step")
    res["runs"][name] = run; json.dump(res, open(X.out, "w"), indent=1)
    keys = ["next", "all"] + [f"pos<={b}" for b in bands]
    print(f"[sb_eval] {name} (BART {CK['layers']}+{CK['layers']}, bge {run['enc_layers']}, table {rows}, step {run['step']}) at {len(cand_text)}+1 candidates [{(time.time() - t0) / 60:.1f} min]", flush=True)
    for k in keys:
        if k in run: print(f"  {k:10s} n {run[k]['n']:6d}  cos {run[k]['cos']:.3f}  top1 {run[k]['top1']:.3f}  top10 {run[k]['top10']:.3f}  z {run[k]['z']:.2f}  cand_cos {run[k]['cand_cos']:.3f}", flush=True)
    del model, enc, C, CK
    if DEV == "cuda": torch.cuda.empty_cache()
print("SB_EVAL_DONE", flush=True)
