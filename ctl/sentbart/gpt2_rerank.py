#!/usr/bin/env python3
"""How far does the context BART have to go when GPT-2 writes each sentence? (2026-10-10, the user's decoding plan: the
BART's predicted sentence vector goes into GPT-2 through an MLP as a few soft tokens, with the text written so far as a raw
prefix, and GPT-2 writes the sentence that vector stands for - GPT-2's fluency on the BART's plan.) Nothing is trained
here; it is a selection proxy, run on CPU: for held-out Dolphin documents the BART predicts a tail sentence's vector from
the true sentences before it (as the dec_top1 evaluation does), its K nearest candidates among every sentence of a
32-document group are scored by GPT-2 given the previous sentences' text, and the two scores are combined (weights fitted
on one half of the documents, measured on the other, both ways). The BART's own quality is swept by moving its prediction
toward the true vector (--lams), so the curve says what combined top-1 a given BART top-1 buys.

  python3 gpt2_rerank.py --ckpt r03=cap_bge14_bart16/model_best.pt --docs docs_001.jsonl --out res.json
"""
import argparse, copy, json, math, os, random, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", nargs="+", required=True, help="name=path of model_best.pt (train_e2e.py's: model, enc, enc_layers, layers)")
ap.add_argument("--docs", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=320, help="documents (a multiple of --group)"); ap.add_argument("--group", type=int, default=32)
ap.add_argument("--k", type=int, default=50); ap.add_argument("--lams", default="0,0.1,0.2,0.3,0.4,0.6")
ap.add_argument("--gpt2", default="openai-community/gpt2"); ap.add_argument("--ctx-tokens", type=int, default=384); ap.add_argument("--cand-tokens", type=int, default=64)
ap.add_argument("--bge", default="BAAI/bge-small-en-v1.5"); ap.add_argument("--seed", type=int, default=5)
ap.add_argument("--threads", type=int, default=4)
X = ap.parse_args()
torch.set_num_threads(X.threads); torch.manual_seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- train.py's arguments and model class, as train_e2e.py sets them ----
src = open(os.path.join(HERE, "train.py")).read().split("\n")
a0 = next(i for i, l in enumerate(src) if l.startswith("ap = argparse")); a1 = next(i for i, l in enumerate(src) if l.startswith("A = ap.parse_args()"))
c0 = next(i for i, l in enumerate(src) if l.startswith("class SentBART")); c1 = next(i for i, l in enumerate(src) if l.startswith("def nce("))


def build(layers):
    ns = {"argparse": argparse, "torch": torch, "nn": nn, "F": F, "np": np, "math": math, "DIM": 384}
    argv = sys.argv; sys.argv = ["train.py", "--data", "x", "--out", "x", "--seq", "128", "--d", "512", "--layers", str(layers), "--heads", "8", "--ffn", "2048", "--page", "1", "--page-input", "one"]
    exec("\n".join(src[a0:a1 + 1]), ns); sys.argv = argv
    exec("\n".join(src[c0:c1]), ns)
    return ns["SentBART"](), ns["A"]


def grow_bge(m, n):
    import copy
    L = m.encoder.layer
    while len(L) < n:
        j = len(L) // 2; L.insert(j + 1, copy.deepcopy(L[j]))
    m.config.num_hidden_layers = len(L)


btok = AutoTokenizer.from_pretrained(X.bge)


@torch.no_grad()
def embed(enc, sents, chunk=128):
    order = sorted(range(len(sents)), key=lambda i: len(sents[i])); parts = []
    for i in range(0, len(order), chunk):
        b = btok([sents[j] for j in order[i:i + chunk]], padding=True, truncation=True, max_length=128, return_tensors="pt")
        parts.append(F.normalize(enc(**b).last_hidden_state[:, 0].float(), dim=-1))
    out = torch.cat(parts); inv = torch.empty(len(order), dtype=torch.long); inv[torch.tensor(order)] = torch.arange(len(order))
    return out[inv]


# ---- the documents and one target per document: a tail sentence, the cut a quarter to three quarters in ----
docs = []
for l in open(X.docs):
    if not l.strip(): continue
    d = json.loads(l); s = d["sents"][:128]; nq = int(d.get("nq", 0) or 0)
    if len(s) >= 12 and nq + 4 < len(s): docs.append((s, nq))
rng = random.Random(X.seed); rng.shuffle(docs); docs = docs[:X.n]
items = []
for s, nq in docs:
    L = len(s); c = rng.randint(max(nq + 1, L // 4), max(nq + 1, 3 * L // 4)); t = rng.randint(c, min(L - 1, c + 7))
    items.append({"sents": s, "nq": nq, "cut": c, "t": t})
print(f"[data] {len(items)} documents, median {sorted(len(it['sents']) for it in items)[len(items)//2]} sentences", flush=True)

# ---- GPT-2: log p(candidate + newline | the previous sentences' text), cached per (document, candidate text) ----
gtok = AutoTokenizer.from_pretrained(X.gpt2); gpt = AutoModelForCausalLM.from_pretrained(X.gpt2).eval()
NL = gtok.encode("\n"); LPC = {}


@torch.no_grad()
def gpt_scores(di, cands):
    it = items[di]; need = [c for c in cands if (di, c) not in LPC]
    if need:
        ctx = gtok.encode("\n".join(it["sents"][:it["t"]]) + "\n")[-X.ctx_tokens:]
        out = gpt(torch.tensor([ctx]), use_cache=True); past = out.past_key_values; last = out.logits[0, -1].log_softmax(-1)
        for i in range(0, len(need), 25):
            part = need[i:i + 25]; toks = [gtok.encode(c)[:X.cand_tokens] + NL for c in part]; M = max(len(x) for x in toks)
            ids = torch.full((len(part), M), gtok.eos_token_id); att = torch.zeros(len(part), len(ctx) + M, dtype=torch.long); att[:, :len(ctx)] = 1
            for r, x in enumerate(toks): ids[r, :len(x)] = torch.tensor(x); att[r, len(ctx):len(ctx) + len(x)] = 1
            pk = copy.deepcopy(past); pk.batch_repeat_interleave(len(part))   # the context's cache, once per candidate
            lg = gpt(ids, past_key_values=pk, attention_mask=att, use_cache=False).logits.log_softmax(-1)
            for r, x in enumerate(toks):
                lp = float(last[x[0]]) + sum(float(lg[r, j - 1, x[j]]) for j in range(1, len(x)))
                LPC[(di, part[r])] = (lp, len(x))
    return [LPC[(di, c)] for c in cands]


res = {"args": vars(X), "runs": {}}
lams = [float(v) for v in X.lams.split(",")]
for spec in X.ckpt:
    name, path = spec.split("=", 1); t0 = time.time()
    CK = torch.load(path, map_location="cpu", weights_only=False)
    model, A = build(CK["layers"]); miss = model.load_state_dict(CK["model"], strict=False); model.eval()
    enc = AutoModel.from_pretrained(X.bge); grow_bge(enc, CK.get("enc_layers", 12)); em = enc.load_state_dict(CK["enc"], strict=False); enc.eval()
    npar = sum(p.numel() for p in model.parameters()) / 1e6; nenc = sum(p.numel() for p in enc.parameters()) / 1e6
    print(f"[{name}] BART {npar:.1f}M ({CK['layers']}+{CK['layers']} layers), bge {nenc:.1f}M ({CK.get('enc_layers', 12)} layers); missing {len(miss.missing_keys)}/{len(em.missing_keys)}, unexpected {len(miss.unexpected_keys)}/{len(em.unexpected_keys)}", flush=True)
    rows = []   # per document: target index in the group's pool, the pool's texts, BART's prediction, the true vector, the seen mask
    for g in range(0, len(items), X.group):
        grp = items[g:g + X.group]; flat = [s for it in grp for s in it["sents"]]
        V = embed(enc, flat); off = np.cumsum([0] + [len(it["sents"]) for it in grp])
        Lm = max(len(it["sents"]) for it in grp); x = torch.zeros(len(grp), Lm, 384)
        valid = torch.zeros(len(grp), Lm, dtype=torch.bool); masked = torch.zeros_like(valid)
        for b, it in enumerate(grp):
            n = len(it["sents"]); x[b, :n] = V[off[b]:off[b + 1]]; valid[b, :n] = True; masked[b, it["cut"]:n] = True
        with torch.no_grad(): _, pd = model(x, valid, masked)
        for b, it in enumerate(grp):
            seen = torch.zeros(len(flat), dtype=torch.bool); seen[off[b]:off[b] + it["t"]] = True
            rows.append({"di": g + b, "tgt": int(off[b] + it["t"]), "pred": pd[b, it["t"]].float(), "true": V[off[b] + it["t"]], "V": V, "texts": flat, "seen": seen})
        print(f"[{name}] group {g // X.group + 1}/{math.ceil(len(items) / X.group)} embedded and predicted ({(time.time() - t0) / 60:.1f} min)", flush=True)
    run = {"bart_mparams": round(npar, 1), "bge_mparams": round(nenc, 1), "lams": {}}
    for lam in lams:
        feats = []   # per document: (target position in its top-K or -1, [(cos, lp, ntok)] for the top-K)
        b1 = 0
        for r in rows:
            p = F.normalize((1 - lam) * r["pred"] + lam * r["true"], dim=-1)
            sc = (r["V"] @ p).masked_fill(r["seen"], -1e9)
            top = sc.topk(X.k).indices.tolist(); b1 += top[0] == r["tgt"]
            gs = gpt_scores(r["di"], [r["texts"][j] for j in top])
            feats.append((top.index(r["tgt"]) if r["tgt"] in top else -1, [(float(sc[j]), lp, n) for j, (lp, n) in zip(top, gs)]))
        N = len(feats); rec = sum(f[0] >= 0 for f in feats) / N
        gpt1 = sum(f[0] >= 0 and max(range(X.k), key=lambda i: f[1][i][1]) == f[0] for f in feats) / N

        def acc(fs, tau, w):
            return sum(f[0] >= 0 and max(range(len(f[1])), key=lambda i: f[1][i][0] / tau + w * f[1][i][1]) == f[0] for f in fs)
        grid = [(tau, w) for tau in (0.01, 0.02, 0.03, 0.05, 0.08) for w in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)]
        half = N // 2; A_, B_ = feats[:half], feats[half:]
        bestA = max(grid, key=lambda g_: acc(A_, *g_)); bestB = max(grid, key=lambda g_: acc(B_, *g_))
        comb = (acc(B_, *bestA) + acc(A_, *bestB)) / N
        run["lams"][str(lam)] = {"bart_top1": round(b1 / N, 3), f"bart_recall@{X.k}": round(rec, 3), "gpt2_only_in_topK": round(gpt1, 3), "combined_top1": round(comb, 3), "fit": [bestA, bestB],
                                 "cos_pred_true": round(float(np.mean([float(F.normalize((1 - lam) * r["pred"] + lam * r["true"], dim=-1) @ r["true"]) for r in rows])), 3)}
        print(f"[{name}] lam {lam}: BART top-1 {b1 / N:.3f}, recall@{X.k} {rec:.3f}, GPT-2 alone among them {gpt1:.3f}, combined {comb:.3f} (cos to true {run['lams'][str(lam)]['cos_pred_true']}) "
              f"[{(time.time() - t0) / 60:.1f} min, {len(LPC)} GPT-2 scores cached]", flush=True)
    res["runs"][name] = run
    json.dump(res, open(X.out, "w"), indent=1)
    del rows, model, enc, CK
print("RERANK_DONE", flush=True)
