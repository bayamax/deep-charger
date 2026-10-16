#!/usr/bin/env python3
"""The context BART and its GPT-2 decoder trained end to end (2026-10-16, the user: with the BART frozen the decoder
is capped by the BART's vector - make the vector carry what the decoder needs).

Per update: --docs conversations, each cut to its first --window sentences, embedded by the BART checkpoint's own bge
(frozen); the BART reads the question and predicts every later sentence from the true ones before it (teacher-forced);
--per-doc of those positions go through the MLP prefix into GPT-2, which writes the sentence. The sentence loss flows
into GPT-2, the MLP and the BART alike. A light cosine term (--w-cos) keeps the BART's prediction near the sentence's own
vector, so it stays a sentence vector while it learns to carry more.

Starts from a BART checkpoint and a decoder from dec_train.py (dec_best: GPT-2 + prefix_mlp.pt). Measured as
dec_train.py measures (word F1, names, round-trip cosine, exact) on held-out conversations, before and during.

  python3 dec_e2e.py --ckpt data/model_best.pt --dec runA/dec_best --docs data/docs_007.jsonl --held am_held=data/am_held.jsonl,dolphin_held=data/dolphin_held.jsonl --out e2e1
"""
import argparse, copy, json, math, os, random, re, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel, GPT2TokenizerFast

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--dec", default="", help="a dec_train.py output dir (GPT-2 + prefix_mlp.pt); empty: GPT-2 small and a fresh MLP")
ap.add_argument("--docs", required=True); ap.add_argument("--held", required=True, help="name=path,... held-out conversations"); ap.add_argument("--out", required=True)
ap.add_argument("--window", type=int, default=128); ap.add_argument("--docs-per-step", type=int, default=4); ap.add_argument("--per-doc", type=int, default=16)
ap.add_argument("--lr-bart", type=float, default=1e-5); ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--lr-mlp", type=float, default=5e-4)
ap.add_argument("--w-cos", type=float, default=0.1); ap.add_argument("--warmup", type=int, default=200); ap.add_argument("--hours", type=float, default=5.0)
ap.add_argument("--k", type=int, default=8); ap.add_argument("--max-tok", type=int, default=96)
ap.add_argument("--eval-every", type=float, default=30.0); ap.add_argument("--eval-docs", type=int, default=150); ap.add_argument("--eval-n", type=int, default=1000); ap.add_argument("--gen-n", type=int, default=300)
ap.add_argument("--hub", default=""); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-docs", type=int, default=0)
A = ap.parse_args()
torch.manual_seed(A.seed); random.seed(A.seed); np.random.seed(A.seed)
DEV = "cuda" if torch.cuda.is_available() else "cpu"; AMP = DEV == "cuda"
os.makedirs(A.out, exist_ok=True); LOGF = os.path.join(A.out, "train.log"); LOG = open(LOGF, "a")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

# ---- the BART (train.py's SentBART, as dec_pairs.py builds it) and its own bge, frozen ---------------------------------
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


CK = torch.load(A.ckpt, map_location="cpu", weights_only=False)
bart = build(CK["layers"], CK["model"]["pos.weight"].shape[0] - 1); bart.load_state_dict(CK["model"]); bart.to(DEV).train()
btok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5"); enc = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
while len(enc.encoder.layer) < CK.get("enc_layers", 12): j = len(enc.encoder.layer) // 2; enc.encoder.layer.insert(j + 1, copy.deepcopy(enc.encoder.layer[j]))
enc.load_state_dict(CK["enc"]); enc.to(DEV).eval()
for p in enc.parameters(): p.requires_grad_(False)
BART_STEP = CK.get("step"); BART_LAYERS = CK["layers"]; ENC_LAYERS = CK.get("enc_layers", 12); del CK


@torch.no_grad()
def embed(sents, chunk=256):
    order = sorted(range(len(sents)), key=lambda i: len(sents[i])); parts = []
    for i in range(0, len(order), chunk):
        b = btok([sents[j] for j in order[i:i + chunk]], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            parts.append(F.normalize(enc(**b).last_hidden_state[:, 0].float(), dim=-1))
    o = torch.cat(parts); inv = torch.empty(len(order), dtype=torch.long, device=DEV); inv[torch.tensor(order, device=DEV)] = torch.arange(len(order), device=DEV)
    return o[inv]


# ---- the decoder ---------------------------------------------------------------------------------------------------------
tok = GPT2TokenizerFast.from_pretrained(A.dec or "gpt2"); EOS = tok.eos_token_id
lm = GPT2LMHeadModel.from_pretrained(A.dec or "gpt2").to(DEV).train(); H = lm.config.n_embd
mlp = nn.Sequential(nn.Linear(384, 4 * H), nn.GELU(), nn.Linear(4 * H, A.k * H)).to(DEV)
if A.dec and os.path.exists(os.path.join(A.dec, "prefix_mlp.pt")):
    pm = torch.load(os.path.join(A.dec, "prefix_mlp.pt"), map_location="cpu", weights_only=False); mlp.load_state_dict(pm["mlp"]); A.k = pm["k"]
else:
    with torch.no_grad(): mlp[2].weight.mul_(0.1); mlp[2].bias.zero_()


def docs_of(path, limit=0, skip=0):
    ds = []
    for i, l in enumerate(open(path)):
        if i < skip or not l.strip(): continue
        d = json.loads(l); d["sents"] = d["sents"][:A.window]; d["nq"] = max(1, int(d.get("nq", 0) or 0))
        if len(d["sents"]) >= d["nq"] + 2: ds.append(d)
        if limit and len(ds) >= limit: break
    return ds


TRAIN = docs_of(A.docs, A.max_docs)
HELD = {n: docs_of(p, A.eval_docs) for n, p in (x.split("=", 1) for x in A.held.split(","))}
out(f"[e2e-dec] {len(TRAIN)} training conversations, window {A.window}; held-out {', '.join(f'{n} {len(v)}' for n, v in HELD.items())}; BART step {BART_STEP}; decoder {A.dec or 'fresh GPT-2'} (k {A.k})")


def bart_predict(docs):
    """teacher-forced predictions for every sentence after each question; returns pd [B, L, 384] (unnormalised), V, ks"""
    flat = [s for d in docs for s in d["sents"]]; V = embed(flat); off = np.cumsum([0] + [len(d["sents"]) for d in docs])
    Lm = max(len(d["sents"]) for d in docs)
    x = torch.zeros(len(docs), Lm, 384, device=DEV); valid = torch.zeros(len(docs), Lm, dtype=torch.bool, device=DEV); masked = torch.zeros_like(valid)
    for b, d in enumerate(docs):
        n = len(d["sents"]); x[b, :n] = V[off[b]:off[b + 1]]; valid[b, :n] = True; masked[b, d["nq"]:n] = True
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
        _, pd = bart(x, valid, masked)
    return pd.float(), V, off


def dec_loss(vecs, texts):
    """sum of the sentence tokens' NLL with the vectors as prefix, and the token count (targets only projected)"""
    B = len(texts); pre = mlp(vecs).view(B, A.k, H); ids = [tok.encode(" " + t.strip())[:A.max_tok] + [EOS] for t in texts]
    L = A.k + max(map(len, ids)); wte = lm.transformer.wte
    emb = torch.zeros(B, L, H, device=DEV); att = torch.zeros(B, L, dtype=torch.long, device=DEV); lab = torch.full((B, L), -100, dtype=torch.long, device=DEV)
    for b, t in enumerate(ids):
        tt = torch.tensor(t, device=DEV, dtype=torch.long); n = A.k + len(t); emb[b, :A.k] = pre[b]; emb[b, A.k:n] = wte(tt); att[b, :n] = 1; lab[b, A.k:n] = tt
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
        h = lm.transformer(inputs_embeds=emb, attention_mask=att).last_hidden_state[:, :-1]; sel = lab[:, 1:] != -100; lg = lm.lm_head(h[sel])
    return F.cross_entropy(lg.float(), lab[:, 1:][sel], reduction="sum"), int(sel.sum())


@torch.no_grad()
def greedy(vecs, n=64):
    B = len(vecs); pre = mlp(vecs).view(B, A.k, H); emb = pre.clone(); att = torch.ones(B, A.k, dtype=torch.long, device=DEV)
    outs = [[] for _ in range(B)]; done = torch.zeros(B, dtype=torch.bool, device=DEV)
    for _ in range(n):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            nxt = lm(inputs_embeds=emb, attention_mask=att).logits[:, -1].float().argmax(-1)
        for b in range(B):
            if not done[b]:
                if int(nxt[b]) == EOS: done[b] = True
                else: outs[b].append(int(nxt[b]))
        emb = torch.cat([emb, lm.transformer.wte(nxt)[:, None]], 1); att = torch.cat([att, (~done).long()[:, None]], 1)
        if bool(done.all()): break
    return [tok.decode(o).strip() for o in outs]


WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-\.]*")
def words(s): return WORD.findall(s)
def names(s):
    w = words(s); return {x.strip(".") for i, x in enumerate(w) if (i > 0 and x[0].isupper()) or any(c.isdigit() for c in x)}
def f1(a, b):
    from collections import Counter
    a, b = [x.lower() for x in words(a)], [x.lower() for x in words(b)]
    if not a or not b: return 0.0
    c = sum((Counter(a) & Counter(b)).values()); return 0.0 if c == 0 else 2 * c / (len(a) + len(b))


@torch.no_grad()
def evaluate(tag):
    bart.eval(); lm.eval(); mlp.eval(); res = {}; rng = random.Random(7)
    for n, ds in HELD.items():
        P, T, texts, prevs = [], [], [], []
        for i in range(0, len(ds), 8):
            part = ds[i:i + 8]; pd, V, off = bart_predict(part)
            for b, d in enumerate(part):
                for t in range(d["nq"], len(d["sents"])):
                    P.append(F.normalize(pd[b, t], dim=-1)); T.append(V[off[b] + t]); texts.append(d["sents"][t]); prevs.append(d["sents"][t - 1])
        idx = list(range(len(texts))); rng.shuffle(idx); idx = idx[:A.eval_n]
        P = torch.stack([P[i] for i in idx]); T = torch.stack([T[i] for i in idx]); texts = [texts[i] for i in idx]; prevs = [prevs[i] for i in idx]
        for which, Vv in (("pred", P), ("true", T)):
            nll = cnt = 0.0
            for i in range(0, len(texts), 64):
                s_, c_ = dec_loss(Vv[i:i + 64], texts[i:i + 64]); nll += float(s_); cnt += c_
            G = min(A.gen_n, len(texts)); g = []
            for i in range(0, G, 32): g += greedy(Vv[i:min(i + 32, G)])
            refs = texts[:G]; nm = [(names(r), names(h)) for r, h in zip(refs, g)]; has = [x for x in nm if x[0]]
            cos = float((embed([x if x else "." for x in g]) * T[:G]).sum(-1).mean())
            r = {"nll": nll / cnt, "word_f1": float(np.mean([f1(h, rf) for h, rf in zip(g, refs)])), "names": float(np.mean([len(a & b) / len(a) for a, b in has])) if has else float("nan"),
                 "names_n": len(has), "cos": cos, "exact": float(np.mean([h.strip() == rf.strip() for h, rf in zip(g, refs)])), "vec_cos": float((P * T).sum(-1).mean())}
            res[f"{n}/{which}"] = r
            out(f"[eval {tag}] {n:13s} {which}: nll {r['nll']:.3f}  word-F1 {r['word_f1']:.3f}  names {r['names']:.3f} (of {r['names_n']})  round-trip cos {r['cos']:.3f}  exact {r['exact']:.3f}" + (f"  | cos(pred, true) {r['vec_cos']:.3f}" if which == "pred" else ""))
            if which == "pred":
                for j in range(3): out(f"[sample {tag}] {n} | prev: {json.dumps(prevs[j][:150])}\n    true: {json.dumps(refs[j][:200])}\n    gen : {json.dumps(g[j][:200])}")
    bart.train(); lm.train(); mlp.train(); return res


def hub_put(local, remote):
    if not A.hub: return
    try:
        from huggingface_hub import HfApi
        repo, pre = A.hub.split(":", 1); api = HfApi()
        if os.path.isdir(local): api.upload_folder(folder_path=local, path_in_repo=f"{pre}/{remote}", repo_id=repo)
        else: api.upload_file(path_or_fileobj=local, path_in_repo=f"{pre}/{remote}", repo_id=repo)
    except Exception as e:
        out(f"[hub] {remote} failed: {str(e)[:150]}")


def save(tag):
    d = os.path.join(A.out, f"e2e_{tag}"); os.makedirs(d, exist_ok=True); lm.save_pretrained(d); tok.save_pretrained(d)
    torch.save({"mlp": mlp.state_dict(), "k": A.k, "args": vars(A)}, os.path.join(d, "prefix_mlp.pt"))
    # loadable as a BART checkpoint (model, enc, layers, enc_layers): the bge is the frozen one it started with
    torch.save({"model": bart.state_dict(), "enc": enc.state_dict(), "layers": BART_LAYERS, "enc_layers": ENC_LAYERS, "step": BART_STEP, "e2e_args": vars(A)}, os.path.join(d, "bart.pt")); return d


res0 = evaluate("start")
opt = torch.optim.AdamW([{"params": bart.parameters(), "lr": A.lr_bart}, {"params": lm.parameters(), "lr": A.lr}, {"params": mlp.parameters(), "lr": A.lr_mlp}], weight_decay=0.0, betas=(0.9, 0.98))
base_lr = [A.lr_bart, A.lr, A.lr_mlp]; T0 = time.time(); last = time.time(); step = 0; acc = []; accc = []; best = 1e9
order = list(range(len(TRAIN))); random.shuffle(order); ptr = 0
while time.time() - T0 < A.hours * 3600:
    el = time.time() - T0; sc = min(1.0, (step + 1) / A.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, el / (A.hours * 3600)))))
    for g_, b_ in zip(opt.param_groups, base_lr): g_["lr"] = b_ * sc
    if ptr + A.docs_per_step > len(order): random.shuffle(order); ptr = 0
    part = [TRAIN[i] for i in order[ptr:ptr + A.docs_per_step]]; ptr += A.docs_per_step
    pd, V, off = bart_predict(part)
    sel = [(b, t) for b, d in enumerate(part) for t in random.sample(range(d["nq"], len(d["sents"])), min(A.per_doc, len(d["sents"]) - d["nq"]))]
    P = F.normalize(torch.stack([pd[b, t] for b, t in sel]), dim=-1); Tv = torch.stack([V[off[b] + t] for b, t in sel])
    s_, c_ = dec_loss(P, [part[b]["sents"][t] for b, t in sel]); lcos = (1 - (P * Tv).sum(-1)).mean()
    loss = s_ / max(c_, 1) + A.w_cos * lcos
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(list(bart.parameters()) + list(lm.parameters()) + list(mlp.parameters()), 1.0); opt.step()
    step += 1; acc.append(float(s_) / max(c_, 1)); accc.append(1 - float(lcos))
    if step % 100 == 0: out(f"[step {step}] sentence nll {np.mean(acc):.3f}, cos(pred, true) {np.mean(accc):.3f}, {(time.time() - T0) / 60:.0f} min"); acc = []; accc = []
    if time.time() - last > A.eval_every * 60:
        r = evaluate(f"step {step}"); last = time.time(); key = np.mean([v["nll"] for k, v in r.items() if k.endswith("/pred")])
        if key < best: best = key; hub_put(save("best"), "e2e_best")
        hub_put(LOGF, "train.log")
r = evaluate("end"); json.dump({"start": res0, "end": r}, open(os.path.join(A.out, "result.json"), "w"), indent=1)
hub_put(save("final"), "e2e_final"); hub_put(os.path.join(A.out, "result.json"), "result.json"); hub_put(LOGF, "train.log")
out("DEC_E2E_DONE")
