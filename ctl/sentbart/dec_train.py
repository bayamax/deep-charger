#!/usr/bin/env python3
"""The context BART's decoder: a sentence vector in front of GPT-2, the sentence written out.

    [MLP(vector) -> k prefix embeddings] [the previous sentence's raw tokens, with --raw-prev 1] [sentence t ... <eos>]

The loss counts sentence t only (and its end). Trained on dec_pairs.py's pairs: --vec pred (the BART's prediction of
sentence t, what the decoder will be given in use), true (sentence t's own vector: how much of a sentence a vector can
give back at all), or mix (true with probability --p-true, else pred). GPT-2 and the MLP are both trained.

Measured on held-out pairs, with the predicted and with the true vector alike:
  nll        - per token of sentence t
  word_f1    - word overlap of the greedy sentence with sentence t
  names      - of sentence t's names and numbers (a capitalised word not first in the sentence, or a word with a digit),
               the share the greedy sentence has
  cos        - the greedy sentence embedded by the BART's own bge, its cosine to sentence t's vector (the round trip)

  python3 dec_train.py --pairs dec --train train --eval am_held,dolphin_held --ckpt once7/model_best.pt --out dec_run1 --vec mix
"""
import argparse, json, math, os, random, re, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel, GPT2TokenizerFast

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True); ap.add_argument("--train", default="train"); ap.add_argument("--eval", default="am_held,dolphin_held")
ap.add_argument("--ckpt", required=True, help="the BART checkpoint whose bge embeds the generated sentences (the round trip)"); ap.add_argument("--out", required=True)
ap.add_argument("--gpt2", default="gpt2"); ap.add_argument("--k", type=int, default=8, help="prefix embeddings"); ap.add_argument("--raw-prev", type=int, default=0)
ap.add_argument("--vec", default="mix", choices=["pred", "true", "mix"]); ap.add_argument("--p-true", type=float, default=0.3)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--lr-mlp", type=float, default=1e-3); ap.add_argument("--warmup", type=int, default=300)
ap.add_argument("--batch", type=int, default=64); ap.add_argument("--hours", type=float, default=6.0); ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--max-tok", type=int, default=96, help="tokens of sentence t at most (longer sentences are cut)"); ap.add_argument("--prev-tok", type=int, default=64)
ap.add_argument("--eval-every", type=float, default=30.0); ap.add_argument("--eval-n", type=int, default=2000); ap.add_argument("--gen-n", type=int, default=300)
ap.add_argument("--hub", default=""); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-train", type=int, default=0)
A = ap.parse_args()
torch.manual_seed(A.seed); random.seed(A.seed); np.random.seed(A.seed)
DEV = "cuda" if torch.cuda.is_available() else "cpu"; AMP = DEV == "cuda"
os.makedirs(A.out, exist_ok=True); LOGF = os.path.join(A.out, "train.log"); LOG = open(LOGF, "a")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

tok = GPT2TokenizerFast.from_pretrained(A.gpt2); EOS = tok.eos_token_id


def load_pairs(name, limit=0):
    z = np.load(os.path.join(A.pairs, f"pairs_{name}.npz")); meta = [json.loads(l) for l in open(os.path.join(A.pairs, f"pairs_{name}.jsonl"))]
    n = len(meta) if not limit else min(limit, len(meta))
    return z["pred"][:n], z["true"][:n], meta[:n]


t0 = time.time()
TRP, TRT, TRM = load_pairs(A.train, A.max_train)
EV = {n: load_pairs(n) for n in A.eval.split(",") if n and os.path.exists(os.path.join(A.pairs, f"pairs_{n}.npz"))}
rng = np.random.RandomState(1)
for n in EV:   # a fixed draw per held-out set
    p, t, m = EV[n]; idx = rng.permutation(len(m))[:A.eval_n]; EV[n] = (p[idx], t[idx], [m[i] for i in idx])
TR_IDS = [tok.encode(" " + m["text"].strip())[:A.max_tok] + [EOS] for m in TRM]
TR_PREV = [tok.encode(m["prev"].strip())[-A.prev_tok:] for m in TRM] if A.raw_prev else None
out(f"[dec] {len(TRM)} training pairs ({sum(map(len, TR_IDS)) / 1e6:.1f}M target tokens); held-out {', '.join(f'{n} {len(v[2])}' for n, v in EV.items())}; "
    f"vec {A.vec}{f' (p_true {A.p_true})' if A.vec == 'mix' else ''}, k {A.k}, raw_prev {A.raw_prev}, {A.gpt2} [{time.time() - t0:.0f}s]")

lm = GPT2LMHeadModel.from_pretrained(A.gpt2).to(DEV); H = lm.config.n_embd
mlp = nn.Sequential(nn.Linear(384, 4 * H), nn.GELU(), nn.Linear(4 * H, A.k * H)).to(DEV)
with torch.no_grad(): mlp[2].weight.mul_(0.1); mlp[2].bias.zero_()
SEP = tok.encode("\n")   # between the previous sentence and sentence t


def make_batch(vecs, tgt_ids, prev_ids=None):
    """inputs_embeds [B, L, H], attention mask, labels (-100 except sentence t)"""
    B = len(tgt_ids); pre = mlp(torch.as_tensor(np.asarray(vecs), dtype=torch.float32, device=DEV)).view(B, A.k, H)
    wte = lm.transformer.wte; rows = []
    for b in range(B):
        mid = (prev_ids[b] + SEP) if prev_ids is not None else []
        rows.append((mid, tgt_ids[b]))
    L = A.k + max(len(m) + len(t) for m, t in rows)
    emb = torch.zeros(B, L, H, device=DEV); att = torch.zeros(B, L, dtype=torch.long, device=DEV); lab = torch.full((B, L), -100, dtype=torch.long, device=DEV)
    for b, (mid, t) in enumerate(rows):
        ids = torch.tensor(mid + t, device=DEV, dtype=torch.long); n = A.k + len(ids)
        emb[b, :A.k] = pre[b]; emb[b, A.k:n] = wte(ids); att[b, :n] = 1; lab[b, A.k + len(mid):n] = ids[len(mid):]
    return emb, att, lab


def loss_of(emb, att, lab):
    """the vocabulary projection only where a target is (the prefix and the previous sentence carry no loss): the
    [B, L, 50257] logits of every position were what ran two runs out of a 12 GB card"""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
        h = lm.transformer(inputs_embeds=emb, attention_mask=att).last_hidden_state[:, :-1]
        sel = lab[:, 1:] != -100
        lg = lm.lm_head(h[sel])
    return F.cross_entropy(lg.float(), lab[:, 1:][sel], reduction="sum"), int(sel.sum())


@torch.no_grad()
def greedy(vecs, prev_ids=None, n=64):
    B = len(vecs); emb, att, _ = make_batch(vecs, [[] for _ in range(B)], prev_ids)
    lens = att.sum(1); outs = [[] for _ in range(B)]; done = [False] * B
    for _ in range(n):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            lg = lm(inputs_embeds=emb, attention_mask=att).logits.float()
        nxt = lg[torch.arange(B, device=DEV), lens - 1].argmax(-1)
        new = torch.zeros(B, 1, H, device=DEV); emb = torch.cat([emb, new], 1); att = torch.cat([att, torch.zeros(B, 1, dtype=torch.long, device=DEV)], 1)
        for b in range(B):
            if done[b]: continue
            t_ = int(nxt[b])
            if t_ == EOS: done[b] = True; continue
            outs[b].append(t_); emb[b, lens[b]] = lm.transformer.wte.weight[t_]; att[b, lens[b]] = 1; lens[b] += 1
        if all(done): break
    return [tok.decode(o).strip() for o in outs]


# ---- the round trip: the BART checkpoint's own bge ----------------------------------------------------------------------
CK = torch.load(A.ckpt, map_location="cpu", weights_only=False)
btok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5"); enc = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
import copy  # noqa: E402
while len(enc.encoder.layer) < CK.get("enc_layers", 12): j = len(enc.encoder.layer) // 2; enc.encoder.layer.insert(j + 1, copy.deepcopy(enc.encoder.layer[j]))
enc.load_state_dict(CK["enc"]); enc.to(DEV).eval(); del CK


@torch.no_grad()
def embed(sents):
    b = btok(sents, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEV)
    return F.normalize(enc(**b).last_hidden_state[:, 0].float(), dim=-1)


WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-\.]*")
def words(s): return WORD.findall(s)
def names(s):
    w = words(s); return {x.strip(".") for i, x in enumerate(w) if (i > 0 and x[0].isupper()) or any(c.isdigit() for c in x)}


def f1(a, b):
    a, b = [x.lower() for x in words(a)], [x.lower() for x in words(b)]
    if not a or not b: return 0.0
    from collections import Counter
    c = sum((Counter(a) & Counter(b)).values()); return 0.0 if c == 0 else 2 * c / (len(a) + len(b))


@torch.no_grad()
def evaluate(tag):
    lm.eval(); mlp.eval(); res = {}
    for n, (P, T, M) in EV.items():
        ids = [tok.encode(" " + m["text"].strip())[:A.max_tok] + [EOS] for m in M]; prev = [tok.encode(m["prev"].strip())[-A.prev_tok:] for m in M] if A.raw_prev else None
        for which, V in (("pred", P), ("true", T)):
            nll = cnt = 0.0
            for i in range(0, len(M), 64):
                s_, c_ = loss_of(*make_batch(V[i:i + 64].astype(np.float32), ids[i:i + 64], prev[i:i + 64] if prev else None)); nll += float(s_); cnt += c_
            g = []; G = min(A.gen_n, len(M))
            for i in range(0, G, 32):
                j = min(i + 32, G); g += greedy(V[i:j].astype(np.float32), prev[i:j] if prev else None)
            refs = [m["text"] for m in M[:G]]
            nm = [(names(r), names(h)) for r, h in zip(refs, g)]; has = [x for x in nm if x[0]]
            cos = float((embed([x if x else '.' for x in g]) * torch.as_tensor(T[:G].astype(np.float32), device=DEV)).sum(-1).mean()) if g else 0.0
            r = {"nll": nll / cnt, "word_f1": float(np.mean([f1(h, rf) for h, rf in zip(g, refs)])), "names": float(np.mean([len(a & b) / len(a) for a, b in has])) if has else float("nan"),
                 "names_n": len(has), "cos": cos, "exact": float(np.mean([h.strip() == rf.strip() for h, rf in zip(g, refs)]))}
            res[f"{n}/{which}"] = r
            out(f"[eval {tag}] {n:13s} {which}: nll {r['nll']:.3f}  word-F1 {r['word_f1']:.3f}  names {r['names']:.3f} (of {r['names_n']})  round-trip cos {r['cos']:.3f}  exact {r['exact']:.3f}")
            if which == "pred":
                for j in range(3):
                    out(f"[sample {tag}] {n} | prev: {json.dumps(M[j]['prev'][:160])}\n    true: {json.dumps(refs[j][:200])}\n    gen : {json.dumps(g[j][:200])}")
    lm.train(); mlp.train(); return res


def hub_put(local, remote):
    if not A.hub: return
    try:
        from huggingface_hub import HfApi
        repo, pre = A.hub.split(":", 1); api = HfApi()
        (api.upload_folder if os.path.isdir(local) else api.upload_file)(**({"folder_path": local} if os.path.isdir(local) else {"path_or_fileobj": local}), path_in_repo=f"{pre}/{remote}", repo_id=repo)
    except Exception as e:
        out(f"[hub] {remote} failed: {str(e)[:150]}")


def save(tag):
    d = os.path.join(A.out, f"dec_{tag}"); os.makedirs(d, exist_ok=True); lm.save_pretrained(d); tok.save_pretrained(d)
    torch.save({"mlp": mlp.state_dict(), "k": A.k, "raw_prev": A.raw_prev, "args": vars(A)}, os.path.join(d, "prefix_mlp.pt")); return d


res0 = evaluate("start")
opt = torch.optim.AdamW([{"params": lm.parameters(), "lr": A.lr}, {"params": mlp.parameters(), "lr": A.lr_mlp}], weight_decay=0.0, betas=(0.9, 0.98))
order = np.random.permutation(len(TR_IDS)); steps = int(A.epochs * len(order) / A.batch); T0 = time.time(); last = time.time(); acc = []; best = 1e9
out(f"[dec] {steps} updates of {A.batch} ({A.epochs} epoch) or {A.hours} h")
for step in range(steps):
    if time.time() - T0 > A.hours * 3600: break
    f_ = step / max(1, steps); sc = min(1.0, (step + 1) / A.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * f_)))
    opt.param_groups[0]["lr"] = A.lr * sc; opt.param_groups[1]["lr"] = A.lr_mlp * sc
    bi = order[(step * A.batch) % len(order):(step * A.batch) % len(order) + A.batch]
    if A.vec == "pred": V = TRP[bi]
    elif A.vec == "true": V = TRT[bi]
    else: V = np.where((np.random.rand(len(bi)) < A.p_true)[:, None], TRT[bi], TRP[bi])
    s_, c_ = loss_of(*make_batch(V.astype(np.float32), [TR_IDS[i] for i in bi], [TR_PREV[i] for i in bi] if TR_PREV else None))
    opt.zero_grad(set_to_none=True); (s_ / max(c_, 1)).backward(); torch.nn.utils.clip_grad_norm_(list(lm.parameters()) + list(mlp.parameters()), 1.0); opt.step()
    acc.append(float(s_) / max(c_, 1))
    if (step + 1) % 200 == 0: out(f"[step {step + 1}/{steps}] train nll {np.mean(acc):.3f}, {(time.time() - T0) / 60:.0f} min"); acc = []
    if time.time() - last > A.eval_every * 60:
        r = evaluate(f"step {step + 1}"); last = time.time(); key = np.mean([v["nll"] for k, v in r.items() if k.endswith("/pred")])
        if key < best: best = key; hub_put(save("best"), "dec_best")
        hub_put(LOGF, "train.log")
r = evaluate("end"); json.dump({"start": res0, "end": r}, open(os.path.join(A.out, "result.json"), "w"), indent=1)
hub_put(save("final"), "dec_final"); hub_put(os.path.join(A.out, "result.json"), "result.json"); hub_put(LOGF, "train.log")
out("DEC_TRAIN_DONE")
