#!/usr/bin/env python3
"""Sentence-sequence BART: a transformer over a document's sequence of sentence vectors.

  input     : a document as the sequence of its sentences' vectors (bge-small, 384-d, unit length), cropped to --seq
  corruption: per document, either spans of sentences (Poisson(--span) lengths, --mask of the sentences in all) or,
              with probability --p-suffix, the whole tail after a random cut (a quarter to three quarters in) replaced
              by a learned [MASK] vector - BART's text infilling one level up (spans not collapsed: positions stay)
              plus prefix-to-continuation, so the decoder learns to WRITE what follows and not only to fill gaps
  encoder   : bidirectional, reads the corrupted sequence; at each masked position it must name the hidden sentence
  decoder   : causal, cross-attends to the encoder, regenerates the sequence one sentence vector at a time; scored
              on the hidden sentences only (an unhidden one can be copied through the encoder, which teaches nothing)
  loss      : both heads are scored contrastively - the predicted vector against every real sentence vector of the
              batch (InfoNCE, temperature --tau): naming the right sentence among thousands, not regressing to an
              average vector (regression alone collapses to the centroid)
  eval      : held-out documents (the last shard): encoder masked-sentence and decoder next-sentence top-1 / top-10
              among all sentences of the eval batch, and a few greedy continuations decoded by nearest neighbour
  usage     : python3 train.py --data /root/sb/data/docs --out /root/sb/run1 [--steps 30000] [--d 768] [--layers 6]
"""
import argparse, glob, json, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=30000); ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--seq", type=int, default=128); ap.add_argument("--d", type=int, default=768)
ap.add_argument("--layers", type=int, default=6, help="encoder layers (the decoder has as many)")
ap.add_argument("--heads", type=int, default=12); ap.add_argument("--ffn", type=int, default=3072)
ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--warmup", type=int, default=2000)
ap.add_argument("--mask", type=float, default=0.3); ap.add_argument("--span", type=float, default=3.0)
ap.add_argument("--tau", type=float, default=0.05); ap.add_argument("--w-enc", type=float, default=1.0)
ap.add_argument("--w-cos", type=float, default=1.0, help="weight of the direct reconstruction term, 1 - cosine(predicted, true sentence vector)")
ap.add_argument("--eval-every", type=int, default=1000); ap.add_argument("--save-every", type=int, default=2000)
ap.add_argument("--min-sents", type=int, default=8)
ap.add_argument("--prev-skip", type=int, default=0, help="1: the decoder's output is (its head + a learned gate x the previous sentence's vector), normalised")
ap.add_argument("--p-suffix", type=float, default=0.5, help="share of documents whose tail is hidden (continuation) instead of spans")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
torch.manual_seed(0); random.seed(0); np.random.seed(0)
DEV = "cuda"

# ---- data: memory-mapped vectors, documents as row ranges; the last shard is held out ----
shards = []
for vf in sorted(glob.glob(os.path.join(A.data, "vec_*.npy"))):
    k = os.path.basename(vf)[4:-4]
    off = np.load(os.path.join(A.data, f"off_{k}.npy"))
    shards.append((k, np.load(vf, mmap_mode="r"), off))
assert len(shards) >= 2, "need at least two embedded shards (one is held out)"
train_sh, eval_sh = shards[:-1], shards[-1]
docs = [(si, off[d], off[d + 1]) for si, (_, _, off) in enumerate(train_sh) for d in range(len(off) - 1) if off[d + 1] - off[d] >= A.min_sents]
eoff = eval_sh[2]
edocs = [(eoff[d], eoff[d + 1]) for d in range(len(eoff) - 1) if eoff[d + 1] - eoff[d] >= A.min_sents]
random.Random(1).shuffle(edocs); edocs = edocs[:2000]
DIM = train_sh[0][1].shape[1]
print(f"[data] {len(docs)} training documents in {len(train_sh)} shards, {len(edocs)} held-out documents (shard {eval_sh[0]}), dim {DIM}", flush=True)


def crop(vec, a, b):
    n = b - a
    if n > A.seq:
        a = a + random.randint(0, n - A.seq); b = a + A.seq
    return np.asarray(vec[a:b], dtype=np.float32)


def make_batch(items, mode="mix"):
    """items: list of float32 [n_i, DIM] arrays -> padded x [B, L, DIM], valid [B, L], masked [B, L], cut [B]
    mode: mix (training), span (spans only), suffix (tails only); cut = first hidden position of a suffix document, else -1"""
    L = max(len(x) for x in items); B = len(items)
    x = np.zeros((B, L, DIM), np.float32); valid = np.zeros((B, L), bool); masked = np.zeros((B, L), bool); cut = np.full(B, -1)
    for i, v in enumerate(items):
        n = len(v); x[i, :n] = v; valid[i, :n] = True
        if mode == "suffix" or (mode == "mix" and random.random() < A.p_suffix):
            k = random.randint(max(1, n // 4), max(1, (3 * n) // 4)); masked[i, k:n] = True; cut[i] = k; continue
        budget = max(1, int(round(A.mask * n)))
        tries = 0
        while masked[i, :n].sum() < budget and tries < 100:
            ln = max(1, np.random.poisson(A.span)); st = random.randint(0, max(0, n - ln))
            masked[i, st:min(n, st + ln)] = True; tries += 1
    t = lambda a: torch.from_numpy(a).to(DEV)
    return t(x), t(valid), t(masked), t(cut)


def train_batch():
    items = []
    for _ in range(A.batch):
        si, a, b = random.choice(docs); items.append(crop(train_sh[si][1], a, b))
    return make_batch(items)[:3]


# ---- model ----
class SentBART(nn.Module):
    def __init__(s):
        super().__init__()
        d = A.d
        s.inp = nn.Linear(DIM, d); s.pos = nn.Embedding(A.seq + 1, d)
        s.mask_vec = nn.Parameter(torch.randn(d) * 0.02); s.bos = nn.Parameter(torch.randn(d) * 0.02)
        el = nn.TransformerEncoderLayer(d, A.heads, A.ffn, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        dl = nn.TransformerDecoderLayer(d, A.heads, A.ffn, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        s.enc = nn.TransformerEncoder(el, A.layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        s.dec = nn.TransformerDecoder(dl, A.layers, norm=nn.LayerNorm(d))
        s.enc_head = nn.Linear(d, DIM); s.dec_head = nn.Linear(d, DIM)
        s.prev_gate = nn.Parameter(torch.tensor(1.0))   # --prev-skip: the decoder writes a change to the sentence before
        if A.prev_skip:   # the head starts at zero, so the decoder starts exactly at the baseline and learns the difference
            nn.init.zeros_(s.dec_head.weight); nn.init.zeros_(s.dec_head.bias)

    def encode(s, x, valid, masked):
        B, L, _ = x.shape
        h = s.inp(x)
        h = torch.where(masked[..., None], s.mask_vec.to(h.dtype).expand_as(h), h)
        h = h + s.pos(torch.arange(L, device=x.device))[None]
        return s.enc(h, src_key_padding_mask=~valid)

    def decode(s, mem, valid, x):
        """teacher-forced: position t sees BOS + the true sentences before t"""
        B, L, _ = x.shape
        y = torch.cat([s.bos.expand(B, 1, -1), s.inp(x[:, :-1])], dim=1) + s.pos(torch.arange(L, device=x.device))[None]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
        return s.dec(y, mem, tgt_mask=causal, tgt_key_padding_mask=~valid, memory_key_padding_mask=~valid)

    def forward(s, x, valid, masked):
        mem = s.encode(x, valid, masked)
        pd = s.dec_head(s.decode(mem, valid, x)).float()
        if A.prev_skip:   # start from "like the sentence before" (the baseline the first run lost to) and learn the difference
            prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1).float()
            pd = pd + s.prev_gate * prev
        return F.normalize(s.enc_head(mem).float(), dim=-1), F.normalize(pd, dim=-1)


def nce(pred, tgt_all, idx):
    """pred [N, DIM] for the targets tgt_all[idx]: the contrastive term (name the right sentence among every real one
    of the batch) plus --w-cos x (1 - cosine to the true vector) - the head writes the sentence vector itself, it does
    not pick an ID from a codebook; the contrastive term only keeps it from settling on the average sentence"""
    logits = pred @ tgt_all.T / A.tau
    cos = (pred * tgt_all[idx]).sum(-1)
    LAST_COS.append(float(cos.mean().detach()))
    return F.cross_entropy(logits, idx) + A.w_cos * (1.0 - cos).mean(), logits


LAST_COS = []


model = SentBART().to(DEV)
npar = sum(p.numel() for p in model.parameters()) / 1e6
opt = torch.optim.AdamW(model.parameters(), lr=A.lr, betas=(0.9, 0.98), weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / A.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / A.steps))))
STATE = os.path.join(A.out, "state.pt"); step0 = 0
if os.path.exists(STATE):
    st = torch.load(STATE, map_location=DEV); model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); step0 = st["step"]
print(f"[model] {npar:.1f}M parameters, d {A.d}, {A.layers}+{A.layers} layers, seq {A.seq}, batch {A.batch}, resume step {step0}", flush=True)


def losses(x, valid, masked):
    pe, pd = model(x, valid, masked)
    tgt = x[valid]                                    # [N, DIM] every real sentence of the batch
    flat = torch.arange(valid.sum(), device=DEV)
    pos = torch.zeros_like(valid, dtype=torch.long); pos[valid] = flat
    le, lge = nce(pe[masked], tgt, pos[masked])       # encoder: the masked sentences
    ld, lgd = nce(pd[masked], tgt, pos[masked])       # decoder: the hidden sentences, from the ones before it
    return le, ld, lge, lgd, pos


@torch.no_grad()
def evaluate():
    """span mode: the encoder names each hidden sentence (infilling); suffix mode: the decoder names each hidden tail
    sentence from the true ones before it (teacher-forced continuation), and 'next' is the first one after the cut alone.
    Candidates: every sentence of the eval batch. Same documents, same masks every time (fixed seeds)."""
    model.eval(); r = {}
    for mode in ("span", "suffix"):
        random.seed(123); np.random.seed(123)
        for i in range(0, len(edocs), A.batch):
            items = [crop(eval_sh[1], a, b) for a, b in edocs[i:i + A.batch]]
            x, valid, masked, cut = make_batch(items, mode)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                le, ld, lge, lgd, pos = losses(x, valid, masked)
            lg, idx, k = (lge, pos[masked], "enc") if mode == "span" else (lgd, pos[masked], "dec")
            r["cos" + k] = r.get("cos" + k, 0.0) + (LAST_COS[-2] if mode == "span" else LAST_COS[-1]); LAST_COS.clear()
            top = lg.topk(10, dim=-1).indices
            hit1 = top[:, 0] == idx; hit10 = (top == idx[:, None]).any(-1)
            r[k + "1"] = r.get(k + "1", 0) + int(hit1.sum()); r[k + "10"] = r.get(k + "10", 0) + int(hit10.sum()); r["n" + k] = r.get("n" + k, 0) + len(idx)
            r["l" + k] = r.get("l" + k, 0.0) + float(le if mode == "span" else ld); r["b" + k] = r.get("b" + k, 0) + 1
            r["cand"] = r.get("cand", 0) + int(valid.sum())
            if mode == "suffix":                      # the first hidden sentence of each document: plain next-sentence prediction
                first = torch.zeros_like(masked); bi = torch.arange(len(cut), device=DEV)
                first[bi, cut.clamp(min=0)] = True; first &= masked
                sel = first[masked]
                # next sentence, model and baseline under the same rule: the sentences already seen (the document's
                # prefix, the one just before included) are not candidates - "repeat what was just said" is never right
                ok = cut > 0; bi2 = bi[ok]; c2 = cut[ok]
                tgt = x[valid].float(); want = pos[bi2, c2]
                seen = torch.zeros(len(bi2), tgt.shape[0], dtype=torch.bool, device=DEV)
                for j in range(len(bi2)): seen[j, pos[bi2[j], :c2[j]]] = True
                rows = torch.nonzero(masked)                       # (b, t) of every hidden sentence, in lgd's row order
                first_row = {(int(b_), int(t_)): k for k, (b_, t_) in enumerate(rows.tolist())}
                lm = torch.stack([lgd[first_row[(int(b_), int(c_))]] for b_, c_ in zip(bi2, c2)]).float().masked_fill(seen, -1e9)
                lb = (x[bi2, c2 - 1].float() @ tgt.T).masked_fill(seen, -1e9)
                for lg_, k in ((lm, "next"), (lb, "base")):
                    tk = lg_.topk(10, dim=-1).indices
                    r[k + "1"] = r.get(k + "1", 0) + int((tk[:, 0] == want).sum()); r[k + "10"] = r.get(k + "10", 0) + int((tk == want[:, None]).any(-1).sum())
                r["nnext"] = r.get("nnext", 0) + len(bi2)
    random.seed(time.time()); np.random.seed(int(time.time()) % 2**31); model.train()
    return {"enc_top1": r["enc1"] / r["nenc"], "enc_top10": r["enc10"] / r["nenc"], "dec_top1": r["dec1"] / r["ndec"], "dec_top10": r["dec10"] / r["ndec"],
            "next_top1": r["next1"] / r["nnext"], "next_top10": r["next10"] / r["nnext"],
            "next_baseline_top1": r["base1"] / r["nnext"], "next_baseline_top10": r["base10"] / r["nnext"],
            "enc_cos": r["cosenc"] / r["benc"], "dec_cos": r["cosdec"] / r["bdec"],
            "enc_loss": r["lenc"] / r["benc"], "dec_loss": r["ldec"] / r["bdec"], "candidates_per_batch": int(r["cand"] / (r["benc"] + r["bdec"]))}


def save(tag="latest"):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "args": vars(A)}, STATE + ".tmp")
    os.replace(STATE + ".tmp", STATE)
    torch.save({"model": model.state_dict(), "args": vars(A), "step": step}, os.path.join(A.out, f"model_{tag}.pt"))


log = open(os.path.join(A.out, "train.log"), "a")
model.train(); t0 = time.time(); acc = []
for step in range(step0 + 1, A.steps + 1):
    x, valid, masked = train_batch()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        le, ld, _, _, _ = losses(x, valid, masked)
        loss = A.w_enc * le + ld
    LAST_COS.clear()
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)); opt.step(); sched.step()
    acc.append((le.item(), ld.item()))
    if step % 100 == 0:
        e, d_ = np.mean(acc, 0); acc = []
        line = f"[step {step}] enc_loss {e:.3f} dec_loss {d_:.3f} |grad| {gn:.2f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / 60:.0f} min"
        print(line, flush=True); log.write(line + "\n"); log.flush()
    if step % A.eval_every == 0 or step == A.steps:
        ev = evaluate(); line = f"[eval {step}] " + " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in ev.items())
        print(line, flush=True); log.write(line + "\n"); log.flush()
    if step % A.save_every == 0 or step == A.steps:
        save(); print(f"[save] step {step}", flush=True)
print("TRAIN_DONE", flush=True)
