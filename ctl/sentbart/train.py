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
import argparse, glob, json, math, os, random, sys, time
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
ap.add_argument("--eval-shard", default="", help="the held-out shard's key (e.g. 001); default the last one")
ap.add_argument("--page", type=int, default=0, help="1: a page token in front of the encoder whose output is the document's vector, trained so each hidden sentence finds its own document's vector among the batch's (sentence -> page retrieval)")
ap.add_argument("--w-page", type=float, default=1.0)
ap.add_argument("--page-res", type=int, default=0, help="1: page vector = normalise(mean of the visible sentence vectors + page_head(page token)), the head starting at zero - it starts at the mean baseline and learns the difference")
ap.add_argument("--page-pool", default="token", help="token: the page token's output; mean: the mean of the encoder's outputs over the visible sentences (both through page_head, no input mean)")
ap.add_argument("--init", default="", help="a model_*.pt to start from (weights only; fresh optimizer and schedule) when --out has no state")
ap.add_argument("--eval-only", type=int, default=0, help="1: evaluate the --init weights once (with the page recall curve) and exit")
ap.add_argument("--skip-grad", type=float, default=0.0, help=">0: a step whose gradient norm (before clipping) exceeds this is skipped, not applied - a guard against the blow-up run4c2 hit")
ap.add_argument("--search-eval", default="", help="queries jsonl ({idx: document index in the held-out shard, q_nat, q_hard}): article search over EVERY document of the held-out shard with full, unmasked documents (index time), run once with the --init weights and exit")
ap.add_argument("--search-out", default="")
ap.add_argument("--page-queue", type=int, default=0, help="N: the last N batches' page vectors (detached) as extra negatives for the page loss")
ap.add_argument("--prev-skip", type=int, default=0, help="1: the decoder's output is (its head + a learned gate x the previous sentence's vector), normalised")
ap.add_argument("--p-suffix", type=float, default=0.5, help="share of documents whose tail is hidden (continuation) instead of spans")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
torch.manual_seed(0); random.seed(0); np.random.seed(0)
DEV = "cuda"

# ---- data: memory-mapped vectors, documents as row ranges; the last shard is held out ----
class Q8:
    """an int8 shard (embed.py --int8): row i is q[i] * scl[i] / 127, read back as float32 slices"""
    def __init__(s, q, scl): s.q, s.scl, s.shape = q, scl, q.shape
    def __getitem__(s, sl): return s.q[sl].astype(np.float32) * (s.scl[sl].astype(np.float32)[:, None] / 127.0)


shards = []
for vf in sorted(glob.glob(os.path.join(A.data, "vec_*.npy"))):
    k = os.path.basename(vf)[4:-4]
    off = np.load(os.path.join(A.data, f"off_{k}.npy"))
    v = np.load(vf, mmap_mode="r"); sf = os.path.join(A.data, f"scl_{k}.npy")
    if v.dtype == np.int8: v = Q8(v, np.load(sf, mmap_mode="r"))
    shards.append((k, v, off))
assert len(shards) >= 2, "need at least two embedded shards (one is held out)"
ei = [k for k, _, _ in shards].index(A.eval_shard) if A.eval_shard else len(shards) - 1
eval_sh = shards[ei]; train_sh = shards[:ei] + shards[ei + 1:]
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
        if mode == "one":
            j = random.randint(1, n - 1); masked[i, j] = True; cut[i] = j; continue
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
        s.page_tok = nn.Parameter(torch.randn(d) * 0.02); s.page_head = nn.Linear(d, DIM); s.last_page = None
        s.mask_vec = nn.Parameter(torch.randn(d) * 0.02); s.bos = nn.Parameter(torch.randn(d) * 0.02)
        el = nn.TransformerEncoderLayer(d, A.heads, A.ffn, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        dl = nn.TransformerDecoderLayer(d, A.heads, A.ffn, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        s.enc = nn.TransformerEncoder(el, A.layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        s.dec = nn.TransformerDecoder(dl, A.layers, norm=nn.LayerNorm(d))
        s.enc_head = nn.Linear(d, DIM); s.dec_head = nn.Linear(d, DIM)
        s.prev_gate = nn.Parameter(torch.tensor(1.0))   # --prev-skip: the decoder writes a change to the sentence before
        if A.prev_skip:   # the head starts at zero, so the decoder starts exactly at the baseline and learns the difference
            nn.init.zeros_(s.dec_head.weight); nn.init.zeros_(s.dec_head.bias)
        if A.page_res:    # likewise the page vector starts at the mean of the visible sentences (run4's page lost to it)
            nn.init.zeros_(s.page_head.weight); nn.init.zeros_(s.page_head.bias)

    def encode(s, x, valid, masked):
        B, L, _ = x.shape
        h = s.inp(x)
        h = torch.where(masked[..., None], s.mask_vec.to(h.dtype).expand_as(h), h)
        h = h + s.pos(torch.arange(L, device=x.device))[None]
        if not A.page:
            return s.enc(h, src_key_padding_mask=~valid)
        # the page token reads the whole (corrupted) document; its output, through page_head, is the page vector
        h = torch.cat([s.page_tok.to(h.dtype).expand(B, 1, -1), h], dim=1)
        out = s.enc(h, src_key_padding_mask=torch.cat([torch.zeros_like(valid[:, :1]), ~valid], dim=1))
        if A.page_pool == "encmean":   # no new parameters: the encoder's own sentence reconstructions (enc_head, already in the sentence-vector space), averaged
            vis = (valid & ~masked).float()[..., None]
            rec = F.normalize(s.enc_head(out[:, 1:]).float(), dim=-1)
            pg = (rec * vis).sum(1) / vis.sum(1).clamp_min(1)
        elif A.page_pool == "mean":
            vis = (valid & ~masked).to(out.dtype)[..., None]
            pg = s.page_head((out[:, 1:] * vis).sum(1) / vis.sum(1).clamp_min(1)).float()
        else:
            pg = s.page_head(out[:, 0]).float()
        if A.page_res:
            keep = (valid & ~masked).float()[..., None]
            pg = pg + F.normalize((x.float() * keep).sum(1) / keep.sum(1).clamp_min(1), dim=-1)
        s.last_page = F.normalize(pg, dim=-1)
        return out[:, 1:]

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
if A.init and not os.path.exists(STATE):
    _miss = model.load_state_dict(torch.load(A.init, map_location=DEV)["model"], strict=False)
    print(f"[init] weights from {A.init} (missing {list(_miss.missing_keys)}, unexpected {list(_miss.unexpected_keys)})", flush=True)
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
    LP[0] = torch.zeros((), device=DEV)
    if A.page:                                         # each hidden sentence must find its own document's page vector
        q = x[masked].float(); doc = torch.nonzero(masked)[:, 0]
        keys = model.last_page
        if A.page_queue and PQ:
            keys = torch.cat([keys, torch.cat(PQ)])     # earlier batches' pages: negatives only (labels index the current batch)
        LP[0] = F.cross_entropy(q @ keys.T / A.tau, doc)
        if A.page_queue:
            PQ.append(model.last_page.detach()); del PQ[:-A.page_queue]
    return le, ld, lge, lgd, pos


LP = [None]; PQ = []; R_CURVE = {}; SKIPPED = [0]; BEST = [-1.0]


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
    # page retrieval over all held-out documents: one sentence hidden in each, that sentence's vector is the query,
    # its document must come first among every held-out page. Baseline: the mean of the document's other sentences.
    if A.page:
        random.seed(321); np.random.seed(321); pv, bv, qv = [], [], []
        for i in range(0, len(edocs), A.batch):
            items = [crop(eval_sh[1], a, b) for a, b in edocs[i:i + A.batch]]
            x, valid, masked, cut = make_batch(items, "one")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(x, valid, masked)
            pv.append(model.last_page.float()); qv.append(x[masked].float())
            keep = (valid & ~masked).float()[..., None]
            bv.append(F.normalize((x.float() * keep).sum(1) / keep.sum(1).clamp_min(1), dim=-1))
        Pm, Pb, Q = torch.cat(pv), torch.cat(bv), torch.cat(qv); tgt_ = torch.arange(len(Q), device=DEV)
        for M_, k in ((Pm, "page"), (Pb, "page_base")):
            tk = (Q @ M_.T).topk(10, dim=-1).indices
            r[k + "1"] = int((tk[:, 0] == tgt_).sum()) / len(Q); r[k + "10"] = int((tk == tgt_[:, None]).any(-1).sum()) / len(Q)
            S_ = Q @ M_.T; rank = (S_ > S_.diagonal()[:, None]).sum(1) + 1     # 1 = first
            curve = " ".join(f"@{kk} {float((rank <= kk).float().mean()):.3f}" for kk in (1, 5, 10, 20, 50, 100, 200))
            k90 = int(torch.quantile(rank.float(), 0.9).ceil()); r[k + "_curve"] = f"{curve} | 90% within top {k90}, median rank {int(rank.float().median())}"
        r["npages"] = len(Q)
        for k in ("page", "page_base"): R_CURVE[k] = r[k + "_curve"]
    random.seed(time.time()); np.random.seed(int(time.time()) % 2**31); model.train()
    pg = ({"page_top1": r["page1"], "page_top10": r["page10"], "page_base_top1": r["page_base1"], "page_base_top10": r["page_base10"], "pages": r["npages"]} if A.page else {})
    return {**pg, "enc_top1": r["enc1"] / r["nenc"], "enc_top10": r["enc10"] / r["nenc"], "dec_top1": r["dec1"] / r["ndec"], "dec_top10": r["dec10"] / r["ndec"],
            "next_top1": r["next1"] / r["nnext"], "next_top10": r["next10"] / r["nnext"],
            "next_baseline_top1": r["base1"] / r["nnext"], "next_baseline_top10": r["base10"] / r["nnext"],
            "enc_cos": r["cosenc"] / r["benc"], "dec_cos": r["cosdec"] / r["bdec"],
            "enc_loss": r["lenc"] / r["benc"], "dec_loss": r["ldec"] / r["bdec"], "candidates_per_batch": int(r["cand"] / (r["benc"] + r["bdec"]))}


def save(tag="latest"):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "args": vars(A)}, STATE + ".tmp")
    os.replace(STATE + ".tmp", STATE)
    torch.save({"model": model.state_dict(), "args": vars(A), "step": step}, os.path.join(A.out, f"model_{tag}.pt"))


if A.search_eval:
    # article search at index time: every held-out document whole (its first --seq sentences, nothing masked) -> one vector
    # per method; the queries are questions written from a passage of one document; bge-small with its query prefix.
    from transformers import AutoModel, AutoTokenizer
    model.eval(); V, off = eval_sh[1], eval_sh[2]; ND = len(off) - 1
    qs = [json.loads(l) for l in open(A.search_eval) if l.strip()]
    btok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5"); benc = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(DEV).eval()
    PFX = "Represent this sentence for searching relevant passages: "
    @torch.no_grad()
    def qvec(texts):
        out = []
        for i in range(0, len(texts), 128):
            b = btok([PFX + t for t in texts[i:i + 128]], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEV)
            out.append(F.normalize(benc(**b).last_hidden_state[:, 0].float(), dim=-1))
        return torch.cat(out)
    P = {k: torch.zeros(ND, DIM, device=DEV) for k in ("page", "mean", "mean_all", "lead")}
    with torch.no_grad():
        for i in range(0, ND, 64):
            items = [np.asarray(V[off[d]:min(off[d + 1], off[d] + A.seq)], dtype=np.float32) for d in range(i, min(i + 64, ND))]
            L = max(len(t) for t in items); B = len(items)
            x = torch.zeros(B, L, DIM, device=DEV); valid = torch.zeros(B, L, dtype=torch.bool, device=DEV)
            for j, t in enumerate(items): x[j, :len(t)] = torch.from_numpy(t).to(DEV); valid[j, :len(t)] = True
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model.encode(x, valid, torch.zeros_like(valid))
            P["page"][i:i + B] = model.last_page.float()
            P["mean"][i:i + B] = F.normalize((x * valid[..., None]).sum(1) / valid.sum(1, keepdim=True), dim=-1)
            P["lead"][i:i + B] = x[:, 0]
        sid = torch.from_numpy(np.repeat(np.arange(ND), np.diff(off))).to(DEV)
        CH = 1 << 18
        for a in range(0, int(off[-1]), CH):
            v = torch.from_numpy(np.asarray(V[a:a + CH], dtype=np.float32)).to(DEV)
            P["mean_all"].index_add_(0, sid[a:a + CH], v)
        P["mean_all"] = F.normalize(P["mean_all"], dim=-1)
    res = {}
    for qk in sorted({k for q in qs for k in q if k.startswith("q_")}):
        sub = [q for q in qs if q.get(qk)]
        if not sub: continue
        Q = qvec([q[qk] for q in sub]); gold = torch.tensor([int(q["idx"]) for q in sub], device=DEV)
        with torch.no_grad():
            S = {k: Q @ M.T for k, M in P.items()}
            mx = torch.full((len(sub), ND), -2.0, device=DEV)   # best single sentence of each document
            for a in range(0, int(off[-1]), CH):
                v = torch.from_numpy(np.asarray(V[a:a + CH], dtype=np.float32)).to(DEV)
                mx.scatter_reduce_(1, sid[a:a + CH][None].expand(len(sub), -1), Q @ v.T, reduce="amax")
            S["maxsent"] = mx
            S["page+mean"] = S["page"] + S["mean_all"]
            # the query through the model too: a one-sentence document -> its page vector, against the pages
            qx = Q[:, None, :]; qv = torch.ones(len(sub), 1, dtype=torch.bool, device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model.encode(qx, qv, torch.zeros_like(qv))
            S["page(q->page)"] = model.last_page.float() @ P["page"].T
        for k, M in S.items():
            rank = (M > M.gather(1, gold[:, None])).sum(1) + 1
            res[(qk, k)] = {f"@{kk}": float((rank <= kk).float().mean()) for kk in (1, 10, 100)} | {"mrr": float((1.0 / rank.float()).mean())}
            print(f"[search] {qk} {k:10s} " + " ".join(f"{a} {b:.3f}" for a, b in res[(qk, k)].items()) + f"  (n {len(sub)}, pool {ND})", flush=True)
    if A.search_out: json.dump({f"{a}|{b}": v for (a, b), v in res.items()}, open(A.search_out, "w"), indent=1)
    sys.exit(0)
if A.eval_only:
    ev = evaluate()
    print("[eval-only] " + " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in ev.items()), flush=True)
    for k in ("page", "page_base"): print(f"[curve] {k}: {R_CURVE[k]}", flush=True)
    sys.exit(0)
log = open(os.path.join(A.out, "train.log"), "a")
model.train(); t0 = time.time(); acc = []
for step in range(step0 + 1, A.steps + 1):
    x, valid, masked = train_batch()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        le, ld, _, _, _ = losses(x, valid, masked)
        loss = A.w_enc * le + ld + (A.w_page * LP[0] if A.page else 0.0)
    LAST_COS.clear()
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    if A.skip_grad > 0 and not (gn < A.skip_grad):       # also catches nan / inf
        SKIPPED[0] += 1; opt.zero_grad(set_to_none=True)
    else:
        opt.step()
    sched.step()
    acc.append((le.item(), ld.item()))
    if step % 100 == 0:
        e, d_ = np.mean(acc, 0); acc = []
        line = f"[step {step}] enc_loss {e:.3f} dec_loss {d_:.3f} |grad| {gn:.2f} lr {sched.get_last_lr()[0]:.2e} skipped {SKIPPED[0]} {(time.time() - t0) / 60:.0f} min"
        print(line, flush=True); log.write(line + "\n"); log.flush()
    if step % A.eval_every == 0 or step == A.steps:
        ev = evaluate(); line = f"[eval {step}] " + " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in ev.items())
        print(line, flush=True); log.write(line + "\n"); log.flush()
        if A.page and ev["page_top1"] > BEST[0]:          # the best-so-far weights by held-out page top-1
            BEST[0] = ev["page_top1"]; save("best"); print(f"[best] step {step} page_top1 {BEST[0]:.3f}", flush=True); log.write(f"[best] step {step} page_top1 {BEST[0]:.3f}\n"); log.flush()
    if step % A.save_every == 0 or step == A.steps:
        save(); print(f"[save] step {step}", flush=True)
print("TRAIN_DONE", flush=True)
