#!/usr/bin/env python3
"""GPT-2 to 4 bits by GPTQ, on the same grid as the app's model: MLX affine, groups of 64 along the input, fp16 scale and
bias per group (q4.affine_params). Runs on the CPU (GPT-2 is small enough).

Each block is done in order and sees the already-quantized blocks before it; inside a block the four matrices go one at a
time (c_attn, attn.c_proj, c_fc, mlp.c_proj), each against the inputs it gets with the earlier ones already quantized.
The tied token table is quantized once, by GPTQ against the final hidden states it scores as the output head (the input
lookup reads the same quantized rows, as MLX's tied head does); the position table stays fp16.

Writes the packed directory mlx-lm loads (--out-mlx) and the same numbers dequantized into a plain HF directory
(--out-hf), and scores the original, round-to-nearest and GPTQ on WikiText-2 test perplexity and next-token top-1.

  python3 gpt2_gptq.py --calib wikitext-103 train parquet --ppl wikitext-2 test parquet --out-mlx gpt2-4bit-gptq --out-hf gpt2-gptq-hf
"""
import argparse, json, math, os, random, shutil, sys, time

import numpy as np
import torch
import torch.nn as nn
from safetensors.numpy import save_file as save_np
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q4  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gpt2"); ap.add_argument("--calib", required=True); ap.add_argument("--ppl", required=True)
ap.add_argument("--out-mlx", required=True); ap.add_argument("--out-hf", required=True)
ap.add_argument("--n", type=int, default=64, help="calibration windows"); ap.add_argument("--len", type=int, default=1024)
ap.add_argument("--group", type=int, default=64); ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--block", type=int, default=128); ap.add_argument("--damp", type=float, default=0.01)
ap.add_argument("--batch", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=4)
A = ap.parse_args()
random.seed(A.seed); torch.manual_seed(A.seed); torch.set_num_threads(A.threads)
N_BINS = float((1 << A.bits) - 1)

import pandas as pd  # noqa: E402
from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # noqa: E402
tok = GPT2TokenizerFast.from_pretrained(A.model)
load = lambda: GPT2LMHeadModel.from_pretrained(A.model, torch_dtype=torch.float32).eval()
model = load(); model.config.use_cache = False
T = model.transformer; BLOCKS = T.h

# ---- calibration: contiguous windows of general text ---------------------------------------------------------------
texts = [t for t in pd.read_parquet(A.calib)["text"].tolist() if len(t.strip()) > 0]
start = random.randrange(0, len(texts) - 200000)
stream = tok.encode("".join(texts[start:start + 200000]))
starts = random.sample(range(0, len(stream) - A.len), A.n)
ids = torch.stack([torch.tensor(stream[s0:s0 + A.len]) for s0 in starts])
print(f"[gptq] {A.n} calibration windows x {A.len} tokens from {len(stream)} WikiText-103 tokens", flush=True)


# ---- the inputs of block 0 and the keyword arguments the model passes its blocks ----------------------------------
class Caught(Exception):
    pass


class Catch(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m; self.inps = []; self.kw = None

    def forward(self, x, *a, **kw):
        self.inps.append(x.detach()); self.kw = kw
        raise Caught


BLOCKS[0] = Catch(BLOCKS[0])
with torch.no_grad():
    for s0 in range(0, len(ids), A.batch):
        try:
            model(input_ids=ids[s0:s0 + A.batch])
        except Caught:
            pass
KW = {k: v for k, v in BLOCKS[0].kw.items() if k in ("attention_mask", "position_ids", "cache_position", "position_embeddings")}
X = torch.cat(BLOCKS[0].inps); BLOCKS[0] = BLOCKS[0].m
print(f"[gptq] block inputs {tuple(X.shape)}; kwargs {list(KW)}", flush=True)
first = lambda o: o[0] if isinstance(o, tuple) else o


def kw_for(b):
    out = {}
    for k, v in KW.items():
        out[k] = v[:b] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == A.batch and b != A.batch else v
    return out


def run_block(blk, X):
    return torch.cat([first(blk(X[s0:s0 + A.batch], **kw_for(len(X[s0:s0 + A.batch])))) for s0 in range(0, len(X), A.batch)])


with torch.no_grad():   # the blocks run by hand reproduce the model (the causal mask is among the kwargs)
    h = X[:2]
    for blk in BLOCKS: h = first(blk(h, **kw_for(2)))
    ref = model(input_ids=ids[:2]).logits; mine = model.lm_head(T.ln_f(h))
    print(f"[gptq] block-by-block check: max |logit diff| {float((ref - mine).abs().max()):.2e}", flush=True)


# ---- GPTQ for one matrix (rows = outputs, cols = inputs), the app's grid -------------------------------------------
def gptq_linear(W, H):
    W = W.clone(); rows, cols = W.shape
    dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
    H += torch.eye(cols) * A.damp * torch.mean(torch.diag(H))
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W); codes = torch.zeros(rows, cols, dtype=torch.uint8)
    ng = cols // A.group; scales = torch.zeros(rows, ng); biases = torch.zeros(rows, ng); s = b = None
    for i1 in range(0, cols, A.block):
        i2 = min(i1 + A.block, cols); W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % A.group == 0:   # the group's grid from the columns as they stand after the error updates so far
                g = col // A.group
                _, s_, b_ = q4.affine_params(torch.cat([W1[:, i:], W[:, i2:]], 1)[:, :A.group], A.group, A.bits)
                s = s_.reshape(rows); b = b_.reshape(rows); scales[:, g] = s; biases[:, g] = b
            w = W1[:, i]; d = Hi[i, i]
            c = torch.clamp(torch.round((w - b) / s), 0.0, N_BINS); q = c * s + b
            Q1[:, i] = q; codes[:, col] = c.to(torch.uint8)
            err = (w - q) / d; W1[:, i:] -= err[:, None] * Hi[i, i:][None]; E1[:, i] = err
        Q[:, i1:i2] = Q1; W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q, codes, scales, biases


def gptq_rows(W, H, chunk=8192):
    parts = [gptq_linear(W[r0:r0 + chunk], H.clone()) for r0 in range(0, W.shape[0], chunk)]
    return tuple(torch.cat([p[i] for p in parts]) for i in range(4))


def rtn(W):
    q, s_, b_ = q4.affine_params(W, A.group, A.bits)
    return (q * s_ + b_).reshape(W.shape), q.reshape(W.shape[0], -1).to(torch.uint8), s_.reshape(W.shape[0], -1), b_.reshape(W.shape[0], -1)


# ---- scoring: WikiText-2 test perplexity (1024-token windows, stride 512) and next-token top-1 ----------------------
ppl_ids = torch.tensor(tok.encode("\n\n".join(pd.read_parquet(A.ppl)["text"].tolist())))


@torch.no_grad()
def score(m, max_windows=None):
    nll = 0.0; n = 0; hit = 0; prev = 0; w = 0
    for b in range(0, len(ppl_ids), 512):
        e = min(b + 1024, len(ppl_ids)); x = ppl_ids[max(0, e - 1024):e][None]; trg = e - prev
        lo = m(input_ids=x).logits[0]                                   # only the tokens no earlier window scored
        lg, y = (lo[-trg - 1:-1], x[0, -trg:]) if trg < x.shape[1] else (lo[:-1], x[0, 1:])
        nll += float(nn.functional.cross_entropy(lg, y, reduction="sum")); hit += int((lg.argmax(-1) == y).sum()); n += len(y)
        prev = e; w += 1
        if e == len(ppl_ids) or (max_windows and w >= max_windows): break
    return math.exp(nll / n), hit / n, n


CONV = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]


def rtn_model():
    m = load()
    with torch.no_grad():
        for blk in m.transformer.h:
            for n in CONV:
                c = blk.get_submodule(n); c.weight.data = rtn(c.weight.data.T.contiguous())[0].T.contiguous()
        m.transformer.wte.weight.data = rtn(m.transformer.wte.weight.data)[0]
    return m


t0 = time.time()
base = score(model); print(f"[score] original fp32: WikiText-2 perplexity {base[0]:.2f}, top-1 {base[1]:.3f} ({base[2]} tokens) [{(time.time()-t0)/60:.1f} min]", flush=True)
r = score(rtn_model()); print(f"[score] round-to-nearest 4-bit: perplexity {r[0]:.2f}, top-1 {r[1]:.3f}", flush=True)

# ---- GPTQ, block by block, matrix by matrix ----------------------------------------------------------------------------
PACK = {}; t0 = time.time()
with torch.no_grad():
    for li, blk in enumerate(BLOCKS):
        for n in CONV:
            c = blk.get_submodule(n); H = torch.zeros(c.weight.shape[0], c.weight.shape[0]); cnt = [0]
            def hook(mod, inp, out):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float(); H.add_(x.T @ x); cnt[0] += x.shape[0]
            hd = c.register_forward_hook(hook); run_block(blk, X); hd.remove()
            Q, codes, s_, b_ = gptq_rows(c.weight.data.T.contiguous(), H / cnt[0])     # Conv1D keeps [in, out]
            c.weight.data = Q.T.contiguous(); PACK[f"h.{li}.{n}"] = (codes, s_, b_)
        X = run_block(blk, X)
        print(f"[gptq] block {li} done ({time.time()-t0:.0f}s)", flush=True)
    Hh = torch.zeros(768, 768); n_tok = 0
    for s0 in range(0, len(X), A.batch):
        x = T.ln_f(X[s0:s0 + A.batch]).reshape(-1, 768); Hh += x.T @ x; n_tok += x.shape[0]
    Q, codes, s_, b_ = gptq_rows(T.wte.weight.data.clone(), Hh / n_tok)
    T.wte.weight.data = Q; PACK["wte"] = (codes, s_, b_)   # tied: lm_head reads the same tensor
    print(f"[gptq] token table done as the output head ({time.time()-t0:.0f}s)", flush=True)
g = score(model); print(f"[score] GPTQ 4-bit: perplexity {g[0]:.2f}, top-1 {g[1]:.3f}", flush=True)

# ---- the two directories ---------------------------------------------------------------------------------------------
os.makedirs(A.out_hf, exist_ok=True); os.makedirs(A.out_mlx, exist_ok=True)
model.save_pretrained(A.out_hf); tok.save_pretrained(A.out_hf)
shifts = torch.arange(0, 32, A.bits, dtype=torch.int64)
pack = lambda codes: ((codes.to(torch.int64).reshape(codes.shape[0], -1, 32 // A.bits) << shifts).sum(-1) & 0xFFFFFFFF).numpy().astype("uint32")
f16 = lambda t: t.detach().to(torch.float16).contiguous().numpy()
mx = {"model.wpe.weight": f16(T.wpe.weight), "model.ln_f.weight": f16(T.ln_f.weight), "model.ln_f.bias": f16(T.ln_f.bias)}
c_, s_, b_ = PACK["wte"]; mx["model.wte.weight"] = pack(c_); mx["model.wte.scales"] = f16(s_); mx["model.wte.biases"] = f16(b_)
for li, blk in enumerate(BLOCKS):
    for ln in ("ln_1", "ln_2"):
        mx[f"model.h.{li}.{ln}.weight"] = f16(blk.get_submodule(ln).weight); mx[f"model.h.{li}.{ln}.bias"] = f16(blk.get_submodule(ln).bias)
    for n in CONV:   # MLX keeps Linear as [out, in]: the packed codes are already in that orientation
        c_, s_, b_ = PACK[f"h.{li}.{n}"]; k = f"model.h.{li}.{n}"
        mx[k + ".weight"] = pack(c_); mx[k + ".scales"] = f16(s_); mx[k + ".biases"] = f16(b_); mx[k + ".bias"] = f16(blk.get_submodule(n).bias)
save_np(mx, os.path.join(A.out_mlx, "model.safetensors"))
cfg = model.config.to_dict(); cfg["quantization"] = {"group_size": A.group, "bits": A.bits, "mode": "affine"}; cfg["torch_dtype"] = "float16"
json.dump(cfg, open(os.path.join(A.out_mlx, "config.json"), "w"), indent=2); tok.save_pretrained(A.out_mlx)
json.dump({"original": base, "rtn4": r, "gptq4": g}, open(os.path.join(A.out_mlx, "scores.json"), "w"), indent=1)
print(f"[out] {A.out_hf} (dequantized) and {A.out_mlx} (packed, {os.path.getsize(os.path.join(A.out_mlx, 'model.safetensors'))/1e6:.0f} MB)", flush=True)
print("GPT2_GPTQ_DONE", flush=True)
