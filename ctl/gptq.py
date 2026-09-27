#!/usr/bin/env python3
"""GPTQ onto the app's 4-bit affine grid: the codes are chosen to preserve each layer's outputs, not its weights.

Round-to-nearest (what mlx_lm.convert does, and what every arm so far started from) picks each weight's
code on its own. GPTQ (Frantar et al.) quantizes a weight matrix one input column at a time and pushes
each column's rounding error onto the columns not yet quantized, weighted by the inverse Hessian of the
layer's inputs (H = X^T X over calibration tokens), so what is preserved is W x on the data the layer
actually sees. The grid is unchanged - per row, groups of 64 along the input, fp16 scale and bias, the
larger-magnitude edge exactly representable, as q4.affine_params defines it and the app's loader
expects - so the result packs into the same directory. Layers are done in order, each seeing the outputs
of the already-quantized layers before it. The embedding table has no input Hessian and is rounded to
nearest; lm_head is quantized against the final hidden states.

Writes the dequantized directory the evaluator measures (--out-hf) and the packed one the app loads
(--out-mlx); checkmlx.py proves they agree.

  python3 gptq.py --base /root/reeval_hf_g14m --data /root/work/qcal_q14.jsonl --out-hf /root/gptq_hf --out-mlx /root/gptq_mlx4
"""
import argparse, json, math, os, random, shutil, sys, time

import numpy as np
import torch
import torch.nn as nn
from safetensors.numpy import save_file as save_np
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q4  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--out-hf", required=True); ap.add_argument("--out-mlx", required=True)
ap.add_argument("--n", type=int, default=128, help="calibration sequences"); ap.add_argument("--len", type=int, default=1024)
ap.add_argument("--group", type=int, default=64); ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--block", type=int, default=128); ap.add_argument("--damp", type=float, default=0.01)
ap.add_argument("--batch", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--skip", default="", help="comma-separated leaf modules left in float (e.g. embed_tokens,lm_head)")
ap.add_argument("--state", default="", help="also save {name: (codes, scales, biases)} here, for jointfit --codes-from and packmlx --codes-from")
ap.add_argument("--no-dirs", type=int, default=0, help="1: write only the state (the directories are 4.7 GB the disk may not have)")
A = ap.parse_args()
random.seed(A.seed); torch.manual_seed(A.seed)
DEV = "cuda"
SKIP = set(x for x in A.skip.split(",") if x)
N_BINS = float((1 << A.bits) - 1)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(A.base)
model = AutoModelForCausalLM.from_pretrained(A.base, torch_dtype=torch.bfloat16).to(DEV).eval()
model.config.use_cache = False
BODY, HEAD = model.model, model.lm_head
LAYERS = BODY.layers

# ---- calibration tokens: the lineage's own traces, cut to --len ------------------------------------------
texts = [json.loads(l)["text"] for l in open(A.data) if l.strip()]
random.shuffle(texts)
stream = []
for t in texts:
    stream += tok.encode(t, add_special_tokens=False) + [tok.eos_token_id]
    if len(stream) >= A.n * A.len * 2:
        break
starts = random.sample(range(0, len(stream) - A.len), A.n)        # contiguous windows of the trace stream
ids = torch.stack([torch.tensor(stream[s0:s0 + A.len]) for s0 in starts])
L = A.len
print(f"[gptq] {len(ids)} calibration windows x {L} tokens from {len(stream)} trace tokens", flush=True)


# ---- the inputs of layer 0, caught on the way in ------------------------------------------------------------
class Caught(Exception):
    pass


class Catch(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m; self.inps = []; self.kw = None

    def forward(self, x, *a, **kw):
        self.inps.append(x.detach()); self.kw = kw
        raise Caught


LAYERS[0] = Catch(LAYERS[0])
with torch.no_grad():
    for s0 in range(0, len(ids), A.batch):
        try:
            model(input_ids=ids[s0:s0 + A.batch].to(DEV))
        except Caught:
            pass
KW = LAYERS[0].kw; X = torch.cat(LAYERS[0].inps)   # (N, L, H) bf16
LAYERS[0] = LAYERS[0].m
KW = {k: v for k, v in KW.items() if k in ("attention_mask", "position_ids", "cache_position", "position_embeddings")}
print(f"[gptq] layer inputs {tuple(X.shape)}; kwargs {list(KW)}", flush=True)


def kw_for(b):
    out = {}
    for k, v in KW.items():
        if k == "position_embeddings" and v is not None:
            out[k] = tuple(t[:b] if t.shape[0] != 1 else t for t in v)
        elif torch.is_tensor(v) and v.dim() > 0 and v.shape[0] not in (1,) and v.shape[0] == A.batch:
            out[k] = v[:b]
        else:
            out[k] = v
    return out


# ---- GPTQ for one linear -----------------------------------------------------------------------------------
def gptq_linear(W, H):
    """W (rows, in) float32, H (in, in) float32 -> dequantized W, codes (rows, in) uint8, scales/biases (rows, in/group)."""
    W = W.clone(); rows, cols = W.shape
    dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0
    damp = A.damp * torch.mean(torch.diag(H))
    H += torch.eye(cols, device=DEV) * damp
    Hc = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(Hc)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W); codes = torch.zeros(rows, cols, dtype=torch.uint8, device=DEV)
    ng = cols // A.group
    scales = torch.zeros(rows, ng, device=DEV); biases = torch.zeros(rows, ng, device=DEV)
    s = b = None
    for i1 in range(0, cols, A.block):
        i2 = min(i1 + A.block, cols); cnt = i2 - i1
        W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(cnt):
            col = i1 + i
            if col % A.group == 0:   # the group's grid, from the columns as they stand after the error updates so far
                g = col // A.group
                _, s_, b_ = q4.affine_params(torch.cat([W1[:, i:], W[:, i2:]], 1)[:, :A.group], A.group, A.bits)
                s = s_.reshape(rows); b = b_.reshape(rows)
                scales[:, g] = s; biases[:, g] = b
            w = W1[:, i]; d = Hinv1[i, i]
            c = torch.clamp(torch.round((w - b) / s), 0.0, N_BINS)
            q = c * s + b
            Q1[:, i] = q; codes[:, col] = c.to(torch.uint8)
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            E1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q, codes, scales, biases


def gptq_rows(W, H, chunk=16384):
    """gptq_linear over row chunks: the rows are independent given H, and the 152k-row lm_head in float is 0.9 GB a copy."""
    if W.shape[0] <= chunk:
        return gptq_linear(W, H.clone())
    parts = [gptq_linear(W[r0:r0 + chunk], H.clone()) for r0 in range(0, W.shape[0], chunk)]
    return tuple(torch.cat([p[i] for p in parts]) for i in range(4))


def rtn(W, chunk=16384):
    outs = []
    for r0 in range(0, W.shape[0], chunk):
        w = W[r0:r0 + chunk]
        q, s_, b_ = q4.affine_params(w, A.group, A.bits)
        outs.append(((q * s_ + b_).reshape(w.shape), q.reshape(w.shape[0], -1).to(torch.uint8), s_.reshape(w.shape[0], -1), b_.reshape(w.shape[0], -1)))
    return tuple(torch.cat([o[i] for o in outs]) for i in range(4))


PACK = {}      # name -> (codes uint8 cpu, scales, biases)
t0 = time.time()
ORDER = [["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], ["self_attn.o_proj"],
         ["mlp.gate_proj", "mlp.up_proj"], ["mlp.down_proj"]]
with torch.no_grad():
    for li, layer in enumerate(LAYERS):
        mods = {n: layer.get_submodule(n) for grp in ORDER for n in grp}
        Hs = {n: torch.zeros(m.in_features, m.in_features, device=DEV) for n, m in mods.items()}
        cnt = {n: 0 for n in mods}
        hooks = []
        for n, m in mods.items():
            def hook(mod, inp, out, _n=n):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                Hs[_n] += x.t() @ x; cnt[_n] += x.shape[0]
            hooks.append(m.register_forward_hook(hook))
        for s0 in range(0, X.shape[0], A.batch):
            xb = X[s0:s0 + A.batch]
            layer(xb, **kw_for(xb.shape[0]))
        for h in hooks:
            h.remove()
        for grp in ORDER:
            for n in grp:
                m = mods[n]; leaf = n.split(".")[-1]
                if leaf in SKIP:
                    continue
                H = Hs[n] / max(cnt[n], 1)
                Q, codes, s_, b_ = gptq_rows(m.weight.data.float(), H)
                del H
                m.weight.data = Q.to(m.weight.dtype)
                PACK[f"model.layers.{li}.{n}.weight"] = (codes.cpu(), s_.cpu(), b_.cpu())
        del Hs
        # the next layer sees this layer as it will run on the phone
        outs = []
        for s0 in range(0, X.shape[0], A.batch):
            xb = X[s0:s0 + A.batch]
            outs.append(layer(xb, **kw_for(xb.shape[0]))[0])
        X = torch.cat(outs)
        print(f"[gptq] layer {li} done ({time.time()-t0:.0f}s)", flush=True)
    # lm_head against the final hidden states; the embedding table to nearest
    if "lm_head" not in SKIP:
        Hh = torch.zeros(HEAD.in_features, HEAD.in_features, device=DEV); n_tok = 0
        for s0 in range(0, X.shape[0], A.batch):
            x = BODY.norm(X[s0:s0 + A.batch]).reshape(-1, HEAD.in_features).float()
            Hh += x.t() @ x; n_tok += x.shape[0]
        del X; torch.cuda.empty_cache()
        Q, codes, s_, b_ = gptq_rows(HEAD.weight.data.float(), Hh / n_tok)
        HEAD.weight.data = Q.to(HEAD.weight.dtype); PACK["lm_head.weight"] = (codes.cpu(), s_.cpu(), b_.cpu())
        del Q; torch.cuda.empty_cache()
        print(f"[gptq] lm_head done", flush=True)
    else:
        del X
    if "embed_tokens" not in SKIP:
        E = BODY.embed_tokens
        Q, codes, s_, b_ = rtn(E.weight.data.float())
        E.weight.data = Q.to(E.weight.dtype); PACK["model.embed_tokens.weight"] = (codes.cpu(), s_.cpu(), b_.cpu())
        del Q; torch.cuda.empty_cache()

# ---- the state first (small), then the two directories --------------------------------------------------------
if A.state:
    torch.save({"q": {k: (c, s_, b_) for k, (c, s_, b_) in PACK.items()}, "group": A.group, "bits": A.bits}, A.state)
    print(f"[out] {A.state}: codes of {len(PACK)} tensors", flush=True)
if A.no_dirs:
    print("GPTQ_DONE", flush=True); sys.exit(0)
os.makedirs(A.out_hf, exist_ok=True); os.makedirs(A.out_mlx, exist_ok=True)
sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
save_file(sd, os.path.join(A.out_hf, "model.safetensors"), metadata={"format": "pt"})
for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
    if os.path.exists(os.path.join(A.base, f)):
        shutil.copy(os.path.join(A.base, f), os.path.join(A.out_hf, f))
        if f != "config.json":
            shutil.copy(os.path.join(A.base, f), os.path.join(A.out_mlx, f))
print(f"[out] {A.out_hf}: {len(sd)} tensors, {len(PACK)} quantized", flush=True)

shifts = torch.arange(0, 32, A.bits, dtype=torch.int64)
mx = {}
for k, v in sd.items():
    if k in PACK:
        codes, s_, b_ = PACK[k]
        rows = codes.shape[0]
        packed = (codes.to(torch.int64).reshape(rows, -1, 32 // A.bits) << shifts).sum(-1)
        mx[k] = (packed & 0xFFFFFFFF).numpy().astype("uint32")
        mx[k[:-len(".weight")] + ".scales"] = s_.to(torch.float16).numpy()
        mx[k[:-len(".weight")] + ".biases"] = b_.to(torch.float16).numpy()
    else:
        mx[k] = v.to(torch.float16).numpy()
save_np(mx, os.path.join(A.out_mlx, "model.safetensors"))
cfg = json.load(open(os.path.join(A.base, "config.json")))
cfg["quantization"] = cfg["quantization_config"] = {"group_size": A.group, "bits": A.bits, "mode": "affine"}
cfg["torch_dtype"] = "float16"
json.dump(cfg, open(os.path.join(A.out_mlx, "config.json"), "w"), indent=2)
print(f"[out] {A.out_mlx}: packed", flush=True)
print("GPTQ_DONE", flush=True)
