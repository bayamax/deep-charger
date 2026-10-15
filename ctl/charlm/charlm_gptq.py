#!/usr/bin/env python3
"""The character model to 4 bits by GPTQ on the app's grid (affine, groups of 64 along the input, fp16 scale and bias
per group - q4.affine_params), block by block, each matrix against the inputs it sees with the earlier ones already
quantized. The 97-row character table and the position table stay fp16 (together under 0.3 MB at d 256), the output head
is quantized against the final hidden states. Reports bits per character on WikiText-2 test before and after, for
round-to-nearest too, and the packed size.

  python3 charlm_gptq.py --ckpt run1/model_latest.pt --calib wikitext-103 parquet --test wikitext-2 test parquet --out run1
"""
import argparse, json, math, os, sys, time, types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import q4  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--calib", required=True); ap.add_argument("--test", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=64, help="calibration windows"); ap.add_argument("--group", type=int, default=64); ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--block", type=int, default=128); ap.add_argument("--damp", type=float, default=0.01); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--test-chars", type=int, default=0, help="0: the whole test set")
X = ap.parse_args(); torch.set_num_threads(X.threads); torch.manual_seed(0); np.random.seed(0)
N_BINS = float((1 << X.bits) - 1)

# ---- the model, as charlm.py defines it (its argparse namespace rebuilt from the checkpoint) ---------------------------
CK = torch.load(X.ckpt, map_location="cpu", weights_only=False)
sys.argv = ["charlm.py", "--train", "", "--test", "", "--out", X.out] + [f"--{k.replace('_', '-')}={v}" for k, v in CK["args"].items() if k in ("d", "heads", "ff", "seq", "layers")]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "charlm.py")).read()
head_src = src[: src.index("import pandas as pd")]; model_src = src[src.index("# ---- the model"): src.index("DEPTHS = ")]
ns = {}; exec(head_src.replace('LOG = open(os.path.join(A.out, "train.log"), "a")', "LOG = None").replace("os.makedirs(A.out, exist_ok=True)", ""), ns)
ns["out"] = lambda s: print(s, flush=True); exec(model_src, ns)
A = ns["A"]; CharLM, encode, bpc_fn = ns["CharLM"], ns["encode"], ns["bpc"]
model = CharLM(CK["layers"]); model.load_state_dict(CK["model"]); model.eval()
import pandas as pd  # noqa: E402
TEST = encode("\n\n".join(pd.read_parquet(X.test)["text"].tolist()))
if X.test_chars: TEST = TEST[: X.test_chars]
tr = pd.read_parquet(X.calib)["text"].tolist(); CAL = encode("".join(tr[len(tr) // 3: len(tr) // 3 + 20000]))
starts = np.random.randint(0, len(CAL) - A.seq - 1, X.n); ids = torch.from_numpy(np.stack([CAL[j:j + A.seq] for j in starts]).astype(np.int64))
LOG = open(os.path.join(X.out, "gptq.log"), "a")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
out(f"[gptq] {CK['layers']} layers, {model.n_params() / 1e6:.2f}M params, {X.n} calibration windows x {A.seq}")
t = time.time(); b0 = bpc_fn(model, TEST); out(f"[score] float: {b0:.3f} bpc [{(time.time() - t) / 60:.1f} min]")

LIN = ["qkv", "proj", "fc", "fc2"]


def gptq_linear(W, H):
    W = W.clone(); rows, cols = W.shape
    dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
    H += torch.eye(cols) * X.damp * torch.mean(torch.diag(H))
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W); codes = torch.zeros(rows, cols, dtype=torch.uint8); ng = cols // X.group
    scales = torch.zeros(rows, ng); biases = torch.zeros(rows, ng); s = b = None
    for i1 in range(0, cols, X.block):
        i2 = min(i1 + X.block, cols); W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % X.group == 0:
                g = col // X.group; _, s_, b_ = q4.affine_params(torch.cat([W1[:, i:], W[:, i2:]], 1)[:, :X.group], X.group, X.bits)
                s = s_.reshape(rows); b = b_.reshape(rows); scales[:, g] = s; biases[:, g] = b
            w = W1[:, i]; d = Hi[i, i]; c = torch.clamp(torch.round((w - b) / s), 0.0, N_BINS); q = c * s + b
            Q1[:, i] = q; codes[:, col] = c.to(torch.uint8); err = (w - q) / d; W1[:, i:] -= err[:, None] * Hi[i, i:][None]; E1[:, i] = err
        Q[:, i1:i2] = Q1; W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q, codes, scales, biases


def rtn(W):
    q, s_, b_ = q4.affine_params(W, X.group, X.bits); return (q * s_ + b_).reshape(W.shape)


def quantize(mode):
    m = CharLM(CK["layers"]); m.load_state_dict(CK["model"]); m.eval(); PACK = {}
    with torch.no_grad():
        x = m.emb(ids) + m.pos(torch.arange(A.seq))
        for li, blk in enumerate(m.blocks):
            for n in LIN:
                lin = getattr(blk, n)
                if mode == "rtn": lin.weight.data = rtn(lin.weight.data); continue
                H = torch.zeros(lin.in_features, lin.in_features); cnt = [0]
                def hook(mod, inp, o): z = inp[0].reshape(-1, inp[0].shape[-1]); H.add_(z.T @ z); cnt[0] += z.shape[0]
                h = lin.register_forward_hook(hook); blk(x); h.remove()
                Q, codes, s_, b_ = gptq_linear(lin.weight.data.clone(), H / cnt[0]); lin.weight.data = Q; PACK[f"blocks.{li}.{n}"] = (codes, s_, b_)
            x = blk(x)
        if mode == "rtn": m.head.weight.data = rtn(m.head.weight.data)
        else:
            z = m.ln_f(x).reshape(-1, A.d); Q, codes, s_, b_ = gptq_linear(m.head.weight.data.clone(), z.T @ z / z.shape[0]); m.head.weight.data = Q; PACK["head"] = (codes, s_, b_)
    return m, PACK


t = time.time(); mr, _ = quantize("rtn"); br = bpc_fn(mr, TEST); out(f"[score] round-to-nearest {X.bits}-bit: {br:.3f} bpc")
t = time.time(); mg, PACK = quantize("gptq"); out(f"[gptq] done [{(time.time() - t) / 60:.1f} min]"); bg = bpc_fn(mg, TEST); out(f"[score] GPTQ {X.bits}-bit: {bg:.3f} bpc")

# ---- the packed file: 4-bit codes 8 to a uint32, fp16 scales and biases, fp16 for everything not quantized -----------------
shifts = torch.arange(0, 32, X.bits, dtype=torch.int64)
pack = lambda c: ((c.to(torch.int64).reshape(c.shape[0], -1, 32 // X.bits) << shifts).sum(-1) & 0xFFFFFFFF).to(torch.int32)
sd = {}; nbytes = 0
for k, v in mg.state_dict().items():
    key = k[: -len(".weight")] if k.endswith(".weight") else None
    if key in PACK:
        c, s_, b_ = PACK[key]; sd[k] = pack(c); sd[key + ".scales"] = s_.half(); sd[key + ".biases"] = b_.half()
    else:
        sd[k] = v.half()
for v in sd.values(): nbytes += v.numel() * v.element_size()
torch.save({"state": sd, "layers": CK["layers"], "args": CK["args"], "chars": CK["chars"], "quantization": {"group_size": X.group, "bits": X.bits, "mode": "affine"}}, os.path.join(X.out, f"model_{X.bits}bit.pt"))
torch.save({"model": mg.state_dict(), "layers": CK["layers"], "args": CK["args"], "chars": CK["chars"]}, os.path.join(X.out, f"model_{X.bits}bit_dequant.pt"))
res = {"float_bpc": b0, "rtn_bpc": br, "gptq_bpc": bg, "packed_bytes": nbytes, "float_bytes": 4 * mg.n_params(), "params": mg.n_params(), "layers": CK["layers"]}
json.dump(res, open(os.path.join(X.out, "gptq_result.json"), "w"), indent=1)
out(f"[out] packed {nbytes / 1e6:.2f} MB (float32 {4 * mg.n_params() / 1e6:.2f} MB): float {b0:.3f} / RTN {br:.3f} / GPTQ {bg:.3f} bpc")
out("CHARLM_GPTQ_DONE")
