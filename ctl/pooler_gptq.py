#!/usr/bin/env python3
"""The pooler onto the app's 4-bit affine grid, by the method that worked for the model itself (gptq.py).

GPTQ (Frantar et al.): each weight matrix is quantized one input column at a time, the rounding error pushed onto the
columns not yet quantized through the inverse Hessian of that matrix's own inputs (H = X^T X over calibration data),
so what is preserved is W x on the inputs the pooler actually sees. Same grid as the model: per row, groups of 64
along the input, fp16 scale and bias (q4.affine_params). Blocks in order, and inside a block the matrices in the
order the data meets them, each calibrated on the outputs of the already-quantized ones before it.

The pooler (docs/iphone_agent_v2_spec.md 3-1): query [32, 1536]; three blocks of cross attention (query lnq1(q),
key/value lnk(past)), self attention (lnq2(q)), FFN (lnq3(q) -> ffn.0 -> GELU -> ffn.2), each residual; out
ln_out(q), L2-normalised, times |out_scale|. The packed in_proj of the cross attention is two matrices for GPTQ: its
query rows see lnq1(q) and its key/value rows see lnk(past), so each part gets its own Hessian. The 18 matrices
(99.9% of the weights) are quantized; query, layernorms, biases and out_scale stay in float.

Calibration: the pooler's real input - the model's (4-bit) embeddings of past trace tokens plus the sinusoidal
positions - as windows of 32..384 tokens cut from the lineage's traces. Held-out windows measure the result: the
cosine between the float pooler's 32 output vectors and the quantized one's.

Writes  <out>/pooler_<method>_dq.safetensors   dequantized float32, same 64 bare keys (the evaluator loads it)
        <out>/pooler_<method>_mlx.safetensors  the app's form: each quantized matrix K as uint32 codes (MLX layout,
                                                8 per word) with K.scales / K.biases in fp16, the rest as fp16
  usage: python3 pooler_gptq.py --pooler pooler.safetensors --model /root/g14q_hf --data calib.jsonl --out /root/pq --method gptq
"""
import argparse, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from safetensors.numpy import save_file as save_np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q4  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pooler", required=True); ap.add_argument("--model", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--method", default="gptq", choices=["gptq", "rtn"])
ap.add_argument("--data2", default="", help="reasoning records {q, thinking, reply} (the pooler also compresses thinking, not only searches)")
ap.add_argument("--mix", type=float, default=0.5, help="share of the windows cut from --data2")
ap.add_argument("--tag", default="", help="suffix of the output names")
ap.add_argument("--n", type=int, default=512, help="calibration windows"); ap.add_argument("--n-eval", type=int, default=128)
ap.add_argument("--minlen", type=int, default=32); ap.add_argument("--maxlen", type=int, default=384)
ap.add_argument("--group", type=int, default=64); ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--block", type=int, default=128); ap.add_argument("--damp", type=float, default=0.01)
ap.add_argument("--heads", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
random.seed(A.seed); torch.manual_seed(A.seed)
DEV = "cuda"; N_BINS = float((1 << A.bits) - 1)
os.makedirs(A.out, exist_ok=True)

P = {k: v.float().to(DEV) for k, v in load_file(A.pooler).items()}
P0 = {k: v.clone() for k, v in P.items()}          # the float pooler, the reference
NB = 1 + max(int(k.split(".")[1]) for k in P if k.startswith("blocks."))
D = P["query"].shape[1]; HD = D // A.heads

# ---- calibration windows: the model's own (quantized) embeddings of trace tokens ----
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(A.model)
emb_w = None
for f in sorted(os.listdir(A.model)):
    if f.endswith(".safetensors"):
        sd = load_file(os.path.join(A.model, f))
        if "model.embed_tokens.weight" in sd: emb_w = sd["model.embed_tokens.weight"].float(); break
assert emb_w is not None, "no embed_tokens in the model directory"
def make_stream(texts, need):
    random.shuffle(texts); st = []
    for t in texts:
        st += tok.encode(t, add_special_tokens=False)
        if len(st) > need: break
    return st


n2 = int(round(A.mix * A.n)) if A.data2 else 0; e2 = int(round(A.mix * A.n_eval)) if A.data2 else 0
stream = make_stream([json.loads(l)["text"] for l in open(A.data) if l.strip()], (A.n - n2 + A.n_eval - e2) * A.maxlen * 3)
stream2 = []
if A.data2:
    recs = [json.loads(l) for l in open(A.data2) if l.strip()]
    stream2 = make_stream([f"<｜User｜>{r['q']}<｜Assistant｜><think>\n{r.get('thinking', '')}\n</think>\n\n{r['reply']}" for r in recs if r.get("q") and r.get("reply")],
                          (n2 + e2) * A.maxlen * 3)


def pe(L):
    pos = torch.arange(L, dtype=torch.float32)[:, None]; i = torch.arange(0, D, 2, dtype=torch.float32)
    ang = pos / torch.pow(10000.0, i / D); out = torch.zeros(L, D)
    out[:, 0::2] = torch.sin(ang); out[:, 1::2] = torch.cos(ang)
    return out


def windows(n, st):
    out = []
    for _ in range(n):
        L = random.randint(A.minlen, A.maxlen); s0 = random.randint(0, len(st) - L - 1)
        ids = torch.tensor(st[s0:s0 + L])
        out.append((emb_w[ids] + pe(L)).to(DEV))
    return out


CAL = windows(A.n - n2, stream) + (windows(n2, stream2) if n2 else [])
EVS = windows(A.n_eval - e2, stream); EVR = windows(e2, stream2) if e2 else []
EVW = EVS + EVR
del emb_w
print(f"[pq] {len(CAL)} calibration ({n2} of them reasoning) + {len(EVW)} held-out windows ({len(EVR)} reasoning) ({A.minlen}..{A.maxlen} tokens) from {len(stream)} trace tokens; "
      f"pooler {NB} blocks, d {D}, {A.heads} heads; method {A.method}", flush=True)


# ---- the pooler, written out so every matrix's input can be caught ----
def ln(x, k):
    return F.layer_norm(x, (D,), P[k + ".weight"], P[k + ".bias"], 1e-5)


def mha(xq, xkv, pre, W, catch):
    """nn.MultiheadAttention layout: packed in_proj [3D, D] (+bias), out_proj; catch: name -> input to record"""
    w, b = W[pre + ".in_proj_weight"], W[pre + ".in_proj_bias"]
    if catch is not None:
        catch.setdefault(pre + ".in_q", []).append(xq); catch.setdefault(pre + ".in_kv", []).append(xkv)
    q = xq @ w[:D].T + b[:D]; k = xkv @ w[D:2 * D].T + b[D:2 * D]; v = xkv @ w[2 * D:].T + b[2 * D:]
    q = q.view(-1, A.heads, HD).transpose(0, 1); k = k.view(-1, A.heads, HD).transpose(0, 1); v = v.view(-1, A.heads, HD).transpose(0, 1)
    att = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(HD), dim=-1)
    o = (att @ v).transpose(0, 1).reshape(-1, D)
    if catch is not None: catch.setdefault(pre + ".out", []).append(o)
    return o @ W[pre + ".out_proj.weight"].T + W[pre + ".out_proj.bias"]


def block(q, past, bi, W, catch=None):
    p = f"blocks.{bi}"
    if past.shape[0] > 0:
        q = q + mha(ln(q, p + ".lnq1"), ln(past, p + ".lnk"), p + ".cross", W, catch)
    x = ln(q, p + ".lnq2"); q = q + mha(x, x, p + ".selfa", W, catch)
    x = ln(q, p + ".lnq3")
    if catch is not None: catch.setdefault(p + ".ffn0", []).append(x)
    h = F.gelu(x @ W[p + ".ffn.0.weight"].T + W[p + ".ffn.0.bias"])
    if catch is not None: catch.setdefault(p + ".ffn2", []).append(h)
    return q + h @ W[p + ".ffn.2.weight"].T + W[p + ".ffn.2.bias"]


def pooler(past, W):
    q = W["query"]
    for bi in range(NB): q = block(q, past, bi, W)
    return F.normalize(F.layer_norm(q, (D,), W["ln_out.weight"], W["ln_out.bias"], 1e-5), dim=-1) * W["out_scale"].abs()


# ---- GPTQ on one matrix (gptq.py's gptq_linear, unchanged) ----
def gptq_linear(W, H):
    W = W.clone(); rows, cols = W.shape
    dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0
    H += torch.eye(cols, device=DEV) * (A.damp * torch.mean(torch.diag(H)))
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W); codes = torch.zeros(rows, cols, dtype=torch.uint8, device=DEV)
    ng = cols // A.group
    scales = torch.zeros(rows, ng, device=DEV); biases = torch.zeros(rows, ng, device=DEV)
    s = b = None
    for i1 in range(0, cols, A.block):
        i2 = min(i1 + A.block, cols); cnt = i2 - i1
        W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(cnt):
            col = i1 + i
            if col % A.group == 0:
                g = col // A.group
                _, s_, b_ = q4.affine_params(torch.cat([W1[:, i:], W[:, i2:]], 1)[:, :A.group], A.group, A.bits)
                s = s_.reshape(rows).float(); b = b_.reshape(rows).float(); scales[:, g] = s; biases[:, g] = b
            w = W1[:, i]; d = Hinv1[i, i]
            c = torch.clamp(torch.round((w - b) / s), 0.0, N_BINS)
            q_ = c * s + b; Q1[:, i] = q_; codes[:, col] = c.to(torch.uint8)
            err = (w - q_) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0); E1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q, codes, scales, biases


def rtn(W):
    q, s_, b_ = q4.affine_params(W, A.group, A.bits)
    return (q * s_ + b_).reshape(W.shape).float(), q.reshape(W.shape[0], -1).to(torch.uint8), s_.reshape(W.shape[0], -1).float(), b_.reshape(W.shape[0], -1).float()


def hessian(xs):
    H = torch.zeros(D if xs[0].shape[-1] == D else xs[0].shape[-1], xs[0].shape[-1], device=DEV); n = 0
    for x in xs:
        x = x.reshape(-1, x.shape[-1]).float(); H += x.T @ x; n += x.shape[0]
    return H / max(n, 1)


PACK = {}
t0 = time.time()
with torch.no_grad():
    Q_in = [P["query"].clone() for _ in CAL]           # each window's query state entering the current block
    for bi in range(NB):
        p = f"blocks.{bi}"
        # (matrix key, row slice or None, catch name) in the order the data meets them
        steps = [(p + ".cross.in_proj_weight", slice(0, D), p + ".cross.in_q"), (p + ".cross.in_proj_weight", slice(D, 3 * D), p + ".cross.in_kv"),
                 (p + ".cross.out_proj.weight", None, p + ".cross.out"),
                 (p + ".selfa.in_proj_weight", None, p + ".selfa.in_q"), (p + ".selfa.out_proj.weight", None, p + ".selfa.out"),
                 (p + ".ffn.0.weight", None, p + ".ffn0"), (p + ".ffn.2.weight", None, p + ".ffn2")]
        parts = {}
        for key, rows, cname in steps:
            if A.method == "gptq":
                catch = {}
                for qs, past in zip(Q_in, CAL): block(qs, past, bi, P, catch)
                H = hessian(catch[cname]); del catch
            Wm = P[key][rows] if rows is not None else P[key]
            Qm, codes, s_, b_ = gptq_linear(Wm.float(), H) if A.method == "gptq" else rtn(Wm.float())
            if rows is not None:
                P[key][rows] = Qm; parts.setdefault(key, []).append((codes, s_, b_))
                if len(parts[key]) == 2:
                    PACK[key] = tuple(torch.cat([x[i] for x in parts[key]]).cpu() for i in range(3))
            else:
                P[key] = Qm; PACK[key] = (codes.cpu(), s_.cpu(), b_.cpu())
        Q_in = [block(qs, past, bi, P) for qs, past in zip(Q_in, CAL)]
        print(f"[pq] block {bi} done ({time.time() - t0:.0f}s)", flush=True)

    def score(W, EVW=EVW):
        """the plain cosine is near 1 by construction (most of each output vector is the same whatever the input), so
        the error is also measured against what the input changes: the float outputs' spread around their own mean
        over windows (rel = |b - a| / |a - mean a|; cos_c = cosine of a - mean a and b - mean a)"""
        As = torch.stack([pooler(past, P0) for past in EVW]); Bs = torch.stack([pooler(past, W) for past in EVW])
        cs = F.cosine_similarity(As, Bs, dim=-1).mean(-1)
        mu = As.mean(0, keepdim=True)
        rel = ((Bs - As).norm(dim=-1) / (As - mu).norm(dim=-1).clamp_min(1e-8)).mean().item()
        cc = F.cosine_similarity(As - mu, Bs - mu, dim=-1).mean().item()
        spread = F.cosine_similarity(As, mu.expand_as(As), dim=-1).mean().item()
        return float(cs.mean()), float(cs.min()), rel, cc, spread
    mc, wc, rel, cc, spread = score(P)
    if EVR:
        for nm, ev in (("search", EVS), ("reasoning", EVR)):
            m_, w_, r_, c_, s_ = score(P, ev)
            print(f"[pq]   {nm} windows only: relative error {r_:.4f}, centred cosine {c_:.4f}", flush=True)
    print(f"[pq] held-out windows: cosine to the float pooler {mc:.5f} (worst window {wc:.5f}); against the input-driven part: "
          f"relative error {rel:.4f}, centred cosine {cc:.4f} (the float outputs sit at cosine {spread:.4f} to their own mean)", flush=True)

# ---- outputs ----
dq = {k: v.detach().float().cpu().contiguous() for k, v in P.items()}
save_file(dq, os.path.join(A.out, f"pooler_{A.method}{A.tag}_dq.safetensors"))
shifts = torch.arange(0, 32, A.bits, dtype=torch.int64); mx = {}
for k, v in dq.items():
    if k in PACK:
        codes, s_, b_ = PACK[k]; rows = codes.shape[0]
        packed = (codes.to(torch.int64).reshape(rows, -1, 32 // A.bits) << shifts).sum(-1)
        mx[k] = (packed & 0xFFFFFFFF).numpy().astype("uint32")
        mx[k + ".scales"] = s_.to(torch.float16).numpy(); mx[k + ".biases"] = b_.to(torch.float16).numpy()
    else:
        mx[k] = v.to(torch.float16).numpy()
save_np(mx, os.path.join(A.out, f"pooler_{A.method}{A.tag}_mlx.safetensors"))
# the packed file must reproduce the dequantized one exactly
bad = 0
for k in PACK:
    w32 = torch.from_numpy(mx[k].astype(np.int64)); c = ((w32[..., None] >> shifts) & ((1 << A.bits) - 1)).reshape(w32.shape[0], -1).float()
    s_ = torch.from_numpy(mx[k + ".scales"]).float().repeat_interleave(A.group, 1); b_ = torch.from_numpy(mx[k + ".biases"]).float().repeat_interleave(A.group, 1)
    bad += int(((c * s_ + b_) - dq[k]).abs().max() > 1e-3)
sz = os.path.getsize(os.path.join(A.out, f"pooler_{A.method}{A.tag}_mlx.safetensors")) / 1e6
print(f"[pq] packed {len(PACK)} matrices, {sz:.1f} MB (float32: {sum(v.numel() for v in dq.values()) * 4 / 1e6:.1f} MB); unpack check {'OK' if bad == 0 else f'{bad} MISMATCH'}", flush=True)
print(f"POOLER_Q_DONE {A.method} cos {mc:.5f} rel {rel:.4f} centred {cc:.4f}", flush=True)
