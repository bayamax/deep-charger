#!/usr/bin/env python3
"""DWQ: train the quantization parameters instead of the weights.

A 4-bit affine tensor is three things - integer codes, a per-group scale and a per-group bias - and
the runtime computes q * scale + bias. Freeze the codes and the loss becomes smooth in the scale and
the bias, so there is no rounding in the gradient path and no straight-through approximation: the
thing being optimised is exactly the thing the file stores. That also reaches embed_tokens and
lm_head, which an adapter on the projections cannot touch and which carry the largest error.

The target is the model's own unquantized self, matched by KL on its logits. The method is
framework-independent; only mlx-lm's implementation of it is not, and this is the torch one.

Two things come out. --out-mlx is the deployable directory in the format the app already loads:
packed uint32 codes, fp16 scales and biases, and the quantization block in config.json. --out-hf is
the same numbers dequantized into a plain model directory, so the existing evaluator can measure it
with no --q4 - the weights already are the 4-bit values.

  python3 dwq.py --base /root/eval_hf200 --data /root/work/dwq_calib/train.jsonl
"""
import argparse, json, math, os, random, shutil, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="/root/eval_hf200", help="the bf16 model to quantize and distil from")
ap.add_argument("--data", default="/root/work/dwq_calib/train.jsonl", help="jsonl with a text field")
ap.add_argument("--out-mlx", default="/root/dwq_mlx4"); ap.add_argument("--out-hf", default="/root/dwq_hf")
ap.add_argument("--ckpt", default="/root/dwq/latest.pt")
ap.add_argument("--lr", type=float, default=1e-6); ap.add_argument("--steps", type=int, default=512)
ap.add_argument("--accum", type=int, default=4); ap.add_argument("--len", type=int, default=1024)
ap.add_argument("--kpos", type=int, default=256, help="positions per sequence scored with the KL")
ap.add_argument("--temp", type=float, default=2.0)
ap.add_argument("--warmup", type=int, default=20); ap.add_argument("--clip", type=float, default=0.0)
ap.add_argument("--save-every", type=int, default=50)
ap.add_argument("--group", type=int, default=64); ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--selftest", type=int, default=0)
A = ap.parse_args()

import torch                                                     # noqa: E402
import torch.nn as nn                                            # noqa: E402
import torch.nn.functional as F                                  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer     # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q4                                                        # noqa: E402

random.seed(A.seed); torch.manual_seed(A.seed)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GS, NBINS = A.group, (1 << A.bits) - 1

tok = AutoTokenizer.from_pretrained(A.base)
model = AutoModelForCausalLM.from_pretrained(A.base, torch_dtype=torch.bfloat16).to(DEV)
model.config.use_cache = False
model.eval()
for p in model.parameters():
    p.requires_grad_(False)


class Dequant(torch.autograd.Function):
    """q * scale + bias, keeping only the uint8 codes alive for the backward pass."""

    @staticmethod
    def forward(ctx, codes, scales, biases, dtype):
        ctx.save_for_backward(codes)
        w = codes.float() * scales + biases
        return w.reshape(codes.shape[0], -1).to(dtype)

    @staticmethod
    def backward(ctx, g):
        (codes,) = ctx.saved_tensors
        g = g.reshape(codes.shape).float()
        return None, (g * codes.float()).sum(-1, keepdim=True), g.sum(-1, keepdim=True), None


class QTensor:
    """One quantized tensor: frozen codes, trainable scale and bias, and the fp original."""

    def __init__(self, name, w):
        self.name, self.shape, self.dtype = name, tuple(w.shape), w.dtype
        q, s, b = q4.affine_params(w.detach().to("cpu", torch.float32), GS, A.bits)
        self.codes = q.to(torch.uint8).to(DEV)                   # [rows, n_groups, group]
        self.scales = s.to(DEV).requires_grad_(True)             # [rows, n_groups, 1]
        self.biases = b.to(DEV).requires_grad_(True)
        self.orig = w.detach().clone()

    def weight(self):
        return Dequant.apply(self.codes, self.scales, self.biases, self.dtype).reshape(self.shape)

    def err(self):
        """(squared error, squared magnitude), so tensors can be pooled into one honest number."""
        with torch.no_grad():
            o = self.orig.float()
            return (self.weight().float() - o).pow(2).sum().item(), o.pow(2).sum().item()

    def rel_err(self):
        se, ss = self.err()
        return (se / max(ss, 1e-30)) ** 0.5


TEACHER = [False]
QS = {}
for name, mod in q4.quantizable(model):
    qt = QTensor(name, mod.weight.data)
    QS[name] = qt
    if isinstance(mod, nn.Embedding):
        def fwd(x, _m=mod, _q=qt):
            return F.embedding(x, _q.orig if TEACHER[0] else _q.weight(), _m.padding_idx)
    else:
        def fwd(x, *a, _m=mod, _q=qt, **kw):
            return F.linear(x, _q.orig if TEACHER[0] else _q.weight(), _m.bias)
    mod.forward = fwd
    mod.weight.data = torch.empty(0, device=DEV, dtype=qt.dtype)   # the original lives in QTensor now

BODY, HEAD = model.model, model.lm_head
params = [t for q in QS.values() for t in (q.scales, q.biases)]
nw = sum(math.prod(q.shape) for q in QS.values())
print(f"[dwq] {len(QS)} tensors, {nw/1e6:.1f}M weights, {sum(p.numel() for p in params)/1e6:.1f}M "
      f"trainable scale/bias, group={GS} bits={A.bits}", flush=True)
def overall():
    se = ss = 0.0
    for q in QS.values():
        a, b = q.err(); se += a; ss += b
    return (se / max(ss, 1e-30)) ** 0.5


print(f"[dwq] starting relative error {100*overall():.2f}% "
      f"(lm_head {100*QS['lm_head'].rel_err():.2f}%)", flush=True)
if DEV == "cuda":
    torch.cuda.empty_cache()

ROWS = [json.loads(l)["text"] for l in open(A.data) if l.strip()]
print(f"[data] {len(ROWS)} calibration sequences", flush=True)


def sample_ids():
    ids = tok.encode(ROWS[random.randrange(len(ROWS))], add_special_tokens=False)
    if len(ids) <= A.len:
        return ids
    start = 0 if random.random() < 0.5 else random.randrange(len(ids) - A.len)
    return ids[start:start + A.len]


opt = torch.optim.Adam(params, lr=A.lr, betas=(0.9, 0.95))
step0 = 0
if os.path.exists(A.ckpt):
    st = torch.load(A.ckpt, map_location=DEV)
    for k, (s, b) in st["qp"].items():
        QS[k].scales.data.copy_(s.to(DEV)); QS[k].biases.data.copy_(b.to(DEV))
    opt.load_state_dict(st["opt"]); step0 = st["step"]
    print(f"[resume] step {step0}", flush=True)


def lr_at(i):
    if i < A.warmup:
        return A.lr * (i + 1) / A.warmup
    t = (i - A.warmup) / max(1, A.steps - A.warmup)
    return A.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def one_sequence():
    ids = torch.tensor([sample_ids()], device=DEV)
    n = ids.shape[1]
    idx = torch.randperm(n, device=DEV)[:min(A.kpos, n)]
    TEACHER[0] = True
    with torch.no_grad():
        lg_t = F.linear(BODY(input_ids=ids).last_hidden_state[0, idx], QS["lm_head"].orig).float()
    TEACHER[0] = False
    lg_s = HEAD(BODY(input_ids=ids).last_hidden_state[0, idx]).float()
    p_t = F.softmax(lg_t / A.temp, -1)
    kl = (p_t * (torch.log(p_t.clamp_min(1e-9)) - F.log_softmax(lg_s / A.temp, -1))).sum(-1).mean()
    return kl * A.temp ** 2, n


def save_ckpt(i):
    os.makedirs(os.path.dirname(A.ckpt), exist_ok=True)
    torch.save({"qp": {k: (v.scales.detach().cpu(), v.biases.detach().cpu()) for k, v in QS.items()},
                "opt": opt.state_dict(), "step": i}, A.ckpt)


log = open("/root/dwq.log", "a")
t0 = time.time()
for i in range(step0, A.selftest if A.selftest else A.steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(i)
    opt.zero_grad(set_to_none=True)
    kls, toks = 0.0, 0
    for _ in range(A.accum):
        kl, n = one_sequence()
        (kl / A.accum).backward()
        kls += kl.item() / A.accum; toks += n
    if A.clip > 0:
        torch.nn.utils.clip_grad_norm_(params, A.clip)
    opt.step()
    line = (f"step {i+1} kl={kls:.4f} lr={lr_at(i):.2e} tok={toks} "
            f"{(time.time()-t0)/(i-step0+1):.1f}s/step")
    print(line, flush=True); log.write(line + "\n"); log.flush()
    if (i + 1) % A.save_every == 0 or i + 1 == A.steps:
        save_ckpt(i + 1); print(f"[ckpt] step {i+1}", flush=True)

if A.selftest:
    print(f"[dwq] relative error after {A.selftest} steps {100*overall():.2f}%", flush=True)
    print("DWQ_SELFTEST_DONE", flush=True)
    raise SystemExit

# ---- what the file stores: fp16 scales and biases, so round them before anything is written -----
with torch.no_grad():
    for q in QS.values():
        q.scales.data = q.scales.data.to(torch.float16).float()
        q.biases.data = q.biases.data.to(torch.float16).float()
print(f"[dwq] final relative error {100*overall():.2f}% (lm_head {100*QS['lm_head'].rel_err():.2f}%)", flush=True)

# ---- the dequantized copy the existing evaluator can read ----------------------------------------
import numpy as np                                                # noqa: E402
from safetensors.torch import save_file                           # noqa: E402
from safetensors.numpy import save_file as save_np                # noqa: E402
os.makedirs(A.out_hf, exist_ok=True)
sd = {}
for k, v in model.state_dict().items():
    if v.numel():
        sd[k] = v.detach().cpu()
for name, q in QS.items():
    with torch.no_grad():
        sd[name + ".weight"] = q.weight().detach().cpu()
save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(A.out_hf, "model.safetensors"),
          metadata={"format": "pt"})
model.config.save_pretrained(A.out_hf); tok.save_pretrained(A.out_hf)
print(f"[out] {A.out_hf}: {len(sd)} tensors", flush=True)

# ---- the deployable directory, in the format the app already loads -------------------------------
os.makedirs(A.out_mlx, exist_ok=True)
mx = {}
for k, v in sd.items():
    if k.endswith(".weight") and k[: -len(".weight")] in QS:
        continue
    mx[k] = v.to(torch.float16)
for name, q in QS.items():
    c = q.codes.to(torch.int64).reshape(q.shape[0], -1)           # row-major, group boundaries aligned
    shifts = torch.arange(0, 32, A.bits, device=c.device, dtype=torch.int64)
    packed = (c.reshape(c.shape[0], -1, 32 // A.bits) << shifts).sum(-1)
    mx[name + ".weight"] = (packed & 0xFFFFFFFF).to(torch.int64).cpu().numpy().astype("uint32")
    mx[name + ".scales"] = q.scales.detach().reshape(q.shape[0], -1).to(torch.float16).cpu()
    mx[name + ".biases"] = q.biases.detach().reshape(q.shape[0], -1).to(torch.float16).cpu()
save_np({k: (v if isinstance(v, np.ndarray) else v.numpy()) for k, v in mx.items()},
        os.path.join(A.out_mlx, "model.safetensors"))
cfg = json.loads(model.config.to_json_string())
cfg["quantization"] = cfg["quantization_config"] = {"group_size": GS, "bits": A.bits, "mode": "affine"}
cfg["torch_dtype"] = "float16"          # everything in this directory is fp16 or packed uint32
json.dump(cfg, open(os.path.join(A.out_mlx, "config.json"), "w"), indent=2)
tok.save_pretrained(A.out_mlx)
print(f"[out] {A.out_mlx}: {len(mx)} tensors", flush=True)
print("DWQ_DONE", flush=True)
