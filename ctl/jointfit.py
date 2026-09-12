#!/usr/bin/env python3
"""Train the quantization parameters and the pooler together, against the 16-bit system's output.

Everything tried before this trained on plain contexts, which means the pooler never received a
gradient - and the pooler is where the phone's compression happens. It takes embed_tokens output,
and on the phone that table is 4-bit at the largest relative error of any tensor, while the pooler
itself is a fixed float32 function that this lineage fitted against the unquantized model.

So the student here is the whole deployed system: the quantized body, whose per-group scales and
biases move, and the pooler, whose weights move. The teacher is the same system in bf16 with the
pooler as it stands. Both run the compressed forward the evaluator runs - query, then SP vectors
over the evicted set, then the raw window - and the loss is the KL between their logits.

The block boundaries and the evicted sets come from the teacher and are handed to the student. If
each side chose its own, the two would not even be reading the same context and the comparison would
mean nothing.

One block per step, drawn at random: rebuilding every block of a trace costs twenty forwards, and
the gradient does not need them all.

  SP_BASE=/root/eval_hf200 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 python3 jointfit.py
"""
import argparse, json, math, os, random, re, shutil, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/root/pooler200.safetensors")
ap.add_argument("--data", default="/root/work/dwq_calib/train.jsonl")
ap.add_argument("--out-hf", default="/root/joint_hf"); ap.add_argument("--out-mlx", default="/root/joint_mlx4")
ap.add_argument("--out-pooler", default="/root/pooler_joint.safetensors")
ap.add_argument("--state", default="/root/joint/latest.pt"); ap.add_argument("--log", default="/root/joint.log")
ap.add_argument("--steps", type=int, default=1200); ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--lr-q", type=float, default=2e-6); ap.add_argument("--lr-p", type=float, default=1e-5)
ap.add_argument("--warmup", type=int, default=30); ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--rw", type=int, default=768); ap.add_argument("--maxd", type=int, default=384)
ap.add_argument("--chunk", type=int, default=128); ap.add_argument("--maxtok", type=int, default=2600)
ap.add_argument("--val", type=int, default=24); ap.add_argument("--val-every", type=int, default=50)
ap.add_argument("--group", type=int, default=64); ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--clip-search", type=int, default=1); ap.add_argument("--policy-only", type=int, default=1)
ap.add_argument("--gen", type=int, default=1500); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--selftest", type=int, default=0)
ap.add_argument("--objective", default="kl", choices=["kl", "ce"],
                help="kl: match the bf16 system's logits; ce: reproduce the trace's own tokens (use with traces that scored)")
A = ap.parse_args()

os.environ.setdefault("SP_HOTPOT2", "0"); os.environ.setdefault("SP_BASE", "/root/eval_hf200")
os.environ.setdefault("SP_RANK", "16"); os.environ.setdefault("SP_NOSYS", "1")
os.environ.setdefault("SP_EPISODIC", "1")
import numpy as np                                        # noqa: E402
import torch                                              # noqa: E402
import torch.nn as nn                                     # noqa: E402
import torch.nn.functional as F                           # noqa: E402
from safetensors.torch import load_file, save_file        # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q4                                                 # noqa: E402

random.seed(A.seed); torch.manual_seed(A.seed)
F_ = "/root/work/grpo_e2e_torch.py"
src = open(F_).read().split("\n")
cut = next(i for i, l in enumerate(src) if l.startswith("if MULTI > 1:"))
sys.argv = ["grpo_e2e_torch.py", "0", "6", "0", str(A.gen)]
sys.path.insert(0, "/root/work")
ns = {"__name__": "jointfit", "__file__": F_}
exec(compile("\n".join(src[:cut]), F_, "exec"), ns)
model, tok, pooler = ns["model"], ns["tok"], ns["pooler"]
emb, sp, DEV = ns["emb"], ns["sp"], ns["DEV"]
ns["MAXD"] = A.maxd; ns["C"] = A.chunk; ns["RWG"] = A.rw
CLM = model.base_model.model if hasattr(model, "base_model") else model
BODY, HEAD = CLM.model, CLM.lm_head
model.config.use_cache = False
for p in model.parameters():
    p.requires_grad_(False)

sd = load_file(A.ckpt)
pl = {k[len("pooler."):]: v for k, v in sd.items() if k.startswith("pooler.")}
if pl:
    print(f"[pooler] restored {pooler.load_sd(pl)} tensors", flush=True)
POOL_REF = {k: v.detach().clone() for k, v in pooler.A.items()}
pooler.make_trainable()
POOL_TRAIN = dict(pooler.A)

TEACHER = [False]


class Dequant(torch.autograd.Function):
    """q * scale + bias, keeping only the uint8 codes alive for the backward pass."""

    @staticmethod
    def forward(ctx, codes, scales, biases, dtype):
        ctx.save_for_backward(codes)
        return (codes.float() * scales + biases).reshape(codes.shape[0], -1).to(dtype)

    @staticmethod
    def backward(ctx, g):
        (codes,) = ctx.saved_tensors
        g = g.reshape(codes.shape).float()
        return None, (g * codes.float()).sum(-1, keepdim=True), g.sum(-1, keepdim=True), None


class QTensor:
    def __init__(self, name, w):
        self.name, self.shape, self.dtype = name, tuple(w.shape), w.dtype
        f = q4.clipped_affine_params if A.clip_search else q4.affine_params
        q, s, b = f(w.detach().float(), A.group, A.bits)
        self.codes = q.to(torch.uint8)
        self.scales = s.to(DEV).float().requires_grad_(True)
        self.biases = b.to(DEV).float().requires_grad_(True)
        self.orig = w.detach().clone()

    def weight(self):
        return Dequant.apply(self.codes, self.scales, self.biases, self.dtype).reshape(self.shape)


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
    mod.weight.data = torch.empty(0, device=DEV, dtype=qt.dtype)
qp = [t for q in QS.values() for t in (q.scales, q.biases)]
pp = list(pooler.parameters())
print(f"[joint] {len(QS)} quantized tensors ({sum(p.numel() for p in qp)/1e6:.1f}M scale/bias) + "
      f"pooler ({sum(p.numel() for p in pp)/1e6:.2f}M)", flush=True)
if DEV == "cuda":
    torch.cuda.empty_cache()


def set_mode(teacher):
    TEACHER[0] = teacher
    pooler.A = POOL_REF if teacher else POOL_TRAIN


# ---- calibration traces, and the block schedule the teacher would have produced -------------------
def encode(text):
    head, sep, body = text.partition("<think>\n")
    q_ids = tok.encode(head + sep, add_special_tokens=False)
    gen, own = [], []
    for seg in re.split(r"(<information>.*?</information>)", body, flags=re.S):
        if not seg:
            continue
        si = tok.encode(seg, add_special_tokens=False)
        gen += si
        own += [0.0 if seg.startswith("<information>") else 1.0] * len(si)
    return q_ids, gen[:A.maxtok], own[:A.maxtok]


@torch.no_grad()
def schedule(gen):
    """(c0, c1, kept) per block, exactly the bookkeeping the rollout loop does: evict past the raw
    window, and once the pooled set is over maxd keep the highest-mass tokens in order."""
    set_mode(True)
    kept, absorbed, segs, c = [], 0, [], 0
    while c < len(gen):
        c0, c1 = c, min(c + A.chunk, len(gen))
        nd = c0 - min(c0, A.rw)
        if nd > absorbed:
            kept.extend(gen[absorbed:nd]); absorbed = nd
            if len(kept) > A.maxd:
                _, mass = pooler.forward_with_mass(emb(kept).to(torch.float32))
                mm = mass[0].float().cpu().numpy()
                kept = [kept[i] for i in np.sort(np.argsort(mm)[-A.maxd:])]
        segs.append((c0, c1, list(kept)))
        c = c1
    return segs


def logits_for(q_ids, gen, seg):
    """One block, rebuilt as the policy saw it, scored at the positions that predict its tokens."""
    c0, c1, kept = seg
    R = min(c0, A.rw)
    spv = sp(kept) if kept else torch.zeros((1, 0, ns["H"]), device=DEV, dtype=ns["MDTYPE"])
    parts = [emb(q_ids), spv] + ([emb(gen[c0 - R:c0])] if R > 0 else []) + [emb(gen[c0:c1])]
    block = torch.cat(parts, dim=1)
    L, cur = block.shape[1], c1 - c0
    h = BODY(inputs_embeds=block, use_cache=False).last_hidden_state[:, L - cur - 1:L - 1, :]
    return HEAD(h).float()[0]


ALL = [json.loads(l)["text"] for l in open(A.data) if l.strip()]
VAL, ROWS = ALL[:A.val], ALL[A.val:]
print(f"[data] {len(ROWS)} traces, {len(VAL)} held back", flush=True)


def kl_of(text, seg_i=None):
    q_ids, gen, own = encode(text)
    if len(gen) < 8:
        return None
    segs = schedule(gen)
    seg = segs[seg_i % len(segs)] if seg_i is not None else segs[random.randrange(len(segs))]
    c0, c1, _ = seg
    keep = [i for i in range(c1 - c0) if own[c0 + i] > 0] if A.policy_only else list(range(c1 - c0))
    if not keep:
        return None
    idx = torch.tensor(keep, device=DEV)
    if A.objective == "ce":
        # The trace scored, so its own tokens are the target: the quantized system learns to
        # produce the behaviour that worked, in the context it actually runs in. No teacher.
        set_mode(False)
        ls = logits_for(q_ids, gen, seg)[idx]
        tgt = torch.tensor([gen[c0 + i] for i in keep], device=DEV)
        return F.cross_entropy(ls, tgt)
    set_mode(True)
    with torch.no_grad():
        lt = logits_for(q_ids, gen, seg)[idx]
    set_mode(False)
    ls = logits_for(q_ids, gen, seg)[idx]
    p = F.softmax(lt / A.temp, -1)
    return (p * (torch.log(p.clamp_min(1e-9)) - F.log_softmax(ls / A.temp, -1))).sum(-1).mean() * A.temp ** 2


@torch.no_grad()
def validate():
    tot, n = 0.0, 0
    for j, t in enumerate(VAL):
        v = kl_of(t, seg_i=j)
        if v is not None:
            tot += v.item(); n += 1
    return tot / max(n, 1)


opt = torch.optim.Adam([{"params": qp, "lr": A.lr_q}, {"params": pp, "lr": A.lr_p}], betas=(0.9, 0.95))
BEST = {"v": float("inf"), "step": 0, "q": None, "p": None}


def keep_if_best(v, i):
    if v < BEST["v"]:
        BEST.update(v=v, step=i,
                    q={k: (q.scales.detach().cpu().clone(), q.biases.detach().cpu().clone()) for k, q in QS.items()},
                    p={k: t.detach().cpu().clone() for k, t in POOL_TRAIN.items()})
        return True
    return False


def lr_scale(i):
    if i < A.warmup:
        return (i + 1) / A.warmup
    t = (i - A.warmup) / max(1, A.steps - A.warmup)
    return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


log = open(A.log, "a")
v0 = validate(); keep_if_best(v0, 0)
print(f"[joint] compressed-context KL before training {v0:.4f} over {len(VAL)} unseen traces", flush=True)
log.write(f"val 0 kl={v0:.4f}\n"); log.flush()
t0 = time.time()
N = A.selftest if A.selftest else A.steps
for i in range(N):
    for g, base in zip(opt.param_groups, (A.lr_q, A.lr_p)):
        g["lr"] = base * lr_scale(i)
    opt.zero_grad(set_to_none=True)
    tot, n = 0.0, 0
    for _ in range(A.accum):
        l = kl_of(ROWS[random.randrange(len(ROWS))])
        if l is None:
            continue
        (l / A.accum).backward(); tot += l.item() / A.accum; n += 1
    if n:
        opt.step()
    line = f"step {i+1} kl={tot:.4f} lrq={A.lr_q*lr_scale(i):.2e} {(time.time()-t0)/(i+1):.1f}s/step"
    print(line, flush=True); log.write(line + "\n"); log.flush()
    if (i + 1) % A.val_every == 0 or i + 1 == N:
        v = validate()
        mark = "best" if keep_if_best(v, i + 1) else f"worse than step {BEST['step']} ({BEST['v']:.4f})"
        vl = f"val {i+1} kl={v:.4f} {mark}"
        print(vl, flush=True); log.write(vl + "\n"); log.flush()
        os.makedirs(os.path.dirname(A.state), exist_ok=True)
        torch.save({"q": BEST["q"], "p": BEST["p"], "step": BEST["step"], "val": BEST["v"]}, A.state)

if A.selftest:
    print("JOINTFIT_SELFTEST_DONE", flush=True); raise SystemExit

with torch.no_grad():
    for k, (s_, b_) in BEST["q"].items():
        QS[k].scales.data.copy_(s_.to(DEV).to(torch.float16).float())
        QS[k].biases.data.copy_(b_.to(DEV).to(torch.float16).float())
    for k, t in BEST["p"].items():
        POOL_TRAIN[k].data.copy_(t.to(DEV))
print(f"[joint] taking step {BEST['step']}, KL {BEST['v']:.4f} (started at {v0:.4f})", flush=True)
save_file({f"pooler.{k}": v.detach().cpu().contiguous() for k, v in POOL_TRAIN.items()}, A.out_pooler)
print(f"[out] {A.out_pooler}", flush=True)

# The evaluator reads a plain model directory, and the one this run started from is already on disk.
# Copy it and swap in the dequantized weights: building the file out of the peft-wrapped state dict
# instead would mean trusting a pile of name rewriting for no benefit.
def plain_name(n):
    return n.replace("base_model.model.", "").replace(".base_layer", "") + ".weight"


BASE = os.environ["SP_BASE"]
os.makedirs(A.out_hf, exist_ok=True)
state = load_file(os.path.join(BASE, "model.safetensors"))
set_mode(False)
with torch.no_grad():
    for name, q in QS.items():
        k = plain_name(name)
        assert k in state, f"{k} is not in {BASE}"
        state[k] = q.weight().detach().cpu().to(state[k].dtype)
save_file({k: v.contiguous() for k, v in state.items()}, os.path.join(A.out_hf, "model.safetensors"),
          metadata={"format": "pt"})
for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
          "special_tokens_map.json"):
    if os.path.exists(os.path.join(BASE, f)):
        shutil.copy(os.path.join(BASE, f), os.path.join(A.out_hf, f))
print(f"[out] {A.out_hf}: {len(state)} tensors, {len(QS)} replaced", flush=True)
print("JOINTFIT_DONE", flush=True)
