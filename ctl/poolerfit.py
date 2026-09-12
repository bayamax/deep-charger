#!/usr/bin/env python3
"""Fit the pooler to the quantized embedding table it is actually fed on the phone.

sp(kept) in the harness is pooler(emb(kept)) - the compressor's input is the embedding TABLE, not
hidden states, and that table is quantized on device at the largest relative error of any tensor.
The pooler is float32 and fixed, and this lineage refitted it against the unquantized model, so
every token that scrolls out of the raw window enters the compressor already perturbed and the
compressor has no way to know.

This trains the pooler to undo exactly that. No language-model forward is involved: the target is
what the current pooler produces from the unquantized embeddings, the input is the same tokens
through the quantized ones, and the parameters that move are the pooler's. Seconds per step.

What comes out is a pooler file. The app loads it separately and keeps it in float32, so this ships
without touching the model directory, its size, or any code.

  SP_BASE=/root/eval_hf200 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 python3 poolerfit.py \
      --ckpt /root/pooler200.safetensors --out /root/pooler_q4.safetensors
"""
import argparse, json, math, os, random, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/root/pooler200.safetensors", help="pooler to start from")
ap.add_argument("--out", default="/root/pooler_q4.safetensors")
ap.add_argument("--data", default="/root/work/dwq_calib/train.jsonl")
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--warmup", type=int, default=50)
ap.add_argument("--minlen", type=int, default=64); ap.add_argument("--maxlen", type=int, default=384)
ap.add_argument("--val", type=int, default=64); ap.add_argument("--val-every", type=int, default=100)
ap.add_argument("--group", type=int, default=64); ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--gen", type=int, default=1500)
ap.add_argument("--selftest", type=int, default=0)
A = ap.parse_args()

os.environ.setdefault("SP_HOTPOT2", "0"); os.environ.setdefault("SP_BASE", "/root/eval_hf200")
os.environ.setdefault("SP_RANK", "16"); os.environ.setdefault("SP_NOSYS", "1")
os.environ.setdefault("SP_EPISODIC", "1")
import torch                                              # noqa: E402
from safetensors.torch import load_file, save_file        # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q4                                                 # noqa: E402

F = "/root/work/grpo_e2e_torch.py"
src = open(F).read().split("\n")
cut = next(i for i, l in enumerate(src) if l.startswith("if MULTI > 1:"))
sys.argv = ["grpo_e2e_torch.py", "0", "6", "0", str(A.gen)]
sys.path.insert(0, "/root/work")
ns = {"__name__": "poolerfit", "__file__": F}
exec(compile("\n".join(src[:cut]), F, "exec"), ns)
model, tok, pooler, emb, DEV = ns["model"], ns["tok"], ns["pooler"], ns["emb"], ns["DEV"]

sd = load_file(A.ckpt)
pl = {k[len("pooler."):]: v for k, v in sd.items() if k.startswith("pooler.")}
if pl:
    print(f"[pooler] restored {pooler.load_sd(pl)} tensors from {A.ckpt}", flush=True)
REF = {k: v.detach().clone() for k, v in pooler.A.items()}     # the pooler as it stands: the target
pooler.make_trainable()
params = pooler.parameters()
print(f"[pooler] {len(params)} tensors, {sum(p.numel() for p in params)/1e6:.2f}M parameters, "
      f"n_sp={pooler.n_sp} H={pooler.H}", flush=True)

# the two embedding tables: what the target sees, and what the phone has
embT = model.get_input_embeddings()
W_FP = embT.weight.data
W_Q = q4.quant_dequant(W_FP.detach().to("cpu", torch.float32), A.group, A.bits).to(W_FP.dtype).to(DEV)
d = (W_Q.float() - W_FP.float())
print(f"[q4] embed_tokens relative error {100*(d.pow(2).sum()/W_FP.float().pow(2).sum()).sqrt():.2f}%", flush=True)
del d
if DEV == "cuda":
    torch.cuda.empty_cache()


def use_quantized(q):
    embT.weight.data = W_Q if q else W_FP


@torch.no_grad()
def target(ids):
    """What the current pooler makes of these tokens through the clean table."""
    use_quantized(False)
    de = emb(ids).to(torch.float32)
    saved, pooler.A = pooler.A, REF
    try:
        return pooler.forward(de).float()
    finally:
        pooler.A = saved


def predicted(ids):
    use_quantized(True)
    return pooler.forward(emb(ids).to(torch.float32)).float()


ALL = [json.loads(l)["text"] for l in open(A.data) if l.strip()]
VAL, ROWS = ALL[:A.val], ALL[A.val:]
print(f"[data] {len(ROWS)} traces, {len(VAL)} held back", flush=True)


def sample_ids(text=None, rnd=True):
    ids = tok.encode(text if text is not None else ROWS[random.randrange(len(ROWS))],
                     add_special_tokens=False)
    n = random.randint(A.minlen, A.maxlen) if rnd else A.maxlen
    if len(ids) <= n:
        return ids
    start = random.randrange(len(ids) - n) if rnd else 0
    return ids[start:start + n]


VAL_IDS = [sample_ids(t, False) for t in VAL]


def loss_of(ids):
    t = target(ids)
    p = predicted(ids)
    err = (p - t).pow(2).sum(-1)
    return (err / t.pow(2).sum(-1).clamp_min(1e-9)).mean()


@torch.no_grad()
def validate():
    tot = 0.0
    for ids in VAL_IDS:
        t, p = target(ids), predicted(ids)
        cos = torch.nn.functional.cosine_similarity(p, t, dim=-1).mean().item()
        tot += 1.0 - cos
    return tot / max(len(VAL_IDS), 1)


opt = torch.optim.Adam(params, lr=A.lr, betas=(0.9, 0.95))
BEST = {"v": float("inf"), "step": 0, "sd": None}


def keep_if_best(v, i):
    if v < BEST["v"]:
        BEST.update(v=v, step=i, sd={k: p.detach().cpu().clone() for k, p in pooler.A.items()})
        return True
    return False


def lr_at(i):
    if i < A.warmup:
        return A.lr * (i + 1) / A.warmup
    t = (i - A.warmup) / max(1, A.steps - A.warmup)
    return A.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


log = open("/root/poolerfit.log", "a")
v0 = validate(); keep_if_best(v0, 0)
print(f"[fit] 1-cos between the phone's pooler input and the target, before training: {v0:.5f}", flush=True)
log.write(f"val 0 cosloss={v0:.5f}\n"); log.flush()
t0 = time.time()
N = A.selftest if A.selftest else A.steps
for i in range(N):
    for g in opt.param_groups:
        g["lr"] = lr_at(i)
    opt.zero_grad(set_to_none=True)
    tot = 0.0
    for _ in range(A.accum):
        l = loss_of(sample_ids())
        (l / A.accum).backward()
        tot += l.item() / A.accum
    opt.step()
    line = f"step {i+1} rel={tot:.5f} lr={lr_at(i):.2e} {(time.time()-t0)/(i+1):.2f}s/step"
    print(line, flush=True); log.write(line + "\n"); log.flush()
    if (i + 1) % A.val_every == 0 or i + 1 == N:
        v = validate()
        mark = "best" if keep_if_best(v, i + 1) else f"worse than step {BEST['step']} ({BEST['v']:.5f})"
        vl = f"val {i+1} cosloss={v:.5f} {mark}"
        print(vl, flush=True); log.write(vl + "\n"); log.flush()

if A.selftest:
    print("POOLERFIT_SELFTEST_DONE", flush=True); raise SystemExit
print(f"[fit] taking step {BEST['step']}, 1-cos {BEST['v']:.5f} (started at {v0:.5f})", flush=True)
save_file({f"pooler.{k}": v.contiguous() for k, v in BEST["sd"].items()}, A.out)
print(f"[out] {A.out}: {len(BEST['sd'])} tensors", flush=True)
print("POOLERFIT_DONE", flush=True)
