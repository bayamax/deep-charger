#!/usr/bin/env python3
"""The 4-bit pooler's grid trained against the system it replaces - the second half of what made the shipped model
(GPTQ's codes, then the grid's scales and biases trained on the model's own traces: release/g14-4bit-gptq-trained).

GPTQ chose the pooler's codes to keep each matrix's own outputs (pooler_gptq.py). This keeps those codes and moves
only the grid - each group's fp16 scale and bias - plus the parts that stay in float (query, layernorms, biases,
out_scale), so that the MODEL's next-token distribution under the 4-bit pooler matches the one under the float pooler.
The body is the shipped 4-bit model, frozen. Each step rebuilds one block of a trace exactly as the evaluator does
(question, soft prompt over the evicted set, raw window, the block), the teacher with the float pooler, the student
with the 4-bit one; the eviction schedule is the teacher's, so both read the same context. Loss: KL on the trace's
own tokens. Traces: the model's own search traces and reasoning (dolphin_v1, the Dolphin held-out hundred left out).

The app's file format does not change: the same codes, new fp16 scales and biases.

  SP_BASE=/root/g14q_hf SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 python3 pooler_gridfit.py --float /path/pooler.safetensors \
      --codes /root/pq/pooler_gptq_mix_mlx.safetensors --search /root/work/dwq_calib/train.jsonl \
      --reason /root/work/dolphin_calib.jsonl --out /root/pq/pooler_gridfit
"""
import argparse, json, math, os, random, re, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--float", required=True, help="the float pooler (bare keys)")
ap.add_argument("--codes", required=True, help="pooler_gptq.py's packed file: uint32 codes + .scales / .biases")
ap.add_argument("--search", required=True); ap.add_argument("--reason", required=True)
ap.add_argument("--out", required=True, help="prefix: <out>_dq.safetensors and <out>_mlx.safetensors")
ap.add_argument("--steps", type=int, default=600); ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr-q", type=float, default=2e-5, help="scales and biases"); ap.add_argument("--lr-f", type=float, default=1e-5, help="the float parts")
ap.add_argument("--warmup", type=int, default=20); ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--rw", type=int, default=768); ap.add_argument("--maxd", type=int, default=384)
ap.add_argument("--chunk", type=int, default=128); ap.add_argument("--maxtok", type=int, default=3000)
ap.add_argument("--val", type=int, default=24); ap.add_argument("--val-every", type=int, default=50)
ap.add_argument("--group", type=int, default=64); ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--compressed-only", type=int, default=0, help="1: train only on blocks with something evicted (an empty past is otherwise drawn too)")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--selftest", type=int, default=0)
A = ap.parse_args()

os.environ.setdefault("SP_HOTPOT2", "0"); os.environ.setdefault("SP_RANK", "16"); os.environ.setdefault("SP_NOSYS", "1"); os.environ.setdefault("SP_EPISODIC", "1")
import numpy as np                                        # noqa: E402
import torch                                              # noqa: E402
import torch.nn.functional as F                           # noqa: E402
from safetensors.torch import load_file, save_file        # noqa: E402
from safetensors.numpy import save_file as save_np        # noqa: E402

random.seed(A.seed); torch.manual_seed(A.seed)
F_ = "/root/work/grpo_e2e_torch.py"
src = open(F_).read().split("\n")
cut = next(i for i, l in enumerate(src) if l.startswith("if MULTI > 1:"))
sys.argv = ["grpo_e2e_torch.py", "0", "6", "0", "1500"]
sys.path.insert(0, "/root/work")
ns = {"__name__": "pooler_gridfit", "__file__": F_}
exec(compile("\n".join(src[:cut]), F_, "exec"), ns)
model, tok, pooler = ns["model"], ns["tok"], ns["pooler"]
emb, sp, DEV = ns["emb"], ns["sp"], ns["DEV"]
ns["MAXD"] = A.maxd; ns["C"] = A.chunk; ns["RWG"] = A.rw
CLM = model.base_model.model if hasattr(model, "base_model") else model
BODY, HEAD = CLM.model, CLM.lm_head
model.config.use_cache = False
for p in model.parameters():
    p.requires_grad_(False)
import torch.utils.checkpoint as _ckp                    # noqa: E402
for _lyr in BODY.layers:                                 # recompute each layer in the backward: a block is ~1000 tokens, the card 12 GB
    def _wrapped(*a, _f=_lyr.forward, **kw):
        return _ckp.checkpoint(_f, *a, use_reentrant=False, **kw)
    _lyr.forward = _wrapped

# ---- the two poolers: the float teacher, and the student whose quantized matrices are codes x trainable grid ----
FL = {(k[len("pooler."):] if k.startswith("pooler.") else k): v.float().to(DEV) for k, v in load_file(A.float).items()}
MX = load_file(A.codes)
SH = torch.arange(0, 32, A.bits, dtype=torch.int64)
QK = [k for k in MX if k + ".scales" in MX]
CODES, SC, BI = {}, {}, {}
for k in QK:
    w = MX[k].to(torch.int64)
    CODES[k] = ((w[..., None] >> SH) & ((1 << A.bits) - 1)).reshape(w.shape[0], -1, A.group).to(torch.uint8).to(DEV)
    SC[k] = MX[k + ".scales"].float().unsqueeze(-1).to(DEV).requires_grad_(True)
    BI[k] = MX[k + ".biases"].float().unsqueeze(-1).to(DEV).requires_grad_(True)
FLOATP = {k: FL[k].clone().requires_grad_(True) for k in FL if k not in CODES}


class Student:
    """the pooler reads self.A[k]; quantized keys are rebuilt from codes and the live grid on every read"""

    def __getitem__(self, k):
        if k in CODES:
            return (CODES[k].float() * SC[k] + BI[k]).reshape(CODES[k].shape[0], -1)
        return FLOATP[k]

    def __contains__(self, k): return k in CODES or k in FLOATP
    def keys(self): return list(CODES) + list(FLOATP)
    def values(self): return [self[k] for k in self.keys()]
    def items(self): return [(k, self[k]) for k in self.keys()]


STUDENT = Student()
pooler.load_sd({k: v for k, v in FL.items()})
TEACHER_A = pooler.A


def set_mode(teacher):
    pooler.A = TEACHER_A if teacher else STUDENT


qp = list(SC.values()) + list(BI.values()); fp = list(FLOATP.values())
print(f"[grid] {len(QK)} quantized matrices ({sum(p.numel() for p in qp)/1e6:.2f}M scale/bias), {len(fp)} float tensors "
      f"({sum(p.numel() for p in fp)/1e6:.2f}M); missing from the codes file: {[k for k in FL if k not in CODES and k not in MX][:3]}", flush=True)


# ---- traces and the teacher's block schedule (jointfit.py's, unchanged) ----
def encode(text):
    head, sep, body = text.partition("<think>\n")
    q_ids = tok.encode(head + sep, add_special_tokens=False)
    gen, own = [], []
    for seg in re.split(r"(<information>.*?</information>)", body, flags=re.S):
        if not seg:
            continue
        si = tok.encode(seg, add_special_tokens=False)
        gen += si; own += [0.0 if seg.startswith("<information>") else 1.0] * len(si)
    return q_ids, gen[:A.maxtok], own[:A.maxtok]


@torch.no_grad()
def schedule(gen):
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
    c0, c1, kept = seg
    R = min(c0, A.rw)
    spv = sp(kept)                                         # an empty past still gives the 32 vectors, as in the rollout
    parts = [emb(q_ids), spv.to(ns["MDTYPE"])] + ([emb(gen[c0 - R:c0])] if R > 0 else []) + [emb(gen[c0:c1])]
    block = torch.cat(parts, dim=1)
    L, cur = block.shape[1], c1 - c0
    h = BODY(inputs_embeds=block, use_cache=False).last_hidden_state[:, L - cur - 1:L - 1, :]
    return HEAD(h).float()[0]


BOS = "<｜begin▁of▁sentence｜>"
search = [json.loads(l)["text"] for l in open(A.search) if l.strip()]
reason = []
for l in open(A.reason):
    if not l.strip(): continue
    r = json.loads(l)
    if r.get("q") and r.get("reply"):
        reason.append(f"{BOS}<｜User｜>{r['q']}<｜Assistant｜><think>\n{r.get('thinking', '')}\n</think>\n\n{r['reply']}")
random.shuffle(search); random.shuffle(reason)
VAL = search[:A.val // 2] + reason[:A.val // 2]
TR_S, TR_R = search[A.val // 2:], reason[A.val // 2:]
print(f"[data] {len(TR_S)} search traces, {len(TR_R)} reasoning traces, {len(VAL)} held back", flush=True)


def kl_of(text, seg_i=None):
    q_ids, gen, own = encode(text)
    if len(gen) < 8:
        return None
    segs = schedule(gen)
    if A.compressed_only:
        segs = [s for s in segs if s[2]] or segs
    seg = segs[seg_i % len(segs)] if seg_i is not None else segs[random.randrange(len(segs))]
    c0, c1, _ = seg
    keep = [i for i in range(c1 - c0) if own[c0 + i] > 0]
    if not keep:
        return None
    idx = torch.tensor(keep, device=DEV)
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
        v = kl_of(t, seg_i=j * 3)
        if v is not None:
            tot += v.item(); n += 1
    return tot / max(n, 1)


opt = torch.optim.Adam([{"params": qp, "lr": A.lr_q}, {"params": fp, "lr": A.lr_f}], betas=(0.9, 0.95))
BEST = {"v": float("inf"), "step": 0}


def keep_if_best(v, i):
    if v < BEST["v"]:
        BEST.update(v=v, step=i, sc={k: t.detach().clone() for k, t in SC.items()}, bi={k: t.detach().clone() for k, t in BI.items()},
                    fp={k: t.detach().clone() for k, t in FLOATP.items()})
        return True
    return False


def lr_scale(i):
    if i < A.warmup: return (i + 1) / A.warmup
    return 0.5 * (1 + math.cos(math.pi * min((i - A.warmup) / max(1, A.steps - A.warmup), 1.0)))


v0 = validate(); keep_if_best(v0, 0)
print(f"[grid] KL of the 4-bit pooler's system to the float one before training {v0:.5f} over {len(VAL)} held-back traces", flush=True)
t0 = time.time()
N = A.selftest or A.steps
for i in range(N):
    for g, base in zip(opt.param_groups, (A.lr_q, A.lr_f)): g["lr"] = base * lr_scale(i)
    opt.zero_grad(set_to_none=True); tot, n = 0.0, 0
    for a in range(A.accum):
        rows = TR_R if (a % 2) else TR_S                # half reasoning, half search
        l = kl_of(rows[random.randrange(len(rows))])
        if l is None: continue
        (l / A.accum).backward(); tot += l.item() / A.accum; n += 1
    if n: opt.step()
    if (i + 1) % 10 == 0 or A.selftest:
        print(f"step {i+1} kl={tot:.5f} {(time.time()-t0)/(i+1):.1f}s/step", flush=True)
    if (i + 1) % A.val_every == 0 or i + 1 == N:
        v = validate()
        print(f"val {i+1} kl={v:.5f} " + ("best" if keep_if_best(v, i + 1) else f"worse than step {BEST['step']} ({BEST['v']:.5f})"), flush=True)
if A.selftest:
    print("GRIDFIT_SELFTEST_DONE", flush=True); raise SystemExit

# ---- the best grid out, in both forms ----
print(f"[grid] taking step {BEST['step']}, KL {BEST['v']:.5f} (started at {v0:.5f})", flush=True)
dq, mx = {}, {}
with torch.no_grad():
    for k in CODES:
        s_ = BEST["sc"][k].to(torch.float16); b_ = BEST["bi"][k].to(torch.float16)
        dq[k] = (CODES[k].float() * s_.float() + b_.float()).reshape(CODES[k].shape[0], -1).cpu().contiguous()
        rows = CODES[k].shape[0]
        packed = (CODES[k].reshape(rows, -1).to(torch.int64).cpu().reshape(rows, -1, 32 // A.bits) << SH).sum(-1)
        mx[k] = (packed & 0xFFFFFFFF).numpy().astype("uint32")
        mx[k + ".scales"] = s_.squeeze(-1).cpu().numpy(); mx[k + ".biases"] = b_.squeeze(-1).cpu().numpy()
    for k, t in BEST["fp"].items():
        dq[k] = t.float().cpu().contiguous(); mx[k] = t.to(torch.float16).cpu().numpy()
save_file(dq, A.out + "_dq.safetensors"); save_np(mx, A.out + "_mlx.safetensors")
print(f"[out] {A.out}_dq.safetensors, {A.out}_mlx.safetensors ({os.path.getsize(A.out + '_mlx.safetensors')/1e6:.1f} MB)", flush=True)
print("GRIDFIT_DONE", flush=True)
