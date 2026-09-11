#!/usr/bin/env python3
"""Quantization-aware training: move the weights so that the phone's 4-bit conversion stops hurting.

What ships is the affine 4-bit conversion of the MERGED weights, so there is no "keep an adapter in
higher precision" option - whatever we train gets folded in and then rounded. The training therefore
optimises the rounded weight directly: the forward pass uses

    W_used = W + (Q(W) - W).detach(),   W = base + B A * scaling

so the matmul sees the 4-bit value while the gradient reaches the underlying weight (straight
through). Only the adapter carries gradient and optimiser state, which is what keeps this inside one
24GB card.

embed_tokens and lm_head are quantized once at load and left frozen: the student must live with
their error, and the adapter on the projections is what compensates for it. They are restored to
their original values before saving, because the deployment converter quantizes them itself and
quantizing an already-quantized tensor is not a no-op.

The objective is the model's own bf16 self, not the task: teacher and student differ only by the
rounding, so any on-distribution token sequence is a valid calibration input and no rollouts, no
search and no reward are needed. That is what makes a step seconds rather than minutes.

  SP_Q4=1 python3 qat.py --base /root/eval_hf200 --data /root/work/rollouts.jsonl --out /root/qat_hf
"""
import argparse, json, math, os, random, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="/root/eval_hf200", help="bf16 model whose 4-bit conversion is being repaired")
ap.add_argument("--data", default="/root/work/rollouts.jsonl", help="jsonl with q + text: the model's own traces")
ap.add_argument("--out", default="/root/qat_hf", help="where the merged result is written")
ap.add_argument("--ckpt", default="/root/qat/latest.pt")
ap.add_argument("--rank", type=int, default=32); ap.add_argument("--alpha", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--accum", type=int, default=4, help="sequences per optimiser step")
ap.add_argument("--len", type=int, default=640, help="tokens per sequence")
ap.add_argument("--kpos", type=int, default=256, help="positions per sequence scored with the logit KL")
ap.add_argument("--beta", type=float, default=1.0, help="weight of the normalised hidden-state MSE")
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--warmup", type=int, default=50)
ap.add_argument("--clip", type=float, default=1.0)
ap.add_argument("--save-every", type=int, default=100)
ap.add_argument("--group", type=int, default=64); ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--selftest", type=int, default=0, help="run this many steps and stop, printing everything")
A = ap.parse_args()

import torch                                            # noqa: E402
import torch.nn.functional as F                         # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402
from peft import LoraConfig, get_peft_model             # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q4                                               # noqa: E402

random.seed(A.seed); torch.manual_seed(A.seed)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

tok = AutoTokenizer.from_pretrained(A.base)
model = AutoModelForCausalLM.from_pretrained(A.base, torch_dtype=torch.bfloat16).to(DEV)
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(
    r=A.rank, lora_alpha=A.alpha, lora_dropout=0.0, target_modules=TARGETS,
    bias="none", task_type="CAUSAL_LM"))
CLM = model.base_model.model
BODY, HEAD, EMB = CLM.model, CLM.lm_head, CLM.model.embed_tokens
for _p in model.parameters():          # the adapter is the only thing that moves; keep it in fp32
    if _p.requires_grad:
        _p.data = _p.data.float()

# ---- the two tables the adapter cannot reach: quantize once, keep the originals for the teacher ----
def _fq(w):
    return q4.quant_dequant(w.detach().to("cpu", torch.float32), A.group, A.bits).to(w.dtype).to(w.device)

ORIG = {"emb": EMB.weight.data, "head": HEAD.weight.data}
QUANT = {"emb": _fq(ORIG["emb"]), "head": _fq(ORIG["head"])}
for k in ORIG:
    d = (QUANT[k].float() - ORIG[k].float())
    print(f"[q4] {k}: relative error {100*(d.pow(2).sum()/ORIG[k].float().pow(2).sum()).sqrt():.2f}%", flush=True)
del d
if DEV == "cuda":
    torch.cuda.empty_cache()

TEACHER = [False]


def set_mode(teacher):
    TEACHER[0] = teacher
    EMB.weight.data = ORIG["emb"] if teacher else QUANT["emb"]
    HEAD.weight.data = ORIG["head"] if teacher else QUANT["head"]


# ---- straight-through forward on every module the conversion quantizes ----
import peft.tuners.lora as _lora                        # noqa: E402
WRAPPED = [m for _, m in model.named_modules() if isinstance(m, _lora.Linear)]
assert len(WRAPPED) == len(TARGETS) * CLM.config.num_hidden_layers, \
    f"{len(WRAPPED)} wrapped modules, expected {len(TARGETS) * CLM.config.num_hidden_layers}"


def _make_forward(mod):
    base, la, lb = mod.base_layer, mod.lora_A["default"], mod.lora_B["default"]
    scale = mod.scaling["default"]

    def fwd(x, *a, **kw):
        if TEACHER[0]:
            return F.linear(x, base.weight, base.bias)
        w = base.weight.float() + (lb.weight.float() @ la.weight.float()) * scale
        with torch.no_grad():                 # the rounding itself carries no gradient and no graph
            d = q4.quant_dequant(w.detach(), A.group, A.bits) - w.detach()
        return F.linear(x, (w + d).to(x.dtype), base.bias)
    return fwd


for m in WRAPPED:
    m.forward = _make_forward(m)

# ---- calibration inputs: any on-distribution tokens will do, teacher and student see the same ----
ROWS = [json.loads(l) for l in open(A.data) if l.strip()]
ROWS = [r for r in ROWS if r.get("q") and r.get("text")]
print(f"[data] {len(ROWS)} traces from {A.data}", flush=True)


def sample_ids():
    r = ROWS[random.randrange(len(ROWS))]
    head = tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": r["q"]}], add_generation_prompt=True, tokenize=False) + "<think>\n")
    ids = head + tok.encode(r["text"], add_special_tokens=False)
    if len(ids) <= A.len:
        return ids
    start = 0 if random.random() < 0.5 else random.randrange(len(ids) - A.len)
    return ids[start:start + A.len]


params = [p for p in model.parameters() if p.requires_grad]
print(f"[lora] r={A.rank} alpha={A.alpha} on {len(WRAPPED)} modules, "
      f"{sum(p.numel() for p in params)/1e6:.1f}M trainable", flush=True)
opt = torch.optim.AdamW(params, lr=A.lr, betas=(0.9, 0.95), weight_decay=0.0)
step0 = 0
if os.path.exists(A.ckpt):
    st = torch.load(A.ckpt, map_location=DEV)
    model.load_state_dict(st["lora"], strict=False); opt.load_state_dict(st["opt"]); step0 = st["step"]
    print(f"[resume] step {step0}", flush=True)


def lr_at(i):
    if i < A.warmup:
        return A.lr * (i + 1) / A.warmup
    t = (i - A.warmup) / max(1, A.steps - A.warmup)
    return A.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def one_sequence():
    """KL between the bf16 teacher and the 4-bit student, plus their hidden-state distance."""
    ids = torch.tensor([sample_ids()], device=DEV)
    n = ids.shape[1]
    k = min(A.kpos, n)
    idx = torch.randperm(n, device=DEV)[:k]
    set_mode(True)
    with torch.no_grad():
        h_t = BODY(input_ids=ids).last_hidden_state
        lg_t = F.linear(h_t[0, idx], ORIG["head"]).float()
    set_mode(False)
    h_s = BODY(input_ids=ids).last_hidden_state
    lg_s = F.linear(h_s[0, idx], QUANT["head"]).float()
    p_t = F.softmax(lg_t / A.temp, -1)
    kl = (p_t * (torch.log(p_t.clamp_min(1e-9)) - F.log_softmax(lg_s / A.temp, -1))).sum(-1).mean() * A.temp ** 2
    den = h_t.float().pow(2).mean()
    mse = (h_s.float() - h_t.float()).pow(2).mean() / den.clamp_min(1e-9)
    return kl, mse, n


log = open("/root/qat.log", "a")
t0 = time.time()
N = A.selftest or A.steps
for i in range(step0, N if A.selftest else A.steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(i)
    opt.zero_grad(set_to_none=True)
    kls, mses, toks = 0.0, 0.0, 0
    for _ in range(A.accum):
        kl, mse, n = one_sequence()
        ((kl + A.beta * mse) / A.accum).backward()
        kls += kl.item() / A.accum; mses += mse.item() / A.accum; toks += n
    gn = torch.nn.utils.clip_grad_norm_(params, A.clip).item()
    opt.step()
    line = (f"step {i+1} kl={kls:.4f} mse={mses:.4f} lr={lr_at(i):.2e} gn={gn:.2f} "
            f"tok={toks} {(time.time()-t0)/(i-step0+1):.1f}s/step")
    print(line, flush=True); log.write(line + "\n"); log.flush()
    if (i + 1) % A.save_every == 0 or i + 1 == A.steps:
        os.makedirs(os.path.dirname(A.ckpt), exist_ok=True)
        torch.save({"lora": {k: v for k, v in model.state_dict().items() if "lora_" in k},
                    "opt": opt.state_dict(), "step": i + 1}, A.ckpt)
        print(f"[ckpt] step {i+1} -> {A.ckpt}", flush=True)

if A.selftest:
    print("QAT_SELFTEST_DONE", flush=True)
    raise SystemExit

# ---- write what the converter will actually read ----
set_mode(True)                      # the deployment converter quantizes these two itself
merged = model.merge_and_unload()
merged.save_pretrained(A.out, safe_serialization=True)
tok.save_pretrained(A.out)
print(f"[out] {A.out}", flush=True)
print("QAT_DONE", flush=True)
