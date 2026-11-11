"""4-bit weight quantization of the LLM (the user: GPTQ / DWQ-style, as little loss as possible; stacks on the fixed KV).
Every linear of the decoder layers (q k v o gate up down) gets 4-bit codes in groups of --group along the input dimension with an
fp16 scale and bias per group (W ~ q * s + b, q in 0..15: 4 + 32/group bits per weight, 4.5 at group 64). The embedding (tied with
the output head) and the norms stay 16-bit.
  rtn  - round to nearest; the scale / bias of each group from a clipped min-max grid (the clip that minimises the group's error).
  gptq - layer by layer, each linear's columns quantised in order with the remaining columns updated to cancel the error, weighted
         by the inputs' second moment (Frantar et al.); true sequential: every linear sees the inputs of the already-quantised net.
  dwq  - gptq, then the codes fixed and every group's scale and bias trained so the 4-bit model's next-token distribution matches
         the 16-bit model's (KL, distillation on calibration text).
Calibration: the model's own conversations on the training rows (rollout outputs, prompt + text), separate from every eval row.
  python tools/wq.py --calib tr_lg1024_s1.jsonl,tr2_lg1024_s1.jsonl --methods rtn,gptq,dwq --out wq
Writes wq_<method>.pt ({linear name: {"q": packed codes uint8 (out, in/2), "s": fp16 (out, groups), "b": fp16 (out, groups)}},
"group") and prints the held-out KL to the 16-bit model and the perplexity for each. tools/sp_qwen3.py --wq wq_<method>.pt loads one
(dequantised into the weights: same numbers as a 4-bit kernel, the memory saving needs such a kernel).
"""
import argparse, json, math, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
ap.add_argument("--calib", required=True, help="comma-separated rollout outputs (rows with prompt + text)")
ap.add_argument("--methods", default="rtn,gptq,dwq"); ap.add_argument("--out", default="wq")
ap.add_argument("--group", type=int, default=64); ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--n-calib", type=int, default=128, help="gptq: calibration sequences"); ap.add_argument("--seq", type=int, default=2048)
ap.add_argument("--n-val", type=int, default=24, help="held-out sequences (whole conversations, not in calibration or training)")
ap.add_argument("--damp", type=float, default=0.01)
ap.add_argument("--init", default="", help="dwq: start from these codes (a wq_<method>.pt) instead of running gptq")
ap.add_argument("--dwq-seq", type=int, default=1024); ap.add_argument("--dwq-epochs", type=int, default=1)
ap.add_argument("--dwq-lr", type=float, default=5e-4, help="Adam on the log-scale and the bias (in units of the scale)")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cuda"); ap.add_argument("--max-dwq", type=int, default=0, help="dwq: at most this many pieces (0: all)")
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)
dev = a.device; QM = 2 ** a.bits - 1
tok = AutoTokenizer.from_pretrained(a.model)
LIN = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
STEPS = [LIN[:3], LIN[3:4], LIN[4:6], LIN[6:]]   # linears that share an input, in the order the layer runs them


def log(s): print(s, flush=True)


# ---- data: whole conversations; the last n-val are held out
rows = [json.loads(l) for f in a.calib.split(",") for l in open(f)]
random.shuffle(rows)
seqs = [tok.encode((r.get("prompt") or "") + r["text"], add_special_tokens=False) for r in rows]
seqs = [s for s in seqs if len(s) >= 256]
VAL = [torch.tensor(s[:a.seq]) for s in seqs[-a.n_val:]]; TRS = seqs[:-a.n_val]
CAL = []   # gptq: n-calib windows of seq tokens (a random window of each conversation)
for s in TRS[:a.n_calib]:
    o = random.randint(0, max(len(s) - a.seq, 0)); CAL.append(torch.tensor(s[o:o + a.seq]))
DWQ = []   # dwq: every conversation cut into dwq-seq pieces
for s in TRS:
    for o in range(0, len(s) - 128, a.dwq_seq): DWQ.append(torch.tensor(s[o:o + a.dwq_seq]))
DWQ = DWQ[:a.max_dwq] if a.max_dwq else DWQ
log(f"[wq] conversations {len(seqs)} | gptq calibration {len(CAL)} x <= {a.seq} | dwq pieces {len(DWQ)} x <= {a.dwq_seq} | held out {len(VAL)}")


def load():
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m


# ---- quantiser: per group min-max with the clip that minimises the squared error
def grid(w):   # w (out, g) -> s, b (out, 1)
    mn0 = w.amin(1, keepdim=True).clamp(max=0); mx0 = w.amax(1, keepdim=True).clamp(min=0); best = None
    for p in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75):
        mn, mx = mn0 * p, mx0 * p; s = ((mx - mn) / QM).clamp(min=1e-8); q = ((w - mn) / s).round().clamp(0, QM)
        e = ((q * s + mn - w) ** 2).sum(1, keepdim=True)
        if best is None: best = [e, s, mn]
        else:
            k = e < best[0]; best = [torch.where(k, e, best[0]), torch.where(k, s, best[1]), torch.where(k, mn, best[2])]
    return best[1], best[2]


def rtn(W):
    W = W.float(); o, n = W.shape; g = a.group; s, b = grid(W.view(o * n // g, g)); s = s.view(o, n // g); b = b.view(o, n // g)
    q = ((W - b.repeat_interleave(g, 1)) / s.repeat_interleave(g, 1)).round().clamp(0, QM)
    return q.to(torch.uint8), s, b


def gptq(W, Hm):
    W = W.float().clone(); o, n = W.shape; g = a.group; bs = 128
    H = Hm.clone(); dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
    H += torch.eye(n, device=W.device) * a.damp * torch.diag(H).mean()
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W, dtype=torch.uint8); S = torch.zeros(o, n // g, device=W.device); B = torch.zeros_like(S)
    for i1 in range(0, n, bs):
        i2 = min(i1 + bs, n); W1 = W[:, i1:i2].clone(); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            c = i1 + i
            if c % g == 0: s_, b_ = grid(W1[:, i:i + g]); S[:, c // g] = s_[:, 0]; B[:, c // g] = b_[:, 0]
            w = W1[:, i]; s_, b_ = S[:, c // g], B[:, c // g]
            q = ((w - b_) / s_).round().clamp(0, QM); Q[:, c] = q.to(torch.uint8)
            e = (w - (q * s_ + b_)) / Hi[i, i]; W1[:, i:] -= e[:, None] * Hi[i, i:][None]; E1[:, i] = e
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q, S, B


def deq(q, s, b): g = q.size(1) // s.size(1); return q.float() * s.repeat_interleave(g, 1) + b.repeat_interleave(g, 1)


def lin(m, li, name):
    mod = m.model.layers[li]
    for p in name.split("."): mod = getattr(mod, p)
    return mod


def set_w(m, QD):
    for k, (q, s, b) in QD.items():
        li, name = k; w = lin(m, li, name).weight; w.data.copy_(deq(q, s, b).to(w.dtype))


class Stop(Exception): pass


def run_gptq(m):
    QD = {}; NLy = len(m.model.layers); t0 = time.time()
    for li in range(NLy):
        for step in STEPS:
            Hs = {nm: None for nm in step}; N = [0]
            def hk(nm):
                def f(mod, inp, out):
                    x = inp[0].reshape(-1, inp[0].size(-1)).float(); Hs[nm] = (x.T @ x) if Hs[nm] is None else Hs[nm] + x.T @ x
                    if nm == step[-1]: N[0] += x.size(0); raise Stop
                return f
            hs = [lin(m, li, nm).register_forward_hook(hk(nm)) for nm in step]
            with torch.no_grad():
                for x in CAL:
                    try: m(input_ids=x[None].to(dev), use_cache=False)
                    except Stop: pass
            for h in hs: h.remove()
            for nm in step:
                W = lin(m, li, nm).weight.data; q, s, b = gptq(W, Hs[nm] * (2 / N[0])); QD[(li, nm)] = (q, s, b)
                W.copy_(deq(q, s, b).to(W.dtype))
            del Hs
        if li % 4 == 3 or li == NLy - 1: log(f"[wq] gptq layer {li + 1}/{NLy} ({time.time() - t0:.0f} s)")
    return QD


# ---- dwq: codes fixed, log-scale and bias (in scale units) trained against the 16-bit model
class QLin(nn.Module):
    def __init__(self, q, s, b):
        super().__init__(); self.register_buffer("q", q); self.register_buffer("s0", s.float()); self.register_buffer("b0", b.float())
        self.ls = nn.Parameter(torch.zeros_like(self.s0)); self.db = nn.Parameter(torch.zeros_like(self.s0)); self.g = q.size(1) // s.size(1)
    def sb(self): s = self.s0 * self.ls.exp(); return s, self.b0 + self.s0 * self.db
    def forward(self, x):
        s, b = self.sb(); w = (self.q.to(x.dtype) * s.repeat_interleave(self.g, 1).to(x.dtype) + b.repeat_interleave(self.g, 1).to(x.dtype))
        return F.linear(x, w)


def run_dwq(m, QD, teacher):
    mods = {}
    for (li, nm), (q, s, b) in QD.items():
        par = m.model.layers[li]; ps = nm.split(".")
        for p in ps[:-1]: par = getattr(par, p)
        ql = QLin(q, s, b).to(dev); setattr(par, ps[-1], ql); mods[(li, nm)] = ql
    prm = [p for ql in mods.values() for p in (ql.ls, ql.db)]
    opt = torch.optim.Adam(prm, lr=a.dwq_lr); m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); m.config.use_cache = False; m.train()
    tot = a.dwq_epochs * len(DWQ); step = 0; run = 0.0; t0 = time.time()
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(i / max(tot, 1), 1))))
    for ep in range(a.dwq_epochs):
        random.shuffle(DWQ)
        for x in DWQ:
            x = x[None].to(dev)
            with torch.no_grad(): tl = teacher(input_ids=x, use_cache=False).logits.float()
            sl = m(input_ids=x, use_cache=False).logits.float()
            loss = F.kl_div(F.log_softmax(sl, -1), F.log_softmax(tl, -1), log_target=True, reduction="none").sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); step += 1; lv = loss.item(); run = 0.98 * run + 0.02 * lv if step > 1 else lv
            if step % 50 == 0: log(f"[wq] dwq step {step}/{tot}: KL {run:.4f} ({time.time() - t0:.0f} s)")
    m.eval(); out = {}
    with torch.no_grad():
        for k, ql in mods.items():
            s, b = ql.sb(); out[k] = (ql.q, s, b)
    return out


@torch.no_grad()
def evaluate(m, teacher, name):
    kl = nll = n = 0.0; kls = []
    for x in VAL:
        x = x[None].to(dev); sl = m(input_ids=x, use_cache=False).logits.float()
        tl = teacher(input_ids=x, use_cache=False).logits.float() if teacher is not None else sl
        k = F.kl_div(F.log_softmax(sl, -1), F.log_softmax(tl, -1), log_target=True, reduction="none").sum(-1)[0, :-1]
        nl = F.cross_entropy(sl[0, :-1], x[0, 1:], reduction="none")
        kl += float(k.sum()); nll += float(nl.sum()); n += k.numel(); kls.append(float(k.mean()))
    log(f"[wq] {name}: held-out KL to 16-bit {kl / n:.4f} (worst conversation {max(kls):.4f}) | ppl {math.exp(nll / n):.3f}")
    SUM[name] = {"kl": kl / n, "worst": max(kls), "ppl": math.exp(nll / n)}; json.dump(SUM, open(f"{a.out}_summary.json", "w"), indent=1)


def load_q(path):
    d = torch.load(path, map_location="cpu"); QD = {}
    for k, v in d.items():
        if not isinstance(v, dict): continue
        ks = k.split("."); p = v["q"]
        QD[(int(ks[2]), ".".join(ks[3:]))] = (torch.stack([p & 15, p >> 4], 2).view(p.size(0), -1).to(dev), v["s"].float().to(dev), v["b"].float().to(dev))
    return QD


SUM = {}


def save(QD, name):
    d = {"group": a.group, "bits": a.bits}
    for (li, nm), (q, s, b) in QD.items():
        q = q.to(torch.uint8); d[f"model.layers.{li}.{nm}"] = {"q": (q[:, 0::2] | (q[:, 1::2] << 4)).cpu(), "s": s.half().cpu(), "b": b.half().cpu()}
    torch.save(d, f"{a.out}_{name}.pt"); log(f"[wq] saved {a.out}_{name}.pt")


teacher = load(); evaluate(teacher, None, "16-bit")
M = a.methods.split(",")
if "rtn" in M:
    m = load(); QD = {(li, nm): rtn(lin(m, li, nm).weight.data) for li in range(len(m.model.layers)) for nm in LIN}
    set_w(m, QD); evaluate(m, teacher, f"rtn {a.bits}-bit g{a.group}"); save(QD, "rtn"); del m, QD; torch.cuda.empty_cache()
if a.init and "dwq" in M:
    m = load(); QD = load_q(a.init); set_w(m, QD); evaluate(m, teacher, f"start ({a.init})")
    QD = run_dwq(m, QD, teacher); evaluate(m, teacher, f"dwq {a.bits}-bit g{a.group}"); save(QD, "dwq")
elif "gptq" in M or "dwq" in M:
    m = load(); QD = run_gptq(m); evaluate(m, teacher, f"gptq {a.bits}-bit g{a.group}"); save(QD, "gptq")
    if "dwq" in M:
        QD = run_dwq(m, QD, teacher); evaluate(m, teacher, f"dwq {a.bits}-bit g{a.group}"); save(QD, "dwq")
