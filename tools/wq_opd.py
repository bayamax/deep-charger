"""On-policy distillation of the 4-bit weights, after DWQ (the user: 4-bit was the open item; push its loss down as far as it goes).
DWQ only read the 16-bit model's text, so the errors that compound while the 4-bit model writes its own text were never seen.
Here the 4-bit model writes continuations of conversations, the 16-bit model reads them and gives its next-token distribution at
every position, and the per-group scales and biases (codes fixed, as in DWQ) are trained to close KL(16-bit || 4-bit) along the
4-bit model's own text.
Metric (before and after): the same KL on continuations the 4-bit model writes from prefixes of the eval conversations (never
used for training), fixed seed.
  python tools/wq_opd.py --init wq4_dwq.pt --calib a.jsonl,b.jsonl --evalc mt_full_s1.jsonl,docc_full_s1.jsonl --minutes 30 --out wq5
Writes <out>_dwq.pt in the tools/wq.py format (sp_qwen3.py --wq, tools/wq_export.py, tools/wq_kl.py read it).
"""
import argparse, json, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-1.7B"); ap.add_argument("--init", required=True)
ap.add_argument("--calib", required=True); ap.add_argument("--evalc", required=True); ap.add_argument("--out", default="wq5")
ap.add_argument("--prefix", type=int, default=768); ap.add_argument("--gen", type=int, default=256)
ap.add_argument("--bgen", type=int, default=16, help="continuations written per step"); ap.add_argument("--mb", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--temp", type=float, default=0.6)
ap.add_argument("--minutes", type=float, default=30); ap.add_argument("--n-eval", type=int, default=96)
ap.add_argument("--device", default="cuda"); ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
random.seed(a.seed); torch.manual_seed(a.seed); dev = a.device; dt = getattr(torch, a.dtype)
tok = AutoTokenizer.from_pretrained(a.model)
STOP = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}


def log(s): print(s, flush=True)


def seqs_of(files):
    out = []
    for f in files.split(","):
        for l in open(f):
            r = json.loads(l); s = tok.encode((r.get("prompt") or "") + r["text"], add_special_tokens=False)
            if len(s) > a.prefix + 32: out.append(s)
    return out


def prefixes(seqs, n, rng):   # n windows of exactly --prefix tokens ending at a random point of a random conversation
    out = []
    for _ in range(n):
        s = rng.choice(seqs); c = rng.randint(a.prefix, len(s) - 1); out.append(s[c - a.prefix:c])
    return torch.tensor(out)


TR = seqs_of(a.calib); EV = prefixes(seqs_of(a.evalc), a.n_eval, random.Random(1234))
log(f"[wq] on-policy: {len(TR)} training conversations, {len(EV)} eval prefixes of {a.prefix} tokens, {a.gen} written per prefix")

T = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, attn_implementation="sdpa").to(dev).eval()
S = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, attn_implementation="sdpa").to(dev).eval()
for p in list(T.parameters()) + list(S.parameters()): p.requires_grad_(False)
VER = [0]


class QLin(nn.Module):   # codes fixed; log-scale and bias (in units of the scale) trained; the weight cached while not training
    def __init__(self, q, s, b):
        super().__init__(); self.register_buffer("q", q); self.register_buffer("s0", s.float()); self.register_buffer("b0", b.float())
        self.ls = nn.Parameter(torch.zeros_like(self.s0)); self.db = nn.Parameter(torch.zeros_like(self.s0)); self.g = q.size(1) // s.size(1)
        self._w = None; self._v = -1
    def sb(self): return self.s0 * self.ls.exp(), self.b0 + self.s0 * self.db
    def w(self, dtype):
        s, b = self.sb(); return self.q.to(dtype) * s.repeat_interleave(self.g, 1).to(dtype) + b.repeat_interleave(self.g, 1).to(dtype)
    def forward(self, x):
        if torch.is_grad_enabled(): return F.linear(x, self.w(x.dtype))
        if self._v != VER[0]: self._w = self.w(x.dtype).detach(); self._v = VER[0]
        return F.linear(x, self._w)


d = torch.load(a.init, map_location="cpu"); MODS = {}
for k, v in d.items():
    if not isinstance(v, dict): continue
    p = v["q"]; q = torch.stack([p & 15, p >> 4], 2).view(p.size(0), -1)
    par = S; ks = k.split(".")
    for x in ks[:-1]: par = getattr(par, x)
    ql = QLin(q.to(dev), v["s"].to(dev), v["b"].to(dev)); setattr(par, ks[-1], ql); MODS[k] = ql
G0, BITS = d["group"], d["bits"]; del d
S.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
log(f"[wq] student: {len(MODS)} 4-bit linears from {a.init}")


@torch.no_grad()
def write(x):   # the 4-bit model continues each prefix; mask = written tokens up to and including the first stop token
    S.eval(); y = S.generate(input_ids=x.to(dev), attention_mask=torch.ones_like(x).to(dev), max_new_tokens=a.gen, min_new_tokens=a.gen,
                             do_sample=True, temperature=a.temp, top_p=1.0, top_k=0, use_cache=True, pad_token_id=tok.eos_token_id)
    g = y[:, x.size(1):]; st = torch.zeros_like(g, dtype=torch.bool)
    for s_ in STOP: st |= g == s_
    after = (st.cumsum(1) - st.long()) > 0   # strictly after the first stop
    return y, (~after).float()


def kl(y, m):   # per-token KL(16-bit || 4-bit) on the written positions
    P = y.size(1) - a.gen
    with torch.no_grad(): lt = F.log_softmax(T.lm_head(T.model(input_ids=y).last_hidden_state[:, P - 1:-1]).float(), -1)
    ls = F.log_softmax(S.lm_head(S.model(input_ids=y).last_hidden_state[:, P - 1:-1]).float(), -1)
    return ((lt.exp() * (lt - ls)).sum(-1) * m).sum() / m.sum().clamp(min=1), m.sum()


def evaluate(name):
    torch.manual_seed(99); tot = n = 0.0
    for i in range(0, len(EV), a.bgen):
        y, m = write(EV[i:i + a.bgen])
        with torch.no_grad():
            for j in range(0, y.size(0), a.mb):
                k, c = kl(y[j:j + a.mb], m[j:j + a.mb]); tot += float(k) * float(c); n += float(c)
    log(f"[wq] {name}: KL on its own continuations of the eval prefixes {tot / n:.4f} ({int(n)} tokens)"); return tot / n


k0 = evaluate("start (" + a.init + ")")
prm = [p for ql in MODS.values() for p in (ql.ls, ql.db)]
opt = torch.optim.Adam(prm, lr=a.lr); rng = random.Random(a.seed); t0 = time.time(); step = 0; run = None
while time.time() - t0 < a.minutes * 60:
    y, m = write(prefixes(TR, a.bgen, rng)); S.train(); opt.zero_grad(); tl = 0.0
    for j in range(0, y.size(0), a.mb):
        k, c = kl(y[j:j + a.mb], m[j:j + a.mb]); (k * a.mb / y.size(0)).backward(); tl += float(k) * a.mb / y.size(0)
    for g_ in opt.param_groups: g_["lr"] = a.lr * min(1.0, (step + 1) / 10)
    opt.step(); VER[0] += 1; step += 1; run = tl if run is None else 0.95 * run + 0.05 * tl
    if step % 10 == 0: log(f"[wq] on-policy step {step}: KL {run:.4f} ({time.time() - t0:.0f} s)")
S.eval(); k1 = evaluate(f"after {step} steps")
out = {"group": G0, "bits": BITS}
with torch.no_grad():
    for k, ql in MODS.items():
        s, b = ql.sb(); q = ql.q.to(torch.uint8)
        out[k] = {"q": (q[:, 0::2] | (q[:, 1::2] << 4)).cpu(), "s": s.half().cpu(), "b": b.half().cpu()}
torch.save(out, f"{a.out}_dwq.pt"); json.dump({"start": k0, "after": k1, "steps": step}, open(f"{a.out}_opd.json", "w"))
log(f"[wq] saved {a.out}_dwq.pt (own-text KL {k0:.4f} -> {k1:.4f})")
