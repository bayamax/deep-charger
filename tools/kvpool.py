"""Learned KV pooler: bounded KV cache trained for TOKEN REPRODUCTION (the full model's next-token distributions).

Cache layout during generation, every layer:  [prompt | m slots | exposure window (last W tokens)]
Every C tokens the oldest C leave the window; the pooler folds them into the m slots of each (layer, kv head):
  inputs   : the current slots (keys, values, log-mass beta) + the evicted tokens (keys, values, beta 0)
  assign   : r[c,j] = softmax_c( (U[c] . k_j) * s / sqrt(D) )      learned codebook U per (layer, kv head)
  slot     : mass M_c = sum_j r[c,j] e^beta_j ; key/value = mass-weighted mean of its members ; beta_c = log M_c + b_c
             key scaled by e^g_c (Jensen: the mean of keys under-scores), b_c and g_c learned
Slots live in KV space (post-RoPE keys), so the window's KV never needs re-computation (no re-prefill on the phone),
and beta enters as an additive attention-mask column. Pooler size: NL*G*m*(D+2)+NL*G parameters (m=64: 1.9M) << KV.

Objective (the user: like GPTQ, match the full model's OUTPUT): KL(full || bounded) on the next-token distribution of
every token after the first eviction, teacher-forced on the model's own full-KV traces. One pooler step is
back-propagated per chunk (the slot state is detached between chunks).

  python kvpool.py train --traces traces.jsonl --W 512 --m 64 --C 128 --out pool_W512_m64.safetensors
  python kvpool.py eval  --traces q17_full.jsonl --W 512 --m 64 --ckpt pool_W512_m64.safetensors   (--m 0: drop)
  python kvpool.py gen   --traces none --W 512 --m 64 --ckpt pool_W512_m64.safetensors --out gen.jsonl   (MATH-500 L4-5)
"""
import argparse, json, math, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["train", "eval", "gen"])
ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
ap.add_argument("--traces", required=True)
ap.add_argument("--W", type=int, default=512)
ap.add_argument("--m", type=int, default=64)
ap.add_argument("--C", type=int, default=128)
ap.add_argument("--max-len", type=int, default=4096, help="tokens after the prompt used per trace")
ap.add_argument("--n", type=int, default=0, help="traces used (0 = all)")
ap.add_argument("--skip", type=int, default=0, help="skip the first traces (held out for eval)")
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--ckpt", default="")
ap.add_argument("--out", default="")
ap.add_argument("--log", default="")
ap.add_argument("--device", default="cuda")
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gen-n", type=int, default=64, help="gen: MATH-500 level>=4 problems (same draw as kvwin_eval)")
ap.add_argument("--gen-tokens", type=int, default=8000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--fold", default="learned", choices=["learned", "fit"], help="fit: no weights - at every fold the slots are "
               "re-fitted (top-mass keys, NNLS mass, least-squares values) so that a reservoir of past queries + the window's "
               "queries read the same attention output as from (old slots + evicted tokens)")
ap.add_argument("--reservoir", type=int, default=256, help="fit: past queries kept per q head (uniform reservoir)")
ap.add_argument("--lora", type=int, default=0, help="unfreeze the model: LoRA rank on q,k,v,o,gate,up,down (0 = frozen)")
ap.add_argument("--lora-ckpt", default="", help="eval/gen: a trained LoRA (kvpool train --lora ... --out)")
ap.add_argument("--ft-layers", type=int, default=0, help="full fine-tune of the first N decoder layers (input side), rest frozen; "
               "fp32 master weights (bf16 alone would round small updates away)")
ap.add_argument("--ft-ckpt", default="", help="eval/gen: the trained input-side layers (kvpool train --ft-layers N --out x.safetensors)")
ap.add_argument("--full", action="store_true", help="eval: never evict (full KV) - checks a LoRA keeps the frozen model's normal behaviour")
ap.add_argument("--init-traces", default="", help="gen without --ckpt: traces whose keys seed the codebook")
a = ap.parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import transformers.models.qwen3.modeling_qwen3 as qm
from safetensors.torch import save_file, load_file

torch.manual_seed(a.seed); random.seed(a.seed)
dev = a.device
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), attn_implementation="eager").to(dev).eval()
for p_ in model.parameters(): p_.requires_grad_(False)
PEFT = None
if a.lora or a.lora_ckpt:
    from peft import LoraConfig, get_peft_model, PeftModel
    if a.lora_ckpt:
        PEFT = PeftModel.from_pretrained(model, a.lora_ckpt, is_trainable=a.mode == "train")
    else:
        PEFT = get_peft_model(model, LoraConfig(r=a.lora, lora_alpha=2 * a.lora, lora_dropout=0.0,
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    # the LoRA layers are injected into `model` in place; PEFT.disable_adapter() gives the frozen teacher
FT_SD0 = None   # the frozen weights of the tuned layers, to serve as the teacher
if a.ft_layers or a.ft_ckpt:
    if a.ft_ckpt:
        sd = load_file(a.ft_ckpt)
        FT_N = 1 + max(int(k.split(".")[2]) for k in sd)
    else:
        sd, FT_N = None, a.ft_layers
    FT_KEYS = [k for k in model.state_dict() if k.startswith("model.layers.") and int(k.split(".")[2]) < FT_N]
    FT_SD0 = {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items() if k in FT_KEYS}   # on CPU: GPU memory goes to the fp32 master + Adam
    if sd is not None:
        missing = model.load_state_dict({k: v.to(dev, model.dtype) for k, v in sd.items()}, strict=False)
from contextlib import nullcontext, contextmanager
@contextmanager
def ft_teacher():
    """swap the frozen input-side layers back in for the teacher pass"""
    cur = {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items() if k in FT_SD0}
    model.load_state_dict(FT_SD0, strict=False)
    try: yield
    finally: model.load_state_dict(cur, strict=False)
def teacher_ctx():
    if FT_SD0 is not None: return ft_teacher()
    return PEFT.disable_adapter() if PEFT is not None else nullcontext()
cfg = model.config
NL, H, G = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
D = getattr(cfg, "head_dim", cfg.hidden_size // H); REP = H // G
LOG = open(a.log, "a") if a.log else None
def log(s):
    print(s, flush=True)
    if LOG: LOG.write(s + "\n"); LOG.flush()

# ---- slot log-mass beta as an additive attention-mask column, per layer (the mask itself is shared by all layers)
BETA = {"v": None, "lo": 0}
_eager = qm.eager_attention_forward
def eager_with_beta(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    b = BETA["v"]
    if b is not None and attention_mask is not None:
        bh = b[module.layer_idx].repeat_interleave(REP, 1)[:, :, None, :].to(attention_mask.dtype)   # (B,H,1,m)
        lo, m = BETA["lo"], bh.shape[-1]
        am = attention_mask[:, :, :, :key.shape[-2]]
        am = am.expand(key.shape[0], H, am.shape[2], am.shape[3])
        attention_mask = torch.cat([am[..., :lo], am[..., lo:lo + m] + bh, am[..., lo + m:]], -1)
    return _eager(module, query, key, value, attention_mask, scaling, dropout=dropout, **kw)
qm.eager_attention_forward = eager_with_beta
try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if "eager" in ALL_ATTENTION_FUNCTIONS: ALL_ATTENTION_FUNCTIONS["eager"] = eager_with_beta
except Exception:
    pass


class KVPooler(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.U = nn.Parameter(torch.randn(NL, G, m, D) * 0.02)
        self.s = nn.Parameter(torch.ones(NL, G))
        self.b = nn.Parameter(torch.zeros(NL, G, m))
        self.g = nn.Parameter(torch.zeros(NL, G, m))

    def init_from_keys(self, keys):   # keys[l]: (G, n, D) sample of real keys -> codebook starts on the data
        with torch.no_grad():
            for l, k in enumerate(keys):
                idx = torch.randperm(k.shape[1])[:self.m]
                self.U[l] = k[:, idx].float() / k.float().norm(dim=-1, keepdim=True).mean()

    def fold(self, l, sk, sv, sb, ek, ev):
        """sk/sv (B,G,m,D) or None, sb (B,G,m); ek/ev (B,G,e,D) -> new (k, v, beta)."""
        eb = torch.zeros(ek.shape[:3], device=ek.device)
        xk = ek.float() if sk is None else torch.cat([sk.float(), ek.float()], 2)
        xv = ev.float() if sv is None else torch.cat([sv.float(), ev.float()], 2)
        xb = eb if sb is None else torch.cat([sb.float(), eb], 2)
        z = torch.einsum("gmd,bgnd->bgmn", self.U[l].float(), xk) * (self.s[l][None, :, None, None] / math.sqrt(D))
        logw = torch.log_softmax(z, 2) + xb[:, :, None, :]            # token j's mass split over slots
        M = torch.logsumexp(logw, -1)                                 # (B,G,m) slot log-mass
        om = torch.softmax(logw, -1)
        k = (om @ xk) * torch.exp(self.g[l])[None, :, :, None]
        v = om @ xv
        return k, v, M + self.b[l][None]


# ---- post-RoPE query capture (fit fold): QBUF[layer] = queries of the tokens being forwarded
from transformers.models.qwen3.modeling_qwen3 import rotate_half
QCAP = {"on": False}; QBUF = {}
def _qhook(li):
    def hook(mod, args, kwargs):
        if not QCAP["on"]: return
        hs = kwargs["hidden_states"]; cos, sin = kwargs["position_embeddings"]
        Bq, n = hs.shape[:2]
        q = mod.q_norm(mod.q_proj(hs).view(Bq, n, H, D)).transpose(1, 2)
        QBUF[li] = (q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)).detach().half()
    return hook
for _li, _layer in enumerate(model.model.layers):
    _layer.self_attn.register_forward_pre_hook(_qhook(_li), with_kwargs=True)


@torch.no_grad()
def fit_fold(m, slot, ek, ev, calq, ctx_k):
    """slot: (k,v,beta) or None; ek/ev (B,G,e,D); calq (B,H,Q,D) calibration queries; ctx_k (B,G,L,D) the other keys
    the queries also see (prompt + window). -> new (k, v, beta) with m slots, attention-matched on calq."""
    SC = D ** -0.5
    xk = ek.float() if slot is None else torch.cat([slot[0].float(), ek.float()], 2)
    xv = ev.float() if slot is None else torch.cat([slot[1].float(), ev.float()], 2)
    xb = torch.zeros(ek.shape[:3], device=dev) if slot is None else torch.cat([slot[2].float(), torch.zeros(ek.shape[:3], device=dev)], 2)
    Bq, Gq, n, _ = xk.shape
    if n <= m:
        return xk, xv, xb
    q = calq.float().view(Bq, Gq, REP * calq.shape[2], D)                           # q heads of a group stacked
    sx = torch.einsum("bgqd,bgnd->bgqn", q, xk) * SC + xb[:, :, None, :]
    sc = torch.einsum("bgqd,bgld->bgql", q, ctx_k.float()) * SC
    lZ = torch.logsumexp(torch.cat([sx, sc], -1), -1, keepdim=True)
    pE = torch.exp(sx - lZ)                                                         # share of each X member
    f = pE.sum(-1)                                                                  # (B,G,Q)
    tgt = pE @ xv                                                                   # (B,G,Q,D)
    idx = pE.sum(2).topk(m, -1).indices                                             # (B,G,m)
    kc = torch.gather(xk, 2, idx[..., None].expand(-1, -1, -1, D))
    vc = torch.gather(xv, 2, idx[..., None].expand(-1, -1, -1, D))
    A = torch.exp(torch.einsum("bgqd,bgmd->bgqm", q, kc) * SC - lZ)                 # (B,G,Q,m)
    w = torch.exp(torch.gather(xb, 2, idx)).clamp(min=1e-6)                         # start from the members' own mass
    AtA = A.transpose(-1, -2) @ A; Atf = (A.transpose(-1, -2) @ f[..., None]).squeeze(-1)
    for _ in range(100):                                                            # NNLS by multiplicative updates
        w = w * Atf / ((AtA @ w[..., None]).squeeze(-1) + 1e-12)
    w = w.clamp(min=1e-8)
    Bm = A * w[:, :, None, :]
    BtB = Bm.transpose(-1, -2) @ Bm
    lam = 1e-3 * BtB.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None].clamp(min=1e-12)
    eye = torch.eye(m, device=dev)
    vfit = torch.linalg.solve(BtB + lam * eye, Bm.transpose(-1, -2) @ tgt + lam * vc)
    return kc, vfit, torch.log(w)


def reservoir_add(res, n_seen, newq, R):
    """uniform reservoir over token positions (all heads keep the same tokens). res (B,H,r,D) or None, newq (B,H,e,D)."""
    if res is None: pool, wts = newq, torch.ones(newq.shape[2], device=dev)
    else:
        pool = torch.cat([res, newq], 2)
        wts = torch.cat([torch.full((res.shape[2],), n_seen / res.shape[2], device=dev), torch.ones(newq.shape[2], device=dev)])
    if pool.shape[2] <= R: return pool
    keep = torch.multinomial(wts, R, replacement=False)
    return pool[:, :, keep]


def kv_of(cache, li):
    if hasattr(cache, "layers"): return cache.layers[li].keys, cache.layers[li].values
    return cache.key_cache[li], cache.value_cache[li]


def make_cache(parts):
    c = DynamicCache()
    for li, (k, v) in enumerate(parts): c.update(k, v, li)
    return c


def chunk_forward(ids_chunk, pos0, prompt_kv, slots, window, need_grad):
    """forward a chunk on [prompt | slots | window]; -> logits (B,C,V) and the chunk's own KV per layer."""
    B, Cn = ids_chunk.shape
    parts = []
    for li in range(NL):
        ks, vs = [prompt_kv[li][0]], [prompt_kv[li][1]]
        if slots is not None: ks.append(slots[li][0].to(model.dtype)); vs.append(slots[li][1].to(model.dtype))
        if window is not None: ks.append(window[li][0]); vs.append(window[li][1])
        parts.append((torch.cat(ks, 2), torch.cat(vs, 2)))
    L = parts[0][0].shape[2]
    mask = torch.zeros(B, 1, Cn, L + Cn, device=dev, dtype=model.dtype)
    mask[:, :, :, L:] = torch.triu(torch.full((Cn, Cn), float("-inf"), device=dev), 1).to(model.dtype)
    BETA["v"] = [s[2] for s in slots] if slots is not None else None
    BETA["lo"] = prompt_kv[0][0].shape[2]
    cache = make_cache(parts)
    with torch.set_grad_enabled(need_grad):
        lg = model(input_ids=ids_chunk, position_ids=torch.arange(pos0, pos0 + Cn, device=dev)[None].expand(B, -1),
                   attention_mask=mask, past_key_values=cache, use_cache=True).logits
    BETA["v"] = None
    new = [tuple(t[:, :, L:].detach() for t in kv_of(cache, li)) for li in range(NL)]
    return lg, new


@torch.no_grad()
def teacher_logp(ids, lo, hi):
    """the full model's log-probs for positions [lo, hi) (predicting lo+1 .. hi), in chunks of the lm head."""
    with teacher_ctx():   # the frozen full-KV model is the teacher, also when the student carries a LoRA
        h = model.model(input_ids=ids[None, :hi]).last_hidden_state[0, lo:hi]
    return torch.cat([F.log_softmax(model.lm_head(h[i:i + 256]).float(), -1).to(torch.bfloat16) for i in range(0, h.shape[0], 256)])


def run_trace(ids, P, pooler, opt=None):
    """one trace through the bounded loop; returns (sum KL, sum agree, n tokens)."""
    T = min(len(ids), P + a.max_len)
    first = P + (256 if a.full else a.W + a.C)   # tokens before this are seen with the full context in both models
    if T - first < a.C: return 0.0, 0.0, 0
    ref = teacher_logp(ids, first, T)   # (T-first, V)
    with torch.no_grad():
        c = DynamicCache()
        model(input_ids=ids[None, :P], past_key_values=c, use_cache=True)
        prompt_kv = [tuple(t.detach() for t in kv_of(c, li)) for li in range(NL)]
        QCAP["on"] = FIT
        _, win = chunk_forward(ids[None, P:first], P, prompt_kv, None, None, False)
        winq = [QBUF[li] for li in range(NL)] if FIT else None
    res, n_seen = [None] * NL, 0
    slots, kl_sum, ag_sum, n = None, 0.0, 0.0, 0
    pos = first
    while pos + 1 < T:
        e = min(pos + a.C, T)
        # evict the oldest tokens beyond W, folding them into the slots (grad through this one fold)
        n_ev = 0 if a.full else win[0][0].shape[2] - a.W
        train = opt is not None
        with torch.set_grad_enabled(train):
            if n_ev > 0:
                ev = [(k[:, :, :n_ev], v[:, :, :n_ev]) for k, v in win]
                win = [(k[:, :, n_ev:], v[:, :, n_ev:]) for k, v in win]
                if FIT:
                    evq = [q[:, :, :n_ev] for q in winq]; winq = [q[:, :, n_ev:] for q in winq]
                    res = [reservoir_add(res[li], n_seen, evq[li], a.reservoir) for li in range(NL)]; n_seen += n_ev
                    slots = [fit_fold(a.m, slots[li] if slots is not None else None, ev[li][0], ev[li][1],
                                      torch.cat([res[li], winq[li]], 2),
                                      torch.cat([prompt_kv[li][0], win[li][0]], 2)) for li in range(NL)]
                elif pooler is not None:
                    prev = slots
                    slots = [pooler.fold(li, *(tuple(t.detach() for t in prev[li]) if prev is not None else (None, None, None)),
                                         ev[li][0], ev[li][1]) for li in range(NL)]
        lg, new = chunk_forward(ids[None, pos:e], pos, prompt_kv, slots, win, train)
        if FIT: winq = [torch.cat([winq[li], QBUF[li]], 2) for li in range(NL)]
        lp = F.log_softmax(lg[0].float(), -1)
        r = ref[pos - first:e - first].float()
        kl = (r.exp() * (r - lp)).sum(-1)
        if train:
            kl.mean().backward()
        kl_sum += kl.sum().item(); ag_sum += (lp.argmax(-1) == r.argmax(-1)).float().sum().item(); n += kl.numel()
        win = [(torch.cat([w[0], nw[0]], 2), torch.cat([w[1], nw[1]], 2)) for w, nw in zip(win, new)]
        if slots is not None: slots = [tuple(t.detach() for t in s) for s in slots]
        pos = e
    QCAP["on"] = False
    if opt is not None:
        do_step()
    return kl_sum, ag_sum, n


rows = [json.loads(l) for l in open(a.traces)] if a.mode != "gen" else []
if rows and not all("prompt" in r for r in rows):   # kvwin_eval rows: rebuild the prompt from MATH-500 by id
    from datasets import load_dataset
    PROB = {x["unique_id"]: x["problem"] for x in load_dataset("HuggingFaceH4/MATH-500", split="test")}
    SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    for r in rows:
        r["prompt"] = tok.apply_chat_template([{"role": "user", "content": PROB[r["id"]] + SUFFIX}], add_generation_prompt=True,
                                              tokenize=False, enable_thinking=True)
rows = rows[a.skip:]
if a.n: rows = rows[:a.n]
data = []
for r in rows:
    pid = tok.encode(r["prompt"], add_special_tokens=False)
    data.append((torch.tensor(pid + tok.encode(r["text"], add_special_tokens=False), device=dev), len(pid)))

INIT = []
if a.mode == "gen" and a.init_traces:
    for r in [json.loads(l) for l in open(a.init_traces)][:4]:
        pid = tok.encode(r["prompt"], add_special_tokens=False)
        INIT.append((torch.tensor(pid + tok.encode(r["text"], add_special_tokens=False), device=dev), len(pid)))
FIT = a.fold == "fit" and a.m > 0
pooler = None
if a.m > 0 and not FIT:
    pooler = KVPooler(a.m).to(dev)
    if a.ckpt:
        pooler.load_state_dict({k: v.to(dev) for k, v in load_file(a.ckpt).items()})
    else:   # codebook from the keys of the first traces
        with torch.no_grad():
            ks = [[] for _ in range(NL)]
            for ids, P in (data or INIT)[:4]:
                c = DynamicCache(); model(input_ids=ids[None, :min(len(ids), P + 2048)], past_key_values=c, use_cache=True)
                for li in range(NL): ks[li].append(kv_of(c, li)[0][0, :, P:])
            pooler.init_from_keys([torch.cat(k, 1) for k in ks])

@torch.no_grad()
def generate(prompts):
    """batched sampling with the bounded cache: [prompt | slots | window]; every C new tokens beyond W fold into the slots."""
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    g = torch.Generator(device=dev).manual_seed(a.seed)
    outs, Tt, TOP_P, TOP_K = [], 0.6, 0.95, 20
    for b0 in range(0, len(prompts), a.batch):
        tok.padding_side = "left"
        enc = tok(prompts[b0:b0 + a.batch], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        B, P = enc.input_ids.shape
        pmask = enc.attention_mask
        pos = (pmask.cumsum(1) - 1).clamp(min=0)
        c = DynamicCache()
        lo = model(input_ids=enc.input_ids, attention_mask=pmask, position_ids=pos, past_key_values=c, use_cache=True,
                   logits_to_keep=1).logits[:, -1].clone()   # all-position logits of a batch prefill would pin GBs
        prompt_kv = [tuple(t for t in kv_of(c, li)) for li in range(NL)]
        pad_bias = torch.zeros(B, 1, 1, P, device=dev, dtype=model.dtype).masked_fill(pmask[:, None, None, :] == 0, float("-inf"))
        nxt = pos[:, -1] + 1
        slots, win, winq, res, n_seen = None, None, None, [None] * NL, 0
        QCAP["on"] = FIT
        gen = [[] for _ in range(B)]; done = torch.zeros(B, dtype=torch.bool, device=dev); t0 = time.time()
        for step in range(a.gen_tokens):
            l_ = lo.float() / Tt
            kth = torch.topk(l_, TOP_K).values[:, -1:]
            l_ = l_.masked_fill(l_ < kth, -1e9)
            pr = torch.softmax(l_, -1); sp, si = pr.sort(-1, descending=True)
            sp = sp.masked_fill(sp.cumsum(-1) - sp > TOP_P, 0)
            nx = si.gather(1, torch.multinomial(sp, 1, generator=g)).squeeze(1)
            nx = torch.where(done, torch.full_like(nx, tok.pad_token_id), nx)
            for i in range(B):
                if not done[i]:
                    gen[i].append(int(nx[i]))
                    if int(nx[i]) in eos: done[i] = True
            if bool(done.all()): break
            if win is not None and win[0][0].shape[2] >= a.W + a.C:      # fold the oldest C tokens
                ev = [(k[:, :, :a.C], v[:, :, :a.C]) for k, v in win]
                win = [(k[:, :, a.C:], v[:, :, a.C:]) for k, v in win]
                if FIT:
                    evq = [q[:, :, :a.C] for q in winq]; winq = [q[:, :, a.C:] for q in winq]
                    res = [reservoir_add(res[li], n_seen, evq[li], a.reservoir) for li in range(NL)]; n_seen += a.C
                    slots = [fit_fold(a.m, slots[li] if slots is not None else None, ev[li][0], ev[li][1],
                                      torch.cat([res[li], winq[li]], 2), torch.cat([prompt_kv[li][0], win[li][0]], 2))
                             for li in range(NL)]
                elif pooler is not None:
                    slots = [pooler.fold(li, *(slots[li] if slots is not None else (None, None, None)), ev[li][0], ev[li][1])
                             for li in range(NL)]
            parts = []
            for li in range(NL):
                ks, vs = [prompt_kv[li][0]], [prompt_kv[li][1]]
                if slots is not None: ks.append(slots[li][0].to(model.dtype)); vs.append(slots[li][1].to(model.dtype))
                if win is not None: ks.append(win[li][0]); vs.append(win[li][1])
                parts.append((torch.cat(ks, 2), torch.cat(vs, 2)))
            L = parts[0][0].shape[2]
            mask = torch.cat([pad_bias, torch.zeros(B, 1, 1, L + 1 - P, device=dev, dtype=model.dtype)], -1)
            BETA["v"] = [s_[2] for s_ in slots] if slots is not None else None; BETA["lo"] = P
            cache = make_cache(parts)
            lo = model(input_ids=nx[:, None], position_ids=nxt[:, None], attention_mask=mask, past_key_values=cache, use_cache=True).logits[:, -1]
            BETA["v"] = None
            new = [tuple(t[:, :, L:].clone() for t in kv_of(cache, li)) for li in range(NL)]   # views would pin the whole cache
            del cache, parts
            win = new if win is None else [(torch.cat([w[0], n_[0]], 2), torch.cat([w[1], n_[1]], 2)) for w, n_ in zip(win, new)]
            if FIT: winq = [QBUF[li] if winq is None else torch.cat([winq[li], QBUF[li]], 2) for li in range(NL)]
            nxt = nxt + 1
        QCAP["on"] = False
        log(f"[kvpool] gen batch {b0}: {step + 1} steps {time.time() - t0:.0f}s")
        outs += [tok.decode(x, skip_special_tokens=True) for x in gen]
        del c, prompt_kv, slots, win; torch.cuda.empty_cache() if dev == "cuda" else None
    return outs


if a.mode == "gen":
    from datasets import load_dataset
    from math_verify import parse, verify
    ds = [r for r in load_dataset("HuggingFaceH4/MATH-500", split="test") if int(r["level"]) >= 4]
    random.Random(0).shuffle(ds); ds = ds[:a.gen_n]
    SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["problem"] + SUFFIX}], add_generation_prompt=True,
                                       tokenize=False, enable_thinking=True) for r in ds]
    texts = generate(prompts)
    n_ok = n_fin = 0
    with open(a.out, "w") as f:
        for r, t in zip(ds, texts):
            fin = "</think>" in t
            ok = bool(fin and verify(parse("$" + r["answer"] + "$"), parse(t.split("</think>")[-1])))
            n_ok += ok; n_fin += fin
            f.write(json.dumps({"id": r["unique_id"], "level": r["level"], "gold": r["answer"], "ok": ok, "fin": fin, "text": t}) + "\n")
    log(f"[kvpool] gen W{a.W} m{a.m} C{a.C} {os.path.basename(a.ckpt) or ('drop' if a.m == 0 else ('fit R%d' % a.reservoir if FIT else 'init'))}: {n_ok}/{len(ds)} = {100 * n_ok / len(ds):.1f}%  finished {n_fin}/{len(ds)}")

tag = f"W{a.W} m{a.m} C{a.C}" + (f" fit R{a.reservoir}" if FIT else "") + (f" LoRA" if PEFT is not None else "") + (" FT-in" if FT_SD0 is not None else "")
def save_all():
    if not a.out: return
    if pooler is not None: save_file({k_: v.detach().cpu().contiguous() for k_, v in pooler.state_dict().items()}, a.out)
    if PEFT is not None: PEFT.save_pretrained(a.out if pooler is None else a.out + ".lora")
    if a.ft_layers:
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k in FT_SD0}, a.out)
if a.mode == "gen":
    pass
elif a.mode == "eval":
    t0 = time.time(); K = A = N = 0
    for ids, P in data:
        k, g_, n = run_trace(ids, P, pooler); K += k; A += g_; N += n
    log(f"[kvpool] eval {tag} {os.path.basename(a.ckpt or a.lora_ckpt or a.ft_ckpt) or ('drop' if a.m == 0 else ('fit' if FIT else 'init'))}: {len(data)} traces {N} tokens  "
        f"KL {K / max(N, 1):.4f}  top1 agree {100 * A / max(N, 1):.2f}%  ({time.time() - t0:.0f}s)")
else:
    TRAINABLE = ([p_ for p_ in pooler.parameters()] if pooler is not None else []) + \
                ([p_ for p_ in model.parameters() if p_.requires_grad] if PEFT is not None else [])
    FTP = []
    if a.ft_layers:
        FTP = [p_ for li in range(a.ft_layers) for p_ in model.model.layers[li].parameters()]
        for p_ in FTP: p_.requires_grad_(True)
    MASTER = [p_.detach().float().clone().requires_grad_(True) for p_ in FTP]
    log(f"[kvpool] trainable params: {(sum(p_.numel() for p_ in TRAINABLE) + sum(p_.numel() for p_ in FTP)) / 1e6:.1f}M "
        f"(pooler {'yes' if pooler is not None else 'no'}, LoRA r{a.lora}, full FT of layers 0-{a.ft_layers - 1})")
    opt = torch.optim.AdamW(TRAINABLE + MASTER, lr=a.lr, weight_decay=0.0)
    def do_step():
        for p_, m_ in zip(FTP, MASTER):
            m_.grad = p_.grad.float() if p_.grad is not None else None; p_.grad = None
        torch.nn.utils.clip_grad_norm_(TRAINABLE + MASTER, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for p_, m_ in zip(FTP, MASTER): p_.copy_(m_.to(p_.dtype))
    steps = int(len(data) * a.epochs); t0 = time.time(); K = A = N = 0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.1, 1 - s / max(steps, 1)))
    for st in range(steps):
        ids, P = data[st % len(data)] if st >= len(data) else data[st]
        k, g_, n = run_trace(ids, P, pooler, opt); sched.step()
        K += k; A += g_; N += n
        if (st + 1) % 25 == 0 or st + 1 == steps:
            log(f"[kvpool] train {tag} step {st + 1}/{steps}: KL {K / max(N, 1):.4f} top1 {100 * A / max(N, 1):.2f}% ({time.time() - t0:.0f}s)")
            K = A = N = 0
            save_all()
    save_all()
    log(f"[kvpool] trained {tag} -> {a.out}")
