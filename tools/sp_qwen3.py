"""deep-charger's SP-evict hypernet, ported to a new base (Qwen3-1.7B): same design, the numbers as arguments.

Design (train_sched_evict.py / pooler_mlx.py, unchanged):
  * AttnPoolSP: S learned queries cross-attend to the token embeddings (+ sinusoidal PE) of the distant buffer, then
    self-attention and an FFN, `layers` blocks; output LayerNorm -> L2-normalised -> x |out_scale| (initialised to the
    mean norm of the model's token embeddings): the soft prompts live on the embedding sphere.
  * distant buffer: tokens that left the raw window, at most maxD, evicted by the pooler's own cross-attention mass.
  * generation / training loop per chunk of C tokens: cache cropped to the prompt, re-prefill [SP | raw window (rw)],
    then the chunk.
  * objective: KL(full-context teacher || SP student) on every token of the chunk, teacher-forced on the model's own
    full-KV traces.
Stage 1 trains the pooler with the model frozen; --ft-layers N also full-fine-tunes the first N decoder layers
(N = all layers: FFT) with fp32 master weights (+ --opt8bit for 24 GB cards); the teacher is the frozen copy.

  python sp_qwen3.py train --traces traces.jsonl --skip 60 --n 400 --S 32 --rw 768 --C 128 --maxd 384 --out sp.pt
  python sp_qwen3.py eval  --traces q17_full.jsonl --ckpt sp.pt          (KL / top-1 agreement vs full KV)
"""
import argparse, json, math, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["train", "eval", "profile"])
ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
ap.add_argument("--traces", required=True)
ap.add_argument("--n", type=int, default=0); ap.add_argument("--skip", type=int, default=0)
ap.add_argument("--max-len", type=int, default=3072, help="trace tokens used after the prompt")
ap.add_argument("--S", type=int, default=32, help="soft prompts")
ap.add_argument("--rw", type=int, default=768, help="raw (exposure) window")
ap.add_argument("--C", type=int, default=128, help="chunk")
ap.add_argument("--maxd", type=int, default=384, help="distant buffer cap (tokens)")
ap.add_argument("--layers", type=int, default=3); ap.add_argument("--heads", type=int, default=8); ap.add_argument("--ffn", type=int, default=2048)
ap.add_argument("--lr", type=float, default=1e-4, help="pooler lr")
ap.add_argument("--ft-layers", type=int, default=0, help="also full fine-tune the first N decoder layers")
ap.add_argument("--ft-lr", type=float, default=1e-5)
ap.add_argument("--opt8bit", action="store_true")
ap.add_argument("--accum", type=int, default=8, help="traces per optimizer step")
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--ckpt", default="")
ap.add_argument("--out", default="")
ap.add_argument("--log", default="")
ap.add_argument("--chunks-per-trace", type=int, default=0, help="train: back-propagate only this many random chunks per trace "
               "(the rest only advance the buffer) - more distinct traces per hour (0 = every chunk)")
ap.add_argument("--par", action="store_true", help="train: one forward of [prompt | SP(buffer at boundary b) | the next rw+C "
               "tokens] scores EVERY next-token prediction in it - window 0 (right after the soft prompts) up to rw+C - instead of "
               "one chunk after a full window; --chunks-per-trace boundaries per trace")
ap.add_argument("--spnext", action="store_true", help="train: window 0 only - [prompt | SP_t] -> token t for (nearly) every "
               "position t of a trace, in batches of --sn-batch positions (memory fixed by the batch, not the trace)")
ap.add_argument("--sn-step", type=int, default=16, help="spnext: buffer snapshot granularity (tokens)")
ap.add_argument("--sn-batch", type=int, default=64, help="spnext: positions per forward")
ap.add_argument("--sn-tokens", type=int, default=16384, help="spnext: cap on batch x (prompt + S) - the prompt KV is "
               "repeated per batch item, so long prompts get smaller batches")
ap.add_argument("--sn-pos", type=int, default=0, help="spnext: positions sampled per trace (0 = all)")
ap.add_argument("--sn-win", default="1", help="spnext curriculum: tokens predicted after SP_t (1 = window 0 only; w = SP_t + the next "
                "w-1 raw tokens, scoring windows 0..w-1), comma list stepped every --sn-win-every traces, e.g. 1,2,4,8,16")
ap.add_argument("--sn-win-every", type=int, default=150, help="spnext curriculum: traces per --sn-win step")
ap.add_argument("--sn-win-plateau", type=float, default=0.0, help="spnext curriculum: step to the next w when the last 100 traces' "
                "KL is not this fraction below the 100 before (0 = fixed --sn-win-every); after the last w plateaus, training stops")
ap.add_argument("--sn-win-block", type=int, default=4, help="spnext plateau rule: logs (25 traces each) per compared block")
ap.add_argument("--sn-win-max", type=int, default=800, help="spnext curriculum with --sn-win-plateau: most traces at one w")
ap.add_argument("--sn-t0", type=int, default=16, help="spnext: first answer position trained")
ap.add_argument("--score-from", type=int, default=0, help="count only answer tokens at index >= this (a common region when rw differs)")
ap.add_argument("--pe-rev", action="store_true", help="pooler positions counted from the NEWEST buffer token (0 = the token "
               "just before the window / the predicted position) instead of from the oldest - saved in the checkpoint")
ap.add_argument("--pe-norm", action="store_true", help="scale the pooler's sinusoidal positions to the mean embedding norm "
                "(raw sinusoids have norm sqrt(H/2) = 32 against token embeddings of ~1.5: the content was ~5%% of the key/value signal)")
ap.add_argument("--fresh-pooler", action="store_true", help="train: take only the fine-tuned model weights from --ckpt, start a new pooler")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda"); ap.add_argument("--dtype", default="bfloat16")
a = ap.parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
torch.manual_seed(a.seed); random.seed(a.seed)
dev, dt = a.device, getattr(torch, a.dtype)
tok = AutoTokenizer.from_pretrained(a.model)
llm = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, attn_implementation="sdpa").to(dev).eval()
for p_ in llm.parameters(): p_.requires_grad_(False)
embed = llm.get_input_embeddings(); H = llm.config.hidden_size; NL = llm.config.num_hidden_layers
LOGF = open(a.log, "a") if a.log else None
def log(s):
    print(s, flush=True)
    if LOGF: LOGF.write(s + "\n"); LOGF.flush()


def sinusoidal(L, Hd, device):
    pos = torch.arange(L, device=device).float().unsqueeze(1)
    i = torch.arange(0, Hd, 2, device=device).float()
    div = torch.exp(-math.log(10000.0) * i / Hd)
    pe = torch.zeros(L, Hd, device=device)
    pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Block(nn.Module):
    def __init__(self, Hd, heads, ffn):
        super().__init__()
        self.lnq1 = nn.LayerNorm(Hd); self.lnk = nn.LayerNorm(Hd)
        self.cross = nn.MultiheadAttention(Hd, heads, batch_first=True)
        self.lnq2 = nn.LayerNorm(Hd)
        self.selfa = nn.MultiheadAttention(Hd, heads, batch_first=True)
        self.lnq3 = nn.LayerNorm(Hd)
        self.ffn = nn.Sequential(nn.Linear(Hd, ffn), nn.GELU(), nn.Linear(ffn, Hd))

    def forward(self, q, past, return_w=False, kpm=None):
        w = None
        if past is not None and past.size(1) > 0:
            kn = self.lnk(past)
            a_, w = self.cross(self.lnq1(q), kn, kn, need_weights=return_w, key_padding_mask=kpm)
            q = q + a_
        s, _ = self.selfa(self.lnq2(q), self.lnq2(q), self.lnq2(q), need_weights=False); q = q + s
        q = q + self.ffn(self.lnq3(q))
        return (q, w) if return_w else q


class AttnPoolSP(nn.Module):
    def __init__(self, Hd, n_sp, heads, layers, ffn, target_norm):
        super().__init__()
        self.H = Hd; self.n_sp = n_sp
        self.query = nn.Parameter(torch.randn(n_sp, Hd) * 0.02)
        self.blocks = nn.ModuleList([Block(Hd, heads, ffn) for _ in range(layers)])
        self.ln_out = nn.LayerNorm(Hd)
        self.out_scale = nn.Parameter(torch.tensor(float(target_norm)))
        self._pe = None
        self.pe_rev = False; self.pe_gain = 1.0

    def pe(self, L, device):
        if self._pe is None or self._pe.size(0) < L: self._pe = sinusoidal(max(L, 1024), self.H, device) * self.pe_gain
        return self._pe[:L]

    def _run(self, past_emb, want_mass, kpm=None):
        B, L = past_emb.size(0), past_emb.size(1)
        if L > 0 and self.pe_rev:   # distance from the newest real token of each row (right padding)
            lens = (~kpm).sum(1) if kpm is not None else torch.full((B,), L, device=past_emb.device)
            idx = (lens[:, None] - 1 - torch.arange(L, device=past_emb.device)[None]).clamp(min=0)
            past = past_emb + self.pe(L, past_emb.device)[idx]
        elif L > 0:
            past = past_emb + self.pe(L, past_emb.device).unsqueeze(0)
        else:
            past = past_emb
        q = self.query.unsqueeze(0).expand(B, -1, -1).contiguous()
        mass = torch.zeros(B, L, device=past_emb.device) if want_mass else None
        for blk in self.blocks:
            if want_mass:
                q, w = blk(q, past, return_w=True)
                if w is not None: mass = mass + w.sum(dim=1)
            else:
                q = blk(q, past, kpm=kpm)
        sp = self.ln_out(q)
        sp = sp / sp.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self.out_scale.abs()   # output normalisation
        return (sp, mass) if want_mass else sp

    def forward(self, past_emb, kpm=None): return self._run(past_emb, False, kpm)   # kpm: (B, L) True = padding
    def forward_with_mass(self, past_emb): return self._run(past_emb, True)


with torch.no_grad():
    target_norm = embed(torch.randint(0, embed.weight.size(0), (2048,), device=dev)).float().norm(dim=-1).mean().item()
pooler = AttnPoolSP(H, a.S, a.heads, a.layers, a.ffn, target_norm).to(dev)
FT_SD = None
if a.ckpt:
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if not (a.fresh_pooler and a.mode == "train"): pooler.load_state_dict(ck["pooler"])
    if a.mode != "train":   # eval follows the checkpoint
        if (ck.get("args") or {}).get("pe_rev"): a.pe_rev = True
        if (ck.get("args") or {}).get("pe_norm"): a.pe_norm = True
    if ck.get("ft"):
        FT_SD = {k: v.detach().to("cpu", copy=True) for k, v in llm.state_dict().items() if k in ck["ft"]}   # frozen copy = teacher
        llm.load_state_dict({k: v.to(dev, dt) for k, v in ck["ft"].items()}, strict=False)
pooler.pe_rev = a.pe_rev
if a.pe_norm: pooler.pe_gain = target_norm / math.sqrt(H / 2)   # sinusoid norm sqrt(H/2) -> the embedding norm
n_pool = sum(x.numel() for x in pooler.parameters())
KV_TOK = NL * 2 * llm.config.num_key_value_heads * getattr(llm.config, "head_dim", H // llm.config.num_attention_heads) * 2
log(f"[sp] pooler {n_pool / 1e6:.1f}M params (4-bit ~{n_pool / 2 / 2**20:.0f} MB) | S={a.S} rw={a.rw} C={a.C} maxd={a.maxd} | "
    f"bounded KV {(a.S + a.rw + a.C) * KV_TOK / 2**20:.0f} MB + prompt | target_norm {target_norm:.3f}")


@torch.no_grad()
def evict_topk(ids, mass, maxD):
    if ids.size(1) <= maxD: return ids
    idx = mass.topk(maxD, dim=1).indices.sort(dim=1).values
    return ids.gather(1, idx)


@torch.no_grad()
def update_buffer(kept, absorbed, src, nd_end):
    if nd_end > absorbed:
        kept = torch.cat([kept, src[:, absorbed:nd_end]], 1); absorbed = nd_end
        if a.maxd and kept.size(1) > a.maxd:
            _, mass = pooler.forward_with_mass(embed(kept).float())
            kept = evict_topk(kept, mass, a.maxd)
    return kept, absorbed


def kv_of(cache, li):
    if hasattr(cache, "layers"): return cache.layers[li].keys, cache.layers[li].values
    return cache.key_cache[li], cache.value_cache[li]


SWAP = {"pairs": None, "depth": 0}   # training: (param, pinned frozen copy, fp32 master) - fast in-place swap


class _Frozen:
    """swap the frozen copy of the tuned layers in for the teacher pass (re-entrant: nested uses are no-ops)"""
    def __enter__(self):
        SWAP["depth"] += 1
        if FT_SD is None or SWAP["depth"] > 1: return
        if SWAP["pairs"] is not None:      # training: frozen copy from pinned CPU, restore from the fp32 master (on GPU)
            with torch.no_grad():
                for p_, f_, m_ in SWAP["pairs"]: p_.copy_(f_, non_blocking=True)
            return
        self.cur = {k: v.detach().to("cpu", copy=True) for k, v in llm.state_dict().items() if k in FT_SD}
        llm.load_state_dict(FT_SD, strict=False)
    def __exit__(self, *e):
        SWAP["depth"] -= 1
        if FT_SD is None or SWAP["depth"] > 0: return
        if SWAP["pairs"] is not None:
            with torch.no_grad():
                for p_, f_, m_ in SWAP["pairs"]: p_.copy_(m_.to(p_.dtype))
            return
        llm.load_state_dict(self.cur, strict=False)


@torch.no_grad()
def teacher_hidden(ids, lo, hi):
    """the full-context teacher's last hidden states whose rows predict tokens lo..hi-1 (lm head applied per chunk)"""
    with _Frozen():
        return llm.model(input_ids=ids[None, :hi]).last_hidden_state[0, lo - 1:hi - 1]


@torch.no_grad()
def teacher_rows(h):
    # the lm head is never tuned, so no weight swap here (swapping in place between the student's forward and backward
    # would invalidate its graph)
    return F.log_softmax(llm.lm_head(h).float(), -1)


def run_trace(ids, P, train, ref_h=None):
    """teacher-forced SP-evict over one trace: returns (sum KL, sum top-1 agree, n tokens) on tokens past the window."""
    T = min(len(ids), P + a.max_len)
    a_ids = ids[None, P:T]; La = a_ids.size(1)
    first = a.rw + a.C          # chunks before this see the whole answer so far in the raw window (no eviction yet)
    if La - first < a.C: return 0.0, 0.0, 0
    ids = ids.to(dev); a_ids = a_ids.to(dev)
    if ref_h is None: ref_h = teacher_hidden(ids, P + first, T)
    c0s = list(range(first, La, a.C))
    pick = set(c0s)
    if train and a.chunks_per_trace and len(c0s) > a.chunks_per_trace:
        pick = set(random.sample(c0s, a.chunks_per_trace))
    with torch.no_grad():
        pc = DynamicCache(); llm(input_ids=ids[None, :P], past_key_values=pc, use_cache=True)
        pkv = [tuple(t for t in kv_of(pc, li)) for li in range(NL)]
    kept = torch.zeros(1, 0, dtype=torch.long, device=dev); absorbed = 0
    K = A_ = N = 0.0
    for c0 in range(first, La, a.C):
        c1 = min(c0 + a.C, La); cur = c1 - c0
        if cur < 4: break
        R = min(c0, a.rw); nd_end = c0 - R
        kept, absorbed = update_buffer(kept, absorbed, a_ids, nd_end)
        if c0 not in pick: continue
        with torch.set_grad_enabled(train):
            sp = pooler(embed(kept).float()) if kept.size(1) > 0 else pooler(torch.zeros(1, 0, H, device=dev))
            block = torch.cat([sp.to(dt), embed(a_ids[:, c0 - R:c0]).to(dt), embed(a_ids[:, c0:c1]).to(dt)], 1)
            nb = block.size(1)
            cache = DynamicCache()
            for li, (k_, v_) in enumerate(pkv): cache.update(k_, v_, li)
            pos = torch.arange(P, P + nb, device=dev)[None]
            o = llm(inputs_embeds=block, attention_mask=torch.ones(1, P + nb, device=dev), past_key_values=cache,
                    position_ids=pos, cache_position=pos[0], use_cache=True, logits_to_keep=cur + 1)
            # kept rows: the last window token + the chunk's tokens but its last -> they predict answer tokens c0 .. c1-1
            sl = F.log_softmax(o.logits[0, :cur].float(), -1)
            tr = teacher_rows(ref_h[c0 - first:c1 - first])     # teacher row j predicts answer token first + j
            kl = (tr.exp() * (tr - sl)).sum(-1)
            if train: (kl.sum() / 1000.0).backward()
        if c0 >= a.score_from:
            K += kl.sum().item(); A_ += (sl.argmax(-1) == tr.argmax(-1)).float().sum().item(); N += kl.numel()
    return K, A_, N


from torch.utils.checkpoint import checkpoint


def _kl_rows(h, tr):
    """KL(teacher rows || student rows from hidden h) summed, plus top-1 agreements (logits rebuilt in backward)"""
    lp = F.log_softmax(llm.lm_head(h).float(), -1)
    return (tr.exp() * (tr - lp)).sum(), (lp.argmax(-1) == tr.argmax(-1)).float().sum().detach()


def run_trace_par(ids, P, ref_h=None):
    """--par training on one trace: buffer snapshots at chunk boundaries (sequential, no grad - eviction never had a
    gradient), then for a few boundaries b one forward of [prompt | SP_b | a[b : b+rw+C]] scoring all rw+C predictions."""
    T = min(len(ids), P + a.max_len)
    ids = ids.to(dev); a_ids = ids[None, P:T]; La = a_ids.size(1)
    L = a.rw + a.C
    bounds = [b for b in range(a.C, La - L + 1, a.C)]
    if not bounds: return 0.0, 0.0, 0
    if ref_h is None: ref_h = teacher_hidden(ids, P + a.C, T)     # row j predicts answer token C + j
    pick = set(random.sample(bounds, min(len(bounds), a.chunks_per_trace or len(bounds))))
    with torch.no_grad():
        pc = DynamicCache(); llm(input_ids=ids[None, :P], past_key_values=pc, use_cache=True)
        pkv = [tuple(t for t in kv_of(pc, li)) for li in range(NL)]
    kept = torch.zeros(1, 0, dtype=torch.long, device=dev); absorbed = 0
    K = A_ = N = 0.0
    for b in bounds:
        kept, absorbed = update_buffer(kept, absorbed, a_ids, b)
        if b not in pick: continue
        sp = pooler(embed(kept).float()) if kept.size(1) > 0 else pooler(torch.zeros(1, 0, H, device=dev))
        block = torch.cat([sp.to(dt), embed(a_ids[:, b:b + L - 1]).to(dt)], 1)       # predicts a[b .. b+L-1]
        nb = block.size(1)
        cache = DynamicCache()
        for li, (k_, v_) in enumerate(pkv): cache.update(k_, v_, li)
        pos = torch.arange(P, P + nb, device=dev)[None]
        hs = llm.model(inputs_embeds=block, attention_mask=torch.ones(1, P + nb, device=dev), past_key_values=cache,
                       position_ids=pos, cache_position=pos[0], use_cache=True).last_hidden_state[0, a.S - 1:]   # (L, H)
        loss = 0.0
        for i in range(0, L, 256):
            tr = teacher_rows(ref_h[b - a.C + i:b - a.C + min(i + 256, L)])
            kl, ag = checkpoint(_kl_rows, hs[i:i + 256], tr, use_reentrant=False)
            loss = loss + kl; K += kl.item(); A_ += ag.item(); N += tr.shape[0]
        (loss / 1000.0).backward()
    return K, A_, N


SNW = {"w": 1, "k0": 0.0, "n0": 0}   # spnext: the current curriculum window and the window-0 KL since the last log


def run_trace_spnext(ids, P, ref_h=None):
    """--spnext training on one trace: the buffer before every position t (snapshots every sn_step tokens, mass-evicted,
    no grad) -> SP_t -> [prompt | SP_t | a[t : t+w-1]] predicts answer tokens t .. t+w-1 (windows 0 .. w-1, w = the
    curriculum step, 1 = SP alone); positions batched, backward per batch."""
    w = SNW["w"]
    T = min(len(ids), P + a.max_len)
    ids = ids.to(dev); a_ids = ids[None, P:T]; La = a_ids.size(1); t0 = a.sn_t0
    if La - t0 - w + 1 < a.sn_batch // 4: return 0.0, 0.0, 0
    if ref_h is None: ref_h = teacher_hidden(ids, P + t0, T)          # row j predicts answer token t0 + j
    with torch.no_grad():
        pc = DynamicCache(); llm(input_ids=ids[None, :P], past_key_values=pc, use_cache=True)
        pkv = [tuple(t for t in kv_of(pc, li)) for li in range(NL)]
        snaps = {}; kept = torch.zeros(1, 0, dtype=torch.long, device=dev); absorbed = 0
        for g in range(0, La, a.sn_step):
            kept, absorbed = update_buffer(kept, absorbed, a_ids, g); snaps[g] = kept[0]
    pos = list(range(t0, La - w + 1))
    npos = max(16, a.sn_pos // w) if a.sn_pos else 0                  # about the same scored tokens per trace at every w
    if npos and len(pos) > npos: pos = sorted(random.sample(pos, npos))
    K = A_ = N = 0.0
    bsz = max(1, min(a.sn_batch, a.sn_tokens // (P + a.S + w), max(1, 256 // w)))   # <= ~256 scored rows (B x w x vocab) per forward
    for i in range(0, len(pos), bsz):
        ts = pos[i:i + bsz]; B = len(ts)
        bufs = []
        for t in ts:
            g = (t // a.sn_step) * a.sn_step
            bufs.append(torch.cat([snaps[g], a_ids[0, g:t]]))             # snapshot + the raw tokens since it
        Lb = max(1, max(len(x) for x in bufs))
        pad = torch.zeros(B, Lb, dtype=torch.long, device=dev); kpm = torch.ones(B, Lb, dtype=torch.bool, device=dev)
        for j, x in enumerate(bufs):
            pad[j, :len(x)] = x; kpm[j, :len(x)] = False
        kpm[:, 0] = False                                                  # an empty buffer still attends to one slot
        sp = pooler(embed(pad).float(), kpm=kpm)                          # (B, S, H)
        x_in = sp.to(dt)
        if w > 1:                                                          # the raw window after SP_t: a[t : t+w-1]
            raw = torch.stack([a_ids[0, t:t + w - 1] for t in ts])
            x_in = torch.cat([x_in, embed(raw).to(dt)], 1)
        nb = a.S + w - 1
        cache = DynamicCache()
        for li, (k_, v_) in enumerate(pkv): cache.update(k_.expand(B, -1, -1, -1), v_.expand(B, -1, -1, -1), li)
        pp = torch.arange(P, P + nb, device=dev)[None].expand(B, -1)
        h = llm.model(inputs_embeds=x_in, attention_mask=torch.ones(B, P + nb, device=dev), past_key_values=cache,
                      position_ids=pp, cache_position=pp[0], use_cache=True).last_hidden_state[:, a.S - 1:]   # (B, w, H)
        rows = (torch.tensor(ts, device=dev) - t0)[:, None] + torch.arange(w, device=dev)[None]          # (B, w)
        tr = teacher_rows(ref_h[rows.reshape(-1)]).view(B, w, -1)
        lp = F.log_softmax(llm.lm_head(h).float(), -1)
        kl = (tr.exp() * (tr - lp)).sum(-1)                                # (B, w)
        (kl.sum() / 1000.0).backward()
        K += kl.sum().item(); A_ += (lp.argmax(-1) == tr.argmax(-1)).float().sum().item(); N += B * w
        SNW["k0"] += kl[:, 0].sum().item(); SNW["n0"] += B
    return K, A_, N


@torch.no_grad()
def profile_trace(ids, P, acc):
    """KL by window length: one [prompt | SP_b | a[b : b+rw+C]] forward per boundary b (as --par training sees it)."""
    T = min(len(ids), P + a.max_len)
    ids = ids.to(dev); a_ids = ids[None, P:T]; La = a_ids.size(1)
    L = a.rw + a.C
    bounds = list(range(a.C, La - L + 1, a.C))
    if not bounds: return
    ref_h = teacher_hidden(ids, P + a.C, T)
    pc = DynamicCache(); llm(input_ids=ids[None, :P], past_key_values=pc, use_cache=True)
    pkv = [tuple(t for t in kv_of(pc, li)) for li in range(NL)]
    kept = torch.zeros(1, 0, dtype=torch.long, device=dev); absorbed = 0
    for b in bounds:
        kept, absorbed = update_buffer(kept, absorbed, a_ids, b)
        sp = pooler(embed(kept).float()) if kept.size(1) > 0 else pooler(torch.zeros(1, 0, H, device=dev))
        block = torch.cat([sp.to(dt), embed(a_ids[:, b:b + L - 1]).to(dt)], 1); nb = block.size(1)
        cache = DynamicCache()
        for li, (k_, v_) in enumerate(pkv): cache.update(k_, v_, li)
        pos = torch.arange(P, P + nb, device=dev)[None]
        hs = llm.model(inputs_embeds=block, attention_mask=torch.ones(1, P + nb, device=dev), past_key_values=cache,
                       position_ids=pos, cache_position=pos[0], use_cache=True).last_hidden_state[0, a.S - 1:]
        for i in range(0, L, 256):
            tr = teacher_rows(ref_h[b - a.C + i:b - a.C + min(i + 256, L)])
            lp = F.log_softmax(llm.lm_head(hs[i:i + 256]).float(), -1)
            kl = (tr.exp() * (tr - lp)).sum(-1)
            for j, v in enumerate(kl.tolist()):
                w = i + j
                key = "0" if w == 0 else "1-7" if w < 8 else "8-63" if w < 64 else "64-511" if w < 512 else "512+"
                acc.setdefault(key, [0.0, 0]); acc[key][0] += v; acc[key][1] += 1


rows = [json.loads(l) for f_ in a.traces.split(",") for l in open(f_)]
if not all("prompt" in r for r in rows):
    from datasets import load_dataset
    PROB = {x["unique_id"]: x["problem"] for x in load_dataset("HuggingFaceH4/MATH-500", split="test")}
    SUF = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    for r in rows:
        r["prompt"] = tok.apply_chat_template([{"role": "user", "content": PROB[r["id"]] + SUF}], add_generation_prompt=True,
                                              tokenize=False, enable_thinking=True)
rows = rows[a.skip:]
if a.n: rows = rows[:a.n]
data = []
for r in rows:
    pid = tok.encode(r["prompt"], add_special_tokens=False)
    data.append((torch.tensor(pid + tok.encode(r["text"], add_special_tokens=False)), len(pid)))   # CPU; moved per trace

tag = f"S{a.S} rw{a.rw} C{a.C} maxd{a.maxd} L{a.layers} ffn{a.ffn}" + (" par" if a.par and a.mode == "train" else "") + (" spnext" + (f" win{a.sn_win}/{a.sn_win_every}" if a.sn_win != "1" else "") if a.spnext and a.mode == "train" else "") + (" pe-rev" if a.pe_rev else "") + (" pe-norm" if a.pe_norm else "") + (f" FT{a.ft_layers}" if a.ft_layers else "") + (f" from{a.score_from}" if a.score_from else "")
if a.mode == "profile":
    pooler.eval(); acc = {}
    for ids, P in data: profile_trace(ids, P, acc)
    tot = sum(v[0] for v in acc.values()); n = sum(v[1] for v in acc.values())
    log(f"[sp] profile {tag} {os.path.basename(a.ckpt) or 'untrained'}: mean KL over all windows {tot / max(n, 1):.4f}")
    for key in ["0", "1-7", "8-63", "64-511", "512+"]:
        if key in acc: log(f"[sp] profile   window {key:>7}: KL {acc[key][0] / acc[key][1]:.4f}  share of the summed KL {100 * acc[key][0] / max(tot, 1e-9):.1f}%  ({acc[key][1]} tokens)")
elif a.mode == "eval":
    pooler.eval(); t0 = time.time(); K = A_ = N = 0
    for ids, P in data:
        k, g, n = run_trace(ids, P, False); K += k; A_ += g; N += n
    log(f"[sp] eval {tag} {os.path.basename(a.ckpt) or 'untrained'}: {len(data)} traces {int(N)} tokens  KL {K / max(N, 1):.4f}  "
        f"top1 agree {100 * A_ / max(N, 1):.2f}%  ({time.time() - t0:.0f}s)")
else:
    pooler.train()
    groups = [{"params": list(pooler.parameters()), "lr": a.lr}]
    FTP, MASTER = [], []
    if a.ft_layers:
        if FT_SD is None:   # fresh run: the frozen original is the teacher; resuming from an FT ckpt keeps the original taken at load
            FT_SD = {k: v.detach().to("cpu", copy=True) for k, v in llm.state_dict().items()
                     if k.startswith("model.layers.") and int(k.split(".")[2]) < a.ft_layers}
        FTP = [p_ for li in range(a.ft_layers) for p_ in llm.model.layers[li].parameters()]
        for p_ in FTP: p_.requires_grad_(True)
        MASTER = [p_.detach().float().clone().requires_grad_(True) for p_ in FTP]
        def _acc(m_):
            def hook(p_):
                if m_.grad is None: m_.grad = p_.grad.float()
                else: m_.grad.add_(p_.grad.float())
                p_.grad = None
            return hook
        for p_, m_ in zip(FTP, MASTER): p_.register_post_accumulate_grad_hook(_acc(m_))
        groups.append({"params": MASTER, "lr": a.ft_lr})
    if a.opt8bit:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(groups, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(groups, weight_decay=0.0)
    steps = int(len(data) * a.epochs); n_upd = max(1, steps // a.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(2, n_upd // 15)) * max(0.1, 1 - s / n_upd))
    log(f"[sp] train {tag}: {steps} traces, {a.accum} per update ({n_upd} updates), pooler lr {a.lr}"
        + (f", FT layers 0-{a.ft_layers - 1} lr {a.ft_lr} ({sum(p_.numel() for p_ in FTP) / 1e6:.0f}M)" if a.ft_layers else ""))

    def save():
        if not a.out: return
        ck = {"pooler": {k: v.detach().cpu() for k, v in pooler.state_dict().items()}, "args": vars(a)}
        if a.ft_layers: ck["ft"] = {k: v.detach().cpu() for k, v in llm.state_dict().items() if k in FT_SD}
        torch.save(ck, a.out)

    t0 = time.time(); K = A_ = N = 0
    if FTP:   # pinned frozen copies in the same order as the tuned params, for the fast swap
        names = [n_ for li in range(a.ft_layers) for n_, _ in llm.model.layers[li].named_parameters(prefix=f"model.layers.{li}")]
        SWAP["pairs"] = [(p_, FT_SD[n_].pin_memory() if dev == "cuda" else FT_SD[n_], m_) for p_, n_, m_ in zip(FTP, names, MASTER)]
    refs = {}; WINS = [int(x) for x in a.sn_win.split(",")]
    WI = {"i": 0, "from": 0, "kl": []}; STOP = {"now": False}
    for st in range(steps):
        ids, P = data[st % len(data)]
        if FTP and st % a.accum == 0:   # the teacher pass for the next group of traces in ONE weight swap
            refs = {}
            with _Frozen():
                for j in range(st, min(st + a.accum, steps)):
                    ids_j, P_j = data[j % len(data)]
                    T_j = min(len(ids_j), P_j + a.max_len)
                    lo_j = a.sn_t0 if a.spnext else (a.C if a.par else a.rw + a.C)
                    if T_j - P_j - lo_j >= (a.sn_batch // 4 if a.spnext else (a.rw + a.C if a.par else a.C)):
                        refs[j] = teacher_hidden(ids_j.to(dev), P_j + lo_j, T_j)
        if a.spnext:
            if a.sn_win_plateau: SNW["w"] = WINS[WI["i"]]
            else: SNW["w"] = WINS[min(st // a.sn_win_every, len(WINS) - 1)]
            k, g, n = run_trace_spnext(ids, P, ref_h=refs.pop(st, None))
        elif a.par: k, g, n = run_trace_par(ids, P, ref_h=refs.pop(st, None))
        else: k, g, n = run_trace(ids, P, True, ref_h=refs.pop(st, None))
        K += k; A_ += g; N += n
        if (st + 1) % a.accum == 0 or st + 1 == steps:
            torch.nn.utils.clip_grad_norm_(list(pooler.parameters()) + MASTER, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True); sched.step()
            if FTP:
                with torch.no_grad():
                    for p_, m_ in zip(FTP, MASTER): p_.copy_(m_.to(p_.dtype))
        if (st + 1) % 25 == 0 or st + 1 == steps:
            x = f" | w{SNW['w']} window-0 KL {SNW['k0'] / max(SNW['n0'], 1):.4f}" if a.spnext and len(WINS) > 1 else ""
            log(f"[sp] train {tag} trace {st + 1}/{steps}: KL {K / max(N, 1):.4f} top1 {100 * A_ / max(N, 1):.2f}%{x} ({time.time() - t0:.0f}s)")
            if a.spnext and a.sn_win_plateau and N:   # curriculum by plateau: 100-trace means at this w, last vs the one before
                WI["kl"].append(K / N); n_at = st + 1 - WI["from"]; hist = WI["kl"]
                b_ = a.sn_win_block
                flat = len(hist) >= 2 * b_ and sum(hist[-b_:]) / b_ > (1 - a.sn_win_plateau) * sum(hist[-2 * b_:-b_]) / b_
                if flat or n_at >= a.sn_win_max:
                    log(f"[sp] curriculum: w{WINS[WI['i']]} {'plateau' if flat else 'cap'} after {n_at} traces "
                        f"(last {25 * len(hist[-a.sn_win_block:])} KL {sum(hist[-a.sn_win_block:]) / len(hist[-a.sn_win_block:]):.4f})")
                    if WI["i"] + 1 >= len(WINS): STOP["now"] = True
                    else: WI["i"] += 1; WI["from"] = st + 1; WI["kl"] = []
            K = A_ = N = 0; SNW["k0"] = 0.0; SNW["n0"] = 0
        if (st + 1) % 200 == 0: save()
        if STOP["now"]: break
    save(); log(f"[sp] trained {tag} -> {a.out}")
