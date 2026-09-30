#!/usr/bin/env python3
"""Teach the pooler to carry a conversation: multi-turn self-distillation.

At turn k the student sees the turn as the user asked it ("when was that film released?") with every earlier token
of the conversation - questions, thinking, search results, replies - reaching it only through the pooler (the current
question pinned, the rest mass-evicted to maxd: `pool_eval --mt-mode stream`, the phone's bounded-KV protocol). The
teacher is the same model on the turn's SELF-CONTAINED version ("when was Spirited Away released?"), no history: what
the model does well, single-turn. The loss is the KL between them on the teacher's own trajectory, one block at a time,
so the student learns to resolve the reference from the compressed history - and, on a topic switch (the
self-contained version is the question itself), to ignore it. The full-history protocol is not the teacher: measured
2026-09-29 (mt2), with earlier turns in the prompt the model answers the FIRST question of the conversation on a topic
switch (0 of 8).

Data: rows of `pool_eval --multiturn ... --mt-mode none --mt-standalone 1 --mt-save-tokens 1` (each turn's
self-contained version, run alone). Turn 0 of a dialogue has no history and is skipped.

  SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 python3 memfit.py --ckpt /root/reeval_g14m_pooler.safetensors \
      --data /root/work/mtt_full.jsonl --out /root/pooler_mem.safetensors
"""
import argparse, json, math, os, random, re, sys, time, collections

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="the pooler to start from (bare or pooler.-prefixed keys)")
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--log", default="/root/memfit.log")
ap.add_argument("--steps", type=int, default=800); ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--lr-lora", type=float, default=0.0, help=">0 also trains the harness's LoRA (layers as SP_RANK set up)")
ap.add_argument("--warmup", type=int, default=30); ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--rw", type=int, default=768); ap.add_argument("--maxd", type=int, default=384); ap.add_argument("--chunk", type=int, default=128)
ap.add_argument("--maxtok", type=int, default=2600); ap.add_argument("--val", type=int, default=24); ap.add_argument("--val-every", type=int, default=50)
ap.add_argument("--reply-bias", type=float, default=0.5, help="share of steps that take the block holding </think> (where the reply starts)")
ap.add_argument("--student", default="stream", choices=["stream", "mix", "full"], help="full: every earlier exchange (question + reply) verbatim in the pinned prompt, the pooler only for this turn's own evicted tokens; stream: every earlier turn through the pooler; mix: the last exchange (question + reply) verbatim in the pinned prompt, older turns through the pooler - names travel verbatim, the pooler carries the rest")
ap.add_argument("--objective", default="kl", choices=["kl", "ce"], help="kl: match the model's own single-turn behaviour on the self-contained version (the default); ce: rejection-sampled self-training - the data are the model's OWN successful runs (rows with correct=true, e.g. pool_eval --mt-mode full --mt-save-tokens 1), their prompt as run and their tokens as the target, so the history-in-view behaviour it already gets right is reinforced and nothing else is asked of it")
ap.add_argument("--ce-native", type=int, default=0, help="ce: rebuild each successful turn's prompt as the NATIVE multi-turn chat (earlier questions and replies of its dialogue as separate turns, then the question) instead of the prompt it was run with - successes collected under another protocol (quote) trained into the native one")
ap.add_argument("--anchor", type=int, default=0, help="single-turn anchors: a solo trace with NO history, the student's prompt the teacher's own - the KL then only holds the LoRA to the model as it was (mem5 without them lost 9 of 41 first turns and doubled the searches). -1: as many as the real examples")
ap.add_argument("--synth-switch", type=int, default=-1, help="topic switches assembled from the traces: one dialogue's history, then a self-contained turn of ANOTHER dialogue (its own solo trajectory is the teacher's). -1: as many as the real examples; 0: none")
ap.add_argument("--gen", type=int, default=1500); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--selftest", type=int, default=0)
A = ap.parse_args()
os.environ.setdefault("SP_HOTPOT2", "0"); os.environ.setdefault("SP_RANK", "16"); os.environ.setdefault("SP_NOSYS", "1"); os.environ.setdefault("SP_EPISODIC", "1")
import numpy as np                                        # noqa: E402
import torch                                              # noqa: E402
import torch.nn.functional as F                           # noqa: E402
from safetensors.torch import load_file, save_file        # noqa: E402

random.seed(A.seed); torch.manual_seed(A.seed)
F_ = "/root/work/grpo_e2e_torch.py"
src = open(F_).read().split("\n")
cut = next(i for i, l in enumerate(src) if l.startswith("if MULTI > 1:"))
sys.argv = ["grpo_e2e_torch.py", "0", "6", "0", str(A.gen)]
sys.path.insert(0, "/root/work")
ns = {"__name__": "memfit", "__file__": F_}
exec(compile("\n".join(src[:cut]), F_, "exec"), ns)
model, tok, pooler = ns["model"], ns["tok"], ns["pooler"]
emb, sp, DEV = ns["emb"], ns["sp"], ns["DEV"]
CLM = model.base_model.model if hasattr(model, "base_model") else model
BODY, HEAD = CLM.model, CLM.lm_head
model.config.use_cache = False
lora = []
for n, p in model.named_parameters():
    train = A.lr_lora > 0 and "lora" in n
    p.requires_grad_(train)
    if train: lora.append(p)
sd = load_file(A.ckpt)
pl = {k[len("pooler."):]: v for k, v in sd.items() if k.startswith("pooler.")}
if not pl and not any(k.startswith(("model.", "lm_head", "base_model")) for k in sd):
    pl = dict(sd)
print(f"[pooler] restored {pooler.load_sd(pl) if pl else 0} tensors from {A.ckpt}", flush=True)
_md = {k: v for k, v in sd.items() if not k.startswith("pooler.") and "lora" in k}
if _md:   # a checkpoint written by an earlier memfit run carries its LoRA: continue from it
    _r = model.load_state_dict(_md, strict=False)
    print(f"[lora] restored {len(_md)} LoRA tensors from {A.ckpt} ({len(_r.unexpected_keys)} unexpected)", flush=True)
POOL_REF = {k: v.detach().clone() for k, v in pooler.A.items()}
pooler.make_trainable(); POOL_TRAIN = dict(pooler.A)
pp = list(pooler.parameters())
if DEV == "cuda":
    import torch.utils.checkpoint as _ckp
    for _lyr in BODY.layers:
        def _wrapped(*a, _f=_lyr.forward, **kw):
            return _ckp.checkpoint(_f, *a, use_reentrant=False, **kw)
        _lyr.forward = _wrapped
print(f"[memfit] pooler {sum(p.numel() for p in pp)/1e6:.2f}M parameters" + (f", LoRA {sum(p.numel() for p in lora)/1e6:.2f}M" if lora else ""), flush=True)
EOS_IDS = tok.encode("<｜end▁of▁sentence｜>", add_special_tokens=False)
BOS = tok.bos_token_id


import contextlib
_null = contextlib.nullcontext


def set_teacher(t):
    pooler.A = POOL_REF if t else POOL_TRAIN


def single_q_ids(q):
    return tok.encode(tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False) + "<think>\n")


def own_mask(gen):
    """1 for the model's own tokens, 0 inside <information> blocks the environment injected."""
    text_ids = gen; own = [1.0] * len(gen)
    s = tok.decode(gen)
    if "<information>" not in s:
        return own
    # walk the ids, tracking whether we are inside an information block
    inside = False; buf = ""
    for i, t in enumerate(gen):
        buf += tok.decode([t]); buf = buf[-40:]
        if not inside and "<information>" in buf:
            inside = True; buf = ""
        own[i] = 0.0 if inside else 1.0
        if inside and "</information>" in buf:
            inside = False; buf = ""
    return own


# ---- data: (history carry for the student, teacher prompt ids, student prompt ids, gen) per turn >= 1 ----------
rows = collections.defaultdict(list)
for l in open(A.data):
    try: r = json.loads(l)
    except Exception: continue
    if "q_ids" in r and "gen" in r: rows[r["dialog"]].append(r)
EX = []
for did, rs in rows.items():
    rs.sort(key=lambda r: r["turn"]); carry = []; older = []; prev = None; msgs = []
    for r in rs:
        sq = single_q_ids(r["q"])                                   # the turn as asked
        tq = r["q_ids"]                                               # the self-contained version the trajectory was run on
        if r["turn"] > 0 and len(r["gen"]) >= 8:
            if A.student == "full":
                sq_full = tok.encode(tok.apply_chat_template(msgs + [{"role": "user", "content": r["q"]}], add_generation_prompt=True, tokenize=False) + "<think>\n")
                EX.append({"dialog": did, "turn": r["turn"], "carry": [], "tq": tq, "sq": sq_full, "gen": r["gen"][:A.maxtok]})
            elif A.student == "mix" and prev is not None:
                pq, preply = prev
                sq_mix = tok.encode(tok.apply_chat_template([{"role": "user", "content": pq}, {"role": "assistant", "content": preply or "(no reply)"},
                                                             {"role": "user", "content": r["q"]}], add_generation_prompt=True, tokenize=False) + "<think>\n")
                EX.append({"dialog": did, "turn": r["turn"], "carry": list(older), "tq": tq, "sq": sq_mix, "gen": r["gen"][:A.maxtok]})
            else:
                EX.append({"dialog": did, "turn": r["turn"], "carry": list(carry), "tq": tq, "sq": sq, "gen": r["gen"][:A.maxtok]})
        this = (sq[1:] if sq and sq[0] == BOS else sq) + r["gen"] + EOS_IDS
        # mix: the pinned exchange is the previous one; the pooler gets everything before it plus that turn's thinking
        older = carry + r["gen"] + EOS_IDS if A.student == "mix" else older
        carry = carry + this
        txt = tok.decode(r["gen"]); prev = (r["q"], txt.split("</think>")[-1].replace("<｜end▁of▁sentence｜>", "").strip() if "</think>" in txt else "")
        msgs = msgs + [{"role": "user", "content": prev[0]}, {"role": "assistant", "content": prev[1] or "(no reply)"}]
# topic switches: the history of dialogue A (all its turns), then a turn of dialogue B that is self-contained (its
# question is its own standalone version), which the model must answer as if nothing came before
HIST = {}
for did, rs in rows.items():
    rs = sorted(rs, key=lambda r: r["turn"]); c = []; m = []
    for r in rs:
        sq = single_q_ids(r["q"]); c = c + (sq[1:] if sq and sq[0] == BOS else sq) + r["gen"] + EOS_IDS
        txt = tok.decode(r["gen"]); rep = txt.split("</think>")[-1].replace("<｜end▁of▁sentence｜>", "").strip() if "</think>" in txt else ""
        m = m + [{"role": "user", "content": r["q"]}, {"role": "assistant", "content": rep or "(no reply)"}]
    HIST[did] = (c, m)
SOLO = [(did, r) for did, rs in rows.items() for r in rs if r.get("standalone", r["q"]) == r["q"] and len(r["gen"]) >= 8]
n_syn = len(EX) if A.synth_switch < 0 else A.synth_switch
rng = random.Random(3); hd = list(HIST)
for _ in range(n_syn if SOLO and len(hd) > 1 else 0):
    bid, r = rng.choice(SOLO); aid = rng.choice(hd)
    if aid == bid: continue
    c, m = HIST[aid]
    if A.student == "full":
        sq = tok.encode(tok.apply_chat_template(m + [{"role": "user", "content": r["q"]}], add_generation_prompt=True, tokenize=False) + "<think>\n"); carry = []
    elif A.student == "mix":
        sq = tok.encode(tok.apply_chat_template(m[-2:] + [{"role": "user", "content": r["q"]}], add_generation_prompt=True, tokenize=False) + "<think>\n"); carry = c
    else:
        sq = single_q_ids(r["q"]); carry = c
    EX.append({"dialog": f"{bid}~{aid}", "turn": -1, "carry": carry, "tq": r["q_ids"], "sq": sq, "gen": r["gen"][:A.maxtok]})
print(f"[data] + {sum(1 for e in EX if e['turn'] == -1)} topic switches assembled from other dialogues", flush=True)
n_real = sum(1 for e in EX if e["turn"] != -1)
n_anc = n_real if A.anchor < 0 else A.anchor
ALLSOLO = [(did, r) for did, rs in rows.items() for r in rs if len(r["gen"]) >= 8]
for k in range(n_anc if ALLSOLO else 0):
    did, r = ALLSOLO[k % len(ALLSOLO)] if k < len(ALLSOLO) else rng.choice(ALLSOLO)
    EX.append({"dialog": f"{did}~anchor", "turn": -2, "carry": [], "tq": r["q_ids"], "sq": r["q_ids"], "gen": r["gen"][:A.maxtok]})
print(f"[data] + {sum(1 for e in EX if e['turn'] == -2)} single-turn anchors (no history, the teacher's own prompt)", flush=True)
if A.objective == "ce":
    EX = []
    for did, rs in rows.items():
        rs = sorted(rs, key=lambda r: r["turn"]); hist = []
        for r in rs:
            if r.get("correct") and len(r["gen"]) >= 8:
                if A.ce_native:
                    sq = tok.encode(tok.apply_chat_template(hist + [{"role": "user", "content": r["q"]}], add_generation_prompt=True, tokenize=False) + "<think>\n")
                else:
                    sq = r["q_ids"]
                EX.append({"dialog": did, "turn": r["turn"], "carry": [], "tq": sq, "sq": sq, "gen": r["gen"][:A.maxtok]})
            hist = hist + [{"role": "user", "content": r["q"]}, {"role": "assistant", "content": (r.get("reply") or "(no reply)")}]
    k0 = sum(1 for e in EX if e["turn"] == 0); k1 = len(EX) - k0
    print(f"[data] ce: {len(EX)} successful turns of the model's own ({k0} first turns, no history; {k1} later turns, history in the prompt)", flush=True)
random.shuffle(EX)
dids = sorted({e["dialog"].split("~")[0] for e in EX}); random.Random(1).shuffle(dids); vd = set(dids[:max(1, len(dids) // 10)])
VAL = [e for e in EX if e["dialog"].split("~")[0] in vd][:A.val]; TRAIN = [e for e in EX if e["dialog"].split("~")[0] not in vd and not any(x in vd for x in e["dialog"].split("~"))]
print(f"[data] {len(EX)} turns with history from {len(rows)} dialogues; train {len(TRAIN)}, validation {len(VAL)}", flush=True)
for e in EX: e["own"] = own_mask(e["gen"])


@torch.no_grad()
def schedule(gen, seed):
    """(c0, c1, kept) per block for one side: kept starts from seed (the student's history, [] for the teacher),
    evicted by the current pooler's mass down to maxd, then extended by what leaves the raw window."""
    kept, absorbed, segs, c = list(seed), 0, [], 0
    while c < len(gen):
        c0, c1 = c, min(c + A.chunk, len(gen)); nd = c0 - min(c0, A.rw)
        if nd > absorbed:
            kept.extend(gen[absorbed:nd]); absorbed = nd
        if len(kept) > A.maxd:
            _, mass = pooler.forward_with_mass(emb(kept).to(torch.float32))
            mm = mass[0].float().cpu().numpy(); kept = [kept[i] for i in np.sort(np.argsort(mm)[-A.maxd:])]
        segs.append((c0, c1, list(kept))); c = c1
    return segs


def logits_for(q_ids, gen, seg):
    c0, c1, kept = seg; R = min(c0, A.rw)
    spv = sp(kept) if kept else torch.zeros((1, 0, ns["H"]), device=DEV, dtype=ns["MDTYPE"])
    parts = [emb(q_ids), spv] + ([emb(gen[c0 - R:c0])] if R > 0 else []) + [emb(gen[c0:c1])]
    block = torch.cat(parts, dim=1); L, cur = block.shape[1], c1 - c0
    h = BODY(inputs_embeds=block, use_cache=False).last_hidden_state[:, L - cur - 1:L - 1, :]
    return HEAD(h).float()[0]


def kl_of(e, seg_i=None):
    gen, own = e["gen"], e["own"]
    set_teacher(True); tsegs = schedule(gen, [])
    set_teacher(False); ssegs = schedule(gen, e["carry"])
    if seg_i is not None:
        j = seg_i % len(tsegs)
    else:
        txt_end = [k for k, (c0, c1, _) in enumerate(tsegs) if "</think>" in tok.decode(gen[c0:c1])]
        j = txt_end[0] if (txt_end and random.random() < A.reply_bias) else random.randrange(len(tsegs))
    c0, c1, _ = tsegs[j]
    keep = [i for i in range(c1 - c0) if own[c0 + i] > 0]
    if not keep: return None
    idx = torch.tensor(keep, device=DEV)
    if A.objective == "ce":
        set_teacher(False)
        ls = logits_for(e["sq"], gen, ssegs[j])[idx]
        tgt = torch.tensor([gen[c0 + i] for i in keep], device=DEV)
        return F.cross_entropy(ls, tgt)
    set_teacher(True)
    with torch.no_grad(), (model.disable_adapter() if lora else _null()):   # the teacher is the model as it stands, LoRA off
        lt = logits_for(e["tq"], gen, tsegs[j])[idx]
    set_teacher(False)
    ls = logits_for(e["sq"], gen, ssegs[j])[idx]
    p = F.softmax(lt / A.temp, -1)
    return (p * (torch.log(p.clamp_min(1e-9)) - F.log_softmax(ls / A.temp, -1))).sum(-1).mean() * A.temp ** 2


@torch.no_grad()
def validate():
    tot, n = 0.0, 0
    for j, e in enumerate(VAL):
        v = kl_of(e, seg_i=None if False else j)
        if v is not None: tot += v.item(); n += 1
    return tot / max(n, 1)


groups = ([{"params": pp, "lr": A.lr}] if A.lr > 0 else []) + ([{"params": lora, "lr": A.lr_lora}] if lora else [])   # --lr 0: the pooler stays as it is
opt = torch.optim.Adam(groups, betas=(0.9, 0.95))
BEST = {"v": float("inf"), "step": 0, "p": None, "l": None}
def keep_if_best(v, i):
    if v < BEST["v"]:
        BEST.update(v=v, step=i, p={k: t.detach().cpu().clone() for k, t in POOL_TRAIN.items()},
                    l={n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad})
        return True
    return False
def lr_scale(i):
    if i < A.warmup: return (i + 1) / A.warmup
    t = (i - A.warmup) / max(1, A.steps - A.warmup); return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
log = open(A.log, "a")
v0 = validate(); keep_if_best(v0, 0)
print(f"[memfit] history-through-pooler KL before training {v0:.4f} over {len(VAL)} unseen turns", flush=True); log.write(f"val 0 kl={v0:.4f}\n"); log.flush()
t0 = time.time(); N = A.selftest if A.selftest else A.steps
for i in range(N):
    for g, base in zip(opt.param_groups, ([A.lr] if A.lr > 0 else []) + ([A.lr_lora] if lora else [])): g["lr"] = base * lr_scale(i)
    opt.zero_grad(set_to_none=True); tot, n = 0.0, 0
    for _ in range(A.accum):
        l = kl_of(TRAIN[random.randrange(len(TRAIN))])
        if l is None: continue
        (l / A.accum).backward(); tot += l.item() / A.accum; n += 1
    if n: torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0); opt.step()
    print(f"step {i+1} kl={tot:.4f} {(time.time()-t0)/(i+1):.1f}s/step", flush=True); log.write(f"step {i+1} kl={tot:.4f}\n")
    if (i + 1) % A.val_every == 0 or i + 1 == N:
        v = validate(); better = keep_if_best(v, i + 1)
        print(f"val {i+1} kl={v:.4f} {'best' if better else 'worse than step %d (%.4f)' % (BEST['step'], BEST['v'])}", flush=True); log.write(f"val {i+1} kl={v:.4f}\n"); log.flush()
if lora and BEST["l"]:
    # one file pool_eval reads as it is: the pooler under pooler.-prefixed keys, the LoRA under the model's own names
    out = {"pooler." + k: v.float().contiguous() for k, v in BEST["p"].items()}
    out.update({n: v.contiguous() for n, v in BEST["l"].items()})
else:
    out = {k: v.float().contiguous() for k, v in BEST["p"].items()}
save_file(out, A.out)
print(f"MEMFIT_DONE best step {BEST['step']} val kl {BEST['v']:.4f} (from {v0:.4f}) -> {A.out}", flush=True)
