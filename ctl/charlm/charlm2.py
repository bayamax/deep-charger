#!/usr/bin/env python3
"""The character model, on a GPU: 4096-character context (about 900-1000 GPT-2 tokens of English), grown from shallow.

Same design as charlm.py - a 98-symbol alphabet (charlm_prep.py) with a learned vector per character, next-character
prediction, depth added on plateau as identity blocks (both output projections zero, so the function is unchanged at the
moment of growth) - with what the GPU allows: bf16, flash attention through SDPA, gradient accumulation to a fixed number
of characters per update, and a micro-batch that halves itself on out-of-memory.

Data: train_wiki.bin and train_fw.bin, a window drawn from Wikipedia with probability --p-wiki. Growth is decided on the
mean of the two validation sets; the test sets decide nothing. At the end: test bits per character on both beside GPT-2
small's on the same raw documents, samples, then GPTQ 4-bit (the app's grid: affine, group 64) of every linear layer
and its test bpc. Checkpoints and the log go to the hub as they are made (--hub).

  python3 charlm2.py --data /root/charlm/data --out /root/charlm/run2 --hours 14 --d 512 --heads 8 --layers 4,8,12,16
"""
import argparse, json, math, os, random, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--d", type=int, default=512); ap.add_argument("--heads", type=int, default=8); ap.add_argument("--ff", type=int, default=0)
ap.add_argument("--layers", default="4,8,12,16"); ap.add_argument("--seq", type=int, default=4096)
ap.add_argument("--chars-per-step", type=int, default=131072, help="characters per update (micro-batches accumulate to it)")
ap.add_argument("--micro", type=int, default=16, help="windows per micro-batch to start with; halves on out-of-memory")
ap.add_argument("--lr", type=float, default=1.5e-3); ap.add_argument("--warmup", type=int, default=300); ap.add_argument("--wd", type=float, default=0.1)
ap.add_argument("--p-chat", type=float, default=0.0, help="a window from train_chat.bin (chat as plain text) with this probability, when the file is there")
ap.add_argument("--p-wiki", type=float, default=0.3); ap.add_argument("--hours", type=float, default=14.0)
ap.add_argument("--eval-every", type=float, default=20.0, help="minutes"); ap.add_argument("--val-chars", type=int, default=2_000_000)
ap.add_argument("--plateau", type=float, default=0.006, help="grow when two validations in a row gain less than this (bpc)")
ap.add_argument("--stage-min", type=float, default=0.10, help="share of the budget a depth gets before it may grow")
ap.add_argument("--hub", default="", help="repo/prefix on the hub for checkpoints and the log")
ap.add_argument("--resume", default=""); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bench", type=int, default=0)
ap.add_argument("--compile", type=int, default=0)
ap.add_argument("--grow-now", type=int, default=0, help="1: after --resume, grow to the next depth at once")
ap.add_argument("--chat-val", default="", help="a held-out chat text (charlm_prep.py --chat-val: SmolTalk's test split, used by no training), scored at every validation")
ap.add_argument("--chat-val-chars", type=int, default=1500000); ap.add_argument("--chat-prompt", default="User: What is photosynthesis? Explain it simply.\nAssistant:")
ap.add_argument("--cool-plateau", type=float, default=0.0, help="at the last depth, when the chat score has gained less than this over --cool-evals "
                "validations, the learning rate falls to a tenth over --cool-hours and the run ends there (0: off; --hours stays the cap)")
ap.add_argument("--cool-evals", type=int, default=4); ap.add_argument("--cool-hours", type=float, default=3.0)
ap.add_argument("--cool-target", type=float, default=0.0, help="the cool-down starts once the chat score reaches this (bpc; 0: no target)")
ap.add_argument("--plateau-cools", type=int, default=1, help="0: a chat plateau before the target is only reported, the run goes on")
ap.add_argument("--ctl", default="", help="a JSON file read at every validation; its keys (cool_target, cool_plateau, plateau_cools, "
                "cool_evals, cool_hours) replace those settings without a restart")
ap.add_argument("--qknorm", type=int, default=1, help="1: LayerNorm on each head's queries and keys (keeps attention logits bounded over the 4096-character context)")
ap.add_argument("--diverge", type=float, default=0.10, help="a validation this much worse than the best: back to the best weights, the learning rate halved")
A = ap.parse_args()
torch.manual_seed(A.seed); np.random.seed(A.seed); random.seed(A.seed)
DEV = "cuda" if torch.cuda.is_available() else "cpu"; FF = A.ff or 4 * A.d
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
os.makedirs(A.out, exist_ok=True); LOGF = os.path.join(A.out, "train.log"); LOG = open(LOGF, "a")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

CHARS = "\n" + "".join(chr(i) for i in range(32, 127)); UNK = len(CHARS); EOD = UNK + 1; V = EOD + 1
def decode(ids): return "".join(CHARS[i] if i < UNK else ("�" if i == UNK else "\n<EOD>\n") for i in ids)
LUT = np.full(256, UNK, np.uint8)
for i, c in enumerate(CHARS): LUT[ord(c)] = i
def encode_ascii(s): return LUT[np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)]

mm = lambda n: np.memmap(os.path.join(A.data, n), dtype=np.uint8, mode="r")
VAC = np.memmap(A.chat_val, dtype=np.uint8, mode="r") if A.chat_val and os.path.exists(A.chat_val) else None
TRC = mm("train_chat.bin") if os.path.exists(os.path.join(A.data, "train_chat.bin")) else None
TRW, TRF = mm("train_wiki.bin"), mm("train_fw.bin"); VAW, VAF = mm("val_wiki.bin"), mm("val_fw.bin"); TEW, TEF = mm("test_wiki.bin"), mm("test_fw.bin")
out(f"[charlm2] train: wiki {len(TRW) / 1e6:.0f}M + fineweb {len(TRF) / 1e6:.0f}M chars; validation {len(VAW) / 1e6:.2f}M + {len(VAF) / 1e6:.1f}M; "
    f"test {len(TEW) / 1e6:.2f}M + {len(TEF) / 1e6:.1f}M; {DEV} {torch.cuda.get_device_name(0) if DEV == 'cuda' else ''}")


class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1 = nn.LayerNorm(A.d); s.qkv = nn.Linear(A.d, 3 * A.d); s.proj = nn.Linear(A.d, A.d)
        s.ln2 = nn.LayerNorm(A.d); s.fc = nn.Linear(A.d, FF); s.fc2 = nn.Linear(FF, A.d)
        s.qk = bool(A.qknorm)
        if s.qk: s.qn = nn.LayerNorm(A.d // A.heads); s.kn = nn.LayerNorm(A.d // A.heads)

    def forward(s, x):
        B, T, D = x.shape; q, k, v = s.qkv(s.ln1(x)).split(D, -1)
        sh = lambda z: z.view(B, T, A.heads, D // A.heads)
        q, k, v = sh(q), sh(k), sh(v)
        if s.qk: q, k = s.qn(q), s.kn(k)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + s.proj(a)
        return x + s.fc2(F.gelu(s.fc(s.ln2(x)), approximate="tanh"))

    def as_identity(s):
        for m in (s.proj, s.fc2): nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        return s


class CharLM(nn.Module):
    def __init__(s, layers):
        super().__init__()
        s.emb = nn.Embedding(V, A.d); s.pos = nn.Embedding(A.seq, A.d)
        s.blocks = nn.ModuleList([Block() for _ in range(layers)]); s.ln_f = nn.LayerNorm(A.d); s.head = nn.Linear(A.d, V)
        s.apply(s._init)
        for b in s.blocks:   # GPT-2's residual scaling
            for m in (b.proj, b.fc2): nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)

    def forward(s, ids):
        x = s.emb(ids) + s.pos(torch.arange(ids.shape[1], device=ids.device))
        for b in s.blocks: x = b(x)
        return s.head(s.ln_f(x))

    def grow(s, to):
        """identity blocks spread evenly between the existing ones - every existing block kept (8 -> 12 used to drop the
        last two: an identity after each block until the count was reached, the rest cut off)"""
        old = list(s.blocks); n_add = to - len(old); new = []
        for i, b in enumerate(old):
            new.append(b)
            for _ in range((i + 1) * n_add // len(old) - i * n_add // len(old)): new.append(Block().as_identity().to(s.head.weight.device))
        assert len(new) == to and all(any(b is o for b in new) for o in old)
        s.blocks = nn.ModuleList(new)

    def n_params(s): return sum(p.numel() for p in s.parameters())


def windows(n):
    rows = []
    for _ in range(n):
        src = TRC if TRC is not None and random.random() < A.p_chat else (TRW if random.random() < A.p_wiki else TRF); j = random.randrange(0, len(src) - A.seq - 1); rows.append(src[j:j + A.seq + 1])
    a = torch.from_numpy(np.stack(rows).astype(np.int64)).pin_memory() if DEV == "cuda" else torch.from_numpy(np.stack(rows).astype(np.int64))
    a = a.to(DEV, non_blocking=True); return a[:, :-1], a[:, 1:]


@torch.no_grad()
def bpc(model, data, max_chars=None, stride=None, eb=8):
    """bits per character: windows of --seq, each after the first scoring only its last `stride` characters"""
    stride = stride or A.seq // 2; model.eval(); T = A.seq; end = len(data) if max_chars is None else min(len(data), max_chars)
    starts = list(range(0, max(1, end - T - 1), stride)); nll = 0.0; n = 0
    for i in range(0, len(starts), eb):
        s_ = starts[i:i + eb]; a = torch.from_numpy(np.stack([data[j:j + T + 1] for j in s_]).astype(np.int64)).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            lg = model(a[:, :-1])
        for r, j in enumerate(s_):
            k = 0 if j == 0 else T - stride
            nll += float(F.cross_entropy(lg[r, k:].float(), a[r, k + 1:], reduction="sum")); n += T - k
    model.train(); return nll / n / math.log(2)


@torch.no_grad()
def sample(model, prompt, n=400, temp=0.8):
    model.eval(); ids = torch.tensor([encode_ascii(prompt).astype(np.int64)], device=DEV)
    for _ in range(n):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            lg = model(ids[:, -A.seq:])[0, -1].float() / temp
        ids = torch.cat([ids, torch.multinomial(F.softmax(lg, -1), 1)[None]], 1)
    model.train(); return decode(ids[0].tolist())


def hub_put(local, remote):
    if not A.hub: return
    try:
        from huggingface_hub import HfApi
        repo, pre = A.hub.split(":", 1) if ":" in A.hub else (A.hub, "charlm")
        HfApi().upload_file(path_or_fileobj=local, path_in_repo=f"{pre}/{remote}", repo_id=repo)
    except Exception as e:
        out(f"[hub] upload of {remote} failed: {str(e)[:120]}")


DEPTHS = [int(v) for v in A.layers.split(",")]
if A.bench:
    for L in DEPTHS:
        m = CharLM(L).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=A.lr, fused=DEV == "cuda"); mb = A.micro
        while True:
            try:
                x, y = windows(mb)
                for _ in range(2):
                    with torch.autocast("cuda", dtype=torch.bfloat16): l = F.cross_entropy(m(x).flatten(0, 1).float(), y.flatten())
                    opt.zero_grad(); l.backward(); opt.step()
                torch.cuda.synchronize(); t = time.time()
                for _ in range(5):
                    with torch.autocast("cuda", dtype=torch.bfloat16): l = F.cross_entropy(m(x).flatten(0, 1).float(), y.flatten())
                    opt.zero_grad(); l.backward(); opt.step()
                torch.cuda.synchronize(); dt = (time.time() - t) / 5; break
            except torch.cuda.OutOfMemoryError:
                del x, y; torch.cuda.empty_cache(); mb //= 2
        out(f"[bench] {L} layers d {A.d}: {m.n_params() / 1e6:.1f}M params, micro {mb} x {A.seq}: {mb * A.seq / dt / 1e3:.0f}k chars/s, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        del m, opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    sys.exit(0)

model = CharLM(DEPTHS[0]).to(DEV); stage = 0; step = 0; hist = []; best = {"bpc": 9.0, "step": 0}; T0 = time.time(); budget = A.hours * 3600
CHATH = []; COOL_T0 = None   # chat-score history (step, minutes, bpc); when the cool-down started (seconds of the run)
if A.resume and os.path.exists(A.resume):
    st = torch.load(A.resume, map_location=DEV, weights_only=False); model = CharLM(st["layers"]).to(DEV); model.load_state_dict(st["model"])
    stage = DEPTHS.index(st["layers"]); step = st["step"]; hist = st.get("hist", []); best = st.get("best", best); T0 = time.time() - st.get("elapsed", 0)
    best.setdefault("layers", st["layers"]); CHATH = st.get("chat_hist", []); COOL_T0 = st.get("cool_t0")
    out(f"[charlm2] resumed at step {step}, {st['layers']} layers, {st.get('elapsed', 0) / 3600:.1f} h in")
if A.grow_now and A.resume and stage + 1 < len(DEPTHS):
    stage += 1; model.grow(DEPTHS[stage]); out(f"[grow] -> {len(model.blocks)} layers ({model.n_params() / 1e6:.1f}M params) at resume, step {step} (--grow-now)")
make_opt = lambda: torch.optim.AdamW(model.parameters(), lr=A.lr, betas=(0.9, 0.95), weight_decay=A.wd, fused=DEV == "cuda")
opt = make_opt(); MICRO = A.micro; OPT_BACK = False
if A.resume and os.path.exists(A.resume) and "opt" in st and not A.grow_now:   # the AdamW moments back too: a fresh optimizer cost every restart a bump
    try: opt.load_state_dict(st["opt"]); OPT_BACK = True; out("[charlm2] optimizer state restored (no warm-up)")
    except Exception as e: out(f"[charlm2] optimizer state not restored: {str(e)[:100]}")
out(f"[charlm2] {len(model.blocks)} layers, d {A.d}, ff {FF}, heads {A.heads}, context {A.seq}: {model.n_params() / 1e6:.2f}M params; "
    f"{A.chars_per_step} chars per update; budget {A.hours} h; depths {DEPTHS}")


def validate():
    vw, vf = bpc(model, VAW, A.val_chars), bpc(model, VAF, A.val_chars); return vw, vf, (vw + vf) / 2


def apply_ctl():
    """the control file's settings; "cool_cancel_before_min": N calls off a cool-down that started before minute N"""
    global COOL_T0
    if not (A.ctl and os.path.exists(A.ctl)): return
    try:
        for k_, v_ in json.load(open(A.ctl)).items():
            if k_ == "cool_cancel_before_min":
                if COOL_T0 is not None and COOL_T0 / 60 < v_: out(f"[ctl] the cool-down started at {COOL_T0 / 60:.0f} min is called off"); COOL_T0 = None
            elif hasattr(A, k_) and getattr(A, k_) != v_: out(f"[ctl] {k_}: {getattr(A, k_)} -> {v_}"); setattr(A, k_, type(getattr(A, k_))(v_))
    except Exception as e: out(f"[ctl] unreadable: {str(e)[:100]}")


def chat_eval():
    """the unseen chat text's bpc, and a reply to --chat-prompt (the base model, no fine-tuning yet)"""
    if VAC is None: return None
    vc = bpc(model, VAC, A.chat_val_chars); CHATH.append((step, (time.time() - T0) / 60, vc))
    out(f"[chat] step {step} {len(model.blocks)} layers: SmolTalk test (unseen) {vc:.3f} bpc")
    out(f"[chat-sample] {json.dumps(sample(model, A.chat_prompt, n=300)[len(A.chat_prompt):])}")
    return vc


def save(tag):
    p = os.path.join(A.out, f"model_{tag}.pt")
    torch.save({"model": model.state_dict(), "layers": len(model.blocks), "step": step, "hist": hist, "best": best, "elapsed": time.time() - T0, "args": vars(A), "chars": CHARS,
                "chat_hist": CHATH, "cool_t0": COOL_T0, **({"opt": opt.state_dict()} if tag == "latest" else {})}, p)
    return p


stage_t0 = time.time(); stage_step0 = step - (A.warmup if OPT_BACK else 0); last_eval = time.time(); LR_SCALE = 1.0
apply_ctl()
vw, vf, v = validate(); hist.append((step, (time.time() - T0) / 60, len(model.blocks), vw, vf)); out(f"[val] step {step} {len(model.blocks)} layers: wiki {vw:.3f} fineweb {vf:.3f} mean {v:.3f} bpc")
chat_eval()
acc = []; t_log = time.time(); seen = step * A.chars_per_step
while True:
    el = time.time() - T0
    if el > budget: break
    if COOL_T0 is not None and el > COOL_T0 + A.cool_hours * 3600: out(f"[cool] cool-down over at {el / 60:.0f} min"); break
    w = min(1.0, (step - stage_step0 + 1) / A.warmup); dec = 1.0 if el < 0.8 * budget else max(0.1, 1.0 - 0.9 * (el - 0.8 * budget) / (0.2 * budget))
    if COOL_T0 is not None: dec = min(dec, max(0.1, 1.0 - 0.9 * (el - COOL_T0) / (A.cool_hours * 3600)))
    for g in opt.param_groups: g["lr"] = A.lr * LR_SCALE * w * dec
    n_win = A.chars_per_step // A.seq
    try:
        opt.zero_grad(set_to_none=True); tot = 0.0
        for i in range(0, n_win, MICRO):
            x, y = windows(min(MICRO, n_win - i))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                loss = F.cross_entropy(model(x).flatten(0, 1).float(), y.flatten())
            (loss * x.shape[0] / n_win).backward(); tot += float(loss.detach()) * x.shape[0] / n_win
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    except torch.cuda.OutOfMemoryError:
        opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); MICRO = max(1, MICRO // 2); out(f"[oom] micro-batch -> {MICRO}"); continue
    step += 1; seen += A.chars_per_step; acc.append(tot)
    if time.time() - t_log > 300:
        out(f"[step {step}] {len(model.blocks)} layers: train {np.mean(acc) / math.log(2):.3f} bpc, lr {A.lr * LR_SCALE * w * dec:.2e}, micro {MICRO}, "
            f"{seen / 1e9:.2f}B chars, {seen / (time.time() - T0) / 1e3:.0f}k chars/s overall, {el / 60:.0f} min"); acc = []; t_log = time.time()
    if time.time() - last_eval > A.eval_every * 60:
        vw, vf, v = validate(); el = time.time() - T0; hist.append((step, el / 60, len(model.blocks), vw, vf)); last_eval = time.time()
        out(f"[val] step {step} {len(model.blocks)} layers: wiki {vw:.3f} fineweb {vf:.3f} mean {v:.3f} bpc ({el / 60:.0f} min)")
        apply_ctl()
        vc = chat_eval()
        if vc is not None and COOL_T0 is None and stage + 1 >= len(DEPTHS):
            gain = CHATH[-1 - A.cool_evals][2] - CHATH[-1][2] if len(CHATH) > A.cool_evals else None
            if A.cool_target > 0 and vc <= A.cool_target:
                COOL_T0 = el; out(f"[cool] the chat score {vc:.3f} reached the target {A.cool_target:.3f}: the learning rate falls to a tenth over {A.cool_hours:g} h, then the run ends")
            elif A.cool_plateau > 0 and gain is not None and gain < A.cool_plateau:
                if A.plateau_cools:
                    COOL_T0 = el; out(f"[cool] the chat score gained {gain:.4f} over the last {A.cool_evals} validations (< {A.cool_plateau}): "
                                      f"the learning rate falls to a tenth over {A.cool_hours:g} h, then the run ends")
                else: out(f"[plateau] the chat score gained only {gain:.4f} over the last {A.cool_evals} validations, still {vc:.3f} against the target {A.cool_target:.3f} - going on")
        if v < best["bpc"]: best = {"bpc": v, "step": step, "layers": len(model.blocks)}; hub_put(save("best"), "model_best.pt")
        if v > best["bpc"] + A.diverge and best.get("layers") == len(model.blocks) and os.path.exists(os.path.join(A.out, "model_best.pt")):
            # diverging: the best weights back (same depth), a fresh optimizer, half the learning rate from here on
            bst = torch.load(os.path.join(A.out, "model_best.pt"), map_location=DEV, weights_only=False); model.load_state_dict(bst["model"])
            LR_SCALE *= 0.5; opt = make_opt(); stage_step0 = step
            out(f"[diverge] validation {v:.3f} against the best {best['bpc']:.3f} (step {best['step']}): back to those weights, learning rate x{LR_SCALE:g} ({A.lr * LR_SCALE:.1e})")
            hist.append((step, (time.time() - T0) / 60, len(model.blocks), bst["hist"][-1][3] if bst.get("hist") else v, bst["hist"][-1][4] if bst.get("hist") else v))
        hub_put(save("latest"), "model_latest.pt"); hub_put(LOGF, "train.log")
        same = [h for h in hist if h[2] == len(model.blocks)]
        m_ = lambda h: (h[3] + h[4]) / 2
        plateau = len(same) >= 3 and (m_(same[-3]) - m_(same[-2]) < A.plateau) and (m_(same[-2]) - m_(same[-1]) < A.plateau)
        if stage + 1 < len(DEPTHS) and plateau and (time.time() - stage_t0) > A.stage_min * budget and el < 0.75 * budget:
            stage += 1; model.grow(DEPTHS[stage]); opt = make_opt(); stage_t0 = time.time(); stage_step0 = step
            vw2, vf2, v2 = validate()
            out(f"[grow] -> {len(model.blocks)} layers ({model.n_params() / 1e6:.1f}M params) at step {step}; validation after the identity blocks {v2:.3f} (before {v:.3f})")

hub_put(save("latest"), "model_latest.pt")
out(f"[charlm2] training over: {step} steps, {seen / 1e9:.2f}B chars, {len(model.blocks)} layers, best validation {best['bpc']:.3f} at step {best['step']}")

# ---- the yardsticks: test bpc here and GPT-2 small's on the same raw documents ---------------------------------------
res = {"layers": len(model.blocks), "params": model.n_params(), "steps": step, "chars_seen": seen, "hist": hist}
res["test_wiki"], res["test_fw"] = bpc(model, TEW), bpc(model, TEF)
out(f"[test] {len(model.blocks)} layers, {model.n_params() / 1e6:.1f}M params: WikiText-2 test {res['test_wiki']:.3f} bpc, FineWeb-Edu test {res['test_fw']:.3f} bpc")
try:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); g2 = GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()

    @torch.no_grad()
    def gpt2_bpc(texts):
        nll = 0.0; nch = 0
        for t in texts:
            ids = torch.tensor(tok.encode(t), device=DEV); nch += len(t); prev = 0
            if len(ids) < 2: continue
            for b in range(0, len(ids), 512):
                e = min(b + 1024, len(ids)); x = ids[max(0, e - 1024):e][None]; trg = e - prev; lo = g2(x).logits[0].float()
                lg, y = (lo[-trg - 1:-1], x[0, -trg:]) if trg < x.shape[1] else (lo[:-1], x[0, 1:])
                nll += float(F.cross_entropy(lg, y, reduction="sum")); prev = e
                if e == len(ids): break
        return nll / nch / math.log(2)
    res["gpt2_test_wiki"] = gpt2_bpc([open(os.path.join(A.data, "test_wiki.txt")).read()])
    res["gpt2_test_fw"] = gpt2_bpc(open(os.path.join(A.data, "test_fw.txt")).read().split("\x00"))
    out(f"[test] GPT-2 small (124M) on the same raw documents: WikiText-2 {res['gpt2_test_wiki']:.3f} bpc, FineWeb-Edu {res['gpt2_test_fw']:.3f} bpc")
    del g2; torch.cuda.empty_cache()
except Exception as e:
    out(f"[test] GPT-2 comparison failed: {str(e)[:200]}")
res["samples"] = [sample(model, p) for p in ("The history of the", "Photosynthesis is the process", "In 1998, the band released", "To solve the equation, we first")]
for s_ in res["samples"]: out(f"[sample] {json.dumps(s_)}")

# ---- GPTQ to 4 bits on the app's grid (affine, group 64), every linear layer, the two tables left in fp16 ----------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import q4
    def gptq_linear(W, H, group=64, bits=4, block=128, damp=0.01):
        W = W.clone().float(); rows, cols = W.shape; nb = float((1 << bits) - 1)
        H = H.double(); dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0; Hinv = None
        for d_ in (damp, 0.1, 1.0, 10.0):   # in float64, the damping raised until the factorisation goes through (it once failed at 0.01)
            try:
                Hd = H + torch.eye(cols, device=W.device, dtype=H.dtype) * d_ * torch.mean(torch.diag(H))
                Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hd)), upper=True).float(); break
            except Exception: continue
        if Hinv is None: raise RuntimeError("GPTQ: the Hessian would not factorise even at damping 10")
        Q = torch.zeros_like(W); codes = torch.zeros(rows, cols, dtype=torch.uint8, device=W.device); sc = torch.zeros(rows, cols // group, device=W.device); bi = torch.zeros_like(sc)
        for i1 in range(0, cols, block):
            i2 = min(i1 + block, cols); W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
            for i in range(i2 - i1):
                col = i1 + i
                if col % group == 0:
                    _, s_, b_ = q4.affine_params(torch.cat([W1[:, i:], W[:, i2:]], 1)[:, :group], group, bits); s = s_.reshape(rows); b = b_.reshape(rows); sc[:, col // group] = s; bi[:, col // group] = b
                c = torch.clamp(torch.round((W1[:, i] - b) / s), 0, nb); q = c * s + b; Q1[:, i] = q; codes[:, col] = c.to(torch.uint8)
                err = (W1[:, i] - q) / Hi[i, i]; W1[:, i:] -= err[:, None] * Hi[i, i:][None]; E1[:, i] = err
            Q[:, i1:i2] = Q1; W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
        return Q, codes, sc, bi
    model.eval(); calib = windows(32)[0]; PACK = {}
    with torch.no_grad():
        x = model.emb(calib) + model.pos(torch.arange(A.seq, device=DEV))
        for li, blk in enumerate(model.blocks):
            for n in ("qkv", "proj", "fc", "fc2"):
                lin = getattr(blk, n); H = torch.zeros(lin.in_features, lin.in_features, device=DEV); cnt = [0]
                def hook(mod, inp, o): z = inp[0].reshape(-1, inp[0].shape[-1]).float(); H.add_(z.T @ z); cnt[0] += z.shape[0]
                h = lin.register_forward_hook(hook); blk(x); h.remove()
                Q, c, s_, b_ = gptq_linear(lin.weight.data, H / cnt[0]); lin.weight.data = Q.to(lin.weight.dtype); PACK[f"blocks.{li}.{n}"] = (c.cpu(), s_.cpu(), b_.cpu())
            x = blk(x)
        z = model.ln_f(x).reshape(-1, A.d).float(); Q, c, s_, b_ = gptq_linear(model.head.weight.data, z.T @ z / z.shape[0]); model.head.weight.data = Q; PACK["head"] = (c.cpu(), s_.cpu(), b_.cpu())
    res["gptq_test_wiki"], res["gptq_test_fw"] = bpc(model, TEW), bpc(model, TEF)
    shifts = torch.arange(0, 32, 4, dtype=torch.int64); sd = {}
    for k, v_ in model.state_dict().items():
        key = k[:-7] if k.endswith(".weight") else None
        if key in PACK:
            c, s_, b_ = PACK[key]; sd[k] = ((c.to(torch.int64).reshape(c.shape[0], -1, 8) << shifts).sum(-1) & 0xFFFFFFFF).to(torch.int32); sd[key + ".scales"] = s_.half(); sd[key + ".biases"] = b_.half()
        else: sd[k] = v_.detach().cpu().half()
    res["packed_bytes"] = sum(v_.numel() * v_.element_size() for v_ in sd.values())
    pq_ = os.path.join(A.out, "model_4bit.pt"); torch.save({"state": sd, "layers": len(model.blocks), "args": vars(A), "chars": CHARS, "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}}, pq_); hub_put(pq_, "model_4bit.pt")
    out(f"[gptq] 4-bit: WikiText-2 {res['gptq_test_wiki']:.3f} bpc, FineWeb-Edu {res['gptq_test_fw']:.3f} bpc; packed {res['packed_bytes'] / 1e6:.1f} MB (bf16 {2 * model.n_params() / 1e6:.1f} MB)")
except Exception as e:
    out(f"[gptq] failed: {str(e)[:300]}")
json.dump(res, open(os.path.join(A.out, "result.json"), "w"), indent=1); hub_put(os.path.join(A.out, "result.json"), "result.json"); hub_put(LOGF, "train.log")
out("CHARLM2_DONE")
