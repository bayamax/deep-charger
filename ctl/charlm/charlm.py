#!/usr/bin/env python3
"""A character-level language model grown from small, on the CPU.

The idea: the token table and its 50k-row output head are what makes a small LM big (GPT-2: 31% of the parameters, and
the part that would not go to 4 bits). Here the vocabulary is the printable ASCII characters plus newline and one
unknown symbol (97 rows), each with a learned vector exactly as a token would have - the next-character prediction task
over text, the way Meta's byte-level models read bytes. The decoder starts shallow and gains layers as it plateaus
(the context BART's growth: a new block starts as the identity, its two output projections zero, so the model's
function is unchanged at the moment of growth and the new depth is trained from there).

Text: WikiText-103 train (curated Good/Featured articles) - the first parquet shard, its last 2% held out as the
validation set that decides growth. The final yardstick is WikiText-2 test (the same held-out articles WikiText-103
tests on), never used for a decision: bits per character, beside GPT-2 small's on the same text.

  python3 charlm.py --train wikitext-103 parquet --test wikitext-2 test parquet --out /root/charlm --hours 5.5 --layers 2,4,8
  python3 charlm.py ... --bench 1        # tokens/s at each depth, then exit
"""
import argparse, codecs, json, math, os, random, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True); ap.add_argument("--test", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--d", type=int, default=384); ap.add_argument("--heads", type=int, default=6); ap.add_argument("--ff", type=int, default=0, help="0: 4d")
ap.add_argument("--layers", default="2,4,8", help="the depths, in order: the model grows from one to the next")
ap.add_argument("--seq", type=int, default=512); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--warmup", type=int, default=200)
ap.add_argument("--hours", type=float, default=5.5, help="the training budget; the last fifth of it decays the learning rate")
ap.add_argument("--eval-every", type=float, default=15.0, help="minutes between validations")
ap.add_argument("--val-chars", type=int, default=300_000); ap.add_argument("--plateau", type=float, default=0.004, help="grow when two validations in a row gain less than this (bpc)")
ap.add_argument("--stage-min", type=float, default=0.12, help="a depth gets at least this share of the budget before it may grow")
ap.add_argument("--threads", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--bench", type=int, default=0); ap.add_argument("--resume", default="")
A = ap.parse_args()
torch.manual_seed(A.seed); random.seed(A.seed); np.random.seed(A.seed); torch.set_num_threads(A.threads)
FF = A.ff or 4 * A.d
os.makedirs(A.out, exist_ok=True)
LOG = open(os.path.join(A.out, "train.log"), "a")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

# ---- the alphabet: newline, the 95 printable ASCII characters, and one symbol for anything else -------------------------
CHARS = "\n" + "".join(chr(i) for i in range(32, 127)); UNK = len(CHARS); V = UNK + 1
codecs.register_error("unk", lambda e: ("\x7f", e.end))          # a run of non-ASCII characters becomes one unknown
LUT = np.full(256, UNK, np.uint8)
for i, c in enumerate(CHARS): LUT[ord(c)] = i
def encode(text): return LUT[np.frombuffer(text.encode("ascii", "unk"), dtype=np.uint8)]
def decode(ids): return "".join(CHARS[i] if i < UNK else "�" for i in ids)

import pandas as pd  # noqa: E402
def read_parquet_text(p): return "".join(t for t in pd.read_parquet(p)["text"].tolist())
t0 = time.time()
TRAIN_TEXT = read_parquet_text(A.train); cut = int(len(TRAIN_TEXT) * 0.98)
TRAIN = encode(TRAIN_TEXT[:cut]); VAL = encode(TRAIN_TEXT[cut:cut + A.val_chars]); del TRAIN_TEXT
TEST_TEXT = "\n\n".join(pd.read_parquet(A.test)["text"].tolist()); TEST = encode(TEST_TEXT)
out(f"[charlm] alphabet {V} (unknown share: train {100 * (TRAIN == UNK).mean():.2f}%, test {100 * (TEST == UNK).mean():.2f}%); "
    f"train {len(TRAIN) / 1e6:.1f}M chars, validation {len(VAL) / 1e3:.0f}k, test {len(TEST) / 1e6:.2f}M [{time.time() - t0:.0f}s]")
# GPT-2 small on this test text, from gpt2_gptq.py's run: perplexity 25.17 over 287,643 BPE tokens -> bits per character
GPT2_BPC = math.log(25.17) * 287643 / (len(TEST_TEXT) * math.log(2))
out(f"[charlm] GPT-2 small on the same test text: {GPT2_BPC:.3f} bits per character (124M parameters)")


# ---- the model ---------------------------------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1 = nn.LayerNorm(A.d); s.qkv = nn.Linear(A.d, 3 * A.d); s.proj = nn.Linear(A.d, A.d)
        s.ln2 = nn.LayerNorm(A.d); s.fc = nn.Linear(A.d, FF); s.fc2 = nn.Linear(FF, A.d)

    def forward(s, x):
        B, T, D = x.shape; q, k, v = s.qkv(s.ln1(x)).split(D, -1)
        sh = lambda z: z.view(B, T, A.heads, D // A.heads).transpose(1, 2)
        a = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + s.proj(a)
        return x + s.fc2(F.gelu(s.fc(s.ln2(x))))

    def as_identity(s):
        """the residual passes through untouched: both output projections zero"""
        for m in (s.proj, s.fc2): nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        return s


class CharLM(nn.Module):
    def __init__(s, layers):
        super().__init__()
        s.emb = nn.Embedding(V, A.d); s.pos = nn.Embedding(A.seq, A.d)
        s.blocks = nn.ModuleList([Block() for _ in range(layers)]); s.ln_f = nn.LayerNorm(A.d); s.head = nn.Linear(A.d, V)
        s.apply(s._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)

    def forward(s, ids):
        x = s.emb(ids) + s.pos(torch.arange(ids.shape[1]))
        for b in s.blocks: x = b(x)
        return s.head(s.ln_f(x))

    def grow(s, to):
        """interleave: after each block an identity one, until there are `to` blocks (the extra ones at the end)"""
        new = []
        for b in s.blocks:
            new.append(b)
            if len(new) < to: new.append(Block().as_identity())
        while len(new) < to: new.append(Block().as_identity())
        s.blocks = nn.ModuleList(new[:to])

    def n_params(s): return sum(p.numel() for p in s.parameters())


def batch(data, n):
    i = np.random.randint(0, len(data) - A.seq - 1, n)
    x = torch.from_numpy(np.stack([data[j:j + A.seq] for j in i]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[j + 1:j + A.seq + 1] for j in i]).astype(np.int64))
    return x, y


@torch.no_grad()
def bpc(model, data, stride=256, max_chars=None):
    """bits per character over `data`: windows of --seq, each scoring its last `stride` characters (the first its whole)"""
    model.eval(); nll = 0.0; n = 0; T = A.seq; end = len(data) if max_chars is None else min(len(data), max_chars)
    starts = list(range(0, end - T - 1, stride)); i = 0; eb = 32
    while i < len(starts):
        s_ = starts[i:i + eb]; x = torch.from_numpy(np.stack([data[j:j + T] for j in s_]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[j + 1:j + T + 1] for j in s_]).astype(np.int64))
        lg = model(x)
        for r, j in enumerate(s_):
            k = 0 if j == 0 else T - stride
            nll += float(F.cross_entropy(lg[r, k:], y[r, k:], reduction="sum")); n += T - k
        i += eb
    model.train(); return nll / n / math.log(2)


@torch.no_grad()
def sample(model, prompt, n=300, temp=0.8):
    model.eval(); ids = torch.tensor([encode(prompt).astype(np.int64)])
    for _ in range(n):
        lg = model(ids[:, -A.seq:])[0, -1] / temp; p = F.softmax(lg, -1); nxt = torch.multinomial(p, 1)
        ids = torch.cat([ids, nxt[None]], 1)
    model.train(); return decode(ids[0].tolist())


DEPTHS = [int(v) for v in A.layers.split(",")]
if A.bench:
    for L in DEPTHS:
        m = CharLM(L); opt = torch.optim.AdamW(m.parameters(), lr=A.lr); x, y = batch(TRAIN, A.batch)
        for _ in range(2): opt.zero_grad(); F.cross_entropy(m(x).flatten(0, 1), y.flatten()).backward(); opt.step()
        t = time.time(); n = 5
        for _ in range(n): opt.zero_grad(); F.cross_entropy(m(x).flatten(0, 1), y.flatten()).backward(); opt.step()
        dt = (time.time() - t) / n
        out(f"[bench] {L} layers, d {A.d}, ff {FF}: {m.n_params() / 1e6:.2f}M params, {dt:.2f}s per step of {A.batch}x{A.seq} = {A.batch * A.seq / dt:.0f} chars/s")
    sys.exit(0)

# ---- training: stage by stage through the depths, the learning rate warmed up after each growth ------------------------
model = CharLM(DEPTHS[0]); stage = 0; step = 0; T0 = time.time(); budget = A.hours * 3600
hist = []   # (step, minutes, depth, val bpc)
best = {"bpc": 9.0, "step": 0}
if A.resume and os.path.exists(A.resume):
    st = torch.load(A.resume, map_location="cpu", weights_only=False); model = CharLM(st["layers"]); model.load_state_dict(st["model"])
    stage = DEPTHS.index(st["layers"]); step = st["step"]; hist = st.get("hist", []); best = st.get("best", best); T0 = time.time() - st.get("elapsed", 0)
    out(f"[charlm] resumed at step {step}, {st['layers']} layers")
opt = torch.optim.AdamW(model.parameters(), lr=A.lr, betas=(0.9, 0.95), weight_decay=0.1)
out(f"[charlm] {DEPTHS[stage]} layers, d {A.d}, ff {FF}, heads {A.heads}: {model.n_params() / 1e6:.2f}M params; budget {A.hours} h, depths {DEPTHS}")
stage_t0 = time.time(); last_eval = time.time(); stage_step0 = step; v = bpc(model, VAL, max_chars=A.val_chars); hist.append((step, 0.0, DEPTHS[stage], v))
out(f"[val] step {step} {DEPTHS[stage]} layers: {v:.3f} bpc")


def save(tag):
    torch.save({"model": model.state_dict(), "layers": len(model.blocks), "step": step, "hist": hist, "best": best, "elapsed": time.time() - T0,
                "args": vars(A), "chars": CHARS}, os.path.join(A.out, f"model_{tag}.pt"))


acc = []; t_log = time.time()
while True:
    el = time.time() - T0
    if el > budget: break
    # the learning rate: warm-up after every growth, the last fifth of the budget a linear decay to a tenth
    w = min(1.0, (step - stage_step0 + 1) / A.warmup); dec = 1.0 if el < 0.8 * budget else max(0.1, 1.0 - 0.9 * (el - 0.8 * budget) / (0.2 * budget))
    for g in opt.param_groups: g["lr"] = A.lr * w * dec
    x, y = batch(TRAIN, A.batch); loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    step += 1; acc.append(float(loss))
    if time.time() - t_log > 120:
        out(f"[step {step}] {len(model.blocks)} layers: train {np.mean(acc) / math.log(2):.3f} bpc, lr {A.lr * w * dec:.2e}, {step * A.batch * A.seq / 1e6:.1f}M chars seen, {el / 60:.0f} min"); acc = []; t_log = time.time()
    if time.time() - last_eval > A.eval_every * 60:
        v = bpc(model, VAL, max_chars=A.val_chars); el = time.time() - T0; hist.append((step, el / 60, len(model.blocks), v)); last_eval = time.time()
        out(f"[val] step {step} {len(model.blocks)} layers: {v:.3f} bpc ({el / 60:.0f} min)")
        if v < best["bpc"]: best = {"bpc": v, "step": step}; save("best")
        save("latest")
        same = [h for h in hist if h[2] == len(model.blocks)]
        plateau = len(same) >= 3 and (same[-3][3] - same[-2][3] < A.plateau) and (same[-2][3] - same[-1][3] < A.plateau)
        long_enough = (time.time() - stage_t0) > A.stage_min * budget
        if stage + 1 < len(DEPTHS) and plateau and long_enough and el < 0.8 * budget:
            stage += 1; model.grow(DEPTHS[stage]); opt = torch.optim.AdamW(model.parameters(), lr=A.lr, betas=(0.9, 0.95), weight_decay=0.1)
            stage_t0 = time.time(); stage_step0 = step
            v2 = bpc(model, VAL, max_chars=A.val_chars)
            out(f"[grow] -> {len(model.blocks)} layers ({model.n_params() / 1e6:.2f}M params) at step {step}; validation unchanged by the identity blocks: {v2:.3f} bpc")

save("latest")
out(f"[charlm] training over: {step} steps, {step * A.batch * A.seq / 1e6:.1f}M chars, {len(model.blocks)} layers, best validation {best['bpc']:.3f} bpc at step {best['step']}")
t = time.time(); tb = bpc(model, TEST)
out(f"[test] final model ({len(model.blocks)} layers, {model.n_params() / 1e6:.2f}M params): WikiText-2 test {tb:.3f} bits per character (GPT-2 small {GPT2_BPC:.3f}) [{(time.time() - t) / 60:.1f} min]")
for p in ("The history of the", "In 1998, the band released", "She was born in"):
    out(f"[sample] {json.dumps(sample(model, p))}")
json.dump({"test_bpc": tb, "gpt2_bpc": GPT2_BPC, "layers": len(model.blocks), "params": model.n_params(), "steps": step, "hist": hist}, open(os.path.join(A.out, "result.json"), "w"), indent=1)
out("CHARLM_DONE")
