#!/usr/bin/env python3
"""Chat fine-tuning of the character model (charlm2.py's checkpoint) on SmolTalk's conversational parts.

The alphabet gains two symbols, USER and ASSISTANT (98 -> 100); a conversation is
    [USER] text [ASSISTANT] text [USER] text [ASSISTANT] text [EOD]
(a system message goes in front of the first user turn, after its own [USER]). The new rows start at the mean of the
character vectors (input) and at zero (output), so the base model is unchanged on plain text. The loss counts only the
assistant's characters and the symbol that ends its turn, so the model learns to answer and to stop, not to write the
user's side. Conversations are packed into the 4096-character windows whole, cut at a turn boundary when too long.

Sources: SmolTalk's chat, rewriting, summarising and instruction parts (no maths derivations, code or function calls,
which a model this size does not carry). Held out: SmolTalk's own test split, scored as bits per assistant character,
before and after. At the end: sample replies, GPTQ 4-bit, uploads.

  python3 charlm_sft.py --base run2/model_latest.pt --data sftdata --out sft1 --hours 3 --hub repo:charlm/sft1
"""
import argparse, json, math, os, random, shutil, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--shards", type=int, default=4, help="SmolTalk train shards (of 9): the first N")
ap.add_argument("--shard-list", default="", help="these SmolTalk train shards instead (e.g. 4,5,6,7,8)")
ap.add_argument("--sources", default="smol-magpie-ultra,everyday-conversations,systemchats-30k,openhermes-100k,smol-rewrite,smol-summarize,smol-constraints,explore-instruct-rewriting,longalign")
ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--warmup", type=int, default=100); ap.add_argument("--hours", type=float, default=3.0)
ap.add_argument("--stop-plateau", type=float, default=0.0, help="stop when the held-out score has not improved by this over --stop-evals evaluations (0: off; --hours stays the cap)")
ap.add_argument("--stop-evals", type=int, default=3)
ap.add_argument("--chars-per-step", type=int, default=65536); ap.add_argument("--micro", type=int, default=8); ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--eval-every", type=float, default=20.0); ap.add_argument("--val-convs", type=int, default=600)
ap.add_argument("--jsonl", default="", help="conversations of one's own, comma-separated .jsonl files of {\"messages\": [...]} (with --shards 0: only these)")
ap.add_argument("--val-jsonl", default="", help="held-out conversations as .jsonl (in place of SmolTalk's test split)")
ap.add_argument("--prompts-json", default="", help="a JSON list of user turns for the sample replies")
ap.add_argument("--min-learned", type=int, default=20, help="drop conversations with fewer assistant characters to learn")
ap.add_argument("--ckpt-every", type=float, default=0.0, help="minutes between resume checkpoints (A.out/resume.pt; a rerun picks it up and continues; 0: off)")
ap.add_argument("--hub", default=""); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-convs", type=int, default=0)
A = ap.parse_args()
torch.manual_seed(A.seed); random.seed(A.seed); np.random.seed(A.seed)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(A.out, exist_ok=True); LOGF = os.path.join(A.out, "train.log"); LOG = open(LOGF, "a")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

CK = torch.load(A.base, map_location="cpu", weights_only=False); BA = CK["args"]
D, HEADS, SEQ = BA["d"], BA["heads"], BA["seq"]; FF = BA.get("ff") or 4 * D
CHARS = "\n" + "".join(chr(i) for i in range(32, 127)); UNK = len(CHARS); EOD = UNK + 1; USR = EOD + 1; AST = USR + 1; V = AST + 1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import charlm_fold as FOLDMOD  # noqa: E402  (the prep's alphabet and Unicode fold)
encode = FOLDMOD.encode
def decode(ids): return "".join(CHARS[i] if i < UNK else {UNK: "�", EOD: "<EOD>", USR: "\n[USER] ", AST: "\n[ASSISTANT] "}[i] for i in ids)


class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1 = nn.LayerNorm(D); s.qkv = nn.Linear(D, 3 * D); s.proj = nn.Linear(D, D)
        s.ln2 = nn.LayerNorm(D); s.fc = nn.Linear(D, FF); s.fc2 = nn.Linear(FF, D)
        s.qk = bool(BA.get("qknorm", 0))
        if s.qk: s.qn = nn.LayerNorm(D // HEADS); s.kn = nn.LayerNorm(D // HEADS)

    def forward(s, x):
        B, T, _ = x.shape; q, k, v = s.qkv(s.ln1(x)).split(D, -1)
        sh = lambda z: z.view(B, T, HEADS, D // HEADS)
        q, k, v = sh(q), sh(k), sh(v)
        if s.qk: q, k = s.qn(q), s.kn(k)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + s.proj(a)
        return x + s.fc2(F.gelu(s.fc(s.ln2(x)), approximate="tanh"))


class CharLM(nn.Module):
    def __init__(s, layers, vocab):
        super().__init__()
        s.emb = nn.Embedding(vocab, D); s.pos = nn.Embedding(SEQ, D)
        s.blocks = nn.ModuleList([Block() for _ in range(layers)]); s.ln_f = nn.LayerNorm(D); s.head = nn.Linear(D, vocab)

    def forward(s, ids):
        x = s.emb(ids) + s.pos(torch.arange(ids.shape[1], device=ids.device))
        for b in s.blocks: x = b(x)
        return s.head(s.ln_f(x))

    def n_params(s): return sum(p.numel() for p in s.parameters())


sd = CK["model"]; V0 = sd["emb.weight"].shape[0]
model = CharLM(CK["layers"], V)
with torch.no_grad():
    for k, v in sd.items():
        p = model.state_dict()[k]
        if p.shape == v.shape: p.copy_(v)
    mean = sd["emb.weight"].mean(0)
    for r in range(V0, V): model.emb.weight[r] = mean + 0.02 * torch.randn(D)
    model.emb.weight[:V0] = sd["emb.weight"]; model.head.weight[:V0] = sd["head.weight"]; model.head.bias[:V0] = sd["head.bias"]
    model.head.weight[V0:] = 0; model.head.bias[V0:] = sd["head.bias"].mean()
model = model.to(DEV)
out(f"[sft] base {A.base}: {CK['layers']} layers, d {D}, context {SEQ}, alphabet {V0} -> {V}; {model.n_params() / 1e6:.1f}M params")

# ---- conversations -> (ids, loss mask) -------------------------------------------------------------------------------
import pyarrow.parquet as pq  # noqa: E402
SRC = set(A.sources.split(","))


def fetch(path):
    dst = os.path.join(A.data, path.replace("/", "_"))
    if not os.path.exists(dst):
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("HuggingFaceTB/smoltalk", path, repo_type="dataset", local_dir=os.path.join(A.data, "dl")); os.replace(p, dst)
    return dst


def conv_ids(msgs):
    """[USER] ... [ASSISTANT] ... [EOD]; mask[i] = 1 where the token at i is a target to learn (assistant text and the
    symbol that closes its turn)"""
    ids, mask = [], []; sys_ = ""; turns = []
    for m in msgs:
        if m["role"] == "system": sys_ += m["content"].strip() + "\n\n"; continue
        c = m["content"]
        if m["role"] == "user" and sys_: c = sys_ + c; sys_ = ""
        turns.append((m["role"], c))
    for role, text in turns:
        a = encode(text.strip()).tolist()
        if role == "user": ids += [USR] + a; mask += [0] * (1 + len(a))
        else: ids += [AST] + a; mask += [0] + [1] * len(a)
        # the symbol after an assistant turn is learned (where the next turn's marker or EOD lands)
    ids.append(EOD); mask.append(1 if turns and turns[-1][0] == "assistant" else 0)
    for i in range(1, len(ids)):
        if ids[i] == USR and ids[i - 1] != EOD and mask[i - 1] == 1: mask[i] = 1
    return ids, mask


def cut(ids, mask, T):
    """a conversation longer than the window ends at its last assistant turn that fits"""
    if len(ids) <= T: return ids, mask
    last = max((i for i in range(1, T) if ids[i] == USR), default=0)
    if last <= 0: return None
    ids, mask = ids[:last] + [EOD], mask[:last] + [1]
    return ids, mask


def jsonl_table(p):
    return [(json.loads(l)["messages"], "jsonl") for l in open(p) if l.strip()]


def load(paths, limit=0):
    convs = []; skipped = collections_counter()
    for p in paths:
        if p.endswith(".jsonl"): rows = jsonl_table(p)
        else: t = pq.read_table(p, columns=["messages", "source"]); rows = zip(t.column("messages").to_pylist(), t.column("source").to_pylist())
        for msgs, src in rows:
            if src != "jsonl" and src not in SRC: skipped["source"] += 1; continue
            ids, mask = conv_ids(msgs)
            if np.mean(np.array(ids) == UNK) > 0.005: skipped["non-ascii"] += 1; continue
            r = cut(ids, mask, SEQ)
            if r is None: skipped["too long"] += 1; continue
            if sum(r[1]) < A.min_learned: skipped["too little to learn"] += 1; continue
            convs.append((np.asarray(r[0], np.uint8), np.asarray(r[1], np.uint8)))   # one byte a symbol: as int lists 695M characters took tens of GB
            if limit and len(convs) >= limit: return convs, skipped
    return convs, skipped


def collections_counter():
    import collections; return collections.Counter()


t0 = time.time()
train_paths = [fetch(f"data/all/train-{i:05d}-of-00009.parquet") for i in ([int(x) for x in A.shard_list.split(",")] if A.shard_list else range(A.shards))] + [x for x in A.jsonl.split(",") if x]
TRAIN, sk = load(train_paths, A.max_convs); VAL, _ = load([A.val_jsonl] if A.val_jsonl else [fetch("data/all/test-00000-of-00001.parquet")], A.val_convs)
ntr = sum(len(c[0]) for c in TRAIN); nas = sum(int(c[1].sum()) for c in TRAIN)
out(f"[sft] {len(TRAIN)} training conversations ({ntr / 1e6:.0f}M chars, {nas / 1e6:.0f}M assistant chars learned); skipped {dict(sk)}; validation {len(VAL)} [{time.time() - t0:.0f}s]")


def pack(convs, shuffle=True):
    """conversations whole into windows of SEQ+1 symbols; the rest of a window is EOD with no loss (uint8 arrays)"""
    order = np.random.permutation(len(convs)) if shuffle else np.arange(len(convs))
    rows, cur = [], []; n = 0
    def flush():
        ids = np.full(SEQ + 1, EOD, np.uint8); msk = np.zeros(SEQ + 1, np.uint8); o = 0
        for a, m in cur: ids[o:o + len(a)] = a; msk[o:o + len(m)] = m; o += len(a)
        rows.append((ids, msk))
    for k in order:
        a, m = convs[k]
        if n + len(a) > SEQ + 1: flush(); cur, n = [], 0
        cur.append((a, m)); n += len(a)
    if cur: flush()
    return rows


def run_rows(rows):
    a = torch.from_numpy(np.stack([r[0] for r in rows]).astype(np.int64)).to(DEV); m = torch.from_numpy(np.stack([r[1] for r in rows]).astype(np.float32)).to(DEV)
    with torch.autocast(DEV, dtype=torch.bfloat16):   # bf16 on the CPU too (about 3x faster there)
        lg = model(a[:, :-1])
    nll = F.cross_entropy(lg.flatten(0, 1).float(), a[:, 1:].flatten(), reduction="none") * m[:, 1:].flatten()
    return nll.sum(), m[:, 1:].sum()


@torch.no_grad()
def val_bpc():
    model.eval(); rows = pack(VAL, shuffle=False); tot = n = 0.0
    for i in range(0, len(rows), 4):
        s_, c_ = run_rows(rows[i:i + 4]); tot += float(s_); n += float(c_)
    model.train(); return tot / n / math.log(2)


@torch.no_grad()
def reply(prompt, n=500, temp=0.7):
    model.eval(); ids = [USR] + encode(prompt).tolist() + [AST]; x = torch.tensor([ids], device=DEV); outp = []
    for _ in range(n):
        with torch.autocast(DEV, dtype=torch.bfloat16):   # bf16 on the CPU too (about 3x faster there)
            lg = model(x[:, -SEQ:])[0, -1].float() / temp
        nxt = int(torch.multinomial(F.softmax(lg, -1), 1))
        if nxt in (USR, EOD): break
        outp.append(nxt); x = torch.cat([x, torch.tensor([[nxt]], device=DEV)], 1)
    model.train(); return decode(outp)


def hub_put(local, remote):
    if not A.hub: return
    try:
        from huggingface_hub import HfApi
        repo, pre = A.hub.split(":", 1); HfApi().upload_file(path_or_fileobj=local, path_in_repo=f"{pre}/{remote}", repo_id=repo)
    except Exception as e:
        out(f"[hub] {remote} failed: {str(e)[:120]}")


PROMPTS = ["Hi! How are you today?", "What is photosynthesis? Explain it simply.", "Write a short poem about the sea.",
           "Can you give me three tips for staying focused while studying?", "Rewrite this sentence to sound more polite: Give me the report now."]
if A.prompts_json: PROMPTS = json.load(open(A.prompts_json))
RESUME = os.path.join(A.out, "resume.pt"); RS = torch.load(RESUME, map_location="cpu", weights_only=False) if os.path.exists(RESUME) else None
if RS: model.load_state_dict(RS["model"]); v0 = RS["v0"]; out(f"[resume] step {RS['step']} from {RESUME} ({RS['elapsed'] / 60:.0f} min done)")
else:
    v0 = val_bpc(); out(f"[val] before fine-tuning: {v0:.3f} bits per assistant character")
    for p in PROMPTS[:2]: out(f"[reply before] {json.dumps(p)} -> {json.dumps(reply(p, 300))}")

opt = torch.optim.AdamW(model.parameters(), lr=A.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=DEV == "cuda")
rows_per_step = max(1, A.chars_per_step // SEQ); epoch_rows = pack(TRAIN); total_steps = int(A.epochs * len(epoch_rows) / rows_per_step)
out(f"[sft] {len(epoch_rows)} windows per epoch, {rows_per_step} per update -> {total_steps} updates for {A.epochs} epochs (or {A.hours} h)")
T0 = time.time(); step = 0; ptr = 0; last_eval = time.time(); best = v0; acc = []; MICRO = A.micro; VH = [v0]; last_ck = time.time()
if RS:
    opt.load_state_dict(RS["opt"]); step, ptr, best, VH, epoch_rows = RS["step"], RS["ptr"], RS["best"], RS["VH"], RS["rows"]
    T0 = time.time() - RS["elapsed"]; np.random.set_state(RS["np"]); del RS


def save_resume():
    """model, optimizer and place in the epoch; written beside the old one and swapped in, or over it when the disk is short"""
    tmp = RESUME + ".tmp"; free = shutil.disk_usage(A.out).free
    if os.path.exists(RESUME) and free < os.path.getsize(RESUME) + 700e6: os.remove(RESUME)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "ptr": ptr, "best": best, "VH": VH, "rows": epoch_rows,
                "v0": v0, "elapsed": time.time() - T0, "np": np.random.get_state()}, tmp); os.replace(tmp, RESUME)
while step < total_steps and time.time() - T0 < A.hours * 3600:
    frac = step / max(1, total_steps); lr = A.lr * min(1.0, (step + 1) / A.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
    for g in opt.param_groups: g["lr"] = lr
    if ptr + rows_per_step > len(epoch_rows): epoch_rows = pack(TRAIN); ptr = 0
    batch = epoch_rows[ptr:ptr + rows_per_step]; ptr += rows_per_step
    try:
        opt.zero_grad(set_to_none=True); tot = cnt = 0.0
        n_all = sum(int(r[1][1:].sum()) for r in batch)
        for i in range(0, len(batch), MICRO):
            s_, c_ = run_rows(batch[i:i + MICRO])
            (s_ / max(n_all, 1)).backward(); tot += float(s_); cnt += float(c_)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    except torch.cuda.OutOfMemoryError:
        opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); MICRO = max(1, MICRO // 2); out(f"[oom] micro -> {MICRO}"); ptr -= rows_per_step; continue
    step += 1; acc.append(tot / max(cnt, 1))
    if step % 100 == 0: out(f"[step {step}/{total_steps}] train {np.mean(acc) / math.log(2):.3f} bits/assistant char, lr {lr:.2e}, {(time.time() - T0) / 60:.0f} min"); acc = []
    if time.time() - last_eval > A.eval_every * 60:
        v = val_bpc(); last_eval = time.time(); out(f"[val] step {step}: {v:.3f} bits per assistant character"); VH.append(v)
        for p in PROMPTS[step // 100 % len(PROMPTS):][:2]: out(f"[reply step {step}] {json.dumps(p)} -> {json.dumps(reply(p, 300))}")
        if v < best:
            best = v; p_ = os.path.join(A.out, "model_best.pt")
            torch.save({"model": model.state_dict(), "layers": CK["layers"], "args": BA, "vocab": V, "chars": CHARS, "special": {"UNK": UNK, "EOD": EOD, "USER": USR, "ASSISTANT": AST}}, p_); hub_put(p_, "model_best.pt")
        hub_put(LOGF, "train.log")
        if A.stop_plateau > 0 and len(VH) > A.stop_evals and min(VH[:-A.stop_evals]) - min(VH[-A.stop_evals:]) < A.stop_plateau:
            out(f"[stop] the held-out score gained {min(VH[:-A.stop_evals]) - min(VH[-A.stop_evals:]):.4f} over the last {A.stop_evals} evaluations (< {A.stop_plateau}) - done"); break

    if A.ckpt_every > 0 and time.time() - last_ck > A.ckpt_every * 60: save_resume(); last_ck = time.time(); out(f"[resume] saved at step {step}")
v1 = val_bpc(); out(f"[val] after: {v1:.3f} bits per assistant character (before {v0:.3f}); {step} updates, {(time.time() - T0) / 60:.0f} min")
p_ = os.path.join(A.out, "model_final.pt")
torch.save({"model": model.state_dict(), "layers": CK["layers"], "args": BA, "vocab": V, "chars": CHARS, "special": {"UNK": UNK, "EOD": EOD, "USER": USR, "ASSISTANT": AST}}, p_); hub_put(p_, "model_final.pt")
if os.path.exists(RESUME): os.remove(RESUME)   # the final model is saved: the checkpoint has done its job
res = {"val_before": v0, "val_after": v1, "steps": step, "replies": {}}
for p in PROMPTS:
    r = reply(p); res["replies"][p] = r; out(f"[reply] {json.dumps(p)} -> {json.dumps(r)}")

# ---- GPTQ 4-bit (the app's grid), every linear layer; the tables fp16 --------------------------------------------------
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
    model.eval(); calib = torch.from_numpy(np.stack([r[0][:-1] for r in pack(TRAIN)[:24]]).astype(np.int64)).to(DEV); PACK = {}
    with torch.no_grad():
        x = model.emb(calib) + model.pos(torch.arange(SEQ, device=DEV))
        for li, blk in enumerate(model.blocks):
            for n in ("qkv", "proj", "fc", "fc2"):
                lin = getattr(blk, n); H = torch.zeros(lin.in_features, lin.in_features, device=DEV); cnt = [0]
                def hook(mod, inp, o): z = inp[0].reshape(-1, inp[0].shape[-1]).float(); H.add_(z.T @ z); cnt[0] += z.shape[0]
                h = lin.register_forward_hook(hook); blk(x); h.remove()
                Q, c, s_, b_ = gptq_linear(lin.weight.data, H / cnt[0]); lin.weight.data = Q.to(lin.weight.dtype); PACK[f"blocks.{li}.{n}"] = (c.cpu(), s_.cpu(), b_.cpu())
            x = blk(x)
        z = model.ln_f(x).reshape(-1, D).float(); Q, c, s_, b_ = gptq_linear(model.head.weight.data, z.T @ z / z.shape[0]); model.head.weight.data = Q; PACK["head"] = (c.cpu(), s_.cpu(), b_.cpu())
    res["val_4bit"] = val_bpc(); out(f"[gptq] 4-bit: {res['val_4bit']:.3f} bits per assistant character (float {v1:.3f})")
    shifts = torch.arange(0, 32, 4, dtype=torch.int64); sdq = {}
    for k, v_ in model.state_dict().items():
        key = k[:-7] if k.endswith(".weight") else None
        if key in PACK:
            c, s_, b_ = PACK[key]; sdq[k] = ((c.to(torch.int64).reshape(c.shape[0], -1, 8) << shifts).sum(-1) & 0xFFFFFFFF).to(torch.int32); sdq[key + ".scales"] = s_.half(); sdq[key + ".biases"] = b_.half()
        else: sdq[k] = v_.detach().cpu().half()
    res["packed_bytes"] = sum(v_.numel() * v_.element_size() for v_ in sdq.values())
    p4 = os.path.join(A.out, "model_4bit.pt")
    torch.save({"state": sdq, "layers": CK["layers"], "args": BA, "vocab": V, "chars": CHARS, "special": {"UNK": UNK, "EOD": EOD, "USER": USR, "ASSISTANT": AST}, "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}}, p4); hub_put(p4, "model_4bit.pt")
    for p in PROMPTS[:2]: out(f"[reply 4-bit] {json.dumps(p)} -> {json.dumps(reply(p, 300))}")
    out(f"[gptq] packed {res['packed_bytes'] / 1e6:.1f} MB")
except Exception as e:
    out(f"[gptq] failed: {str(e)[:300]}")
json.dump(res, open(os.path.join(A.out, "result.json"), "w"), indent=1); hub_put(os.path.join(A.out, "result.json"), "result.json"); hub_put(LOGF, "train.log")
out("CHARLM_SFT_DONE")
