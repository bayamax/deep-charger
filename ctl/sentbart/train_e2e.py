#!/usr/bin/env python3
"""End to end: the sentence encoder (bge-small) trained together with the sentence-sequence BART.

The BART (train.py's model, losses and evaluation, unchanged) reads sentence vectors; here those vectors are computed
by bge-small inside the step, from the sentences' text, so every BART loss - masked-sentence infilling (encoder),
continuation (decoder) and sentence -> page retrieval - sends its gradient into bge too: the inputs AND the targets are
bge's outputs. Two things keep bge from collapsing into whatever makes the BART's job trivial:
  - the BART losses are contrastive already: each prediction must name its sentence among every sentence of the batch,
    so vectors cannot merge;
  - an anchor in bge's own terms (--w-anchor): each sentence's trained vector must find the FROZEN bge's vector of the
    same sentence among the frozen vectors of every sentence of the batch (InfoNCE, in-batch negatives) - bge keeps its
    space and its neighbourhoods and moves only where the BART gains.
Starts from a trained BART (--init, e.g. abl_one32 at 58.1% page top-1) and stock bge-small, so step 0 is that BART's
own score. Evaluation is train.py's: the same 2000 held-out documents of shard 001, embedded by the CURRENT bge each
time (index and query from one encoder, as in use); the page top-1 is comparable to the frozen-encoder runs.

  python3 train_e2e.py --vec /root/sb/data/docs --text /root/sb/data/e2e --eval-text /root/sb/data/docs/docs_001.jsonl \
      --init /root/sb/abl_one32/model_best.pt --out /root/sb/e2e1 --steps 30000
"""
import argparse, glob, json, os, random, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--vec", required=True, help="the embedded shards train.py reads (only shard 001's layout is used: which 2000 documents are evaluated)")
ap.add_argument("--text", required=True, help="training documents as text: docs_XXX.jsonl (prep.py)")
ap.add_argument("--eval-text", required=True, help="docs_001.jsonl, the held-out shard's text")
ap.add_argument("--init", default="", help="a BART (model_*.pt); an e2e checkpoint also carries its encoder (\"enc\"), which is loaded too")
ap.add_argument("--out", required=True)
ap.add_argument("--layers", type=int, default=8, help="BART layers per stack")
ap.add_argument("--grow", default="", help="instead of --init: an e2e checkpoint with fewer BART layers, grown to --layers (train.py --grow: the new layers start as the identity); its encoder is loaded too")
ap.add_argument("--enc-layers", type=int, default=0, help="bge layers: 0 = what the checkpoint has (stock: 12); more = the checkpoint's encoder grown by duplicating middle layers")
ap.add_argument("--page-queue", type=int, default=16, help="earlier batches' page vectors kept as extra negatives for the page loss")
ap.add_argument("--patience", type=int, default=0, help=">0: stop after this many evaluations in a row without a new best (the selection metric)")
ap.add_argument("--min-gain", type=float, default=0.002)
ap.add_argument("--pool", type=int, default=0, help=">0: also page retrieval among this many held-out articles (the 2000 + others of shard 001), one sentence hidden in each, the 2000's hidden sentences as queries")
ap.add_argument("--select", default="page_top1", help="the metric the best checkpoint and the patience follow (page_top1, or pool_top1 with --pool)")
ap.add_argument("--steps", type=int, default=30000); ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-5, help="the BART"); ap.add_argument("--lr-enc", type=float, default=1e-5, help="bge")
ap.add_argument("--w-anchor", type=float, default=1.0); ap.add_argument("--tau-anchor", type=float, default=0.05)
ap.add_argument("--anchor-q", type=int, default=0, help=">0: the anchor also on this many of the step's sentences in QUERY form (bge's query instruction in front) - the path real searches take, which the BART losses never see; without it e2e1 drifted there (app-query search fell)")
ap.add_argument("--maxlen", type=int, default=128, help="tokens per sentence into bge (embed.py's 128, so step 0 reads the stored vectors' equal)")
ap.add_argument("--eval-every", type=int, default=2000); ap.add_argument("--save-every", type=int, default=2000)
ap.add_argument("--bge", default="BAAI/bge-small-en-v1.5")
E = ap.parse_args()
os.makedirs(E.out, exist_ok=True)

# ---- train.py's definitions (data layout, model, losses, evaluation), its own arguments ----
TP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
src = open(TP).read(); cut = src.index("\nif A.search_eval:")
sys.argv = ["train.py", "--data", E.vec, "--out", E.out, "--eval-shard", "001", "--batch", str(E.batch), "--seq", "128", "--d", "512",
            "--layers", str(E.layers), "--heads", "8", "--ffn", "2048", "--lr", str(E.lr), "--warmup", "500", "--steps", str(max(1, E.steps)),
            "--page", "1", "--page-queue", str(E.page_queue), "--page-input", "one", "--skip-grad", "100"] + (["--grow", E.grow] if E.grow else ["--init", E.init])
ns = {"__name__": "train_e2e", "__file__": TP}
exec(compile(src[:cut], TP, "exec"), ns)
A, model, DEV, DIM = ns["A"], ns["model"], ns["DEV"], ns["DIM"]
import numpy as np, torch, torch.nn.functional as F  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402
assert DIM == 384, DIM

btok = AutoTokenizer.from_pretrained(E.bge)
enc = AutoModel.from_pretrained(E.bge)


def grow_bge(m, n):
    """duplicate middle layers until the encoder has n (deterministic, so a saved grown encoder rebuilds the same shape)"""
    import copy
    L = m.encoder.layer
    while len(L) < n:
        j = len(L) // 2; L.insert(j + 1, copy.deepcopy(L[j]))
    m.config.num_hidden_layers = len(L)


CK0 = torch.load(E.grow or E.init, map_location="cpu") if (E.grow or E.init) else {}
n0 = CK0.get("enc_layers", 12) if "enc" in CK0 else 12
grow_bge(enc, n0)
if "enc" in CK0: enc.load_state_dict(CK0["enc"]); print(f"[e2e] encoder from {E.grow or E.init} ({n0} layers)", flush=True)
if E.enc_layers > n0: grow_bge(enc, E.enc_layers); print(f"[e2e] encoder grown {n0} -> {E.enc_layers} layers (middle layers duplicated)", flush=True)
del CK0
enc = enc.to(DEV); enc.gradient_checkpointing_enable(); enc.train()
ref = AutoModel.from_pretrained(E.bge).to(DEV).eval()
for p in ref.parameters(): p.requires_grad_(False)


QPFX = "Represent this sentence for searching relevant passages: "


def embed(m, sents, chunk=256):
    """CLS, L2-normalised (as embed.py made the stored vectors); length-sorted chunks keep the padding small"""
    order = sorted(range(len(sents)), key=lambda i: len(sents[i])); parts = []
    for i in range(0, len(order), chunk):
        b = btok([sents[j] for j in order[i:i + chunk]], padding=True, truncation=True, max_length=E.maxlen, return_tensors="pt").to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            parts.append(F.normalize(m(**b).last_hidden_state[:, 0].float(), dim=-1))
    inv = torch.empty(len(order), dtype=torch.long, device=DEV); inv[torch.tensor(order, device=DEV)] = torch.arange(len(order), device=DEV)
    return torch.cat(parts)[inv]


# ---- training text: line offsets only, documents read on demand ----
class Lines:
    def __init__(s, path):
        s.path, s.off = path, []
        with open(path, "rb") as f:
            pos = 0
            for line in f: s.off.append(pos); pos += len(line)
        s.fh = open(path, "rb")
    def __len__(s): return len(s.off)
    def __getitem__(s, i): s.fh.seek(s.off[i]); return json.loads(s.fh.readline())


TR = [Lines(p) for p in sorted(glob.glob(os.path.join(E.text, "docs_*.jsonl"))) if not p.endswith("docs_001.jsonl")]
tdocs = [(i, j) for i, L in enumerate(TR) for j in range(len(L))]
print(f"[e2e] {len(tdocs)} training documents as text in {len(TR)} shards", flush=True)

# ---- the held-out 2000: the documents train.py evaluates, matched to their text by sentence count ----
eoff = ns["eval_sh"][2]; start = {int(a): d for d, a in enumerate(eoff[:-1])}
ET = Lines(E.eval_text); ev_text = []; bad = 0
for a, b in ns["edocs"]:
    d = ET[start[int(a)]]
    if len(d["sents"]) != b - a: bad += 1
    ev_text.append(d["sents"])
assert bad == 0, f"{bad} of {len(ev_text)} held-out documents do not match their text"
print(f"[e2e] {len(ev_text)} held-out documents matched to their text", flush=True)


@torch.no_grad()
def eval_now():
    """embed the held-out documents with the current bge, hand them to train.py's evaluate()"""
    enc.eval(); vs, off = [], [0]
    flat = [s for d in ev_text for s in d]
    for i in range(0, len(flat), 1024): vs.append(embed(enc, flat[i:i + 1024]).cpu().numpy())
    for d in ev_text: off.append(off[-1] + len(d))
    V = np.concatenate(vs).astype(np.float32)
    ns["eval_sh"] = ("001", V, np.array(off)); ns["edocs"] = [(off[k], off[k + 1]) for k in range(len(ev_text))]
    r = ns["evaluate"](); enc.train()
    if POOL: r.update(eval_pool())
    return r


POOL = []
if E.pool > len(ev_text):   # the other documents of the pool: shard 001's, not among the 2000, fixed
    used = {start[int(a)] for a, _ in ns["edocs"]}
    cand = [d for d in range(len(ET)) if d not in used]; random.Random(7).shuffle(cand)
    for d in cand:
        s_ = ET[d]["sents"]
        if len(s_) >= A.min_sents: POOL.append(s_[:A.seq])
        if len(POOL) >= E.pool - len(ev_text): break
    print(f"[e2e] pool: {len(ev_text)} + {len(POOL)} held-out articles", flush=True)


@torch.no_grad()
def eval_pool():
    """page retrieval among --pool articles: each with one sentence hidden (fixed draw), its page vector; the 2000
    evaluation articles' hidden sentences are the queries"""
    enc.eval(); model.eval()
    docs = [d[:A.seq] for d in ev_text] + POOL; rng = random.Random(55)
    hid = [rng.randint(1, len(d) - 1) for d in docs]
    P, Q = [], []
    for i in range(0, len(docs), 64):
        part = docs[i:i + 64]; flat = [s for d in part for s in d]
        v = torch.cat([embed(enc, flat[k:k + 1024]) for k in range(0, len(flat), 1024)])
        L = max(len(d) for d in part); x = torch.zeros(len(part), L, DIM, device=DEV)
        valid = torch.zeros(len(part), L, dtype=torch.bool, device=DEV); masked = torch.zeros_like(valid); o = 0
        for r, d in enumerate(part):
            x[r, :len(d)] = v[o:o + len(d)]; valid[r, :len(d)] = True; masked[r, hid[i + r]] = True
            if i + r < len(ev_text): Q.append(v[o + hid[i + r]])
            o += len(d)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.encode(x, valid, masked)
        P.append(model.last_page.float())
    P = torch.cat(P); Q = torch.stack(Q).float(); S = Q @ P.T
    rank = (S > S.diagonal()[:, None]).sum(1) + 1
    enc.train(); model.train()
    return {"pool_top1": float((rank <= 1).float().mean()), "pool_top10": float((rank <= 10).float().mean()), "pool_size": len(docs)}


opt = torch.optim.AdamW([{"params": [p for p in model.parameters()], "lr": E.lr},
                         {"params": [p for p in enc.parameters()], "lr": E.lr_enc}], betas=(0.9, 0.98), weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(1, E.steps)))))
STATE = os.path.join(E.out, "state_e2e.pt"); step0 = 0
if os.path.exists(STATE):
    st = torch.load(STATE, map_location=DEV); model.load_state_dict(st["model"]); enc.load_state_dict(st["enc"])
    opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); step0 = st["step"]


def save(tag, step):
    meta = {"enc_layers": len(enc.encoder.layer), "layers": E.layers, "step": step, "args": vars(E)}
    torch.save({"model": model.state_dict(), "enc": enc.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), **meta}, STATE + ".tmp")
    os.replace(STATE + ".tmp", STATE)
    torch.save({"model": model.state_dict(), "enc": enc.state_dict(), **meta}, os.path.join(E.out, f"model_{tag}.pt"))


log = open(os.path.join(E.out, "train.log"), "a")
def out(line): print(line, flush=True); log.write(line + "\n"); log.flush()


fmt = lambda r: " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items())
BEST = -1.0; STALE = 0
if step0 == 0:
    r = eval_now(); BEST = r[E.select]; out(f"[eval 0] {fmt(r)}"); save("best", 0); out(f"[best] step 0 {E.select} {BEST:.3f}")
model.train(); t0 = time.time(); acc = []
for step in range(step0 + 1, E.steps + 1) if E.steps > 0 else []:
    ns["STEP"][0] = step
    sents, items = [], []
    for _ in range(E.batch):
        i, j = random.choice(tdocs); s_ = TR[i][j]["sents"]
        while len(s_) < A.min_sents: i, j = random.choice(tdocs); s_ = TR[i][j]["sents"]
        if len(s_) > A.seq: k = random.randint(0, len(s_) - A.seq); s_ = s_[k:k + A.seq]
        items.append(np.zeros((len(s_), DIM), np.float32)); sents.append(s_)
    _, valid, masked, _ = ns["make_batch"](items)            # train.py's corruption (spans or a hidden tail)
    flat = [s for d in sents for s in d]
    v = embed(enc, flat)                                      # gradient into bge
    with torch.no_grad(): v0 = embed(ref, flat)
    x = torch.zeros(valid.shape + (DIM,), device=DEV, dtype=v.dtype); x[valid] = v   # row-major order = the documents' order
    with torch.autocast("cuda", dtype=torch.bfloat16):
        le, ld, _, _, _ = ns["losses"](x, valid, masked)
        la = F.cross_entropy(v @ v0.T / E.tau_anchor, torch.arange(len(v), device=DEV))
        if E.anchor_q > 0:
            qs_ = [QPFX + t for t in random.sample(flat, min(E.anchor_q, len(flat)))]
            vq = embed(enc, qs_)
            with torch.no_grad(): vq0 = embed(ref, qs_)
            la = la + F.cross_entropy(vq @ vq0.T / E.tau_anchor, torch.arange(len(vq), device=DEV))
        loss = A.w_enc * le + ld + A.w_page * ns["LP"][0] + E.w_anchor * la
    ns["LAST_COS"].clear()
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(enc.parameters()), 1.0))
    if gn < 100: opt.step()
    sched.step()
    acc.append((le.item(), ld.item(), float(ns["LP"][0]), la.item(), float((v * v0).sum(-1).mean())))
    if step % 100 == 0:
        e, d_, p_, a_, c_ = np.mean(acc, 0); acc = []
        out(f"[step {step}] enc {e:.3f} dec {d_:.3f} page {p_:.3f} anchor {a_:.3f} cos_to_stock {c_:.3f} |grad| {gn:.2f} {(time.time()-t0)/60:.0f} min")
    if step % E.eval_every == 0 or step == E.steps:
        r = eval_now(); out(f"[eval {step}] {fmt(r)}")
        if r[E.select] > BEST + E.min_gain: STALE = 0
        else: STALE += 1
        if r[E.select] > BEST: BEST = r[E.select]; save("best", step); out(f"[best] step {step} {E.select} {BEST:.3f}")
        if E.patience and STALE >= E.patience: out(f"[plateau] {STALE} evaluations without a gain of {E.min_gain}: stopping at {step}"); break
    if step % E.save_every == 0: save("latest", step)
out("TRAIN_DONE")
