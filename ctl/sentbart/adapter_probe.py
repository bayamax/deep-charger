#!/usr/bin/env python3
"""Can GPT-2 write the sentence the context BART means? (2026-10-10, the user's decoding plan, and how far the BART has to
go for it.) A small probe, CPU: an MLP turns a sentence vector into --n soft tokens in front of GPT-2 (frozen), the
previous sentences' text follows as a raw prefix, and GPT-2 is trained to write the sentence. The vector it is trained
on is the true one moved toward the BART's own prediction by a random amount, so it learns to write from vectors as far
off as the BART's. Then, on held-out positions, it writes from: no vector (GPT-2 alone), the BART's prediction, the
prediction moved toward the truth (--lams), and the true vector. Each written sentence is re-embedded (the BART's bge)
and scored: cosine to the true sentence, and top-1 among every sentence of its 32-document group. The texts are saved
for a judge (is it a sensible next sentence; does it say what the reference says).

  python3 adapter_probe.py --ckpt cap_bge14_bart16/model_best.pt --docs docs_001.jsonl --work probe/
"""
import argparse, json, math, os, random, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--docs", required=True); ap.add_argument("--work", required=True)
ap.add_argument("--train-docs", type=int, default=1024); ap.add_argument("--per-doc", type=int, default=8)
ap.add_argument("--eval-n", type=int, default=192); ap.add_argument("--group", type=int, default=32)
ap.add_argument("--n", type=int, default=8, help="soft tokens"); ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--ctx-tokens", type=int, default=160); ap.add_argument("--tgt-tokens", type=int, default=48)
ap.add_argument("--lams", default="0.15,0.3"); ap.add_argument("--gpt2", default="openai-community/gpt2")
ap.add_argument("--bge", default="BAAI/bge-small-en-v1.5"); ap.add_argument("--threads", type=int, default=3); ap.add_argument("--seed", type=int, default=5)
X = ap.parse_args()
torch.set_num_threads(X.threads); torch.manual_seed(0); os.makedirs(X.work, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], "--ckpt", "x", "--docs", "x", "--out", "x"]   # gpt2_rerank's loaders, without running it
src = open(os.path.join(HERE, "gpt2_rerank.py")).read(); ns = {"__name__": "probe", "__file__": os.path.join(HERE, "gpt2_rerank.py")}
exec(src[:src.index("# ---- the documents")], ns)
build, grow_bge, embed = ns["build"], ns["grow_bge"], ns["embed"]

CK = torch.load(X.ckpt, map_location="cpu", weights_only=False)
model, _ = build(CK["layers"]); model.load_state_dict(CK["model"]); model.eval()
enc = AutoModel.from_pretrained(X.bge); grow_bge(enc, CK.get("enc_layers", 12)); enc.load_state_dict(CK["enc"]); enc.eval()
del CK

# ---- documents: the evaluation ones exactly as gpt2_rerank.py drew them (same seed), the training ones from the rest ----
docs = []
for l in open(X.docs):
    if not l.strip(): continue
    d = json.loads(l); s = d["sents"][:128]; nq = int(d.get("nq", 0) or 0)
    if len(s) >= 12 and nq + 4 < len(s): docs.append((s, nq))
rng = random.Random(X.seed); rng.shuffle(docs)
ev_docs, tr_docs = docs[:X.eval_n], docs[2000:2000 + X.train_docs]


@torch.no_grad()
def predict(group, cuts):
    """the BART's tail predictions (teacher-forced, as dec_top1) for a group; returns the group's vectors and predictions"""
    flat = [s for d, _ in group for s in d]; V = embed(enc, flat); off = np.cumsum([0] + [len(d) for d, _ in group])
    Lm = max(len(d) for d, _ in group); x = torch.zeros(len(group), Lm, 384)
    valid = torch.zeros(len(group), Lm, dtype=torch.bool); masked = torch.zeros_like(valid)
    for b, (d, _) in enumerate(group):
        x[b, :len(d)] = V[off[b]:off[b + 1]]; valid[b, :len(d)] = True; masked[b, cuts[b]:len(d)] = True
    _, pd = model(x, valid, masked)
    return flat, V, off, pd


cache = os.path.join(X.work, "pairs.pt")
if os.path.exists(cache):
    P = torch.load(cache, weights_only=False)
else:
    t0 = time.time(); P = {"train": [], "eval": []}
    for split, dd in (("eval", ev_docs), ("train", tr_docs)):
        r2 = random.Random(X.seed + (0 if split == "eval" else 1))
        for g in range(0, len(dd), X.group):
            grp = dd[g:g + X.group]
            cuts = [r2.randint(max(nq + 1, len(s) // 4), max(nq + 1, 3 * len(s) // 4)) for s, nq in grp]
            flat, V, off, pd = predict(grp, cuts)
            for b, (s, nq) in enumerate(grp):
                ts = [r2.randint(cuts[b], len(s) - 1)] if split == "eval" else r2.sample(range(cuts[b], len(s)), min(X.per_doc, len(s) - cuts[b]))
                for t in ts:
                    rec = {"ctx": s[:t], "tgt": s[t], "true": V[off[b] + t].clone(), "pred": pd[b, t].float().clone()}
                    if split == "eval": rec.update({"group": g // X.group, "gi": int(off[b] + t), "seen": (int(off[b]), int(off[b] + t))})
                    P[split].append(rec)
            if split == "eval": P.setdefault("pools", []).append({"texts": flat, "V": V})
            print(f"[pairs] {split} group {g // X.group + 1}/{math.ceil(len(dd) / X.group)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    torch.save(P, cache)
cosp = float(np.mean([float(F.normalize(r["pred"], dim=-1) @ r["true"]) for r in P["eval"]]))
print(f"[pairs] {len(P['train'])} training pairs, {len(P['eval'])} evaluation positions; the BART's prediction at cos {cosp:.3f} to the truth", flush=True)

# ---- GPT-2 and the adapter ----
gtok = AutoTokenizer.from_pretrained(X.gpt2); gpt = AutoModelForCausalLM.from_pretrained(X.gpt2)
for p in gpt.parameters(): p.requires_grad_(False)
gpt.eval(); H = gpt.config.n_embd; NL = gtok.encode("\n")
adapter = nn.Sequential(nn.Linear(384, 1024), nn.GELU(), nn.Linear(1024, X.n * H))
WTE = gpt.get_input_embeddings()


def mix(true, pred, lam):
    return F.normalize(lam * true + (1 - lam) * F.normalize(pred, dim=-1), dim=-1)


def inputs(vecs, ctxs, tgts=None):
    """[soft tokens | previous sentences' text, newline | target, newline] as embeddings, left-padded; labels on the target"""
    rows = []
    for v, c, t in zip(vecs, ctxs, tgts or [None] * len(ctxs)):
        ci = gtok.encode("\n".join(c) + "\n")[-X.ctx_tokens:]; ti = (gtok.encode(t)[:X.tgt_tokens] + NL) if t is not None else []
        rows.append((v, ci, ti))
    M = max(len(c) + len(t) for _, c, t in rows) + (X.n if vecs[0] is not None else 0)
    E = torch.zeros(len(rows), M, H); att = torch.zeros(len(rows), M, dtype=torch.long); lab = torch.full((len(rows), M), -100)
    for r, (v, c, t) in enumerate(rows):
        ids = torch.tensor(c + t); e = WTE(ids)
        if v is not None: e = torch.cat([adapter(v[None]).view(X.n, H), e])
        E[r, M - len(e):] = e; att[r, M - len(e):] = 1
        if t: lab[r, M - len(t):] = torch.tensor(t)
    return E, att, lab


ack = os.path.join(X.work, "adapter.pt")
if os.path.exists(ack):
    adapter.load_state_dict(torch.load(ack))
else:
    opt = torch.optim.AdamW(adapter.parameters(), lr=X.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / X.steps)))
    tr = P["train"]; r3 = random.Random(1); t0 = time.time(); acc = []
    for step in range(1, X.steps + 1):
        bt = r3.sample(tr, X.batch)
        vecs = [mix(r["true"], r["pred"], 1.0 if r3.random() < 0.25 else r3.random()) for r in bt]   # a quarter the true vector, the rest anywhere between
        E, att, lab = inputs(vecs, [r["ctx"] for r in bt], [r["tgt"] for r in bt])
        out = gpt(inputs_embeds=E, attention_mask=att, position_ids=(att.cumsum(1) - 1).clamp(min=0), labels=lab)
        opt.zero_grad(); out.loss.backward(); torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0); opt.step(); sched.step(); acc.append(float(out.loss))
        if step % 50 == 0:
            print(f"[train] step {step} loss {np.mean(acc):.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True); acc = []
    torch.save(adapter.state_dict(), ack)


@torch.no_grad()
def write(vecs, ctxs, maxnew=40):
    E, att, _ = inputs(vecs, ctxs); out = [[] for _ in ctxs]; done = [False] * len(ctxs)
    pos = (att.cumsum(1) - 1).clamp(min=0)   # left padding: positions count the real tokens only, as in training
    o = gpt(inputs_embeds=E, attention_mask=att, position_ids=pos, use_cache=True); past = o.past_key_values; lg = o.logits[:, -1]; pos = pos[:, -1:]
    for _ in range(maxnew):
        nxt = lg.argmax(-1)
        for r in range(len(ctxs)):
            if not done[r]:
                if int(nxt[r]) in NL or int(nxt[r]) == gtok.eos_token_id: done[r] = True
                else: out[r].append(int(nxt[r]))
        if all(done): break
        att = torch.cat([att, torch.ones(len(ctxs), 1, dtype=torch.long)], 1)
        pos = pos + 1
        o = gpt(input_ids=nxt[:, None], past_key_values=past, attention_mask=att, position_ids=pos, use_cache=True); past = o.past_key_values; lg = o.logits[:, -1]
    return [gtok.decode(x).strip() for x in out]


adapter.eval(); EV = P["eval"]; conds = [("none", None), ("bart", 0.0)] + [(f"lam{l}", float(l)) for l in X.lams.split(",")] + [("true", 1.0)]
res = {"args": vars(X), "cos_pred": cosp, "conds": {}}; texts = [{"ctx_tail": r["ctx"][-6:], "ref": r["tgt"]} for r in EV]
for name, lam in conds:
    gen = []
    for i in range(0, len(EV), 16):
        part = EV[i:i + 16]
        vecs = [None] * len(part) if lam is None else [mix(r["true"], r["pred"], lam) for r in part]
        gen += write(vecs, [r["ctx"] for r in part])
    G = embed(enc, [g or "." for g in gen]); cs, t1 = [], 0
    for r, gv in zip(EV, G):
        cs.append(float(gv @ r["true"])); pool = P["pools"][r["group"]]; sc = pool["V"] @ gv
        sc[r["seen"][0]:r["seen"][1]] = -1e9; t1 += int(sc.argmax()) == r["gi"]
    vin = None if lam is None else float(np.mean([float(mix(r["true"], r["pred"], lam) @ r["true"]) for r in EV]))
    res["conds"][name] = {"vector_cos_in": vin, "written_cos": round(float(np.mean(cs)), 3), "written_top1": round(t1 / len(EV), 3)}
    for k, g in enumerate(gen): texts[k][name] = g
    print(f"[write] {name}: vector in at cos {vin if vin is None else round(vin, 3)} -> written sentence at cos {np.mean(cs):.3f} to the truth, top-1 {t1 / len(EV):.3f}", flush=True)
json.dump(res, open(os.path.join(X.work, "result.json"), "w"), indent=1)
open(os.path.join(X.work, "written.jsonl"), "w").write("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in texts))
print("PROBE_DONE", flush=True)
