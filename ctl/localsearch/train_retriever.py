#!/usr/bin/env python3
"""Fine-tune bge-small on the lineage's own (query, page Wikipedia's search served) pairs.

Contrastive: each query against its page's `title. opening` plus in-batch pages and hard negatives (the pages the
current search ranks near the right one). The query keeps bge's instruction prefix and the page text the same cut
the index uses, so the trained model drops into embed.py / search.py unchanged.
"""
import argparse, json, random, sys, time, os
import torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from store import Store
from search import QUERY_PREFIX

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="/tmp/bge_base"); ap.add_argument("--store", default="/tmp/wstore")
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ap.add_argument("--pairs", default=os.path.join(D, "train_pairs.json")); ap.add_argument("--negs", default=os.path.join(D, "train_negs.json"))
ap.add_argument("--out", default="/tmp/bge_ft"); ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--nneg", type=int, default=2); ap.add_argument("--temp", type=float, default=0.05)
ap.add_argument("--qlen", type=int, default=48); ap.add_argument("--dlen", type=int, default=128)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--device", default="auto")
A = ap.parse_args()
random.seed(A.seed); torch.manual_seed(A.seed)
dev = ("cuda" if torch.cuda.is_available() else "cpu") if A.device == "auto" else A.device
tok = AutoTokenizer.from_pretrained(A.base); model = AutoModel.from_pretrained(A.base).to(dev)
print(f"[retriever] {len(json.load(open(A.pairs)))} pairs, base {A.base}, device {dev}", flush=True)
st = Store(A.store)
pairs = json.load(open(A.pairs)); negs = json.load(open(A.negs)) if os.path.exists(A.negs) else {}
if A.limit: pairs = pairs[:A.limit]
gold_of = {}
for p in pairs: gold_of.setdefault(p["query"], set()).add(p["doc"])
cache = {}
def text(i):
    if i not in cache:
        t, b = st.doc(i); cache[i] = f"{t}. {b[:600]}"
    return cache[i]
def enc(texts, n):
    return {k: v.to(dev) for k, v in tok(texts, padding=True, truncation=True, max_length=n, return_tensors="pt").items()}
def embed(batch):
    out = model(**batch).last_hidden_state[:, 0]
    return F.normalize(out, dim=1)
opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=0.01)
steps = A.epochs * ((len(pairs) + A.bs - 1) // A.bs); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))
model.train(); t0 = time.time(); step = 0
for ep in range(A.epochs):
    random.shuffle(pairs)
    for b0 in range(0, len(pairs), A.bs):
        batch = pairs[b0:b0 + A.bs]
        # one page per batch slot; a query whose page is also another slot's positive is not penalised for it
        docs = [p["doc"] for p in batch]
        for p in batch:
            hard = [i for i in negs.get(p["query"], []) if i not in gold_of[p["query"]]][:6]
            docs += random.sample(hard, min(A.nneg, len(hard)))
        docs = list(dict.fromkeys(docs)); col = {d: j for j, d in enumerate(docs)}
        q = embed(enc([QUERY_PREFIX + p["query"] for p in batch], A.qlen))
        d = embed(enc([text(i) for i in docs], A.dlen))
        logits = q @ d.T / A.temp
        mask = torch.zeros_like(logits, dtype=torch.bool)
        for r, p in enumerate(batch):
            for g in gold_of[p["query"]]:
                if g in col and g != p["doc"]: mask[r, col[g]] = True
        logits = logits.masked_fill(mask, -1e4)
        target = torch.tensor([col[p["doc"]] for p in batch], device=dev)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
        if step % 10 == 0 or step == 1:
            acc = (logits.argmax(1) == target).float().mean().item()
            print(f"ep {ep} step {step}/{steps} loss {loss.item():.3f} in-batch acc {acc:.2f} docs {len(docs)} {time.time()-t0:.0f}s", flush=True)
model.eval(); os.makedirs(A.out, exist_ok=True)
model.cpu().save_pretrained(A.out); tok.save_pretrained(A.out)
import shutil
for f in ("tokenizer.json",):
    if not os.path.exists(os.path.join(A.out, f)): shutil.copy(os.path.join(A.base, f), A.out)
print(f"TRAIN_DONE {A.out} {steps} steps {time.time()-t0:.0f}s", flush=True)
