#!/usr/bin/env python3
"""The retriever trained against the frozen model's verdicts (pool_eval --force rows): for each question, the
candidate pages it was served and whether it then answered right. Group-relative policy gradient on the
retriever's softmax over a question's candidates (GRPO with the model frozen): advantage = reward - the
question's mean reward, loss = -sum(advantage * log p(page)), p from the ranking score = cosine(bge, trained
here) + the fixed lexical terms search.py uses. Questions are split 80/20 for a held-out read; the
retriever is then judged on the 400 test queries by test_retriever.py and end to end.

  python3 train_ranker.py --rows /root/work/rew1_out_0.jsonl --base /root/bge-small --store /root/wiki_store --out /root/bge_rl1
"""
import argparse, json, os, re, sys, random, time, collections
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from store import Store
from search import QUERY_PREFIX, WORD, STOP, W_TITLE, W_BODY, W_FULL, build_idf
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True); ap.add_argument("--base", default="/tmp/bge_base"); ap.add_argument("--store", default="/tmp/wstore")
ap.add_argument("--questions", default=os.path.join(D, "reward_questions.jsonl")); ap.add_argument("--out", default="/tmp/bge_rl1")
ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--temp", type=float, default=0.05)
ap.add_argument("--qlen", type=int, default=48); ap.add_argument("--dlen", type=int, default=96); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="auto"); ap.add_argument("--holdout", type=float, default=0.2)
A = ap.parse_args()
random.seed(A.seed); torch.manual_seed(A.seed)
dev = ("cuda" if torch.cuda.is_available() else "cpu") if A.device == "auto" else A.device
import pickle
st = Store(A.store); IDF = os.path.join(A.store, "idf.pkl")
idf, idf_max = pickle.load(open(IDF, "rb")) if os.path.exists(IDF) else build_idf(st)
query_of = {}
for l in open(A.questions):
    r = json.loads(l); query_of[r["q"]] = r["query"]
groups = collections.defaultdict(dict)   # q -> {doc: [rewards]}
for l in open(A.rows):
    r = json.loads(l)
    if r.get("force") is None: continue
    groups[r["q"]].setdefault(int(r["force"]), []).append(float(bool(r.get("correct"))))
items = []
for q, d in groups.items():
    if q not in query_of or len(d) < 2: continue
    docs = sorted(d); rew = np.array([np.mean(d[i]) for i in docs], dtype=np.float32)
    if rew.max() == rew.min(): continue   # no signal in this group
    items.append((q, query_of[q], docs, rew))
random.shuffle(items); nh = int(len(items) * A.holdout); held, train = items[:nh], items[nh:]
print(f"[ranker] {len(groups)} questions with rows, {len(items)} with a signal; train {len(train)} held-out {len(held)}; device {dev}", flush=True)
tok = AutoTokenizer.from_pretrained(A.base); model = AutoModel.from_pretrained(A.base).to(dev)
def lex(query, docs):
    ql = " " + re.sub(r"[^a-z0-9 ]", " ", query.lower()) + " "
    qw = [w.lower() for w in WORD.findall(query) if w.lower() not in STOP and len(w) > 1]
    W = sum(idf.get(w, idf_max) for w in qw) or 1.0
    out = []
    for i in docs:
        t, b = st.doc(i)
        tl = " " + re.sub(r"[^a-z0-9 ]", " ", t.lower()).strip() + " "
        full = W_FULL if (len(tl.strip()) > 2 and tl in ql) else 0.0
        tw = set(w.lower() for w in WORD.findall(t)); bw = set(w.lower() for w in WORD.findall(b[:600]))
        ft = sum(idf.get(w, idf_max) for w in qw if w in tw) / W; fb = sum(idf.get(w, idf_max) for w in qw if w in tw or w in bw) / W
        out.append(W_TITLE * ft + W_BODY * fb + full)
    return torch.tensor(out, device=dev)
def enc(texts, n): return {k: v.to(dev) for k, v in tok(texts, padding=True, truncation=True, max_length=n, return_tensors="pt").items()}
def scores(query, docs):
    q = F.normalize(model(**enc([QUERY_PREFIX + query], A.qlen)).last_hidden_state[:, 0], dim=1)
    d = F.normalize(model(**enc([f"{st.doc(i)[0]}. {st.doc(i)[1][:400]}" for i in docs], A.dlen)).last_hidden_state[:, 0], dim=1)
    return (d @ q.T).squeeze(1) + lex(query, docs)
def evaluate(items):
    model.eval(); top = []; pick = []
    with torch.no_grad():
        for q, query, docs, rew in items:
            s = scores(query, docs).cpu().numpy(); j = int(np.argmax(s)); top.append(rew[j]); pick.append(rew.max())
    model.train()
    return 100 * float(np.mean(top)), 100 * float(np.mean(pick))
r0 = evaluate(held); r0t = evaluate(train)
print(f"[ranker] before: reward of the page ranked first - held-out {r0[0]:.1f}% (best possible {r0[1]:.1f}), train {r0t[0]:.1f}%", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=0.01); t0 = time.time(); step = 0
for ep in range(A.epochs):
    random.shuffle(train); tot = 0.0
    for q, query, docs, rew in train:
        s = scores(query, docs) / A.temp
        adv = torch.tensor(rew - rew.mean(), device=dev)
        loss = -(adv * F.log_softmax(s, 0)).sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1; tot += loss.item()
    r = evaluate(held); rt = evaluate(train)
    print(f"[ranker] epoch {ep+1}: loss {tot/len(train):.3f} | first-ranked page's reward: held-out {r[0]:.1f}% train {rt[0]:.1f}% {time.time()-t0:.0f}s", flush=True)
model.eval(); os.makedirs(A.out, exist_ok=True); model.cpu().save_pretrained(A.out); tok.save_pretrained(A.out)
import shutil
for f in ("tokenizer.json",):
    if not os.path.exists(os.path.join(A.out, f)): shutil.copy(os.path.join(A.base, f), A.out)
print(f"RANKER_DONE {A.out} held-out {r0[0]:.1f} -> {r[0]:.1f} (best possible {r0[1]:.1f})", flush=True)
