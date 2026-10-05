#!/usr/bin/env python3
"""Article search with the app model's own queries, one vector per article, then a second stage inside the candidates.

Stage 1 (what a device would store): one vector per article of the held-out shard (all 118,398) - the BART's page
vector (the document's first --seq sentences, nothing hidden), or a no-model baseline (the lead sentence, the mean of
the sentences). The query is embedded by the same sentence encoder (bge's query instruction), cosine.
Stage 2 (what opening the candidates would do): the top --k articles of stage 1 are opened and every sentence in them
is compared with the query; an article scores its best sentence (+ its stage-1 score in the "sum" variant). Articles
stage 1 missed stay missed. Reported: final @1 / @10 and stage 1's recall at k.

A checkpoint that carries a trained encoder ("enc", from train_e2e.py) embeds every sentence of the shard and the
queries with THAT encoder; otherwise stock bge-small and the stored vectors.

  python3 search_cascade.py --vec /root/sb/data/docs --text /root/sb/data/docs/docs_001.jsonl --ckpt /root/sb/abl_one32/model_best.pt \
      --queries /root/sb/se2/dl/sentbart/searcheval/dcq.jsonl --out /root/sb/sc/one32.json
"""
import argparse, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--vec", required=True); ap.add_argument("--text", required=True); ap.add_argument("--ckpt", required=True)
ap.add_argument("--queries", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--k", default="10,50"); ap.add_argument("--bge", default="BAAI/bge-small-en-v1.5")
E = ap.parse_args()
TP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
src = open(TP).read(); cut = src.index("\nif A.search_eval:")
sys.argv = ["train.py", "--data", E.vec, "--out", os.path.dirname(E.out) or ".", "--eval-shard", "001", "--batch", "32", "--seq", "128", "--d", "512",
            "--layers", "8", "--heads", "8", "--ffn", "2048", "--page", "1", "--init", E.ckpt]
ns = {"__name__": "search_cascade", "__file__": TP}
exec(compile(src[:cut], TP, "exec"), ns)
model, DEV, DIM = ns["model"], ns["DEV"], ns["DIM"]
import numpy as np, torch, torch.nn.functional as F  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402
model.eval()
ck = torch.load(E.ckpt, map_location="cpu")
btok = AutoTokenizer.from_pretrained(E.bge); enc = AutoModel.from_pretrained(E.bge)
trained = "enc" in ck
if trained: enc.load_state_dict(ck["enc"])
enc = enc.to(DEV).eval()


@torch.no_grad()
def embed(sents, pfx="", chunk=512):
    order = sorted(range(len(sents)), key=lambda i: len(sents[i])); out = torch.empty(len(sents), DIM, dtype=torch.float16)
    for i in range(0, len(order), chunk):
        idx = order[i:i + chunk]
        b = btok([pfx + sents[j] for j in idx], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[idx] = F.normalize(enc(**b).last_hidden_state[:, 0].float(), dim=-1).half().cpu()
    return out


_, Vst, off = ns["eval_sh"]; ND = len(off) - 1
t0 = time.time()
if trained:   # every sentence of the shard again, by the trained encoder
    sents = []
    for line in open(E.text):
        sents.extend(json.loads(line)["sents"])
    assert len(sents) == off[-1], (len(sents), off[-1])
    V = embed(sents); del sents
else:
    V = torch.empty(int(off[-1]), DIM, dtype=torch.float16)
    for a in range(0, int(off[-1]), 1 << 20): V[a:a + (1 << 20)] = torch.from_numpy(np.asarray(Vst[a:a + (1 << 20)], dtype=np.float32)).half()
print(f"[cascade] {ND} articles, {V.shape[0]} sentences ({'trained' if trained else 'stock'} encoder) {(time.time()-t0)/60:.1f} min", flush=True)

# stage-1 vectors
P = {k: torch.zeros(ND, DIM) for k in ("page", "mean", "lead")}
with torch.no_grad():
    for i in range(0, ND, 64):
        ds = range(i, min(i + 64, ND)); items = [V[off[d]:min(off[d + 1], off[d] + 128)].float() for d in ds]
        L = max(len(t) for t in items); B = len(items)
        x = torch.zeros(B, L, DIM, device=DEV); valid = torch.zeros(B, L, dtype=torch.bool, device=DEV)
        for j, t in enumerate(items): x[j, :len(t)] = t.to(DEV); valid[j, :len(t)] = True
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.encode(x, valid, torch.zeros_like(valid))
        P["page"][i:i + B] = model.last_page.float().cpu()
        P["lead"][i:i + B] = x[:, 0].cpu()
    sid = torch.from_numpy(np.repeat(np.arange(ND), np.diff(off)))
    for a in range(0, V.shape[0], 1 << 20): P["mean"].index_add_(0, sid[a:a + (1 << 20)], V[a:a + (1 << 20)].float())
    P["mean"] = F.normalize(P["mean"], dim=-1)
P = {k: v.to(DEV) for k, v in P.items()}
print(f"[cascade] stage-1 vectors {(time.time()-t0)/60:.1f} min", flush=True)

qs = [json.loads(l) for l in open(E.queries) if l.strip()]
KS = [int(k) for k in E.k.split(",")]; res = {}
for qk in ("q_api", "q_good"):
    sub = [q for q in qs if q.get(qk)]
    if not sub: continue
    Q = embed([q[qk] for q in sub], "Represent this sentence for searching relevant passages: ").float().to(DEV)
    gold = torch.tensor([int(q["idx"]) for q in sub], device=DEV)
    for m, M in P.items():
        S = Q @ M.T; g = S.gather(1, gold[:, None]); rank = (S > g).sum(1) + 1
        r = {"@1": float((rank <= 1).float().mean()), "@10": float((rank <= 10).float().mean()), "@100": float((rank <= 100).float().mean())}
        for K in KS:
            top = S.topk(K, dim=1)
            hit = (top.indices == gold[:, None]).any(1)
            fin1 = fin10 = finS1 = 0
            for qi in range(len(sub)):
                cand = top.indices[qi].tolist(); s1 = top.values[qi]
                best = torch.stack([(V[off[d]:off[d + 1]].float().to(DEV) @ Q[qi]).max() for d in cand])
                for nm, sc in (("max", best), ("sum", best + s1)):
                    o = [cand[j] for j in sc.argsort(descending=True).tolist()]
                    g_ = int(gold[qi])
                    if nm == "max": fin1 += o[0] == g_; fin10 += g_ in o[:10]
                    else: finS1 += o[0] == g_
            n = len(sub)
            r[f"recall@{K}"] = float(hit.float().mean()); r[f"k{K}_final@1"] = fin1 / n; r[f"k{K}_final@10"] = fin10 / n; r[f"k{K}_sum@1"] = finS1 / n
        res[f"{qk}|{m}"] = r
        print(f"[cascade] {qk} {m:5s} " + " ".join(f"{a} {b:.3f}" for a, b in r.items()) + f"  (n {len(sub)})", flush=True)
json.dump(res, open(E.out, "w"), indent=1)
print("CASCADE_DONE", flush=True)
