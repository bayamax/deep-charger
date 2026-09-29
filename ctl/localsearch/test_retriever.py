#!/usr/bin/env python3
"""The search on the 400 test queries (the model's own queries against the page Wikipedia's search served; their
questions are out of the retriever's training): does the page come first / in the top three.

  python3 test_retriever.py --store /root/wiki_store --model /root/bge-small [--n 400]
  SP_LOCAL_RERANK_MODEL=/root/bge_ft1 python3 test_retriever.py ...   (the trained model for the ranking only)
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search import LocalSearch
ap = argparse.ArgumentParser()
ap.add_argument("--store", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--queries", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test_queries.json"))
ap.add_argument("--n", type=int, default=400); ap.add_argument("--tag", default="")
ap.add_argument("--serve", type=int, default=0, help="1: also score the answer within the first 1000 characters of the text the search would serve (SP_LOCAL_PASSAGE applies)")
A = ap.parse_args()
ls = LocalSearch(A.store, A.model); T = json.load(open(A.queries))[:A.n]
import re
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()
def carries(t, b, gold): return bool(gold) and (" " + norm(gold) + " ") in (" " + norm(t + " " + b[:1600]) + " ")
from search import compose
t0 = time.time(); h1 = h3 = h10 = 0; g1 = g3 = 0; gw = 0; ng = 0; s1 = 0; sw = 0
for r in T:
    h = ls.search(r["query"], 10); ids = [i for _, i, _, _ in h]
    h1 += bool(ids) and ids[0] == r["doc"]; h3 += r["doc"] in ids[:3]; h10 += r["doc"] in ids
    if r.get("gold"):   # does the page served carry the question's answer (the API's page as the yardstick)
        ng += 1; ok = [carries(t, b, r["gold"]) for _, _, t, b in h[:3]]
        g1 += bool(ok) and ok[0]; g3 += any(ok)
        tw, bw = ls.st.doc(r["doc"]); gw += carries(tw, bw, r["gold"])
        if A.serve:
            _, _, t0_, b0_ = h[0]
            b0_ = compose(r["query"], b0_, ls.idf, ls.idf_max) if os.environ.get("SP_LOCAL_PASSAGE", "0") == "1" else b0_
            s1 += carries(t0_, b0_[:1000], r["gold"]); sw += carries(tw, bw[:1000], r["gold"])
n = len(T)
print(f"RETR_TEST {A.tag} store={A.store} model={A.model} rerank={os.environ.get('SP_LOCAL_RERANK_MODEL', '-')}: "
      f"same page as Wikipedia top1 {100*h1/n:.1f}% top3 {100*h3/n:.1f}% top10 {100*h10/n:.1f}% ({n}) | "
      f"page carries the answer: top1 {100*g1/max(ng,1):.1f}% top3 {100*g3/max(ng,1):.1f}% vs Wikipedia's page {100*gw/max(ng,1):.1f}% ({ng})"
      + (f" | answer in the first 1000 chars served: {100*s1/max(ng,1):.1f}% (Wikipedia's page's opening {100*sw/max(ng,1):.1f}%)" if A.serve else "")
      + f" | {1000*(time.time()-t0)/n:.0f} ms/q", flush=True)
