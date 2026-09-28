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
A = ap.parse_args()
ls = LocalSearch(A.store, A.model); T = json.load(open(A.queries))[:A.n]
t0 = time.time(); h1 = h3 = h10 = 0
for r in T:
    h = ls.search(r["query"], 10); ids = [i for _, i, _, _ in h]
    h1 += bool(ids) and ids[0] == r["doc"]; h3 += r["doc"] in ids[:3]; h10 += r["doc"] in ids
n = len(T)
print(f"RETR_TEST {A.tag} store={A.store} model={A.model} rerank={os.environ.get('SP_LOCAL_RERANK_MODEL', '-')}: "
      f"top1 {100*h1/n:.1f}% top3 {100*h3/n:.1f}% top10 {100*h10/n:.1f}% ({n}) {1000*(time.time()-t0)/n:.0f} ms/q", flush=True)
