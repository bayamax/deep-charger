#!/usr/bin/env python3
"""What the search costs in memory and time on the whole store, as a process of its own.

Loads the search the way the device would (int8 ONNX embedder, IVF layout of the sign index, the lexical
index and the store memory-mapped), runs the held-out questions' first queries, and reports the process's
peak resident size and the time per query. The number to compare with the phone's budget is the RSS.

  python3 memcheck.py --store /root/wiki_store --model /root/bge-small --queries /root/work/ev_0.jsonl --n 40
"""
import argparse, json, os, resource, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--store", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--queries", required=True); ap.add_argument("--n", type=int, default=40)
A = ap.parse_args()

rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
from search import LocalSearch  # noqa: E402
t0 = time.time()
ls = LocalSearch(A.store, A.model)
t_load = time.time() - t0
rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
qs = [json.loads(l)["q"] for l in open(A.queries) if l.strip()][:A.n]
t0 = time.time(); found = 0
for q in qs:
    h = ls.search(q, 1); found += bool(h)
dt = (time.time() - t0) / max(len(qs), 1)
rss2 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
mode = f"ivf(nprobe {ls.ivf.nprobe})" if ls.ivf else "flat"
model = "int8" if "int8" in str(getattr(getattr(ls.emb, "sess", None), "_model_path", "")) else ("torch" if ls.emb.torch is not None else "fp32")
print(f"MEMCHECK store={ls.st.n_docs} articles {mode} embedder={model}: load {t_load:.1f}s, {dt*1000:.0f} ms/query over {len(qs)} queries, "
      f"peak RSS {rss2:.2f} GB (after import {rss0:.2f}, after load {rss1:.2f})", flush=True)
