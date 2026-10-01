#!/usr/bin/env python3
"""Sentences -> vectors with a frozen sentence encoder (bge-small-en-v1.5: CLS pooling, L2-normalised, 384-d).

For each docs_XXX.jsonl shard writes, next to it:
  vec_XXX.npy   float16 [n_sentences, 384]  every sentence of every document, in document order
                (--int8 1: int8, with scl_XXX.npy float16 [n_sentences] the per-row max |x|; row = q * scl / 127)
  off_XXX.npy   int64   [n_docs + 1]        document d's sentences are rows off[d]:off[d+1]
Sentences are batched by length (sorted) so padding is small; the order on disk is the original one.

  usage: python3 embed.py --dir /root/sb/data/docs [--model BAAI/bge-small-en-v1.5] [--batch 1024] [--maxlen 128]
"""
import argparse, glob, json, os, time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--maxlen", type=int, default=128)
ap.add_argument("--int8", type=int, default=0, help="1: vec_XXX.npy as int8 with a per-row scale in scl_XXX.npy (row = q * scl / 127), half the disk")
A = ap.parse_args()
tok = AutoTokenizer.from_pretrained(A.model)
enc = AutoModel.from_pretrained(A.model, torch_dtype=torch.float16).cuda().eval()


@torch.no_grad()
def embed(sents):
    order = np.argsort([len(s) for s in sents])
    out = np.zeros((len(sents), enc.config.hidden_size), dtype=np.float16)
    for i in range(0, len(sents), A.batch):
        idx = order[i:i + A.batch]
        b = tok([sents[j] for j in idx], padding=True, truncation=True, max_length=A.maxlen, return_tensors="pt").to("cuda")
        h = enc(**b).last_hidden_state[:, 0]
        out[idx] = torch.nn.functional.normalize(h.float(), dim=-1).half().cpu().numpy()
    return out


for path in sorted(glob.glob(os.path.join(A.dir, "docs_*.jsonl"))):
    k = os.path.basename(path)[5:-6]
    vf, of = os.path.join(A.dir, f"vec_{k}.npy"), os.path.join(A.dir, f"off_{k}.npy")
    if os.path.exists(vf) and os.path.exists(of):
        print(f"[embed] {k} done already", flush=True); continue
    t0 = time.time(); sents, off = [], [0]
    for line in open(path):
        d = json.loads(line); sents.extend(d["sents"]); off.append(len(sents))
    v = embed(sents)
    if A.int8:
        v = v.astype(np.float32); scl = np.abs(v).max(1).clip(1e-6)
        np.save(os.path.join(A.dir, f"scl_{k}.npy"), scl.astype(np.float16))
        v = np.round(v / scl.astype(np.float16).astype(np.float32)[:, None] * 127).clip(-127, 127).astype(np.int8)
    np.save(vf + ".tmp.npy", v); os.replace(vf + ".tmp.npy", vf)
    np.save(of, np.array(off, dtype=np.int64))
    dt = time.time() - t0
    print(f"[embed] {k}: {len(off) - 1} documents, {len(sents)} sentences in {dt:.0f}s ({len(sents) / dt:.0f}/s)", flush=True)
print("EMBED_DONE", flush=True)
