#!/usr/bin/env python3
"""An inverted-file layout of the sign index, so a query touches a few megabytes of it instead of all 307.

The flat scan reads every article's 48 bytes for every query: fine on a box, but on the phone it keeps
the whole index resident. Here the articles are grouped by the nearest of K centroids (k-means on their
sign vectors as +-1), the sign index is rewritten in cluster order, and a query scores the K centroids
first (K x 384 floats, ~3 MB), then only the articles of its --nprobe best clusters. At K=2048 and 48
probes that is about 2% of the articles, ~7 MB of index read per query, and the recall against the flat
scan is measured by search.py's test sets.

Files written beside emb.bin: ivf_centroids.npy (K, 384) float32, ivf_order.npy (n,) int32 = article ids
in cluster order, ivf_offsets.npy (K+1,) int64 = where each cluster starts in that order, emb_ivf.bin =
emb.bin's rows in that order.

  python3 ivf.py --store /root/wiki_store --k 2048
"""
import argparse, os, sys, time

import numpy as np

DIM = 384
UNPACK = (np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).astype(np.int8) * 2 - 1)


def signs(bits):
    """(m, 48) uint8 -> (m, 384) int8 in {-1, +1}"""
    return UNPACK[bits].reshape(bits.shape[0], DIM)


def build(store, k, sample, iters, seed):
    bits = np.memmap(os.path.join(store, "emb.bin"), dtype=np.uint8, mode="r").reshape(-1, DIM // 8)
    n = bits.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, min(sample, n), replace=False))
    x = signs(np.asarray(bits[idx])).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    t0 = time.time()
    c = x[rng.choice(len(x), k, replace=False)].copy()
    for it in range(iters):
        a = np.concatenate([(x[s0:s0 + 50000] @ c.T).argmax(1) for s0 in range(0, len(x), 50000)])
        for j in range(k):
            m = a == j
            if m.any():
                v = x[m].mean(0); c[j] = v / (np.linalg.norm(v) + 1e-9)
            else:
                c[j] = x[rng.integers(len(x))]
        print(f"[ivf] k-means iteration {it + 1}/{iters} ({time.time()-t0:.0f}s)", flush=True)
    # assign every article
    assign = np.empty(n, dtype=np.int32); CH = 200_000
    for s0 in range(0, n, CH):
        s = signs(np.asarray(bits[s0:s0 + CH])).astype(np.float32)
        assign[s0:s0 + CH] = (s @ c.T).argmax(1)
    order = np.argsort(assign, kind="stable").astype(np.int32)
    counts = np.bincount(assign, minlength=k)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    np.save(os.path.join(store, "ivf_centroids.npy"), c.astype(np.float32))
    np.save(os.path.join(store, "ivf_order.npy"), order)
    np.save(os.path.join(store, "ivf_offsets.npy"), offsets)
    out = np.memmap(os.path.join(store, "emb_ivf.bin"), dtype=np.uint8, mode="w+", shape=(n, DIM // 8))
    for s0 in range(0, n, CH):
        out[s0:s0 + CH] = bits[order[s0:s0 + CH]]
    out.flush()
    print(f"[ivf] {n} articles in {k} clusters (largest {counts.max()}, empty {(counts == 0).sum()}) in {time.time()-t0:.0f}s", flush=True)
    print("IVF_DONE", flush=True)


class IVFIndex:
    def __init__(self, store, nprobe=48):
        self.c = np.load(os.path.join(store, "ivf_centroids.npy"))
        self.order = np.load(os.path.join(store, "ivf_order.npy"), mmap_mode="r")
        self.offsets = np.load(os.path.join(store, "ivf_offsets.npy"))
        self.bits = np.memmap(os.path.join(store, "emb_ivf.bin"), dtype=np.uint8, mode="r").reshape(-1, DIM // 8)
        self.n = self.bits.shape[0]; self.nprobe = nprobe

    def top(self, q, k):
        q = q.astype(np.float32)
        probes = np.argsort(-(self.c @ q))[:self.nprobe]
        rows, ids = [], []
        for p in probes:
            a, b = int(self.offsets[p]), int(self.offsets[p + 1])
            if b > a:
                rows.append(np.asarray(self.bits[a:b])); ids.append(np.asarray(self.order[a:b]))
        if not rows:
            return np.zeros(0, dtype=np.int64)
        rows = np.concatenate(rows); ids = np.concatenate(ids)
        sc = signs(rows).astype(np.float32) @ q
        kk = min(k, len(sc))
        idx = np.argpartition(-sc, kk - 1)[:kk]
        return ids[idx[np.argsort(-sc[idx])]].astype(np.int64)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True); ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--sample", type=int, default=300_000); ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    A = ap.parse_args()
    build(A.store, A.k, A.sample, A.iters, A.seed)
