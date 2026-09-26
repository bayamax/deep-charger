#!/usr/bin/env python3
"""Product quantization of the article vectors: 48 bytes an article, like the sign bits, but asymmetric.

The 384 dimensions are cut into 48 slices of 8; each slice has a codebook of 256 centroids learned by
k-means on a sample, and an article is the 48 centroid indices of its slices. A query is then scored
against every article by 48 table lookups: the query's inner product with each centroid of each slice is
computed once (48 x 256 floats), and an article's score is the sum of its 48 entries. Same storage as
the sign bits, but the query keeps its precision and the centroids keep most of the article's, which is
what recall at a few hundred candidates depends on.

Codes are stored slice-major (48 rows of n_docs bytes) so each lookup reads one contiguous row.

  python3 pq.py train  --store /root/wiki_store           # codebooks from a 200k sample of emb_f16.bin
  python3 pq.py encode --store /root/wiki_store           # pq_codes.bin for every article
"""
import argparse, os, sys, time

import numpy as np

DIM, M, K = 384, 48, 256
SUB = DIM // M


def kmeans(x, k, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        d = (x * x).sum(1)[:, None] - 2 * x @ c.T + (c * c).sum(1)[None, :]
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(0)
            else:
                c[j] = x[rng.integers(len(x))]
    return c


def train(store, sample=200_000, seed=0):
    f16 = np.memmap(os.path.join(store, "emb_f16.bin"), dtype=np.float16, mode="r").reshape(-1, DIM)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(f16.shape[0], min(sample, f16.shape[0]), replace=False))
    x = np.asarray(f16[idx], dtype=np.float32)
    t0 = time.time()
    books = np.stack([kmeans(x[:, m * SUB:(m + 1) * SUB], K, seed=seed + m) for m in range(M)])   # (M, K, SUB)
    np.save(os.path.join(store, "pq_codebook.npy"), books.astype(np.float32))
    print(f"[pq] codebooks from {len(x)} vectors in {time.time()-t0:.0f}s", flush=True)
    return books


def encode(store, chunk=200_000):
    books = np.load(os.path.join(store, "pq_codebook.npy"))
    f16 = np.memmap(os.path.join(store, "emb_f16.bin"), dtype=np.float16, mode="r").reshape(-1, DIM)
    n = f16.shape[0]
    codes = np.lib.format.open_memmap(os.path.join(store, "pq_codes.npy"), mode="w+", dtype=np.uint8, shape=(M, n))
    cc = (books * books).sum(2)   # (M, K)
    t0 = time.time()
    for s0 in range(0, n, chunk):
        x = np.asarray(f16[s0:s0 + chunk], dtype=np.float32)
        for m in range(M):
            xs = x[:, m * SUB:(m + 1) * SUB]
            d = cc[m][None, :] - 2 * xs @ books[m].T           # |x|^2 is constant per row
            codes[m, s0:s0 + len(x)] = d.argmin(1)
        print(f"[pq] {min(s0 + chunk, n)}/{n} ({time.time()-t0:.0f}s)", flush=True)
    codes.flush()
    print(f"PQ_DONE {n} articles, {M} bytes each", flush=True)


class PQIndex:
    def __init__(self, store, gpu=False):
        self.books = np.load(os.path.join(store, "pq_codebook.npy"))          # (M, K, SUB)
        self.codes = np.load(os.path.join(store, "pq_codes.npy"), mmap_mode="r")   # (M, n)
        self.n = self.codes.shape[1]
        self.gpu = None
        if gpu:
            import torch
            self.gpu = torch.from_numpy(np.ascontiguousarray(self.codes)).cuda().long()
            self.books_t = torch.from_numpy(self.books).cuda()

    def top(self, q, k):
        lut = np.einsum("mkd,md->mk", self.books, q.reshape(M, SUB).astype(np.float32))   # (M, K)
        if self.gpu is not None:
            import torch
            lut_t = torch.from_numpy(lut).cuda()
            sc = torch.zeros(self.n, device="cuda")
            for m in range(M):
                sc += lut_t[m][self.gpu[m]]
            return torch.topk(sc, k).indices.cpu().numpy()
        sc = np.zeros(self.n, dtype=np.float32)
        for m in range(M):
            sc += lut[m][self.codes[m]]
        idx = np.argpartition(-sc, k)[:k]
        return idx[np.argsort(-sc[idx])]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "encode"]); ap.add_argument("--store", required=True)
    ap.add_argument("--sample", type=int, default=200_000)
    A = ap.parse_args()
    if A.cmd == "train":
        train(A.store, A.sample)
    else:
        encode(A.store)
