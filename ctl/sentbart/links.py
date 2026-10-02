#!/usr/bin/env python3
"""The link graph between the training documents, from cgscsystems/wikipedia-ordered-links (nlink_sequences: each
page's body links in order, as page ids).

Global document index = the order train.py builds `docs` in: the training shards (every vec_XXX.npy but --eval-shard,
sorted), each shard's documents in file order; ids_XXX.npy (prep.py --ids-only) gives each one's page id and must
match off_XXX.npy's count. Writes links.npz: CSR over global indices, undirected (a link either way), with `pos` =
the link's position in the linking page's body (the smaller of the two directions) - early links are the ones the
article is about.

  usage: python3 links.py --data /root/sb/data/docs --eval-shard 001 --seq nlink_sequences.parquet --out links.npz
"""
import argparse, glob, os, time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--eval-shard", default="001")
ap.add_argument("--seq", required=True); ap.add_argument("--out", required=True)
A = ap.parse_args()
ids = []
for vf in sorted(glob.glob(os.path.join(A.data, "vec_*.npy"))):
    k = os.path.basename(vf)[4:-4]
    if k == A.eval_shard: continue
    n = len(np.load(os.path.join(A.data, f"off_{k}.npy"))) - 1
    i = np.load(os.path.join(A.data, f"ids_{k}.npy")); assert len(i) == n, (k, len(i), n)
    ids.append(i)
ids = np.concatenate(ids); N = len(ids)
order = np.argsort(ids); sid = ids[order]
def glob_of(pid):                                   # page ids -> global index (-1: not a training document)
    j = np.searchsorted(sid, pid); j = np.clip(j, 0, N - 1); hit = sid[j] == pid
    return np.where(hit, order[j], -1)
print(f"[links] {N} training documents", flush=True)
t0 = time.time(); U, V, P = [], [], []
pf = pq.ParquetFile(A.seq)
for g in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(g, columns=["page_id", "link_sequence"])
    src = glob_of(t["page_id"].to_numpy().astype(np.int64))
    m = src >= 0
    if not m.any(): continue
    t = t.filter(pa.array(m)); src = src[m]
    ls = t["link_sequence"].combine_chunks()
    flat = pc.list_flatten(ls).to_numpy(zero_copy_only=False).astype(np.int64)
    par = pc.list_parent_indices(ls).to_numpy(zero_copy_only=False)
    offs = ls.offsets.to_numpy()[:-1]
    pos = np.arange(len(flat)) - offs[par]
    dst = glob_of(flat); k = (dst >= 0) & (dst != src[par])
    U.append(src[par][k]); V.append(dst[k]); P.append(pos[k].astype(np.int32))
    if g % 10 == 0: print(f"[links] row group {g}/{pf.metadata.num_row_groups}: {sum(len(u) for u in U)} in-corpus links ({time.time() - t0:.0f}s)", flush=True)
u = np.concatenate(U); v = np.concatenate(V); p = np.concatenate(P)
u, v = np.concatenate([u, v]), np.concatenate([v, u]); p = np.concatenate([p, p])
o = np.lexsort((p, v, u)); u, v, p = u[o], v[o], p[o]
first = np.ones(len(u), bool); first[1:] = (u[1:] != u[:-1]) | (v[1:] != v[:-1])
u, v, p = u[first], v[first], p[first]
ptr = np.zeros(N + 1, np.int64); np.add.at(ptr, u + 1, 1); ptr = np.cumsum(ptr)
np.savez(A.out, ptr=ptr, nbr=v.astype(np.int32), pos=p, ids=ids)
deg = np.diff(ptr)
print(f"[links] {len(u)} undirected-pair entries; degree: mean {deg.mean():.1f}, median {np.median(deg):.0f}, zero {100 * (deg == 0).mean():.1f}%; first-10 links {100 * (p < 10).mean():.0f}%", flush=True)
print("LINKS_DONE", flush=True)
