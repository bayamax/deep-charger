#!/usr/bin/env python3
"""The whole-text term index at query time, and its packing.

Raw layout (build_terms.py): terms.txt, term_offsets.npy (V+1,) int64, postings.bin uint32 article ids in
increasing order per term, term_df.npy. Packed layout (pack_terms.py): postings_vb.bin, the same lists as
delta + LEB128 varints, term_voffsets.npy (V+1,) int64 byte offsets. TermIndex reads whichever is present
(packed first) and decodes one term's list on demand; memory is the memory-mapped file plus the vocabulary
(a dict of V strings, ~100 MB for a few million terms) - the phone would keep the vocabulary hashed.
"""
import os, sys
import numpy as np


def pack(src, dst=None, chunk=20_000_000):
    """Delta + LEB128 varint per term list, vectorised over chunks of whole terms. Written to a .part file and
    renamed at the end, so a partial file is never mistaken for a packed index."""
    dst = dst or src
    off = np.load(os.path.join(src, "term_offsets.npy")); post = np.memmap(os.path.join(src, "postings.bin"), dtype=np.uint32, mode="r")
    V = len(off) - 1; voff = np.zeros(V + 1, dtype=np.int64)
    part = os.path.join(dst, "postings_vb.bin.part"); out = open(part, "wb"); pos = 0; t = 0
    while t < V:
        t2 = int(np.searchsorted(off, off[t] + chunk, side="right")) - 1; t2 = min(max(t2, t + 1), V)
        a, b = int(off[t]), int(off[t2])
        ids = np.asarray(post[a:b], dtype=np.int64)
        d = np.empty_like(ids); d[1:] = ids[1:] - ids[:-1] - 1; d[:1] = ids[:1]
        starts = off[t:t2] - a; starts = starts[starts < len(ids)]; d[starts] = ids[starts]   # each term's first id as is
        nb = 1 + (d >= 1 << 7) + (d >= 1 << 14) + (d >= 1 << 21) + (d >= 1 << 28)
        tot = int(nb.sum()); buf = np.empty(tot, dtype=np.uint8)
        first = np.concatenate([[0], np.cumsum(nb)[:-1]])
        for k in range(5):
            m = nb > k
            byte = (d[m] >> (7 * k)) & 0x7F
            cont = (nb[m] > k + 1).astype(np.int64) << 7
            buf[first[m] + k] = (byte | cont).astype(np.uint8)
        out.write(buf.tobytes())
        ends = np.cumsum(nb)   # byte end of each value; a term's byte end is the end of its last value
        tend = off[t + 1:t2 + 1] - a   # value index one past each term's last
        voff[t + 1:t2 + 1] = pos + np.where(tend > 0, ends[np.maximum(tend - 1, 0)], 0)
        pos += tot; t = t2; nchunk = getattr(pack, "_n", 0) + 1; pack._n = nchunk
        if nchunk % 10 == 0 or t == V:
            print(f"[pack] {t}/{V} terms, {pos/1e9:.2f} GB", flush=True)
    out.close(); np.save(os.path.join(dst, "term_voffsets.npy"), voff); os.replace(part, os.path.join(dst, "postings_vb.bin"))
    print(f"PACK_DONE {V} terms {pos/1e9:.2f} GB (raw {int(off[-1])*4/1e9:.2f} GB)", flush=True)


def _decode(buf):
    out = []; x = 0; shift = 0; prev = -1
    for byte in buf:
        x |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
        else:
            prev = x if prev < 0 else prev + x + 1; out.append(prev); x = 0; shift = 0
    return np.array(out, dtype=np.int64)


class TermIndex:
    def __init__(self, path):
        self.path = path
        # the vocabulary stays on disk: terms.txt is sorted, a term is found by binary search over its line offsets
        # (term_lines.npy, built once), so memory holds neither a dict of millions of strings nor the file
        tf = os.path.join(path, "terms.txt"); lf = os.path.join(path, "term_lines.npy")
        if not os.path.exists(lf):
            raw = np.fromfile(tf, dtype=np.uint8); nl = np.nonzero(raw == 10)[0]
            np.save(lf, np.concatenate([[0], nl + 1]).astype(np.int64)); del raw
        self.lines = np.load(lf, mmap_mode="r"); self.tbytes = np.memmap(tf, dtype=np.uint8, mode="r")
        self.V = len(self.lines) - 1
        self.df = np.load(os.path.join(path, "term_df.npy"), mmap_mode="r")
        if os.path.exists(os.path.join(path, "postings_vb.bin")):
            self.voff = np.load(os.path.join(path, "term_voffsets.npy")); self.vb = np.memmap(os.path.join(path, "postings_vb.bin"), dtype=np.uint8, mode="r"); self.raw = None
        else:
            self.off = np.load(os.path.join(path, "term_offsets.npy")); self.raw = np.memmap(os.path.join(path, "postings.bin"), dtype=np.uint32, mode="r")
        self.n_docs = None

    def _term(self, i):
        return bytes(self.tbytes[int(self.lines[i]):int(self.lines[i + 1]) - 1])

    def lookup(self, term):
        key = term.encode("utf-8"); lo, hi = 0, self.V
        while lo < hi:
            mid = (lo + hi) // 2
            if self._term(mid) < key: lo = mid + 1
            else: hi = mid
        return lo if lo < self.V and self._term(lo) == key else None

    def has(self, term):
        return self.lookup(term) is not None

    def docs(self, term):
        """The ids of the articles whose text mentions the term anywhere (sorted)."""
        t = self.lookup(term)
        if t is None:
            return np.zeros(0, dtype=np.int64)
        if self.raw is not None:
            return np.asarray(self.raw[int(self.off[t]):int(self.off[t + 1])], dtype=np.int64)
        a, b = int(self.voff[t]), int(self.voff[t + 1])
        return _decode(bytes(self.vb[a:b])) if b > a else np.zeros(0, dtype=np.int64)

    def df_of(self, term):
        t = self.lookup(term)
        return int(self.df[t]) if t is not None else 0


if __name__ == "__main__":
    pack(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
