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


def pack(src, dst=None):
    dst = dst or src
    off = np.load(os.path.join(src, "term_offsets.npy")); post = np.memmap(os.path.join(src, "postings.bin"), dtype=np.uint32, mode="r")
    V = len(off) - 1; voff = np.zeros(V + 1, dtype=np.int64)
    out = open(os.path.join(dst, "postings_vb.bin"), "wb"); pos = 0
    for t in range(V):
        a, b = int(off[t]), int(off[t + 1])
        if b > a:
            ids = np.asarray(post[a:b], dtype=np.int64); d = np.diff(ids, prepend=-1) - 1 + (ids[:1] * 0)   # first as is, then gaps - 1
            d[0] = ids[0]
            buf = bytearray()
            for x in d.tolist():
                while x >= 0x80:
                    buf.append((x & 0x7F) | 0x80); x >>= 7
                buf.append(x)
            out.write(buf); pos += len(buf)
        voff[t + 1] = pos
        if t % 200000 == 0:
            print(f"[pack] {t}/{V} terms, {pos/1e9:.2f} GB", flush=True)
    out.close(); np.save(os.path.join(dst, "term_voffsets.npy"), voff)
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
        self.terms = {}
        for i, t in enumerate(open(os.path.join(path, "terms.txt"), encoding="utf-8")):
            self.terms[t.rstrip("\n")] = i
        self.df = np.load(os.path.join(path, "term_df.npy"))
        if os.path.exists(os.path.join(path, "postings_vb.bin")):
            self.voff = np.load(os.path.join(path, "term_voffsets.npy")); self.vb = np.memmap(os.path.join(path, "postings_vb.bin"), dtype=np.uint8, mode="r"); self.raw = None
        else:
            self.off = np.load(os.path.join(path, "term_offsets.npy")); self.raw = np.memmap(os.path.join(path, "postings.bin"), dtype=np.uint32, mode="r")
        self.n_docs = None

    def has(self, term):
        return term in self.terms

    def docs(self, term):
        """The ids of the articles whose text mentions the term anywhere (sorted)."""
        t = self.terms.get(term)
        if t is None:
            return np.zeros(0, dtype=np.int64)
        if self.raw is not None:
            return np.asarray(self.raw[int(self.off[t]):int(self.off[t + 1])], dtype=np.int64)
        a, b = int(self.voff[t]), int(self.voff[t + 1])
        return _decode(bytes(self.vb[a:b])) if b > a else np.zeros(0, dtype=np.int64)

    def df_of(self, term):
        t = self.terms.get(term)
        return int(self.df[t]) if t is not None else 0


if __name__ == "__main__":
    pack(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
