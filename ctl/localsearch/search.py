#!/usr/bin/env python3
"""The local search: a query -> the article the model should read, from the store and its bit index.

Three stages, all cheap enough for a phone next to the 1.5B model:
  1. candidates by meaning: the query's bge vector, binarised, against every article's 48 bytes
     (the float query against every article's 48 bytes of signs, memory-mapped); the --coarse best are kept.
  2. candidates by name: the title index (SQLite FTS5 over titles.txt) for queries that name the thing.
  3. ranking: the candidates' opening text is re-embedded in float and scored by cosine against the
     float query vector, with a bonus when the query contains the title. The float vectors of the whole
     store are never stored; only these few dozen are computed, per query.

fetch(kw) returns "Title: text" in the shape the rollout loop serves, or "" when nothing is found, so it
drops into pool_eval/online_loop in place of the Wikipedia call.

  python3 search.py --store /root/wiki_store --model /root/bge-small "who designed the Welcome to Las Vegas sign"
"""
import argparse, os, re, sqlite3, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from store import Store  # noqa: E402

DIM = 384
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
UNPACK = (np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).astype(np.int8) * 2 - 1)   # byte -> 8 signs
UNPACK_T = None


class Embedder:
    def __init__(self, model_dir, threads=0, maxlen=160):
        from tokenizers import Tokenizer
        import onnxruntime as ort
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tok.enable_truncation(maxlen); self.tok.enable_padding(length=None)
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(os.path.join(model_dir, "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.names = [i.name for i in self.sess.get_inputs()]

    def __call__(self, texts):
        enc = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64); am = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": am}
        if "token_type_ids" in self.names:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self.sess.run(None, feed)[0][:, 0, :]
        return out / np.linalg.norm(out, axis=1, keepdims=True).clip(1e-6)


def build_title_index(store_path):
    """SQLite FTS5 over the titles, rowid = article id + 1. Built once; ~2% of the text store."""
    db = os.path.join(store_path, "titles.sqlite")
    if os.path.exists(db):
        return db
    con = sqlite3.connect(db + ".tmp")
    con.execute("CREATE VIRTUAL TABLE t USING fts5(title, tokenize='unicode61 remove_diacritics 2')")
    with open(os.path.join(store_path, "titles.txt"), encoding="utf-8") as f:
        con.executemany("INSERT INTO t(rowid, title) VALUES (?, ?)", ((i + 1, ln.rstrip("\n")) for i, ln in enumerate(f)))
    con.commit(); con.close()
    os.rename(db + ".tmp", db)
    return db


WORD = re.compile(r"[A-Za-z0-9]+")
STOP = set("the a an of in on at to for by with and or is was were are be been who what which when where how why did does do year".split())


def build_idf(store, n=40000):
    """Document frequencies over the titles and first 300 characters of a sample of articles: the weights of the
    lexical part of the ranking. Built once beside the index (~1 MB)."""
    import math, pickle
    path = os.path.join(store.path, "idf.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    import collections
    df = collections.Counter(); step = max(1, store.n_docs // n); cnt = 0
    for i in range(0, store.n_docs, step):
        t, b = store.doc(i); df.update(set(w.lower() for w in WORD.findall(t + " " + b[:300]))); cnt += 1
    idf = {w: math.log((cnt + 1) / (c + 1)) + 1 for w, c in df.items() if c > 1}
    pickle.dump((idf, math.log(cnt + 1) + 1), open(path, "wb"))
    return idf, math.log(cnt + 1) + 1


class LocalSearch:
    def __init__(self, store_path, model_dir, threads=0, coarse=128, title_k=24, rerank=128):
        self.st = Store(store_path)
        self.emb = Embedder(model_dir, threads)
        self.bits = np.memmap(os.path.join(store_path, "emb.bin"), dtype=np.uint8, mode="r").reshape(-1, DIM // 8)
        assert self.bits.shape[0] == self.st.n_docs, f"index {self.bits.shape[0]} vs store {self.st.n_docs}"
        self.gpu = None
        if os.environ.get("SP_LOCAL_GPU", "0") == "1":   # on the training box the whole index sits on the card (300 MB)
            import torch
            global UNPACK_T
            self.gpu = torch.from_numpy(np.ascontiguousarray(self.bits)).cuda()
            UNPACK_T = torch.from_numpy(UNPACK).cuda()
        self.con = sqlite3.connect("file:" + build_title_index(store_path) + "?mode=ro", uri=True)
        self.idf, self.idf_max = build_idf(self.st)
        self.coarse, self.title_k, self.rerank = coarse, title_k, rerank

    def _coarse_top(self, qv, k):
        """Asymmetric scoring: the float query against each article's signs. On shard 0 of the dump this finds the
        page Wikipedia's search returned 73% of the time within 96 candidates; binary-against-binary Hamming
        found it 23% of the time, so the query is never binarised."""
        if self.gpu is not None:
            import torch
            q = torch.from_numpy(qv.astype(np.float32)).to(self.gpu.device)
            sc = torch.empty(self.gpu.shape[0], device=self.gpu.device)
            CH = 1_000_000
            for s0 in range(0, self.gpu.shape[0], CH):
                pm = UNPACK_T[self.gpu[s0:s0 + CH].long()].reshape(-1, DIM)      # ±1 as int8
                sc[s0:s0 + CH] = pm.float() @ q
            return torch.topk(sc, k).indices.cpu().numpy()
        sc = np.empty(self.bits.shape[0], dtype=np.float32); CH = 100_000; q = qv.astype(np.float32)
        for s0 in range(0, self.bits.shape[0], CH):
            pm = UNPACK[self.bits[s0:s0 + CH]].reshape(-1, DIM)                    # ±1 as int8, one chunk at a time
            sc[s0:s0 + CH] = pm @ q
        idx = np.argpartition(-sc, k)[:k]
        return idx[np.argsort(-sc[idx])]

    def _title_hits(self, query, k):
        words = [w for w in WORD.findall(query) if w.lower() not in STOP and len(w) > 1]
        if not words:
            return []
        out = []
        # every term, then any term, ranked by FTS5's bm25 - the exact name comes first when it exists
        for q in (" AND ".join(f'"{w}"' for w in words), " OR ".join(f'"{w}"' for w in words)):
            try:
                rows = self.con.execute("SELECT rowid FROM t WHERE t MATCH ? ORDER BY bm25(t) LIMIT ?", (q, k)).fetchall()
            except sqlite3.OperationalError:
                rows = []
            out += [r[0] - 1 for r in rows]
            if len(out) >= k:
                break
        return list(dict.fromkeys(out))[:k]

    def search(self, query, k=3):
        qv = self.emb([QUERY_PREFIX + query])[0]
        cands = list(self._coarse_top(qv, self.coarse)[:self.rerank]) + self._title_hits(query, self.title_k)
        cands = list(dict.fromkeys(int(c) for c in cands))
        docs = [self.st.doc(i) for i in cands]
        dv = self.emb([f"{t}. {b[:800]}" for t, b in docs])
        sims = dv @ qv
        ql = " " + re.sub(r"[^a-z0-9 ]", " ", query.lower()) + " "
        qw = [w.lower() for w in WORD.findall(query) if w.lower() not in STOP and len(w) > 1]
        W = sum(self.idf.get(w, self.idf_max) for w in qw) or 1.0
        scored = []
        for (t, b), s, i in zip(docs, sims, cands):
            tl = " " + re.sub(r"[^a-z0-9 ]", " ", t.lower()).strip() + " "
            full = 0.1 if (len(tl.strip()) > 2 and tl in ql) else 0.0   # the query names the article
            tw = set(w.lower() for w in WORD.findall(t)); bw = set(w.lower() for w in WORD.findall(b[:600]))
            ft = sum(self.idf.get(w, self.idf_max) for w in qw if w in tw) / W        # query terms in the title
            fb = sum(self.idf.get(w, self.idf_max) for w in qw if w in tw or w in bw) / W   # ... or in the opening
            # cosine plus the lexical overlap the model's keyword queries were shaped by (weights from the shard-0 grid)
            scored.append((float(s) + 0.4 * ft + 0.2 * fb + full, i, t, b))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]

    def fetch(self, kw, k=1):
        hits = self.search(kw, k)
        return "\n\n".join(f"{t}: {b}" for _, _, t, b in hits) if hits else ""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=3); ap.add_argument("query", nargs="+")
    A = ap.parse_args()
    t0 = time.time(); ls = LocalSearch(A.store, A.model); print(f"[load] {ls.st.n_docs} articles, {time.time()-t0:.1f}s")
    for q in A.query:
        t0 = time.time(); hits = ls.search(q, A.k)
        print(f"\n== {q}  ({(time.time()-t0)*1000:.0f} ms)")
        for s, i, t, b in hits:
            print(f"  {s:.3f}  {t}: {b[:120]!r}")
