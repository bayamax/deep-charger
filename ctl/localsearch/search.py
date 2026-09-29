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
import argparse, math, os, re, sqlite3, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from store import Store  # noqa: E402

DIM = 384
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
UNPACK = (np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).astype(np.int8) * 2 - 1)   # byte -> 8 signs
UNPACK_T = None
W_TITLE, W_BODY, W_BM25, W_FULL = 0.2, 0.4, 0.0, 0.2   # ranking weights from the whole-dump grid (200 of the model's queries: top-1 49%, top-3 52%)
W_DEEP = float(os.environ.get("SP_LOCAL_WDEEP", "0"))    # query terms found in the body beyond the opening (a store with more than the opening)


class Embedder:
    """bge-small: CLS-pooled, normalised. onnxruntime on the CPU (the device path); torch when it is not installed
    (the training box), on the GPU when there is one."""

    def __init__(self, model_dir, threads=0, maxlen=160):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tok.enable_truncation(maxlen); self.tok.enable_padding(length=None)
        self.torch = None
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = threads
            f = "model_int8.onnx" if os.path.exists(os.path.join(model_dir, "model_int8.onnx")) and os.environ.get("SP_LOCAL_FP32", "0") != "1" else "model.onnx"
            self.sess = ort.InferenceSession(os.path.join(model_dir, f), so, providers=["CPUExecutionProvider"])
            self.names = [i.name for i in self.sess.get_inputs()]
        except ImportError:
            import torch
            from transformers import AutoModel
            self.torch = torch
            self.dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float16 if self.dev == "cuda" else torch.float32).to(self.dev).eval()

    def __call__(self, texts, maxlen=None, batch=16):
        if maxlen:
            self.tok.enable_truncation(maxlen)
        if len(texts) > batch:   # a batch at a time: a 128-text batch through onnxruntime kept ~1 GB of arena
            out = [self(texts[i:i + batch], maxlen) for i in range(0, len(texts), batch)]
            return np.concatenate(out)
        enc = self.tok.encode_batch(texts)
        if self.torch is not None:
            with self.torch.no_grad():
                ids = self.torch.tensor([e.ids for e in enc], device=self.dev); am = self.torch.tensor([e.attention_mask for e in enc], device=self.dev)
                out = self.model(input_ids=ids, attention_mask=am).last_hidden_state[:, 0, :].float()
                return self.torch.nn.functional.normalize(out, dim=1).cpu().numpy()
        ids = np.array([e.ids for e in enc], dtype=np.int64); am = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": am}
        if "token_type_ids" in self.names:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self.sess.run(None, feed)[0][:, 0, :]
        return out / np.linalg.norm(out, axis=1, keepdims=True).clip(1e-6)


def build_lex_index(store_path):
    """SQLite FTS5 over each article's title and first 300 characters, contentless (the text lives in the store),
    rowid = article id + 1. The model's queries are keyword lists shaped by Wikipedia's own lexical search, and on
    shard 0 of the dump BM25 over this much text finds Wikipedia's page first 48% of the time against 29% for
    the embedding alone; the two channels together cover 79% within 128 candidates, the float embedding's own
    ceiling. About 1.8 GB for the whole dump. Built once."""
    db = os.path.join(store_path, "lex.sqlite")
    if os.path.exists(db):
        return db
    from store import Store
    st = Store(store_path)
    con = sqlite3.connect(db + ".tmp")
    con.execute("CREATE VIRTUAL TABLE d USING fts5(title, body, tokenize='unicode61 remove_diacritics 2', content='')")
    def rows():
        for i in range(st.n_docs):
            t, b = st.doc(i)
            yield (i + 1, t, b[:300])
    con.executemany("INSERT INTO d(rowid, title, body) VALUES (?, ?, ?)", rows())
    con.execute("INSERT INTO d(d) VALUES('optimize')")
    con.commit(); con.close()
    os.rename(db + ".tmp", db)
    return db


build_title_index = build_lex_index   # the name the box's corpus job calls at the end of its build

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


SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


def compose(query, body, idf, idf_max, head=500, passage=500):
    """What a store holding more than the opening serves: the article's opening (about `head` characters, whole
    sentences), then the `passage`-character window of the rest that carries most of the query's IDF-weighted
    terms, then everything else in order. The model reads the first 256 tokens (~1000 characters), so a fact
    deep in the article reaches it in the first block when the query points at it; <more> still walks the rest."""
    if len(body) <= head + passage:
        return body
    qw = set(w.lower() for w in WORD.findall(query) if w.lower() not in STOP and len(w) > 1)
    sents = SENT.split(body)
    # the opening: whole sentences up to about head characters
    k = 0; acc = 0
    while k < len(sents) and acc + len(sents[k]) <= head:
        acc += len(sents[k]) + 1; k += 1
    k = max(k, 1)
    rest = sents[k:]
    if not qw or not rest:
        return body
    # windows of consecutive sentences up to `passage` characters; score = IDF mass of distinct query terms present
    best = (0.0, None)
    for a in range(len(rest)):
        L = 0; b = a; seen = set()
        while b < len(rest) and L + len(rest[b]) <= passage:
            seen.update(w.lower() for w in WORD.findall(rest[b]) if w.lower() in qw); L += len(rest[b]) + 1; b += 1
        b = max(b, a + 1)
        sc = sum(idf.get(w, idf_max) for w in seen)
        if sc > best[0]:
            best = (sc, (a, b))
    if best[1] is None:
        return body
    a, b = best[1]
    if a == 0:
        return body   # the best window is already next
    return " ".join(sents[:k] + rest[a:b] + rest[:a] + rest[b:])


class CrossEncoder:
    """(query, page) -> relevance logit. onnxruntime when model_int8.onnx / model.onnx is in the directory (the
    device path), else torch + transformers."""

    def __init__(self, model_dir, threads=0, maxlen=160):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tok.enable_truncation(maxlen, strategy="longest_first"); self.tok.enable_padding(length=None)
        self.torch = None
        onnx = next((f for f in ("model_int8.onnx", "model.onnx") if os.path.exists(os.path.join(model_dir, f))), None)
        if onnx:
            import onnxruntime as ort
            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = threads
            self.sess = ort.InferenceSession(os.path.join(model_dir, onnx), so, providers=["CPUExecutionProvider"])
            self.names = [i.name for i in self.sess.get_inputs()]
        else:
            import torch
            from transformers import AutoModelForSequenceClassification
            self.torch = torch; self.dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.dev).eval()

    def __call__(self, query, texts, batch=16):
        if len(texts) > batch:
            return np.concatenate([self(query, texts[i:i + batch]) for i in range(0, len(texts), batch)])
        enc = self.tok.encode_batch([(query[:300], t) for t in texts])
        ids = np.array([e.ids for e in enc], dtype=np.int64); am = np.array([e.attention_mask for e in enc], dtype=np.int64)
        tt = np.array([e.type_ids for e in enc], dtype=np.int64)
        if self.torch is not None:
            with self.torch.no_grad():
                out = self.model(input_ids=self.torch.tensor(ids, device=self.dev), attention_mask=self.torch.tensor(am, device=self.dev), token_type_ids=self.torch.tensor(tt, device=self.dev)).logits
                return out.float().squeeze(-1).cpu().numpy()
        feed = {"input_ids": ids, "attention_mask": am}
        if "token_type_ids" in self.names:
            feed["token_type_ids"] = tt
        return self.sess.run(None, feed)[0].reshape(len(texts), -1)[:, -1]


class LocalSearch:
    def __init__(self, store_path, model_dir, threads=0, coarse=48, lex_k=48, title_k=12, rerank=64):
        self.st = Store(store_path)
        self.emb = Embedder(model_dir, threads)
        # SP_LOCAL_RERANK_MODEL: a second embedder for the ranking only (a retriever trained on the lineage's queries,
        # judged before the index is rebuilt with it); the coarse channel keeps the model the sign index was built with
        rr = os.environ.get("SP_LOCAL_RERANK_MODEL")
        self.emb_rank = Embedder(rr, threads) if rr and os.path.abspath(rr) != os.path.abspath(model_dir) else self.emb
        # SP_LOCAL_CE: a cross-encoder (MiniLM-class) reranking the top SP_LOCAL_CEK pages of the fused ranking by its logit
        self.ce = CrossEncoder(os.environ["SP_LOCAL_CE"], threads) if os.environ.get("SP_LOCAL_CE") else None
        self.ce_k = int(os.environ.get("SP_LOCAL_CEK", "16"))
        self.bits = np.memmap(os.path.join(store_path, "emb.bin"), dtype=np.uint8, mode="r").reshape(-1, DIM // 8)
        assert self.bits.shape[0] == self.st.n_docs, f"index {self.bits.shape[0]} vs store {self.st.n_docs}"
        self.maxlen_doc = int(os.environ.get("SP_LOCAL_DOCLEN", "96"))
        # the link channel: pages whose body links to a page the query names (Wikipedia's search reaches the
        # answer's own page through its body; the store holds openings only, the inlinks stand in for the body)
        self.links = None
        lp = os.environ.get("SP_LOCAL_LINKS", store_path)
        if os.path.exists(os.path.join(lp, "inlinks.bin")) and os.environ.get("SP_LOCAL_NOLINKS", "0") != "1":
            self.link_off = np.load(os.path.join(lp, "inlinks_offsets.npy"), mmap_mode="r")
            self.links = np.memmap(os.path.join(lp, "inlinks.bin"), dtype=np.int32, mode="r")
            self.link_k = int(os.environ.get("SP_LOCAL_LINKK", "16")); self.link_lex = int(os.environ.get("SP_LOCAL_LINKLEX", "400"))
        # SP_LOCAL_TERMS: the whole-text term index (terms.py) - articles mentioning the query's rare terms anywhere in
        # their body, the reach Wikipedia's search has; a channel of SP_LOCAL_TERMK candidates ahead of the others
        self.terms = None
        tp = os.environ.get("SP_LOCAL_TERMS", os.path.join(store_path, "terms"))
        if os.path.exists(os.path.join(tp, "terms.txt")) and os.environ.get("SP_LOCAL_NOTERMS", "0") != "1":
            from terms import TermIndex
            self.terms = TermIndex(tp); self.term_k = int(os.environ.get("SP_LOCAL_TERMK", "24")); self.term_cap = int(os.environ.get("SP_LOCAL_TERMCAP", "20000"))
        self.gpu = None
        self.pq = None
        self.ivf = None
        if os.path.exists(os.path.join(store_path, "emb_ivf.bin")) and os.environ.get("SP_LOCAL_FLAT", "0") != "1":
            from ivf import IVFIndex   # the phone's layout: a query touches ~2% of the index
            self.ivf = IVFIndex(store_path, nprobe=int(os.environ.get("SP_LOCAL_NPROBE", "48")))
            assert self.ivf.n == self.st.n_docs, f"ivf index {self.ivf.n} vs store {self.st.n_docs}"
        if os.environ.get("SP_LOCAL_PQ", "0") == "1" and os.path.exists(os.path.join(store_path, "pq_codes.npy")):   # no better than the sign bits on shard 0; kept as an option
            from pq import PQIndex
            self.pq = PQIndex(store_path, gpu=os.environ.get("SP_LOCAL_GPU", "0") == "1")
            assert self.pq.n == self.st.n_docs, f"pq index {self.pq.n} vs store {self.st.n_docs}"
        if os.environ.get("SP_LOCAL_GPU", "0") == "1":   # on the training box the whole index sits on the card (300 MB)
            import torch
            global UNPACK_T
            self.gpu = torch.from_numpy(np.ascontiguousarray(self.bits)).cuda()
            UNPACK_T = torch.from_numpy(UNPACK).cuda()
        self.con = sqlite3.connect("file:" + build_lex_index(store_path) + "?mode=ro", uri=True)
        self.con.execute("PRAGMA cache_size=-4096")   # 4 MB of page cache; the rest of the index stays on disk
        self.idf, self.idf_max = build_idf(self.st)
        e = os.environ
        self.coarse, self.lex_k, self.title_k, self.rerank = int(e.get("SP_LOCAL_COARSE", coarse)), int(e.get("SP_LOCAL_LEXK", lex_k)), int(e.get("SP_LOCAL_TITLEK", title_k)), int(e.get("SP_LOCAL_RERANK", rerank))
        self.lex_or = e.get("SP_LOCAL_LEXOR", "0") == "1"   # 1: the four-rarest-terms OR always joins the AND, not only when the AND falls short

    def _coarse_top(self, qv, k):
        """Asymmetric scoring: the float query against each article's signs. On shard 0 of the dump this finds the
        page Wikipedia's search returned 73% of the time within 96 candidates; binary-against-binary Hamming
        found it 23% of the time, so the query is never binarised."""
        if self.ivf is not None:
            return self.ivf.top(qv, k)
        if self.pq is not None:
            return self.pq.top(qv, k)
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

    @staticmethod
    def _terms(query):
        return [w for w in WORD.findall(query) if w.lower() not in STOP and len(w) > 1]

    def _lex_hits(self, query, k, column=None):
        """BM25 over title (x3) and opening text; with column='title', over the title alone. Returns
        [(article id, bm25)] best first (FTS5's bm25 is negative, more negative = better).
        FTS5 scores every matching row, so the query is shaped to match few: the two rarest terms together
        first (1 ms on the whole dump), then any of the four rarest (100 ms); every term OR-ed took 3 s."""
        words = list(dict.fromkeys(self._terms(query)))
        if not words:
            return []
        words.sort(key=lambda w: -self.idf.get(w.lower(), self.idf_max))
        shapes = []
        if len(words) >= 2:
            shapes.append(" AND ".join(f'"{w}"' for w in words[:2]))
        shapes.append(" OR ".join(f'"{w}"' for w in words[:4]))
        out = []
        for q in shapes:
            if column:
                q = f"{column}: ({q})"
            try:
                rows = self.con.execute("SELECT rowid, bm25(d, 3.0, 1.0) FROM d WHERE d MATCH ? ORDER BY bm25(d, 3.0, 1.0) LIMIT ?", (q, k)).fetchall()
            except sqlite3.OperationalError:
                rows = []
            out += [(r[0] - 1, -r[1]) for r in rows]
            if len(out) >= k and not self.lex_or:
                break
        seen = set(); res = []
        for i, sc in out:
            if i not in seen:
                seen.add(i); res.append((i, sc))
        return res[:k]

    def _link_hits(self, query, k):
        """Pages that link to a page the query names, and whose opening matches the query too: the title hits
        whose title sits in the query, their inlink sources, intersected with a wide lexical hit list."""
        if self.links is None:
            return []
        ql = " " + re.sub(r"[^a-z0-9 ]", " ", query.lower()) + " "
        named = []
        for i, _ in self._lex_hits(query, self.title_k * 2, column="title"):
            t = self.st.title(i) if hasattr(self.st, "title") else self.st.doc(i)[0]
            tl = " " + re.sub(r"[^a-z0-9 ]", " ", t.lower()).strip() + " "
            if len(tl.strip()) > 2 and tl in ql:
                named.append(i)
        if not named:
            return []
        src = set()
        for i in named[:4]:
            a, b = int(self.link_off[i]), int(self.link_off[i + 1])
            src.update(int(x) for x in np.asarray(self.links[a:b]))
        if not src:
            return []
        wide = self._lex_hits(query, self.link_lex)
        out = [i for i, _ in wide if i in src and i not in named]
        return out[:k]

    def _term_hits(self, query, k):
        """Articles whose whole text carries the query's rarest terms: the postings of the four rarest vocabulary
        terms, each list capped (a term in more than term_cap articles is too common to point anywhere), scored by
        the IDF mass of the terms an article carries; the best k."""
        if self.terms is None:
            return []
        words = [w.lower() for w in dict.fromkeys(WORD.findall(query)) if len(w) >= 3 and w.lower() not in STOP]
        words = [w for w in words if self.terms.has(w)]
        if not words:
            return []
        words.sort(key=lambda w: self.terms.df_of(w))
        n = self.st.n_docs; score = {}
        used = 0
        for w in words:
            if used >= 4:
                break
            d = self.terms.docs(w)
            if len(d) == 0 or len(d) > self.term_cap:
                continue
            used += 1; idf = math.log((n + 1) / (len(d) + 1)) + 1
            for i in d.tolist():
                score[i] = score.get(i, 0.0) + idf
        if not score:
            return []
        top = sorted(score.items(), key=lambda x: -x[1])[:k]
        return [i for i, _ in top]

    def search(self, query, k=3):
        qv = self.emb([QUERY_PREFIX + query])[0]
        lex = self._lex_hits(query, self.lex_k); bm = {i: s for i, s in lex}; bmax = max(bm.values(), default=1.0) or 1.0
        emb_c = [int(c) for c in self._coarse_top(qv, self.coarse)]
        cands = [x for pair in zip(emb_c, [i for i, _ in lex] + [None] * len(emb_c)) for x in pair if x is not None]   # interleaved
        cands += [i for i, _ in self._lex_hits(query, self.title_k, column="title")]
        if self.links is not None:
            cands = self._link_hits(query, self.link_k) + cands   # ahead of the cut: they are few and Wikipedia's search would have reached them
        if self.terms is not None:
            cands = self._term_hits(query, self.term_k) + cands   # the whole-text reach, ahead of the cut
        cands = list(dict.fromkeys(cands))[:self.rerank]
        docs = [self.st.doc(i) for i in cands]
        dv = self.emb_rank([f"{t}. {b[:400]}" for t, b in docs], maxlen=self.maxlen_doc)
        sims = dv @ (qv if self.emb_rank is self.emb else self.emb_rank([QUERY_PREFIX + query])[0])
        ql = " " + re.sub(r"[^a-z0-9 ]", " ", query.lower()) + " "
        qw = [w.lower() for w in WORD.findall(query) if w.lower() not in STOP and len(w) > 1]
        W = sum(self.idf.get(w, self.idf_max) for w in qw) or 1.0
        scored = []
        for (t, b), s, i in zip(docs, sims, cands):
            tl = " " + re.sub(r"[^a-z0-9 ]", " ", t.lower()).strip() + " "
            full = W_FULL if (len(tl.strip()) > 2 and tl in ql) else 0.0   # the query names the article
            tw = set(w.lower() for w in WORD.findall(t)); bw = set(w.lower() for w in WORD.findall(b[:600]))
            ft = sum(self.idf.get(w, self.idf_max) for w in qw if w in tw) / W        # query terms in the title
            fb = sum(self.idf.get(w, self.idf_max) for w in qw if w in tw or w in bw) / W   # ... or in the opening
            fd = 0.0
            if W_DEEP and len(b) > 600:
                dw = set(w.lower() for w in WORD.findall(b[600:]))
                fd = sum(self.idf.get(w, self.idf_max) for w in qw if w in dw and w not in tw and w not in bw) / W   # ... only deeper in the article
            # cosine plus the lexical overlap the model's keyword queries were shaped by (weights from the shard-0 grid)
            scored.append((float(s) + W_TITLE * ft + W_BODY * fb + W_DEEP * fd + full + W_BM25 * bm.get(i, 0.0) / bmax, i, t, b))
        scored.sort(key=lambda x: -x[0])
        if self.ce is not None and scored:
            top = scored[:self.ce_k]
            ce = self.ce(query, [f"{t}. {b[:500]}" for _, _, t, b in top])
            top = [(float(c), i, t, b) for c, (_, i, t, b) in zip(ce, top)]
            top.sort(key=lambda x: -x[0]); return top[:k]
        return scored[:k]

    def fetch(self, kw, k=None, chars=None):
        """What the rollout loop serves for a query. One page by default, as Wikipedia's search gave; SP_LOCAL_K pages
        of SP_LOCAL_CHARS characters each is the variant where the second-best page rides in the same first block."""
        k = k or int(os.environ.get("SP_LOCAL_K", "1")); chars = chars or int(os.environ.get("SP_LOCAL_CHARS", "0"))
        hits = self.search(kw, k)
        if not hits:
            return ""
        if os.environ.get("SP_LOCAL_PASSAGE", "0") == "1":
            hits = [(s, i, t, compose(kw, b, self.idf, self.idf_max)) for s, i, t, b in hits]
        if k == 1 or not chars:
            return "\n\n".join(f"{t}: {b}" for _, _, t, b in hits)
        return "\n\n".join(f"{t}: {b[:chars]}" for _, _, t, b in hits)


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
