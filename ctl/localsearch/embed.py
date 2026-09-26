#!/usr/bin/env python3
"""Embed every article of a store with bge-small and write the binary index the device searches.

One vector per article, from "title. opening text" cut to --maxlen tokens, CLS-pooled and normalised as
bge does. What is stored is the sign of each of the 384 dimensions: 48 bytes an article, so the whole
English Wikipedia is about 300 MB and a query is one XOR-and-popcount pass over a memory-mapped file.
The float vector is not kept: the search re-embeds its few dozen candidates from the stored text to
rank them, which costs a fraction of a second and saves 2.4 GB.

Two backends: onnxruntime on the CPU (the device path, and enough for a shard), torch on a GPU for the
whole dump. Either writes emb.bin (packed bits, article order) and can be resumed: articles already
embedded are skipped.

  python3 embed.py --store /root/wiki_store --model /root/bge-small --backend torch --batch 256
"""
import argparse, json, os, struct, sys, time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--store", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--backend", default="onnx", choices=["onnx", "torch"])
ap.add_argument("--batch", type=int, default=64); ap.add_argument("--maxlen", type=int, default=160)
ap.add_argument("--threads", type=int, default=0); ap.add_argument("--limit", type=int, default=0)
A = ap.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from store import Store  # noqa: E402

DIM = 384
st = Store(A.store)
EMB = os.path.join(A.store, "emb.bin")
done = os.path.getsize(EMB) // (DIM // 8) if os.path.exists(EMB) else 0
total = st.n_docs if not A.limit else min(A.limit, st.n_docs)
print(f"[embed] {st.n_docs} articles in the store, {done} already embedded, backend {A.backend}", flush=True)

from tokenizers import Tokenizer  # noqa: E402
tok = Tokenizer.from_file(os.path.join(A.model, "tokenizer.json"))
tok.enable_truncation(A.maxlen); tok.enable_padding(length=None)

if A.backend == "onnx":
    import onnxruntime as ort
    so = ort.SessionOptions()
    if A.threads:
        so.intra_op_num_threads = A.threads
    sess = ort.InferenceSession(os.path.join(A.model, "model.onnx"), so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]

    def encode(texts):
        enc = tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64); am = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": am}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = sess.run(None, feed)[0][:, 0, :]          # CLS
        return out / np.linalg.norm(out, axis=1, keepdims=True).clip(1e-6)
else:
    import torch
    from transformers import AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(A.model, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()

    @torch.no_grad()
    def encode(texts):
        enc = tok.encode_batch(texts)
        ids = torch.tensor([e.ids for e in enc], device=dev); am = torch.tensor([e.attention_mask for e in enc], device=dev)
        out = model(input_ids=ids, attention_mask=am).last_hidden_state[:, 0, :].float()
        return torch.nn.functional.normalize(out, dim=1).cpu().numpy()


def pack(v):
    return np.packbits((v > 0).astype(np.uint8), axis=1)   # 384 bits -> 48 bytes, dimension order


t0 = time.time(); n = done
with open(EMB, "ab") as f:
    batch = []
    for i in range(done, total):
        title, body = st.doc(i)
        batch.append(f"{title}. {body[:800]}")
        if len(batch) == A.batch or i == total - 1:
            f.write(pack(encode(batch)).tobytes()); n += len(batch); batch = []
            if (n // A.batch) % 50 == 0:
                rate = (n - done) / max(time.time() - t0, 1e-6)
                print(f"[embed] {n}/{total}  {rate:.0f} articles/s  eta {(total - n) / max(rate, 1e-6) / 60:.0f} min", flush=True)
print(f"EMBED_DONE {n} articles in {(time.time()-t0)/60:.1f} min", flush=True)
