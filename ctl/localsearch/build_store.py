#!/usr/bin/env python3
"""Build the local article store the on-device search reads: Wikipedia intros, block-compressed.

The model was trained against Wikipedia's own search (a query -> the best page -> its extract, served
256 tokens at a time), so what has to live on the device is the text it learned to read: the opening of
every article. Each article is cut to --chars characters at a sentence boundary, packed 256 articles to a
block, and every block is one zstd frame appended to docs.bin; blocks.idx holds the frame offsets, so a
document is (block, position) and reading one costs one frame. Titles go to titles.txt, one per line, in
document order, for the title index. Shards are appended in the order given, so the whole dump can be
processed one parquet file at a time and each file deleted after it is read.

  python3 build_store.py --out /root/wiki_store 20231101.en/train-0000*.parquet
"""
import argparse, io, json, os, re, struct, sys

import pyarrow.parquet as pq
import zstandard as zstd

ap = argparse.ArgumentParser()
ap.add_argument("parquet", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--chars", type=int, default=1500, help="characters of each article kept, cut at a sentence end")
ap.add_argument("--min-chars", type=int, default=80, help="articles shorter than this are stubs and are dropped")
ap.add_argument("--block", type=int, default=256, help="articles per zstd frame")
ap.add_argument("--level", type=int, default=9)
A = ap.parse_args()

os.makedirs(A.out, exist_ok=True)
META = os.path.join(A.out, "meta.json")
meta = json.load(open(META)) if os.path.exists(META) else {"n_docs": 0, "n_blocks": 0, "block": A.block, "chars": A.chars, "shards": []}
assert meta["block"] == A.block and meta["chars"] == A.chars, "an existing store has a different layout"

SENT_END = re.compile(r"[.!?][\"')\]]?(?=\s)")
WS = re.compile(r"[ \t]+")
BLANK = re.compile(r"\n{2,}")


def clean(text):
    """The dump's plain text: collapse whitespace, keep paragraph breaks, drop empty section headers at the tail."""
    text = WS.sub(" ", text.replace("\r", "").replace("\x00", ""))
    text = BLANK.sub("\n", text)
    return text.strip()


def cut(text, n):
    if len(text) <= n:
        return text
    head = text[:n]
    ends = [m.end() for m in SENT_END.finditer(head)]
    if ends and ends[-1] >= n // 3:
        return head[:ends[-1]]
    sp = head.rfind(" ")
    return head[:sp] if sp > n // 3 else head


cctx = zstd.ZstdCompressor(level=A.level)
docs_f = open(os.path.join(A.out, "docs.bin"), "ab")
idx_f = open(os.path.join(A.out, "blocks.idx"), "ab")
titles_f = open(os.path.join(A.out, "titles.txt"), "a", encoding="utf-8")
buf, n_in_buf = io.BytesIO(), 0
n_docs, n_blocks, n_dropped = meta["n_docs"], meta["n_blocks"], 0


def flush():
    global buf, n_in_buf, n_blocks
    if n_in_buf == 0:
        return
    frame = cctx.compress(buf.getvalue())
    off = docs_f.tell()
    docs_f.write(frame)
    idx_f.write(struct.pack("<QIH", off, len(frame), n_in_buf))   # offset, frame length, documents in it
    n_blocks += 1
    buf, n_in_buf = io.BytesIO(), 0


for path in A.parquet:
    if path in meta["shards"]:
        print(f"[store] {path} already in the store, skipped", flush=True)
        continue
    pf = pq.ParquetFile(path)
    n_before = n_docs
    for batch in pf.iter_batches(batch_size=4096, columns=["title", "text"]):
        for title, text in zip(batch.column("title").to_pylist(), batch.column("text").to_pylist()):
            title = (title or "").replace("\n", " ").strip()
            body = cut(clean(text or ""), A.chars)
            if not title or len(body) < A.min_chars:
                n_dropped += 1
                continue
            rec = (title + "\n" + body + "\n\x00").encode("utf-8")   # NUL ends a document inside the frame
            buf.write(rec); n_in_buf += 1
            titles_f.write(title + "\n")
            n_docs += 1
            if n_in_buf >= A.block:
                flush()
    flush()
    meta["shards"].append(path)
    meta.update(n_docs=n_docs, n_blocks=n_blocks)
    json.dump(meta, open(META, "w"), indent=1)
    print(f"[store] {os.path.basename(path)}: +{n_docs - n_before} articles ({n_dropped} stubs dropped so far); "
          f"store {n_docs} articles, {n_blocks} blocks, docs.bin {docs_f.tell()/1e9:.2f} GB", flush=True)

docs_f.close(); idx_f.close(); titles_f.close()
print(f"STORE_DONE {n_docs} articles {n_blocks} blocks", flush=True)
