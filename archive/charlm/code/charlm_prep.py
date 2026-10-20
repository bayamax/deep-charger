#!/usr/bin/env python3
"""The character model's text, as one byte per character, on disk.

Alphabet (98): newline, the 95 printable ASCII characters, UNK (anything else; a run of them is one UNK) and EOD (between
documents). Typographic Unicode is folded to its ASCII look-alike one-for-one before that (curly quotes, dashes, the
non-breaking space, accented Latin letters lose their accents), so the web text's punctuation does not become UNK.

Sources (quality first): WikiText-103 (Wikipedia Good/Featured articles, both train shards) and FineWeb-Edu (English web
pages kept by an educational-quality classifier, score >= 3) - its sample-10BT shards, each ~3B characters.
Held out: WikiText-103 validation and the first 3000 documents of the last FineWeb shard taken (validation, decides
growth), WikiText-2 test and the next 3000 FineWeb documents (test, decides nothing). The raw test texts are kept too,
for GPT-2 on the very same documents.

  python3 charlm_prep.py --out /root/charlm/data --fineweb 1
  python3 charlm_prep.py --out /root/charlm/data --more 2,3      (more training text later, appended by the caller)
"""
import argparse, codecs, os, sys, time, unicodedata

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--fineweb", type=int, default=1, help="FineWeb-Edu sample-10BT shards")
ap.add_argument("--src", default="", help="a folder that already holds the parquet files (else they are downloaded there)")
ap.add_argument("--chat-val", type=int, default=0, help="only this: the first N conversations of SmolTalk's test split (used by no training) as "
                "plain text - 'User: ...' / 'Assistant: ...' lines - into val_chat.bin (+ val_chat.txt), the chat yardstick")
ap.add_argument("--chat-train", default="", help="only this: SmolTalk train shards (e.g. 4,5,6,7,8 of 9) as the same plain 'User:'/'Assistant:' "
                "text into train_chat.bin, for pre-training (the other shards stay for the chat fine-tuning; the test split stays unseen)")
ap.add_argument("--more", default="", help="only this: more FineWeb-Edu training text from these sample-10BT shards (e.g. 2,3) into train_fw_more.bin, "
                "nothing held out (the held-out documents stay those of the original run's last shard)")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True); SRC = A.src or os.path.join(A.out, "src"); os.makedirs(SRC, exist_ok=True)

CHARS = "\n" + "".join(chr(i) for i in range(32, 127)); UNK = len(CHARS); EOD = UNK + 1; V = EOD + 1
codecs.register_error("unk", lambda e: ("\x7f", e.end))
LUT = np.full(256, UNK, np.uint8)
for i, c in enumerate(CHARS): LUT[ord(c)] = i
FOLD = {0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'", 0x2032: "'", 0x00B4: "'", 0x201C: '"', 0x201D: '"', 0x201E: '"', 0x201F: '"',
        0x2033: '"', 0x00AB: '"', 0x00BB: '"', 0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-", 0x2014: "-", 0x2015: "-", 0x2212: "-",
        0x00A0: " ", 0x2002: " ", 0x2003: " ", 0x2009: " ", 0x200A: " ", 0x202F: " ", 0x2026: ".", 0x2022: "*", 0x00B7: "*", 0x00D7: "x",
        0x2044: "/", 0x00F7: "/", 0x2122: "T", 0x00AE: "R", 0x00A9: "C", 0x00B0: "o", 0x0009: " ", 0x000D: "\n"}
for cp in range(0xC0, 0x250):   # accented Latin: the base letter when it is one ASCII letter
    if cp in FOLD: continue
    base = unicodedata.normalize("NFKD", chr(cp)).encode("ascii", "ignore").decode()
    if len(base) == 1: FOLD[cp] = base
TABLE = str.maketrans({k: v for k, v in FOLD.items()})


def norm(text): return text.translate(TABLE)
def encode(text): return LUT[np.frombuffer(norm(text).encode("ascii", "unk"), dtype=np.uint8)]


def fetch(repo, path):
    dst = os.path.join(SRC, path.replace("/", "_"))
    if not os.path.exists(dst):
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo, path, repo_type="dataset", local_dir=os.path.join(SRC, "dl")); os.replace(p, dst)
    return dst


import pyarrow.parquet as pq  # noqa: E402


def docs_of(path, col="text"):
    f = pq.ParquetFile(path)
    for rg in range(f.num_row_groups):
        for t in f.read_row_group(rg, columns=[col]).column(col).to_pylist():
            yield t


def write(name, docs, raw=None):
    """docs -> one uint8 file, EOD between them; returns characters written"""
    n = 0; t0 = time.time(); unk = 0
    with open(os.path.join(A.out, name), "wb") as f:
        for d in docs:
            if not d or not d.strip(): continue
            a = encode(d); f.write(a.tobytes()); f.write(bytes([EOD])); n += len(a) + 1; unk += int((a == UNK).sum())
            if raw is not None: raw.append(d)
    print(f"[prep] {name}: {n / 1e6:.1f}M chars, UNK {100 * unk / max(n, 1):.3f}% [{time.time() - t0:.0f}s]", flush=True)
    return n


def wikitext_articles(path):
    """WikiText rows are lines; an article starts at a ' = Title = ' line"""
    cur = []
    for line in docs_of(path):
        if line.startswith(" = ") and not line.startswith(" = = ") and cur and "".join(cur).strip():
            yield "".join(cur); cur = []
        cur.append(line)
    if cur: yield "".join(cur)


def chat_text(msgs):
    """a conversation as plain text: 'User: ...' / 'Assistant: ...' lines, a system prompt folded into the first user turn"""
    lines = []; sys_ = ""
    for m in msgs:
        if m["role"] == "system": sys_ += m["content"].strip() + "\n\n"; continue
        c = m["content"].strip()
        if m["role"] == "user" and sys_: c = sys_ + c; sys_ = ""
        lines.append(("User: " if m["role"] == "user" else "Assistant: ") + c)
    return "\n".join(lines) + "\n" if lines else ""


if A.chat_val or A.chat_train:
    from huggingface_hub import hf_hub_download
    st_ = lambda path: hf_hub_download("HuggingFaceTB/smoltalk", path, repo_type="dataset", local_dir=os.path.join(SRC, "dl"))
if A.chat_val:
    t = pq.read_table(st_("data/all/test-00000-of-00001.parquet"), columns=["messages"])
    convs = []
    for msgs in t.column("messages").to_pylist():
        c = chat_text(msgs)
        if c: convs.append(c)
        if len(convs) >= A.chat_val: break
    raw = []; write("val_chat.bin", convs, raw); open(os.path.join(A.out, "val_chat.txt"), "w").write("\x00".join(raw))
    print("CHARLM_CHATVAL_DONE", flush=True); sys.exit(0)
if A.chat_train:
    paths = [st_(f"data/all/train-{int(i):05d}-of-00009.parquet") for i in A.chat_train.split(",")]
    write("train_chat.bin", (chat_text(m) for p_ in paths for m in pq.read_table(p_, columns=["messages"]).column("messages").to_pylist()))
    for p_ in paths: os.remove(p_)
    print("CHARLM_CHATTRAIN_DONE", flush=True); sys.exit(0)

if A.more:
    more = [fetch("HuggingFaceFW/fineweb-edu", f"sample/10BT/{int(i):03d}_00000.parquet") for i in A.more.split(",")]
    write("train_fw_more.bin", (d for p_ in more for d in docs_of(p_)))
    for p_ in more: os.remove(p_)
    print("CHARLM_MORE_DONE", flush=True); sys.exit(0)

W = "Salesforce/wikitext"
wtr = [fetch(W, f"wikitext-103-raw-v1/train-0000{i}-of-00002.parquet") for i in range(2)]
wva = fetch(W, "wikitext-103-raw-v1/validation-00000-of-00001.parquet"); wte = fetch(W, "wikitext-2-raw-v1/test-00000-of-00001.parquet")
write("train_wiki.bin", (a for p in wtr for a in wikitext_articles(p)))
write("val_wiki.bin", wikitext_articles(wva))
raw = []; write("test_wiki.bin", ["\n\n".join(pq.read_table(wte, columns=["text"]).column("text").to_pylist())], raw)
open(os.path.join(A.out, "test_wiki.txt"), "w").write(raw[0])

FW = "HuggingFaceFW/fineweb-edu"
shards = [fetch(FW, f"sample/10BT/{i:03d}_00000.parquet") for i in range(A.fineweb)]
it = docs_of(shards[-1])
held = [next(it) for _ in range(6000)]
write("val_fw.bin", held[:3000]); raw = []; write("test_fw.bin", held[3000:], raw)
open(os.path.join(A.out, "test_fw.txt"), "w").write("\x00".join(raw))      # NUL between documents, for GPT-2's own run


def fw_train():
    for p in shards[:-1]:
        yield from docs_of(p)
    yield from it                      # the last shard after its held-out 6000


write("train_fw.bin", fw_train())
for p in shards: os.remove(p)          # the parquet files go; the bins stay
open(os.path.join(A.out, "PREP_DONE"), "w").write(f"{V} symbols\n")
print("CHARLM_PREP_DONE", flush=True)
