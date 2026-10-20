"""The character model's alphabet and the Unicode fold (shared by charlm_prep.py's data and charlm_sft.py's chats)."""
import codecs, unicodedata

import numpy as np

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


