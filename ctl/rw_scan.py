#!/usr/bin/env python3
"""How large would the raw window (RW) have to be for the gold information to still be verbatim at answer time?
usage: rw_scan.py teacher600.jsonl pooleval_post.jsonl"""
import json, re, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/root/fft_hf")
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()
def has(t, g): return (" " + norm(g) + " ") in (" " + norm(t) + " ")
def rows(p):
    out = []
    for l in open(p):
        try: out.append(json.loads(l))
        except Exception: pass
    return out
def dist(text, gold):
    """tokens from the last gold occurrence inside an <information> block to </think> (None if never served / unlanded)."""
    if "</think>" not in text: return None
    think = text.split("</think>")[0]; best = None
    for m in re.finditer(r"<information>(.*?)</information>", think, re.S):
        body = m.group(1); low = re.sub(r"[^a-z0-9 ]", " ", body.lower()); g = norm(gold)
        i = low.rfind(" " + g + " ")
        if i < 0: i = low.rfind(g)
        if i >= 0: best = m.start(1) + i + len(g)
    if best is None: return None
    return len(tok.encode(think[best:], add_special_tokens=False))
T = {}
for r in rows(sys.argv[1]): T.setdefault(r["q"].strip(), []).append(r)
S = [r for r in rows(sys.argv[2]) if r["q"].strip() in T]
sf = [(dist(r["text"], r["gold"]), r) for r in S if not r["correct"] and r["grounded"]]
sf = [(d, r) for d, r in sf if d is not None]
sc = [d for d in (dist(r["text"], r["gold"]) for r in S if r["correct"]) if d is not None]
tc = [d for d in (dist(x["text"], x["gold"]) for r in S for x in T[r["q"].strip()] if x["correct"]) if d is not None]
print(f"=== RW SCAN n={len(S)}  student gold-served-but-wrong={len(sf)}  student-correct={len(sc)}  teacher-correct-rolls={len(tc)}")
print("student wrong, gold distance sorted:", sorted(d for d, _ in sf))
print("RW    | wrong rows with gold inside | student-correct inside | teacher-correct inside")
for rw in [512, 640, 768, 1024, 1280, 1536, 2048, 3072, 4096]:
    print(f"{rw:5d} | {sum(d <= rw for d, _ in sf):3d}/{len(sf)} | {sum(d <= rw for d in sc):3d}/{len(sc)} | {sum(d <= rw for d in tc):3d}/{len(tc)}")
for d, r in sorted(sf, key=lambda z: -z[0])[:8]:
    print(f"  d={d:5d} ns={r['ns']} more={r['more']} gold={r['gold']!r} answer={r['answer'][:60]!r}")
print("=== RW SCAN END")
