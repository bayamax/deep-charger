#!/usr/bin/env python3
"""The vote over sampled passes of pool_eval.py on the same questions: per question, the single-pass rate (the mean over
the passes), the majority of the passes' answers (grounded answers first, ties by count then by the first seen), and the
ceiling (any pass right). The answers are compared as pool_eval scores them (the first answer sentence, normalised).

  python3 vote_count.py vote_p0_0.jsonl vote_p0_1.jsonl ...
"""
import collections, json, re, sys


def norm(s): return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()
def has(t, g): return (" " + norm(g) + " ") in (" " + norm(t) + " ")


def key(ans):
    m = re.search(r"answer is (.+)", ans or "", re.I); a = m.group(1) if m else (ans or "")
    return re.sub(r"\s+", " ", norm(a)).strip(" .")


Q = collections.defaultdict(list)
for f in sys.argv[1:]:
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        if "q" in r and "gold" in r: Q[r["q"]].append(r)
single = maj = majg = oracle = 0; n = 0; rolls = 0; per_pass = collections.Counter()
for q, rs in Q.items():
    n += 1; rolls += len(rs); single += sum(bool(r.get("correct")) for r in rs) / len(rs)
    oracle += any(r.get("correct") for r in rs)
    c = collections.Counter(key(r.get("answer")) for r in rs if key(r.get("answer")))
    cg = collections.Counter(key(r.get("answer")) for r in rs if key(r.get("answer")) and r.get("grounded"))
    gold = rs[0]["gold"]
    if c: maj += has(c.most_common(1)[0][0], gold)
    pick = (cg.most_common(1)[0][0] if cg else (c.most_common(1)[0][0] if c else ""))
    majg += has(pick, gold)
k = rolls / max(n, 1)
print(f"[vote] {n} questions, {k:.1f} passes each: single pass {100 * single / n:.1f}%, majority {100 * maj / n:.1f}%, majority of the grounded {100 * majg / n:.1f}%, any pass right {100 * oracle / n:.1f}%")
