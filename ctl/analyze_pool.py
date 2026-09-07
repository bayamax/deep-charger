#!/usr/bin/env python3
"""Where does the compressed student lose against the teacher on the same questions?
usage: analyze_pool.py teacher600.jsonl pooleval_post.jsonl [RW=512]"""
import json, re, sys
from collections import Counter
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/root/fft_hf")
RW = int(sys.argv[3]) if len(sys.argv) > 3 else 512
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()
def has(t, g): return (" " + norm(g) + " ") in (" " + norm(t) + " ")
def rows(p):
    out = []
    for l in open(p):
        try: out.append(json.loads(l))
        except Exception: pass
    return out
def ntok(s): return len(tok.encode(s, add_special_tokens=False)) if s else 0
def gold_pos(text, gold):
    """token distance from the end of the LAST <information> block containing gold to the answer position (</think>)."""
    if "</think>" not in text: return None
    think = text.split("</think>")[0]; g = norm(gold); last = None
    for m in re.finditer(r"<information>(.*?)</information>", think, re.S):
        if has(m.group(1), gold): last = m.end()
    if last is None: return None
    return ntok(think[last:])
def think_has_gold(text, gold):
    """the model itself wrote the gold string in its reasoning (outside <information>) before </think>."""
    if "</think>" not in text: return False
    think = re.sub(r"<information>.*?</information>", " ", text.split("</think>")[0], flags=re.S)
    return has(think, gold)
T = {}
for r in rows(sys.argv[1]): T.setdefault(r["q"].strip(), []).append(r)
S = [r for r in rows(sys.argv[2]) if r["q"].strip() in T]
n = len(S); print(f"=== ANALYZE n={n} RW={RW}")
# 1. outcome buckets, student vs teacher (teacher = 2 rolls; 'teacher right' = at least one roll)
b = Counter()
for r in S:
    t_any = any(x["correct"] for x in T[r["q"].strip()])
    b[("S+" if r["correct"] else "S-") + ("T+" if t_any else "T-")] += 1
print("buckets S=student T=teacher(any of 2):", dict(b))
# 2. student failures: grounded (gold served) or not
fail = [r for r in S if not r["correct"]]
gr = [r for r in fail if r["grounded"]]; ng = [r for r in fail if not r["grounded"]]
print(f"student wrong={len(fail)}: gold-served-but-wrong={len(gr)} gold-never-served={len(ng)} dead={sum(r.get('dead',False) for r in fail)}")
# 3. for gold-served-but-wrong: was gold still inside the raw window at answer time?
d_in = d_out = d_none = 0; outs = []
for r in gr:
    d = gold_pos(r["text"], r["gold"])
    if d is None: d_none += 1
    elif d <= RW: d_in += 1
    else: d_out += 1; outs.append((d, r))
print(f"  gold->answer distance: inside RW({RW})={d_in} evicted(>RW)={d_out} unmeasurable={d_none}")
# same measure on the student's CORRECT rows
ok = [r for r in S if r["correct"]]; ci = co = 0
for r in ok:
    d = gold_pos(r["text"], r["gold"])
    if d is not None: (ci, co) = (ci + 1, co) if d <= RW else (ci, co + 1)
print(f"  student correct rows: gold inside RW={ci} evicted={co}")
# 4. teacher correct rolls on the same questions: how far back was gold when it answered? (compression exposure)
ti = to = tn = 0
for r in S:
    for x in T[r["q"].strip()]:
        if not x["correct"]: continue
        d = gold_pos(x["text"], x["gold"])
        if d is None: tn += 1
        elif d <= RW: ti += 1
        else: to += 1
print(f"teacher correct rolls: gold inside RW={ti} evicted(>RW)={to} unmeasurable={tn}")
# 5. search behaviour
sc = Counter(r["ns"] for r in S); tc = Counter(x["ns"] for r in S for x in T[r["q"].strip()])
print("student searches/q:", dict(sorted(sc.items()))); print("teacher searches/roll:", dict(sorted(tc.items())))
same_q1 = sum(1 for r in S if r["queries"] and any(x["queries"] and norm(x["queries"][0]) == norm(r["queries"][0]) for x in T[r["q"].strip()]))
print(f"first query identical to a teacher roll: {same_q1}/{n}")
stop1 = [r for r in S if r["ns"] == 1]
print(f"student stopped after 1 search: {len(stop1)} (correct {sum(r['correct'] for r in stop1)}, grounded {sum(r['grounded'] for r in stop1)})")
# 6. never-served failures: did the teacher get gold served on those?
tg = sum(1 for r in ng if any(x["grounded"] for x in T[r["q"].strip()]))
print(f"gold-never-served ({len(ng)}): teacher served gold on {tg} of them")
# 7. length
print(f"student mean gen tokens (whole text): {sum(ntok(r['text']) for r in S)/n:.0f} | teacher: {sum(ntok(x['text']) for r in S for x in T[r['q'].strip()])/(2*n):.0f}")
# 8. samples: evicted-gold failures
for d, r in sorted(outs, key=lambda z: -z[0])[:3]:
    print(f"--- evicted sample d={d} q={r['q'][:90]!r} gold={r['gold']!r} answer={r['answer'][:80]!r}")
    tail = r["text"].split("</think>")[0][-500:].replace("\n", " ")
    print("    think tail:", tail[-300:])
# 8b. decoding-guard suspects: model wrote gold in its own reasoning, final answer still wrong
tw = [r for r in fail if think_has_gold(r["text"], r["gold"])]
print(f"wrote gold in think but final answer wrong: {len(tw)}  (grounded {sum(r['grounded'] for r in tw)})")
for r in tw[:6]:
    print(f"    gold={r['gold']!r} answer={r['answer'][:70]!r}")
tw_t = sum(1 for r in S for x in T[r["q"].strip()] if not x["correct"] and think_has_gold(x["text"], x["gold"]))
print(f"teacher: wrote gold in think but final answer wrong: {tw_t} of {2*n} rolls")
# 9. samples: gold inside window but wrong (reading/answering failure)
k = 0
for r in gr:
    d = gold_pos(r["text"], r["gold"])
    if d is not None and d <= RW and k < 3:
        k += 1; print(f"--- in-window-wrong d={d} q={r['q'][:90]!r} gold={r['gold']!r} answer={r['answer'][:80]!r}")
print("=== ANALYZE END")
