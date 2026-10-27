"""Score multi-problem bundles (ctl/membench.jsonl rollouts): the ANSWERS line after </think> vs the gold list - per-problem
accuracy, bundles fully right, how many closed, and per-position accuracy (does the model still remember problem 1's
answer when it writes the summary line?).

  python sp_memcheck.py roll_a.jsonl roll_b.jsonl ...
"""
import json, re, sys


def answers(t):
    t = t.split("</think>")[-1] if "</think>" in t else t
    m = re.findall(r"ANSWERS?\s*[:：]\s*(.+)", t)
    if not m: return []
    return [x.strip().replace("$", "").replace(",", "") for x in re.split(r"[;,]\s*|\s{2,}", m[-1].strip()) if x.strip()]


def num(s):
    m = re.search(r"-?\d+(?:\.\d+)?", s or ""); return float(m.group()) if m else None


for f in sys.argv[1:]:
    rs = [json.loads(l) for l in open(f)]
    ok = tot = full = closed = 0; pos = {}
    for r in rs:
        a = answers(r["text"]); g = r["golds"]; closed += "</think>" in r["text"]; allok = True
        for j, gj in enumerate(g):
            hit = j < len(a) and num(a[j]) is not None and abs(num(a[j]) - float(gj)) < 1e-6
            ok += hit; tot += 1; allok &= hit; pos.setdefault(j, [0, 0]); pos[j][0] += hit; pos[j][1] += 1
        full += allok
    pp = " ".join(f"p{j + 1}:{100 * v[0] / v[1]:.0f}" for j, v in sorted(pos.items()))
    print(f"{f}: problems {ok}/{tot} ({100 * ok / max(tot, 1):.1f}%) | bundles fully right {full}/{len(rs)} | closed {closed} | by position {pp}")
