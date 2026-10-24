"""Score MATH rollouts (tools/sp_qwen3.py rollout output with "gold"): the last \\boxed{...} after </think> vs the gold answer.

  python sp_mathcheck.py roll_full.jsonl roll_sp512.jsonl ...
Prints, per file: accuracy, how many closed </think>, how many gave a boxed answer, median length, and a 4-gram repetition
rate (a collapsed / looping generation shows up there, not in KL).
"""
import json, re, sys


def boxed(t):
    t = t.split("</think>")[-1] if "</think>" in t else t
    i = t.rfind("\\boxed{")
    if i < 0: return None
    j = i + 7; d = 1; out = []
    while j < len(t) and d:
        c = t[j]; d += (c == "{") - (c == "}")
        if d: out.append(c)
        j += 1
    return "".join(out)


def norm(s):
    s = (s or "").strip().replace(" ", "").replace("\\!", "").replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "").replace("^\\circ", "").replace("^{\\circ}", "").replace("\\$", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s).rstrip(".")
    return s


def rep4(t):
    w = t.split()
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(g)) / max(len(g), 1)


for f in sys.argv[1:]:
    rs = [json.loads(l) for l in open(f)]
    ok = sum(norm(boxed(r["text"])) == norm(r.get("gold")) for r in rs)
    closed = sum("</think>" in r["text"] for r in rs); bx = sum(boxed(r["text"]) is not None for r in rs)
    L = sorted(len(r["text"]) for r in rs); rp = sum(rep4(r["text"]) for r in rs) / max(len(rs), 1)
    print(f"{f}: {ok}/{len(rs)} correct ({100 * ok / max(len(rs), 1):.0f}%) | closed think {closed} | boxed {bx} | "
          f"median chars {L[len(L) // 2] if L else 0} | 4-gram repeat {rp:.3f}")
