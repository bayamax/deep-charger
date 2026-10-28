"""Score MATH rollouts (tools/sp_qwen3.py rollout output with "gold"): the last \\boxed{...} after </think> vs the gold answer.

  python sp_mathcheck.py roll_full.jsonl roll_sp512.jsonl ...
Prints, per file: accuracy, how many closed </think>, how many gave a boxed answer, median length, and a 4-gram repetition
rate (a collapsed / looping generation shows up there, not in KL). With math-verify installed, equivalent LaTeX answers count.
"""
import json, re, sys

try:   # equivalence of LaTeX answers (\\frac 34 = \\dfrac{3}{4}, (B) = \\text{(B)}, .35625 = 0.35625); string match as the fallback
    from math_verify import parse as _mv_parse, verify as _mv_verify
except ImportError:
    _mv_parse = None


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


def correct(r):
    b = boxed(r["text"])
    if b is None: return False
    if norm(b) == norm(r.get("gold")): return True
    if re.fullmatch(r"[A-J]", r.get("gold") or ""):   # multiple choice: the letter, however it is wrapped ((C), \\text{C}, C. ...)
        m = re.match(r"\(?([A-J])\b", norm(b).lstrip("("))
        return bool(m) and m.group(1) == r["gold"]
    if _mv_parse is None: return False
    try: return bool(_mv_verify(_mv_parse("$" + (r.get("gold") or "") + "$"), _mv_parse("$" + b + "$")))
    except Exception: return False


for f in (sys.argv[1:] if __name__ == "__main__" else []):
    rs = [json.loads(l) for l in open(f)]
    ok = sum(correct(r) for r in rs)
    closed = sum("</think>" in r["text"] for r in rs); bx = sum(boxed(r["text"]) is not None for r in rs)
    L = sorted(len(r["text"]) for r in rs); rp = sum(rep4(r["text"]) for r in rs) / max(len(rs), 1)
    print(f"{f}: {ok}/{len(rs)} correct ({100 * ok / max(len(rs), 1):.0f}%) | closed think {closed} | boxed {bx} | "
          f"median chars {L[len(L) // 2] if L else 0} | 4-gram repeat {rp:.3f}")
