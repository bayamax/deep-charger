"""Score tools/mt_build.py rollouts (ctl/mt_eval.jsonl through sp_qwen3.py rollout --kv-evict): per task, the final assistant
turn's \\boxed{} answer against the gold. chain: the sum is also checked against the model's OWN two earlier answers (memory of
what it said, apart from whether it solved them). 'reached' = the dialogue got to its last turn.
  python tools/mt_score.py out1.jsonl out2.jsonl ...
"""
import json, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sp_mathcheck import correct, boxed, norm

TASKS = ["unpin", "fact", "chain", "code", "docq", "docf"]


def replies(text):
    return [p.split("<|im_end|>")[0] for p in text.split("<|im_start|>assistant\n")[1:]]


def key(s): return re.sub(r"[^0-9a-z\-]", "", (s or "").lower().replace("\\text", ""))


def num(s):
    s = norm(s or "").replace(",", "")
    try: return float(s)
    except ValueError: return None


def score(r):
    rp = replies(r["text"]); n_t = len(r.get("turns", [1])); reached = len(rp) >= n_t
    out = {"reached": reached, "ok": False}
    if not reached: return out
    last = rp[-1]; b = boxed(last)
    t = r.get("task", "")
    if t == "unpin": out["ok"] = correct({"text": last, "gold": r["gold"]})
    elif t == "chain":
        out["ok"] = b is not None and num(b) is not None and num(b) == float(r["gold"])
        a1, a2 = num(boxed(rp[0])), num(boxed(rp[1]))
        out["solved"] = [correct({"text": rp[k], "gold": r["golds"][k]}) for k in (0, 1)]
        if a1 is not None and a2 is not None: out["own"] = b is not None and num(b) is not None and abs(num(b) - (a1 + a2)) < 1e-6
    elif t == "code": out["ok"] = b is not None and num(b) is not None and num(b) == float(r["gold"])
    else: out["ok"] = b is not None and key(b) == key(r["gold"])
    return out


for f in sys.argv[1:]:
    rs = [json.loads(l) for l in open(f)]; S = {t: [] for t in TASKS}
    for r in rs: S.setdefault(r.get("task", "?"), []).append((r, score(r)))
    parts = []; tot = [0, 0]
    for t, xs in S.items():
        if not xs: continue
        ok = sum(s["ok"] for _, s in xs); tot[0] += ok; tot[1] += len(xs); p = f"{t} {ok}/{len(xs)}"
        if t == "chain":
            own = [s["own"] for _, s in xs if "own" in s]; sv = sum(sum(s.get("solved", [])) for _, s in xs)
            p += f" (own-answer sum {sum(own)}/{len(own)}, turns 1-2 solved {sv}/{2 * len(xs)})"
        nr = sum(not s["reached"] for _, s in xs)
        if nr: p += f" [{nr} did not reach the last turn]"
        parts.append(p)
    print(f"{os.path.basename(f)}: total {tot[0]}/{tot[1]} | " + " | ".join(parts))
