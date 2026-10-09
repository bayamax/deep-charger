#!/usr/bin/env python3
"""How well the screens' replies are written (2026-10-10, the user: a copy whose second- and third-turn replies fit better
is worth taking even when single-turn accuracy stays where it is). Every reply that is correct is scored by the nano judge
on the rubric the GRPO reward uses (online_loop.py SEARCH_SYS, plus MT_SYS for a turn after the first, with the earlier
exchange shown): sound, natural, clean. A reply is good when it is correct and all three hold; a wrong reply is never good.
Works on pool_eval.py output, single-turn rows or multi-turn rows (dialog, turn). Writes <file>_rq.jsonl and prints
  RQ <tag> good G/N (correct C) turn0 g/c turn1 g/c ...

  OAI_KEY=... python3 reply_judge.py /root/work/br3_<tag>.jsonl <tag>
"""
import json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
src = open("/root/work/online_loop.py").read()
def const(name):
    i = src.index(name + " = "); j = src.index('"""', src.index('"""', i) + 3) + 3; ns = {}; exec(src[i:j], ns); return ns[name]
SEARCH_SYS, MT_SYS = const("SEARCH_SYS"), const("MT_SYS")
INFO = re.compile(r"<information>(.*?)</information>", re.S)
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]; tag = sys.argv[2]; key = os.environ.get("OAI_KEY", "")
out = sys.argv[1].replace(".jsonl", "_rq.jsonl")
prev = {}
for r in rows:
    if "dialog" in r: prev.setdefault(r["dialog"], {})[r["turn"]] = r


def reply_of(r):
    if "reply" in r: return r["reply"] or ""
    t = r.get("text", ""); return re.sub(r"<｜end▁of▁sentence｜>.*", "", t.split("</think>")[-1], flags=re.S).strip() if "</think>" in t else ""


def judge(r):
    reply = reply_of(r)
    if not r.get("correct") or not reply: return {}
    served = "\n\n".join(m.strip() for m in INFO.findall(r.get("text", "")))[-4000:]
    sys_, ctx = SEARCH_SYS, ""
    if r.get("turn", 0) > 0:
        h = [prev[r["dialog"]][i] for i in range(r["turn"]) if i in prev.get(r["dialog"], {})]
        ctx = "EARLIER IN THE CONVERSATION:\n" + "\n".join(f"USER: {x['q'][:600]}\nASSISTANT: {reply_of(x)[:600] or '(no reply)'}" for x in h) + "\n\nNEW QUESTION:\n"
        sys_ = SEARCH_SYS + MT_SYS
    body = {"model": "gpt-5-nano", "max_completion_tokens": 2000, "messages": [{"role": "system", "content": sys_},
            {"role": "user", "content": f"{ctx or 'QUESTION:'}{chr(10) if not ctx else ''}{r['q']}\n\nWHAT THE SEARCH RETURNED:\n{served}\n\nREPLY:\n{reply[:2000]}"}]}
    err = "?"
    for _ in range(3):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}), timeout=120))
            c = d["choices"][0]["message"].get("content") or ""; return json.loads(c[c.find("{"): c.rfind("}") + 1])
        except Exception as e: time.sleep(3); err = type(e).__name__
    return {"error": err}


if os.path.exists(out) and sum(1 for l in open(out) if l.strip()) == len(rows):
    res = [json.loads(l)["v"] for l in open(out) if l.strip()]   # already judged: the screens are re-run and reuse it
else:
    with ThreadPoolExecutor(max_workers=8) as ex: res = list(ex.map(judge, rows))
    with open(out, "w") as o:
        for r, v in zip(rows, res): o.write(json.dumps({"q": r["q"], "turn": r.get("turn", 0), "correct": bool(r.get("correct")), "v": v}, ensure_ascii=False) + "\n")
good = [bool(r.get("correct")) and all(bool(v.get(k)) for k in ("sound", "natural", "clean")) for r, v in zip(rows, res)]
by = {}
for r, g in zip(rows, good):
    b = by.setdefault(r.get("turn", 0), [0, 0]); b[0] += g; b[1] += bool(r.get("correct"))
errs = sum(1 for v in res if "error" in v)
print(f"RQ {tag} good {sum(good)}/{len(rows)} (correct {sum(b[1] for b in by.values())}) " + " ".join(f"turn{t} {b[0]}/{b[1]}" for t, b in sorted(by.items())) + f" judge errors {errs}", flush=True)
