#!/usr/bin/env python3
"""The Dolphin reasoning screen: nano judges each held-out reply against the R1 reference (the judge the mix0 run introduced).
  OAI_KEY=... python3 dl_judge.py /root/work/dl_<tag>.jsonl <tag>    (DL_REF: the held-out hundred, default /root/work/dolphin_heldout100.jsonl)
"""
import json, os, sys, urllib.request, time
from concurrent.futures import ThreadPoolExecutor
src = open("/root/work/online_loop.py").read(); i = src.index("REASON_SYS = "); j = src.index('"""', src.index('"""', i) + 3) + 3
ns = {}; exec(src[i:j], ns); SYS = ns["REASON_SYS"]
ref = {json.loads(l)["q"].strip(): json.loads(l)["ref"] for l in open(os.environ.get("DL_REF", "/root/work/dolphin_heldout100.jsonl")) if l.strip()}
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]; key = os.environ.get("OAI_KEY", "")
def judge(r):
    t = r["text"]; reply = t.split("</think>")[-1].strip() if "</think>" in t else ""
    if not reply: return 0, "unfinished"
    body = {"model": "gpt-5-nano", "max_completion_tokens": 2000, "messages": [{"role": "system", "content": SYS},
            {"role": "user", "content": f"QUESTION:\n{r['q'][:2000]}\n\nREFERENCE ANSWER:\n{ref.get(r['q'].strip(), '')[:3000]}\n\nASSISTANT ANSWER:\n{reply[:3000]}"}]}
    err = "?"
    for _ in range(3):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}), timeout=120))
            c = d["choices"][0]["message"].get("content") or ""; v = json.loads(c[c.find("{"): c.rfind("}") + 1])
            return int(all(bool(v.get(k)) for k in ("solves_it", "follows_the_request", "language_english", "clean"))), v
        except Exception as e: time.sleep(3); err = type(e).__name__
    return 0, {"error": err}
with ThreadPoolExecutor(max_workers=8) as ex: res = list(ex.map(judge, rows))
with open(sys.argv[1].replace(".jsonl", "_judged.jsonl"), "w") as o:
    for r, (a, v) in zip(rows, res): o.write(json.dumps({"q": r["q"], "pass": a, "why": str(v)[:300], "text": r["text"]}, ensure_ascii=False) + "\n")
ok = sum(a for a, _ in res); unf = sum(1 for _, v in res if v == "unfinished"); errs = sum(1 for _, v in res if isinstance(v, dict) and "error" in v)
print(f"[mix0] DOLPHIN {sys.argv[2]} {100*ok/max(len(rows),1):.1f}% ({ok}/{len(rows)}) unfinished {unf} judge errors {errs}", flush=True)
