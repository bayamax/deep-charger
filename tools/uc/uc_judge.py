#!/usr/bin/env python3
"""Which reply the person would rather have, the problem solved first: a strong judge reads the person's message and two
replies, both orders. Step 1, does each reply actually solve what was asked (right facts, working reasoning or code, sound
advice, the text produced, nothing made up) - a solved reply always beats an unsolved one; step 2, only between equally
solved replies, which one this person would prefer (clear, useful, a length for a phone chat, the tone). Many of these
prompts have no single right answer, so the judge decides; the solve rate of each side is reported beside the preference. A pair is a win only when the two orders agree (or one
says win and the other tie); otherwise a tie. Prints the win / tie / loss of A against B, overall and per kind.

  OPENAI_API_KEY=... python3 -I uc_judge.py --a dc.jsonl --b ans_4omini.jsonl --out j.jsonl --judge gpt-5.4
  (each file: {"id", "q", "kind", "reply"} per line; rows matched by id)
"""
import argparse, collections, json, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--judge", default="gpt-5.4-mini", help="gpt-5.4-mini for the routine reads (its totals within 2.5 points of gpt-5.4 on the 4o vs 4o-mini check, the winner flipped on 6%% of pairs); gpt-5.4 for a final call"); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
A = ap.parse_args()
KEY = os.environ.get("OPENAI_API_KEY", "")
SYS = ("A person sent the message below to an assistant on their phone. You see two replies. Judge in two steps.\n"
       "STEP 1 - the problem itself. For each reply decide whether it actually solves what was asked: the facts are right, the "
       "reasoning or code works, the advice is sound, the requested text is produced, nothing is made up. A reply that is wrong, "
       "fabricates, dodges, or does not do the task is NOT solved, however well written.\n"
       "STEP 2 - only between replies that are equally solved (both solved, or both not): which one would THIS person rather receive - "
       "clear and useful, a length right for a phone chat, the right tone. Do not prefer a reply for being longer.\n"
       "A solved reply always beats an unsolved one. If they are equal on both steps, say tie. Reply with JSON only: "
       '{"reply1_solved": true/false, "reply2_solved": true/false, "reply1_has_error": true/false, "reply2_has_error": true/false, '
       '"winner": "1" | "2" | "tie", "why": "one short sentence"}')


def ask(q, r1, r2):
    msg = f"MESSAGE:\n{q[:3000]}\n\nREPLY 1:\n{(r1 or '(empty)')[:4000]}\n\nREPLY 2:\n{(r2 or '(empty)')[:4000]}"
    body = {"model": A.judge, "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": msg}], "max_completion_tokens": 2000,
            "response_format": {"type": "json_object"}}
    for t in range(4):
        try:
            req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
            d = json.load(urllib.request.urlopen(req, timeout=300)); return json.loads(d["choices"][0]["message"]["content"]), d.get("usage", {})
        except Exception as ex:
            err = ex; time.sleep(3 * (t + 1))
    return {"winner": "error", "why": str(err)[:120]}, {}


def judge(pair):
    a, b = pair
    j1, u1 = ask(a["q"], a["reply"], b["reply"])   # A first
    j2, u2 = ask(a["q"], b["reply"], a["reply"])   # B first
    v1 = {"1": "A", "2": "B", "tie": "tie"}.get(str(j1.get("winner")), "err"); v2 = {"1": "B", "2": "A", "tie": "tie"}.get(str(j2.get("winner")), "err")
    s = {v1, v2} - {"tie"}
    res = "err" if "err" in (v1, v2) else ("tie" if not s or len(s) == 2 else s.pop())
    err_a = bool(j1.get("reply1_has_error")) + bool(j2.get("reply2_has_error")); err_b = bool(j1.get("reply2_has_error")) + bool(j2.get("reply1_has_error"))
    sol_a = bool(j1.get("reply1_solved")) + bool(j2.get("reply2_solved")); sol_b = bool(j1.get("reply2_solved")) + bool(j2.get("reply1_solved"))
    return {"id": a["id"], "kind": a.get("kind"), "q": a["q"], "result": res, "orders": [v1, v2], "a_error": err_a, "b_error": err_b, "a_solved": sol_a, "b_solved": sol_b,
            "why": [j1.get("why"), j2.get("why")]}, [u1, u2]


ra = {json.loads(l)["id"]: json.loads(l) for l in open(A.a) if l.strip()}; rb = {json.loads(l)["id"]: json.loads(l) for l in open(A.b) if l.strip()}
ids = [i for i in ra if i in rb]
if A.limit: ids = ids[:A.limit]
done = {}
if os.path.exists(A.out):
    for l in open(A.out):
        try: d = json.loads(l); done[d["id"]] = d
        except Exception: pass
todo = [(ra[i], rb[i]) for i in ids if i not in done]; tin = tout = 0
with open(A.out, "a") as fh, ThreadPoolExecutor(A.threads) as ex:
    for res, us in ex.map(judge, todo):
        done[res["id"]] = res; fh.write(json.dumps(res, ensure_ascii=False) + "\n"); fh.flush()
        for u in us: tin += u.get("prompt_tokens", 0); tout += u.get("completion_tokens", 0)
R = [done[i] for i in ids if i in done and done[i]["result"] != "err"]
def line(rs):
    n = len(rs); w = sum(r["result"] == "A" for r in rs); t = sum(r["result"] == "tie" for r in rs); l = n - w - t
    sa = sum(r.get("a_solved", 0) for r in rs) / 2; sb = sum(r.get("b_solved", 0) for r in rs) / 2
    return (f"solved A {100 * sa / max(n, 1):.0f}% B {100 * sb / max(n, 1):.0f}% | preferred: win {w} tie {t} loss {l} of {n} (A {100 * (w + t / 2) / max(n, 1):.1f}%; "
            f"errors A {sum(r['a_error'] > 0 for r in rs)} B {sum(r['b_error'] > 0 for r in rs)})")
print(f"[judge {A.judge}] A={os.path.basename(A.a)} vs B={os.path.basename(A.b)}: " + line(R), flush=True)
K = collections.defaultdict(list)
for r in R: K[r["kind"]].append(r)
for k, rs in sorted(K.items(), key=lambda x: -len(x[1])): print(f"[judge]   {k:<10} " + line(rs), flush=True)
print(f"[judge] tokens in {tin} out {tout} this run", flush=True)
