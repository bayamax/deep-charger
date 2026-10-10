#!/usr/bin/env python3
"""How the screens' conversations read as conversations (2026-10-10, the user: with an LLM scoring the replies, score the
conversation's quality too). reply_judge.py scores each correct reply on its own; this reads each multi-turn dialogue whole
(every user message and the model's reply, no answers given) and the nano judge says whether it holds together:
follows_the_thread (each reply answers its own message, a follow-up's "he" / "that city" taken as the earlier turn meant it),
consistent (nothing contradicts an earlier turn), no_carryover (no earlier answer repeated or earlier topic dragged in where
the new message does not ask for it), natural (a natural spoken exchange, replies of fitting length, not templated).
A dialogue is good when all four hold. Correctness is not judged here. Writes <file>_cq.jsonl and prints
  CQ <tag> good G/D (follows_the_thread F, consistent C, no_carryover N, natural T)

  OAI_KEY=... python3 conv_judge.py /root/work/ch3_<tag>.jsonl <tag>
"""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
KEYS = ("follows_the_thread", "consistent", "no_carryover", "natural")
SYS = """You are reading a whole conversation between a user and a small assistant that can search an encyclopedia. You judge how the conversation reads, not whether its facts are right. Reply with JSON only:
{"follows_the_thread": true/false, "consistent": true/false, "no_carryover": true/false, "natural": true/false}
follows_the_thread means every reply answers the message it follows, and a follow-up that points back ("he", "that city", "the company") is taken to mean what the earlier turn was about; consistent means no reply contradicts an earlier one; no_carryover means no reply repeats an earlier answer or drags the earlier topic in where the new message does not ask for it; natural means it reads as a natural spoken exchange: replies of fitting length, not templated, no tool tags, no repetition loops, nothing left unfinished. An empty reply fails follows_the_thread and natural."""
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]; tag = sys.argv[2]; key = os.environ.get("OAI_KEY", "")
out = sys.argv[1].replace(".jsonl", "_cq.jsonl")
dl = {}
for r in rows: dl.setdefault(r["dialog"], {})[r["turn"]] = r
dialogs = [(d, [t[i] for i in sorted(t)]) for d, t in dl.items() if len(t) == max(x["n_turns"] for x in t.values())]


def judge(item):
    d, turns = item
    conv = "\n\n".join(f"USER: {t['q'][:600]}\nASSISTANT: {(t.get('reply') or '').strip()[:1200] or '(no reply)'}" for t in turns)
    body = {"model": "gpt-5-nano", "max_completion_tokens": 2000, "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": "CONVERSATION:\n\n" + conv}]}
    err = "?"
    for _ in range(3):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}), timeout=120))
            c = r["choices"][0]["message"].get("content") or ""; return json.loads(c[c.find("{"): c.rfind("}") + 1])
        except Exception as e: time.sleep(3); err = type(e).__name__
    return {"error": err}


if os.path.exists(out) and sum(1 for l in open(out) if l.strip()) == len(dialogs):
    res = [json.loads(l)["v"] for l in open(out) if l.strip()]   # already judged: the screens are re-run and reuse it
else:
    with ThreadPoolExecutor(max_workers=8) as ex: res = list(ex.map(judge, dialogs))
    with open(out, "w") as o:
        for (d, _), v in zip(dialogs, res): o.write(json.dumps({"dialog": d, "v": v}, ensure_ascii=False) + "\n")
good = sum(all(bool(v.get(k)) for k in KEYS) for v in res)
part = ", ".join(f"{k} {sum(bool(v.get(k)) for v in res)}" for k in KEYS)
print(f"CQ {tag} good {good}/{len(dialogs)} ({part}) judge errors {sum(1 for v in res if 'error' in v)}", flush=True)
