#!/usr/bin/env python3
"""Three-turn conversations for the GRPO screens (2026-10-10, the user: a copy that answers the second and third turn
better, and replies that fit better, is worth taking even when single-turn accuracy stays where it is). The two-turn br3
set has no third turn, so this builds one from MuSiQue's linear three-hop chains (3hop1, the train split - the rounds train
on its 2-hop questions only, so no question here is trained on): each hop becomes one turn, the second and third turns
referring back to the previous answer the way a person would ("Where was he born?") instead of naming it. The teacher
writes the turns; each must leave the earlier answers unnamed, every gold is the hop's own answer (1-4 words).
Dialogues {id, kind: "chain", turns: [{q, gold, standalone}]} as the other mt_eval sets.

  DSK_KEY=... python3 chain3_items.py --n 60 --out /root/work/mt_eval_chain3.jsonl
"""
import argparse, json, os, random, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=60); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--jsonl", default="", help="a local copy of musique_ans_v1.0_train.jsonl (else downloaded)")
ap.add_argument("--model", default="deepseek-flash")
A = ap.parse_args()
KEY = os.environ.get("DSK_KEY", "")


def norm(s): return " ".join(re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split())


def short(a):
    a = " ".join((a or "").split())
    return a if a and 1 <= len(a.split()) <= 4 and len(a) >= 2 else ""


SYS = """You turn a three-step fact chain into a short conversation a curious person might have with an assistant: three user messages, one per step, each answered by the step's answer. Reply with JSON only:
{"turns": ["<message 1>", "<message 2>", "<message 3>"], "standalone": ["<message 1 on its own>", "<message 2 on its own>", "<message 3 on its own>"]}
Message 1 asks step 1 as a plain, natural question. Messages 2 and 3 are follow-ups: each refers to the previous step's answer only indirectly (he, she, it, they, that band, that city, the company, ...), never by its name, and never names any earlier answer. Each message asks exactly one thing, the step's fact, in everyday wording; no "#1", no "according to". The standalone versions ask the same thing with the earlier answers written out."""


def ask(chain):
    hops = "\n".join(f"step {i + 1}: {h['question']}  (answer: {h['answer']})" for i, h in enumerate(chain))
    body = {"model": A.model, "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": "In the steps, #1 and #2 stand for the answers of steps 1 and 2; a step written as 'X >> relation' asks for the relation of X.\n\n" + hops}],
            "max_tokens": 1500, "temperature": 0.0, "response_format": {"type": "json_object"}}
    for _ in range(3):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request("https://api.deepseek.com/chat/completions", data=json.dumps(body).encode(),
                headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}), timeout=120))
            c = d["choices"][0]["message"].get("content") or ""
            return json.loads(c[c.find("{"): c.rfind("}") + 1])
        except Exception: time.sleep(3)
    return None


p = A.jsonl
if not p:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("dgslibisey/MuSiQue", "musique_ans_v1.0_train.jsonl", repo_type="dataset")
cands, tails = [], set()
for l in open(p):
    if not l.strip(): continue
    r = json.loads(l)
    if not r["id"].startswith("3hop1__") or not r.get("answerable", True): continue
    ch = r.get("question_decomposition") or []
    if len(ch) != 3 or not all(short(h.get("answer")) for h in ch): continue
    if len({norm(h["answer"]) for h in ch}) < 3: continue
    tail = (norm(ch[1]["answer"]), norm(ch[2]["answer"]))   # many chains share their last two hops (three Nielsen works -> his wife -> Copenhagen)
    if tail in tails: continue
    tails.add(tail)
    cands.append((r["id"], ch, [a for a in [r.get("answer")] + list(r.get("answer_aliases") or []) if short(a)]))
random.Random(A.seed).shuffle(cands)
cands = cands[:int(A.n * 1.8)]
with ThreadPoolExecutor(max_workers=8) as ex: outs = list(ex.map(lambda c: ask(c[1]), cands))
kept, why = [], {"no reply": 0, "names an earlier answer": 0, "shape": 0}
for (rid, ch, finals), o in zip(cands, outs):
    if not o: why["no reply"] += 1; continue
    t, s = o.get("turns") or [], o.get("standalone") or []
    if len(t) != 3 or len(s) != 3 or not all(isinstance(x, str) and 3 <= len(x.split()) <= 40 for x in t + s): why["shape"] += 1; continue
    if any(norm(ch[j]["answer"]) in norm(t[i]) for i in (1, 2) for j in range(i)) or "#" in "".join(t): why["names an earlier answer"] += 1; continue
    golds = [ch[0]["answer"], ch[1]["answer"], finals[0] if finals else ch[2]["answer"]]
    kept.append({"id": f"chain{len(kept):04d}", "kind": "chain", "src": rid,
                 "turns": [{"q": " ".join(t[i].split()), "gold": short(golds[i]), "standalone": " ".join(s[i].split())} for i in range(3)]})
    if len(kept) >= A.n: break
open(A.out, "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in kept))
print(f"CHAIN3_DONE {len(kept)} dialogues ({len(cands)} chains asked; dropped: {why}) -> {A.out}", flush=True)
