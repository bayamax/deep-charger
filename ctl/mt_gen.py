#!/usr/bin/env python3
"""Multi-turn dialogues with gold answers, built from single-turn questions by the teacher (DeepSeek).

Kinds:
  bridge  - a two-hop question split into two turns: turn 1 asks the first hop (gold: the bridge entity), turn 2
            asks for the final answer referring back only by pronoun / "that ..." (gold: the original answer).
  memory  - turn 1: the user mentions a personal detail in passing; turn 2: an unrelated question that needs a search
            (gold); turn 3: the user asks the assistant to recall the detail (gold: the detail).
  switch  - two unrelated questions in a row (both gold): history must not hurt an independent turn.

  python3 mt_gen.py --seeds /root/work/eval300.jsonl --out /root/work/mt_eval.jsonl --n-bridge 40 --n-memory 30 --n-switch 20
  python3 mt_gen.py --seeds /root/work/selfq_all.jsonl --exclude /root/work/eval300.jsonl --out /root/work/mt_train.jsonl ...
"""
import argparse, json, random, re, sys, time, urllib.request, urllib.error, concurrent.futures as cf
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", required=True); ap.add_argument("--exclude", default=""); ap.add_argument("--out", required=True)
ap.add_argument("--n-bridge", type=int, default=40); ap.add_argument("--n-memory", type=int, default=30); ap.add_argument("--n-switch", type=int, default=20)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--model", default="deepseek-flash")
A = ap.parse_args()
KEY = open("/root/.dsk").read().strip()
def chat(system, user, tries=4):
    body = {"model": A.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 800, "temperature": 0.7, "response_format": {"type": "json_object"}}
    for t in range(tries):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request("https://api.deepseek.com/chat/completions", data=json.dumps(body).encode(),
                headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}), timeout=120))
            return json.loads(d["choices"][0]["message"]["content"])
        except Exception as e:
            time.sleep(3 * (t + 1))
    return None
seeds = []
for l in open(A.seeds):
    try: r = json.loads(l)
    except Exception: continue
    q, g = (r.get("q") or "").strip(), (r.get("gold") or "").strip()
    if q and g: seeds.append((q, g))
if A.exclude:
    ex = set()
    for l in open(A.exclude):
        try: ex.add(json.loads(l).get("q", "").strip())
        except Exception: pass
    seeds = [x for x in seeds if x[0] not in ex]
seeds = list(dict.fromkeys(seeds)); random.seed(A.seed); random.shuffle(seeds)
print(f"[mtgen] {len(seeds)} seed questions", flush=True)
BRIDGE_SYS = ("You turn a two-hop trivia question into a natural two-turn chat between a user and an assistant. Output JSON only.")
BRIDGE_USER = """Question: {q}
Answer: {g}

Split it into two user turns:
- turn1: a question for the FIRST hop, whose answer is the intermediate entity (the bridge) the final question depends on.
- turn2: the follow-up asking for the final answer ({g}). It must refer to the turn-1 answer ONLY by a pronoun or a phrase like "that film" / "that company" - it must NOT name the bridge entity, and it must not be answerable without turn 1.
Both turns should sound like a person chatting (short, casual is fine). If the question cannot be split this way (single hop, comparison, yes/no), output {{"skip": true}}.
Output: {{"bridge": "<the intermediate entity, a short name>", "turn1": "...", "turn2": "..."}}"""
MEMORY_SYS = "You write short, natural things a person says to a chat assistant. Output JSON only."
MEMORY_USER = """Write (1) a casual remark a user makes to an assistant at the start of a chat, mentioning in passing ONE specific, memorable personal detail - a person's name, a pet's name, a town, a number, a date - inside a sentence about their plans, family, work or hobbies (vary the topic: {topic}); (2) a question the same user asks much later asking the assistant to recall that detail WITHOUT restating it (e.g. "what did I say my dog's name was?"). The detail must be a short string that the correct recall answer would contain verbatim.
Output: {{"remark": "...", "detail": "...", "recall": "..."}}"""
TOPICS = ["a trip", "a pet", "a sibling", "a new job", "a hobby", "a birthday", "moving house", "a recipe", "a sports team", "a book club", "a garden", "a car", "a wedding", "a course they are taking", "a neighbour"]
def make_bridge(qg):
    q, g = qg; r = chat(BRIDGE_SYS, BRIDGE_USER.format(q=q, g=g))
    if not r or r.get("skip") or not r.get("turn1") or not r.get("turn2") or not r.get("bridge"): return None
    b = r["bridge"].strip()
    if b.lower() in r["turn2"].lower(): return None   # the follow-up must not name the bridge
    return {"kind": "bridge", "seed": q, "turns": [{"q": r["turn1"].strip(), "gold": b, "standalone": r["turn1"].strip()},
                                                   {"q": r["turn2"].strip(), "gold": g, "standalone": q}]}
def make_memory(args):
    (q, g), topic = args; r = chat(MEMORY_SYS, MEMORY_USER.format(topic=topic))
    if not r or not r.get("remark") or not r.get("detail") or not r.get("recall"): return None
    det = r["detail"].strip()
    if det.lower() not in r["remark"].lower() or det.lower() in r["recall"].lower(): return None
    rem, rec = r["remark"].strip(), r["recall"].strip()
    return {"kind": "memory", "seed": q, "turns": [{"q": rem, "gold": "", "standalone": rem}, {"q": q, "gold": g, "standalone": q},
                                                   {"q": rec, "gold": det, "standalone": f'Earlier in our chat I told you: "{rem}" {rec}'}]}
out = []; i = 0
with cf.ThreadPoolExecutor(A.workers) as ex:
    pool = seeds[:]
    bridges = [x for x in ex.map(make_bridge, pool[:int(A.n_bridge * 2.5)]) if x][:A.n_bridge]; pool = pool[int(A.n_bridge * 2.5):]
    print(f"[mtgen] bridge {len(bridges)}", flush=True)
    mem = [x for x in ex.map(make_memory, [(pool[k], TOPICS[k % len(TOPICS)]) for k in range(int(A.n_memory * 1.5))]) if x][:A.n_memory]; pool = pool[int(A.n_memory * 1.5):]
    print(f"[mtgen] memory {len(mem)}", flush=True)
sw = [{"kind": "switch", "seed": pool[2 * k][0], "turns": [{"q": pool[2 * k][0], "gold": pool[2 * k][1], "standalone": pool[2 * k][0]},
                                                           {"q": pool[2 * k + 1][0], "gold": pool[2 * k + 1][1], "standalone": pool[2 * k + 1][0]}]} for k in range(min(A.n_switch, len(pool) // 2))]
allv = bridges + mem + sw
for k, d in enumerate(allv): d["id"] = f"{d['kind']}{k:04d}"
random.shuffle(allv)
with open(A.out, "w") as fh:
    for d in allv: fh.write(json.dumps(d, ensure_ascii=False) + "\n")
print(f"MTGEN_DONE {len(allv)} dialogues (bridge {len(bridges)}, memory {len(mem)}, switch {len(sw)}) -> {A.out}", flush=True)
