#!/usr/bin/env python3
"""Simulated conversations for the multi-turn fluency stage: gpt-5-nano plays the user, R1 (DeepSeek) the reference
assistant. Each dialogue is 3-5 user turns; the MOVE of each turn is scheduled here (so switches and follow-ups are
guaranteed, not left to the simulator's taste) and nano writes the message in character:

  open      - the first message: a factual question from the seed pool (gold kept), phrased as a person would.
  follow    - a follow-up on the assistant's last reply (a detail it mentioned, a why/how, "and what about ...").
  refer     - refers back to something named earlier with a pronoun or "that ..." (not the last turn only).
  switch    - a new topic: the NEXT seed question, rephrased casually, no link to what came before (gold kept).
  correct   - pushes back or corrects ("no, I meant ...", "are you sure?").
  chat      - small talk or a personal remark (no question to look up).

R1 answers every turn given the whole conversation (its final answer only, no reasoning) - the reference the GRPO
judge scores the model's replies against, and the history the next turn is written on. Rows:
  {id, persona, turns: [{move, q, gold, ref}], ...}

  python3 mt_sim.py --seeds /root/work/selfq_all.jsonl --exclude /root/work/eval300.jsonl --out /root/work/mt_sim.jsonl --n 200
"""
import argparse, json, random, time, urllib.request, urllib.error, concurrent.futures as cf
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", required=True); ap.add_argument("--exclude", default=""); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=200); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--user-model", default="gpt-5-nano"); ap.add_argument("--ref-model", default="deepseek-reasoner")
ap.add_argument("--ref-fallback", default="deepseek-flash")
A = ap.parse_args()
OAI = open("/root/.oai").read().strip(); DSK = open("/root/.dsk").read().strip()


def post(url, key, body, timeout=240):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode(),
                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}), timeout=timeout))


def nano(system, user, tries=4):
    body = {"model": A.user_model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_completion_tokens": 3000, "response_format": {"type": "json_object"}}
    for t in range(tries):
        try:
            c = post("https://api.openai.com/v1/chat/completions", OAI, body)["choices"][0]["message"].get("content") or ""
            return json.loads(c[c.find("{"): c.rfind("}") + 1])
        except Exception:
            time.sleep(3 * (t + 1))
    return None


REF = [A.ref_model]
def r1(messages, tries=3):
    sys_msg = {"role": "system", "content": "You are a helpful, friendly assistant in a chat. Answer the user's LATEST message; use the earlier conversation only where the latest message depends on it. Two to four sentences, conversational, accurate; if a fact is uncertain, say so briefly."}
    for model in (REF[0], A.ref_fallback):
        body = {"model": model, "messages": [sys_msg] + messages, "max_tokens": 4000}
        for t in range(tries):
            try:
                m = post("https://api.deepseek.com/chat/completions", DSK, body)["choices"][0]["message"]
                c = (m.get("content") or "").strip()
                if c: return c, model
            except urllib.error.HTTPError as e:
                if e.code in (400, 404): break   # the model name is not served: fall back
                time.sleep(3 * (t + 1))
            except Exception:
                time.sleep(3 * (t + 1))
        REF[0] = A.ref_fallback
    return None, None


USER_SYS = ("You role-play a person chatting with an AI assistant. Write only that person's next message, in character: natural, often casual, "
            "sometimes terse - ONE or TWO short sentences, like a real chat message. Never state your persona, job or age outright; let it show only "
            "in tone. Output JSON {\"message\": \"...\"}.")
PERSONAS = ["a retired teacher who likes history", "a college student cramming for a quiz", "a curious 12-year-old", "a sports fan in a bar argument",
            "a nurse on a night shift", "a software engineer on a coffee break", "a novelist researching details", "a tourist planning a trip",
            "a parent helping with homework", "a trivia-night regular", "a film buff", "a music producer"]
MOVES = {
    "follow": "Ask a natural follow-up about something the assistant just said (a detail it mentioned, why/how, or 'and what about ...').",
    "refer": "Ask about something named EARLIER in the conversation (not necessarily the last reply) but refer to it only with a pronoun or a phrase like 'that film' / 'that guy' - do not repeat its name.",
    "correct": "Briefly push back on or question the assistant's last answer ('are you sure?', 'no, I meant ...', 'that doesn't sound right') - one sentence, no long argument.",
    "chat": "Say something personal or make small talk related loosely to the conversation (a feeling, an anecdote, a plan) - not a question to look up.",
}


def transcript(turns):
    return "\n".join(f"USER: {t['q']}\nASSISTANT: {t['ref']}" for t in turns)


def make(args):
    i, persona, seeds = args
    rng = random.Random(A.seed * 1000 + i)
    n_turns = rng.choice([3, 4, 4, 5])
    plan = ["open"] + [rng.choices(["follow", "refer", "switch", "correct", "chat"], weights=[3, 1, 3, 1, 2])[0] for _ in range(n_turns - 1)]   # names are the memory bank's job (the user, 2026-09-29): references back are kept rare
    if "switch" not in plan: plan[rng.randrange(1, n_turns)] = "switch"
    turns, si, msgs = [], 0, []
    for mv in plan:
        gold = ""
        if mv in ("open", "switch"):
            q0, gold = seeds[si]; si += 1
            ask = (f"You are {persona}. Ask this, in your own words (keep every name and fact needed to answer it): {q0}" if mv == "open" else
                   f"You are {persona}. Conversation so far:\n{transcript(turns)}\n\nNow change the subject completely and ask this, in your own words, with no link to what came before (keep every name and fact needed to answer it): {q0}")
        else:
            ask = f"You are {persona}. Conversation so far:\n{transcript(turns)}\n\n{MOVES[mv]}"
        u = nano(USER_SYS, ask)
        if not u or not u.get("message"): return None
        q = u["message"].strip(); msgs = msgs + [{"role": "user", "content": q}]
        ref, model = r1(msgs)
        if not ref: return None
        msgs = msgs + [{"role": "assistant", "content": ref}]
        turns.append({"move": mv, "q": q, "gold": gold, "ref": ref, "ref_model": model})
    return {"id": f"sim{i:04d}", "persona": persona, "turns": turns}


seeds = []
for l in open(A.seeds):
    try: r = json.loads(l)
    except Exception: continue
    if r.get("q") and r.get("gold"): seeds.append((r["q"].strip(), r["gold"].strip()))
ex = set()
if A.exclude:
    for l in open(A.exclude):
        try: ex.add(json.loads(l).get("q", "").strip())
        except Exception: pass
seeds = [x for x in dict.fromkeys(seeds) if x[0] not in ex]; random.Random(A.seed).shuffle(seeds)
jobs = [(i, PERSONAS[i % len(PERSONAS)], seeds[3 * i: 3 * i + 3]) for i in range(min(A.n, len(seeds) // 3))]
out = []; t0 = time.time()
with cf.ThreadPoolExecutor(A.workers) as pool:
    for k, d in enumerate(pool.map(make, jobs)):
        if d: out.append(d)
        if (k + 1) % 20 == 0: print(f"[mtsim] {k+1}/{len(jobs)} dialogues, {len(out)} kept, {time.time()-t0:.0f}s (reference model {REF[0]})", flush=True)
with open(A.out, "w") as fh:
    for d in out: fh.write(json.dumps(d, ensure_ascii=False) + "\n")
import collections
mv = collections.Counter(t["move"] for d in out for t in d["turns"])
print(f"MTSIM_DONE {len(out)} dialogues, {sum(mv.values())} turns {dict(mv)} -> {A.out} (reference model {REF[0]})", flush=True)
