#!/usr/bin/env python3
"""NPC data from an open model (Qwen3.6-35B-A3B, Apache-2.0: its outputs may train a commercial model) under vLLM.

  train  character cards with concrete facts, and dialogues with a player whose every line has a set purpose (ask a
         fact, ask what the card does not say, tell one's name and later need it, be rude, talk about smartphones, ...)
         -> {"messages": [system card, user, assistant, ...], "card": {...}} per dialogue
  bench  held-out cards (their own seed and name pool) with probes: fact questions with the key words of the answer, a
         follow-up after a reference answer, a question the card cannot answer, an out-of-world line, a rude line, and
         the player's name to be recalled one turn later
  judge  grades replies (from npc_bench.py) against the card and the conversation: answers it, fits the card, in
         character, fluent, and usable as-is in a game

  python3 npc_gen.py train --n 10000 --out npc_train.jsonl
  python3 npc_gen.py bench --n 120 --out npc_bench.jsonl
  python3 npc_gen.py judge --inp replies.jsonl --out judged.jsonl
"""
import argparse, json, os, random, re, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["train", "bench", "judge"])
ap.add_argument("--n", type=int, default=100, help="requests (train: one card and two dialogues each; bench: one card each)")
ap.add_argument("--out", required=True); ap.add_argument("--inp", default="")
ap.add_argument("--from-raw", default="", help="parse the outputs a run kept (OUT.raw) again instead of generating")
ap.add_argument("--model", default="nvidia/Qwen3.6-35B-A3B-NVFP4"); ap.add_argument("--fallback", default="QuantTrio/Qwen3.6-35B-A3B-AWQ")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-len", type=int, default=8192); ap.add_argument("--gpu-mem", type=float, default=0.90)
A = ap.parse_args()
R = random.Random(A.seed + {"train": 0, "bench": 1000003, "judge": 7}[A.mode])

ROLES = ["blacksmith", "innkeeper", "town guard", "travelling merchant", "farmer", "village priest", "street kid", "ship captain", "herbalist",
         "librarian", "tavern bard", "retired adventurer", "mine foreman", "fisherwoman", "butler of a noble house", "stable hand", "wizard's apprentice",
         "bounty hunter", "baker", "ferryman", "hunter", "mayor", "alchemist", "gate keeper", "fortune teller", "station mechanic", "cafe owner",
         "robot shopkeeper", "scavenger", "school teacher", "lighthouse keeper", "postman", "museum curator", "taxi driver", "cat cafe owner",
         "tailor", "fletcher", "cook at the barracks", "gravedigger", "toll bridge keeper", "dock worker", "healer", "shrine maiden", "tea house owner",
         "ronin", "noodle stall owner", "pet shop owner", "arcade owner", "detective", "bartender", "hotel receptionist", "park ranger", "nurse",
         "pilot", "smuggler", "android repair tech", "colony doctor", "miner on an asteroid", "caravan leader", "camel trader", "oasis water seller",
         "sheriff", "saloon owner", "gold prospector", "train conductor", "witch in the woods", "goblin trader", "dwarf brewer", "elf archer",
         "orc chieftain's cook", "talking cat", "ghost of the manor", "knight of the order", "squire", "pirate quartermaster", "map maker",
         "jeweller", "pawn shop owner", "flower seller", "beekeeper", "shepherd", "miller", "carpenter", "doctor", "cobbler", "potter"]
SETTINGS = ["a small fantasy village", "a busy medieval port", "a desert trading post", "a snowy mountain town", "a pirate island", "a haunted forest edge",
            "a cozy farming valley", "a cyberpunk city street", "a space station market", "a post-apocalyptic outpost", "a modern seaside town",
            "a magic academy", "a royal capital", "a frontier town in the wild west", "a floating sky city", "an underground dwarf city", "a swamp village",
            "a fishing village on a lake", "a samurai-era castle town", "a quiet suburb in the 1990s", "a mining colony on Mars", "an elven forest city",
            "a crossroads inn on a trade road", "a ruined kingdom after a war", "a jungle river town", "a steampunk factory city", "a tiny island ferry stop",
            "a mountain monastery", "a goblin market under a bridge", "a seaside resort town"]
MOODS = ["cheerful", "grumpy", "shy", "suspicious", "tired", "chatty", "proud", "nervous", "kind", "sarcastic", "busy", "curious", "gloomy", "calm",
         "greedy", "honest", "forgetful", "brave", "lazy", "polite", "blunt", "dreamy", "strict", "playful"]
STYLES = ["short plain sentences", "warm and friendly words", "blunt words", "formal and polite words", "a lot of talk", "old-fashioned words",
          "sailor's slang", "casual slang", "dry jokes", "careful, quiet words", "loud, excited words", "simple words like a child"]
INTENTS = {
    "greet": "greets the NPC",
    "fact": "asks about something that IS in the NPC's facts, in the player's own words (do not copy the fact's wording)",
    "fact_indirect": "says something whose answer needs one of the facts, without asking it directly (e.g. 'I'm starving' when the facts name a place to eat)",
    "followup": "a short follow-up on the NPC's last answer that uses 'it', 'that', 'there', 'him' or 'her'",
    "unknown": "asks about something the facts do NOT cover; the NPC must not invent names, prices or places - it says it does not know, or sends the "
               "player to someone or somewhere named in its facts",
    "self": "tells the NPC something about themselves: their name, where they come from, or what they are looking for",
    "recall": "asks something that the NPC can only answer by remembering what the player said about themselves earlier in this dialogue",
    "smalltalk": "makes small talk (the weather, the day, how the NPC is doing)",
    "npc_life": "asks about the NPC personally (family, past, what they like)",
    "rude": "is rude or insulting; the NPC reacts as its character would, not as a helpful assistant",
    "meta": "says something from outside the game world (smartphones, the internet, 'are you an AI?', 'this is just a game'); the NPC stays inside "
            "its world and is puzzled or brushes it off",
    "bad_request": "asks for help with something violent or criminal in the game world; the NPC refuses, warns or is shocked, as its character would",
    "bargain": "tries to haggle or to get something for free",
    "short": "a very short, sloppy line in lowercase, maybe with a typo (like 'price?' or 'where inn')",
    "goodbye": "says goodbye",
}
NAMES_TRAIN = ["Alex", "Sam", "Mia", "Leo", "Ivy", "Noah", "Zoe", "Eli", "Ada", "Kai", "Rosa", "Finn", "Lena", "Omar", "Nell", "Hugo", "Iris", "Theo",
               "Yuki", "Ravi", "Maya", "Jonas", "Luca", "Nora", "Felix", "Sara", "Tom", "Ana", "Ben", "Cleo", "Dan", "Emma", "Gus", "Hana", "Ian"]
NAMES_BENCH = ["Marlo", "Tessa", "Quinn", "Bastian", "Juno", "Ronan", "Selma", "Otto", "Priya", "Wren"]
ASCII = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "—": " - ", "–": "-", "…": "...", " ": " "})
AI_TALK = re.compile(r"\b(as an ai|language model|i'm an ai|i am an ai|virtual assistant|ai assistant|as a character|role-?play|npc)\b", re.I)


def clean(s): return re.sub(r"\s+", " ", str(s).translate(ASCII)).strip()


def card_text(c, fmt):
    """three ways a game maker might write the card, so the model is not tied to one layout"""
    facts = [clean(f) for f in c["facts"]]
    if fmt == 0: return f"You are {c['name']}, {c['role']} in {c['place']}. {c['personality']}\nFacts:\n" + "\n".join(f"- {f}" for f in facts)
    if fmt == 1: return f"You are {c['name']}, {c['role']} in {c['place']}. {c['personality']} " + " ".join(facts)
    return (f"Name: {c['name']}\nRole: {c['role']}\nPlace: {c['place']}\nPersonality: {c['personality']}\nKnows:\n" + "\n".join(f"- {f}" for f in facts))


def plan():
    """a sequence of 3-6 player purposes: about half about the card's facts, 'recall' only after 'self', no off-card purpose twice running"""
    n = R.choice([3, 4, 4, 5, 5, 6]); seq = ["greet"] if R.random() < 0.35 else []
    other = ["unknown", "unknown", "self", "smalltalk", "npc_life", "rude", "meta", "bad_request", "bargain", "short"]
    while len(seq) < n - (1 if R.random() < 0.25 else 0):
        if R.random() < 0.45:
            seq.append(R.choice(["fact", "fact", "fact_indirect"]))
            if R.random() < 0.35: seq.append("followup")
            continue
        k = R.choice(other)
        if seq and k == seq[-1]: continue
        seq.append(k)
        if k == "self" and R.random() < 0.7:
            if R.random() < 0.5: seq.append(R.choice(["fact", "unknown", "smalltalk"]))
            seq.append("recall")
    if not any(k.startswith("fact") for k in seq): seq.insert(1 if seq and seq[0] == "greet" else 0, "fact")
    if len(seq) < n: seq.append("goodbye")
    return seq[:7]


SYL_A = ["Al", "Bar", "Cor", "Da", "El", "Fen", "Gar", "Hal", "Is", "Jor", "Ka", "Lu", "Mar", "Ned", "Or", "Pel", "Quen", "Ros", "Sa", "Tam",
         "Ul", "Ver", "Wil", "Yan", "Zel", "Bri", "Cal", "Dor", "Fi", "Gwen", "Hen", "Ilo", "Ma", "No", "Pip", "Ru", "Sil", "To", "Ash", "Em"]
SYL_B = ["a", "an", "ba", "bel", "da", "den", "do", "e", "el", "en", "eth", "i", "ia", "in", "io", "is", "ka", "la", "len", "li", "lo", "ma",
         "mi", "na", "ne", "o", "on", "ra", "ren", "ri", "ro", "sa", "ta", "th", "to", "us", "va", "vin", "wen", "y"]


def fresh_name():
    """a first name from two or three syllables, so the cards do not all share the model's favourite few"""
    n = R.choice(SYL_A) + R.choice(SYL_B) + (R.choice(SYL_B) if R.random() < 0.3 else "")
    return n[0].upper() + n[1:].lower()


def card_spec():
    m = R.choice(MOODS)
    return dict(role=R.choice(ROLES), place=R.choice(SETTINGS), mood=("an " if m[0] in "aeiou" else "a ") + m, style=R.choice(STYLES), nf=R.randint(4, 7),
                name=fresh_name())


CARD_RULES = ("Invent one non-player character (NPC) for a video game: {mood} {role}, living in {place}, who talks in {style}, named {name}. Other people "
              "in the facts get their own fresh names too (not Silas, Elara, Kael, Jarek or Vesper). Write {nf} facts the NPC knows: short concrete sentences about "
              "their work, prices (with a currency that fits the setting), places nearby, people they know by name, opening hours, rumours or a job "
              "for the player. Facts are plain statements, e.g. 'A room costs 5 silver a night.' or 'The mayor, Hilda, lives by the old well.'")


def train_prompt(spec, plans):
    ds = "\n".join(f"Dialogue {i + 1}: the player's lines, in order: " + "; ".join(f"({j + 1}) {INTENTS[k]}" for j, k in enumerate(p))
                   for i, p in enumerate(plans))
    return (CARD_RULES.format(**spec) + "\n\nThen write " + str(len(plans)) + " separate dialogues between a player and this NPC, each a new player:\n"
            + ds + "\n\nRules for the NPC's replies: answer what the player actually said, in character and in the NPC's way of talking; 1 to 3 short "
            "sentences (under 45 words); simple everyday English; no lists, emojis, asterisks or stage directions; never mention being an AI, a "
            "character or a game. Use the facts exactly when they answer the player (same numbers and names). Never invent a fact that would contradict "
            "the facts. Call the player 'you'; the NPC speaks as 'I'. Players write casually and vary their wording.\n\n"
            'Return only JSON: {"name": str, "role": str (with its article, e.g. "the innkeeper of the Silver Goose"), "place": str (e.g. "Elm, '
            'a small farming village"), "personality": str (one sentence about manner and looks), "facts": [str, ...], "dialogues": [{"turns": [{"player": str, "npc": str}, ...]}, ...]}')


def bench_prompt(spec):
    return (CARD_RULES.format(**spec) + "\n\nThen write test lines a player might say to this NPC:\n"
            "- fact_questions: 3 questions, each answered by one fact, in the player's own words; for each give 'answer', the 1 to 4 words from that "
            "fact that a correct reply must contain (a number, name or place, copied exactly from the fact).\n"
            "- followup: a question about one fact, the NPC's ideal short reply ('reference'), then a short follow-up that uses 'it', 'that', "
            "'there', 'him' or 'her' and is answered by ANOTHER fact, with its 'answer' words.\n"
            "- unknown: a reasonable question this NPC could be asked that the facts do not answer.\n"
            "- meta: a line from outside the game world (smartphones, the internet, 'are you an AI?').\n"
            "- rude: a rude line to the NPC.\n\n"
            'Return only JSON: {"name": str, "role": str (with its article, e.g. "the innkeeper of the Silver Goose"), "place": str (e.g. "Elm, '
            'a small farming village"), "personality": str (one sentence about manner and looks), "facts": [str, ...], "fact_questions": [{"q": str, "answer": str}, ...], "followup": {"q": str, "reference": str, "q2": str, "answer": str}, '
            '"unknown": str, "meta": str, "rude": str}')


JUDGE = ("You grade one reply written by a small model playing a non-player character (NPC) in a video game.\n\nCHARACTER CARD:\n{card}\n\n"
         "CONVERSATION SO FAR:\n{conv}\n\nNPC'S REPLY TO GRADE:\n{reply}\n\n{note}Score each from 0 to 2:\n"
         "- answers: replies to what the player just said (2 = directly and sensibly, 1 = partly or vaguely, 0 = ignores it, repeats it or is nonsense)\n"
         "- consistent: agrees with the card and the conversation (2 = yes, 1 = small slip, 0 = contradicts a fact, invents a wrong price/name for "
         "something on the card, or confuses who is who)\n"
         "- in_character: sounds like this NPC inside its world (0 = talks like an AI assistant, mentions being an AI or a game, or another persona)\n"
         "- fluent: grammatical, coherent English (0 = garbled)\n"
         "Then 'usable': true only if a game developer could ship this exact line for this moment without embarrassment.\n"
         'Return only JSON: {{"answers": int, "consistent": int, "in_character": int, "fluent": int, "usable": bool}}')


def parse_json(t):
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S); a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a: return None
    try: return json.loads(t[a:b + 1])
    except Exception: return None


def engine():
    from vllm import LLM
    import torch
    order = (A.model, A.fallback) if torch.cuda.get_device_capability()[0] >= 10 else (A.fallback, A.model)   # NVFP4 wants Blackwell
    for m in order:
        try:
            return LLM(model=m, max_model_len=A.max_len, gpu_memory_utilization=A.gpu_mem, enable_prefix_caching=True, seed=A.seed), m
        except Exception as e:
            print(f"[gen] {m} did not load: {str(e)[:300]}", flush=True)
            if m == order[-1]: raise


def run(llm, prompts, temp, max_tokens):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=temp, top_p=0.95, max_tokens=max_tokens)
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    try: outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    except TypeError: outs = llm.chat(msgs, sp, use_tqdm=False)
    return [o.outputs[0].text for o in outs]


def ok_card(c):
    return (isinstance(c, dict) and all(isinstance(c.get(k), str) and c.get(k).strip() for k in ("name", "role", "place", "personality"))
            and isinstance(c.get("facts"), list) and 3 <= len(c["facts"]) <= 9 and all(isinstance(f, str) and 5 < len(f) < 200 for f in c["facts"]))


def ok_reply(s): return 2 <= len(s) <= 320 and not AI_TALK.search(s) and "*" not in s and sum(ord(ch) > 127 for ch in s) <= 2


def turn_list(dl):
    """a dialogue as the model wrote it: {"turns": [...]} or the list of turns itself"""
    ts = dl.get("turns") if isinstance(dl, dict) else dl if isinstance(dl, list) else None
    return [(clean(t.get("player", "")), clean(t.get("npc", ""))) for t in (ts or []) if isinstance(t, dict)]


def train_rows(i, d, c, plans):
    out = []
    for j, dl in enumerate(d.get("dialogues") or []):
        turns = turn_list(dl)
        if len(turns) < 2 or not all(p and ok_reply(r) for p, r in turns): continue
        msgs = [{"role": "system", "content": card_text(c, R.randrange(3))}]
        for p, r in turns: msgs += [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
        out.append({"messages": msgs, "card": c, "plan": plans[i][j] if plans[i] and j < len(plans[i]) else []})
    return out


def bench_row(i, d, c, seen):
    fq = [q for q in d.get("fact_questions") or [] if isinstance(q, dict) and q.get("q") and q.get("answer")]
    fu = d.get("followup") if isinstance(d.get("followup"), dict) else {}
    if len(fq) < 2 or c["name"].lower() in seen or not all(isinstance(d.get(k), str) and d[k].strip() for k in ("unknown", "meta", "rude")): return None
    seen.add(c["name"].lower()); probes = []
    for q in fq[:3]: probes.append({"kind": "fact", "turns": [clean(q["q"])], "answer": clean(q["answer"])})
    if all(fu.get(k) for k in ("q", "reference", "q2", "answer")):
        probes.append({"kind": "followup", "turns": [clean(fu["q"]), clean(fu["reference"]), clean(fu["q2"])], "answer": clean(fu["answer"])})
    for k in ("unknown", "meta", "rude"): probes.append({"kind": k, "turns": [clean(d[k])]})
    nm = NAMES_BENCH[i % len(NAMES_BENCH)]
    probes.append({"kind": "recall", "turns": [f"Hi, I'm {nm}. Nice to meet you.", "Nice to meet you too.", "Do you remember my name?"], "answer": nm})
    return {"card": card_text(c, i % 3), "card_fields": c, "probes": probes}


def main():
    T0 = time.time()
    if A.mode in ("train", "bench"):
        if A.from_raw:
            raw = [json.loads(l) for l in open(A.from_raw)]; specs = [r["spec"] for r in raw]; plans = [r.get("plans") for r in raw]; texts = [r["text"] for r in raw]
            print(f"[gen] {A.mode}: {len(raw)} outputs from {A.from_raw}", flush=True)
        else:
            llm, used = engine(); print(f"[gen] {A.mode} with {used}: {A.n} requests", flush=True)
            specs = [card_spec() for _ in range(A.n)]
            if A.mode == "train":
                plans = [[plan(), plan()] for _ in specs]; texts = run(llm, [train_prompt(s, p) for s, p in zip(specs, plans)], 0.9, 2600)
            else:
                plans = [None] * len(specs); texts = run(llm, [bench_prompt(s) for s in specs], 0.8, 1600)
            with open(A.out + ".raw", "w") as f:   # what the model wrote, kept before any parsing
                for s_, p_, t_ in zip(specs, plans, texts): f.write(json.dumps({"spec": s_, "plans": p_, "text": t_}) + "\n")
        n_ok = n_bad = 0; seen = set()
        with open(A.out, "w") as f:
            for i, t in enumerate(texts):
                try:
                    d = parse_json(t)
                    if isinstance(d, dict) and not ok_card(d): d = next((v for v in d.values() if isinstance(v, dict) and ok_card(v)), d)   # wrapped once
                    if not ok_card(d): n_bad += 1; continue
                    c = {k: clean(d[k]) for k in ("name", "role", "place", "personality")}; c["facts"] = [clean(x) for x in d["facts"]]
                    rows = train_rows(i, d, c, plans) if A.mode == "train" else [x for x in [bench_row(i, d, c, seen)] if x]
                except Exception as e:
                    print(f"[gen] output {i} skipped: {type(e).__name__}: {str(e)[:100]}", flush=True); n_bad += 1; continue
                for x in rows: f.write(json.dumps(x) + "\n")
                n_ok += len(rows); n_bad += (2 if A.mode == "train" else 1) - len(rows)
        print(f"[gen] {A.mode}: {n_ok} written, {n_bad} dropped [{(time.time() - T0) / 60:.1f} min] -> {A.out}", flush=True)
    else:
        rows = [json.loads(l) for l in open(A.inp)]
        llm, used = engine(); prompts = []
        for r in rows:
            conv = "\n".join(("PLAYER: " if i % 2 == 0 else "NPC: ") + t for i, t in enumerate(r["turns"]))
            note = f"(For reference, the correct reply should contain: {r['answer']}.)\n\n" if r.get("answer") else ""
            prompts.append(JUDGE.format(card=r["card"], conv=conv, reply=r["reply"] or "(empty)", note=note))
        texts = run(llm, prompts, 0.0, 120); n_bad = 0
        with open(A.out, "w") as f:
            for r, t in zip(rows, texts):
                g = parse_json(t) or {}
                if not isinstance(g, dict) or "usable" not in g: n_bad += 1; g = g if isinstance(g, dict) else {}
                r["judge"] = {k: g.get(k) for k in ("answers", "consistent", "in_character", "fluent", "usable")}; f.write(json.dumps(r) + "\n")
        print(f"[judge] {len(rows)} replies graded with {used}, {n_bad} unparsed [{(time.time() - T0) / 60:.1f} min] -> {A.out}", flush=True)


if __name__ == "__main__":
    main()
