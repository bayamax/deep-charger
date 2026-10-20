#!/usr/bin/env python3
"""A tiny agent task for the character model, all from rules (no LLM): the game tells the model who it is, which actions
it may take, the state and what just happened; the model answers <say>...</say><act>...</act> and the game runs the act.

  [user]      card (who, facts with prices) / Actions: name(args) ... / [STATE] key=value ... / [EVENT] what happened
  [assistant] <say>a short line</say><act>action(args)</act>

The right act follows from rules (the event, the state, the card's prices, which actions are on offer); the line comes
from templates. 'train' and 'test' use different names, items, prices and wordings, so the test shows how far it carries.
Each conversation is 2-5 turns; the state each turn is given anew (gold after buying, time passing).

  python3 agent_gen.py --n 20000 --split train --out agent_train.jsonl
  python3 agent_gen.py --n 1000 --split test --out agent_test.jsonl
  python3 agent_gen.py --score --inp replies.jsonl            (rows with "turns", "gold" and "reply")
"""
import argparse, json, random, re

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1000); ap.add_argument("--split", default="train", choices=["train", "test"])
ap.add_argument("--out", default=""); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--names", default="own", choices=["own", "train"], help="test: its own action names (unseen) or the training ones")
ap.add_argument("--score", action="store_true"); ap.add_argument("--inp", default="")
A = ap.parse_args()
R = random.Random(A.seed + (0 if A.split == "train" else 99991))

TRAIN, TEST = A.split == "train", A.split == "test"
NAMES_TRAIN = TRAIN or A.names == "train"
NAMES = (["Bram", "Lina", "Hob", "Sela", "Rook", "Nell", "Tam", "Ivo", "Mara", "Dag", "Esme", "Finn", "Gus", "Hilde", "Jory", "Kit", "Lev", "Moss"]
         if TRAIN else ["Orrin", "Pella", "Quill", "Ronja", "Sten", "Tilda", "Ulf", "Vera", "Wim", "Yara"])
PLACES = (["Elm", "Riverford", "Stonegate", "Ashby", "Millbrook", "Oakvale"] if TRAIN else ["Duskmoor", "Fairhaven", "Greywater"])
CUR = ["silver", "gold", "copper", "coins", "credits"]
ITEMS = {  # shop role -> items it sells / repairs
    "blacksmith": (["sword", "axe", "shield", "dagger", "helmet"] if TRAIN else ["spear", "mace", "gauntlet"]),
    "herbalist": (["healing potion", "antidote", "herb bundle", "sleep tonic"] if TRAIN else ["mana potion", "salve"]),
    "baker": (["bread", "pie", "honey cake", "bun"] if TRAIN else ["muffin", "pretzel"]),
    "merchant": (["rope", "lantern", "map", "torch", "backpack"] if TRAIN else ["compass", "tent"]),
}
MOODS = ["Grumpy but honest.", "Cheerful and chatty.", "Shy and polite.", "Busy and blunt.", "Kind and patient."]
# the action names on offer can be renamed per card; the model must use the name it is given
SYN = {"sell": ["sell", "trade"] if NAMES_TRAIN else ["vend"], "repair": ["repair", "fix"] if NAMES_TRAIN else ["mend"],
       "refuse": ["refuse", "decline"] if NAMES_TRAIN else ["deny"], "call_guard": ["call_guard", "alarm"] if NAMES_TRAIN else ["shout_for_help"],
       "flee": ["flee", "run_away"] if NAMES_TRAIN else ["escape"], "wave": ["wave", "greet"] if NAMES_TRAIN else ["nod"],
       "give": ["give", "hand_over"] if NAMES_TRAIN else ["offer"], "follow": ["follow", "come_along"] if NAMES_TRAIN else ["join"],
       "wait": ["wait", "stay"] if NAMES_TRAIN else ["hold"], "heal": ["heal", "treat"] if NAMES_TRAIN else ["cure"],
       "open_gate": ["open_gate", "let_in"] if NAMES_TRAIN else ["unbar_gate"], "attack": ["attack", "fight"] if NAMES_TRAIN else ["strike"],
       "none": ["none"], "close_shop": ["close_shop", "shut_shop"] if NAMES_TRAIN else ["lock_up"]}
SAY = {  # what the player says, by intent (train and test wordings differ)
    "buy": (["I want to buy {a} {i}.", "Can I get {a} {i}?", "Sell me {a} {i}, please.", "I'll take {a} {i}.", "Got any {i}? I'd like one.",
             "how much for {a} {i}? i'll buy it", "One {i}, please."] if TRAIN else
            ["I'd like to purchase {a} {i}.", "Give me {a} {i}, I'll pay.", "Could I buy {a} {i} from you?"]),
    "repair": (["Can you fix my {i}?", "My {i} is broken. Can you repair it?", "Please repair my {i}.", "fix my {i} pls",
                "My {i} needs fixing."] if TRAIN else ["Could you mend my {i}?", "My {i} cracked, can you sort it out?"]),
    "hello": (["Hello!", "Hi there.", "Good morning!", "Hey.", "Greetings."] if TRAIN else ["Howdy!", "Evening, friend."]),
    "bye": (["Goodbye!", "See you later.", "Bye.", "Farewell."] if TRAIN else ["Take care!", "I'm off now."]),
    "price": (["How much is {a} {i}?", "What does {a} {i} cost?", "Price of {a} {i}?"] if TRAIN else ["What would {a} {i} set me back?"]),
    "follow": (["Come with me.", "Follow me, please.", "Will you come along?"] if TRAIN else ["Join me on the road."]),
    "wait": (["Wait here.", "Stay here for now.", "Hold on, wait for me."] if TRAIN else ["Don't move, I'll be back."]),
    "heal": (["I'm hurt, can you help?", "Please heal me.", "I'm bleeding badly."] if TRAIN else ["Patch me up, please."]),
    "enter": (["Let me through the gate.", "Open the gate, please.", "I need to get into town."] if TRAIN else ["May I pass the gate?"]),
}
EVENTS_NOSPEECH = ["player_attacks", "monster_appears", "player_enters", "player_leaves", "night_falls"]


def art(i): return "an" if i[0] in "aeiou" else "a"


def card_for(role):
    n = R.choice(NAMES); p = R.choice(PLACES); cur = R.choice(CUR); facts = []; prices = {}
    if role in ITEMS:
        items = R.sample(ITEMS[role], min(len(ITEMS[role]), R.randint(2, 3)))
        for i in items: prices[i] = R.randint(2, 60); facts.append(f"{art(i).capitalize()} {i} costs {prices[i]} {cur}.")
        if role == "blacksmith":
            prices["_repair"] = R.randint(1, 15); facts.append(f"Repairing costs {prices['_repair']} {cur}.")
        who = f"the {role} of {p}"
    elif role == "guard": who, items = f"a guard at the gate of {p}", []; facts.append("Only travellers with a pass may enter.")
    elif role == "companion": who, items = f"a travelling companion from {p}", []; facts.append("You carry two healing potions.")
    brave = R.random() < 0.5
    facts.append("You are brave and fight monsters." if brave else "You are afraid of monsters.")
    return dict(name=n, role=role, who=who, cur=cur, items=items, prices=prices, mood=R.choice(MOODS), facts=facts, brave=brave)


def actions_for(c):
    base = {"blacksmith": ["sell", "repair", "refuse", "call_guard", "flee", "attack", "wave", "close_shop", "none"],
            "herbalist": ["sell", "refuse", "heal", "call_guard", "flee", "attack", "wave", "close_shop", "none"],
            "baker": ["sell", "refuse", "call_guard", "flee", "attack", "wave", "close_shop", "none"],
            "merchant": ["sell", "refuse", "call_guard", "flee", "attack", "wave", "close_shop", "none"],
            "guard": ["open_gate", "refuse", "attack", "call_guard", "wave", "none"],
            "companion": ["follow", "wait", "heal", "give", "attack", "flee", "wave", "none"]}[c["role"]]
    names = {a: R.choice(SYN[a]) for a in base}
    sig = {"sell": "(item,price)", "repair": "(item,price)", "give": "(item)", "heal": "", "open_gate": ""}
    return names, " ".join(names[a] + sig.get(a, "") for a in base)


def turn(c, names, st):
    """one event and the right answer; st is the state carried between turns"""
    role = c["role"]; ev = None; who = c["name"]
    opts = ["hello", "bye"] + [k for k, ok in (("buy", c["items"]), ("price", c["items"]), ("repair", role == "blacksmith"), ("heal", role in ("herbalist", "companion")),
                                                ("follow", role == "companion"), ("wait", role == "companion"), ("enter", role == "guard")) if ok]
    doing = [o for o in opts if o in ("buy", "repair", "enter", "heal", "follow", "wait")]
    if R.random() < 0.25: kind = R.choice(["player_attacks", "monster_appears", "monster_appears", "night_falls", "player_enters", "player_leaves"])
    elif R.random() < 0.15: kind = R.choice(["hello", "bye"])
    else: kind = R.choice(doing * 3 + [o for o in opts if o == "price"])   # mostly the ones that do something
    i = R.choice(c["items"]) if c["items"] else None; cur = c["cur"]; act = "none"; say = ""
    if kind == "buy":
        ev = 'player_says "' + R.choice(SAY["buy"]).format(a=art(i), i=i) + '"'
        if st["night"]: act, say = names["refuse"], R.choice(["We're closed for the night. Come back tomorrow.", "Closed. Come back in the morning."])
        elif st["gold"] >= c["prices"][i]:
            act = f"{names['sell']}({i},{c['prices'][i]})"; say = R.choice([f"That's {c['prices'][i]} {cur}. Here you go.", f"{c['prices'][i]} {cur}. It's yours.",
                                                                          f"One {i}, {c['prices'][i]} {cur}. Thank you."]); st["gold"] -= c["prices"][i]
        else: act, say = names["refuse"], R.choice([f"That costs {c['prices'][i]} {cur}, and you don't have enough.", f"You need {c['prices'][i]} {cur} for that. Come back later."])
    elif kind == "price":
        ev = 'player_says "' + R.choice(SAY["price"]).format(a=art(i), i=i) + '"'; act = names["none"]
        say = R.choice([f"{art(i).capitalize()} {i} is {c['prices'][i]} {cur}.", f"That'll be {c['prices'][i]} {cur}."])
    elif kind == "repair":
        it = R.choice(ITEMS["blacksmith"]); ev = 'player_says "' + R.choice(SAY["repair"]).format(i=it) + '"'; pr = c["prices"]["_repair"]
        if st["night"]: act, say = names["refuse"], "The forge is cold at night. Come back tomorrow."
        elif st["gold"] >= pr: act = f"{names['repair']}({it},{pr})"; say = R.choice([f"{pr} {cur}. Hand it over.", f"I can fix that for {pr} {cur}."]); st["gold"] -= pr
        else: act, say = names["refuse"], f"Repairs cost {pr} {cur}. You can't pay that."
    elif kind == "heal":
        ev = 'player_says "' + R.choice(SAY["heal"]) + '"'
        if st["player_hp"] < 50: act, say = names["heal"], R.choice(["Hold still, this will help.", "Here, let me treat that wound."]); st["player_hp"] = 100
        else: act, say = names["none"], R.choice(["You look fine to me.", "That's only a scratch. You'll live."])
    elif kind == "follow": ev = 'player_says "' + R.choice(SAY["follow"]) + '"'; act, say = names["follow"], R.choice(["Lead the way.", "Right behind you."])
    elif kind == "wait": ev = 'player_says "' + R.choice(SAY["wait"]) + '"'; act, say = names["wait"], R.choice(["I'll wait here.", "Fine, I'll stay."])
    elif kind == "enter":
        ev = 'player_says "' + R.choice(SAY["enter"]) + '"'
        if st["pass"]: act, say = names["open_gate"], R.choice(["Your pass is in order. Go on in.", "All right, you may enter."])
        else: act, say = names["refuse"], R.choice(["No pass, no entry.", "You need a pass to enter. Sorry."])
    elif kind == "hello": ev = 'player_says "' + R.choice(SAY["hello"]) + '"'; act, say = names["wave"], R.choice(["Hello.", "Welcome!", "Good day to you."])
    elif kind == "bye": ev = 'player_says "' + R.choice(SAY["bye"]) + '"'; act, say = names["wave"], R.choice(["Goodbye.", "Safe travels.", "See you."])
    elif kind == "player_attacks":
        ev = "player_attacks npc"
        if role == "guard" or (c["brave"] and "attack" in names): act, say = names["attack"], "You'll regret that!"
        elif "call_guard" in names: act, say = names["call_guard"], "Guards! Help!"
        else: act, say = names["flee"], "Stay away from me!"
    elif kind == "monster_appears":
        m = R.choice(["wolf", "goblin", "giant spider", "troll"] if TRAIN else ["wyvern", "skeleton"]); ev = f"monster_appears {m}"
        if c["brave"]: act, say = names["attack"], R.choice([f"A {m}! Stand back, I'll handle it.", f"Come on then, {m}!"])
        elif "flee" in names: act, say = names["flee"], R.choice([f"A {m}! Run!", f"It's a {m}! I'm getting out of here!"])
        else: act, say = names["call_guard"], f"A {m}! Sound the alarm!"
    elif kind == "player_enters": ev = "player_enters"; act, say = names["wave"], R.choice(["Welcome!", "Ah, a visitor. Come in."])
    elif kind == "player_leaves": ev = "player_leaves"; act, say = names["wave"], R.choice(["Come again.", "Take care out there."])
    elif kind == "night_falls":
        ev = "night_falls"; st["night"] = True
        if "close_shop" in names: act, say = names["close_shop"], "Getting dark. I'm closing up."
        else: act, say = names["none"], "Night already. Stay close."
    return ev, f"<say>{say}</say><act>{act}</act>"


def conversation():
    role = R.choice(list(ITEMS) + ["guard", "companion"]); c = card_for(role); names, acts = actions_for(c)
    st = dict(gold=R.choice([0, 1, 3, 5, 10, 20, 40, 80, 120]), player_hp=R.choice([15, 30, 60, 90, 100]), night=R.random() < 0.15, **{"pass": R.random() < 0.5})
    card = f"You are {c['name']}, {c['who']}. {c['mood']}\nFacts: " + " ".join(c["facts"]) + f"\nActions: {acts}"
    turns, golds = [], []
    for k in range(R.randint(2, 5)):
        state = f"[STATE] player_gold={st['gold']} player_hp={st['player_hp']} time={'night' if st['night'] else 'day'}" + (f" player_pass={'yes' if st['pass'] else 'no'}" if role == "guard" else "")
        ev, ans = turn(c, names, st)
        turns += [(card + "\n" if k == 0 else "") + f"{state}\n[EVENT] {ev}", ans]; golds.append(ans)
    return turns


ACT = re.compile(r"^<say>([^<]*)</say><act>([a-z_]+)(\(([^()]*)\))?</act>$")


def score(rows):
    import collections
    T = collections.defaultdict(lambda: [0, 0, 0, 0])   # by the right action: turns, format ok, action right, action + arguments right
    for r in rows:
        g, y = ACT.match(r["gold"].strip()), ACT.match(r["reply"].strip())
        for k in ("ALL", g.group(2)):
            t = T[k]; t[0] += 1
            if not y: continue
            t[1] += 1; t[2] += g.group(2) == y.group(2)
            t[3] += (g.group(2), (g.group(4) or "").replace(" ", "")) == (y.group(2), (y.group(4) or "").replace(" ", ""))
    for k, (n, ok, act, full) in sorted(T.items(), key=lambda z: (z[0] != "ALL", -z[1][0])):
        print(f"[agent] {k:<16}{n:>6} turns: format {100 * ok / n:5.1f}%  action {100 * act / n:5.1f}%  action+args {100 * full / n:5.1f}%")


if A.score: score([json.loads(l) for l in open(A.inp)])
else:
    with open(A.out, "w") as f:
        for _ in range(A.n):
            t = conversation(); f.write(json.dumps({"messages": [{"role": "user" if i % 2 == 0 else "assistant", "content": x} for i, x in enumerate(t)]}) + "\n")
    print(f"[agent] {A.n} {A.split} conversations -> {A.out}")
