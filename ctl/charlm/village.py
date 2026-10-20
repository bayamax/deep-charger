#!/usr/bin/env python3
"""A small village run by several agents on the one char model: each tick the game sends every villager the goal that
matters most to them now, their state and their operations (the goal_gen.py format), the model answers with a tag, the
game checks it and applies it; the market is shared, so what one sells another can buy. In the evening two villagers
talk (the chat side of the same model), with the day's facts in their cards.

  python3 village.py --model agent6/model_4bit.pt --days 4 --out village.md
  python3 village.py --policy rule ...           (the rule the traces spell out, in place of the model: a baseline)
"""
import argparse, os, random, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=""); ap.add_argument("--policy", default="model", choices=["model", "rule"])
ap.add_argument("--days", type=int, default=4); ap.add_argument("--ticks", type=int, default=4, help="decisions per villager per day")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="village.md"); ap.add_argument("--talk", type=int, default=3, help="model lines per evening talk (0: none)")
A = ap.parse_args()
R = random.Random(A.seed)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0], "data"]; import goal_gen as G  # noqa: E402  (its policy, show_ops and state format; the parse wants a mode)


def op(name, eff, needs=(), market=None, desc=""):
    return {"name": name, "eff": list(eff.items()), "needs": list(needs), "market": market or {}, "desc": desc}


VILLAGERS = [
    {"name": "Mira", "card": "You are Mira, a farmer in the village of Elm. You grow wheat and sell it at the market. You are cheerful and practical.",
     "st": {"energy": 3, "food": 1, "gold": 1}, "goals": [("energy", ">=", 2), ("food", ">=", 2), ("gold", ">=", 6)],
     "ops": [op("farm", {"food": 3, "energy": -2}, [("energy", 2)], desc="worked the wheat field"),
             op("eat", {"food": -1, "energy": 2}, [("food", 1)], desc="had a meal"),
             op("sell", {"food": -2, "gold": 3}, [("food", 2)], {"food": 2}, desc="sold wheat at the market"),
             op("rest", {"energy": 1}, desc="took a rest")]},
    {"name": "Tob", "card": "You are Tob, the blacksmith of Elm. You forge tools and sell them at the market. You are gruff but kind.",
     "st": {"energy": 3, "food": 1, "gold": 2, "tools": 0}, "goals": [("energy", ">=", 2), ("food", ">=", 1), ("tools", ">=", 1), ("gold", ">=", 6)],
     "ops": [op("forge", {"tools": 2, "energy": -2}, [("energy", 2)], desc="forged tools"),
             op("sell", {"tools": -1, "gold": 3}, [("tools", 1)], {"tools": 1}, desc="sold a tool at the market"),
             op("buy", {"food": 2, "gold": -2}, [("gold", 2)], {"food": -2}, desc="bought food at the market"),
             op("eat", {"food": -1, "energy": 2}, [("food", 1)], desc="had a meal"),
             op("rest", {"energy": 1}, desc="took a rest")]},
    {"name": "Lena", "card": "You are Lena, a traveling trader who has settled in Elm. You buy goods at the market and carry them to the city. You are curious and talkative.",
     "st": {"energy": 3, "food": 2, "gold": 4, "wares": 0}, "goals": [("energy", ">=", 2), ("food", ">=", 1), ("wares", ">=", 1), ("gold", ">=", 9)],
     "ops": [op("trade", {"wares": -1, "gold": 4}, [("wares", 1)], desc="carried goods to the city and sold them"),
             op("stock", {"wares": 1, "gold": -2}, [("gold", 2)], {"tools": -1}, desc="bought a tool at the market to trade"),
             op("buy", {"food": 2, "gold": -2}, [("gold", 2)], {"food": -2}, desc="bought food at the market"),
             op("eat", {"food": -1, "energy": 2}, [("food", 1)], desc="had a meal"),
             op("rest", {"energy": 1}, desc="took a rest")]},
]
MARKET = {"food": 4, "tools": 1}


def goal_now(v):
    """the first goal in the villager's order that does not hold (one at a time: the model is surest so)"""
    for g in v["goals"]:
        if not G.holds(v["st"][g[0]], g[1], g[2]): return [g]
    return [v["goals"][-1][:2] + (v["goals"][-1][2] + 3,)]   # all fine: aim a little richer


def usable(v):
    """the operations the market can serve right now: the game lists only those (a shorter list, fewer misreadings)"""
    return [o for o in v["ops"] if all(MARKET[k] + d >= 0 for k, d in o["market"].items()) and o["name"] != v.get("hide")]


def prompt(v, goals):
    """the goal's variable first in STATE: the training episodes always put it there, and the model leans on that"""
    st = {k: v["st"][k] for k in sorted(v["st"], key=lambda k: k != goals[0][0])}
    return G.head(goals, usable(v)) + "\n" + G.state_line(st)


def check(v, name):
    """the game's side: a known operation, its needs met, and the market able to give what it takes"""
    o = next((o for o in usable(v) if o["name"] == name), None)
    if o is None: return None, f"<{name}> is not an operation"
    if not G.can(o, v["st"]): return None, f"<{name}> needs " + ", ".join(f"{x}>={k}" for x, k in o["needs"])
    if any(MARKET[k] + d < 0 for k, d in o["market"].items()): return None, f"the market has no {next(k for k, d in o['market'].items() if MARKET[k] + d < 0)}"
    return o, ""


m = None
if A.policy == "model":
    import charlm_run as CR
    m = CR.Model(A.model)
log, stats = [], {"decisions": 0, "rejected": 0, "retried_ok": 0, "same_as_rule": 0, "no_tag": 0}
for v in VILLAGERS: v["turns"], v["goal"], v["day"] = [], None, []


def decide(v):
    goals = goal_now(v)
    if goals != v["goal"] or usable(v) != v.get("shown"): v["turns"], v["goal"], v["shown"] = [], goals, usable(v)   # a new goal starts a new episode, as in training
    v["turns"] = v["turns"][-6:]; v["turns"].append(prompt(v, goals))
    rule, tr = G.policy(v["st"], goals, usable(v))
    if A.policy == "rule": y = f"{{{tr}}}<{rule}>"
    else: y = m.reply(v["turns"], n=260).strip()
    mm = re.search(r"<([a-z_]+)>$", y); pick = mm.group(1) if mm else None
    stats["decisions"] += 1; stats["same_as_rule"] += pick == rule; stats["no_tag"] += pick is None
    o, why = check(v, pick) if pick and pick != "none" else (None, "no action" if pick == "none" else "no tag")
    note = ""
    if o is None and pick != "none" and A.policy == "model":   # rejected: ask once more with that operation left off the list
        stats["rejected"] += 1; v["hide"] = pick
        y2 = m.reply([prompt(v, goals)], n=260).strip(); v["hide"] = None; mm = re.search(r"<([a-z_]+)>$", y2); p2 = mm.group(1) if mm else None
        o2, why2 = check(v, p2) if p2 and p2 != "none" else (None, "no action")
        note = f" (rejected: {why}; asked again -> {y2[-60:]!r}{' ok' if o2 else ' rejected: ' + why2})"
        stats["retried_ok"] += o2 is not None; o, y = o2, y2
    v["turns"].append(y)
    if o:
        st0 = dict(v["st"]); v["st"] = G.apply(o, v["st"])
        for k, d in o["market"].items(): MARKET[k] += d
        v["day"].append(o["desc"])
        return f"- **{v['name']}** goal `{goals[0][0]} {goals[0][1]} {goals[0][2]}` · {G.state_line(st0)[7:]} → `{y}`{note} → {G.state_line(v['st'])[7:]}"
    return f"- **{v['name']}** goal `{goals[0][0]} {goals[0][1]} {goals[0][2]}` · {G.state_line(v['st'])[7:]} → `{y}`{note} → (nothing happens)"


def talk(a, b):
    """a evening chat: b's opener comes from the game, then the model speaks as a and b in turn, each with their own card"""
    fact = lambda v: (f" Today you {', '.join(dict.fromkeys(v['day'])) or 'did little'}." + f" You have {v['st']['gold']} gold" +
                      (f" and feel tired." if v["st"]["energy"] < 2 else ".") + (" You are hungry." if v["st"]["food"] < 1 else ""))
    card = lambda v, o: v["card"] + fact(v) + f" It is evening at the market and you are talking with {o['name']}."
    line = f"Evening, {a['name']}! How did your day go?"; lines = [(b["name"], line)]
    ctx = {a["name"]: [card(a, b) + "\n\n" + line], b["name"]: [card(b, a) + "\n\n(You greet " + a["name"] + ".)", line]}
    who, other = a, b
    for _ in range(A.talk):
        y = m.reply(ctx[who["name"]], n=220, temp=0.7).strip().split("\n")[0]; lines.append((who["name"], y))
        ctx[who["name"]].append(y); ctx[other["name"]].append(y); who, other = other, who
    return lines


for day in range(1, A.days + 1):
    log.append(f"\n## Day {day}\n\nMarket at dawn: {MARKET['food']} food, {MARKET['tools']} tools\n")
    for v in VILLAGERS: v["day"] = []
    for t in range(A.ticks):
        log.append(f"\n**Hour {t + 1}**\n")
        for v in R.sample(VILLAGERS, len(VILLAGERS)): log.append(decide(v))
    for v in VILLAGERS:   # the night: everyone gets hungrier and wearier
        v["st"]["energy"] = max(0, v["st"]["energy"] - 1); v["st"]["food"] = max(0, v["st"]["food"] - 1)
    if A.talk and m:
        a, b = R.sample(VILLAGERS, 2); log.append(f"\n**Evening: {a['name']} and {b['name']} at the market**\n")
        for who, s in talk(a, b): log.append(f"> **{who}:** {s}  ")
    log.append("\nEnd of day: " + "; ".join(f"{v['name']} {G.state_line(v['st'])[7:]}" for v in VILLAGERS))
s = stats; head = (f"# Elm village, {A.days} days ({A.policy}{' ' + os.path.basename(A.model) if A.model else ''})\n\n"
                   f"{s['decisions']} decisions: the same as the rule {100 * s['same_as_rule'] / s['decisions']:.0f}%, no tag {s['no_tag']}, "
                   f"rejected by the game {s['rejected']} (fixed on asking again {s['retried_ok']})\n")
open(A.out, "w").write(head + "\n".join(log) + "\n"); print(head)
