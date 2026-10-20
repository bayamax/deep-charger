#!/usr/bin/env python3
"""One pattern for any use (the user): a goal on numbers, the state, and operations that change it; the model picks the
operation that moves the state toward the goal, the game applies it and sends the new state, until the goal holds.

  [user]      GOAL: life >= 1
              STATE: life=0 gold=7
              OPS: <a> life+1 gold-3 | <b> gold+2 (needs gold>=1) | <c> life-1
  [assistant] <a>                (or <none> once every goal holds)

Levels: 1 one goal, an operation that helps; 2 distractors (wrong way, other variables) and needs; 3 the helping one is
blocked until another raises what it needs (two steps); 4 two goals, side effects that must not break the other.
Train and test use different variable and operation names (test: made-up words), so the test shows whether the model
reads the lines rather than recalls words. The episode is played in a loop: the right pick is the one a short search
finds first on the way to the goal (the fewest steps, then the order of the list).

  python3 goal_gen.py data --n 30000 --split train --out goal_train.jsonl
  python3 goal_gen.py play --model agent2/model_final.pt --n 400 --split test        (closed loop, success rate by level)
"""
import argparse, itertools, json, random, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["data", "play"])
ap.add_argument("--n", type=int, default=1000); ap.add_argument("--split", default="train", choices=["train", "test"])
ap.add_argument("--levels", default="1,2,3,4"); ap.add_argument("--out", default=""); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--model", default=""); ap.add_argument("--max-steps", type=int, default=8)
ap.add_argument("--think", type=int, default=1, help="1: the answer starts with a short worked trace {...} (the steps the rule takes)")
ap.add_argument("--repeat", type=int, default=1, help="1: the game sends the goal and the operations again every turn (they stay near)")
ap.add_argument("--check", type=int, default=1, help="1: the trace reads out every need of a helper before calling it ok (agent5 and before: 0)")
A = ap.parse_args()
R = random.Random(A.seed + (0 if A.split == "train" else 77777))
TRAIN = A.split == "train"
VARS = (["life", "gold", "food", "mana", "heat", "fuel", "ammo", "water", "wood", "stone", "energy", "score", "hp", "keys", "coins", "light",
         "speed", "trust", "noise", "stock", "power", "air", "rest", "fun"] if TRAIN else
        ["zorb", "plim", "quax", "tenk", "vasp", "drell", "mup", "yolt", "frib", "glon"])
OPN = list("abcdefghijklmnopqrstuvwxyz") + (["eat", "rest", "buy", "sell", "dig", "chop", "heal", "work", "pray", "run", "hide", "fix", "drink", "mine"]
                                           if TRAIN else ["blip", "kesh", "vorn", "tazz", "wib"])
CMP = [">=", "<=", ">", "<", "=="]


def holds(v, c, k): return {">=": v >= k, "<=": v <= k, ">": v > k, "<": v < k, "==": v == k}[c]


def ok_state(st, goals): return all(holds(st[g[0]], g[1], g[2]) for g in goals)


def can(op, st): return all(st[v] >= k for v, k in op["needs"])


def apply(op, st):
    s = dict(st)
    for v, d in op["eff"]: s[v] += d
    return s


def plan(st, goals, ops, depth=4):
    """the fewest steps to the goal (breadth first, list order on ties); [] when it holds, None when out of reach"""
    if ok_state(st, goals): return []
    frontier = [(st, [])]; seen = {tuple(sorted(st.items()))}
    for _ in range(depth):
        nxt = []
        for s, path in frontier:
            for op in ops:
                if not can(op, s): continue
                t = apply(op, s); key = tuple(sorted(t.items()))
                if key in seen or any(v < -20 or v > 60 for v in t.values()): continue
                seen.add(key)
                if ok_state(t, goals): return path + [op["name"]]
                nxt.append((t, path + [op["name"]]))
        frontier = nxt
    return None


def direction(st, g):
    v, c, k = g
    if holds(st[v], c, k): return "ok"
    return "up" if (c in (">=", ">") or (c == "==" and st[v] < k)) else "down"


def policy(st, goals, ops):
    """the rule the trace spells out: the first goal not met -> the operations that move its variable the right way (list
    order) -> the first that may run and keeps the other goals that hold; if none may run, the first operation that raises
    what the first helper needs. Returns (pick, trace)."""
    parts = [f"{v} {st[v]} {c}{k} {direction(st, (v, c, k))}" for v, c, k in goals]
    bad = [g for g in goals if direction(st, g) != "ok"]
    if not bad: return "none", "; ".join(parts)
    v, c, k = bad[0]; d = direction(st, bad[0])
    helpers = [o for o in ops if any(x == v and ((dd > 0) == (d == "up")) for x, dd in o["eff"])]
    if not helpers: return "none", "; ".join(parts) + f". {d}: nothing"
    parts.append(f"{d}: " + " ".join(o["name"] + next(f"{dd:+d}" for x, dd in o["eff"] if x == v) for o in helpers))
    for o in helpers:
        if not can(o, st):
            nv, nk = next((x, kk) for x, kk in o["needs"] if st[x] < kk); parts.append(f"{o['name']} needs {nv}>={nk} {nv} {st[nv]} no"); continue
        need = "".join(f" needs {x}>={kk} {x} {st[x]} yes" for x, kk in o["needs"]) if A.check else ""
        t = apply(o, st); broken = [g for g in goals if g is not bad[0] and direction(st, g) == "ok" and direction(t, g) != "ok"]
        if broken: parts.append(f"{o['name']}{need} breaks {broken[0][0]}"); continue
        parts.append(f"{o['name']}{need} ok"); return o["name"], ". ".join(parts)
    first = helpers[0]   # all blocked: raise what the first one needs
    nv, nk = next(((x, kk) for x, kk in first["needs"] if st[x] < kk), (None, None))
    if nv:
        raisers = [o for o in ops if can(o, st) and any(x == nv and dd > 0 for x, dd in o["eff"])]
        if raisers: parts.append(f"{nv} up: {raisers[0]['name']}"); return raisers[0]["name"], ". ".join(parts)
    return "none", ". ".join(parts) + ". stuck"


def greedy_run(st, goals, ops, steps=6):
    for _ in range(steps):
        pick, _ = policy(st, goals, ops)
        if pick == "none": return ok_state(st, goals)
        st = apply(next(o for o in ops if o["name"] == pick), st)
    return False


def episode(level):
    """a solvable start: goals, state, operations (with the helping one(s) placed anywhere)"""
    for _ in range(200):
        nv = {1: 2, 2: 3, 3: 3, 4: 3}[level]; vs = R.sample(VARS, nv); names = R.sample(OPN, R.randint(3, 5) if level > 1 else R.randint(2, 3))
        st = {v: R.randint(0, 12) for v in vs}
        goals = []
        for g in vs[:1 if level < 4 else 2]:
            c = R.choice(CMP if level > 1 else [">=", "<="]); k = R.randint(1, 10)
            if c == "==": k = st[g] + R.choice([-3, -2, -1, 1, 2, 3])
            goals.append((g, c, k))
        ops = []
        for nm in names:
            eff = [(v, R.choice([-3, -2, -1, 1, 2, 3])) for v in R.sample(vs, R.randint(1, 2 if level > 1 else 1))]
            needs = [(v, R.randint(1, 6)) for v in R.sample(vs, 1)] if level >= 2 and R.random() < 0.35 else []
            ops.append({"name": nm, "eff": eff, "needs": needs})
        if level == 3:   # make the helper need something only another op gives
            g = goals[0][0]; other = [v for v in vs if v != g][0]; st[other] = 0
            ops[0] = {"name": ops[0]["name"], "eff": [(g, 3 if goals[0][1] in (">=", ">") else -3)], "needs": [(other, 2)]}
            ops[1] = {"name": ops[1]["name"], "eff": [(other, 2)], "needs": []}
            R.shuffle(ops)
        if ok_state(st, goals) and R.random() < 0.85: continue
        p = plan(st, goals, ops)
        if p is None or len(p) > 4: continue
        if level == 3 and len(p) < 2: continue
        if A.think and not greedy_run(dict(st), goals, ops): continue   # the rule must get there itself
        return st, goals, ops
    return None


def show_ops(ops):
    return " | ".join(f"<{o['name']}> " + " ".join(f"{v}{d:+d}" for v, d in o["eff"]) + (" (needs " + ", ".join(f"{v}>={k}" for v, k in o["needs"]) + ")" if o["needs"] else "")
                      for o in ops)


def head(goals, ops): return "GOAL: " + " and ".join(f"{g} {c} {k}" for g, c, k in goals) + "\nOPS: " + show_ops(ops)
def state_line(st): return "STATE: " + " ".join(f"{v}={x}" for v, x in st.items())


def data(n):
    rows = []; levels = [int(x) for x in A.levels.split(",")]
    while len(rows) < n:
        lv = R.choice(levels); e = episode(lv)
        if not e: continue
        st, goals, ops = e; msgs = []
        for step in range(A.max_steps):
            if A.think: pick, tr = policy(st, goals, ops); ans = f"{{{tr}}}<{pick}>"   # braces: a trace never starts like an operation tag
            else: p = plan(st, goals, ops); pick = p[0] if p else "none"; ans = f"<{pick}>"
            msgs += [{"role": "user", "content": (head(goals, ops) + "\n" if step == 0 or A.repeat else "") + state_line(st)}, {"role": "assistant", "content": ans}]
            if pick == "none": break
            st = apply(next(o for o in ops if o["name"] == pick), st)
        rows.append({"messages": msgs, "level": lv})
    return rows


if __name__ == "__main__" and A.mode == "data":
    rows = data(A.n)
    with open(A.out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"[goal] {len(rows)} {A.split} episodes ({sum(len(r['messages']) // 2 for r in rows)} decisions) -> {A.out}")
elif __name__ == "__main__":   # closed loop with the model: each turn it picks, the game applies, until <none> or the step limit
    sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
    import charlm_run as CR
    m = CR.Model(A.model); res = {}; ex = []
    for lv in [int(x) for x in A.levels.split(",")]:
        won = steps = opt = bad = 0; k = 0
        while k < A.n:
            e = episode(lv)
            if not e: continue
            k += 1; st, goals, ops = e; best = len(plan(st, goals, ops)); turns = []; done = False
            for step in range(A.max_steps):
                turns.append((head(goals, ops) + "\n" if step == 0 or A.repeat else "") + state_line(st))
                y = m.reply(turns, n=260 if A.think else 24).strip(); turns.append(y); mm = re.search(r"<([a-z_]+)>$", y)
                if not mm: bad += 1; break
                if mm.group(1) == "none": done = ok_state(st, goals); break
                op = next((o for o in ops if o["name"] == mm.group(1)), None)
                if op is None or not can(op, st): bad += 1; break
                st = apply(op, st)
            won += done; steps += (len(turns) // 2 - 1) if done else 0; opt += best if done else 0
            if not done and len(ex) < 6: ex.append(turns)
        res[lv] = (won / A.n, steps / max(won, 1), opt / max(won, 1), bad / A.n)
        print(f"[goal] level {lv}: reached {100 * won / A.n:.1f}% of {A.n}; steps {steps / max(won, 1):.2f} (shortest {opt / max(won, 1):.2f}); "
              f"bad or impossible picks {100 * bad / A.n:.1f}%", flush=True)
    for t in ex: print("[goal] failed:", json.dumps(t)[:600])
