"""Build the long-range eval for a bounded KV cache (ctl/mt_eval.jsonl): only a short system prompt is pinned, every user turn
is fed into the stream as forced tokens (tools/sp_qwen3.py rollout, rows with "turns"), so the problem text, documents and
earlier turns can all be evicted. Each task needs something from thousands of tokens back:
  unpin  - a math problem whose statement is NOT pinned (the 60-problem eval pinned it, which is why it never tested memory)
  fact   - 5 details in turn 1, a long math turn in between, then one detail is asked for
  chain  - two math problems, then: add your two final answers (scored against the model's own answers and against the golds)
  docq   - a ~3.1k-token shipment log (100 entries), the question after it about an entry near the start
  docf   - the same, the question stated before the log
  code   - a small Python function in turn 1, a long math turn, then: what does f(x) return
  python tools/mt_build.py ctl/rl_eval120.jsonl ctl/math96.jsonl ctl/mt_eval.jsonl
"""
import json, random, re, sys

R = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 20261110)   # another seed: a calibration set with other facts, logs, functions
SYS = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
ev120, m96, out = sys.argv[1:4]


def body(p): return p.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]


rs = [json.loads(l) for l in open(ev120)]
ev = [r for i, r in enumerate(rs) if i % 24 < 12]; rest = [r for i, r in enumerate(rs) if i % 24 >= 12]
pool = [json.loads(l) for l in open(m96)] + [r for r in rest if r["src"].startswith(("math", "comp"))]
isint = lambda g: bool(re.fullmatch(r"-?\d+", str(g).strip()))
ints = [r for r in pool if isint(r["gold"])]
R.shuffle(pool); R.shuffle(ints)
dist = iter(pool * 3)
rows = []

# unpin: 30 of the 60 eval problems (every other one: all sources)
for k, r in enumerate(ev[::2]):
    rows.append({"id": f"unpin/{k}", "task": "unpin", "gold": r["gold"], "src": r["src"], "turns": [body(r["prompt"])], "caps": [12000]})

# fact
SUR = ["Okafor", "Lindqvist", "Ramirez", "Takahashi", "Novak", "Haddad", "Brennan", "Kowalczyk", "Mbeki", "Varga", "Castellano", "Iyer"]
ADJ = ["Amber", "Silent", "Copper", "Northern", "Velvet", "Crimson", "Hollow", "Bright", "Granite", "Misty"]
ANI = ["Falcon", "Otter", "Lynx", "Heron", "Badger", "Marlin", "Raven", "Bison", "Gecko", "Puma"]
WRD = ["maple", "harbor", "pixel", "tundra", "saffron", "quartz", "lagoon", "ember", "cobalt", "willow"]
FACTS = [("locker code", lambda: str(R.randint(1000, 9999))),
         ("project codename", lambda: f"{R.choice(ADJ)} {R.choice(ANI)}"),
         ("meeting room", lambda: f"{R.randint(2, 9)}{R.choice('ABCDEF')}"),
         ("contact's surname", lambda: R.choice(SUR)),
         ("flight number", lambda: f"{R.choice(['QR', 'LH', 'NH', 'KL', 'AF', 'SQ'])} {R.randint(100, 9999)}"),
         ("Wi-Fi password", lambda: f"{R.choice(WRD)}{R.randint(10, 99)}"),
         ("parking spot", lambda: f"{R.choice('PQRSTU')}-{R.randint(10, 99)}"),
         ("order number", lambda: str(R.randint(100000, 9999999)))]
for k in range(30):
    fs = R.sample(FACTS, 5); vals = [(n, g()) for n, g in fs]; ask = R.choice(vals)
    t1 = ("Please keep the following details in mind; I will ask about one of them later. For now just reply \"Noted.\"\n\n"
          + "\n".join(f"- {n[0].upper() + n[1:]}: {v}" for n, v in vals))
    t3 = (f"Back to the details I asked you to keep in mind at the very start of our conversation: what was the {ask[0]}? "
          "Put only the value in \\boxed{}.")
    rows.append({"id": f"fact/{k}", "task": "fact", "gold": ask[1], "turns": [t1, body(next(dist)["prompt"]), t3], "caps": [600, 3000, 1500]})

# chain: integer-gold pairs
for k in range(30):
    p1, p2 = ints[(2 * k) % len(ints)], ints[(2 * k + 1) % len(ints)]
    t3 = ("Now take the final answer you gave to the FIRST problem in this conversation and the final answer you gave to the SECOND "
          "problem, and add them together. Put the sum in \\boxed{}.")
    rows.append({"id": f"chain/{k}", "task": "chain", "gold": str(int(p1["gold"]) + int(p2["gold"])), "golds": [p1["gold"], p2["gold"]],
                 "turns": [body(p1["prompt"]), body(p2["prompt"]), t3], "caps": [3500, 3500, 1500]})

# docq / docf: a shipment log of 100 entries, the asked entry among the first 8
CITY = ["Lyon", "Osaka", "Porto", "Tallinn", "Recife", "Hanoi", "Perth", "Quito", "Bergen", "Tunis", "Kraków", "Cusco", "Leeds", "Izmir",
        "Dakar", "Busan", "Graz", "Mysore", "Cork", "Austin", "Malmö", "Sapporo", "Valencia", "Nairobi"]
ITEM = ["ceramic tiles", "bicycle parts", "coffee beans", "solar panels", "wool blankets", "glass bottles", "printer toner", "olive oil",
        "copper wire", "garden tools", "violin strings", "rice paper", "lab gloves", "marble slabs", "tea leaves", "LED bulbs",
        "cotton thread", "brake pads", "maple syrup", "vinyl records", "ski boots", "honey jars", "chess sets", "fishing nets"]
for k in range(40):
    combos = R.sample([(i, c) for i in ITEM for c in CITY], 100); lines = []
    for n, (it, c) in enumerate(combos):
        dst = R.choice([x for x in CITY if x != c]); code = f"{R.choice('BCDFGHJKLMNPQRSTVWXZ')}{R.choice('BCDFGHJKLMNPQRSTVWXZ')}-{R.randint(1000, 9999)}"
        lines.append((f"S{n + 1:02d}: {R.randint(3, 95)} kg of {it} left {c} for {dst} on day {R.randint(1, 60)}, tracking code {code}.", it, c, code))
    tgt = lines[R.randint(0, 7)]
    log_ = "Shipment log:\n" + "\n".join(x[0] for x in lines)
    q = f"What was the tracking code of the shipment of {tgt[1]} that left {tgt[2]}? Put only the code in \\boxed{{}}."
    if k < 20:
        t = log_ + "\n\n" + q; task = "docq"
    else:
        t = ("I will give you a shipment log. After reading it, answer this question: " + q + "\n\n" + log_
             + "\n\nNow answer the question I asked before the log."); task = "docf"
    rows.append({"id": f"{task}/{k % 20}", "task": task, "gold": tgt[3], "turns": [t], "caps": [1500]})

# code
for k in range(30):
    A, B, m, c, x = R.randint(2, 9), R.randint(1, 20), R.randint(2, 5), R.randint(2, 6), R.randint(5, 40)
    rr = R.randint(0, m - 1); off = R.randint(1, 9)
    src = (f"def f(x):\n    a = {A}\n    b = {B}\n    if x % {m} == {rr}:\n        return x * a - b\n"
           f"    y = x + b\n    if y > {R.randint(20, 50)}:\n        return y - {off}\n    return y * {c}\n")
    env = {}; exec(src, env); g = env["f"](x)
    t1 = "Here is a Python function. Keep it in mind; I will ask about it later. For now just reply \"Got it.\"\n\n```python\n" + src + "```"
    t3 = f"What does f({x}) return for the Python function I showed you at the start of our conversation? Put only the number in \\boxed{{}}."
    rows.append({"id": f"code/{k}", "task": "code", "gold": str(g), "turns": [t1, body(next(dist)["prompt"]), t3], "caps": [600, 3000, 2000]})

with open(out, "w") as f:
    for r in rows: r["prompt"] = SYS; f.write(json.dumps(r, ensure_ascii=False) + "\n")
from collections import Counter
print(len(rows), Counter(r["task"] for r in rows))
