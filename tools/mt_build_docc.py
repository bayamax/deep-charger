"""The log lookup fed in pieces (the user: just feed the document in small pieces): from the docf rows of ctl/mt_eval.jsonl, the
question first, then the 100-entry log in 10 user turns of 10 entries; after each piece the model says the code if the piece
holds the asked shipment, else "Not here."; the last turn asks for the code without repeating which shipment.
  python tools/mt_build_docc.py ctl/mt_eval.jsonl ctl/mt_docc.jsonl
"""
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]; out = []
for r in rows:
    if r["task"] != "docf": continue
    t = r["turns"][0]; q = t.split("answer this question: ", 1)[1].split("\n\n", 1)[0]
    lines = [x for x in t.split("Shipment log:\n", 1)[1].split("\n\n", 1)[0].split("\n") if x.strip()]
    assert len(lines) == 100, len(lines)
    q0 = q.split(" Put only")[0]; NL = "\n"
    turns = [f"I will give you a shipment log in 10 parts. My question: {q0}\n"
             "After each part, if that part contains the shipment I asked about, reply with its tracking code; otherwise reply "
             "\"Not here.\" Keep each reply short.\n\nPart 1 of 10:\n" + NL.join(lines[:10])]
    turns += [f"Part {k + 1} of 10:\n" + NL.join(lines[10 * k:10 * k + 10]) for k in range(1, 10)]
    turns.append("That was the whole log. What was the tracking code of the shipment I asked about at the start? Put only the code in \\boxed{}.")
    out.append({"id": r["id"].replace("docf", "docc"), "task": "docc", "gold": r["gold"], "prompt": r["prompt"], "turns": turns,
                "caps": [700] * 10 + [1200]})
with open(sys.argv[2], "w") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(out), "rows")
