#!/usr/bin/env python3
"""The candidate pages whose worth the frozen model will measure (pool_eval --force): for N training questions
(their answer known, the test questions out), the question's most frequent query, the search's top-4 pages,
plus the best-ranked page not named by the query whose opening carries the answer when the top-4 lack one
(so the pages Wikipedia's search would have reached get measured too). Writes data/reward_questions.jsonl
(q, gold) and data/reward_force.json {q: [doc ids]}."""
import json, re, os, sys, collections, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search import LocalSearch
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200; K = 5
store = sys.argv[2] if len(sys.argv) > 2 else "/tmp/wstore"; model = sys.argv[3] if len(sys.argv) > 3 else "/tmp/wstore/bge"
ls = LocalSearch(store, model)
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()
def carries(t, b, g): return (" " + norm(g) + " ") in (" " + norm(t + " " + b[:1600]) + " ")
gold_of = {}
for l in open("/tmp/hubt/box_recover/corpus.jsonl"):
    r = json.loads(l); gold_of[r["q"]] = r.get("gold", "")
P = json.load(open(os.path.join(D, "train_pairs.json")))
byq = collections.defaultdict(collections.Counter)
for p in P: byq[p["gold"]][p["query"]] += 1
qs = [q for q in byq if gold_of.get(q)]; random.seed(1); random.shuffle(qs); qs = qs[:N]
force = {}; stats = collections.Counter()
with open(os.path.join(D, "reward_questions.jsonl"), "w") as fh:
    for q in qs:
        query = byq[q].most_common(1)[0][0]; gold = gold_of[q]
        hits = ls.search(query, 64); ids = [i for _, i, _, _ in hits]
        top = ids[:K - 1]
        ql = " " + norm(query) + " "
        extra = next((i for _, i, t, b in hits[K - 1:] if carries(t, b, gold) and (" " + norm(t) + " ") not in ql), None)
        if extra is None or any(carries(t, b, gold) for _, i, t, b in hits[:K - 1]):
            extra = ids[K - 1] if len(ids) >= K else None
        else:
            stats["answer page added"] += 1
        docs = top + ([extra] if extra is not None else [])
        force[q] = docs; stats["questions"] += 1; stats["pages"] += len(docs)
        stats["questions with an answer-bearing page in the set"] += any(carries(t, b, gold) for _, i, t, b in hits if i in docs)
        fh.write(json.dumps({"q": q, "gold": gold, "query": query}, ensure_ascii=False) + "\n")
json.dump(force, open(os.path.join(D, "reward_force.json"), "w"))
print("REWARD_PREP", dict(stats))
