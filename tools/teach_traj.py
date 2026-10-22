#!/usr/bin/env python3
"""A hosted teacher (DeepSeek by default) solves training questions in the student's own search environment - the local
store, pages of 256 tokens, at most --maxs searches - writing in the student's trace style, and the trajectories that
pass the rules become supervised rows for memfit.py --objective ce (q_ids, gen, the served blocks masked by own_mask):
the NPC recipe (teacher data for the weak spots, rules-filtered) applied to the search model.

Rules a trajectory must pass: the reply's first sentence carries the gold; the gold appeared in a served block
(grounded); no query contains the gold (no answer-first searches); at most --maxs searches; the trace has no
<information> text of the teacher's own making.

  DEEPSEEK_API_KEY=... python3 teach_traj.py --store S --bge B --tok T --corpus corpus.jsonl --exclude eval300.jsonl --n 300 --out traj.jsonl
"""
import argparse, collections, json, os, random, re, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--store", required=True); ap.add_argument("--bge", required=True); ap.add_argument("--tok", required=True)
ap.add_argument("--corpus", required=True); ap.add_argument("--exclude", default=""); ap.add_argument("--n", type=int, default=300); ap.add_argument("--skip", type=int, default=0)
ap.add_argument("--out", required=True); ap.add_argument("--model", default="deepseek-chat"); ap.add_argument("--api", default="deepseek", choices=["deepseek", "openai"])
ap.add_argument("--temp", type=float, default=0.7); ap.add_argument("--maxs", type=int, default=5); ap.add_argument("--maxm", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--localsearch", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ctl", "localsearch"))
A = ap.parse_args()
PAGE_STEP = 256
NOTICE = "(no searches left - answer from what you have read)"; NOMORE = "(no more of this page - search again or answer)"
sys.path.insert(0, A.localsearch)
from search import LocalSearch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(A.tok)
LS = LocalSearch(A.store, A.bge)
cache = {}; CF = A.out + ".pages"
if os.path.exists(CF):
    for l in open(CF):
        try: d = json.loads(l); cache[d["kw"]] = d["page"]
        except Exception: pass
cfh = open(CF, "a")


def get_page(kw):
    if kw not in cache:
        cache[kw] = LS.fetch(kw); cfh.write(json.dumps({"kw": kw, "page": cache[kw]}, ensure_ascii=False) + "\n"); cfh.flush()
    return cache[kw]


def norm(s): return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()
def has(t, g): return (" " + norm(g) + " ") in (" " + norm(t) + " ")
ABBR = {"st", "mr", "mrs", "ms", "dr", "jr", "sr", "mt", "vs", "no", "inc", "ltd", "co"}


def head_sentence(a):
    h = (a or "").strip().split("\n")[0]
    for m in re.finditer(r"[.!?](?=\s|$)", h):
        t = re.split(r"[\s(\"]", h[:m.start()])[-1]
        if len(t) == 1 or t.lower() in ABBR: continue
        return h[:m.end()][:120]
    return h[:120]


EXAMPLE = ("We need answer. Need find state common between 2008 McDonald's All-American Girls Game and Bradley Center. Need use search. "
           "Query: 2008 McDonald's All-American Girls Game. Let's search.\n<search>2008 McDonald's All-American Girls Game</search>\n"
           "[the page is shown here]\n"
           "From the chunk, the game was at the Bradley Center in Milwaukee, Wisconsin. So the state is Wisconsin. The Bradley Center is "
           "in Milwaukee too, same state. The answer is Wisconsin.\n</think> The answer is Wisconsin.")
SYS = ("You solve a question with ONE tool, a Wikipedia search, thinking aloud in short plain sentences exactly in the style of the "
       "example below (clipped notes: 'We need answer. Need find ... Let's search.'). To search, write <search>keywords</search> and stop "
       "at once; the first part of the best page is then shown to you inside <information>...</information>. Write <more> and stop to "
       f"read the next part of the same page. At most {A.maxs} searches. Search for what you need to find, never for the answer itself. "
       "When you know the answer, write a line of reasoning ending with 'The answer is X.' then on a new line '</think> The answer is X.' "
       "(X as short as possible, as Wikipedia would name it). Never write <information> yourself.\n\nExample of the style:\n" + EXAMPLE)
URL = "https://api.deepseek.com/chat/completions" if A.api == "deepseek" else "https://api.openai.com/v1/chat/completions"
KEY = os.environ.get("DEEPSEEK_API_KEY" if A.api == "deepseek" else "OPENAI_API_KEY", "")
usage = collections.Counter()


def chat(msgs):
    body = {"model": A.model, "messages": msgs, "temperature": A.temp, "max_tokens": 500, "stop": ["</search>", "<more>", "</think>"]}
    for t in range(4):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
            d = json.load(urllib.request.urlopen(req, timeout=180)); u = d.get("usage", {})
            usage["in"] += u.get("prompt_tokens", 0); usage["out"] += u.get("completion_tokens", 0)
            return d["choices"][0]["message"]["content"] or "", d["choices"][0].get("finish_reason")
        except Exception as ex:
            err = ex; time.sleep(3 * (t + 1))
    print(f"[api] failed: {str(err)[:200]}", file=sys.stderr); return "", "error"


def rollout(q, gold):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    trace = ""; ns = nm = 0; served, queries, page_ids, off = [], [], [], 0
    for step in range(A.maxs + A.maxm + 3):
        out, fin = chat(msgs)
        out = out.replace("</search>", "").replace("<more>", "")
        m = re.search(r"<search>(.*)$", out, re.S)
        if "<information>" in out: return None, "wrote information"
        if m and "answer is" not in m.group(1).lower():
            ns += 1; kw = m.group(1).strip(); queries.append(kw)
            if ns > A.maxs: blk = NOTICE
            else:
                pg = get_page(kw) if kw else ""
                if not pg: blk = "(no results)"; page_ids, off = [], 0
                else: page_ids = tok.encode(pg, add_special_tokens=False); blk = tok.decode(page_ids[:PAGE_STEP]); off = PAGE_STEP; served.append(blk)
            trace += out.rstrip() + "</search\n<information>\n" + blk + "\n[READER] (no extraction)\n</information>\n"
            msgs.append({"role": "assistant", "content": out + "</search>"}); msgs.append({"role": "user", "content": f"<information>\n{blk}\n</information>"}); continue
        if out.rstrip().endswith("<more") or re.search(r"<more\s*/?>\s*$", out):
            nxt = page_ids[off:off + PAGE_STEP] if nm < A.maxm else []
            if nxt: nm += 1; off += len(nxt); blk = tok.decode(nxt); served.append(blk)
            else: blk = NOMORE
            trace += re.sub(r"<more\s*/?>?\s*$", "", out).rstrip() + "\n<more>\n<information>\n" + blk + "\n[READER] (no extraction)\n</information>\n"
            msgs.append({"role": "assistant", "content": out + "<more>"}); msgs.append({"role": "user", "content": f"<information>\n{blk}\n</information>"}); continue
        # the end: reasoning that ends with 'The answer is X.', then the reply
        body = out.split("</think>")[0].strip()
        am = re.search(r"[Tt]he answer is (.+?)\.?\s*$", body.split("\n")[-1]) or re.search(r"[Tt]he answer is ([^\n.]+)", body)
        if not am: return None, "no answer line"
        reply = "The answer is " + am.group(1).strip().rstrip(".") + "."
        trace += body + "\n</think> " + reply
        break
    else:
        return None, "too many steps"
    ans = head_sentence(reply)
    rec = {"q": q, "gold": gold, "ns": ns, "more": nm, "queries": queries, "answer": ans, "grounded": any(has(s, gold) for s in served), "correct": has(ans, gold),
           "answer_first": any(has(kw, gold) for kw in queries), "text": trace}
    return rec, None


rows = [json.loads(l) for l in open(A.corpus) if l.strip()]
ex = set()
if A.exclude:
    for l in open(A.exclude):
        try: ex.add(json.loads(l)["q"])
        except Exception: pass
items = [(r["q"], r["gold"].strip()) for r in rows if r.get("q") and (r.get("gold") or "").strip() and len(r["gold"].split()) <= 6 and r["q"] not in ex]
random.Random(A.seed).shuffle(items); items = items[A.skip:A.skip + A.n]
done = set()
if os.path.exists(A.out):
    for l in open(A.out):
        try: done.add(json.loads(l)["q"])
        except Exception: pass
stat = collections.Counter(); t0 = time.time()
with open(A.out, "a") as fh:
    for i, (q, gold) in enumerate(items):
        if q in done: continue
        rec, why = rollout(q, gold)
        if rec is None: stat["dropped:" + why] += 1; fh.write(json.dumps({"q": q, "gold": gold, "dropped": why}) + "\n"); fh.flush(); continue
        passed = rec["correct"] and rec["grounded"] and not rec["answer_first"] and rec["ns"] <= A.maxs
        rec["passed"] = passed
        if passed:   # the supervised row: the student's prompt ids, then the trace as it would have generated it (served blocks included)
            q_ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False) + "<think>\n")
            rec["q_ids"] = q_ids; rec["gen"] = tok.encode(rec["text"], add_special_tokens=False); rec["dialog"] = f"t{i}"; rec["turn"] = 0
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
        stat["n"] += 1; stat["correct"] += rec["correct"]; stat["grounded"] += rec["grounded"]; stat["answer_first"] += rec["answer_first"]; stat["passed"] += passed; stat["searches"] += rec["ns"]
        if (i + 1) % 10 == 0:
            n = max(stat["n"], 1)
            print(f"[teach] {i + 1}/{len(items)}: correct {100 * stat['correct'] / n:.0f}% grounded {100 * stat['grounded'] / n:.0f}% answer-first {100 * stat['answer_first'] / n:.0f}% "
                  f"passed {stat['passed']} ({100 * stat['passed'] / n:.0f}%) searches {stat['searches'] / n:.1f}; dropped {sum(v for k, v in stat.items() if k.startswith('dropped'))} "
                  f"[{(time.time() - t0) / 60:.1f} min, tokens in {usage['in']} out {usage['out']}]", flush=True)
print(f"[teach] done: {dict(stat)}; tokens in {usage['in']} out {usage['out']}", flush=True)
