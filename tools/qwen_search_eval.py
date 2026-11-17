"""Qwen3-1.7B (4-bit weights with --wq) on the app's search questions with the app's search environment (tools/wiki_env.py:
Wikipedia's API, <search>query</search> -> the top page 256 tokens at a time, <more> for the next slice). The protocol is
explained in the system prompt (the model was not trained on it); the recommended assistant system prompt in front, no
thinking, greedy, repetition_penalty 1.05. Scored like tools/dc_eval.py: the gold answer in the first sentence of the answer.
  python tools/qwen_search_eval.py questions.jsonl --wq qwen3-1.7b-4bit-dwq.pt --wiki cache.json --out qs.jsonl [--n 30]
"""
import argparse, json, os, re, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wiki_env import WikiEnv

ap = argparse.ArgumentParser()
ap.add_argument("questions"); ap.add_argument("--wq", default=""); ap.add_argument("--wiki", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=30); ap.add_argument("--max-new", type=int, default=1200); ap.add_argument("--think", action="store_true")
ap.add_argument("--system", default="ctl/assist_system_ja.txt")
ap.add_argument("--force-search", type=int, default=1, help="1: the reply starts with <search> (the app searches first; the model writes the query)")
a = ap.parse_args()
torch.set_num_threads(os.cpu_count() or 4)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float32).eval()
if a.wq:
    d = torch.load(a.wq, map_location="cpu"); g = d["group"]
    for k, v in d.items():
        if not isinstance(v, dict): continue
        w = m.get_submodule(k).weight; p = v["q"]; q = torch.stack([p & 15, p >> 4], 2).view(p.size(0), -1).float()
        w.data.copy_((q * v["s"].float().repeat_interleave(g, 1) + v["b"].float().repeat_interleave(g, 1)).to(w.dtype))
    del d
SEARCH = ("For questions about facts, do not answer from memory: search first. You can search English Wikipedia. To search, write <search>keywords</search> and stop; the top page comes back as "
          "<information>...</information>. Search again with different "
          "keywords if the page does not have the answer (up to 5 searches). When you know the answer, reply with one sentence "
          "that states the answer (\"The answer is ...\").")
SYS = (open(a.system).read().strip() + "\n\n" if a.system else "") + SEARCH
env = WikiEnv(tok, a.wiki)


class Stop(StoppingCriteria):
    def __init__(s, n0): s.n0 = n0
    def __call__(s, ids, scores, **kw):
        t = tok.decode(ids[0, s.n0:][-12:]); return "</search>" in t


def run(q):
    env.reset(); text = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": q}],
                                                add_generation_prompt=True, enable_thinking=a.think, tokenize=False)
    p0 = len(text); used = 0
    if a.force_search: text += "<search>"
    for _ in range(8):
        x = tok(text, return_tensors="pt", add_special_tokens=False)
        kw = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20) if a.think else dict(do_sample=False)
        with torch.no_grad():
            y = m.generate(**x, max_new_tokens=max(a.max_new - used, 1), repetition_penalty=1.05, stopping_criteria=StoppingCriteriaList([Stop(x["input_ids"].size(1))]), **kw)
        new = y[0, x["input_ids"].size(1):]; used += new.numel(); out = tok.decode(new, skip_special_tokens=True); text += out
        if "</search>" in out:
            qq = text.rsplit("<search>", 1)[-1].split("</search>")[0]; text += env.search(qq)
        else: break
        if used >= a.max_new: break
    return text[p0:], used


def first_sentence(t):
    t = t.split("</think>")[-1]; t = t.rsplit("</information>", 1)[-1].strip(); return re.split(r"(?<=[.!?。])\s", t, maxsplit=1)[0]


rows, seen = [], set()
for l in open(a.questions):
    r = json.loads(l)
    if r["q"] not in seen: seen.add(r["q"]); rows.append(r)
rows = rows[:a.n]
done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
out = open(a.out, "a"); t0 = time.time(); nc = 0; nt = 0
for i, r in enumerate(rows):
    if f"s{i}" in done: continue
    text, n = run(r["q"]); nt += n; ok = r["gold"].lower() in first_sentence(text).lower(); nc += ok
    out.write(json.dumps({"id": f"s{i}", "cat": "search", "gold": r["gold"], "n_tok": n, "searches": env.ns, "correct": ok,
                          "page_has_gold": env.grounded(r["gold"]), "api_failed": env.failed, "text": text}, ensure_ascii=False) + "\n"); out.flush()
    print(f"[as] {i + 1}/{len(rows)}, correct {nc}, {nt / (time.time() - t0):.1f} tok/s", flush=True)
