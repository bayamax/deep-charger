"""Use the model as an AI assistant on ctl/assist_prompts.jsonl (built by ctl/assist_prompts.py) and write every answer.
Thinking rows use the recommended sampling (T 0.6, top-p 0.95, top-k 20, fixed seed); the rest are greedy, so the 16-bit and the
4-bit model can be compared answer by answer. --wq loads 4-bit weights from tools/wq.py / wq_opd.py (dequantised: the same
numbers a 4-bit kernel computes). Runs on the CPU in float32 (bfloat16 matmuls are very slow there).
  python tools/assist_eval.py --wq qwen3-1.7b-4bit-dwq.pt --out assist_q4.jsonl
Score with tools/assist_grade.py.
"""
import argparse, json, os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-1.7B"); ap.add_argument("--wq", default="")
ap.add_argument("--rows", default="ctl/assist_prompts.jsonl"); ap.add_argument("--out", required=True)
ap.add_argument("--system", default="", help="a system prompt put in front of every row (before the row's own system message)")
ap.add_argument("--sample", action="store_true", help="non-thinking rows sampled too (Qwen's non-thinking setting: T 0.7, top-p 0.8, top-k 20)")
ap.add_argument("--rep", type=float, default=1.0, help="repetition_penalty")
ap.add_argument("--only", default="", help="comma-separated categories to run")
ap.add_argument("--device", default="cpu"); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.set_num_threads(os.cpu_count() or 4)
dev = a.device; dt = torch.float32 if dev == "cpu" else torch.bfloat16
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).to(dev).eval()
if a.wq:
    d = torch.load(a.wq, map_location="cpu"); g = d["group"]; n = 0
    for k, v in d.items():
        if not isinstance(v, dict): continue
        w = m.get_submodule(k).weight; p = v["q"]; q = torch.stack([p & 15, p >> 4], 2).view(p.size(0), -1).float()
        w.data.copy_((q * v["s"].float().repeat_interleave(g, 1) + v["b"].float().repeat_interleave(g, 1)).to(w.dtype)); n += 1
    print(f"[as] {a.wq}: {n} linears at {d['bits']}-bit", flush=True); del d
rows = [json.loads(l) for l in open(a.rows)]
done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
todo = [r for r in rows if r["id"] not in done and (not a.only or r["cat"] in a.only.split(","))]
def msgs(r):
    if not a.system: return r["messages"]
    ms = [dict(x) for x in r["messages"]]
    if ms[0]["role"] == "system": ms[0]["content"] = a.system + "\n\n" + ms[0]["content"]
    else: ms = [{"role": "system", "content": a.system}] + ms
    return ms
BATCHES = []
for th in (False, True):   # never mix thinking and non-thinking rows in one batch; similar lengths together
    rs = sorted([r for r in todo if r["think"] == th], key=lambda r: r["max_new"])
    BATCHES += [rs[i:i + a.batch] for i in range(0, len(rs), a.batch)]
out = open(a.out, "a"); t0 = time.time(); ntok = 0; nd = 0
for b in BATCHES:
    th = b[0]["think"]
    txt = [tok.apply_chat_template(msgs(r), add_generation_prompt=True, enable_thinking=th, tokenize=False) for r in b]
    x = tok(txt, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    torch.manual_seed(a.seed)
    kw = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20) if th else (dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20) if a.sample else dict(do_sample=False))
    if a.rep != 1.0: kw["repetition_penalty"] = a.rep
    with torch.no_grad():
        y = m.generate(**x, max_new_tokens=max(r["max_new"] for r in b), pad_token_id=tok.pad_token_id, **kw)
    for r, row in zip(b, y[:, x["input_ids"].size(1):]):
        ids = row.tolist(); ids = ids[:ids.index(tok.pad_token_id)] if tok.pad_token_id in ids else ids
        ids = ids[:r["max_new"]]; ntok += len(ids)
        out.write(json.dumps({"id": r["id"], "cat": r["cat"], "think": th, "n_tok": len(ids), "text": tok.decode(ids, skip_special_tokens=True)}, ensure_ascii=False) + "\n"); out.flush()
    nd += len(b); print(f"[as] {nd}/{len(todo)} rows, {ntok} tokens, {ntok / (time.time() - t0):.1f} tok/s", flush=True)
