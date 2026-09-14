#!/usr/bin/env python3
"""On-policy candidates for the no-search side: the lineage's base writes K replies per real prompt.

  python3 selfgen_gpu.py --base deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --prompts chat_prompts.jsonl --out self_cands.jsonl --k 4
Batched over prompts; a candidate is usable only if it reached </think>, ended, and has no tool tags or CJK."""
import argparse, json, re, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--prompts", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=4); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--maxnew", type=int, default=360); ap.add_argument("--temp", type=float, default=0.7)
A = ap.parse_args()
tok = AutoTokenizer.from_pretrained(A.base); tok.padding_side = "left"
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(A.base, torch_dtype=torch.bfloat16).cuda().eval()
rows = [json.loads(l) for l in open(A.prompts) if l.strip()]
CJK = re.compile(r"[぀-ヿ一-鿿]")
def head(p):
    h = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
    return h if h.rstrip().endswith("<think>") else h + "<think>\n"
jobs = [(i, k) for i in range(len(rows)) for k in range(A.k)]
done = {}
t0 = time.time()
with open(A.out, "w") as fh:
    for b in range(0, len(jobs), A.batch):
        chunk = jobs[b:b + A.batch]
        enc = tok([head(rows[i]["prompt"]) for i, _ in chunk], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, do_sample=True, temperature=A.temp, top_p=0.95, max_new_tokens=A.maxnew, pad_token_id=tok.pad_token_id)
        for (i, k), seq in zip(chunk, g):
            new = seq[enc["input_ids"].shape[1]:].tolist()
            ended = tok.eos_token_id in new
            txt = tok.decode(new, skip_special_tokens=True)
            ok = "</think>" in txt
            th, rep = (txt.split("</think>", 1) if ok else (txt, ""))
            th, rep = th.strip(), rep.strip()
            usable = ok and bool(rep) and ended and not CJK.search(txt) and "<search>" not in txt and "<information>" not in txt
            fh.write(json.dumps({"prompt": rows[i]["prompt"], "category": rows[i].get("category"), "src": rows[i].get("src"), "k": k,
                                 "thinking": th, "reply": rep, "usable": usable, "ended": ended, "words": len(rep.split())}, ensure_ascii=False) + "\n")
        fh.flush()
        if (b // A.batch) % 10 == 0:
            print(f"[selfgen] {b + len(chunk)}/{len(jobs)} candidates, {time.time() - t0:.0f}s", flush=True)
print("SELFGEN_DONE", flush=True)
