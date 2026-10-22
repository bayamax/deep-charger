#!/usr/bin/env python3
"""An open model's replies to the use-case prompts, thinking on, no search (vLLM): what a candidate base for the app says
before any of our training. The reply is the text after </think>; an unfinished thought (no </think> within --max-tokens)
is an empty reply, as the app would show nothing. Writes {id, q, kind, model, reply, think_tokens} rows for uc_judge.py.

  python3 uc_vllm.py --model Qwen/Qwen3-1.7B --inp uc_app.jsonl --out q17.jsonl --max-tokens 4096
"""
import argparse, json, time

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--max-tokens", type=int, default=4096); ap.add_argument("--max-len", type=int, default=8192); ap.add_argument("--gpu-mem", type=float, default=0.88)
ap.add_argument("--temp", type=float, default=0.6); ap.add_argument("--top-p", type=float, default=0.95); ap.add_argument("--thinking", type=int, default=1)
A = ap.parse_args()
from vllm import LLM, SamplingParams  # noqa: E402
rows = [json.loads(l) for l in open(A.inp) if l.strip()]
llm = LLM(model=A.model, max_model_len=A.max_len, gpu_memory_utilization=A.gpu_mem, max_num_seqs=32)
tok = llm.get_tokenizer()
prompts = [tok.apply_chat_template([{"role": "user", "content": r["q"]}], tokenize=False, add_generation_prompt=True, enable_thinking=bool(A.thinking)) for r in rows]
sp = SamplingParams(temperature=A.temp, top_p=A.top_p, top_k=20, max_tokens=A.max_tokens)
t0 = time.time(); outs = llm.generate(prompts, sp); empty = 0; n_tok = 0
with open(A.out, "w") as f:
    for r, o in zip(rows, outs):
        t = o.outputs[0].text; n_tok += len(o.outputs[0].token_ids)
        if A.thinking: rep = t.split("</think>", 1)[1].strip() if "</think>" in t else ""
        else: rep = t.strip()
        empty += not rep
        f.write(json.dumps({"id": r["id"], "q": r["q"], "kind": r.get("kind"), "model": A.model, "reply": rep,
                            "think_tokens": len(tok.encode(t.split("</think>")[0])) if "</think>" in t else len(o.outputs[0].token_ids)}, ensure_ascii=False) + "\n")
print(f"[uc_vllm] {A.model} thinking={A.thinking}: {len(rows)} replies ({empty} empty), {n_tok} tokens in {time.time() - t0:.0f}s -> {A.out}", flush=True)
