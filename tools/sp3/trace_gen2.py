"""One shard of the big trace set: the model's own full-KV traces (thinking on, vLLM) for a slice of prompts_big.jsonl.
  python trace_gen2.py --prompts prompts_big.jsonl --shard 0 --nshards 6 --max-tokens 3200 --out traces_big_0.jsonl
"""
import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-1.7B"); ap.add_argument("--prompts", required=True)
ap.add_argument("--shard", type=int, required=True); ap.add_argument("--nshards", type=int, required=True)
ap.add_argument("--max-tokens", type=int, default=3200); ap.add_argument("--out", required=True)
a = ap.parse_args()
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
rows = [json.loads(l) for i, l in enumerate(open(a.prompts)) if i % a.nshards == a.shard]
tok = AutoTokenizer.from_pretrained(a.model)
prompts = [tok.apply_chat_template([{"role": "user", "content": r["q"]}], add_generation_prompt=True, tokenize=False, enable_thinking=True) for r in rows]
llm = LLM(a.model, max_model_len=a.max_tokens + 1024, gpu_memory_utilization=0.9, seed=a.shard)
sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=a.max_tokens, seed=a.shard)
n_tok = 0
with open(a.out, "w") as f:
    for r, p, o in zip(rows, prompts, llm.generate(prompts, sp)):
        n_tok += len(o.outputs[0].token_ids)
        f.write(json.dumps({"src": r["src"], "q": r["q"], "prompt": p, "text": o.outputs[0].text}) + "\n")
print(f"[trace_gen2] shard {a.shard}/{a.nshards}: {len(rows)} traces, {n_tok} tokens -> {a.out}", flush=True)
