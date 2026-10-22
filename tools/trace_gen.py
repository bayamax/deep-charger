"""The model's own reasoning traces (thinking on, vLLM, full KV) for pooler training: use-case prompts + MATH train.

  python trace_gen.py --model Qwen/Qwen3-1.7B --uc uc_pool.jsonl --n-uc 700 --n-math 500 --out traces.jsonl
Each row: {"src", "q", "prompt" (chat-templated), "text" (everything generated, <think> ... </think> answer)}.
"""
import argparse, json, random

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
ap.add_argument("--uc", required=True)
ap.add_argument("--n-uc", type=int, default=700)
ap.add_argument("--n-math", type=int, default=500)
ap.add_argument("--max-tokens", type=int, default=6000)
ap.add_argument("--out", required=True)
a = ap.parse_args()

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset

rng = random.Random(0)
items = [("uc_" + r["kind"], r["q"]) for r in map(json.loads, open(a.uc))]
rng.shuffle(items); items = items[:a.n_uc]
math = []
for sub in ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]:
    try:
        math += [r["problem"] for r in load_dataset("EleutherAI/hendrycks_math", sub, split="train") if r["level"] in ("Level 3", "Level 4", "Level 5")]
    except Exception as e:
        print(f"[trace_gen] {sub}: {e}", flush=True)
if not math:
    math = [r["question"] for r in load_dataset("openai/gsm8k", "main", split="train")]
rng.shuffle(math)
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
items += [("math", q + SUFFIX) for q in math[:a.n_math]]
rng.shuffle(items)

tok = AutoTokenizer.from_pretrained(a.model)
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False, enable_thinking=True)
           for _, q in items]
llm = LLM(a.model, max_model_len=a.max_tokens + 2048, gpu_memory_utilization=0.9, seed=0)
sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=a.max_tokens, seed=0)
outs = llm.generate(prompts, sp)
n_tok = 0
with open(a.out, "w") as f:
    for (src, q), p, o in zip(items, prompts, outs):
        t = o.outputs[0].text; n_tok += len(o.outputs[0].token_ids)
        f.write(json.dumps({"src": src, "q": q, "prompt": p, "text": t}) + "\n")
print(f"[trace_gen] {len(items)} traces, {n_tok} tokens -> {a.out}", flush=True)
