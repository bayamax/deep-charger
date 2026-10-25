"""Teacher traces without a length cap from the full-KV model (vLLM): reads rows with "prompt", writes them back with "text"
(the generated thought + answer) and "finish" (stop / length). Qwen3's recommended thinking sampling.

  python gen_long_vllm.py prompts.jsonl out.jsonl [max_tokens]
"""
import json, sys
from vllm import LLM, SamplingParams

src, dst = sys.argv[1], sys.argv[2]; mx = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
rows = [json.loads(l) for l in open(src)]
llm = LLM("Qwen/Qwen3-1.7B", dtype="bfloat16", gpu_memory_utilization=0.88, max_model_len=mx + 1024, seed=0)
sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=mx, seed=0)
outs = llm.generate([r["prompt"] for r in rows], sp)
with open(dst, "w") as f:
    for r, o in zip(rows, outs):
        r["text"] = o.outputs[0].text; r["finish"] = o.outputs[0].finish_reason; r["n_tok"] = len(o.outputs[0].token_ids)
        f.write(json.dumps(r) + "\n")
L = sorted(r["n_tok"] for r in rows)
print(f"[gen] {len(rows)} traces, closed {sum('</think>' in r['text'] for r in rows)}, stopped {sum(r['finish'] == 'stop' for r in rows)}, "
      f"tokens median {L[len(L) // 2]} p90 {L[int(len(L) * .9)]} max {L[-1]}, over 3200: {sum(x > 3200 for x in L)}")
