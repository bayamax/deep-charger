"""Teacher traces without a length cap from the full-KV model (vLLM): reads rows with "prompt", writes them back with "text"
(the generated thought + answer), "finish" (stop / length) and "n_tok". Qwen3's recommended thinking sampling. Prompts go in
groups and every group is appended at once, so a crash keeps what is done (a rerun skips the ids already written).

  python gen_long_vllm.py prompts.jsonl out.jsonl [max_tokens] [group]
"""
import json, os, sys, time
from vllm import LLM, SamplingParams

src, dst = sys.argv[1], sys.argv[2]; mx = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
grp = int(sys.argv[4]) if len(sys.argv) > 4 else 150
rows = [json.loads(l) for l in open(src)]
done = {json.loads(l)["id"] for l in open(dst)} if os.path.exists(dst) else set()
todo = [r for r in rows if r["id"] not in done]
llm = LLM("Qwen/Qwen3-1.7B", dtype="bfloat16", gpu_memory_utilization=0.88, max_model_len=mx + 1024, seed=0)
sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=mx, seed=0)
t0 = time.time()
for i in range(0, len(todo), grp):
    part = todo[i:i + grp]
    outs = llm.generate([r["prompt"] for r in part], sp, use_tqdm=False)
    with open(dst, "a") as f:
        for r, o in zip(part, outs):
            r["text"] = o.outputs[0].text; r["finish"] = o.outputs[0].finish_reason; r["n_tok"] = len(o.outputs[0].token_ids)
            f.write(json.dumps(r) + "\n")
    print(f"[gen] {len(done) + i + len(part)}/{len(rows)} written ({time.time() - t0:.0f}s)", flush=True)
rs = [json.loads(l) for l in open(dst)]
L = sorted(r["n_tok"] for r in rs)
print(f"[gen] {len(rs)} traces, closed {sum('</think>' in r['text'] for r in rs)}, stopped {sum(r['finish'] == 'stop' for r in rs)}, "
      f"tokens median {L[len(L) // 2]} p90 {L[int(len(L) * .9)]} max {L[-1]}, over 3200: {sum(x > 3200 for x in L)}")
