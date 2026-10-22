"""Long-CoT under a bounded KV cache, WITHOUT a hypernet: the floor the pooler has to beat.

Same problems, same sampling, two ways of holding the reasoning:
  --engine vllm            full KV (nothing dropped)                -> the ceiling
  --engine hf --recent R   keep the prompt + the last R tokens only -> naive window (what bounded KV costs
                           when the evicted span is simply thrown away, i.e. the SP-evict loop with 0 soft prompts)
Evicted tokens keep their original RoPE positions (no re-rotation), as in the SP-evict loop.

Data: MATH-500 levels >= --min-level (long chains of thought), graded with math-verify on the text after </think>.
  python kvwin_eval.py --engine hf --recent 768 --model Qwen/Qwen3-1.7B --n 64 --out win768.jsonl
"""
import argparse, json, random, time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--engine", choices=["hf", "vllm"], required=True)
ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
ap.add_argument("--recent", type=int, default=768, help="hf: tokens kept after the prompt (0 = keep everything)")
ap.add_argument("--slack", type=int, default=64, help="hf: evict only when this many tokens over budget (amortised)")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--min-level", type=int, default=4)
ap.add_argument("--gen", type=int, default=8000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", required=True)
a = ap.parse_args()

from datasets import load_dataset
from transformers import AutoTokenizer
from math_verify import parse, verify

ds = [r for r in load_dataset("HuggingFaceH4/MATH-500", split="test") if int(r["level"]) >= a.min_level]
random.Random(0).shuffle(ds); ds = ds[:a.n]
tok = AutoTokenizer.from_pretrained(a.model)
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
prompts = [tok.apply_chat_template([{"role": "user", "content": r["problem"] + SUFFIX}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=True) for r in ds]
T, TOP_P, TOP_K = 0.6, 0.95, 20   # Qwen3 thinking-mode recommendation


def run_vllm():
    from vllm import LLM, SamplingParams
    llm = LLM(a.model, max_model_len=a.gen + 1024, gpu_memory_utilization=0.9, seed=a.seed)
    sp = SamplingParams(temperature=T, top_p=TOP_P, top_k=TOP_K, max_tokens=a.gen, seed=a.seed)
    return [o.outputs[0].text for o in llm.generate(prompts, sp)]


def layers_of(cache):
    if hasattr(cache, "layers"):   # transformers >= 4.56
        return [(L, "keys", "values") for L in cache.layers]
    return [(cache, i, i) for i in range(len(cache.key_cache))]


def evict(cache, keep_head, keep_tail):
    """drop cache columns [keep_head, len-keep_tail) in every layer."""
    for L, kk, vk in layers_of(cache):
        if isinstance(kk, str):
            for name in (kk, vk):
                t = getattr(L, name); setattr(L, name, torch.cat([t[:, :, :keep_head], t[:, :, -keep_tail:]], 2))
        else:
            for lst in (cache.key_cache, cache.value_cache):
                t = lst[kk]; lst[kk] = torch.cat([t[:, :, :keep_head], t[:, :, -keep_tail:]], 2)


@torch.no_grad()
def run_hf():
    from transformers import AutoModelForCausalLM, DynamicCache
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(a.device).eval()
    tok.padding_side = "left"
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    g = torch.Generator(device=a.device).manual_seed(a.seed)
    outs = []
    for b0 in range(0, len(prompts), a.batch):
        enc = tok(prompts[b0:b0 + a.batch], return_tensors="pt", padding=True, add_special_tokens=False).to(a.device)
        B, P = enc.input_ids.shape
        mask = enc.attention_mask
        pos = (mask.cumsum(1) - 1).clamp(min=0)
        cache = DynamicCache()
        lo = model(input_ids=enc.input_ids, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True).logits[:, -1]
        nxt_pos = pos[:, -1] + 1
        gen = [[] for _ in range(B)]; done = torch.zeros(B, dtype=torch.bool, device=a.device); peak = P
        t0 = time.time()
        for step in range(a.gen):
            lo = lo.float() / T
            kth = torch.topk(lo, TOP_K).values[:, -1:]
            lo = lo.masked_fill(lo < kth, -1e9)
            pr = torch.softmax(lo, -1); sp, si = pr.sort(-1, descending=True)
            cut = sp.cumsum(-1) - sp > TOP_P; sp = sp.masked_fill(cut, 0)
            nx = si.gather(1, torch.multinomial(sp, 1, generator=g)).squeeze(1)
            nx = torch.where(done, torch.full_like(nx, tok.pad_token_id), nx)
            for i in range(B):
                if not done[i]:
                    gen[i].append(int(nx[i]))
                    if int(nx[i]) in eos: done[i] = True
            if bool(done.all()): break
            mask = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype, device=a.device)], 1)
            L = mask.shape[1]
            if a.recent and L > P + a.recent + a.slack:
                evict(cache, P, a.recent - 1)   # -1: the token being fed now becomes the newest column
                mask = torch.cat([mask[:, :P], mask[:, -a.recent:]], 1)
            peak = max(peak, mask.shape[1])
            lo = model(input_ids=nx[:, None], attention_mask=mask, position_ids=nxt_pos[:, None],
                       past_key_values=cache, use_cache=True).logits[:, -1]
            nxt_pos = nxt_pos + 1
        print(f"[kvwin] batch {b0}: {step + 1} steps {time.time() - t0:.0f}s peak cache {peak}", flush=True)
        outs += [tok.decode(x, skip_special_tokens=True) for x in gen]
        del cache; torch.cuda.empty_cache()
    return outs


texts = run_vllm() if a.engine == "vllm" else run_hf()
n_ok = n_fin = 0
with open(a.out, "w") as f:
    for r, t in zip(ds, texts):
        fin = "</think>" in t
        ans = t.split("</think>")[-1] if fin else ""
        ok = bool(fin and verify(parse("$" + r["answer"] + "$"), parse(ans)))
        n_ok += ok; n_fin += fin
        f.write(json.dumps({"id": r["unique_id"], "level": r["level"], "gold": r["answer"], "ok": ok, "fin": fin,
                            "ntok": len(tok.encode(t, add_special_tokens=False)), "text": t}) + "\n")
tag = "full" if a.engine == "vllm" else f"recent{a.recent}"
print(f"[kvwin] {a.model} {tag}: {n_ok}/{len(ds)} = {100 * n_ok / len(ds):.1f}%  finished {n_fin}/{len(ds)}", flush=True)
