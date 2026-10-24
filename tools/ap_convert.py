"""Turn the original deep-charger data (dolphin_subset.jsonl from archive/data_prep.py) into sp_qwen3 traces, with the same
filter as diag_qwen_ablation.load_samples (answer >= min_chars, no CJK / tool calls, >= 200 answer tokens; query raw text with
the tokenizer's special tokens, cut at 256; answer cut at max_ans tokens) so both evaluators score the same contexts.
  python ap_convert.py dolphin_subset.jsonl out.jsonl --skip 200 --n 24
"""
import argparse, json, re, sys
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"); ap.add_argument("--min-chars", type=int, default=3500)
ap.add_argument("--max-ans", type=int, default=768); ap.add_argument("--skip", type=int, default=0); ap.add_argument("--n", type=int, default=24)
a = ap.parse_args()
CJK = re.compile(r"[一-鿿぀-ゟ゠-ヿ]")
TOOL = re.compile(r'</?tool_call>|<tools>|</tools>|"name"\s*:\s*"[A-Za-z_][A-Za-z0-9_]*"\s*,\s*"arguments"')
tok = AutoTokenizer.from_pretrained(a.model)
kept = []
for line in open(a.src):
    r = json.loads(line); q = next((t["content"] for t in r["input"] if t["role"] == "user"), ""); ans = r["output"]
    if not q or not ans or len(ans) < a.min_chars or CJK.search(ans) or CJK.search(q) or TOOL.search(ans) or TOOL.search(q): continue
    qi = tok(q, max_length=256, truncation=True, add_special_tokens=True).input_ids
    ai = tok(ans, max_length=a.max_ans, truncation=True, add_special_tokens=False).input_ids
    if len(ai) < 200: continue
    kept.append((qi, ai))
    if len(kept) >= a.skip + a.n: break
with open(a.out, "w") as f:
    for qi, ai in kept[a.skip:a.skip + a.n]:
        f.write(json.dumps({"prompt": tok.decode(qi), "text": tok.decode(ai)}) + "\n")
print(f"[ap] {len(kept[a.skip:a.skip + a.n])} contexts -> {a.out}")
