#!/usr/bin/env python3
"""Reply quality, copy against 2.1.0, judged by a reward model NOT used in training (2026-10-15, the user: what matters is
how good the replies read, not the accuracy; and the training reward model's own reading is no yardstick). For two screen
outputs over the same questions (single-turn: q; multi-turn: dialog + turn), every pair where both replies are correct is
scored by the judge; the copy wins a pair when its reply scores higher. Also: the pairs where one side is right only,
the mean reply lengths, and a sample of pairs written out for a reader.
  PYTHONPATH=/root/rmenv python3 rm_pair.py <judge[@4bit]> <copy.jsonl> <start.jsonl> <tag> <pairs_out.jsonl>
Prints: PAIR <tag> both_right <n> copy_wins <w> win_rate <r> tie <t> len_copy <a> len_start <b> right_only_copy <x> right_only_start <y>
"""
import json, sys, random
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
judge, fc, fs, tag, outp = sys.argv[1:6]
dev = "cuda" if torch.cuda.is_available() else "cpu"
name, _, q = judge.partition("@")
kw = {"num_labels": 1}
if q in ("8bit", "4bit"):
    from transformers import BitsAndBytesConfig
    kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True) if q == "8bit" else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    kw["device_map"] = {"": 0}; kw["torch_dtype"] = torch.bfloat16
    model = AutoModelForSequenceClassification.from_pretrained(name, **kw).eval()
else:
    model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32, **kw).to(dev).eval()
tok = AutoTokenizer.from_pretrained(name)


def key(r): return (r.get("dialog"), r.get("turn")) if r.get("dialog") is not None else (r.get("q"), 0)
def reply(r): return (r.get("reply") or r.get("answer") or "").strip()


def load(f):
    out = {}
    for l in open(f):
        if l.strip():
            r = json.loads(l); out[key(r)] = r
    return out


A, B = load(fc), load(fs)
both = [k for k in A if k in B]
def conv(r):
    h = [{"role": m["role"], "content": m["content"]} for m in (r.get("hist") or []) if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    return h + [{"role": "user", "content": r.get("ran") or r.get("q") or ""}, {"role": "assistant", "content": reply(r)}]
pairs = [k for k in both if A[k].get("correct") and B[k].get("correct") and reply(A[k]) and reply(B[k])]
texts = []
for k in pairs: texts += [tok.apply_chat_template(conv(A[k]), tokenize=False), tok.apply_chat_template(conv(B[k]), tokenize=False)]
sc = []
with torch.no_grad():
    for i in range(0, len(texts), 8):
        enc = tok(texts[i:i + 8], return_tensors="pt", padding=True, truncation=True, max_length=2048, add_special_tokens=False).to(model.device)
        sc += model(**enc).logits[:, 0].float().cpu().tolist()
w = t = 0; rows = []
for j, k in enumerate(pairs):
    a, b = sc[2 * j], sc[2 * j + 1]; w += a > b; t += a == b
    rows.append({"q": A[k].get("ran") or A[k].get("q"), "turn": k[1], "copy": reply(A[k]), "start": reply(B[k]), "judge_copy": round(a, 3), "judge_start": round(b, 3)})
random.Random(3).shuffle(rows)
open(outp, "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
ro_c = sum(1 for k in both if A[k].get("correct") and not B[k].get("correct")); ro_s = sum(1 for k in both if B[k].get("correct") and not A[k].get("correct"))
la = [len(reply(A[k])) for k in pairs]; lb = [len(reply(B[k])) for k in pairs]; m = lambda x: sum(x) / len(x) if x else float("nan")
print(f"PAIR {tag} both_right {len(pairs)} copy_wins {w} win_rate {w / max(len(pairs), 1):.3f} tie {t} len_copy {m(la):.0f} len_start {m(lb):.0f} right_only_copy {ro_c} right_only_start {ro_s}", flush=True)
