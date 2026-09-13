#!/usr/bin/env python3
"""Supervised fine-tuning of the search student on teacher-continued traces.

Each record holds the user question, the student's own search prefix (its queries and the
<information> blocks the environment served), the teacher's continuation of the thinking, and the
teacher's reply. The sequence is  chat-template(question) + "<think>\n" + prefix + thinking +
"\n</think>\n" + reply + eos.  The loss is taken on everything the model would generate itself:
the prefix (its own searches), the continuation and the reply. The prompt and every
<information>...</information> block are masked, since the environment writes those.

  python3 sft_lora.py --base /root/eval_hf200 --data traces.jsonl --out /root/sft2_hf
"""
import argparse, json, math, os, random, re, time
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="/root/eval_hf200"); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--rank", type=int, default=32); ap.add_argument("--alpha", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--epochs", type=float, default=3); ap.add_argument("--accum", type=int, default=8); ap.add_argument("--maxlen", type=int, default=3072)
ap.add_argument("--val", type=int, default=12); ap.add_argument("--log", default="/root/sft2.log"); ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
random.seed(A.seed); torch.manual_seed(A.seed)
DEV = "cuda"
tok = AutoTokenizer.from_pretrained(A.base)
model = AutoModelForCausalLM.from_pretrained(A.base, torch_dtype=torch.bfloat16).to(DEV)
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
cfg = LoraConfig(r=A.rank, lora_alpha=A.alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, cfg)
print(f"[sft] trainable {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M", flush=True)
INFO = re.compile(r"(<information>.*?</information>\n?)", re.S)
eos = tok.eos_token_id

def encode(rec):
    head = tok.apply_chat_template([{"role": "user", "content": rec["q"]}], add_generation_prompt=True, tokenize=False)
    if not head.rstrip().endswith("<think>"):
        head += "<think>\n"
    pre = rec["prefix"]
    if pre.startswith("<think>"):
        pre = pre[len("<think>"):].lstrip("\n")
    body = pre.rstrip("\n") + "\n" + rec["thinking"].strip() + "\n</think>\n" + rec["reply"].strip()
    ids, mask = tok.encode(head), []
    mask = [0] * len(ids)
    for piece in INFO.split(body):
        if not piece:
            continue
        t = tok.encode(piece, add_special_tokens=False)
        ids += t; mask += [0 if piece.startswith("<information>") else 1] * len(t)
    ids.append(eos); mask.append(1)
    return ids[:A.maxlen], mask[:A.maxlen]

rows = [json.loads(l) for l in open(A.data) if l.strip()]
random.shuffle(rows)
val, train = rows[:A.val], rows[A.val:]
enc_train = [encode(r) for r in train]; enc_val = [encode(r) for r in val]
print(f"[sft] {len(train)} train / {len(val)} val, mean len {sum(len(i) for i, _ in enc_train)/len(enc_train):.0f}, "
      f"trained tokens/seq {sum(sum(m) for _, m in enc_train)/len(enc_train):.0f}", flush=True)

def loss_of(ids, mask):
    x = torch.tensor([ids], device=DEV); m = torch.tensor([mask], device=DEV)
    logits = model(input_ids=x).logits[:, :-1].float()
    tgt = x[:, 1:]; w = m[:, 1:].float()
    l = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").reshape(w.shape)
    return (l * w).sum() / w.sum().clamp(min=1)

@torch.no_grad()
def validate():
    model.eval(); tot = 0.0
    for ids, mask in enc_val:
        tot += loss_of(ids, mask).item()
    model.train(); return tot / max(len(enc_val), 1)

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=A.lr, weight_decay=0.0)
steps = int(math.ceil(len(enc_train) * A.epochs / A.accum)); warm = max(1, steps // 20)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
log = open(A.log, "a"); v0 = validate(); log.write(f"step 0 val {v0:.4f}\n"); log.flush()
print(f"[sft] steps {steps}, val0 {v0:.4f}", flush=True)
model.train(); order = []; t0 = time.time(); best = (v0, 0)
for step in range(1, steps + 1):
    acc = 0.0
    for _ in range(A.accum):
        if not order:
            order = list(range(len(enc_train))); random.shuffle(order)
        ids, mask = enc_train[order.pop()]
        l = loss_of(ids, mask) / A.accum; l.backward(); acc += l.item()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if step % 10 == 0 or step == steps:
        v = validate(); log.write(f"step {step} train {acc:.4f} val {v:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s\n"); log.flush()
        print(f"[sft] step {step}/{steps} train {acc:.4f} val {v:.4f}", flush=True)
        if v < best[0]:
            best = (v, step); model.save_pretrained(A.out + "_adapter")
print(f"[sft] best val {best[0]:.4f} at step {best[1]}", flush=True)
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(A.base, torch_dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, A.out + "_adapter").merge_and_unload()
merged.save_pretrained(A.out, safe_serialization=True); tok.save_pretrained(A.out)
print(f"[out] {A.out}", flush=True); print("SFT2_DONE", flush=True)
