#!/bin/bash
# DPO data (the user: DPO with the raw-window trajectories as the ideal). 300 problems from grpo_prompts.jsonl, none of the
# evaluation set (math 180 / comp 60 / mc 60; ctl/dpo_prompts300.jsonl), generated under the deployment loop to 12288 tokens:
# the raw window 1024 alone (memory unused) and the KV memory (--ctl-window segment 1, kvr.pt: 40/60, 3 closed-but-wrong,
# 18 unclosed). In blocks of 50 problems, both sides per block, each block uploaded as it finishes. Pairs are made from these
# afterwards: the same problem, the raw window right and closed (chosen) against the memory unclosed or wrong (rejected).
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
for _t in $(seq 60); do curl -sS -o dpo_prompts300.jsonl "$RAW/ctl/dpo_prompts300.jsonl?nocache=$(date +%s)"; [ "$(wc -l < dpo_prompts300.jsonl)" = 300 ] && break; sleep 120; done
LG=/root/w/sp_kvd.log
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
[ -s dpo_ref.pt ] || cp kvr.pt dpo_ref.pt
echo "[kd] ready $(date -u) | prompts $(wc -l < dpo_prompts300.jsonl) | memory weights: ctl-window segment $(cat kvr_seg)"
score() { python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open(sys.argv[1])]
print('[kd] ' + sys.argv[2] + f': {sum(correct(r) for r in rs)}/{len(rs)} correct, closed {sum(\"</think>\" in r[\"text\"] for r in rs)}')" "$1" "$2"; }
for b in 0 1 2 3 4 5; do
  sed -n "$((b * 50 + 1)),$((b * 50 + 50))p" dpo_prompts300.jsonl > dpo_b$b.jsonl
  for side in raw mem; do
    out=dpo_${side}_b$b.jsonl; [ -s $out ] && continue
    x="--ckpt dpo_ref.pt"; [ $side = raw ] && x="--ckpt dpo_ref.pt --no-sp"
    python3 sp_qwen3.py rollout --traces dpo_b$b.jsonl $M $x --max-len 12288 --roll-batch 48 --grpo-tok-budget 60000 --seed $((b + 11)) --roll-out $out --log $LG > run_kdgen.log 2>&1
    grep -E "Error|Traceback" run_kdgen.log | tail -1 | sed "s/^/[kd] gen $side b$b: /"
    score $out "block $b $side"; put $out dpo/$out
  done
done
date -u > DONE_KD; put DONE_KD dpo/DONE_KD; echo "[kd] JOB_DONE $(date -u)"
