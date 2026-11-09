#!/bin/bash
# A fixed KV budget of REAL entries, no synthesis (the user, after every trained pooler turned out to be a constant prefix: stop
# synthesising, bound the KV instead). sp_qwen3.py rollout --kv-evict --kv-keep B: a true KV cache; every C tokens a stream keeps
# its prompt, its newest rw=1024 answer entries and the B older entries the model itself attended most (attention received from
# the generated tokens, layers 8-19, per step cached), as computed, at their true positions. The 60 problems at seed 1 for each
# B in $BS (default: 512, then 0 = the window alone in this exact code path, then 2048), scored, uploaded as kvk/kvk_b<B>_s1.jsonl.
# The references: the re-prefilled raw window 42 / 44, v38 44 / 41, the full-context model ~50.
# 24 streams (48 ran out of memory at B=512: the rebuild held two full caches; the old layers are freed as they are copied now).
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && return 0; sleep 60; done; echo "[kk] upload of $2 FAILED"; }
for _t in $(seq 60); do curl -sS -o sp_new.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q "def rollout_kvkeep" sp_new.py && { mv sp_new.py sp_qwen3.py; break; }; sleep 120; done
curl -sS -o sp_mathcheck.py "$RAW/tools/sp_mathcheck.py?nocache=$(date +%s)"
score() { python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open(sys.argv[1])]
print(f'{sum(correct(r) for r in rs)}/{len(rs)} correct, closed {sum(\"</think>\" in r[\"text\"] for r in rs)}')" "$1"; }
SEED=${SEED:-1}; RW=${RW:-1024}
echo "[kk] ready $(date -u) | kv-keep in the script: $(grep -c 'def rollout_kvkeep' sp_qwen3.py) | budgets ${BS:-512 0 2048} | rw $RW | seed $SEED"
for B in ${BS:-512 0 2048}; do
  o=kvk_b${B}_s$SEED.jsonl; [ -s $o ] && { echo "[kk] B=$B seed $SEED already: $(score $o)"; continue; }
  t0=$(date +%s); rm -f $o.part
  python3 sp_qwen3.py rollout --traces rl_eval60.jsonl --no-sp --kv-evict --kv-keep $B --kv-keep-path --att-layers 8-19 --rw $RW --C 128 --maxd 0 --max-len 12288 \
    --roll-batch ${RB:-24} --kv-budget 90000 --seed $SEED --roll-out $o.part --log sp_kvk.log > run_kvk.log 2>&1
  if [ -s $o.part ]; then mv $o.part $o; put $o kvk/$o
    echo "[kk] B=$B seed $SEED: $(score $o) ($(( $(date +%s) - t0 )) s; $(grep -oE 'kv-keep [0-9]+: .*' sp_kvk.log | tail -1 | cut -c1-120)) | refs: raw window 42 / 44, v38 44 / 41, full context ~50 $(date -u)"
  else echo "[kk] B=$B seed $SEED FAILED: $(grep -E 'Error|error' run_kvk.log | tail -1 | cut -c1-200)"; fi
  [ -e STOP_KK ] && break
done
echo "[kk] JOB_DONE $(date -u)"
