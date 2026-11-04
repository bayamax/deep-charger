#!/bin/bash
# DPO with the model unfrozen (the user: the hypernet alone did not move it - unfreeze; no KL anchor, the raw-window trajectories
# the original model wrote are the anchor; collect in parallel on cheap boxes). The trainer (this box):
#  1. DPO on the 53 pairs of the first data (raw window right vs memory unclosed/wrong, memory right vs failures) from the DPO
#     reference (--ctl-window segment 1): all 28 layers + the hypernet, beta 0.1, model lr 1e-6 (8-bit AdamW, fp32 masters),
#     hypernet lr 2e-5, 2 epochs; published as archive/sp3/spKVF_latest.pt + kvf/VERSION for the collectors;
#  2. the 60 problems, seeds 1-2;
#  3. then rounds: once the collectors (ctl/kvc_job.sh, archive/sp3/dpo_col/s*/pairs_*.jsonl) have NEWP new pairs, DPO from the
#     current weights (the reference) on them, publish, the 60 problems at seed 1.
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
for _t in $(seq 60); do curl -sS -o sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q "class PrefixCache" sp_qwen3.py && break; sleep 120; done
python3 -c "import bitsandbytes" 2>/dev/null || python3 -m pip install -q bitsandbytes 2>&1 | tail -1
LG=/root/w/sp_kvf.log; NEWP=${NEWP:-50}; NSH=${NSH:-4}
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
FT="--ft-layers 28 --opt8bit --ft-lr ${FTLR:-1e-6} --lr 2e-5 --spin-beta 0.1 --epochs 2 --accum 4 --const-lr"
echo "[kf] ready $(date -u) | PrefixCache in the script: $(grep -c 'class PrefixCache' sp_qwen3.py) | $FT"
ok60() { python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open(sys.argv[1])]; print(sum(correct(r) for r in rs), sum('</think>' in r['text'] for r in rs))" "$1"; }
gen60() { [ -s $1 ] && return; python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $M --ckpt kvf.pt --max-len 12288 --roll-batch 48 --grpo-tok-budget 60000 --seed $2 --roll-out $1 --log $LG > run_kfgen.log 2>&1
  grep -E "Error|Traceback" run_kfgen.log | tail -1 | sed "s/^/[kf] gen $1: /"; put $1 kvf/$1; }
train() {   # train <pairs> <from ckpt> <round>
  python3 sp_qwen3.py train --spin --traces $1 $M --max-len 12288 --ckpt $2 $FT --seed $((4000 + $3)) --out kvf_next.pt --log $LG > run_kf_tr.log 2>&1
  grep -E "\[sp\] train .* pair|Error|Traceback|out of memory" run_kf_tr.log | tail -n 3 | cut -c1-220 | sed "s/^/[kf] round $3 train: /"
  [ -s kvf_next.pt ] && grep -q "^\[sp\] trained" run_kf_tr.log || { echo "[kf] round $3: training FAILED"; return 1; }
  mv kvf_next.pt kvf.pt; echo $3 > kvf_round; echo $3 > VERSION
  put kvf.pt spKVF_latest.pt && put VERSION kvf/VERSION; }
rd=$(cat kvf_round 2>/dev/null || echo 0)
if [ $rd = 0 ]; then
  echo "[kf] round 1: $(wc -l < dpo_pairs.jsonl) pairs from the first data"
  train dpo_pairs.jsonl dpo_ref.pt 1 || exit 1; rd=1
  for sd in 1 2; do gen60 kvf_r1_s$sd.jsonl $sd; echo "[kf] round 1 (model unfrozen) seed $sd: $(ok60 kvf_r1_s$sd.jsonl | awk '{print $1 "/60 correct, closed " $2}')"; done
fi
touch used_pairs.txt
while [ ! -e STOP_KF ]; do
  python3 - <<'PY' > new_list.txt
from huggingface_hub import HfApi
used = set(open('used_pairs.txt').read().split())
fs = [f for f in HfApi().list_repo_files('baya1116/hypernet-sp-distill') if f.startswith('archive/sp3/dpo_col/') and '/pairs_' in f and f not in used]
print('\n'.join(sorted(fs)))
PY
  : > new_pairs.jsonl
  for f in $(cat new_list.txt); do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" "$H/$f" >> new_pairs.jsonl 2>/dev/null; done
  n=$(wc -l < new_pairs.jsonl)
  if [ $n -lt $NEWP ]; then   # meanwhile collect one block of its own shard (the last one)
    curl -sS -o kvc_job.sh "$RAW/ctl/kvc_job.sh?nocache=$(date +%s)"
    SHARD=$((NSH - 1)) NSHARD=$NSH ONE=1 GB="--roll-batch 48 --grpo-tok-budget 60000" bash kvc_job.sh | grep "^\[kc\] block" | sed 's/^\[kc\]/[kf] own/' || sleep 600
    continue; fi
  rd=$((rd + 1)); cat new_list.txt >> used_pairs.txt; cp new_pairs.jsonl kvf_pairs_r$rd.jsonl; put kvf_pairs_r$rd.jsonl kvf/kvf_pairs_r$rd.jsonl
  echo "[kf] round $rd: $n new pairs from $(wc -l < new_list.txt) collector blocks ($(python3 -c "
import json, collections; print(dict(collections.Counter(json.loads(l)['why'] for l in open('kvf_pairs_r$rd.jsonl'))))"))"
  cp kvf.pt kvf_prev.pt; train kvf_pairs_r$rd.jsonl kvf_prev.pt $rd || break
  gen60 kvf_r${rd}_s1.jsonl 1; echo "[kf] round $rd seed 1: $(ok60 kvf_r${rd}_s1.jsonl | awk '{print $1 "/60 correct, closed " $2}')"
  put $LG kvf/kvf.log
done
echo "[kf] JOB_DONE $(date -u)"
