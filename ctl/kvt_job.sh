#!/bin/bash
# After the para-window decoding: the same weights (--ctl-para segment 2; plain decoding 35 / 41, para-window 37 at seed 1)
# decoded with the memory-masked twin in two more ways, seed 1 each: --twin-minent (per token, the lower-entropy side) and
# --twin-mix 0.5 (half the memory's logits, half the raw window's).
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
for _t in $(seq 60); do curl -sS -o sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q '"--twin-minent"' sp_qwen3.py && break; sleep 120; done
LG=/root/w/sp_kvt.log
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
echo "[kt] ready $(date -u) | twin options in the script: $(grep -c '"--twin-m' sp_qwen3.py) | weights: ctl-para segment $(cat kvp_seg)"
score() { python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open(sys.argv[1])]
print('[kt] ' + sys.argv[2] + f': {sum(correct(r) for r in rs)}/{len(rs)} correct, closed {sum(\"</think>\" in r[\"text\"] for r in rs)}')" "$1" "$2"; }
for cfg in "minent:--twin-minent" "mix05:--twin-mix 0.5"; do n=${cfg%%:*}; x=${cfg#*:}
  [ -s kvt_${n}_s1.jsonl ] && continue
  python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $M $x --max-len 12288 --ckpt kvp.pt --roll-batch 24 --grpo-tok-budget 60000 --seed 1 --roll-out kvt_${n}_s1.jsonl --log $LG > run_ktgen.log 2>&1
  grep -E "Error|Traceback" run_ktgen.log | tail -1 | sed "s/^/[kt] gen $n: /"; grep "steps taken from" $LG | tail -1 | grep -oP "steps taken.*" | sed "s/^/[kt] $n /"
  score kvt_${n}_s1.jsonl "ctl-para segment $(cat kvp_seg) + $n decoding seed 1"; put kvt_${n}_s1.jsonl kvt/kvt_${n}_s1.jsonl
done
put $LG kvt/kvt.log; date -u > DONE_KT; put DONE_KT kvt/DONE_KT; echo "[kt] JOB_DONE $(date -u)"
