#!/bin/bash
# --ctl-window segment 1 (KV memory): 40/60 (segment 8 without it 35 / 37, raw window 42 / 44) and only 3 closed-but-wrong (raw
# 7-10), but still 18 unclosed (raw 8-10) - nearly the same problems as before. Teacher-forced, the control tokens' mass already
# sits at the raw-window model's; the drift is in what each paragraph opens with (raw window: So 12.4% / Wait 5.3% /
# Alternatively 3.4%; memory: 7.7 / 8.9 / 6.3; full KV 8.8 / 6.5 / 6.2). So --ctl-para: at paragraph openings the whole target
# is the raw-window model's distribution, everything else as --ctl-window. From ctl segment 1, 2 segments; after each the
# held-out statistics per domain and the 60 problems (seed 1; seed 2 too after the last).
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
for _t in $(seq 60); do curl -sS -o sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q '"--ctl-para"' sp_qwen3.py && break; sleep 120; done
LG=/root/w/sp_kvp.log
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
[ -s kvp.pt ] || { cp kvr.pt kvp.pt; echo "[kp] start from ctl-window segment $(cat kvr_seg)"; }
echo "[kp] ready $(date -u) | ctl-para in the script: $(grep -c '"--ctl-para"' sp_qwen3.py)"
st() {
  local lab=$1 tr=$2 ml=$3; shift 3
  timeout 2700 python3 sp_qwen3.py eval --traces $tr --max-len $ml $M --end-stat "$@" --log $LG > run_kp.log 2>&1
  echo "[kp] $lab: $(grep '\[sp\] eval' run_kp.log | tail -1 | grep -oP 'KL [0-9.]+  top1 agree [0-9.]+%  reverse KL [0-9.]+') | $(grep '\[sp\] end-stat' run_kp.log | tail -1 | grep -oP 'mean p\(Wait.*')"
  grep -E "Error|Traceback" run_kp.log | tail -1 | sed "s/^/[kp] $lab error: /"; }
std() { local lab=$1; shift
  for d in chat code comp math mc; do python3 -c "
import json, sys; [sys.stdout.write(l) for l in open('rm_held.jsonl') if json.loads(l).get('src', '?').split('-')[0] == sys.argv[1]]" $d > held_$d.jsonl
    st "$lab [$d]" held_$d.jsonl 6144 "$@"; done; }
score() { python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open(sys.argv[1])]
print('[kp] ' + sys.argv[2] + f': {sum(correct(r) for r in rs)}/{len(rs)} correct, closed {sum(\"</think>\" in r[\"text\"] for r in rs)}')" "$1" "$2"; }
gen() {
  [ -s kvp_$1_s$2.jsonl ] && return
  python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $M --max-len 12288 --ckpt kvp.pt --roll-batch 48 --grpo-tok-budget 60000 --seed $2 --roll-out kvp_$1_s$2.jsonl --log $LG > run_kpgen.log 2>&1
  grep -E "Error|Traceback" run_kpgen.log | tail -1 | sed "s/^/[kp] gen $1 s$2: /"
  score kvp_$1_s$2.jsonl "ctl-para segment $(cat kvp_seg) KV memory seed $2"; put kvp_$1_s$2.jsonl kvp/kvp_$1_s$2.jsonl; }
seg=$(cat kvp_seg 2>/dev/null || echo 0)
while [ $seg -lt 2 ] && [ ! -e STOP_KP ]; do seg=$((seg + 1)); t0=$(date +%s)
  python3 sp_qwen3.py train --traces rm_pool.jsonl --skip $(( ((seg + 9) * 600) % 9600 )) --n 600 --max-len 8192 $M --chunks-per-trace 16 --accum 8 --lr 1e-4 --const-lr --ctl-window --ctl-para --ckpt kvp.pt --seed $((seg + 200)) --out kvp_next.pt --log $LG > run_kp_tr.log 2>&1
  if [ -s kvp_next.pt ] && grep -q "^\[sp\] trained" run_kp_tr.log; then mv kvp_next.pt kvp.pt; echo $seg > kvp_seg
  else echo "[kp] segment $seg FAILED: $(grep -E 'Error' run_kp_tr.log | tail -1 | cut -c1-200)"; break; fi
  echo "[kp] ctl-para segment $seg ($(( $(date +%s) - t0 )) s): train $(grep -oP 'KL [0-9.]+ top1 [0-9.]+%.*openings [0-9.]+%' run_kp_tr.log | tail -1)"
  put kvp.pt spKVP_latest.pt
  std "held-out, ctl-para segment $seg" --ckpt kvp.pt; put $LG kvp/kvp.log
  gen seg$seg 1
done
gen seg2 2
put $LG kvp/kvp.log; date -u > DONE_KP; put DONE_KP kvp/DONE_KP; echo "[kp] JOB_DONE $(date -u)"
