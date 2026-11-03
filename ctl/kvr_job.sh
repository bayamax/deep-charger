#!/bin/bash
# Overnight follow-up to the KV-memory hypernet (segment 8: held-out KL -39% vs the raw window, but 35 / 37 of 60 against the raw
# window's 42 / 44). Its generations lose on not finishing (17 unclosed vs 8-10). Their style is the full-KV model's (on the same
# 60 problems full KV: 50 correct, 2 unclosed; "Alternatively" 5.1 / "Let me check" 2.7 / "So the answer" 0.28 per 10k chars -
# the memory model 5.0 / 3.0 / 0.28, the raw window 2.9 / 1.8 / 1.03): the memory teaches the full-KV model's habit of exploring
# and cross-checking, which converges there because it can see its earlier checks, but not the recall that makes it converge.
#  1. stopping / hesitation / entropy statistics (--end-stat), teacher-forced, with the memory and with the raw window only:
#     on the held-out 65 traces and on the memory model's own seed-1 generations (teacher = full KV on that text);
#  2. from segment 8, train with --ctl-window: the full-KV teacher's distribution, but the total mass on the control tokens
#     (Wait/But/Alternatively/Hmm, </think>) set to the raw-window model's - content from full KV, the decision to re-check or
#     stop from the bounded model; 1 segment, the 60 problems at seed 1, 2 more segments, the 60 problems at seeds 1-2.
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
for _t in $(seq 60); do curl -sS -o sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q '"--ctl-window"' sp_qwen3.py && break; sleep 120; done
LG=/root/w/sp_kvr.log
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
echo "[kr] ready $(date -u) | ctl-window in the script: $(grep -c '"--ctl-window"' sp_qwen3.py)"
st() {   # st <label> <traces> <max-len> <args...>: one eval line + its end-stat line
  local lab=$1 tr=$2 ml=$3; shift 3
  timeout 2700 python3 sp_qwen3.py eval --traces $tr --max-len $ml $M --end-stat "$@" --log $LG > run_kr.log 2>&1
  echo "[kr] $lab: $(grep '\[sp\] eval' run_kr.log | tail -1 | grep -oP 'KL [0-9.]+  top1 agree [0-9.]+%  reverse KL [0-9.]+') | $(grep '\[sp\] end-stat' run_kr.log | tail -1 | cut -d: -f2- | cut -c1-400)"
  grep -E "Error|Traceback" run_kr.log | tail -1 | sed "s/^/[kr] $lab error: /"; }
std() {   # std <label> <args...>: st per held-out domain (the 65-trace file in one go once spun on the CPU for 45 min with the memory)
  local lab=$1; shift
  for d in chat code comp math mc; do python3 -c "
import json, sys; [sys.stdout.write(l) for l in open('rm_held.jsonl') if json.loads(l).get('src', '?').split('-')[0] == sys.argv[1]]" $d > held_$d.jsonl
    st "$lab [$d]" held_$d.jsonl 6144 "$@"; done; }
score() { python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open(sys.argv[1])]
print('[kr] ' + sys.argv[2] + f': {sum(correct(r) for r in rs)}/{len(rs)} correct, closed {sum(\"</think>\" in r[\"text\"] for r in rs)}')" "$1" "$2"; }
if [ ! -e kr_diag_done ]; then
  st "held-out, segment 8 memory" rm_held.jsonl 6144 --ckpt kvm.pt
  st "held-out, raw window only" rm_held.jsonl 6144 --ckpt kvm.pt --no-sp
  st "own seed-1 text, segment 8 memory" kvm_mem_s1.jsonl 12288 --ckpt kvm.pt
  st "own seed-1 text, raw window only" kvm_mem_s1.jsonl 12288 --ckpt kvm.pt --no-sp
  touch kr_diag_done; put $LG kvr/kvr.log
fi
[ -e kr_mem_held_done ] || { std "held-out, segment 8 memory" --ckpt kvm.pt; touch kr_mem_held_done; put $LG kvr/kvr.log; }
[ -s kvr.pt ] || cp kvm.pt kvr.pt
gen() {   # gen <tag> <seed>
  [ -s kvr_$1_s$2.jsonl ] && return
  python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $M --max-len 12288 --ckpt kvr.pt --roll-batch 48 --grpo-tok-budget 60000 --seed $2 --roll-out kvr_$1_s$2.jsonl --log $LG > run_krgen.log 2>&1
  grep -E "Error|Traceback" run_krgen.log | tail -1 | sed "s/^/[kr] gen $1 s$2: /"
  score kvr_$1_s$2.jsonl "ctl-window segment $(cat kvr_seg) KV memory ($1) seed $2"; put kvr_$1_s$2.jsonl kvr/kvr_$1_s$2.jsonl; }
seg=$(cat kvr_seg 2>/dev/null || echo 0)
while [ $seg -lt 3 ] && [ ! -e STOP_KR ]; do seg=$((seg + 1)); t0=$(date +%s)
  python3 sp_qwen3.py train --traces rm_pool.jsonl --skip $(( ((seg + 7) * 600) % 9600 )) --n 600 --max-len 8192 $M --chunks-per-trace 16 --accum 8 --lr 1e-4 --const-lr --ctl-window --ckpt kvr.pt --seed $((seg + 100)) --out kvr_next.pt --log $LG > run_kr_tr.log 2>&1
  if [ -s kvr_next.pt ] && grep -q "^\[sp\] trained" run_kr_tr.log; then mv kvr_next.pt kvr.pt; echo $seg > kvr_seg
  else echo "[kr] ctl segment $seg FAILED: $(grep -E 'Error' run_kr_tr.log | tail -1 | cut -c1-200)"; break; fi
  echo "[kr] ctl segment $seg ($(( $(date +%s) - t0 )) s, OOM skips $(grep -c 'out of memory' run_kr_tr.log)): train $(grep -oP 'KL [0-9.]+ top1 [0-9.]+%' run_kr_tr.log | awk 'NR==1{f=$0} {l=$0} END{print "first 25 " f " / last 25 " l}')"
  std "held-out, ctl segment $seg (KL to the true full-KV teacher)" --ckpt kvr.pt
  put kvr.pt spKVR_latest.pt; put $LG kvr/kvr.log
  [ $seg = 1 ] && gen seg1 1
done
gen seg3 1; gen seg3 2
st "own seed-1 text, ctl-window memory" kvr_seg3_s1.jsonl 12288 --ckpt kvr.pt
put $LG kvr/kvr.log; date -u > DONE_KR; put DONE_KR kvr/DONE_KR; echo "[kr] JOB_DONE $(date -u)"
