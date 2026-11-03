#!/bin/bash
# Overnight follow-up to the KV-memory hypernet (segment 8: held-out KL -39% vs the raw window, but 35 / 37 of 60 against the raw
# window's 42 / 44). Its generations lose on not finishing (17 unclosed vs 8-10) and carry ~60% more "Wait" / "Alternatively" /
# "Let me check" and ~75% fewer "So the answer" - the same hesitation drift the notes found for SP (stage S). Here:
#  1. stopping / hesitation / entropy statistics (--end-stat), teacher-forced, with the memory and with the raw window only:
#     on the held-out 65 traces and on the memory model's own seed-1 generations (teacher = full KV on that text);
#  2. from segment 8, train with a mixed objective (--rkl-weight 0.5: half reverse KL, mode-seeking, so a student unsure what
#     the far past said does not spread mass onto the "check again" branches), 3 segments, the same statistics after each;
#  3. the 60 problems (seeds 1-2) with that memory.
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
curl -sS -o sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
LG=/root/w/sp_kvr.log
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
echo "[kr] ready $(date -u) | entropy in end-stat: $(grep -c 'mean entropy' sp_qwen3.py)"
st() {   # st <label> <traces> <max-len> <args...>: one eval line + its end-stat line
  local lab=$1 tr=$2 ml=$3; shift 3
  python3 sp_qwen3.py eval --traces $tr --max-len $ml $M --end-stat "$@" --log $LG > run_kr.log 2>&1
  echo "[kr] $lab: $(grep '\[sp\] eval' run_kr.log | tail -1 | grep -oP 'KL [0-9.]+  top1 agree [0-9.]+%  reverse KL [0-9.]+') | $(grep '\[sp\] end-stat' run_kr.log | tail -1 | cut -d: -f2- | cut -c1-400)"
  grep -E "Error|Traceback" run_kr.log | tail -1 | sed "s/^/[kr] $lab error: /"; }
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
[ -s kvr.pt ] || cp kvm.pt kvr.pt
seg=$(cat kvr_seg 2>/dev/null || echo 0)
while [ $seg -lt 3 ] && [ ! -e STOP_KR ]; do seg=$((seg + 1)); t0=$(date +%s)
  python3 sp_qwen3.py train --traces rm_pool.jsonl --skip $(( ((seg + 7) * 600) % 9600 )) --n 600 --max-len 8192 $M --chunks-per-trace 16 --accum 8 --lr 1e-4 --const-lr --rkl-weight 0.5 --ckpt kvr.pt --seed $((seg + 100)) --out kvr_next.pt --log $LG > run_kr_tr.log 2>&1
  if [ -s kvr_next.pt ] && grep -q "^\[sp\] trained" run_kr_tr.log; then mv kvr_next.pt kvr.pt; echo $seg > kvr_seg
  else echo "[kr] rkl segment $seg FAILED: $(grep -E 'Error' run_kr_tr.log | tail -1 | cut -c1-200)"; break; fi
  echo "[kr] rkl segment $seg ($(( $(date +%s) - t0 )) s): train $(grep -oP 'KL [0-9.]+ top1 [0-9.]+%.*reverse KL [0-9.]+' run_kr_tr.log | tail -1)"
  st "held-out, rkl segment $seg" rm_held.jsonl 6144 --ckpt kvr.pt
  put kvr.pt spKVR_latest.pt; put $LG kvr/kvr.log
done
for sd in 1 2; do
  [ -s kvr_mem_s$sd.jsonl ] && continue
  python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $M --max-len 12288 --ckpt kvr.pt --roll-batch 48 --grpo-tok-budget 60000 --seed $sd --roll-out kvr_mem_s$sd.jsonl --log $LG > run_krgen.log 2>&1
  grep -E "Error|Traceback" run_krgen.log | tail -1 | sed "s/^/[kr] gen s$sd: /"
  score kvr_mem_s$sd.jsonl "rkl segment $(cat kvr_seg) KV memory seed $sd"; put kvr_mem_s$sd.jsonl kvr/kvr_mem_s$sd.jsonl
done
st "own seed-1 text, rkl memory" kvr_mem_s1.jsonl 12288 --ckpt kvr.pt
put $LG kvr/kvr.log; date -u > DONE_KR; put DONE_KR kvr/DONE_KR; echo "[kr] JOB_DONE $(date -u)"
