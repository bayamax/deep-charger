#!/bin/bash
# Iterative DPO on the KV-memory hypernet (the user: work out roughly how many pairs it needs, grow the corpus to that, not all
# at once - train, and fetch more when it runs out). Target ~200-300 pairs in all; pairs come at ~13-16% of problems, so each
# iteration takes the next 300 problems of ctl/dpo_pool.jsonl (grpo_prompts minus the eval set and the first 300) and:
#  1. the raw window 1024 (fixed - the hypernet does not touch it) and the memory with the CURRENT weights (on-policy: the
#     pairs are the current model's failures), one draw each; both sides redrawn where both failed;
#  2. pairs both ways (ctl/dpo_pairs.py); DPO from the current weights with them as the reference (beta 0.1, lr 2e-5, 2 epochs);
#  3. the 60 problems (seed 1) and the held-out KL to full KV (does the memory still carry the past?).
# Stops after MAXIT iterations, or when the 60-problem score is below the starting one by 4+ two iterations running.
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
for f in ctl/dpo_pool.jsonl ctl/dpo_pairs.py; do
  for _t in $(seq 30); do curl -sS -o $(basename $f) "$RAW/$f?nocache=$(date +%s)"; [ -s $(basename $f) ] && ! grep -q "404: Not Found" $(basename $f) && break; sleep 120; done; done
LG=/root/w/sp_kvi.log; MAXIT=${MAXIT:-6}; NP=${NP:-300}
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
[ -s kvi.pt ] || cp kvo.pt kvi.pt
it=$(cat kvi_it 2>/dev/null || echo 0)
echo "[ki] ready $(date -u) | pool $(wc -l < dpo_pool.jsonl) | from iteration $it | start: DPO round 0 (kvo.pt)"
ok60() { python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open(sys.argv[1])]; print(sum(correct(r) for r in rs), sum('</think>' in r['text'] for r in rs))" "$1"; }
gen() {   # gen <prompts> <out> <seed> <args...>
  local p=$1 o=$2 sd=$3; shift 3; [ -s $o ] && return
  python3 sp_qwen3.py rollout --traces $p $M "$@" --max-len 12288 --roll-batch 48 --grpo-tok-budget 60000 --seed $sd --roll-out $o --log $LG > run_kigen.log 2>&1
  grep -E "Error|Traceback" run_kigen.log | tail -1 | sed "s/^/[ki] gen $o: /"; put $o kvi/$o; }
# the starting point on the 60 problems (seed 1): DPO round 0 if it ran, else score it now
[ -s ki_eval_0.jsonl ] || { [ -s kvo_s1.jsonl ] && cp kvo_s1.jsonl ki_eval_0.jsonl; }
base=$(cat kvi_base 2>/dev/null); [ -n "$base" ] || { gen rl_eval60.jsonl ki_eval_0.jsonl 1 --ckpt kvi.pt; base=$(ok60 ki_eval_0.jsonl | cut -d' ' -f1); echo $base > kvi_base; }
echo "[ki] start: $base/60 at seed 1"; low=0
while [ $it -lt $MAXIT ] && [ ! -e STOP_KI ]; do it=$((it + 1)); t0=$(date +%s)
  sed -n "$(( (it - 1) * NP + 1 )),$(( it * NP ))p" dpo_pool.jsonl > ki_p$it.jsonl
  [ -s ki_p$it.jsonl ] || { echo "[ki] pool used up"; break; }
  cp kvi.pt ki_w$it.pt   # the weights that generate this iteration's memory side (and its DPO reference)
  gen ki_p$it.jsonl ki_raw_$it.jsonl $((3000 + it)) --ckpt ki_w$it.pt --no-sp
  gen ki_p$it.jsonl ki_mem_$it.jsonl $((3100 + it)) --ckpt ki_w$it.pt
  python3 - $it <<'PY'
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
it = sys.argv[1]; ok = lambda r: '</think>' in r['text'] and correct(r)
raw = {json.loads(l)['id']: json.loads(l) for l in open(f'ki_raw_{it}.jsonl')}; mem = {json.loads(l)['id']: json.loads(l) for l in open(f'ki_mem_{it}.jsonl')}
P = {json.loads(l)['id']: l for l in open(f'ki_p{it}.jsonl')}
open(f'ki_todo_{it}.jsonl', 'w').writelines(P[k] for k in P if k in raw and k in mem and not ok(raw[k]) and not ok(mem[k]))
PY
  gen ki_todo_$it.jsonl ki_raw2_$it.jsonl $((3200 + it)) --ckpt ki_w$it.pt --no-sp
  gen ki_todo_$it.jsonl ki_mem2_$it.jsonl $((3300 + it)) --ckpt ki_w$it.pt
  echo "[ki] iteration $it data: $(python3 dpo_pairs.py ki_pairs_$it.jsonl ki_raw_$it.jsonl ki_raw2_$it.jsonl -- ki_mem_$it.jsonl ki_mem2_$it.jsonl) | redrawn $(wc -l < ki_todo_$it.jsonl)"
  put ki_pairs_$it.jsonl kvi/ki_pairs_$it.jsonl; cat ki_pairs_$it.jsonl >> ki_pairs_all.jsonl
  if [ "$(wc -l < ki_pairs_$it.jsonl)" -lt 4 ]; then echo "[ki] iteration $it: too few pairs, no training"; echo $it > kvi_it; continue; fi
  python3 sp_qwen3.py train --spin --traces ki_pairs_$it.jsonl $M --max-len 12288 --ckpt ki_w$it.pt --accum 4 --lr 2e-5 --const-lr --epochs 2 --spin-beta 0.1 --seed $((3400 + it)) --out kvi_next.pt --log $LG > run_ki_tr.log 2>&1
  grep -E "\[sp\] train .* pair|Error|Traceback" run_ki_tr.log | tail -n 2 | cut -c1-200 | sed "s/^/[ki] iteration $it train: /"
  [ -s kvi_next.pt ] || { echo "[ki] iteration $it: training FAILED"; break; }
  mv kvi_next.pt kvi.pt; echo $it > kvi_it; put kvi.pt spKVI_latest.pt
  gen rl_eval60.jsonl ki_eval_$it.jsonl 1 --ckpt kvi.pt; sc=$(ok60 ki_eval_$it.jsonl)
  out=""; for d in chat code comp math mc; do python3 -c "
import json, sys; [sys.stdout.write(l) for l in open('rm_held.jsonl') if json.loads(l).get('src', '?').split('-')[0] == sys.argv[1]]" $d > held_$d.jsonl
    k=$(timeout 2700 python3 sp_qwen3.py eval --traces held_$d.jsonl --max-len 6144 $M --ckpt kvi.pt --log $LG 2>&1 | grep "\[sp\] eval" | tail -1 | grep -oP "KL \K[0-9.]+(?=  top1)"); out="$out $d $k"; done
  echo "[ki] iteration $it ($(( ($(date +%s) - t0) / 60 )) min): 60 problems seed 1 $(echo $sc | cut -d' ' -f1)/60 correct, closed $(echo $sc | cut -d' ' -f2) (start $base) | held-out KL to full KV:$out | pairs so far $(wc -l < ki_pairs_all.jsonl)"
  put $LG kvi/kvi.log
  if [ $(echo $sc | cut -d' ' -f1) -le $((base - 4)) ]; then low=$((low + 1)); else low=0; fi
  [ $low -ge 2 ] && { echo "[ki] two iterations running 4+ below the start - stopping"; break; }
done
date -u > DONE_KI; put DONE_KI kvi/DONE_KI; echo "[ki] JOB_DONE $(date -u)"
