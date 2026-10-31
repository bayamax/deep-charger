#!/bin/bash
# Box RL (a 24 GB card): GRPO from G under the deployment loop ([prompt | SP | raw window 1024], C 128, maxd 384, cap 12288,
# Qwen3 thinking sampling). Problems: ctl/grpo_prompts.jsonl (tools/build_grpo_prompts.py: MATH train L3-5 1200, competition
# 900, MMLU-Pro all 14 categories 896). 16 problems x 8 samples per update, reward = correct final answer. The 96-problem
# check (2 seeds) on the starting weights and every 8 updates; the best total is kept; stop after 3 checks without a new best.
# Results to archive/sp3/rl/ (grpo_best.pt, eval/train jsonl), live log archive/sp3/rl_live.log, marker DONE_RL1.
# serial 2: run 1 ran out of memory generating update 2 (48 streams of long MMLU-Pro prompts) - streams now also capped by a
# token budget (--grpo-tok-budget), fewer streams on an OOM; the starting check is known (G: 79 + 81 = 160), so no eval0.
# serial 3: 8 updates at model lr 1e-6 left the model where it was (96-problem check 156 vs 160, answer lengths within a few
# characters) - Adam moves each weight ~lr per update, and an update costs an hour here. Again from G at lr 5e-6 (pooler
# 5e-5), checks every 6 updates, each with the deployment KL to the full-KV teacher (64 MATH traces, 4096) to watch the
# drift. Outputs grpo2_*.
# serial 4: GRPO usually dips before it climbs, and serial 3 would have stopped at its next check (patience 3 on the
# 96-problem total) without ever saving its weights (only a new best was saved): u12 closed 184 of 192 (G 173) at 157 correct
# (G 160), teacher KL 0.0746 (G 0.0623). Again from G, no early stop, 60 updates; at every check the whole training state
# (weights, fp32 masters, optimizer) goes to grpo3_latest.pt (also on the hub), and a crashed run carries on from it.
# serial 5 (on the box running serial 4): no 96-problem check any more (the user: the model is not made for it, and it
# costs ~45 min per check) - the trend to watch is the GRPO problems' own accuracy. A check is now the teacher KL and the
# saved state, every 3 updates; up to 100 updates (earlier GRPO work needed ~100). Carries on from grpo3_latest.pt (u18).
# serial 6 (same box, right after the u21 save): 32 problems x 4 samples per update instead of 16 x 8 - more problems per
# step for the same compute (the user); a stricter reward (an answer whose thought never closes earns 0).
# serial 7 (replaces 6 before the box picked it up): the user wants at least 12 answers per problem - 12 problems x 12
# per update (144 answers, ~ the time of 16 x 8); a problem the model nearly always (or never) solves still splits.
# serial 8: GRPO stopped (u45 saved as grpo3_latest.pt). The user: on the GRPO problems the SP model trails the full-KV
# teacher far more than on MATH-500 (MATH L5 ~38% vs ~68% from the old teacher traces) - something is off. Measure the same
# 120 held-out GRPO problems (ctl/rl_eval120.jsonl: 24 each MATH L3/L4/L5, competition, MMLU-Pro), 2 seeds, cap 12288,
# Qwen3 thinking sampling, under: full KV (vLLM), the raw window alone (base weights, --no-sp --rw 1056), G (SP) and the
# GRPO u45 state. Results to archive/sp3/rl/ev120_*.jsonl, live log archive/sp3/rl_live.log, marker DONE_EV1.
# serial 9: the user wants it as a rough reference only - 60 of the problems (12 per type), 1 seed.
# serial 10: serial 9's restart left serial 8's vLLM engine on the GPU (its engine process outlived the killed parent), so
# serial 9's vLLM died; now every GPU process is killed first, and the four runs go two at a time (the user wants it quick):
# full KV (vLLM, half the GPU) with the raw window, then G with GRPO u45 (30 streams each).
# serial 11: nvidia-smi lists no PIDs inside the container, so serial 10 killed nothing - the orphaned vLLM engine still held
# 21 GB and every run died. Kill the vLLM venv's processes by name; if the GPU is still not free, stop and say so (the box
# then needs a container restart, after which this serial runs again).
# serial 12: two at a time ran out of memory (vLLM took more than its share, and two SP rollouts of 30 streams did not fit
# together) - full KV (50/60) and G (36/60) finished; the raw window and GRPO u45 now one after the other.
# serial 13: on the 60 problems full KV 50, raw window 44, G 36, GRPO u45 36 - the SP model is worse than the untouched
# model with only the raw window. Split the cause: G's fine-tuned weights WITHOUT SP, under full KV (exported to a HF
# folder for vLLM) and under the raw window alone. Near 50 / 44: the weights are fine and the SP path hurts; far below:
# the fine-tuning damaged the model.
# serial 14: SP rebuilt from zero (the user): the base model untouched (no fine-tuned layers), a fresh pooler, trained only
# on the rows the deployment loop scores (--par-deploy: the chunk after a full raw window of 1024), on the 2678 long
# teacher traces, 600 per segment. Stage 1 without eviction (maxd 0: everything that left the window goes into the pooler -
# an untrained pooler's attention mass would throw away what it has not yet learned to need; first learn to compress a
# finite text into the soft prompts, the hypernet's own ability), 3 segments; stage 2 with eviction (maxd 384), the rest.
# Deployment KL after each segment with and without eviction; the 60 GRPO problems after each stage.
# serial 15 (replaces 14 right after it started): no eviction on 16k traces would squeeze up to ~15k tokens into 32 soft
# prompts, far past what attention pooling holds (the user found ~4096 the limit). Step 1 of a probe: the distant buffer
# keeps the newest 2048 tokens (--fifo 2048, no learned selection), 3 segments of 600, deployment KL after each, the 60
# problems at the end; the cap / the switch to mass eviction are decided from these.
# serial 16: the cap as a curriculum (the user): 1024 -> 2048 -> 4096 over the three segments (the traces stay whole, so the
# thought's end is still trained); deployment KL at all three caps after each segment.
# serial 17/18: no curriculum - the cap at 4096 from the start (the user), and the deployment KL at caps 2048 / 4096 / 8192 after
# each segment shows where compressing more context into the soft prompts stops paying - on the 64 MATH teacher traces at
# 16384 tokens (at 4096 the distant part never passes ~2900, so caps above that would read the same).
# serial 19: nothing thrown away (the user) - the whole past outside the window goes into the 32 soft prompts (no cap, no
# eviction); the deployment KL with everything, and with caps 2048 / 4096 for comparison.
# serial 20: the training log now gives the KL by how much past lies outside the window (<1k ... 8k+) every 25 traces -
# where it starts to climb with length is the compression limit (the user); the live log carries those lines.
# serial 21 (after serial 20's three window-1024 segments and its 60-problem check): pooler pretraining at raw window 0
# (the user) - the next 128 tokens from the 32 soft prompts alone, so the pooler has to learn to compress the past (with a
# 1024 window most of what it needs is in the window and its signal is weak). Base frozen, a fresh pooler, the whole past
# (no cap); a block is [SP | 128] (8x lighter), 32 blocks per trace. Training KL by past length every 25 traces; after each
# 600 the window-0 deployment KL with all / cap 4096 / cap 2048. No generation in this stage.
# serial 22: serial 21 keeps running (its job is not touched); a side task profiles every new window-0 checkpoint by
# position in the chunk (the first token = predicted from the soft prompts alone; old pooler G: ~5.0, E: 3.0) on 16
# MATH teacher traces at 16384, into the live log.
# serial 23: one more side task - the same window-0 profile with no soft prompts (prompt + the chunk so far), the floor
# the pooler has to beat; run once with a separate copy of the trainer (sp_prof.py) so the running training is untouched.
SER=23
if [ -f /root/.boxrl_serial ] && [ "$(cat /root/.boxrl_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxrl_serial
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w
cat > /root/status.sh <<'ST'
grep -E "^\[(sz|ev|hub)\]|Error|Traceback" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-200
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.job21 ]; then touch /root/.job21
  for J in $(pgrep -f "^bash /root/job.sh") $(pgrep -f "^bash /root/live.sh"); do pkill -P $J; kill $J; done; sleep 3
  pkill -f "sp_qwen3.py"; pkill -f "pip install"; pkill -f gen_long_vllm; sleep 5
  pkill -9 -f "/root/vv/bin/python"; pkill -9 -f EngineCore; pkill -9 -f "multiprocessing.spawn"; pkill -9 -f "multiprocessing.resource_tracker"
  sleep 15; rm -f /root/w/DONE_SZ1
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${U:-0}" -gt 2000 ]; then echo "[ev] GPU still holds $U MiB after the kills: $(ps -eo pid,rss,cmd --sort=-rss | head -4 | tr '\n' ';' | cut -c1-300)"; rm -f /root/.job21; exit 0; fi
  echo "[ev] GRPO stopped $(date -u); GPU now $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  cat > /root/live.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
while [ ! -e DONE_SZ1 ]; do
  { grep -E "^\[(sz|ev|hub)\]|Error|Traceback" w.log | tail -n 30; grep "\[sp\] train .* trace" sp.log 2>/dev/null | tail -n 8; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; date -u; } > rl_live.log 2>&1
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='rl_live.log', path_in_repo='archive/sp3/rl_live.log', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1
  sleep 600
done
JOB
  cat > /root/job.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
get() { for t in 1 2 3; do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o "$2" "$H/$1" && return 0; sleep 10; done; return 1; }
for f in tools/sp_qwen3.py tools/sp_mathcheck.py ctl/rl_eval120.jsonl; do curl -sS -o $(basename $f) "$RAW/$f?nocache=$(date +%s)"; done
python3 -c "
rs = open('rl_eval120.jsonl').readlines(); open('rl_eval60.jsonl', 'w').writelines(rs[i] for i in range(len(rs)) if i % 24 < 12)"
for f in longtr_math_train_16k:lt_math longtr_diverse_16k:lt_div longtr2_mixed_16k:lt2; do [ -s ${f#*:}.jsonl ] || get archive/sp3/${f%%:*}.jsonl ${f#*:}.jsonl; done
python3 -c "
import json, random
rs = [json.loads(l) for f in ('lt_math.jsonl', 'lt_div.jsonl', 'lt2.jsonl') for l in open(f)]
ok = [r for r in rs if r.get('finish') == 'stop' and '</think>' in r['text']]; random.seed(0); random.shuffle(ok)
open('poolZ.jsonl', 'w').writelines(json.dumps(r) + '\n' for r in ok); print('[sz] pool', len(ok), 'finished long teacher traces')"
LG=/root/w/sp.log; base="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128"
klm() { python3 sp_qwen3.py eval --traces kv_full_p.jsonl --max-len 16384 $base "$@" --log $LG 2>&1 | grep -oP "KL \K[0-9.]+" | tail -1; }
echo "[sz] stage W0: pooler at raw window 0 (reference at window 1024: raw window alone 0.1405, G 0.0754, serial 20's SP 0.1004 after 1200)"
gen() {   # gen NAME CKPT: the 60 GRPO problems, 1 seed, scored by type
  python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $base --maxd $3 --max-len 12288 --roll-cb --roll-batch 48 --grpo-tok-budget 60000 --ckpt $2 --seed 1 --roll-out ev_$1_s1.jsonl --log $LG > run_ev_$1.log 2>&1
  python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open(sys.argv[1])]; by = {}
for r in rs:
    s = r['src'] if r['src'].startswith('math') else r['src'].split('-')[0]; v = by.setdefault(s, [0, 0])
    v[0] += correct(r); v[1] += 1
print('[sz] ' + sys.argv[2] + f': {sum(correct(r) for r in rs)}/{len(rs)} correct, closed {sum(\"</think>\" in r[\"text\"] for r in rs)} | ' + ' | '.join(f'{k} {v[0]}/{v[1]}' for k, v in sorted(by.items())))" ev_$1_s1.jsonl "$1 (60 GRPO problems; full KV 50, raw window 44, G 36)"
  bash /root/put_rl.sh ev_$1_s1.jsonl; }
N=$(wc -l < poolZ.jsonl); off=0; seg=0; cur=""; W="--rw 0 --C 128 --maxd 0"
while [ $off -lt $N ]; do seg=$((seg + 1)); t0=$(date +%s)
  python3 sp_qwen3.py train --traces poolZ.jsonl --skip $off --n 600 --max-len 16384 $base $W ${cur:+--ckpt $cur} --par --par-deploy --end-train --end-weight 4 \
    --chunks-per-trace 32 --accum 8 --kl-rows 64 --lr 1e-4 --seed $seg --out spW_next.pt --log $LG > run_w0_$seg.log 2>&1
  grep -E "Error|Traceback" run_w0_$seg.log | tail -2 | sed "s/^/[sz] W0.$seg: /"
  [ -s spW_next.pt ] || { echo "[sz] W0.$seg: no checkpoint - stop"; break; }
  mv spW_next.pt spW0.pt; cur=spW0.pt; off=$((off + 600))
  echo "[sz] W0.$seg (traces $off, $(( $(date +%s) - t0 )) s): train KL $(grep -oP 'KL \K[0-9.]+' run_w0_$seg.log | tail -1), window-0 deployment KL all $(klm --ckpt spW0.pt $W) / cap 4096 $(klm --ckpt spW0.pt $W --fifo 4096) / cap 2048 $(klm --ckpt spW0.pt $W --fifo 2048)"
  put spW0.pt spW0_window0.pt; put $LG spP_w0.log
done
date -u > DONE_SZ1; put DONE_SZ1 DONE_SZ1; echo "[sz] DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/live.sh' > /dev/null 2>&1 < /dev/null &
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[ev] job launched $(date -u)"
fi
if [ ! -e /root/.side22 ]; then touch /root/.side22
  cat > /root/side22.sh <<'JOB'
cd /root/w; last=0
while [ ! -e DONE_SZ1 ]; do
  if [ -s spW0.pt ] && [ $(stat -c %Y spW0.pt) -gt $last ]; then last=$(stat -c %Y spW0.pt); sleep 30
    python3 sp_qwen3.py profile --traces kv_full_p.jsonl --n 16 --max-len 16384 --S 32 --heads 8 --layers 3 --ffn 2048 --rw 0 --C 128 --maxd 0 \
      --ckpt spW0.pt 2>&1 | grep -E "profile" | sed "s/^/[sz] W0 profile ($(date -u +%H:%M)): /" | cut -c1-200 >> w.log
  fi
  sleep 120
done
JOB
  setsid nohup bash -c 'bash /root/side22.sh' > /dev/null 2>&1 < /dev/null &
  echo "[sz] side profiler launched $(date -u)"
fi
if [ ! -e /root/.side23 ]; then touch /root/.side23
  ( cd /root/w; curl -sS -o sp_prof.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/tools/sp_qwen3.py?nocache=$(date +%s)"
    python3 sp_prof.py profile --traces kv_full_p.jsonl --n 16 --max-len 16384 --S 32 --heads 8 --layers 3 --ffn 2048 --rw 0 --C 128 --maxd 0 --no-sp 2>&1 \
      | grep -E "profile" | sed "s/^/[sz] W0 profile no SP: /" | cut -c1-200 >> w.log ) > /dev/null 2>&1 < /dev/null &
  echo "[sz] no-SP profile launched $(date -u)"
fi
# CTL-END
