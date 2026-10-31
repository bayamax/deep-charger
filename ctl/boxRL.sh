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
SER=12
if [ -f /root/.boxrl_serial ] && [ "$(cat /root/.boxrl_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxrl_serial
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w
cat > /root/status.sh <<'ST'
grep -E "^\[(ev|hub)\]|Error|Traceback" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-200
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.job12 ]; then touch /root/.job12
  for J in $(pgrep -f "^bash /root/job.sh") $(pgrep -f "^bash /root/live.sh"); do pkill -P $J; kill $J; done; sleep 3
  pkill -f "sp_qwen3.py"; pkill -f "pip install"; pkill -f gen_long_vllm; sleep 5
  pkill -9 -f "/root/vv/bin/python"; pkill -9 -f EngineCore; pkill -9 -f "multiprocessing.spawn"; pkill -9 -f "multiprocessing.resource_tracker"
  sleep 15; rm -f /root/w/ev_raw_s1.jsonl /root/w/ev_u45_s1.jsonl /root/w/DONE_EV1
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${U:-0}" -gt 2000 ]; then echo "[ev] GPU still holds $U MiB after the kills: $(ps -eo pid,rss,cmd --sort=-rss | head -4 | tr '\n' ';' | cut -c1-300)"; rm -f /root/.job12; exit 0; fi
  echo "[ev] GRPO stopped $(date -u); GPU now $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  cat > /root/live.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
while [ ! -e DONE_EV1 ]; do
  { grep -E "^\[(ev|hub)\]|Error|Traceback" w.log | tail -n 30; tail -n 2 sp.log 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; date -u; } > rl_live.log 2>&1
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='rl_live.log', path_in_repo='archive/sp3/rl_live.log', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1
  sleep 600
done
JOB
  cat > /root/job.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
for f in tools/sp_qwen3.py tools/sp_mathcheck.py tools/gen_long_vllm.py ctl/rl_eval120.jsonl; do curl -sS -o $(basename $f) "$RAW/$f?nocache=$(date +%s)"; done
echo "[ev] $(wc -l < rl_eval120.jsonl) problems; ckpts: spG $(stat -c %s spG.pt 2>/dev/null) grpo3_latest $(stat -c %s grpo3_latest.pt 2>/dev/null)"
python3 -c "
import json; rs = [l for l in open('rl_eval120.jsonl')]
open('rl_eval60.jsonl', 'w').writelines(rs[i] for i in range(len(rs)) if i % 24 < 12)"; echo "[ev] using $(wc -l < rl_eval60.jsonl) problems, 1 seed"
/root/vv/bin/python -c "import vllm" 2>/dev/null || { [ -x /root/vv/bin/python ] || python3 -m venv /root/vv; /root/vv/bin/pip install -q vllm > vllm_install.log 2>&1; }
/root/vv/bin/python -c "import vllm; print('[ev] vllm', vllm.__version__)" 2>&1 | tail -1
LG=/root/w/sp.log; base="--S 32 --heads 8 --layers 3 --ffn 2048 --max-len 12288 --roll-cb"
score() { python3 sp_mathcheck.py "$1" | sed "s/^/[ev] $2: /"; python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open(sys.argv[1])]; by = {}
for r in rs:
    s = r['src'] if r['src'].startswith('math') else r['src'].split('-')[0]; v = by.setdefault(s, [0, 0, 0])
    v[0] += correct(r); v[1] += '</think>' in r['text']; v[2] += 1
print('[ev] ' + sys.argv[2] + ' by type: ' + ' | '.join(f'{k} {v[0]}/{v[2]} closed {v[1]}' for k, v in sorted(by.items())))" "$1" "$2"; }
run() {   # run NAME ARGS...: one SP-loop rollout of the 60 problems, scored and uploaded
  n=$1; shift; t0=$(date +%s)
  python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $base --roll-batch 30 "$@" --seed 1 --roll-out ev_${n}_s1.jsonl --log $LG > run_ev_$n.log 2>&1
  echo "[ev] $n done in $(( $(date +%s) - t0 )) s"; grep -E "Error|Traceback" run_ev_$n.log | tail -2 | sed 's/^/[ev] /'
  score ev_${n}_s1.jsonl "$n"; bash /root/put_rl.sh ev_${n}_s1.jsonl; }
full() { t0=$(date +%s); VLLM_UTIL=0.45 /root/vv/bin/python gen_long_vllm.py rl_eval60.jsonl ev_full_s1.jsonl 12288 60 1 > run_vllm.log 2>&1
  echo "[ev] full KV done in $(( $(date +%s) - t0 )) s"; grep -E "Error|Traceback" run_vllm.log | tail -2 | sed 's/^/[ev] /'
  score ev_full_s1.jsonl "full KV"; bash /root/put_rl.sh ev_full_s1.jsonl; }
run raw --no-sp --rw 1056 --C 128 --maxd 0 --roll-batch 48 --grpo-tok-budget 60000
run u45 --rw 1024 --C 128 --maxd 384 --ckpt grpo3_latest.pt --roll-batch 48 --grpo-tok-budget 60000
put $LG spP_ev.log; date -u > DONE_EV1; put DONE_EV1 DONE_EV1; echo "[ev] DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/live.sh' > /dev/null 2>&1 < /dev/null &
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[ev] job launched $(date -u)"
fi
# CTL-END
