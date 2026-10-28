#!/bin/bash
# Box RL (a 24 GB card): GRPO from G under the deployment loop ([prompt | SP | raw window 1024], C 128, maxd 384, cap 12288,
# Qwen3 thinking sampling). Problems: ctl/grpo_prompts.jsonl (tools/build_grpo_prompts.py: MATH train L3-5 1200, competition
# 900, MMLU-Pro all 14 categories 896). 16 problems x 8 samples per update, reward = correct final answer. The 96-problem
# check (2 seeds) on the starting weights and every 8 updates; the best total is kept; stop after 3 checks without a new best.
# Results to archive/sp3/rl/ (grpo_best.pt, eval/train jsonl), live log archive/sp3/rl_live.log, marker DONE_RL1.
SER=1
if [ -f /root/.boxrl_serial ] && [ "$(cat /root/.boxrl_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxrl_serial
echo "[rl] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w
cat > /root/status.sh <<'ST'
grep -E "^\[(rl|hub|grpo)\]|Error|Traceback" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-200
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.job ]; then touch /root/.job
  cat > /root/live.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
while [ ! -e DONE_RL1 ]; do
  { grep -E "^\[(rl|hub)\]|Error|Traceback" w.log | tail -n 12; grep -E "^\[grpo\]" sp.log 2>/dev/null | tail -n 30; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; date -u; } > rl_live.log 2>&1
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='rl_live.log', path_in_repo='archive/sp3/rl_live.log', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1
  sleep 600
done
JOB
  cat > /root/put_rl.sh <<'JOB'
# put_rl.sh FILE...: upload to archive/sp3/rl/ (called in the background by the GRPO run after each check)
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
for f in "$@"; do
  for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/rl/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$f" "$(basename $f)" > put_rl.err 2>&1 && { echo "[hub] rl/$(basename $f)" >> w.log; break; }; sleep 60; done
done
JOB
  cat > /root/job.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
get() { for t in 1 2 3; do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o "$2" "$H/$1" && return 0; echo "[rl] get $1 attempt $t failed"; sleep 10; done; return 1; }
for f in tools/sp_qwen3.py tools/sp_mathcheck.py ctl/math96.jsonl ctl/grpo_prompts.jsonl; do curl -sS -o $(basename $f) "$RAW/$f?nocache=$(date +%s)"; done
python3 -c "import torch; x = torch.randn(4096, 4096, device='cuda'); print('[rl] GPU ok', torch.cuda.get_device_name(), float((x @ x).sum()) != 0)" 2>&1 | tail -1
python3 -c "import torch; torch.zeros(1).cuda()" 2>/dev/null || { echo "[rl] GPU FAIL - stop"; exit 1; }
python3 -m pip install -q "huggingface_hub[hf_xet]" transformers bitsandbytes "math-verify[antlr4_13_2]" 2>&1 | grep -v -i warn | tail -1
( while [ ! -e dl.done ]; do sleep 120; echo "[rl] downloads $(du -sh /root/.cache/huggingface 2>/dev/null | cut -f1) model, $(du -ch /root/w/*.pt 2>/dev/null | tail -1 | cut -f1) ckpt $(date -u +%H:%M)"; done ) &
t0=$(date +%s); python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-1.7B'))" > dl.log 2>&1
get archive/sp3/spG_longtraces_fft1024.pt spG.pt
touch dl.done; echo "[rl] downloads done in $(( $(date +%s) - t0 )) s: spG $(stat -c %s spG.pt) bytes, $(wc -l < grpo_prompts.jsonl) problems $(date -u)"
LG=/root/w/sp.log
python3 sp_qwen3.py grpo --traces grpo_prompts.jsonl --max-len 12288 --S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 384 \
  --ckpt spG.pt --ft-layers 28 --opt8bit --ft-lr 1e-6 --lr 1e-5 --group 8 --grpo-q 16 --grpo-slots 48 --grpo-updates 200 --grpo-hours 30 \
  --grpo-eval math96.jsonl --grpo-eval-every 8 --grpo-eval-seeds 2 --grpo-eval0 --grpo-patience 3 --grpo-hook "bash /root/put_rl.sh" \
  --seed 1 --out grpo_best.pt --log $LG > run_grpo.log 2>&1
echo "[rl] grpo exit $?: $(grep -E '^\[grpo\] (done|stop)' run_grpo.log | tail -2 | tr '\n' ' ')"; grep -E "Error|Traceback" run_grpo.log | tail -3 | sed 's/^/[rl] /'
sleep 120; bash /root/put_rl.sh grpo_best_train.jsonl; put $LG spP_rl.log
date -u > DONE_RL1; put DONE_RL1 DONE_RL1; echo "[rl] DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/live.sh' > /dev/null 2>&1 < /dev/null &
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[rl] job launched $(date -u)"
fi
# CTL-END
