#!/bin/bash
# Interruptible GRPO rollout worker (the user: a long run - spend the money better). An interruptible box is paused when
# outbid and started again later: its onstart runs ctl.sh again, which runs this script again, so nothing here is guarded by
# a one-time flag - every run (re)starts the worker loop unless it is already running. The worker loop keeps ctl/kvc_job.sh
# going on grpo_pool2 shard 1 of 6 (dpo_col_g/n2/): on its first start it begins two blocks after the last one on the
# hub for the shard (the block the replaced box may still be running is skipped); afterwards kvc_job's own block counter
# (kcg_n2_block, on this box's disk) carries on. Touch /root/w/PAUSE_W to hold it.
# serial 1: first start. serial 2: streams / token budget from /root/w/kvc_rb, kvc_tb (default 12 / 16000).
SER=2; SHARD=1
if [ -f /root/.boxw_serial ] && [ "$(cat /root/.boxw_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxw_serial
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w
cat > /root/status.sh <<'ST'
grep -E "^\[(kc|hub|w)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
tail -n 1 $(ls -t /root/w/sp_*.log 2>/dev/null | head -1) 2>/dev/null | cut -c1-400
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
cat > /root/wloop.sh <<'JOB'
SHARD=1; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
echo "[w] worker loop up, shard $SHARD of 6, $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader) $(date -u)"
python3 -c "import huggingface_hub" 2>/dev/null || python3 -m pip install -q "huggingface_hub[hf_xet]" 2>&1 | tail -1
while true; do
  if [ ! -e PAUSE_W ] && ! pgrep -f "bash /root/kvc_job.sh" > /dev/null; then
    rm -f STOP_KC
    if [ ! -s kcg_n2_block ]; then   # first start: after the replaced box's blocks
      st=$(python3 -c "
import re; from huggingface_hub import HfApi
b = [int(re.search(r'/g12_b(\d+)_v', f).group(1)) for f in HfApi().list_repo_files('baya1116/hypernet-sp-distill') if f.startswith('archive/sp3/dpo_col_g/n2/s$SHARD/')]
print(max(b) + 2 if b else 0)" 2>/dev/null)
      [ -n "$st" ] && echo $st > kcg_n2_block; echo "[w] shard $SHARD starts at block $(cat kcg_n2_block 2>/dev/null)"
    fi
    curl -sS -o /root/kvc_job.sh "$RAW/ctl/kvc_job.sh?nocache=$(date +%s)"
    echo "[w] kvc_job (re)started $(date -u)"
    RB=$(cat kvc_rb 2>/dev/null || echo 12) TB=$(cat kvc_tb 2>/dev/null || echo 16000) POOL=grpo_pool2.jsonl TAG=n2 MODE=grpo SHARD=$SHARD NSHARD=6 bash /root/kvc_job.sh >> /root/w/w.log 2>&1
  fi
  sleep 60
done
JOB
if [ -n "16" ] && [ "$(cat /root/w/kvc_rb 2>/dev/null)" != "16" ]; then   # new streams / budget: the running kvc_job stops after its block
  echo 16 > /root/w/kvc_rb; echo 20000 > /root/w/kvc_tb; touch /root/w/STOP_KC; fi
if [ ! -e /root/.wloop2 ]; then touch /root/.wloop2; pkill -f "bash /root/wloop.sh"; fi   # the loop script changed: restart it (the running kvc_job carries on)
pgrep -f "bash /root/wloop.sh" > /dev/null || setsid nohup bash -c 'bash /root/wloop.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
echo "[w] ctl run $(date -u)"
# CTL-END
