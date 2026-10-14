#!/bin/bash
# box RW (2026-10-14, the user: rent a second box, as cheap as possible, and try GRPO with a reward model's 0-1 score as
# the reward). First step: can a dedicated reward model tell Deep Charger's better replies from its worse ones? The
# Skywork-Reward-V2 (Qwen3) family scores the archived GRPO rollouts of mtg7-mtg10 (rm_scan.py). Pull-based as the others.
BOXRW_SERIAL=1
if [ -f /root/.boxrw_serial ] && [ "$(cat /root/.boxrw_serial)" -gt "$BOXRW_SERIAL" ] 2>/dev/null; then echo "BOXRW_STALE $BOXRW_SERIAL"; exit 0; fi
echo $BOXRW_SERIAL > /root/.boxrw_serial
mkdir -p /root/work /root/hfdl; cd /root/work
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
for f in rm_scan.py; do
  for try in 1 2 3; do curl -sS -o /root/work/$f "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/work/$f && break; sleep 5; done
done

# ---- one-time bootstrap ----
if [ ! -f /root/.bootstrapped ]; then
  mkdir /root/.bootstrap_lock 2>/dev/null || { echo "bootstrap already running"; exit 0; }
  echo "=== bootstrap $(date -u) ==="
  pip install -q "transformers>=4.51,<4.57" accelerate "huggingface_hub>=0.34,<1.0" safetensors 2>&1 | tail -1
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /root | tail -1; free -g | head -2
  touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"
fi

# ---- the mirror: the box log to the hub every 10 minutes ----
if ! pgrep -f "rwmirro[r].sh" >/dev/null; then
  cat > /root/rwmirror.sh <<'MR'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
while :; do
  { echo "=== boxRW $(date -u) ==="; echo "--- ctl.log ---"; tail -n 30 /root/ctl.log 2>/dev/null
    for f in /root/rm_scan.log; do [ -s $f ] && { echo "--- $f ---"; grep -vE "Warning|warn" $f | tail -n 40 | cut -c1-600; }; done
    echo "--- gpu ---"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
    echo "--- disk ---"; df -h /root | tail -1
    echo "--- processes ---"; pgrep -fa "python3|kee[p].sh" | cut -c1-200; } > /root/boxlog.txt 2>&1
  hf upload $R /root/boxlog.txt pooler_distill/chatsft/audit/boxlog_RW.txt > /root/mirror_up.txt 2>&1 || echo "MIRROR_UP_FAIL $(date -u)" > /proc/1/fd/1
  sleep 600
done
MR
  setsid nohup bash /root/rwmirror.sh > /dev/null 2>&1 < /dev/null &
  echo "RWMIRROR_LAUNCHED $(date -u)"
fi

# ---- step 1: the reward models on the archived rollouts ----
if [ ! -e /root/.rmscan ]; then touch /root/.rmscan
  cat > /root/rmscankeep.sh <<'RS'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
until [ -f /root/.bootstrapped ]; do sleep 30; done
A=pooler_distill/chatsft/archive_G2/online
for r in mtg7 mtg8 mtg9 mtg10; do
  for try in 1 2 3 4; do hf download $R $A/online_$r/rollouts.jsonl --local-dir /root/hfdl >/dev/null 2>&1 && break; sleep 30; done
done
FS=$(ls /root/hfdl/$A/online_*/rollouts.jsonl); echo "[rm_scan] rollouts: $(cat $FS | wc -l) lines in $(echo $FS | wc -w) files"
python3 -u /root/work/rm_scan.py --rollouts $FS --models Skywork/Skywork-Reward-V2-Qwen3-0.6B Skywork/Skywork-Reward-V2-Qwen3-1.7B Skywork/Skywork-Reward-V2-Qwen3-4B --out /root/rm_scan.json --batch 8
hf upload $R /root/rm_scan.json pooler_distill/chatsft/rm_scan/rm_scan.json >/dev/null 2>&1 && echo "[rm_scan] on the hub: pooler_distill/chatsft/rm_scan/rm_scan.json"
echo "RMSCAN_JOB_DONE $(date -u)"
RS
  setsid nohup bash -c 'bash /root/rmscankeep.sh 2>&1 | tee -a /root/rm_scan.log' > /dev/null 2>&1 < /dev/null &
  echo "RMSCAN_LAUNCHED $(date -u)"
fi
if [ -e /root/.rmscan ] && [ -s /root/rmscankeep.sh ] && ! pgrep -f "rmscankee[p].sh" >/dev/null && ! grep -q "RMSCAN_JOB_DONE" /root/rm_scan.log 2>/dev/null; then
  setsid nohup bash -c 'bash /root/rmscankeep.sh 2>&1 | tee -a /root/rm_scan.log' > /dev/null 2>&1 < /dev/null &
fi
echo "BOXRW_OK serial $BOXRW_SERIAL $(date -u)"
# CTL-END
