#!/bin/bash
# Collector candidate Cb (the user: parallel DPO data collection on cheaper boxes; start-up check, GPU check, keep the cheapest
# that work). Serial 1: GPU empty?, bf16 matmul speed, model download time - one [pr] RESULT line. COLLECT=1 (a later serial)
# starts ctl/kvc_job.sh with this box's shard.
# serial 2: passed the probe (GPU empty, ~25 TFLOPS bf16) - collect shard 0 of 4.
# serial 3: the first generation ran out of memory at 24 streams / 30000 tokens - restart the collector with the current
# kvc_job.sh (12 streams / 16000 tokens, halved again on an out-of-memory).
# serial 4: no block finished in 3.5 h (the others take 80-90 min) - print what it is doing.
SER=4; COLLECT=1; SHARD=0; NSHARD=4
if [ -f /root/.boxc_serial ] && [ "$(cat /root/.boxc_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxc_serial
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w
cat > /root/status.sh <<'ST'
grep -E "^\[(pr|kc|hub)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 4 | cut -c1-300
tail -n 1 $(ls -t /root/w/sp_*.log 2>/dev/null | head -1) 2>/dev/null | cut -c1-200
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.probe ]; then touch /root/.probe
  cat > /root/probe.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
U0=$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)
echo "[pr] preflight GPU: $U0"
python3 -m pip install -q "huggingface_hub[hf_xet]" transformers "math-verify[antlr4_13_2]" 2>&1 | grep -v -i warn | tail -1
t0=$(date +%s); timeout 1500 python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B')" > dl.log 2>&1; dl=$(( $(date +%s) - t0 ))
tf=$(python3 -c "
import torch, time
x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16); y = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
for _ in range(5): x @ y
torch.cuda.synchronize(); t = time.time()
for _ in range(50): x @ y
torch.cuda.synchronize(); print(round(50 * 2 * 4096**3 / (time.time() - t) / 1e12, 1))" 2>&1 | tail -1)
echo "[pr] RESULT gpu ($U0) | model download ${dl}s | bf16 matmul ${tf} TFLOPS | $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/probe.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ "$COLLECT" = 1 ] && [ ! -e /root/.collect3 ]; then touch /root/.collect3 /root/.collect
  for J in $(pgrep -f "bash /root/kvc_job.sh"); do pkill -P $J; kill $J; done; sleep 3; pkill -f "sp_qwen3.py"; sleep 3
  curl -sS -o /root/kvc_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvc_job.sh?nocache=$(date +%s)"
  setsid nohup bash -c "SHARD=$SHARD NSHARD=$NSHARD bash /root/kvc_job.sh >> /root/w/w.log 2>&1" > /dev/null 2>&1 < /dev/null &
  echo "[kc] collector launched: shard $SHARD of $NSHARD $(date -u)"
fi
if [ ! -e /root/.dbg4 ]; then touch /root/.dbg4
  cd /root/w; echo "[kc] dbg $(date -u) block $(cat kc_block 2>/dev/null)"; ls -la --time-style=+%H:%M *_1.jsonl p_1.jsonl todo_1.jsonl 2>/dev/null | awk '{print "[kc] dbg file", $6, $7, $5}'
  for f in raw_1 mem_1 todo_1 raw2_1 mem2_1; do [ -e $f.jsonl ] && echo "[kc] dbg $f lines $(wc -l < $f.jsonl)"; done
  ps aux | grep "sp_qwen3.py" | grep -v grep | cut -c1-230 | sed 's/^/[kc] dbg ps: /'; tail -n 3 run_kc.log | cut -c1-200 | sed 's/^/[kc] dbg log: /'
fi
# CTL-END
