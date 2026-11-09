#!/bin/bash
# A fresh box for the long-range eval (ctl/mt_job.sh), because the stopped 3090 (58853599) could not restart: its host's GPU
# was taken ("resources_unavailable"). Only this job runs here.
SER=1
if [ -f /root/.boxmt_serial ] && [ "$(cat /root/.boxmt_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxmt_serial
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w
cat > /root/status.sh <<'ST'
grep -E "^\[mt\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 8 | cut -c1-600
tail -n 1 /root/w/sp_mt.log 2>/dev/null | cut -c1-250
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.mt1 ]; then touch /root/.mt1
  cat > /root/mt1.sh <<'JOB'
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
cd /root/w; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
echo "[mt] box up: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader) $(date -u)"
python3 -m pip install -q "huggingface_hub[hf_xet]" transformers "math-verify[antlr4_13_2]" 2>&1 | grep -v -i warn | tail -1
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
timeout 1500 python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B')" > dl.log 2>&1
echo "[mt] setup done: transformers $(python3 -c 'import transformers; print(transformers.__version__)' 2>&1 | tail -1) $(date -u)"
for _t in $(seq 60); do curl -sS -o /root/mt_job.sh "$RAW/ctl/mt_job.sh?nocache=$(date +%s)"; grep -q "CONDS" /root/mt_job.sh && break; sleep 120; done
CONDS="k512:1024:512 w1536:1536:0 full:16384:0 w1024:1024:0 k1024:1024:1024 w2048:2048:0" exec bash /root/mt_job.sh
JOB
  setsid nohup bash -c 'bash /root/mt1.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
# CTL-END
