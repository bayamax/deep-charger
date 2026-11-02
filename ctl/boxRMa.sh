#!/bin/bash
# Candidate box for the RMT run (the user: start-up check, GPU check, then keep the cheapest that works): GPU empty?,
# bf16 matmul speed, model download time. Prints one [pr] RESULT line and waits; the winner gets the training serial.
SER=1
if [ -f /root/.boxrmx_serial ] && [ "$(cat /root/.boxrmx_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxrmx_serial
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w
cat > /root/status.sh <<'ST'
grep -E "^\[(pr|rm|hub)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 4 | cut -c1-300
tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-200
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
# CTL-END
