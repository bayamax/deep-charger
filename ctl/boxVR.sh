#!/bin/bash
# Box VR (3060, vLLM image): candidate bases for the app, as they come - Qwen3-1.7B and Qwen3-4B, thinking on, no search -
# answer the use-case bench's app kinds (197 real prompts); the replies go to archive/uc/ for the judge against 4o-mini.
BOXVR_SERIAL=1
if [ -f /root/.boxvr_serial ] && [ "$(cat /root/.boxvr_serial)" -gt "$BOXVR_SERIAL" ] 2>/dev/null; then echo "BOXVR_STALE $BOXVR_SERIAL"; exit 0; fi
echo $BOXVR_SERIAL > /root/.boxvr_serial
[ -e /root/.vr_pre ] || { touch /root/.vr_pre; echo "[vr] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"; }
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
for f in $(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv $f $f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/vr; RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i"
curl -sS -o /root/vr/uc_vllm.py "$RAW/tools/uc/uc_vllm.py?nocache=$(date +%s)"
[ -f /root/.bootstrapped ] || { python3 -m pip install -q "huggingface_hub[hf_xet]" 2>&1 | tail -1; touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"; }
cat > /root/status.sh <<'ST'
grep -E "^\[(vr|uc_vllm|hub)\]|Error|Traceback|out of memory" /root/vr/vr.log 2>/dev/null | tail -n 5 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.vrjob ]; then touch /root/.vrjob
  cat > /root/vrjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vr
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/uc/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
curl -sSfL -o uc_app.jsonl "$H/archive/uc/uc_app.jsonl"
for spec in q17:Qwen/Qwen3-1.7B q4:Qwen/Qwen3-4B; do t=${spec%%:*}; m=${spec#*:}
  fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3
  python3 uc_vllm.py --model $m --inp uc_app.jsonl --out ans_$t.jsonl --max-tokens 4096 --max-len 8192 2>&1 | grep -E "^\[uc_vllm\]|Error|Traceback"
  put ans_$t.jsonl ans_$t.jsonl
done
put /root/vr/vr.log run_vr.log; date -u > DONE_VR; put DONE_VR DONE_VR; echo "[vr] VRJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vrjob.sh >> /root/vr/vr.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VRJOB_LAUNCHED $(date -u)"
fi
# CTL-END
