#!/bin/bash
# Box VS (3060, vLLM image, short): the end-to-end check box VR ran out of memory on - 32 MATH-500 level 4-5 problems,
# Qwen3-1.7B thinking, 8000 tokens, exposure window 256: the weight-free fit fold (128 slots, reservoir 256) vs dropping.
BOXVS_SERIAL=1
if [ -f /root/.boxvs_serial ] && [ "$(cat /root/.boxvs_serial)" -gt "$BOXVS_SERIAL" ] 2>/dev/null; then echo "BOXVS_STALE $BOXVS_SERIAL"; exit 0; fi
echo $BOXVS_SERIAL > /root/.boxvs_serial
echo "[vs] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
for f in $(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv $f $f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/vs; RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i"
curl -sS -o /root/vs/kvpool.py "$RAW/tools/kvpool.py?nocache=$(date +%s)"
cat > /root/status.sh <<'ST'
grep -E "^\[(vs|kvpool|hub)\]|Error|Traceback|out of memory" /root/vs/vs.log 2>/dev/null | tail -n 5 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.vsjob ]; then touch /root/.vsjob
  cat > /root/vsjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/kvpool/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
python3 -m pip install -q "huggingface_hub[hf_xet]" datasets "math-verify[antlr4_13_2]" 2>&1 | grep -v -i warn | tail -1
echo "[vs] setup done $(date -u)"
LG=/root/vs/kvpool.log
for spec in "256 128" "256 0"; do set -- $spec; x=""; [ $2 != 0 ] && x="--fold fit --reservoir 256"
  for b in 32 16; do
    python3 kvpool.py gen --traces none --W $1 --m $2 --C 128 $x --gen-n 32 --gen-tokens 8000 --batch $b --out gen_W$1_m$2.jsonl --log $LG > run_W$1_m$2.log 2>&1
    grep -E "^\[kvpool\] gen|Error|Traceback|out of memory" run_W$1_m$2.log | tail -3
    grep -q "out of memory" run_W$1_m$2.log || break
    echo "[vs] OOM at batch $b, retry smaller"
  done
  put gen_W$1_m$2.jsonl gen_W$1_m$2.jsonl; put $LG kvpool_vs.log
done
put /root/vs/vs.log run_vs.log; date -u > DONE_VS; put DONE_VS DONE_VS; echo "[vs] VSJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vsjob.sh >> /root/vs/vs.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VSJOB_LAUNCHED $(date -u)"
fi
# CTL-END
