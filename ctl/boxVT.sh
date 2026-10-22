#!/bin/bash
# Box VT (3060, vLLM image, short): the decisive end-to-end check of the unfrozen model - 32 MATH-500 level 4-5
# problems, Qwen3-1.7B + the LoRA trained under the bounded cache (W256 + 128 fitted slots, reservoir 256), thinking,
# 8000 tokens. Compare with the same 32 under full KV (vLLM run) and a 768 naive window (both from archive/kvwin).
BOXVT_SERIAL=1
if [ -f /root/.boxvt_serial ] && [ "$(cat /root/.boxvt_serial)" -gt "$BOXVT_SERIAL" ] 2>/dev/null; then echo "BOXVT_STALE $BOXVT_SERIAL"; exit 0; fi
echo $BOXVT_SERIAL > /root/.boxvt_serial
echo "[vt] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
for f in $(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv $f $f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/vt; RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i"
curl -sS -o /root/vt/kvpool.py "$RAW/tools/kvpool.py?nocache=$(date +%s)"
cat > /root/status.sh <<'ST'
grep -E "^\[(vt|kvpool|hub)\]|Error|Traceback|out of memory" /root/vt/vt.log 2>/dev/null | tail -n 4 | cut -c1-300
tail -n 1 /root/vt/kvpool.log 2>/dev/null | cut -c1-200
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.vtjob ]; then touch /root/.vtjob
  cat > /root/vtjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/kvpool/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
python3 -m pip install -q "huggingface_hub[hf_xet]" peft datasets "math-verify[antlr4_13_2]" 2>&1 | grep -v -i warn | tail -1
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$R', allow_patterns=['archive/kvpool/lora_W256_m128/*'], local_dir='/root/vt/hub')" > /dev/null 2>&1
ls /root/vt/hub/archive/kvpool/lora_W256_m128/ | tr '\n' ' ' | sed 's/^/[vt] lora: /'; echo
LG=/root/vt/kvpool.log
python3 kvpool.py gen --traces none --W 256 --m 128 --C 128 --fold fit --reservoir 256 --lora-ckpt /root/vt/hub/archive/kvpool/lora_W256_m128 \
  --gen-n 32 --gen-tokens 8000 --batch 16 --out gen_lora_W256_m128.jsonl --log $LG > run_gen.log 2>&1
grep -E "^\[kvpool\] gen W|Error|Traceback|out of memory" run_gen.log | tail -3
put gen_lora_W256_m128.jsonl gen_lora_W256_m128.jsonl; put $LG kvpool_vt.log
put /root/vt/vt.log run_vt.log; date -u > DONE_VT; put DONE_VT DONE_VT; echo "[vt] VTJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vtjob.sh >> /root/vt/vt.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VTJOB_LAUNCHED $(date -u)"
fi
# CTL-END
