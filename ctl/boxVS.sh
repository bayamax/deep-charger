#!/bin/bash
# Box VS (3060, vLLM image, short): the end-to-end check box VR ran out of memory on - 32 MATH-500 level 4-5 problems,
# Qwen3-1.7B thinking, 8000 tokens, exposure window 256: the weight-free fit fold (128 slots, reservoir 256) vs dropping.
BOXVS_SERIAL=2
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
grep -E "^\[(vs|kvpool|hub|math|mix)\]|Error|Traceback|out of memory" /root/vs/vs.log 2>/dev/null | tail -n 5 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
# serial 2 (the user: unfreezing the model is a must): the frozen-model gen check is stopped. Job 2: LoRA r16 on all
# attention+MLP projections, trained under the bounded cache (W256 + 128 fitted slots, reservoir 256) to reproduce the
# FROZEN full-KV model's next-token distributions (KL), 300 own traces; then held-out KL (MATH 64 / mixed 60) vs the
# frozen model's 0.094 / 0.077, and the LoRA with a full cache (W 8192, nothing evicted) to see normal behaviour is kept.
if [ ! -e /root/.vsjob2 ]; then touch /root/.vsjob2
  pkill -f vsjob.sh; pkill -f "kvpool.py"; sleep 5; echo "[vs] s2: gen check stopped $(date -u)"
  curl -sS -o /root/vs/kvpool.py "$RAW/tools/kvpool.py?nocache=$(date +%s)"
  cat > /root/vsjob2.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/kvpool/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
putd() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_folder(folder_path=sys.argv[1], path_in_repo='archive/kvpool/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2/" || echo "[hub] $2/ FAILED"; }
python3 -m pip install -q peft 2>&1 | grep -v -i warn | tail -1
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o traces.jsonl "$H/archive/kvpool/traces.jsonl"
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o kv_full.jsonl "$H/archive/kvwin/q17_full.jsonl"
echo "[vs] s2 data: $(wc -l < traces.jsonl) traces, $(wc -l < kv_full.jsonl) math"
fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3
LG=/root/vs/kvpool_lora.log; F="--W 256 --m 128 --C 128 --fold fit --reservoir 256"
python3 kvpool.py train --traces traces.jsonl --skip 60 --n 300 --max-len 3072 $F --lora 16 --lr 1e-4 --out lora_W256_m128 --log $LG 2>&1 | grep -E "^\[kvpool\] (trainable|trained)|Error|Traceback|out of memory"
putd lora_W256_m128 lora_W256_m128; put $LG kvpool_lora.log
for e in kv_full.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 kvpool.py eval --traces $f $x --max-len 4096 $F --lora-ckpt lora_W256_m128 --log $LG 2>&1 | grep -E "^\[kvpool\]|Error|Traceback|out of memory" | sed "s/^/[$n] /"; done
python3 kvpool.py eval --traces traces.jsonl --n 60 --max-len 4096 --full --W 256 --m 0 --C 128 --lora-ckpt lora_W256_m128 --log $LG 2>&1 | grep -E "^\[kvpool\]|Error|Traceback" | sed "s/^/[mix fullKV] /"
put $LG kvpool_lora.log; put /root/vs/vs.log run_vs2.log; date -u > DONE_VS2; put DONE_VS2 DONE_VS2; echo "[vs] VSJOB2_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vsjob2.sh >> /root/vs/vs.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[vs] s2: job 2 (LoRA unfreeze under the bounded cache) launched $(date -u)"
fi
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
