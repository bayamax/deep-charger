#!/bin/bash
# Box VW (3090 24 GB): full fine-tune of ALL decoder layers (embeddings / final norm / lm head frozen), fp32 master
# weights + 8-bit AdamW states, same setting as the LoRA and input-side runs (W256 + 128 fitted slots, reservoir 256, 300
# own traces, KL to the frozen full-KV model). Held-out KL (MATH 64 / mixed 60) plus the full-KV check.
BOXVW_SERIAL=1
if [ -f /root/.boxvw_serial ] && [ "$(cat /root/.boxvw_serial)" -gt "$BOXVW_SERIAL" ] 2>/dev/null; then echo "BOXVU_STALE $BOXVW_SERIAL"; exit 0; fi
echo $BOXVW_SERIAL > /root/.boxvw_serial
echo "[vw] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
for f in $(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv $f $f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/vw; RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i"
curl -sS -o /root/vw/kvpool.py "$RAW/tools/kvpool.py?nocache=$(date +%s)"
cat > /root/status.sh <<'ST'
grep -E "^\[(vw|kvpool|hub|math|mix)\]|Error|Traceback|out of memory" /root/vw/vw.log 2>/dev/null | tail -n 4 | cut -c1-300
tail -n 1 /root/vw/kvpool_fft.log 2>/dev/null | cut -c1-200
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.vwjob ]; then touch /root/.vwjob
  cat > /root/vwjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vw
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/kvpool/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
python3 -m pip install -q "huggingface_hub[hf_xet]" bitsandbytes datasets "math-verify[antlr4_13_2]" 2>&1 | grep -v -i warn | tail -1
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o traces.jsonl "$H/archive/kvpool/traces.jsonl"
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o kv_full.jsonl "$H/archive/kvwin/q17_full.jsonl"
echo "[vw] data: $(wc -l < traces.jsonl) traces, $(wc -l < kv_full.jsonl) math"
LG=/root/vw/kvpool_fft.log; F="--W 256 --m 128 --C 128 --fold fit --reservoir 256"
NL=28
python3 kvpool.py train --traces traces.jsonl --skip 60 --n 300 --max-len 3072 $F --ft-layers 28 --opt8bit --accum 8 --epochs 2 --lr 1e-5 --out ftin$NL.safetensors --log $LG > run_train$NL.log 2>&1
grep -E "^\[kvpool\] (trainable|trained)|Error|Traceback|out of memory" run_train$NL.log | tail -3
put ftin$NL.safetensors fft_W256_m128.safetensors; put $LG kvpool_fft.log
for e in kv_full.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 kvpool.py eval --traces $f $x --max-len 4096 $F --ft-ckpt ftin$NL.safetensors --log $LG 2>&1 | grep -E "^\[kvpool\]|Error|Traceback|out of memory" | sed "s/^/[$n] /"; done
python3 kvpool.py eval --traces traces.jsonl --n 60 --max-len 4096 --full --W 256 --m 0 --C 128 --ft-ckpt ftin$NL.safetensors --log $LG 2>&1 | grep -E "^\[kvpool\]|Error|Traceback" | sed "s/^/[mix fullKV] /"
put $LG kvpool_fft.log; put /root/vw/vw.log run_vw.log; date -u > DONE_VW; put DONE_VW DONE_VW; echo "[vw] VWJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vwjob.sh >> /root/vw/vw.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VWJOB_LAUNCHED $(date -u)"
fi
# CTL-END
