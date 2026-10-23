#!/bin/bash
# Box VY (3060): deep-charger's SP-evict hypernet ported to Qwen3-1.7B (tools/sp_qwen3.py), stage 1 = the pooler alone
# with the model frozen, deep-charger 1.0.2's numbers (S32, raw window 768, chunk 128, maxD 384, 3 blocks, ffn 2048),
# KL to the full-context model on its own traces (600 train, 4 per update). Held-out KL (MATH 64 / mixed 60) before/after.
BOXVY_SERIAL=1
if [ -f /root/.boxvy_serial ] && [ "$(cat /root/.boxvy_serial)" -gt "$BOXVY_SERIAL" ] 2>/dev/null; then echo "BOXVY_STALE $BOXVY_SERIAL"; exit 0; fi
echo $BOXVY_SERIAL > /root/.boxvy_serial
echo "[vy] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
for f in $(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv $f $f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/vy; RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i"
curl -sS -o /root/vy/sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
cat > /root/status.sh <<'ST'
grep -E "^\[(vy|sp|hub|math|mix)\]|Error|Traceback|out of memory" /root/vy/vy.log 2>/dev/null | tail -n 3 | cut -c1-300
tail -n 1 /root/vy/sp.log 2>/dev/null | cut -c1-200
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.vyjob ]; then touch /root/.vyjob
  cat > /root/vyjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
python3 -m pip install -q "huggingface_hub[hf_xet]" datasets 2>&1 | grep -v -i warn | tail -1
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o traces.jsonl "$H/archive/kvpool/traces.jsonl"
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o kv_full.jsonl "$H/archive/kvwin/q17_full.jsonl"
echo "[vy] data: $(wc -l < traces.jsonl) traces, $(wc -l < kv_full.jsonl) math"
LG=/root/vy/sp.log; F="--S 32 --rw 768 --C 128 --maxd 384 --layers 3 --ffn 2048"
ev() { for e in kv_full.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces $f $x --max-len 3072 $F "$@" --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback|out of memory" | sed "s/^/[$n] /"; done; }
ev
python3 sp_qwen3.py train --traces traces.jsonl --skip 60 --n 600 --max-len 3072 $F --accum 4 --lr 2e-4 --out sp1.pt --log $LG > run_train.log 2>&1
grep -E "^\[sp\] (train .*: [0-9]+ traces|trained)|Error|Traceback|out of memory" run_train.log | tail -3
put sp1.pt sp1_S32_rw768.pt; put $LG sp1.log
ev --ckpt sp1.pt
put $LG sp1.log; put /root/vy/vy.log run_vy.log; date -u > DONE_VY; put DONE_VY DONE_VY; echo "[vy] VYJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vyjob.sh >> /root/vy/vy.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VYJOB_LAUNCHED $(date -u)"
fi
# CTL-END
