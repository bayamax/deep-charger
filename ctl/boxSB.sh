#!/bin/bash
# Box SB (3060): SP-pooler numbers sweep, stage 1 (model frozen); pooler shrunk to 2 blocks / ffn 1024 so pooler <= KV at rw 256; then re-scores box VY's 1.0.2-numbers pooler (untrained and trained) on the common region: --S 32 --rw 256 --C 128 --layers 2 --ffn 1024 --maxd 384
SER=1
if [ -f /root/.boxsb_serial ] && [ "$(cat /root/.boxsb_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxsb_serial
echo "[sb] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
for f in $(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv $f $f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w; RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i"
curl -sS -o /root/w/sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
cat > /root/status.sh <<'ST'
grep -E "^\[(sb|sp|hub|math|mix)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-200
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.job ]; then touch /root/.job
  cat > /root/job.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
python3 -m pip install -q "huggingface_hub[hf_xet]" datasets 2>&1 | grep -v -i warn | tail -1
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o traces.jsonl "$H/archive/kvpool/traces.jsonl"
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o kv_full.jsonl "$H/archive/kvwin/q17_full.jsonl"
LG=/root/w/sp.log; F="--S 32 --rw 256 --C 128 --layers 2 --ffn 1024 --maxd 384"
ev() { for e in kv_full.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 --score-from 1792 $F "$@" --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback|out of memory" | sed "s/^/[$n] /"; done; }
python3 sp_qwen3.py train --traces traces.jsonl --skip 60 --n 600 --max-len 3072 $F --accum 4 --lr 2e-4 --out sp.pt --log $LG > run_train.log 2>&1
grep -E "^\[sp\] (train .*: [0-9]+ traces|trained)|Error|Traceback|out of memory" run_train.log | tail -3
put sp.pt sp_SB.pt; put $LG sp_SB.log
ev --ckpt sp.pt
# the 1.0.2-numbers run (box VY, sp1) re-scored on the same common region
until curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o sp1.pt "$H/archive/sp3/sp1_S32_rw768.pt"; do sleep 120; done
F="--S 32 --rw 768 --C 128 --layers 3 --ffn 2048 --maxd 384"; ev; ev --ckpt sp1.pt
put $LG sp_SB.log; put /root/w/w.log run_SB.log; date -u > DONE_SB; put DONE_SB DONE_SB; echo "[sb] JOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "JOB_LAUNCHED $(date -u)"
fi
# CTL-END
