#!/bin/bash
# Box TA (3090 24 GB): the single-pass run on the big fresh trace set. Stage A = the pooler alone (model frozen) on
# shards 0-2 (~6.5k traces, each seen once, 4 chunks back-propagated per trace), numbers from the sweep: S32, raw window
# 1024, chunk 128, maxD 384, 1 block / ffn 1024 (~19 MB at 4 bits). Held-out KL (MATH 64 / mixed 60) on the default region
# (from the first eviction, 4096 tokens) and on the common region (1792+). Stage B (full FT + pooler) comes as serial 2.
SER=1
if [ -f /root/.boxta_serial ] && [ "$(cat /root/.boxta_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxta_serial
echo "[ta] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
for f in $(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv $f $f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w; RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i"
curl -sS -o /root/w/sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
cat > /root/status.sh <<'ST'
grep -E "^\[(ta|sp|hub|math|mix)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-200
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.joba ]; then touch /root/.joba
  cat > /root/joba.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
python3 -m pip install -q "huggingface_hub[hf_xet]" bitsandbytes datasets 2>&1 | grep -v -i warn | tail -1
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o traces.jsonl "$H/archive/kvpool/traces.jsonl"
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o kv_full.jsonl "$H/archive/kvwin/q17_full.jsonl"
for i in 0 1 2; do until curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o traces_big_$i.jsonl "$H/archive/sp3/traces_big_$i.jsonl"; do sleep 120; done; done
echo "[ta] data: $(cat traces_big_[012].jsonl | wc -l) big traces (shards 0-2)"
LG=/root/w/sp.log; F="--S 32 --rw 1024 --C 128 --maxd 384 --layers 1 --ffn 1024"
ev() { for e in kv_full.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  for sf in 0 1792; do python3 sp_qwen3.py eval --traces $f $x --max-len 4096 --score-from $sf $F "$@" --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback|out of memory" | sed "s/^/[$n] /"; done; done; }
ev
python3 sp_qwen3.py train --traces traces_big_0.jsonl,traces_big_1.jsonl,traces_big_2.jsonl --max-len 3072 $F --chunks-per-trace 4 --accum 8 --lr 2e-4 --out spA.pt --log $LG > run_trainA.log 2>&1
grep -E "^\[sp\] (train .*: [0-9]+ traces|trained)|Error|Traceback|out of memory" run_trainA.log | tail -3
put spA.pt spA_rw1024_L1.pt; put $LG spA.log
ev --ckpt spA.pt
put $LG spA.log; date -u > DONE_TA_A; put DONE_TA_A DONE_TA_A; echo "[ta] STAGE_A_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/joba.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "JOBA_LAUNCHED $(date -u)"
fi
# CTL-END
