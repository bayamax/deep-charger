#!/bin/bash
# Box SN (3060, pytorch image): does the window-0 "SP-next" training help? From stage A's pooler (spA, rw1024, 1 block),
# model frozen, 600 unseen traces (shard 3), every 4th position... 512 positions per trace in batches of 64:
# [prompt | SP_t] -> token t. Scored with the deployment loop (MATH 64 / mixed 60, default + from1792) vs spA.
SER=1
if [ -f /root/.boxsn_serial ] && [ "$(cat /root/.boxsn_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxsn_serial
echo "[sn] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w; RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i"
curl -sS -o /root/w/sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
cat > /root/status.sh <<'ST'
grep -E "^\[(sn|sp|hub|math|mix)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
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
python3 -m pip install -q "huggingface_hub[hf_xet]" transformers datasets 2>&1 | grep -v -i warn | tail -1
for t in 1 2 3; do x=""; [ $t -gt 1 ] && x="HF_HUB_DISABLE_XET=1"
  env $x timeout 600 python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-1.7B'))" > dl.log 2>&1 && break; echo "[sn] model download attempt $t failed"; done
echo "[sn] model: $(tail -1 dl.log | cut -c1-100)"
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o traces.jsonl "$H/archive/kvpool/traces.jsonl"
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o kv_full_p.jsonl "$H/archive/kvwin/q17_full_p.jsonl"
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o traces_big_3.jsonl "$H/archive/sp3/traces_big_3.jsonl"
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o spA.pt "$H/archive/sp3/spA_rw1024_L1.pt"
echo "[sn] data: $(wc -l < traces_big_3.jsonl) traces, spA $(du -h spA.pt | cut -f1)"
LG=/root/w/sp.log; F="--S 32 --rw 1024 --C 128 --maxd 384 --layers 1 --ffn 1024"
ev() { for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  for sf in 0 1792; do python3 sp_qwen3.py eval --traces $f $x --max-len 4096 --score-from $sf $F "$@" --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback|out of memory" | sed "s/^/[$n] /"; done; done; }
python3 sp_qwen3.py train --traces traces_big_3.jsonl --n 600 --max-len 3072 $F --ckpt spA.pt --spnext --sn-pos 512 --sn-batch 64 --accum 8 --lr 1e-4 --out spSN.pt --log $LG > run_train.log 2>&1
grep -E "^\[sp\] (train .*: [0-9]+ traces|trained)|Error|Traceback|out of memory" run_train.log | tail -3
put spSN.pt spSN.pt; put $LG spSN.log
ev --ckpt spSN.pt
put $LG spSN.log; date -u > DONE_SN; put DONE_SN DONE_SN; echo "[sn] JOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "JOB_LAUNCHED $(date -u)"
fi
# CTL-END
