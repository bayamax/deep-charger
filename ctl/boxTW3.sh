#!/bin/bash
# Box TW3 (3090 24 GB, pytorch image): finish W2's window curriculum (it stopped right at w = 1024, likely out of memory - the
# trainer now scores window 0 + 255 sampled windows when w > 256): from W2 (the 400-trace checkpoint: deployment MATH 0.091 /
# mixed 0.083), w = 512 then 1024, each until it stops falling (100-trace means < 2%, max 500), middle lr, on W's unseen
# traces (pool from 1776). Uploads retry (box TD's all failed and were lost); held-out profile + deployment loop at the end.
SER=1
if [ -f /root/.boxtw3_serial ] && [ "$(cat /root/.boxtw3_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxtw3_serial
echo "[tw3] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w; RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i"
curl -sS -o /root/w/sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
cat > /root/status.sh <<'ST'
grep -E "^\[(tw3|sp|hub|math|mix)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-220
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.job ]; then touch /root/.job
  cat > /root/live.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
while [ ! -e DONE_TW3 ]; do
  { grep -E "^\[(tw3|hub|math|mix)\]" w.log | tail -n 20; tail -n 12 sp.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; df -h / | tail -1; date -u; } > tw3_live.log 2>&1
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='tw3_live.log', path_in_repo='archive/sp3/tw3_live.log', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1
  sleep 600
done
JOB
  cat > /root/job.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; echo "[tw3] put $2 attempt $t: $(tail -1 put.err | cut -c1-160)"; sleep 90; done; echo "[hub] $2 FAILED"; }
get() { for t in 1 2 3; do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o "$2" "$H/$1" && return 0; echo "[tw3] get $1 attempt $t failed"; sleep 10; done; return 1; }
python3 -m pip install -q "huggingface_hub[hf_xet]" transformers bitsandbytes datasets 2>&1 | grep -v -i warn | tail -1
for t in 1 2 3; do
  x=""; [ $t -gt 1 ] && x="HF_HUB_DISABLE_XET=1"
  env $x timeout 600 python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-1.7B'))" > dl.log 2>&1 && break
  echo "[tw3] model download attempt $t failed"
done
echo "[tw3] model: $(tail -1 dl.log | cut -c1-120)"
get archive/kvpool/traces.jsonl traces.jsonl; get archive/kvwin/q17_full_p.jsonl kv_full_p.jsonl
for i in 0 1 2 3 4 5; do get archive/sp3/traces_big_$i.jsonl traces_big_$i.jsonl; done
get archive/sp3/spW2_wincurr1024_fft.pt spW2.pt; echo "[tw3] start ckpt $(stat -c %s spW2.pt) bytes"
cat traces_big_[012].jsonl | tail -n +3201 > restA.jsonl; cat restA.jsonl traces_big_3.jsonl traces_big_4.jsonl traces_big_5.jsonl > trainB.jsonl
{ cat traces_big_[012].jsonl | head -n 3200; head -n 5425 trainB.jsonl; } > trainE.jsonl
{ sed -n '701,2000p' trainE.jsonl; tail -n +2801 trainE.jsonl; } > trainW.jsonl
echo "[tw3] data: trainW $(wc -l < trainW.jsonl) (expect 7125)"
LG=/root/w/sp.log; F="--S 32 --rw 1024 --C 128 --maxd 384 --layers 1 --ffn 1024"
score() { python3 sp_qwen3.py profile --traces kv_full_p.jsonl --n 16 --max-len 4096 $F --ckpt $1 --log $LG 2>&1 | grep -E "^\[sp\] profile|Error|Traceback" | tail -7 | sed "s/^/[math] /"
  for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
    python3 sp_qwen3.py eval --traces $f $x --max-len 4096 $F --ckpt $1 --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback|out of memory" | sed "s/^/[$n] /"; done; }
echo "[tw3] training launched $(date -u)"
python3 sp_qwen3.py train --traces trainW.jsonl --skip 1775 --n 1500 --max-len 3072 $F --ckpt spW2.pt --pe-rev --pe-norm --ft-layers 28 --opt8bit --ft-lr 5e-6 --lr 2e-4 --spnext --sn-win 512,1024 --sn-win-plateau 0.02 --sn-win-block 4 --sn-win-max 500 --sn-pos 256 --sn-batch 8 --sn-tokens 3000 --accum 1 --out spW3.pt --log $LG > run_trainW3.log 2>&1
grep -E "^\[sp\] (trained|curriculum)|Error|Traceback|out of memory" run_trainW3.log | tail -4
put spW3.pt spW3_wincurr1024_fft.pt
score spW3.pt
put $LG spW3.log; date -u > DONE_TW3; put DONE_TW3 DONE_TW3; echo "[tw3] DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/live.sh' > /dev/null 2>&1 < /dev/null &
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tw3] job launched $(date -u)"
fi
# CTL-END
