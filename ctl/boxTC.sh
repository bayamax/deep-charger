#!/bin/bash
# Box TC (3090 24 GB, pytorch image): stage C' = window-0 SP-next training (--spnext) + all-layer FFT, with the pooler's
# positions counted from the NEWEST buffer token (--pe-rev): stage C (forward positions) sat at KL ~5.7 for 625 traces.
# Starts from stage C's weights (spC_spnext_fft_t400.pt); data = the stage-C traces it has not seen (stage B's 8000 unseen
# traces from 4801, minus the 625 stage C used) - every trace once. The original model stays the teacher. At the end:
# held-out KL by window length (profile) and the deployment loop, both against stages A / B.
SER=1
if [ -f /root/.boxtc_serial ] && [ "$(cat /root/.boxtc_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxtc_serial
echo "[tc] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w; RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i"
curl -sS -o /root/w/sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
cat > /root/status.sh <<'ST'
grep -E "^\[(tc|sp|hub|math|mix)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
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
get() { for t in 1 2 3; do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o "$2" "$H/$1" && return 0; echo "[tc] get $1 attempt $t failed"; sleep 10; done; return 1; }
python3 -m pip install -q "huggingface_hub[hf_xet]" transformers bitsandbytes datasets 2>&1 | grep -v -i warn | tail -1
for t in 1 2 3; do
  x=""; [ $t -gt 1 ] && x="HF_HUB_DISABLE_XET=1"
  env $x timeout 600 python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-1.7B'))" > dl.log 2>&1 && break
  echo "[tc] model download attempt $t failed"
done
echo "[tc] model: $(tail -1 dl.log | cut -c1-120)"
get archive/kvpool/traces.jsonl traces.jsonl; get archive/kvwin/q17_full_p.jsonl kv_full_p.jsonl
for i in 0 1 2 3 4 5; do get archive/sp3/traces_big_$i.jsonl traces_big_$i.jsonl; done
get archive/sp3/spC_spnext_fft_t400.pt spC0.pt; echo "[tc] start ckpt $(stat -c %s spC0.pt) bytes"
cat traces_big_[012].jsonl | tail -n +3201 > restA.jsonl; cat restA.jsonl traces_big_3.jsonl traces_big_4.jsonl traces_big_5.jsonl > trainB.jsonl
tail -n +4801 trainB.jsonl | tail -n +626 > trainC2.jsonl
echo "[tc] data: trainB $(wc -l < trainB.jsonl) (expect 8000), stage C' $(wc -l < trainC2.jsonl) unseen traces"
LG=/root/w/sp.log; F="--S 32 --rw 1024 --C 128 --maxd 384 --layers 1 --ffn 1024"
ev() { for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 $F "$@" --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback|out of memory" | sed "s/^/[$n] /"; done; }
echo "[tc] training launched $(date -u)"
python3 sp_qwen3.py train --traces trainC2.jsonl --max-len 3072 $F --ckpt spC0.pt --pe-rev --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 1e-4 --spnext --sn-pos 256 --sn-batch 8 --sn-tokens 3000 --accum 8 --out spC2.pt --log $LG > run_train.log 2>&1
grep -E "^\[sp\] (train .*: [0-9]+ traces|trained)|Error|Traceback|out of memory" run_train.log | tail -3
put spC2.pt spC2_spnext_perev_fft.pt; put $LG spC2.log
python3 sp_qwen3.py profile --traces kv_full_p.jsonl --n 16 --max-len 4096 $F --ckpt spC2.pt --log $LG 2>&1 | grep -E "^\[sp\]|Error|Traceback" | tail -3 | sed "s/^/[math] /"
ev --ckpt spC2.pt
put $LG spC2.log; date -u > DONE_TC; put DONE_TC DONE_TC; echo "[tc] DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tc] job launched $(date -u)"
fi
# CTL-END
