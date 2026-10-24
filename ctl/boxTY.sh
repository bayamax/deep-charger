#!/bin/bash
# Box TY (3090 24 GB, pytorch image): the user - the model must not stay frozen. The 3-layer pooler from stage F2 (window-0
# KL 3.32 held-out with the model frozen) + all-layer FFT from the original model, window-0 SP-next until it stops falling
# (200-trace means, < 1% better), at most 600 traces, on traces neither F2's pooler nor the model has seen (stage E's pool
# from 3851). Then held-out KL by window length (window 0 = the pooler's score; stage E = 3.02 with a 1-layer pooler).
SER=1
if [ -f /root/.boxty_serial ] && [ "$(cat /root/.boxty_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxty_serial
echo "[ty] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w; RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i"
curl -sS -o /root/w/sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
cat > /root/status.sh <<'ST'
grep -E "^\[(ty|sp|hub|math|mix)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-220
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.job ]; then touch /root/.job
  cat > /root/job.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
get() { for t in 1 2 3; do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o "$2" "$H/$1" && return 0; echo "[ty] get $1 attempt $t failed"; sleep 10; done; return 1; }
python3 -m pip install -q "huggingface_hub[hf_xet]" transformers bitsandbytes datasets 2>&1 | grep -v -i warn | tail -1
for t in 1 2 3; do
  x=""; [ $t -gt 1 ] && x="HF_HUB_DISABLE_XET=1"
  env $x timeout 600 python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-1.7B'))" > dl.log 2>&1 && break
  echo "[ty] model download attempt $t failed"
done
echo "[ty] model: $(tail -1 dl.log | cut -c1-120)"
get archive/kvpool/traces.jsonl traces.jsonl; get archive/kvwin/q17_full_p.jsonl kv_full_p.jsonl
for i in 0 1 2 3 4 5; do get archive/sp3/traces_big_$i.jsonl traces_big_$i.jsonl; done
get archive/sp3/spF2_L3_spnext.pt spF20.pt; echo "[ty] start ckpt $(stat -c %s spF20.pt) bytes"
cat traces_big_[012].jsonl | tail -n +3201 > restA.jsonl; cat restA.jsonl traces_big_3.jsonl traces_big_4.jsonl traces_big_5.jsonl > trainB.jsonl
{ cat traces_big_[012].jsonl | head -n 3200; head -n 5425 trainB.jsonl; } > trainE.jsonl
tail -n +3851 trainE.jsonl > trainY.jsonl
echo "[ty] data: trainE $(wc -l < trainE.jsonl) (expect 8625), pool $(wc -l < trainY.jsonl) unseen traces"
LG=/root/w/sp.log; F="--S 32 --rw 1024 --C 128 --maxd 384 --layers 3 --ffn 1024"
echo "[ty] training launched $(date -u)"
python3 sp_qwen3.py train --traces trainY.jsonl --n 600 --max-len 3072 $F --ckpt spF20.pt --pe-rev --pe-norm --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 5e-4 --spnext --sn-win 1 --sn-win-plateau 0.01 --sn-win-block 8 --sn-win-max 600 --sn-pos 256 --sn-batch 8 --sn-tokens 3000 --accum 1 --out spY.pt --log $LG > run_train.log 2>&1
grep -E "^\[sp\] (train .*: [0-9]+ traces|trained|curriculum)|Error|Traceback|out of memory" run_train.log | tail -3
python3 sp_qwen3.py profile --traces kv_full_p.jsonl --n 16 --max-len 4096 $F --ckpt spY.pt --log $LG 2>&1 | grep -E "^\[sp\] profile|Error|Traceback" | tail -7 | sed "s/^/[math] /"
put spY.pt spY_L3_spnext_fft.pt; put $LG spY.log; date -u > DONE_TY; put DONE_TY DONE_TY; echo "[ty] DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[ty] job launched $(date -u)"
fi
# CTL-END
