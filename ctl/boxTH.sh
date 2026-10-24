#!/bin/bash
# Box TH (3090 24 GB, pytorch image): the next lever from the original deep-charger INVESTIGATION.md - feed the pooler the
# frozen model's CONTEXTUALISED hidden states of the distant tokens instead of raw embeddings (their overfit floor 0.16 -> 0.11).
# Same recipe as box TP's frozen run (3-layer pooler, rw 128, C 64, 768-token contexts, 4000 once, lr 3e-4), layer 14 and the
# last layer (28) side by side; the same scores as the raw-embedding pooler (rw 128: MATH 0.150 / mixed 0.221; rw 512 on 4096
# tokens: 0.140 / 0.129; stage-A conditions: 0.084 / 0.071).
SER=1
if [ -f /root/.boxth_serial ] && [ "$(cat /root/.boxth_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxth_serial
echo "[th] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w; RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i"
curl -sS -o /root/w/sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
cat > /root/status.sh <<'ST'
grep -E "^\[(th|sp|hub|math|mix)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-220
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.job ]; then touch /root/.job
  cat > /root/live.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
while [ ! -e DONE_TH ]; do
  { grep -E "^\[(th|hub|math|mix)\]" w.log | tail -n 20; tail -n 4 spH_L14.log; tail -n 4 spH_L28.log; tail -n 6 sp.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; df -h / | tail -1; date -u; } > th_live.log 2>&1
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='th_live.log', path_in_repo='archive/sp3/th_live.log', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1
  sleep 600
done
JOB
  cat > /root/job.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; echo "[th] put $2 attempt $t: $(tail -1 put.err | cut -c1-160)"; sleep 90; done; echo "[hub] $2 FAILED"; }
get() { for t in 1 2 3; do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o "$2" "$H/$1" && return 0; echo "[th] get $1 attempt $t failed"; sleep 10; done; return 1; }
python3 -m pip install -q "huggingface_hub[hf_xet]" transformers bitsandbytes datasets 2>&1 | grep -v -i warn | tail -1
for t in 1 2 3; do
  x=""; [ $t -gt 1 ] && x="HF_HUB_DISABLE_XET=1"
  env $x timeout 600 python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-1.7B'))" > dl.log 2>&1 && break
  echo "[th] model download attempt $t failed"
done
echo "[th] model: $(tail -1 dl.log | cut -c1-120)"
get archive/kvpool/traces.jsonl traces.jsonl; get archive/kvwin/q17_full_p.jsonl kv_full_p.jsonl
for i in 0 1 2 3 4 5; do get archive/sp3/traces_big_$i.jsonl traces_big_$i.jsonl; done
LG=/root/w/sp.log
P="--S 32 --heads 8 --layers 3 --ffn 2048 --C 64 --maxd 0"
echo "[th] training two hidden-input poolers (layer 14 | layer 28) $(date -u)"
python3 sp_qwen3.py train --traces traces_big_0.jsonl,traces_big_1.jsonl --n 4000 --max-len 768 --rw 128 $P --ctx-hidden 14 --lr 3e-4 --accum 1 --out spH_L14.pt --log spH_L14.log > run_L14.log 2>&1 &
sleep 120
python3 sp_qwen3.py train --traces traces_big_0.jsonl,traces_big_1.jsonl --n 4000 --max-len 768 --rw 128 $P --ctx-hidden 28 --lr 3e-4 --accum 1 --out spH_L28.pt --log spH_L28.log > run_L28.log 2>&1 &
wait
for v in L14 L28; do grep -E "^\[sp\] trained|Error|Traceback|out of memory" run_$v.log | tail -2 | sed "s/^/[th] $v: /"; done
put spH_L14.pt spH_L14_ctxhidden.pt; put spH_L28.pt spH_L28_ctxhidden.pt; cat spH_L14.log spH_L28.log >> $LG; put $LG spH.log
for c in spH_L14.pt spH_L28.pt; do for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces $f $x --max-len 768 --score-from 224 --rw 128 $P --ckpt $c --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] rw128: /"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 --score-from 608 --rw 512 --S 32 --heads 8 --layers 3 --ffn 2048 --C 64 --maxd 4096 --ckpt $c --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] rw512: /"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 --rw 1024 --S 32 --heads 8 --layers 3 --ffn 2048 --C 128 --maxd 384 --ckpt $c --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] stageA-cond: /"; done; done
put $LG spH.log; date -u > DONE_TH; put DONE_TH DONE_TH; echo "[th] DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/live.sh' > /dev/null 2>&1 < /dev/null &
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[th] job launched $(date -u)"
fi
# CTL-END
