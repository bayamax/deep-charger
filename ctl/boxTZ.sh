#!/bin/bash
# Box TZ (3090 24 GB, pytorch image): box TY froze (GPU/CPU idle, reboot ignored) during the window-0 continuation; resume
# here from TY's 600-trace weights on the hub (3-layer pooler + all-layer FFT, held-out window-0 KL 2.79): window 0 until it
# stops falling (200-trace means < 1%, up to 1500), score window 0, then the fine window curriculum 2 .. 1024 by plateau,
# then the held-out profile + deployment loop. Pool traces 1-850 are skipped (TY used up to ~800). A live log goes to the
# hub every 10 min (Vast's log relay was unreliable).
SER=1
if [ -f /root/.boxtz_serial ] && [ "$(cat /root/.boxtz_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxtz_serial
echo "[tz] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w; RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i"
curl -sS -o /root/w/sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
cat > /root/status.sh <<'ST'
grep -E "^\[(tz|sp|hub|math|mix)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-220
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.job ]; then touch /root/.job
  cat > /root/live.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
while [ ! -e DONE_TZ ]; do
  { grep -E "^\[(tz|hub|math|mix)\]" w.log | tail -n 20; tail -n 12 sp.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; df -h / | tail -1; date -u; } > tz_live.log 2>&1
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='tz_live.log', path_in_repo='archive/sp3/tz_live.log', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1
  sleep 600
done
JOB
  cat > /root/job.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
get() { for t in 1 2 3; do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o "$2" "$H/$1" && return 0; echo "[tz] get $1 attempt $t failed"; sleep 10; done; return 1; }
python3 -m pip install -q "huggingface_hub[hf_xet]" transformers bitsandbytes datasets 2>&1 | grep -v -i warn | tail -1
for t in 1 2 3; do
  x=""; [ $t -gt 1 ] && x="HF_HUB_DISABLE_XET=1"
  env $x timeout 600 python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-1.7B'))" > dl.log 2>&1 && break
  echo "[tz] model download attempt $t failed"
done
echo "[tz] model: $(tail -1 dl.log | cut -c1-120)"
get archive/kvpool/traces.jsonl traces.jsonl; get archive/kvwin/q17_full_p.jsonl kv_full_p.jsonl
for i in 0 1 2 3 4 5; do get archive/sp3/traces_big_$i.jsonl traces_big_$i.jsonl; done
get archive/sp3/spY_L3_spnext_fft.pt spY.pt; echo "[tz] start ckpt $(stat -c %s spY.pt) bytes"
cat traces_big_[012].jsonl | tail -n +3201 > restA.jsonl; cat restA.jsonl traces_big_3.jsonl traces_big_4.jsonl traces_big_5.jsonl > trainB.jsonl
{ cat traces_big_[012].jsonl | head -n 3200; head -n 5425 trainB.jsonl; } > trainE.jsonl
tail -n +3851 trainE.jsonl > trainY.jsonl
echo "[tz] data: trainE $(wc -l < trainE.jsonl) (expect 8625), pool $(wc -l < trainY.jsonl) (expect 4775)"
LG=/root/w/sp.log; F="--S 32 --rw 1024 --C 128 --maxd 384 --layers 3 --ffn 1024"
NY=850
echo "[tz] window-0 resumed from spY (600 traces) at pool trace $NY $(date -u)"
python3 sp_qwen3.py train --traces trainY.jsonl --skip $NY --n 1500 --max-len 3072 $F --ckpt spY.pt --pe-rev --pe-norm --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 5e-4 --spnext --sn-win 1 --sn-win-plateau 0.01 --sn-win-block 8 --sn-win-max 1500 --sn-pos 256 --sn-batch 8 --sn-tokens 3000 --accum 1 --out spY2.pt --log $LG > run_trainY2.log 2>&1
grep -E "^\[sp\] (trained|curriculum)|Error|Traceback|out of memory" run_trainY2.log | tail -3
N2=$(grep -o 'trace [0-9]*/1500' sp.log | tail -1 | grep -o '[0-9]*/' | tr -d '/'); NS=$(( NY + N2 ))
python3 sp_qwen3.py profile --traces kv_full_p.jsonl --n 16 --max-len 4096 $F --ckpt spY2.pt --log $LG 2>&1 | grep -E "^\[sp\] profile|Error|Traceback" | tail -7 | sed "s/^/[math] /"
put spY2.pt spY2_L3_spnext_fft.pt; put $LG spY.log
echo "[tz] curriculum launched after $NS window-0 traces $(date -u)"
python3 sp_qwen3.py train --traces trainY.jsonl --skip $NS --n 3000 --max-len 3072 $F --ckpt spY2.pt --pe-rev --pe-norm --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 5e-4 --spnext --sn-win 2,4,8,16,32,64,128,256,512,1024 --sn-win-plateau 0.02 --sn-win-block 4 --sn-win-max 400 --sn-pos 256 --sn-batch 8 --sn-tokens 3000 --accum 1 --out spZ.pt --log $LG > run_trainZ.log 2>&1
grep -E "^\[sp\] (trained|curriculum)|Error|Traceback|out of memory" run_trainZ.log | tail -3
python3 sp_qwen3.py profile --traces kv_full_p.jsonl --n 16 --max-len 4096 $F --ckpt spZ.pt --log $LG 2>&1 | grep -E "^\[sp\] profile|Error|Traceback" | tail -7 | sed "s/^/[math] /"
for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 $F --ckpt spZ.pt --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback|out of memory" | sed "s/^/[$n] /"; done
put spZ.pt spZ_L3_wincurr_fft.pt; put $LG spZ.log; date -u > DONE_TZ; put DONE_TZ DONE_TZ; echo "[tz] DONE_Z $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/live.sh' > /dev/null 2>&1 < /dev/null &
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tz] job launched $(date -u)"
fi
# CTL-END
