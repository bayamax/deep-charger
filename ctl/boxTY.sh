#!/bin/bash
# Box TY (3090 24 GB, pytorch image): the user - the model must not stay frozen. The 3-layer pooler from stage F2 (window-0
# KL 3.32 held-out with the model frozen) + all-layer FFT from the original model, window-0 SP-next until it stops falling
# (200-trace means, < 1% better), at most 600 traces, on traces neither F2's pooler nor the model has seen (stage E's pool
# from 3851). Then held-out KL by window length (window 0 = the pooler's score; stage E = 3.02 with a 1-layer pooler).
SER=7
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
# serial 7: the box went idle (GPU 0%, no log relay) during the window-0 continuation - after a reboot: put diagnostics on the
# hub, then resume from the newest checkpoint (spY2.pt if the continuation saved one, else spY.pt), skipping every trace
# already used, with the same plan (window 0 until it stops falling, then the fine curriculum), and restart the live log.
if [ ! -e /root/.job7 ]; then touch /root/.job7
  cd /root/w
  { echo "== diag $(date -u)"; ls -la /root/w | head -40; df -h / | tail -1; free -g | head -2; nvidia-smi --query-gpu=memory.used --format=csv,noheader
    echo "== w.log"; grep -v "hf_[A-Za-z0-9]\{20\}" w.log | tail -n 25; echo "== run_trainY2.log"; tail -n 25 run_trainY2.log 2>/dev/null
    echo "== sp.log"; tail -n 6 sp.log; } > ty_diag.log 2>&1
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='ty_diag.log', path_in_repo='archive/sp3/ty_diag.log', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1 && echo "[ty] s7: diag uploaded"
  pkill -f job5.sh; pkill -f sp_qwen3.py; pkill -f live.sh; sleep 3
  N2=$(grep -o 'trace [0-9]*/1500' sp.log | tail -1 | grep -o '[0-9]*/' | tr -d '/'); N2=${N2:-0}
  if [ -s spY2.pt ] && [ "$N2" -ge 200 ]; then CK=spY2.pt; else CK=spY.pt; fi
  SK=$(( 600 + N2 + 25 )); echo "[ty] s7: resume from $CK, skip $SK traces (continuation logged $N2)"
  sed -e "s#^NY=.*#NY=$SK#" -e 's#^echo "\[ty\] window-0 continued after $NY traces.*#echo "[ty] window-0 resumed at trace $NY from '"$CK"' $(date -u)"#' \
      -e "s#--out spY2.pt#--out spY3.pt#; s#--ckpt spY2.pt#--ckpt spY3.pt#g; s#put spY2.pt spY2_L3_spnext_fft.pt#put spY3.pt spY3_L3_spnext_fft.pt#" \
      -e "s#--ckpt spY.pt --pe-rev --pe-norm --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 5e-4 --spnext --sn-win 1 #--ckpt $CK --pe-rev --pe-norm --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 5e-4 --spnext --sn-win 1 #" \
      -e "s#^while \[ ! -e DONE_TY \]; do sleep 30; done##" -e "s#run_trainY2.log#run_trainY3.log#g" /root/job5.sh > /root/job7.sh
  grep -q -- "--ckpt $CK" /root/job7.sh && grep -q "spY3.pt" /root/job7.sh && echo "[ty] s7: job7 ready" || echo "[ty] s7: WARNING job7 not as expected"
  cd /root; setsid nohup bash -c 'bash /root/job7.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  sed -i 's/tail -n 12 sp.log; date -u;/tail -n 12 sp.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; df -h \/ | tail -1; date -u;/' /root/live.sh
  setsid nohup bash -c 'bash /root/live.sh' > /dev/null 2>&1 < /dev/null &
  echo "[ty] s7: resumed + live log restarted $(date -u)"
fi
# serial 6: Vast's log relay stopped answering - push the training log to the hub every 10 min instead (archive/sp3/ty_live.log)
if [ ! -e /root/.live6 ]; then touch /root/.live6
  cat > /root/live.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
while [ ! -e DONE_TZ ]; do
  { grep -E "^\[(ty|hub|math|mix)\]" w.log | tail -n 20; tail -n 12 sp.log; date -u; } > ty_live.log
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='ty_live.log', path_in_repo='archive/sp3/ty_live.log', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1
  sleep 600
done
JOB
  setsid nohup bash -c 'bash /root/live.sh' > /dev/null 2>&1 < /dev/null &
  echo "[ty] s6: live log uploader started $(date -u)"
fi
# serial 5: the window-0 run is still falling at its 600 cap (3.9 -> 3.12 by 575) - continue window 0 from its weights until
# it stops (200-trace means < 1%, up to 1500 more), score window 0 held-out, THEN the fine curriculum from those weights.
if [ ! -e /root/.job5 ]; then touch /root/.job5
  pkill -f job2.sh
  cat > /root/job5.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
while [ ! -e DONE_TY ]; do sleep 30; done
LG=/root/w/sp.log; F="--S 32 --rw 1024 --C 128 --maxd 384 --layers 3 --ffn 1024"
NY=$(grep -o 'trace [0-9]*/600' sp.log | tail -1 | grep -o '[0-9]*/' | tr -d '/')
echo "[ty] window-0 continued after $NY traces $(date -u)"
python3 sp_qwen3.py train --traces trainY.jsonl --skip $NY --n 1500 --max-len 3072 $F --ckpt spY.pt --pe-rev --pe-norm --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 5e-4 --spnext --sn-win 1 --sn-win-plateau 0.01 --sn-win-block 8 --sn-win-max 1500 --sn-pos 256 --sn-batch 8 --sn-tokens 3000 --accum 1 --out spY2.pt --log $LG > run_trainY2.log 2>&1
grep -E "^\[sp\] (trained|curriculum)|Error|Traceback|out of memory" run_trainY2.log | tail -3
N2=$(grep -o 'trace [0-9]*/1500' sp.log | tail -1 | grep -o '[0-9]*/' | tr -d '/'); NS=$(( NY + N2 ))
python3 sp_qwen3.py profile --traces kv_full_p.jsonl --n 16 --max-len 4096 $F --ckpt spY2.pt --log $LG 2>&1 | grep -E "^\[sp\] profile|Error|Traceback" | tail -7 | sed "s/^/[math] /"
put spY2.pt spY2_L3_spnext_fft.pt; put $LG spY.log
echo "[ty] curriculum launched after $NS window-0 traces $(date -u)"
python3 sp_qwen3.py train --traces trainY.jsonl --skip $NS --n 3000 --max-len 3072 $F --ckpt spY2.pt --pe-rev --pe-norm --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 5e-4 --spnext --sn-win 2,4,8,16,32,64,128,256,512,1024 --sn-win-plateau 0.02 --sn-win-block 4 --sn-win-max 400 --sn-pos 256 --sn-batch 8 --sn-tokens 3000 --accum 1 --out spZ.pt --log $LG > run_trainZ.log 2>&1
grep -E "^\[sp\] (trained|curriculum)|Error|Traceback|out of memory" run_trainZ.log | tail -3
python3 sp_qwen3.py profile --traces kv_full_p.jsonl --n 16 --max-len 4096 $F --ckpt spZ.pt --log $LG 2>&1 | grep -E "^\[sp\] profile|Error|Traceback" | tail -7 | sed "s/^/[math] /"
for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 $F --ckpt spZ.pt --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback|out of memory" | sed "s/^/[$n] /"; done
put spZ.pt spZ_L3_wincurr_fft.pt; put $LG spZ.log; date -u > DONE_TZ; put DONE_TZ DONE_TZ; echo "[ty] DONE_Z $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job5.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[ty] s5: window-0 continuation + curriculum queued $(date -u)"
fi
# serial 4 (the user topped up the credit - fine steps are fine): cancel the serial-3 cut (the window-0 run goes on to its
# plateau / 600) and restore a fine curriculum by plateau: w = 2, 4, 8, ... 1024, 100-trace means < 2%, min 200, max 400
# per w, up to 3000 traces; MATH + mixed scoring.
if [ ! -e /root/.job4 ]; then touch /root/.job4
  pkill -f "trace 225/600"; echo "[ty] s4: window-0 cut cancelled"
  sed -i 's/--n 300/--n 3000/; s/--sn-win 4,32,256,1024 --sn-win-every 75/--sn-win 2,4,8,16,32,64,128,256,512,1024 --sn-win-plateau 0.02 --sn-win-block 4 --sn-win-max 400/; s/for e in kv_full_p.jsonl:math; do/for e in kv_full_p.jsonl:math traces.jsonl:mix; do/' /root/job2.sh
  grep -q -- "--n 3000" /root/job2.sh && grep -q -- "--sn-win 2,4,8,16,32,64,128,256,512,1024 --sn-win-plateau 0.02" /root/job2.sh && echo "[ty] s4: fine curriculum set" || echo "[ty] s4: WARNING job2 not changed"
  pgrep -f "sp_qwen3.py train" > /dev/null && echo "[ty] s4: window-0 run still training" || echo "[ty] s4: WARNING window-0 run not running"
fi
# serial 3: the credit (~$0.6 at $0.30/h) does not cover 600 window-0 traces at ~9 s each plus the curriculum: stop the
# window-0 run at its 200-trace checkpoint (killed after the 225 log; job.sh then profiles and uploads it as usual), and run
# the curriculum fixed at 75 traces per w (4, 32, 256, 1024 = 300 traces), scoring MATH only.
if [ ! -e /root/.cut3 ]; then touch /root/.cut3
  sed -i 's/--n 800/--n 300/; s/--sn-win-plateau 0.02 --sn-win-block 2 --sn-win-max 200/--sn-win-every 75/; s/for e in kv_full_p.jsonl:math traces.jsonl:mix; do/for e in kv_full_p.jsonl:math; do/' /root/job2.sh
  grep -q -- "--n 300" /root/job2.sh && grep -q -- "--sn-win-every 75" /root/job2.sh && echo "[ty] s3: curriculum cut to 300 traces" || echo "[ty] s3: WARNING job2 not changed"
  setsid nohup bash -c 'for i in $(seq 1 120); do grep -q "trace 225/600" /root/w/sp.log && break; sleep 20; done; pkill -f "sp_qwen3.py train"; echo "[ty] s3: window-0 run stopped at its 200-trace checkpoint $(date -u)" >> /root/w/w.log' > /dev/null 2>&1 < /dev/null &
  echo "[ty] s3: cut armed $(date -u)"
fi
# serial 2 (the user: then the window, without a reset): after the window-0 run, from its weights on the same box, the
# window curriculum w = 4, 32, 256, 1024 stepped when the KL stops falling (50-trace means, < 2%; min 100, max 200 per w -
# the credit is short), on the next unseen traces; then held-out profile + the deployment loop vs stage A 0.079 / 0.066.
if [ ! -e /root/.job2 ]; then touch /root/.job2
  cat > /root/job2.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
while [ ! -e DONE_TY ]; do sleep 30; done
NY=$(grep -o 'trace [0-9]*/600' sp.log | tail -1 | grep -o '[0-9]*/' | tr -d '/')
LG=/root/w/sp.log; F="--S 32 --rw 1024 --C 128 --maxd 384 --layers 3 --ffn 1024"
echo "[ty] curriculum launched after window-0 run ($NY traces) $(date -u)"
python3 sp_qwen3.py train --traces trainY.jsonl --skip $NY --n 800 --max-len 3072 $F --ckpt spY.pt --pe-rev --pe-norm --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 5e-4 --spnext --sn-win 4,32,256,1024 --sn-win-plateau 0.02 --sn-win-block 2 --sn-win-max 200 --sn-pos 256 --sn-batch 8 --sn-tokens 3000 --accum 1 --out spZ.pt --log $LG > run_trainZ.log 2>&1
grep -E "^\[sp\] (train .*: [0-9]+ traces|trained|curriculum)|Error|Traceback|out of memory" run_trainZ.log | tail -3
python3 sp_qwen3.py profile --traces kv_full_p.jsonl --n 16 --max-len 4096 $F --ckpt spZ.pt --log $LG 2>&1 | grep -E "^\[sp\] profile|Error|Traceback" | tail -7 | sed "s/^/[math] /"
for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 $F --ckpt spZ.pt --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback|out of memory" | sed "s/^/[$n] /"; done
put spZ.pt spZ_L3_wincurr_fft.pt; put $LG spZ.log; date -u > DONE_TZ; put DONE_TZ DONE_TZ; echo "[ty] DONE_Z $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job2.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[ty] s2: curriculum queued after the window-0 run $(date -u)"
fi
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
