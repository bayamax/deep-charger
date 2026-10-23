#!/bin/bash
# Box TA (3090 24 GB, pytorch image): the single-pass run on the big fresh trace set. Stage A = the pooler alone (model
# frozen) on shards 0-2 (~6.5k traces, each seen once, 4 chunks back-propagated per trace), numbers from the sweep: S32,
# raw window 1024, chunk 128, maxD 384, 1 block / ffn 1024 (~19 MB at 4 bits). Held-out KL (MATH 64 / mixed 60) on the
# default region (from the first eviction, 4096 tokens) and on the common region (1792+). The model is fetched first
# (timeout + retry without xet): an earlier box hung inside the first eval on a stalled download.
SER=9
if [ -f /root/.boxta_serial ] && [ "$(cat /root/.boxta_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxta_serial
echo "[ta] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w; RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i"
curl -sS -o /root/w/sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
cat > /root/status.sh <<'ST'
grep -E "^\[(ta|sp|hub|math|mix)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-200
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
# serial 9: stage B crashed at its first backward (the teacher's lm-head step swapped the frozen weights in place, between
# the student's forward and backward) - fixed in sp_qwen3.py; relaunch stage B with the new code
if [ ! -e /root/.jobb3 ]; then touch /root/.jobb3
  pkill -f jobb2.sh; pkill -f sp_qwen3.py; sleep 3
  curl -sS -o /root/w/sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
  sed 's/s7: stage B training on/s9: stage B training on/' /root/jobb2.sh > /root/jobb3.sh
  setsid nohup bash -c 'bash /root/jobb3.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[ta] s9: stage B relaunched with the fix $(date -u)"
fi
# serial 8: diagnostics - stage B header printed but the GPU looks empty
if [ ! -e /root/.diag8 ]; then touch /root/.diag8
  echo "[ta] d8 ps: $(ps aux | grep -E 'sp_qwen3|jobb' | grep -v grep | awk '{print $3"%cpu", $6/1e6"GB", $12, $13, $14}' | tr '\n' ';')"
  echo "[ta] d8 smi: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) | $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')"
  echo "[ta] d8 trainB: $(tail -c 700 /root/w/run_trainB.log | tr '\n' ' ')"
  echo "[ta] d8 mem: $(free -g | awk 'NR==2{print $3"G used of "$2"G"}')"
fi
# serial 7: serial 6 stopped the started stage-B training without relaunching it - relaunch training + scoring on the fixed data
if [ ! -e /root/.jobb2 ]; then touch /root/.jobb2
  pkill -f jobb.sh; pkill -f sp_qwen3.py; sleep 5
  cat > /root/jobb2.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
LG=/root/w/sp.log; F="--S 32 --rw 1024 --C 128 --maxd 384 --layers 1 --ffn 1024"
ev() { for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  for sf in 0 1792; do python3 sp_qwen3.py eval --traces $f $x --max-len 4096 --score-from $sf $F "$@" --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback|out of memory" | sed "s/^/[$n] /"; done; done; }
echo "[ta] s7: stage B training on $(wc -l < trainB.jsonl) unseen traces $(date -u)"
python3 sp_qwen3.py train --traces trainB.jsonl --max-len 3072 $F --ckpt spA_stop.pt --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 1e-4 --chunks-per-trace 4 --accum 8 --out spB.pt --log $LG > run_trainB.log 2>&1
grep -E "^\[sp\] (train .*: [0-9]+ traces|trained)|Error|Traceback|out of memory" run_trainB.log | tail -3
put spB.pt spB_rw1024_L1_fft.pt; put $LG spB.log
ev --ckpt spB.pt
put $LG spB.log; date -u > DONE_TA_B; put DONE_TA_B DONE_TA_B; echo "[ta] STAGE_B_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/jobb2.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[ta] s7: stage B relaunched $(date -u)"
fi
# serial 6: stage A had reached trace 5075 (spA.pt = trace 5000), so stage B starts at trace 5001 of shards 0-2, not 3201
if [ ! -e /root/.fixb ]; then touch /root/.fixb
  cd /root/w
  if [ -e run_trainB.log ]; then pkill -f "sp_qwen3.py train"; echo "[ta] s6: stage B training had started - restarting it"; RESTART=1; fi
  cat traces_big_[012].jsonl | tail -n +5001 > restA.jsonl; cat restA.jsonl traces_big_3.jsonl traces_big_4.jsonl traces_big_5.jsonl > trainB.jsonl
  echo "[ta] s6: stage B data fixed: $(wc -l < trainB.jsonl) unseen traces (spA saw 1-5000)"
  cd /root
fi
# serial 5 (the user: KL is falling - move on now): stop stage A where it is (spA.pt is saved every 200 traces), score it,
# then stage B = full fine-tune of all decoder layers (fp32 master, 8-bit AdamW) + the pooler, on traces never seen:
# the rest of shards 0-2 (from 3201) + shards 3-5 (~9.8k), 4 chunks per trace, 8 traces per update.
if [ ! -e /root/.jobb ]; then touch /root/.jobb
  pkill -f joba.sh; pkill -f "sp_qwen3.py train"; sleep 5; echo "[ta] s5: stage A stopped at $(grep -o 'trace [0-9]*/6501' /root/w/sp.log | tail -1) $(date -u)"
  cat > /root/jobb.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
for i in 3 4 5; do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o traces_big_$i.jsonl "$H/archive/sp3/traces_big_$i.jsonl"; done
cat traces_big_[012].jsonl | tail -n +3201 > restA.jsonl; cat restA.jsonl traces_big_3.jsonl traces_big_4.jsonl traces_big_5.jsonl > trainB.jsonl
echo "[ta] stage B data: $(wc -l < trainB.jsonl) unseen traces"
LG=/root/w/sp.log; F="--S 32 --rw 1024 --C 128 --maxd 384 --layers 1 --ffn 1024"
ev() { for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  for sf in 0 1792; do python3 sp_qwen3.py eval --traces $f $x --max-len 4096 --score-from $sf $F "$@" --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback|out of memory" | sed "s/^/[$n] /"; done; done; }
cp spA.pt spA_stop.pt; put spA_stop.pt spA_rw1024_L1.pt
ev --ckpt spA_stop.pt
put $LG spA.log
python3 sp_qwen3.py train --traces trainB.jsonl --max-len 3072 $F --ckpt spA_stop.pt --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 1e-4 --chunks-per-trace 4 --accum 8 --out spB.pt --log $LG > run_trainB.log 2>&1
grep -E "^\[sp\] (train .*: [0-9]+ traces|trained)|Error|Traceback|out of memory" run_trainB.log | tail -3
put spB.pt spB_rw1024_L1_fft.pt; put $LG spB.log
ev --ckpt spB.pt
put $LG spB.log; date -u > DONE_TA_B; put DONE_TA_B DONE_TA_B; echo "[ta] STAGE_B_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/jobb.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[ta] s5: stage B job launched $(date -u)"
fi
if [ ! -e /root/.joba ]; then touch /root/.joba
  cat > /root/joba.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
python3 -m pip install -q "huggingface_hub[hf_xet]" transformers bitsandbytes datasets 2>&1 | grep -v -i warn | tail -1
for t in 1 2 3; do
  x=""; [ $t -gt 1 ] && x="HF_HUB_DISABLE_XET=1"
  env $x timeout 600 python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-1.7B'))" > dl.log 2>&1 && break
  echo "[ta] model download attempt $t failed"
done
echo "[ta] model: $(tail -1 dl.log | cut -c1-120)"
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o traces.jsonl "$H/archive/kvpool/traces.jsonl"
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o kv_full_p.jsonl "$H/archive/kvwin/q17_full_p.jsonl"
for i in 0 1 2; do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o traces_big_$i.jsonl "$H/archive/sp3/traces_big_$i.jsonl"; done
echo "[ta] data: $(cat traces_big_[012].jsonl | wc -l) big traces (shards 0-2)"
LG=/root/w/sp.log; F="--S 32 --rw 1024 --C 128 --maxd 384 --layers 1 --ffn 1024"
ev() { for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
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
