#!/bin/bash
# Box TA (3090 24 GB, pytorch image): the single-pass run on the big fresh trace set. Stage A = the pooler alone (model
# frozen) on shards 0-2 (~6.5k traces, each seen once, 4 chunks back-propagated per trace), numbers from the sweep: S32,
# raw window 1024, chunk 128, maxD 384, 1 block / ffn 1024 (~19 MB at 4 bits). Held-out KL (MATH 64 / mixed 60) on the
# default region (from the first eviction, 4096 tokens) and on the common region (1792+). The model is fetched first
# (timeout + retry without xet): an earlier box hung inside the first eval on a stalled download.
SER=16
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
# serial 16: stop stage C (window-0 KL only 6.06 -> 5.7 in 500 traces: the pooler's positions count from the oldest
# token, so "the newest token" has no fixed index); keep its weights: upload the latest checkpoints, then the box goes
if [ ! -e /root/.stopc ]; then touch /root/.stopc
  pkill -f jobc2.sh; pkill -f sp_qwen3.py; sleep 3; cd /root/w
  for f in spC.pt:spC_spnext_fft_t400.pt spB_par_stop.pt:spB_par_fft_t4800.pt; do src=${f%%:*}; dst=${f#*:}
    python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='$src', path_in_repo='archive/sp3/$dst', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1 && echo "[hub] $dst" || echo "[hub] $dst FAILED"
  done
  python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='sp.log', path_in_repo='archive/sp3/spC.log', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1
  cp trainC.jsonl /root/w/trainC_all.jsonl; date -u > STOPPED_C
  python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='STOPPED_C', path_in_repo='archive/sp3/STOPPED_C', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1
  echo "[ta] s16: stage C stopped, checkpoints uploaded $(date -u)"; cd /root
fi
# serial 15: stage C ran out of memory (FFT state ~17 GB + 32 positions x prompt KV) - rerun with 8 positions per batch
if [ ! -e /root/.jobc2 ]; then touch /root/.jobc2
  pkill -f jobc.sh; pkill -f sp_qwen3.py; sleep 3
  sed 's/--sn-batch 32 --sn-tokens 12000/--sn-batch 8 --sn-tokens 3000/' /root/jobc.sh > /root/jobc2.sh
  grep -q -- "--sn-batch 8 --sn-tokens 3000" /root/jobc2.sh && echo "[ta] s15: batch 8 set" || echo "[ta] s15: WARNING batch not changed"
  setsid nohup bash -c 'bash /root/jobc2.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[ta] s15: stage C relaunched $(date -u)"
fi
# serial 14: diagnostics - stage C stopped within minutes
if [ ! -e /root/.diag14 ]; then touch /root/.diag14
  echo "[ta] d14 err: $(grep -E 'Error' /root/w/run_trainC.log | tail -2 | tr '\n' ' ' | cut -c1-500)"
  echo "[ta] d14 frames: $(grep -E 'File \"/root/w/sp_qwen3.py' /root/w/run_trainC.log | tail -3 | tr '\n' ' ')"
fi
# serial 13 (the user's hypothesis holds: SP alone predicts almost nothing (window-0 KL ~5), the raw window hides it):
# stop the --par stage B at its last checkpoint and continue with window-0 SP-next training + all-layer FFT on the
# traces stage B has not seen; scored with the deployment loop at the end.
if [ ! -e /root/.jobc ]; then touch /root/.jobc
  cd /root/w
  LAST=$(grep -o 'par FT28 trace [0-9]*/8000' sp.log | tail -1 | grep -o '[0-9]*/' | tr -d '/'); NS=$(( LAST / 200 * 200 ))
  pkill -f upl.sh; pkill -f jobb5.sh; pkill -f sp_qwen3.py; sleep 5
  cp spB.pt spB_par_stop.pt
  tail -n +$(( NS + 1 )) trainB.jsonl > trainC.jsonl
  echo "[ta] s13: stage B (par) stopped at trace $LAST, checkpoint = trace $NS; stage C data: $(wc -l < trainC.jsonl) unseen traces"
  curl -sS -o /root/w/sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
  cd /root
  cat > /root/jobc.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
LG=/root/w/sp.log; F="--S 32 --rw 1024 --C 128 --maxd 384 --layers 1 --ffn 1024"
ev() { for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  for sf in 0 1792; do python3 sp_qwen3.py eval --traces $f $x --max-len 4096 --score-from $sf $F "$@" --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback|out of memory" | sed "s/^/[$n] /"; done; done; }
python3 sp_qwen3.py train --traces trainC.jsonl --max-len 3072 $F --ckpt spB_par_stop.pt --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 1e-4 --spnext --sn-pos 256 --sn-batch 32 --sn-tokens 12000 --accum 8 --out spC.pt --log $LG > run_trainC.log 2>&1
grep -E "^\[sp\] (train .*: [0-9]+ traces|trained)|Error|Traceback|out of memory" run_trainC.log | tail -3
put spC.pt spC_spnext_fft.pt; put $LG spC.log
ev --ckpt spC.pt
put $LG spC.log; date -u > DONE_TA_C; put DONE_TA_C DONE_TA_C; echo "[ta] STAGE_C_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/jobc.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[ta] s13: stage C (spnext + FFT) launched $(date -u)"
fi
# serial 12: a background uploader - the stage-B checkpoint (saved every 200 traces) to the hub every 45 min, so it can
# be scored mid-run elsewhere without stopping the training
if [ ! -e /root/.upl ]; then touch /root/.upl
  cat > /root/upl.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
while pgrep -f jobb5.sh > /dev/null; do
  sleep 2700
  [ -s spB.pt ] && cp spB.pt spB_mid.pt && python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='spB_mid.pt', path_in_repo='archive/sp3/spB_mid.pt', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1 \
    && echo "[ta] mid ckpt uploaded at $(grep -o 'trace [0-9]*/8000' /root/w/sp.log | tail -1) $(date -u)"
done
JOB
  setsid nohup bash -c 'bash /root/upl.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[ta] s12: mid-run checkpoint uploader started $(date -u)"
fi
# serial 11 (the user: use the parallel all-window training if it runs smoothly): stage B again from spA_stop.pt with
# --par - one forward of [prompt | SP_b | the next rw+C tokens] scores every prediction in it (window 0 .. rw+C), 4
# boundaries per trace, all-layer FFT + pooler, the same 8000 unseen traces; scoring unchanged (deployment loop).
if [ ! -e /root/.jobb5 ]; then touch /root/.jobb5
  pkill -f jobb4.sh; pkill -f sp_qwen3.py; sleep 3
  curl -sS -o /root/w/sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
  sed 's/s7: stage B training on/s11: stage B (par) training on/; s/--chunks-per-trace 4 --accum 8 --out spB.pt/--par --chunks-per-trace 4 --accum 8 --out spB.pt/' /root/jobb2.sh > /root/jobb5.sh
  grep -q -- "--par --chunks-per-trace" /root/jobb5.sh && echo "[ta] s11: job uses --par" || echo "[ta] s11: WARNING --par not set"
  setsid nohup bash -c 'bash /root/jobb5.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[ta] s11: stage B (par) launched $(date -u)"
fi
# serial 10: stage B ran at ~20 s/trace (a full 2.8 GB weight swap through the CPU for every trace's teacher pass) -
# new sp_qwen3.py does one swap per 8 traces, frozen copy pinned, restore from the fp32 master on the GPU; restart stage B
if [ ! -e /root/.jobb4 ]; then touch /root/.jobb4
  pkill -f jobb3.sh; pkill -f sp_qwen3.py; sleep 3
  curl -sS -o /root/w/sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
  sed 's/s7: stage B training on/s10: stage B training on/' /root/jobb2.sh > /root/jobb4.sh
  setsid nohup bash -c 'bash /root/jobb4.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[ta] s10: stage B restarted with the fast swap $(date -u)"
fi
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
