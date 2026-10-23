#!/bin/bash
# Box TC (3090 24 GB, pytorch image): stage C' = window-0 SP-next training (--spnext) + all-layer FFT, with the pooler's
# positions counted from the NEWEST buffer token (--pe-rev): stage C (forward positions) sat at KL ~5.7 for 625 traces.
# Starts from stage C's weights (spC_spnext_fft_t400.pt); data = the stage-C traces it has not seen (stage B's 8000 unseen
# traces from 4801, minus the 625 stage C used) - every trace once. The original model stays the teacher. At the end:
# held-out KL by window length (profile) and the deployment loop, both against stages A / B.
SER=5
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
# serial 5: stage D (new pooler, model frozen, pe-norm) took window-0 KL 6.8 -> ~3.9 by trace ~1800 and is flattening
# (copy ceiling 0.79 at S=32). Stage E = unfreeze: all-layer FFT (fp32 master, 8-bit AdamW) + the pooler from D's
# 1800-trace checkpoint, pooler lr 1e-3, one update per trace, 2000 traces neither D's pooler nor the model has seen
# (shards 0-2 lines 1-3200, then trainB lines 1-5425 - only discarded poolers / discarded FFT runs saw them).
if [ ! -e /root/.job5 ]; then touch /root/.job5
  cd /root/w
  for i in $(seq 1 40); do n=$(grep -o 'pe-norm trace [0-9]*/3725' sp.log | tail -1 | grep -o '[0-9]*/' | tr -d '/'); [ "${n:-0}" -ge 1825 ] && break; sleep 30; done
  pkill -f job3.sh; pkill -f sp_qwen3.py; sleep 5
  cp spD.pt spD_t1800.pt; echo "[tc] s5: stage D stopped at trace $n (checkpoint 1800)"
  { cat traces_big_[012].jsonl | head -n 3200; head -n 5425 trainB.jsonl; } > trainE.jsonl; echo "[tc] s5: stage E pool $(wc -l < trainE.jsonl) traces"
  curl -sS -o /root/w/sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
  { sed -n '1,4p' /root/job3.sh; grep -E '^(LG=|ev\(\)|  python3 sp_qwen3.py eval)' /root/job3.sh; } > /root/job5.sh
  cat >> /root/job5.sh <<'JOB'
put spD_t1800.pt spD_spnext_penorm_t1800.pt; put $LG spD.log
echo "[tc] stage E launched $(date -u)"
python3 sp_qwen3.py train --traces trainE.jsonl --n 2000 --max-len 3072 $F --ckpt spD_t1800.pt --pe-rev --pe-norm --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 1e-3 --spnext --sn-pos 256 --sn-batch 8 --sn-tokens 3000 --accum 1 --out spE.pt --log $LG > run_trainE.log 2>&1
grep -E "^\[sp\] (train .*: [0-9]+ traces|trained)|Error|Traceback|out of memory" run_trainE.log | tail -3
put spE.pt spE_spnext_penorm_fft.pt; put $LG spE.log
python3 sp_qwen3.py profile --traces kv_full_p.jsonl --n 16 --max-len 4096 $F --ckpt spE.pt --log $LG 2>&1 | grep -E "^\[sp\]|Error|Traceback" | tail -7 | sed "s/^/[math] /"
ev --ckpt spE.pt
put $LG spE.log; date -u > DONE_TE; put DONE_TE DONE_TE; echo "[tc] DONE_E $(date -u)"
JOB
  echo "[tc] s5: job5 $(wc -l < /root/job5.sh) lines"; cd /root
  setsid nohup bash -c 'bash /root/job5.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tc] s5: launched $(date -u)"
fi
# serial 4: side measurement while stage D trains - the window-0 ceiling of a pooler that just copies: [prompt | the
# embeddings of the last S tokens] -> KL vs the full model (raw and fixed-norm embeddings; S = 1, 8, 32).
if [ ! -e /root/.orc ]; then touch /root/.orc
  cd /root/w; curl -sS -o sp_oracle.py "$RAW/tools/sp_oracle.py?nocache=$(date +%s)"
  setsid nohup bash -c 'cd /root/w; python3 sp_oracle.py kv_full_p.jsonl Qwen/Qwen3-1.7B 8 1,8,32 2>&1 | grep -E "^\{|Error" | tail -2 | sed "s/^/[tc] oracle /" >> /root/w/w.log' > /dev/null 2>&1 < /dev/null &
  echo "[tc] s4: oracle started $(date -u)"; cd /root
fi
# serial 3: the pooler's sinusoidal positions had norm sqrt(H/2) = 32 against token embeddings of ~1.5, so the keys and
# values it reads were ~95% position and ~5% content: the SP carried almost nothing (window-0 KL ~5.5 in every run).
# Fix (--pe-norm: positions scaled to the embedding norm) + newest-first positions; a NEW pooler on the original model
# (frozen, pooler only - the stage-A recipe) on window-0 SP-next, one update per trace; FFT comes after if this moves.
if [ ! -e /root/.job3 ]; then touch /root/.job3
  pkill -f job2.sh; pkill -f sp_qwen3.py; sleep 5; cd /root/w
  curl -sS -o /root/w/sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
  grep -q "pe_norm" sp_qwen3.py && echo "[tc] s3: trainer has --pe-norm" || echo "[tc] s3: WARNING old trainer"
  sed -e 's/--ckpt spC0.pt --pe-rev --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 5e-4/--pe-rev --pe-norm --lr 5e-4/' \
      -e 's/--sn-batch 8 --sn-tokens 3000 --accum 1 --skip 250 --out spC2.pt/--sn-batch 32 --sn-tokens 12000 --accum 1 --skip 650 --out spD.pt/' \
      -e 's/spC2\.pt/spD.pt/g; s/spC2_spnext_perev_fft\.pt/spD_spnext_penorm.pt/; s/spC2\.log/spD.log/g; s/DONE_TC/DONE_TD/g' /root/job2.sh > /root/job3.sh
  grep -q -- "--pe-rev --pe-norm --lr 5e-4 --spnext" /root/job3.sh && grep -q -- "--skip 650 --out spD.pt" /root/job3.sh && echo "[tc] s3: args set" || echo "[tc] s3: WARNING args not set"
  cd /root; setsid nohup bash -c 'bash /root/job3.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tc] s3: launched $(date -u)"
fi
# serial 2: KL flat at 5.6 after 100 traces = only 12 optimizer steps (8 traces per update). Restart with one update per
# trace (256 positions each) and pooler lr 5e-4 so it can learn to carry the newest tokens; the ~250 traces the first run used are skipped.
if [ ! -e /root/.job2 ]; then touch /root/.job2
  pkill -f job.sh; pkill -f sp_qwen3.py; sleep 5; cd /root/w
  curl -sS -o /root/w/sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
  { sed -n '1,4p' /root/job.sh; grep -E '^(LG=|ev\(\)|  python3 sp_qwen3.py eval)' /root/job.sh; sed -n '/^echo "\[tc\] training launched/,$p' /root/job.sh; } > /root/job2.sh
  sed -i 's/--accum 8 --out spC2.pt/--accum 1 --skip 250 --out spC2.pt/; s/--ft-lr 1e-5 --lr 1e-4 --spnext/--ft-lr 1e-5 --lr 5e-4 --spnext/' /root/job2.sh
  grep -q -- "--accum 1 --skip 250" /root/job2.sh && grep -q -- "--lr 5e-4 --spnext" /root/job2.sh && echo "[tc] s2: args set" || echo "[tc] s2: WARNING args not set"
  echo "[tc] s2: job2 lines $(wc -l < /root/job2.sh)"; cd /root
  setsid nohup bash -c 'bash /root/job2.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tc] s2: relaunched $(date -u)"
fi
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
