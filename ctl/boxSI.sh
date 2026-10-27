#!/bin/bash
# Box SI (a 3090): grow S the way the character model grew depth. From G (S 32), S -> 64 -> 128: the new SP queries start as
# copies of the old ones plus noise, then end-aware --par FFT on the long teacher traces (stage G's 1414 + stage H2's 1264,
# shuffled per stage), 600 traces per segment, after each segment the deployment MATH KL (4096) and the window profile (window
# 0 = SP alone); a stage stops after two segments without a 1% KL gain (or the pool runs out); its best segment then gets the
# 96-problem check (2 seeds), and the next S grows from it. Results to archive/sp3/ (spS64_*, spS128_*, gen96_S*), DONE_SI.
SER=1
if [ -f /root/.boxsi_serial ] && [ "$(cat /root/.boxsi_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxsi_serial
echo "[si] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w
cat > /root/status.sh <<'ST'
grep -E "^\[(si|hub|math)\]|Error|Traceback" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-200
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.job ]; then touch /root/.job
  cat > /root/live.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
while [ ! -e DONE_SI ]; do
  { grep -E "^\[(si|hub|math)\]|Error|Traceback" w.log | tail -n 24; tail -n 3 sp.log 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; date -u; } > si_live.log 2>&1
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='si_live.log', path_in_repo='archive/sp3/si_live.log', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1
  sleep 600
done
JOB
  cat > /root/job.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
get() { for t in 1 2 3; do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o "$2" "$H/$1" && return 0; echo "[si] get $1 attempt $t failed"; sleep 10; done; return 1; }
acc() { python3 sp_mathcheck.py "$1" 2>/dev/null | grep -oP ": \K[0-9]+(?=/)" | head -1; }
for f in tools/sp_qwen3.py tools/sp_mathcheck.py ctl/math96.jsonl; do curl -sS -o $(basename $f) "$RAW/$f?nocache=$(date +%s)"; done
python3 -m pip install -q "huggingface_hub[hf_xet]" transformers bitsandbytes "math-verify[antlr4_13_2]" 2>&1 | grep -v -i warn | tail -1
( while [ ! -e dl.done ]; do sleep 120; echo "[si] downloads $(du -sh /root/.cache/huggingface 2>/dev/null | cut -f1) model, $(du -ch /root/w/*.pt /root/w/*.jsonl 2>/dev/null | tail -1 | cut -f1) files $(date -u +%H:%M)"; done ) &
t0=$(date +%s); python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-1.7B'))" > dl.log 2>&1
get archive/sp3/spG_longtraces_fft1024.pt spG.pt
get archive/sp3/longtr_math_train_16k.jsonl lt_math.jsonl; get archive/sp3/longtr_diverse_16k.jsonl lt_div.jsonl; get archive/sp3/longtr2_mixed_16k.jsonl lt2.jsonl
get archive/kvwin/q17_full_p.jsonl kv_full_p.jsonl; get archive/kvpool/traces.jsonl traces.jsonl
touch dl.done; echo "[si] downloads done in $(( $(date +%s) - t0 )) s: spG $(stat -c %s spG.pt) bytes $(date -u)"
python3 -c "
import json
rs = [json.loads(l) for f in ('lt_math.jsonl', 'lt_div.jsonl', 'lt2.jsonl') for l in open(f)]
ok = [r for r in rs if r.get('finish') == 'stop' and '</think>' in r['text']]
open('pool.jsonl', 'w').writelines(json.dumps(r) + '\n' for r in ok); print('[si] pool', len(ok), 'finished long traces')"
LG=/root/w/sp.log; base="--heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 384"
klm() { python3 sp_qwen3.py eval --traces kv_full_p.jsonl --max-len 4096 --S $1 $base --ckpt $2 --log $LG 2>&1 | grep -oP "KL \K[0-9.]+" | tail -1; }
w0() { python3 sp_qwen3.py profile --traces kv_full_p.jsonl --n 16 --max-len 4096 --S $1 $base --ckpt $2 --log $LG 2>&1 | grep -E "window +0:" | grep -oP "KL \K[0-9.]+" | head -1; }
echo "[math] S32 (G): deployment KL $(klm 32 spG.pt), window 0 $(w0 32 spG.pt)"
start=spG.pt
for S in 64 128; do
  python3 -c "
import random, sys; rs = open('pool.jsonl').readlines(); random.seed(int(sys.argv[1])); random.shuffle(rs); open('poolS.jsonl', 'w').writelines(rs)" $S
  N=$(wc -l < poolS.jsonl); off=0; seg=0; bad=0; best=9; cur=$start
  echo "[si] stage S$S from $start over $N traces $(date -u)"
  while [ $off -lt $N ] && [ $bad -lt 2 ]; do seg=$((seg + 1))
    python3 sp_qwen3.py train --traces poolS.jsonl --skip $off --n 600 --max-len 16384 --S $S $base --ckpt $cur --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 1e-4 --par --end-train --end-weight 4 --chunks-per-trace 8 --accum 8 --kl-rows 64 --seed $seg --out nxt.pt --log $LG > run_S${S}_$seg.log 2>&1
    grep -E "S grown|Error|Traceback" run_S${S}_$seg.log | tail -2 | sed "s/^/[si] S$S.$seg: /"; echo "[si] S$S.$seg: $(grep -c 'out of memory' run_S${S}_$seg.log) skipped (memory)"
    [ -s nxt.pt ] || { echo "[si] S$S.$seg: no checkpoint, stop stage"; break; }
    mv nxt.pt spS${S}_cur.pt; cur=spS${S}_cur.pt; off=$((off + 600))
    k=$(klm $S $cur); z=$(w0 $S $cur); echo "[math] S$S.$seg (traces $off): deployment KL $k, window 0 $z"
    if python3 -c "import sys; sys.exit(0 if $k < $best * 0.99 else 1)"; then best=$k; bad=0; cp $cur spS${S}_best.pt; echo "[si] S$S.$seg: new best"
    else bad=$((bad + 1)); python3 -c "import sys; sys.exit(0 if $k < $best else 1)" && { best=$k; cp $cur spS${S}_best.pt; }; fi
    put $LG spP_si.log
  done
  put spS${S}_best.pt spS${S}_best.pt
  G="--traces math96.jsonl --n 96 --max-len 12288 --S $S $base --roll-batch 48 --ckpt spS${S}_best.pt --log $LG"
  for sd in 1 2; do
    python3 sp_qwen3.py rollout $G --seed $sd --roll-out g96_S${S}_s$sd.jsonl > run_g96_S${S}_s$sd.log 2>&1
    python3 sp_mathcheck.py g96_S${S}_s$sd.jsonl | sed "s/^/[si] S$S seed $sd: /"; put g96_S${S}_s$sd.jsonl gen96_S${S}_s$sd.jsonl
  done
  echo "[si] stage S$S done: best deployment KL $best $(date -u)"; put $LG spP_si.log
  start=spS${S}_best.pt; rm -f spS${S}_cur.pt
done
date -u > DONE_SI; put DONE_SI DONE_SI; echo "[si] DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/live.sh' > /dev/null 2>&1 < /dev/null &
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[si] job launched $(date -u)"
fi
# CTL-END
