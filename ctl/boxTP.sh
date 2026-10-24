#!/bin/bash
# Box TP (3090 24 GB, pytorch image): the deep-charger recipe that made ap32_large (HF squash b5989f09: archive/train_attnpool.py,
# archive/HANDOFF.md), numbers moved to Qwen3-1.7B: model FROZEN, a 3-layer AttnPoolSP (S 32, 8 heads, ffn 2048) reads ALL past
# raw-token embeddings (no eviction), output KL on every chunk, raw window 128, chunk 64, 768-token contexts, 4000 contexts once,
# lr 3e-4. Two poolers side by side: the original (raw sinusoids, oldest-first) and pe-rev + pe-norm. Then held-out deep KL vs
# the equal-budget raw window (160) and the plain 128, the inference-time wider window (512 on 4096-token traces), and a
# generation check: 32 MATH problems, full KV vs SP + window 512 vs raw window 544 alone (accuracy, </think>, repetition).
SER=10
if [ -f /root/.boxtp_serial ] && [ "$(cat /root/.boxtp_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxtp_serial
echo "[tp] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w; RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i"
curl -sS -o /root/w/sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
cat > /root/status.sh <<'ST'
grep -E "^\[(tp|sp|hub|math|mix)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-220
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
# serial 10: stage R ran out of memory at ~850 traces (its 800-trace checkpoint already scores 0.0706 / 0.0621 under the stage-A
# conditions - level with the best so far). The trainer now skips a trace that does not fit. R2 = continue from spR.pt on the
# rest (trainR from 851, 1150 traces), then the same scores and the generation check (8192 tokens, rollout without the
# empty-buffer SP) at rw 1024.
if [ ! -e /root/.job10 ]; then touch /root/.job10
  cat > /root/job10.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 90; done; echo "[hub] $2 FAILED"; }
RAWB=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/tools
curl -sS -o sp_qwen3.py "$RAWB/sp_qwen3.py?nocache=$(date +%s)"
grep -q "out of memory, skipped" sp_qwen3.py && grep -q "sp_empty" sp_qwen3.py && echo "[tp] s10: trainer has the OOM skip + rollout fix" || echo "[tp] s10: WARNING old trainer"
LG=/root/w/sp.log; F="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 384"
echo "[tp] stage R2: continue from spR.pt $(date -u)"
python3 sp_qwen3.py train --traces trainR.jsonl --skip 850 --n 1150 --max-len 3072 $F --ckpt spR.pt --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 1e-4 --par --chunks-per-trace 4 --accum 8 --out spR2.pt --log $LG > run_trainR2.log 2>&1
grep -E "^\[sp\] (trained)|Error|Traceback|out of memory" run_trainR2.log | tail -3 | sed "s/^/[tp] R2: /"
put spR2.pt spR2_ap32recipe_fft1024.pt
for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 $F --ckpt spR2.pt --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] R2 stageA-cond: /"; done
put $LG spP.log
G="--traces kv_full_p.jsonl --n 32 --max-len 8192 --roll-batch 16 --S 32 --heads 8 --layers 3 --ffn 2048 --log $LG"
python3 sp_qwen3.py rollout $G --rw 1024 --C 128 --maxd 384 --ckpt spR2.pt --roll-out g8_R2.jsonl > run_gR2.log 2>&1
grep -E "Error|Traceback|out of memory" run_gR2.log | tail -2 | sed "s/^/[tp] /"
python3 sp_mathcheck.py g8_R2.jsonl | sed "s/^/[tp] /"
put g8_R2.jsonl ap32gen8_R2.jsonl; put $LG spP.log; date -u > DONE_TR2; put DONE_TR2 DONE_TR2; echo "[tp] DONE_TR2 $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job10.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  sed -e 's/DONE_TR /DONE_TR2 /' /root/live8.sh > /root/live10.sh; setsid nohup bash -c 'bash /root/live10.sh' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s10: stage R2 launched $(date -u)"
fi
# serial 9: two copies of the 8k generation job are running (the serial-6 one was not stopped before serial 7 relaunched it)
# - stop the newer copy and its current rollout; the shared output files keep both copies' rows (extra samples)
if [ ! -e /root/.dup9 ]; then touch /root/.dup9
  J=$(pgrep -f "^bash /root/job6.sh" | sort -n); echo "[tp] s9: job6 copies: $(echo $J | tr '\n' ' ')"
  if [ $(echo "$J" | wc -w) -ge 2 ]; then P=$(echo "$J" | tail -1); pkill -P $P; kill $P; echo "[tp] s9: stopped copy $P"; fi
fi
# serial 8: stage Q (FFT at rw 128) improved rw 128 slightly but HURT the deployment window (stage-A conditions 0.084 / 0.071
# -> 0.096 / 0.088): the model specialised to the narrow window. Stage R = the frozen recipe pooler (spP_orig) + all-layer FFT
# AT the deployment window, the way stage B did it (--par: every window 0..rw+C at 4 boundaries per trace, rw 1024, C 128,
# maxD 384, 8 traces per update), on traces nobody has seen (trainQ from line 1501), after the 8k generation check.
if [ ! -e /root/.job8 ]; then touch /root/.job8
  cat > /root/job8.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 90; done; echo "[hub] $2 FAILED"; }
while [ ! -e DONE_TG8 ]; do sleep 30; done
LG=/root/w/sp.log; F="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 384"
echo "[tp] stage R: recipe pooler + FFT at rw 1024 (--par) $(date -u)"
tail -n +1501 trainQ.jsonl > trainR.jsonl
python3 sp_qwen3.py train --traces trainR.jsonl --n 2000 --max-len 3072 $F --ckpt spP_orig.pt --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 1e-4 --par --chunks-per-trace 4 --accum 8 --out spR.pt --log $LG > run_trainR.log 2>&1
grep -E "^\[sp\] (trained)|Error|Traceback|out of memory" run_trainR.log | tail -3 | sed "s/^/[tp] R: /"
put spR.pt spR_ap32recipe_fft1024.pt
for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 $F --ckpt spR.pt --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] R stageA-cond: /"; done
python3 sp_qwen3.py profile --traces kv_full_p.jsonl --n 16 --max-len 4096 $F --ckpt spR.pt --log $LG 2>&1 | grep -E "^\[sp\] profile|Error|Traceback" | tail -7 | sed "s/^/[math] R /"
put $LG spP.log; date -u > DONE_TR; put DONE_TR DONE_TR; echo "[tp] DONE_TR $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job8.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  cat > /root/live8.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
while [ ! -e DONE_TR ]; do
  { grep -E "^\[(tp|hub|math|mix)\]" w.log | tail -n 24; tail -n 8 sp.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; date -u; } > tp_live.log 2>&1
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='tp_live.log', path_in_repo='archive/sp3/tp_live.log', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1
  sleep 600
done
JOB
  pkill -f live5.sh; setsid nohup bash -c 'bash /root/live8.sh' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s8: stage R queued after the 8k generation check; live log until DONE_TR $(date -u)"
fi
# serial 7: relaunch the (still waiting) 8k generation job with the full-KV run in batches of 8
if [ ! -e /root/.job7 ]; then touch /root/.job7
  pkill -f job6.sh; sleep 2
  sed -i 's#python3 sp_qwen3.py rollout $G --no-sp --rw 16384#python3 sp_qwen3.py rollout $G --roll-batch 8 --no-sp --rw 16384#' /root/job6.sh
  grep -q -- "--roll-batch 8 --no-sp --rw 16384" /root/job6.sh && echo "[tp] s7: job6 patched" || echo "[tp] s7: WARNING job6 not patched"
  setsid nohup bash -c 'bash /root/job6.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s7: 8k generation job relaunched $(date -u)"
fi
# serial 6: the 4096-token generation check is not usable (full KV left 19/32 thoughts unfinished; the window-only path
# stops early and "wins") - after stage Q, generate to 8192 tokens: full KV, SP (stage Q weights) at the deployment setting
# (rw 1024, C 128, maxD 384), the frozen recipe pooler the same way, and the equal-budget window alone (1056).
if [ ! -e /root/.job6 ]; then touch /root/.job6
  cat > /root/job6.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 90; done; echo "[hub] $2 FAILED"; }
while [ ! -e DONE_TQ ]; do sleep 30; done
LG=/root/w/sp.log; G="--traces kv_full_p.jsonl --n 32 --max-len 8192 --roll-batch 16 --S 32 --heads 8 --layers 3 --ffn 2048 --log $LG"
echo "[tp] generation check at 8192 tokens $(date -u)"
python3 sp_qwen3.py rollout $G --roll-batch 8 --no-sp --rw 16384 --C 512 --maxd 0 --roll-out g8_full.jsonl > run_g8.log 2>&1
python3 sp_qwen3.py rollout $G --rw 1024 --C 128 --maxd 384 --ckpt spQ.pt --roll-out g8_Q1024.jsonl >> run_g8.log 2>&1
python3 sp_qwen3.py rollout $G --rw 1024 --C 128 --maxd 384 --ckpt spP_orig.pt --roll-out g8_P1024.jsonl >> run_g8.log 2>&1
python3 sp_qwen3.py rollout $G --no-sp --rw 1056 --C 128 --maxd 0 --roll-out g8_raw1056.jsonl >> run_g8.log 2>&1
grep -E "Error|Traceback|out of memory" run_g8.log | tail -2
python3 sp_mathcheck.py g8_full.jsonl g8_Q1024.jsonl g8_P1024.jsonl g8_raw1056.jsonl | sed "s/^/[tp] /"
for g in full Q1024 P1024 raw1056; do put g8_$g.jsonl ap32gen8_$g.jsonl; done
put $LG spP.log; date -u > DONE_TG8; put DONE_TG8 DONE_TG8; echo "[tp] DONE_TG8 $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job6.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  sed -i 's/while \[ ! -e DONE_TQ \]/while [ ! -e DONE_TG8 ]/' /root/live5.sh 2>/dev/null
  echo "[tp] s6: 8192-token generation check queued $(date -u)"
fi
# serial 5: the live log stopped at DONE_TP - restart it until stage Q is done, following the FFT run's log
if [ ! -e /root/.live5 ]; then touch /root/.live5
  sed -e 's/while \[ ! -e DONE_TP \]/while [ ! -e DONE_TQ ]/' -e 's#tail -n 4 spP_orig.log; tail -n 4 spP_norm.log; tail -n 8 sp.log;#tail -n 12 sp.log;#' /root/live.sh > /root/live5.sh
  setsid nohup bash -c 'bash /root/live5.sh' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s5: live log restarted until DONE_TQ"
fi
# serial 4 (the user: if the loss does not get low enough with the model frozen, unfreeze it): after the frozen-model
# results (DONE_TP2), stage Q = the better recipe pooler + all-layer FFT (fp32 master, 8-bit AdamW), the same rw 128 / C 64,
# 3072-token contexts (distant up to ~2.9k, cap 4096 = no eviction in practice), 8 chunks per trace, 1500 traces nobody has
# seen (shards 2-5), teacher = the original model; then the same scores (deep KL at rw 128, rw 512, stage-A conditions) and
# the generation check (MATH 32: SP + rw 512 / 1024 vs full KV).
if [ ! -e /root/.job4 ]; then touch /root/.job4
  cat > /root/job4.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 90; done; echo "[hub] $2 FAILED"; }
while [ ! -e DONE_TP2 ]; do sleep 30; done
LG=/root/w/sp.log; L3="--S 32 --heads 8 --layers 3 --ffn 2048"
B=$(python3 - <<'PY'
import re
L=open("/root/w/sp.log").read()
def k(c):
    m=re.findall(r"%s: 64 traces \d+ tokens  KL ([0-9.]+)" % c, L); return float(m[0]) if m else 9
print("spP_orig.pt" if k("spP_orig.pt") <= k("spP_norm.pt") else "spP_norm.pt")
PY
)
PE=""; [ "$B" = spP_norm.pt ] && PE="--pe-rev --pe-norm"
echo "[tp] stage Q: FFT + pooler from $B $PE $(date -u)"
cat traces_big_2.jsonl traces_big_3.jsonl traces_big_4.jsonl traces_big_5.jsonl > trainQ.jsonl
python3 sp_qwen3.py train --traces trainQ.jsonl --n 1500 --max-len 3072 $L3 --rw 128 --C 64 --maxd 4096 --ckpt $B $PE --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 1e-4 --chunks-per-trace 8 --accum 1 --out spQ.pt --log $LG > run_trainQ.log 2>&1
grep -E "^\[sp\] (trained)|Error|Traceback|out of memory" run_trainQ.log | tail -3 | sed "s/^/[tp] Q: /"
put spQ.pt spQ_ap32recipe_fft.pt; put $LG spP.log
for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces $f $x --max-len 768 --score-from 224 --rw 128 $L3 --C 64 --maxd 0 --ckpt spQ.pt --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] Q rw128: /"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 --score-from 608 --rw 512 $L3 --C 64 --maxd 4096 --ckpt spQ.pt --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] Q rw512: /"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 --rw 1024 $L3 --C 128 --maxd 384 --ckpt spQ.pt --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] Q stageA-cond: /"; done
put $LG spP.log
G="--traces kv_full_p.jsonl --n 32 --max-len 4096 --roll-batch 16 $L3 --log $LG"
python3 sp_qwen3.py rollout $G --rw 512 --C 64 --maxd 4096 --ckpt spQ.pt --roll-out genQ_sp512.jsonl > run_gq.log 2>&1
python3 sp_qwen3.py rollout $G --rw 1024 --C 128 --maxd 384 --ckpt spQ.pt --roll-out genQ_sp1024.jsonl >> run_gq.log 2>&1
grep -E "Error|Traceback|out of memory" run_gq.log | tail -2
python3 sp_mathcheck.py gen_full.jsonl genQ_sp512.jsonl genQ_sp1024.jsonl | sed "s/^/[tp] /"
for g in sp512 sp1024; do put genQ_$g.jsonl ap32genQ_$g.jsonl; done
put $LG spP.log; date -u > DONE_TQ; put DONE_TQ DONE_TQ; echo "[tp] DONE_TQ $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job4.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s4: stage Q (FFT) queued after the frozen-model results $(date -u)"
fi
# serial 3 (the user: score them under the same conditions as the earlier 0.06-0.08 numbers): after the pipeline, the deployment
# loop exactly as stages A / B / W3 were scored - rw 1024, C 128, maxD 384, 4096-token traces, MATH 64 / mixed 60 - for both
# poolers, plus the no-SP equal-budget window (1056).
if [ ! -e /root/.job3 ]; then touch /root/.job3
  cat > /root/job3.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 90; done; echo "[hub] $2 FAILED"; }
while [ ! -e DONE_TP ]; do sleep 30; done
LG=/root/w/sp.log; Q="--S 32 --heads 8 --layers 3 --ffn 2048 --C 128 --maxd 384 --max-len 4096"
echo "[tp] same conditions as stage A (rw 1024, C 128, maxD 384, 4096 tokens) $(date -u)"
for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  for c in spP_orig.pt spP_norm.pt; do python3 sp_qwen3.py eval --traces $f $x --rw 1024 $Q --ckpt $c --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] /"; done
  python3 sp_qwen3.py eval --traces $f $x --rw 1056 $Q --no-sp --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] /"; done
put $LG spP.log; date -u > DONE_TP2; put DONE_TP2 DONE_TP2; echo "[tp] DONE_TP2 $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job3.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s3: same-condition scoring queued $(date -u)"
fi
# serial 2: the two trainers write their own logs - show both in the status line and the live log
if [ ! -e /root/.st2 ]; then touch /root/.st2
  sed -i 's#tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-220#for v in orig norm; do tail -n 1 /root/w/spP_$v.log 2>/dev/null | cut -c1-200; done; tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-200#' /root/status.sh
  sed -i 's#tail -n 12 sp.log;#tail -n 4 spP_orig.log; tail -n 4 spP_norm.log; tail -n 8 sp.log;#' /root/live.sh
  pkill -f live.sh; setsid nohup bash -c 'bash /root/live.sh' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s2: status/live follow both trainer logs"
fi
if [ ! -e /root/.job ]; then touch /root/.job
  cat > /root/live.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
while [ ! -e DONE_TP ]; do
  { grep -E "^\[(tp|hub|math|mix)\]" w.log | tail -n 20; tail -n 12 sp.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; df -h / | tail -1; date -u; } > tp_live.log 2>&1
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='tp_live.log', path_in_repo='archive/sp3/tp_live.log', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1
  sleep 600
done
JOB
  cat > /root/job.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; echo "[tp] put $2 attempt $t: $(tail -1 put.err | cut -c1-160)"; sleep 90; done; echo "[hub] $2 FAILED"; }
get() { for t in 1 2 3; do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o "$2" "$H/$1" && return 0; echo "[tp] get $1 attempt $t failed"; sleep 10; done; return 1; }
python3 -m pip install -q "huggingface_hub[hf_xet]" transformers bitsandbytes datasets 2>&1 | grep -v -i warn | tail -1
for t in 1 2 3; do
  x=""; [ $t -gt 1 ] && x="HF_HUB_DISABLE_XET=1"
  env $x timeout 600 python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-1.7B'))" > dl.log 2>&1 && break
  echo "[tp] model download attempt $t failed"
done
echo "[tp] model: $(tail -1 dl.log | cut -c1-120)"
get archive/kvpool/traces.jsonl traces.jsonl; get archive/kvwin/q17_full_p.jsonl kv_full_p.jsonl
for i in 0 1 2 3 4 5; do get archive/sp3/traces_big_$i.jsonl traces_big_$i.jsonl; done
curl -sS -o sp_mathcheck.py "$RAW/tools/sp_mathcheck.py?nocache=$(date +%s)" || curl -sS -o sp_mathcheck.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/tools/sp_mathcheck.py"
LG=/root/w/sp.log
P="--S 32 --heads 8 --layers 3 --ffn 2048 --C 64 --maxd 0"
echo "[tp] training two poolers (orig PE | pe-rev+pe-norm) $(date -u)"
python3 sp_qwen3.py train --traces traces_big_0.jsonl,traces_big_1.jsonl --n 4000 --max-len 768 --rw 128 $P --lr 3e-4 --accum 1 --out spP_orig.pt --log spP_orig.log > run_orig.log 2>&1 &
sleep 120
python3 sp_qwen3.py train --traces traces_big_0.jsonl,traces_big_1.jsonl --n 4000 --max-len 768 --rw 128 $P --pe-rev --pe-norm --lr 3e-4 --accum 1 --out spP_norm.pt --log spP_norm.log > run_norm.log 2>&1 &
wait
for v in orig norm; do grep -E "^\[sp\] trained|Error|Traceback|out of memory" run_$v.log | tail -2 | sed "s/^/[tp] $v: /"; done
put spP_orig.pt spP_orig_ap32recipe.pt; put spP_norm.pt spP_norm_ap32recipe.pt; cat spP_orig.log spP_norm.log >> $LG; put $LG spP.log
echo "[tp] deep KL on 768-token held-out traces (scored from 224) $(date -u)"
dk() { for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces $f $x --max-len 768 --score-from 224 "$@" --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] /"; done; }
dk --rw 160 $P --no-sp
dk --rw 128 $P --no-sp
dk --rw 128 $P --ckpt spP_orig.pt
dk --rw 128 $P --ckpt spP_norm.pt
put $LG spP.log
B=$(python3 - <<'PY'
import re
L=open("/root/w/sp.log").read()
def k(c):
    m=re.findall(r"rw128 C64 maxd0 L3 ffn2048[^\n]*%s: 64 traces \d+ tokens  KL ([0-9.]+)" % c, L); return float(m[-1]) if m else 9
print("spP_orig.pt" if k("spP_orig.pt") <= k("spP_norm.pt") else "spP_norm.pt")
PY
)
echo "[tp] better pooler: $B"
echo "[tp] wider window at inference: 4096-token traces, window 512, buffer cap 4096 $(date -u)"
for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 --score-from 608 --rw 512 --S 32 --heads 8 --layers 3 --ffn 2048 --C 64 --maxd 4096 --ckpt $B --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] /"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 --score-from 608 --rw 544 $P --no-sp --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] /"; done
put $LG spP.log
echo "[tp] generation check: 32 MATH problems, 4096 tokens $(date -u)"
G="--traces kv_full_p.jsonl --n 32 --max-len 4096 --roll-batch 16 --S 32 --heads 8 --layers 3 --ffn 2048 --log $LG"
python3 sp_qwen3.py rollout $G --no-sp --rw 8192 --C 512 --maxd 0 --roll-out gen_full.jsonl > run_gfull.log 2>&1
python3 sp_qwen3.py rollout $G --rw 512 --C 64 --maxd 4096 --ckpt $B --roll-out gen_sp512.jsonl > run_gsp.log 2>&1
python3 sp_qwen3.py rollout $G --no-sp --rw 544 --C 64 --maxd 0 --roll-out gen_raw544.jsonl > run_graw.log 2>&1
grep -h -E "Error|Traceback|out of memory" run_gfull.log run_gsp.log run_graw.log | tail -3
python3 sp_mathcheck.py gen_full.jsonl gen_sp512.jsonl gen_raw544.jsonl | sed "s/^/[tp] /"
for g in full sp512 raw544; do put gen_$g.jsonl ap32gen_$g.jsonl; done
put $LG spP.log; date -u > DONE_TP; put DONE_TP DONE_TP; echo "[tp] DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/live.sh' > /dev/null 2>&1 < /dev/null &
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tp] job launched $(date -u)"
fi
# CTL-END
