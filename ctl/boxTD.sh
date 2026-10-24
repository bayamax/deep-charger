#!/bin/bash
# Box TD (3090 24 GB, pytorch image): DAgger, the step the original deep-charger had (train_sched_evict.py) and this line
# skipped - train on the bounded model's OWN SP-evict generations. Round r: the current model (start: W2, window-0 2.70,
# deployment MATH 0.091 / mixed 0.083) generates 384 rollouts through the SP-evict loop (rw 1024, C 128, sampled), then the
# pooler + all layers are trained on them against the original full-KV model (--par, every window at 4 boundaries per
# rollout), then held-out profile + deployment loop. Round 2 only if round 1 helps. Live log on the hub every 10 min.
SER=1
if [ -f /root/.boxtd_serial ] && [ "$(cat /root/.boxtd_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxtd_serial
echo "[td] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w; RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i"
curl -sS -o /root/w/sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
cat > /root/status.sh <<'ST'
grep -E "^\[(td|sp|hub|math|mix)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-220
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.job ]; then touch /root/.job
  cat > /root/live.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
while [ ! -e DONE_TD ]; do
  { grep -E "^\[(td|hub|math|mix)\]" w.log | tail -n 20; tail -n 12 sp.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; df -h / | tail -1; date -u; } > td_live.log 2>&1
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='td_live.log', path_in_repo='archive/sp3/td_live.log', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1
  sleep 600
done
JOB
  cat > /root/job.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
get() { for t in 1 2 3; do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o "$2" "$H/$1" && return 0; echo "[td] get $1 attempt $t failed"; sleep 10; done; return 1; }
python3 -m pip install -q "huggingface_hub[hf_xet]" transformers bitsandbytes datasets 2>&1 | grep -v -i warn | tail -1
for t in 1 2 3; do
  x=""; [ $t -gt 1 ] && x="HF_HUB_DISABLE_XET=1"
  env $x timeout 600 python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-1.7B'))" > dl.log 2>&1 && break
  echo "[td] model download attempt $t failed"
done
echo "[td] model: $(tail -1 dl.log | cut -c1-120)"
get archive/kvpool/traces.jsonl traces.jsonl; get archive/kvwin/q17_full_p.jsonl kv_full_p.jsonl
for i in 0 1 2 3 4 5; do get archive/sp3/traces_big_$i.jsonl traces_big_$i.jsonl; done
get archive/sp3/spW2_wincurr1024_fft.pt spW2.pt; echo "[td] start ckpt $(stat -c %s spW2.pt) bytes"
cat traces_big_[012].jsonl | tail -n +3201 > restA.jsonl; cat restA.jsonl traces_big_3.jsonl traces_big_4.jsonl traces_big_5.jsonl > trainB.jsonl
{ cat traces_big_[012].jsonl | head -n 3200; head -n 5425 trainB.jsonl; } > trainE.jsonl
{ sed -n '701,2000p' trainE.jsonl; tail -n +2801 trainE.jsonl; } > trainW.jsonl
echo "[td] data: trainW $(wc -l < trainW.jsonl) (expect 7125); rollout prompts from line 2401"
LG=/root/w/sp.log; F="--S 32 --rw 1024 --C 128 --maxd 384 --layers 1 --ffn 1024"
score() { python3 sp_qwen3.py profile --traces kv_full_p.jsonl --n 16 --max-len 4096 $F --ckpt $1 --log $LG 2>&1 | grep -E "^\[sp\] profile|Error|Traceback" | tail -7 | sed "s/^/[math] /"
  for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
    python3 sp_qwen3.py eval --traces $f $x --max-len 4096 $F --ckpt $1 --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback|out of memory" | sed "s/^/[$n] /"; done; }
CK=spW2.pt
for R in 1 2; do
  SKIP=$(( 2400 + (R - 1) * 384 ))
  echo "[td] round $R: rollouts from $CK on prompts $SKIP+ $(date -u)"
  python3 sp_qwen3.py rollout --traces trainW.jsonl --skip $SKIP --n 384 --max-len 3072 $F --ckpt $CK --roll-batch 24 --roll-out roll$R.jsonl --log $LG > run_roll$R.log 2>&1
  grep -E "Error|Traceback|out of memory" run_roll$R.log | tail -2; echo "[td] round $R: $(wc -l < roll$R.jsonl) rollouts"
  put roll$R.jsonl dagger_roll$R.jsonl
  echo "[td] round $R: training on them $(date -u)"
  python3 sp_qwen3.py train --traces roll$R.jsonl --max-len 3072 $F --ckpt $CK --pe-rev --pe-norm --ft-layers 28 --opt8bit --ft-lr 5e-6 --lr 2e-4 --par --chunks-per-trace 4 --accum 4 --out spD$R.pt --log $LG > run_trainD$R.log 2>&1
  grep -E "^\[sp\] (train .*: [0-9]+ traces|trained)|Error|Traceback|out of memory" run_trainD$R.log | tail -3
  score spD$R.pt
  put spD$R.pt dagger_r$R.pt; put $LG dagger.log
  MATH=$(grep -o "spD$R.pt: 64 traces [0-9]* tokens  KL [0-9.]*" $LG | tail -1 | awk '{print $NF}')
  echo "[td] round $R MATH KL $MATH (W2 0.0906)"
  if [ $R = 1 ] && python3 -c "import sys; sys.exit(0 if float('${MATH:-9}') < 0.0906 else 1)"; then CK=spD$R.pt; else break; fi
done
put $LG dagger.log; date -u > DONE_TD; put DONE_TD DONE_TD; echo "[td] DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/live.sh' > /dev/null 2>&1 < /dev/null &
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[td] job launched $(date -u)"
fi
# CTL-END
