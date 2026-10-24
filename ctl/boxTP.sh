#!/bin/bash
# Box TP (3090 24 GB, pytorch image): the deep-charger recipe that made ap32_large (HF squash b5989f09: archive/train_attnpool.py,
# archive/HANDOFF.md), numbers moved to Qwen3-1.7B: model FROZEN, a 3-layer AttnPoolSP (S 32, 8 heads, ffn 2048) reads ALL past
# raw-token embeddings (no eviction), output KL on every chunk, raw window 128, chunk 64, 768-token contexts, 4000 contexts once,
# lr 3e-4. Two poolers side by side: the original (raw sinusoids, oldest-first) and pe-rev + pe-norm. Then held-out deep KL vs
# the equal-budget raw window (160) and the plain 128, the inference-time wider window (512 on 4096-token traces), and a
# generation check: 32 MATH problems, full KV vs SP + window 512 vs raw window 544 alone (accuracy, </think>, repetition).
SER=3
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
