#!/bin/bash
# Box VR (3060, vLLM image): candidate bases for the app, as they come - Qwen3-1.7B and Qwen3-4B, thinking on, no search -
# answer the use-case bench's app kinds (197 real prompts); the replies go to archive/uc/ for the judge against 4o-mini.
BOXVR_SERIAL=6
if [ -f /root/.boxvr_serial ] && [ "$(cat /root/.boxvr_serial)" -gt "$BOXVR_SERIAL" ] 2>/dev/null; then echo "BOXVR_STALE $BOXVR_SERIAL"; exit 0; fi
echo $BOXVR_SERIAL > /root/.boxvr_serial
[ -e /root/.vr_pre ] || { touch /root/.vr_pre; echo "[vr] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"; }
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
for f in $(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv $f $f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/vr; RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i"
curl -sS -o /root/vr/uc_vllm.py "$RAW/tools/uc/uc_vllm.py?nocache=$(date +%s)"
[ -f /root/.bootstrapped ] || { python3 -m pip install -q "huggingface_hub[hf_xet]" 2>&1 | tail -1; touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"; }
cat > /root/status.sh <<'ST'
grep -E "^\[(vr|uc_vllm|hub|kvwin|kvfit|trace_gen|kvpool)\]|Error|Traceback|out of memory" /root/vr/vr.log 2>/dev/null | tail -n 5 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
# serial 6: job 6 after job 5 - the learned KV pooler (tools/kvpool.py), trained for token reproduction (KL to the full
# model) on 500 own traces; three (exposure window W, slots m, chunk C) points, each scored against dropping the evicted
# span (m=0) on held-out traces (MATH-500 64 + 60 mixed); then MATH-500 accuracy with the pooler in the sampling loop.
if [ ! -e /root/.vr_s6 ]; then touch /root/.vr_s6
  curl -sS -o /root/vr/kvpool.py "$RAW/tools/kvpool.py?nocache=$(date +%s)"
  cat > /root/vrjob6.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vr
R=baya1116/hypernet-sp-distill
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/kvpool/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
while [ ! -s DONE_VR5 ]; do sleep 60; done
fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3
K="python3 kvpool.py"; LG=/root/vr/kvpool.log
for cfg in "512 64 128" "256 128 128" "128 64 64"; do set -- $cfg; W=$1; M=$2; C=$3; T=W${W}_m${M}_C${C}
  for ev in kv_full.jsonl:math traces.jsonl:mix; do f=${ev%%:*}; n=${ev#*:}; x=""; [ $n = mix ] && x="--n 60"
    $K eval --traces $f $x --W $W --m 0 --C $C --max-len 4096 --log $LG 2>&1 | grep -E "^\[kvpool\]|Error|Traceback|out of memory" | sed "s/^/[$n] /"; done
  $K train --traces traces.jsonl --skip 60 --n 500 --max-len 3072 --W $W --m $M --C $C --lr 2e-3 --out pool_$T.safetensors --log $LG 2>&1 | grep -E "^\[kvpool\] trained|Error|Traceback|out of memory"
  put pool_$T.safetensors pool_$T.safetensors
  for ev in kv_full.jsonl:math traces.jsonl:mix; do f=${ev%%:*}; n=${ev#*:}; x=""; [ $n = mix ] && x="--n 60"
    $K eval --traces $f $x --W $W --m $M --C $C --max-len 4096 --ckpt pool_$T.safetensors --log $LG 2>&1 | grep -E "^\[kvpool\]|Error|Traceback|out of memory" | sed "s/^/[$n] /"; done
  put $LG kvpool.log
done
for spec in "512 0 128 drop" "512 64 128 pool_W512_m64_C128.safetensors"; do set -- $spec; ck=""; [ $4 != drop ] && ck="--ckpt $4"
  $K gen --traces none --W $1 --m $2 --C $3 $ck --gen-n 64 --gen-tokens 8000 --batch 32 --out gen_W$1_m$2.jsonl --log $LG 2>&1 | grep -E "^\[kvpool\] gen W|Error|Traceback|out of memory"
  put gen_W$1_m$2.jsonl gen_W$1_m$2.jsonl; put $LG kvpool.log
done
put /root/vr/vr.log run_vr6.log; date -u > DONE_VR6; put DONE_VR6 DONE_VR6; echo "[vr] VRJOB6_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vrjob6.sh >> /root/vr/vr.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[vr] s6: job 6 (kvpool train/eval/gen) queued after job 5 $(date -u)"
fi
# serial 5 (the user: the pooler's objective is token reproduction - output-level, like GPTQ matching outputs): job 5,
# the model's own full-KV traces for training a learned pooler (700 use-case prompts + 500 MATH train, thinking on).
if [ ! -e /root/.vr_s5 ]; then touch /root/.vr_s5
  curl -sS -o /root/vr/trace_gen.py "$RAW/tools/trace_gen.py?nocache=$(date +%s)"
  cat > /root/vrjob5.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vr
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/kvpool/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
curl -sSfL -o uc_pool.jsonl "$H/archive/uc/uc_pool.jsonl"
fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3
python3 trace_gen.py --model Qwen/Qwen3-1.7B --uc uc_pool.jsonl --n-uc 700 --n-math 500 --out traces.jsonl 2>&1 | grep -E "^\[trace_gen\]|Error|Traceback"
put traces.jsonl traces.jsonl
date -u > DONE_VR5; put DONE_VR5 DONE_VR5; echo "[vr] VRJOB5_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vrjob5.sh >> /root/vr/vr.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[vr] s5: job 5 (trace_gen 1200 traces) launched $(date -u)"
fi
# serial 4 (the user: rethink the pooler from scratch, GPTQ's idea - reproduce the FULL model's generation; pooler <= KV):
# job 4 after job 3, on job 3's own full-KV traces - which exposure window W and how many slots m can stand in for the
# evicted span, by selection, by a least-squares fit on the window's queries, on a query bank, on all past queries, and
# the oracle fit on the future queries (capacity bound). Teacher-forced KL / top-1 agreement vs full KV.
if [ ! -e /root/.vr_s4 ]; then touch /root/.vr_s4
  curl -sS -o /root/vr/kvfit_study.py "$RAW/tools/kvfit_study.py?nocache=$(date +%s)"
  cat > /root/vrjob4.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vr
R=baya1116/hypernet-sp-distill
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/kvwin/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
python3 -c "import scipy" 2>/dev/null || python3 -m pip install -q scipy 2>&1 | grep -v -i warn | tail -1
while [ ! -s DONE_VR3 ]; do sleep 60; done
fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3
python3 kvfit_study.py --model Qwen/Qwen3-1.7B --traces kv_full.jsonl --n-traces 24 --cuts 1024,2048,3072 --N 256 --windows 128,256,512,1024 --slots 16,64 --bank 512 --out kvfit.jsonl 2>&1 | grep -E "^\[kvfit\]|Error|Traceback|out of memory"
put kvfit.jsonl q17_kvfit.jsonl
put /root/vr/vr.log run_vr4.log; date -u > DONE_VR4; put DONE_VR4 DONE_VR4; echo "[vr] VRJOB4_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vrjob4.sh >> /root/vr/vr.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[vr] s4: job 4 (kvfit study, Qwen3-1.7B) queued after job 3 $(date -u)"
fi
# serial 3 (the user: not flagship-only - the floor is a 4 GB phone, ~1.3-1.5 GB for the whole app -> Qwen3-1.7B
# at 4 bits is the ceiling; 8-9B are out): job 2 is cancelled. Job 3 after job 1: long CoT under a bounded KV cache
# WITHOUT the hypernet (prompt + last R tokens, the rest dropped) vs full KV - the floor the new pooler has to beat.
if [ ! -e /root/.vr_s3 ]; then touch /root/.vr_s3
  [ -s /root/vr/DONE_VR2 ] || { pkill -f vrjob2.sh; echo "[vr] s3: job 2 (8-9B) cancelled $(date -u)"; }
  curl -sS -o /root/vr/kvwin_eval.py "$RAW/tools/kvwin_eval.py?nocache=$(date +%s)"
  cat > /root/vrjob3.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vr
R=baya1116/hypernet-sp-distill
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/kvwin/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
python3 -m pip install -q datasets "math-verify[antlr4_13_2]" 2>&1 | grep -v -i warn | tail -1
while [ ! -s DONE_VR ]; do sleep 60; done
M=Qwen/Qwen3-1.7B
fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3
python3 kvwin_eval.py --engine vllm --model $M --n 64 --gen 8000 --out kv_full.jsonl 2>&1 | grep -E "^\[kvwin\]|Error|Traceback"
put kv_full.jsonl q17_full.jsonl
for spec in 768:32 2048:16; do r=${spec%%:*}; b=${spec#*:}
  fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3
  python3 kvwin_eval.py --engine hf --recent $r --batch $b --model $M --n 64 --gen 8000 --out kv_r$r.jsonl 2>&1 | grep -E "^\[kvwin\]|Error|Traceback|out of memory"
  put kv_r$r.jsonl q17_r$r.jsonl
done
put /root/vr/vr.log run_vr3.log; date -u > DONE_VR3; put DONE_VR3 DONE_VR3; echo "[vr] VRJOB3_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vrjob3.sh >> /root/vr/vr.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[vr] s3: job 3 (Qwen3-1.7B full KV vs naive window 768/2048, MATH-500 L4-5 x64) queued after job 1 $(date -u)"
fi
# serial 2 (the user: the largest model the app can hold, made KV-light by the hypernet): job 2 after job 1 - the 8-9B
# class at 4 bits, Qwen3-8B (AWQ, ~4.6 GB at 4 bits on the phone) and Qwen3.5-9B (AWQ; hybrid attention, 8 of 32 layers
# keep a KV cache), the same 197 prompts, thinking on, no search.
if [ ! -e /root/.vr_s2 ]; then touch /root/.vr_s2
  cat > /root/vrjob2.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vr
R=baya1116/hypernet-sp-distill
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/uc/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
while [ ! -s DONE_VR ]; do sleep 60; done
for spec in q8:Qwen/Qwen3-8B-AWQ q35_9:QuantTrio/Qwen3.5-9B-AWQ; do t=${spec%%:*}; m=${spec#*:}
  fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3
  python3 uc_vllm.py --model $m --inp uc_app.jsonl --out ans_$t.jsonl --max-tokens 4096 --max-len 8192 --gpu-mem 0.92 2>&1 | grep -E "^\[uc_vllm\]|Error|Traceback"
  put ans_$t.jsonl ans_$t.jsonl
done
put /root/vr/vr.log run_vr2.log; date -u > DONE_VR2; put DONE_VR2 DONE_VR2; echo "[vr] VRJOB2_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vrjob2.sh >> /root/vr/vr.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[vr] s2: job 2 (Qwen3-8B, Qwen3.5-9B) queued after job 1 $(date -u)"
fi
if [ ! -e /root/.vrjob ]; then touch /root/.vrjob
  cat > /root/vrjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vr
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/uc/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
curl -sSfL -o uc_app.jsonl "$H/archive/uc/uc_app.jsonl"
for spec in q17:Qwen/Qwen3-1.7B q4:Qwen/Qwen3-4B; do t=${spec%%:*}; m=${spec#*:}
  fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3
  python3 uc_vllm.py --model $m --inp uc_app.jsonl --out ans_$t.jsonl --max-tokens 4096 --max-len 8192 2>&1 | grep -E "^\[uc_vllm\]|Error|Traceback"
  put ans_$t.jsonl ans_$t.jsonl
done
put /root/vr/vr.log run_vr.log; date -u > DONE_VR; put DONE_VR DONE_VR; echo "[vr] VRJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vrjob.sh >> /root/vr/vr.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VRJOB_LAUNCHED $(date -u)"
fi
# CTL-END
