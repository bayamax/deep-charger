#!/bin/bash
# Box VG (3060, pytorch image): the parallel SP-evict experiment on deep-charger's g14 (the model in the app, 16-bit here):
# N bounded streams as one batch vs one at a time vs full KV - tokens/s, peak memory, GSM8K accuracy - and the task split
# into parallel sub-streams. ctl/par_sp.py; results to the hub (archive/parsp/)
BOXVG_SERIAL=3
if [ -f /root/.boxvg_serial ] && [ "$(cat /root/.boxvg_serial)" -gt "$BOXVG_SERIAL" ] 2>/dev/null; then echo "BOXVG_STALE $BOXVG_SERIAL"; exit 0; fi
echo $BOXVG_SERIAL > /root/.boxvg_serial
if [ ! -e /root/.vg_pre ]; then touch /root/.vg_pre
  echo "[vg] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader)"
  echo "[vg] preflight hub: $(curl -sSL -o /dev/null -m 90 -w '%{size_download} bytes in %{time_total}s' https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/release/g14-bf16/model.safetensors -r 0-104857599 2>&1)"
fi
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
mkdir -p /root/vg
for try in 1 2 3; do curl -sS -o /root/vg/par_sp.py "$RAW/par_sp.py?nocache=$(date +%s)" && python3 -m py_compile /root/vg/par_sp.py && break; sleep 5; done
[ -f /root/.bootstrapped ] || { python3 -m pip install -q "transformers>=4.46" safetensors pyarrow "huggingface_hub[hf_xet]" 2>&1 | tail -1; touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"; }
cat > /root/status.sh <<'ST'
grep -E "^\[(vg|par|hub)\]|Error|Traceback|out of memory" /root/vg/vg.log 2>/dev/null | tail -n 6 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
# serial 2, the user: the question is whether SPLITTING a task and running the parts in parallel raises the score, with the
# split made by deep-charger itself, or by a large model when that is too hard for it. Job 2 waits for job 1, then runs the
# decomp arm on 48 bundles (24 of 3 and 24 of 5 GSM8K questions): planner none / g14 / api (DeepSeek's splits, in bundles.jsonl)
if [ ! -e /root/.vg_s2 ]; then touch /root/.vg_s2
  cat > /root/vgjob2.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vg
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/parsp/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
while [ ! -s DONE ]; do sleep 60; done
curl -sS -o par_sp.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/par_sp.py?nocache=$(date +%s)"
curl -sSfL -o bundles.jsonl https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/archive/parsp/bundles.jsonl
R="python3 -u par_sp.py --model g14 --gsm gsm8k_test.parquet --bundles bundles.jsonl --out results_decomp.jsonl --arm decomp --n 24"
echo "[vg] decomp arm: 48 tasks (24 x 3 questions, 24 x 5); one stream (gen 3000) vs the parts in parallel (gen 1024 each), split by g14 (plan 600) or by DeepSeek"
$R --planner none --gen 3000 --tag decomp 2>&1 | grep -E "^\[par\]|Error|Traceback"; put results_decomp.jsonl results_decomp.jsonl > /dev/null
$R --planner api --gen 3000 --gen-split 1024 --tag decomp 2>&1 | grep -E "^\[par\]|Error|Traceback"; put results_decomp.jsonl results_decomp.jsonl > /dev/null
$R --planner g14 --gen 3000 --gen-split 1024 --gen-plan 600 --tag decomp 2>&1 | grep -E "^\[par\]|Error|Traceback"
put results_decomp.jsonl results_decomp.jsonl; put /root/vg/vg.log run.log; date -u > DONE2; put DONE2 DONE2; echo "[vg] VGJOB2_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vgjob2.sh >> /root/vg/vg.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[vg] s2: job 2 (the decomp arm) queued after job 1 $(date -u)"
fi
# serial 3: the rebuild step computed the logits of the whole block (N x 800 x 152k), so n=32 went out of memory; par_sp.py
# now keeps only the last position's (re-fetched above, every later run uses it). Job 3 waits for job 2 and runs the two arms
# that fell over: sp n=32 and the long arm's sp n=16
if [ ! -e /root/.vg_s3 ]; then touch /root/.vg_s3
  cat > /root/vgjob3.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vg
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/parsp/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
while [ ! -s DONE2 ]; do sleep 60; done
curl -sS -o par_sp.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/par_sp.py?nocache=$(date +%s)"
R="python3 -u par_sp.py --model g14 --gsm gsm8k_test.parquet --out results.jsonl"
grep -q '"tag": "gsm", "mode": "sp", "n": 32' results.jsonl 2>/dev/null || grep -q '"mode": "sp", "n": 32, "gen": 1024' results.jsonl 2>/dev/null || $R --arm gsm --mode sp --n 32 --questions 64 --gen 1024 --tag gsm 2>&1 | grep -E "^\[par\]|Error|Traceback"
grep -q '"mode": "sp", "n": 16, "gen": 3000' results.jsonl 2>/dev/null || $R --arm gsm --mode sp --n 16 --questions 16 --gen 3000 --tag long 2>&1 | grep -E "^\[par\]|Error|Traceback"
put results.jsonl results.jsonl; put /root/vg/vg.log run.log; date -u > DONE3; put DONE3 DONE3; echo "[vg] VGJOB3_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vgjob3.sh >> /root/vg/vg.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[vg] s3: par_sp.py fixed (last logits only); job 3 (the runs that went out of memory) queued after job 2 $(date -u)"
fi
if [ ! -e /root/.vgjob ]; then touch /root/.vgjob
  cat > /root/vgjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vg
H=https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/release/g14-bf16
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/parsp/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
mkdir -p g14; for f in config.json generation_config.json pooler.safetensors pooler_config.json special_tokens_map.json tokenizer.json tokenizer_config.json model.safetensors; do [ -s g14/$f ] || curl -sSfL -o g14/$f $H/$f; done
[ -s gsm8k_test.parquet ] || curl -sSfL -o gsm8k_test.parquet "https://huggingface.co/api/datasets/openai/gsm8k/parquet/main/test/0.parquet"
R="python3 -u par_sp.py --model g14 --gsm gsm8k_test.parquet --out results.jsonl"
echo "[vg] gsm arm: 64 questions, gen 1024, sp vs full, n = 1 / 8 / 32"
for n in 32 8 1; do $R --arm gsm --mode sp --n $n --questions 64 --gen 1024 --tag gsm 2>&1 | grep -E "^\[par\]|Error|Traceback"; put results.jsonl results.jsonl > /dev/null; done
for n in 32 8; do $R --arm gsm --mode full --n $n --questions 64 --gen 1024 --tag gsm 2>&1 | grep -E "^\[par\]|Error|Traceback"; put results.jsonl results.jsonl > /dev/null; done
echo "[vg] long arm: 16 streams, gen 3000: the KV footprint, sp vs full"
$R --arm gsm --mode sp --n 16 --questions 16 --gen 3000 --tag long 2>&1 | grep -E "^\[par\]|Error|Traceback"
$R --arm gsm --mode full --n 16 --questions 16 --gen 3000 --tag long 2>&1 | grep -E "^\[par\]|Error|Traceback"
put results.jsonl results.jsonl > /dev/null
echo "[vg] split arm: 24 three-part problems, one stream (gen 2048) vs three parallel sub-streams (gen 1024 each)"
$R --arm split --mode sp --n 24 --problems 24 --gen 2048 --gen-split 1024 --tag split 2>&1 | grep -E "^\[par\]|Error|Traceback"
put results.jsonl results.jsonl; put /root/vg/vg.log run.log; date -u > DONE; put DONE DONE; echo "[vg] VGJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vgjob.sh >> /root/vg/vg.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VGJOB_LAUNCHED $(date -u)"
fi
echo "BOXVG_OK serial $BOXVG_SERIAL $(date -u)"
# CTL-END
