#!/bin/bash
# Box VJ (3060, pytorch image): self-consistency on deep-charger's g14 - the same GSM8K question as K sampled streams run in
# parallel (bounded SP-evict KV), the answer the majority vote: does the parallelism buy accuracy where the task split did not?
# 64 questions, gen 1024, n=32: K=1 (one sample) vs K=8 (majority / any right). ctl/par_sp.py --votes; results to archive/parsp/
BOXVJ_SERIAL=1
if [ -f /root/.boxvj_serial ] && [ "$(cat /root/.boxvj_serial)" -gt "$BOXVJ_SERIAL" ] 2>/dev/null; then echo "BOXVJ_STALE $BOXVJ_SERIAL"; exit 0; fi
echo $BOXVJ_SERIAL > /root/.boxvj_serial
if [ ! -e /root/.vj_pre ]; then touch /root/.vj_pre
  echo "[vj] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader)"
  echo "[vj] preflight hub: $(curl -sSL -o /dev/null -m 90 -w '%{size_download} bytes in %{time_total}s' https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/release/g14-bf16/model.safetensors -r 0-104857599 2>&1)"
fi
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
mkdir -p /root/vj
for try in 1 2 3; do curl -sS -o /root/vj/par_sp.py "$RAW/par_sp.py?nocache=$(date +%s)" && python3 -m py_compile /root/vj/par_sp.py && break; sleep 5; done
[ -f /root/.bootstrapped ] || { python3 -m pip install -q "transformers>=4.46" safetensors pyarrow "huggingface_hub[hf_xet]" 2>&1 | tail -1; touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"; }
cat > /root/status.sh <<'ST'
grep -E "^\[(vj|par|hub)\]|Error|Traceback|out of memory" /root/vj/vj.log 2>/dev/null | tail -n 6 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.vjjob ]; then touch /root/.vjjob
  cat > /root/vjjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vj
H=https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/release/g14-bf16
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/parsp/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
mkdir -p g14; for f in config.json generation_config.json pooler.safetensors pooler_config.json special_tokens_map.json tokenizer.json tokenizer_config.json model.safetensors; do [ -s g14/$f ] || curl -sSfL -o g14/$f $H/$f; done
[ -s gsm8k_test.parquet ] || curl -sSfL -o gsm8k_test.parquet "https://huggingface.co/api/datasets/openai/gsm8k/parquet/main/test/0.parquet"
R="python3 -u par_sp.py --model g14 --gsm gsm8k_test.parquet --out results_votes.jsonl --arm gsm --mode sp --n 32 --questions 64 --gen 1024"
echo "[vj] votes arm: 64 questions, gen 1024, sp n=32: K=8 sampled streams per question, majority vote (and K=1 again for the same seed)"
$R --votes 8 --tag votes8 2>&1 | grep -E "^\[par\]|Error|Traceback"; put results_votes.jsonl results_votes.jsonl > /dev/null
$R --votes 1 --tag votes1 2>&1 | grep -E "^\[par\]|Error|Traceback"
put results_votes.jsonl results_votes.jsonl; put /root/vj/vj.log run_votes.log; date -u > DONE; put DONE DONE_VOTES; echo "[vj] VJJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vjjob.sh >> /root/vj/vj.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VJJOB_LAUNCHED $(date -u)"
fi
# CTL-END
