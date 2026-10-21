#!/bin/bash
# Box VI (3060, pytorch image): the two runs box VG lost when its watcher deleted it early (the decomp arm with g14 as the planner, and sp n=32); results to archive/parsp/. Was: the parallel SP-evict experiment on deep-charger's g14 (the model in the app, 16-bit here):
# N bounded streams as one batch vs one at a time vs full KV - tokens/s, peak memory, GSM8K accuracy - and the task split
# into parallel sub-streams. ctl/par_sp.py; results to the hub (archive/parsp/)
BOXVI_SERIAL=3
if [ -f /root/.boxvi_serial ] && [ "$(cat /root/.boxvi_serial)" -gt "$BOXVI_SERIAL" ] 2>/dev/null; then echo "BOXVI_STALE $BOXVI_SERIAL"; exit 0; fi
echo $BOXVI_SERIAL > /root/.boxvi_serial
if [ ! -e /root/.vi_pre ]; then touch /root/.vi_pre
  echo "[vi] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader)"
  echo "[vi] preflight hub: $(curl -sSL -o /dev/null -m 90 -w '%{size_download} bytes in %{time_total}s' https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/release/g14-bf16/model.safetensors -r 0-104857599 2>&1)"
fi
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
mkdir -p /root/vi
for try in 1 2 3; do curl -sS -o /root/vi/par_sp.py "$RAW/par_sp.py?nocache=$(date +%s)" && python3 -m py_compile /root/vi/par_sp.py && break; sleep 5; done
[ -f /root/.bootstrapped ] || { python3 -m pip install -q "transformers>=4.46" safetensors pyarrow "huggingface_hub[hf_xet]" 2>&1 | tail -1; touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"; }
cat > /root/status.sh <<'ST'
grep -E "^\[(vg|par|hub)\]|Error|Traceback|out of memory" /root/vi/vi.log 2>/dev/null | tail -n 6 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.vijob ]; then touch /root/.vijob
  cat > /root/vijob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vi
H=https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/release/g14-bf16
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/parsp/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
mkdir -p g14; for f in config.json generation_config.json pooler.safetensors pooler_config.json special_tokens_map.json tokenizer.json tokenizer_config.json model.safetensors; do [ -s g14/$f ] || curl -sSfL -o g14/$f $H/$f; done
[ -s gsm8k_test.parquet ] || curl -sSfL -o gsm8k_test.parquet "https://huggingface.co/api/datasets/openai/gsm8k/parquet/main/test/0.parquet"
curl -sSfL -o bundles.jsonl https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/archive/parsp/bundles.jsonl
R="python3 -u par_sp.py --model g14 --gsm gsm8k_test.parquet --bundles bundles.jsonl"
echo "[vi] decomp arm, planner g14: 48 tasks, the model splits them itself (plan 600 tokens), the parts in parallel (gen 1024)"
$R --out results_decomp2.jsonl --arm decomp --n 24 --planner g14 --gen 3000 --gen-split 1024 --gen-plan 600 --tag decomp 2>&1 | grep -E "^\[par\]|Error|Traceback"
put results_decomp2.jsonl results_decomp2.jsonl
echo "[vi] gsm arm, sp n=32 (the run that went out of memory on box VG)"
$R --out results2.jsonl --arm gsm --mode sp --n 32 --questions 64 --gen 1024 --tag gsm 2>&1 | grep -E "^\[par\]|Error|Traceback"
put results2.jsonl results2.jsonl; put /root/vi/vi.log run_vi.log; date -u > DONE_VI; put DONE_VI DONE_VI; echo "[vi] VIJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vijob.sh >> /root/vi/vi.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VIJOB_LAUNCHED $(date -u)"
fi
echo "BOXVI_OK serial $BOXVI_SERIAL $(date -u)"
# CTL-END
