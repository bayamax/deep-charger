#!/bin/bash
# Box LT (a 3090, vLLM only): new long teacher traces - the full-KV Qwen3-1.7B on 1421 unseen prompts (ctl/longtr2.jsonl:
# MATH-train 500, open-domain reasoning 396, MMLU-Pro 215, instructions 214, programming 96) to 16384 tokens, Qwen3 thinking
# sampling, written in groups of 200 (archive/sp3/longtr2_mixed_16k.jsonl), DONE_LT at the end. Box TP's stage H2 waits for it.
SER=1
if [ -f /root/.boxlt_serial ] && [ "$(cat /root/.boxlt_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxlt_serial
echo "[lt] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w
cat > /root/status.sh <<'ST'
grep -E "^\[(lt|hub|gen)\]|Error|Traceback" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.job ]; then touch /root/.job
  cat > /root/live.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
while [ ! -e DONE_LT ]; do
  { grep -E "^\[(lt|hub|gen)\]|Error|Traceback" w.log | tail -n 12; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; date -u; } > lt_live.log 2>&1
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='lt_live.log', path_in_repo='archive/sp3/lt_live.log', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1
  sleep 600
done
JOB
  cat > /root/job.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
python3 -m pip install -q "huggingface_hub[hf_xet]" 2>&1 | grep -v -i warn | tail -1
python3 -m venv /root/vv && /root/vv/bin/pip install -q vllm > vllm_install.log 2>&1; /root/vv/bin/python -c "import vllm; print('[lt] vllm', vllm.__version__)" 2>&1 | tail -1
curl -sS -o longtr2.jsonl "$RAW/ctl/longtr2.jsonl?nocache=$(date +%s)"; curl -sS -o gen_long_vllm.py "$RAW/tools/gen_long_vllm.py?nocache=$(date +%s)"
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o lt2tr.jsonl "$H/archive/sp3/longtr2_mixed_16k.jsonl" 2>/dev/null || rm -f lt2tr.jsonl
echo "[lt] $(wc -l < longtr2.jsonl) prompts, $(wc -l < lt2tr.jsonl 2>/dev/null || echo 0) already done $(date -u)"
( while pgrep -f gen_long_vllm.py > /dev/null || [ ! -s lt2tr.jsonl ]; do sleep 900; [ -s lt2tr.jsonl ] && put lt2tr.jsonl longtr2_mixed_16k.jsonl > /dev/null; done ) &
/root/vv/bin/python gen_long_vllm.py longtr2.jsonl lt2tr.jsonl 16384 200 > run_gen.log 2>&1
grep -E "^\[gen\]|Error|Traceback" run_gen.log | tail -2 | sed "s/^/[lt] /"
put lt2tr.jsonl longtr2_mixed_16k.jsonl; date -u > DONE_LT; put DONE_LT DONE_LT; echo "[lt] DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/live.sh' > /dev/null 2>&1 < /dev/null &
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[lt] job launched $(date -u)"
fi
# CTL-END
