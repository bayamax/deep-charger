#!/bin/bash
# Box GM (one 24 GB card): more full-KV teacher traces (vLLM, Qwen3 thinking sampling, up to 16384 tokens) for the SP pooler
# - ctl/gen_more.jsonl (11660 prompts: MATH train L2-5 4000, competition 2884, MMLU-Pro 2494, UltraChat 2000, code_contests
# 282), in parts of 500; each finished part goes to archive/sp3/more/gm_NN.jsonl (the training box picks them up).
# Live log archive/sp3/gm_live.log, marker DONE_GM1.
SER=1
if [ -f /root/.boxgm_serial ] && [ "$(cat /root/.boxgm_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxgm_serial
echo "[gm] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w
cat > /root/status.sh <<'ST'
grep -E "^\[(gm|hub)\]|Error|Traceback" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
tail -n 1 /root/w/run_gen.log 2>/dev/null | cut -c1-200
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.job1 ]; then touch /root/.job1
  cat > /root/live.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
while [ ! -e DONE_GM1 ]; do
  { grep -E "^\[(gm|hub)\]|Error|Traceback" w.log | tail -n 20; tail -n 2 run_gen.log 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; date -u; } > gm_live.log 2>&1
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='gm_live.log', path_in_repo='archive/sp3/gm_live.log', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1
  sleep 600
done
JOB
  cat > /root/job.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
R=baya1116/hypernet-sp-distill
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
python3 -c "import torch; torch.zeros(1).cuda()" 2>/dev/null || { echo "[gm] GPU FAIL - stop"; exit 1; }
U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${U:-0}" -gt 2000 ] && { echo "[gm] GPU already holds $U MiB of someone else's - stop"; exit 1; }
for f in tools/gen_long_vllm.py ctl/gen_more.jsonl; do curl -sS -o $(basename $f) "$RAW/$f?nocache=$(date +%s)"; done
python3 -m pip install -q "huggingface_hub[hf_xet]" 2>&1 | tail -1
t0=$(date +%s); python3 -m venv /root/vv && /root/vv/bin/pip install -q vllm > vllm_install.log 2>&1
/root/vv/bin/python -c "import vllm; print('[gm] vllm', vllm.__version__)" 2>&1 | tail -1; echo "[gm] install $(( $(date +%s) - t0 )) s"
python3 -c "
rs = open('gen_more.jsonl').readlines()
for i in range(0, len(rs), 500): open(f'part_{i // 500:02d}.jsonl', 'w').writelines(rs[i:i + 500])"
echo "[gm] $(wc -l < gen_more.jsonl) prompts in $(ls part_*.jsonl | wc -l) parts $(date -u)"
for p in part_*.jsonl; do n=${p#part_}; n=${n%.jsonl}; t0=$(date +%s)
  /root/vv/bin/python gen_long_vllm.py $p gm_$n.jsonl 16384 250 $((100 + 10#$n)) > run_gen.log 2>&1
  echo "[gm] part $n: $(tail -n 1 run_gen.log | cut -c1-160) ($(( $(date +%s) - t0 )) s)"
  put gm_$n.jsonl more/gm_$n.jsonl
done
date -u > DONE_GM1; put DONE_GM1 DONE_GM1; echo "[gm] DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/live.sh' > /dev/null 2>&1 < /dev/null &
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[gm] job launched $(date -u)"
fi
# CTL-END
