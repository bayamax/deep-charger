# box C: launch GRPO for the pooler lineage (teacher recipe + compression) from the distilled model.
#   trainer  : /root/work/grpo_pool.py  (see its docstring)   out: /root/grpo_pool/{grpo.log,rollouts.jsonl,latest.safetensors,state.json}
#   data     : the teacher's 2857-question pool (HF box_recover/corpus.jsonl) minus the held-out 300
# Safe to re-run: never relaunches while a trainer is alive; a dead trainer is NOT auto-restarted (OOM rule) - look first.
cd /root/work
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
if ! pgrep -f "/root/ctl\.s[h]" >/dev/null; then
  if [ -f /root/start_ctl.sh ]; then setsid nohup bash /root/start_ctl.sh > /dev/null 2>&1 < /dev/null & else setsid nohup bash /root/ctl.sh > /dev/null 2>&1 < /dev/null & fi
  sleep 2; echo "ctl revived: $(pgrep -fc '/root/ctl\.s[h]')"
fi
pkill -f "pool_eva[l].py" && echo "leftover evaluator killed"
python3 -c "import transformers.modeling_utils" 2>/dev/null || pip install -q "huggingface_hub>=0.34,<1.0" 2>&1 | tail -1
if [ ! -s /root/work/corpus_box_final.jsonl ]; then
  curl -sSL --retry 3 -o /root/work/corpus_box_final.jsonl "https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/box_recover/corpus.jsonl"
  echo "corpus downloaded: $(wc -l < /root/work/corpus_box_final.jsonl) rows"
fi
curl -sS --retry 3 -o /root/work/grpo_pool.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/grpo_pool.py?nocache=$(date +%s)"
python3 -m py_compile /root/work/grpo_pool.py && echo "trainer fetched: $(wc -l < /root/work/grpo_pool.py) lines"
echo "gpu: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader) | disk: $(df -h /root | awk 'NR==2{print $4}') free"
mkdir -p /root/grpo_pool
if ! pgrep -f "grpo_poo[l].py" >/dev/null; then
  export SP_BASE=/root/fft_hf SP_RANK=128 SP_NOSYS=1 SP_EPISODIC=1 SP_HOTPOT2=0 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  echo "=== LAUNCH $(date -u) ===" >> /root/grpo_pool.log
  setsid nohup python3 /root/work/grpo_pool.py /root/fft_new_all.safetensors /root/grpo_pool --steps 200 --g 12 --rw 768 --maxd 384 >> /root/grpo_pool.log 2>&1 < /dev/null &
  echo "trainer launched"
else
  echo "trainer already running"
fi
cat > /usr/local/bin/t <<'TT'
#!/bin/bash
echo "$(date -u +%H:%M)Z  grpo $(pgrep -fc 'grpo_poo[l].py')  ctl $(pgrep -fc '/root/ctl\.s[h]')  gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
grep "^\[step" /root/grpo_pool.log | tail -3 | sed 's/ landed=[0-9]*%//;s/ more=[0-9.]*//;s/ |grad|=[0-9.]*//;s/ skip=[01]//' | cut -c1-90
grep -i "error\|Traceback\|Killed\|GRPO_POOL_DONE" /root/grpo_pool.log | tail -2 | cut -c1-120
TT
chmod +x /usr/local/bin/t
cat > /root/status_pub.sh <<'SP'
#!/bin/bash
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
while true; do
  { echo "=== $(date -u) ==="; t 2>/dev/null; echo "--- last log lines"; tail -3 /root/grpo_pool.log | cut -c1-200; } > /root/work/status.txt 2>&1
  echo "--- STATUS $(date -u +%H:%M) ---"; cat /root/work/status.txt
  hf upload baya1116/hypernet-sp-distill /root/work/status.txt pooler_distill/status.txt >/dev/null 2>&1
  sleep 300
done
SP
chmod +x /root/status_pub.sh
for i in 1 2 3 4 5 6; do curl -sS -o /root/work/gold_pooled.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/gold_pooled.py?nocache=$(date +%s)" && python3 -m py_compile /root/work/gold_pooled.py && break; sleep 5; done
for f in "pooleval_all_c.jsonl 768" "pooleval_all_nc.jsonl 768" "pooleval_post.jsonl 512" "/root/grpo_pool/rollouts.jsonl 768"; do set -- $f; p=$1; [ "${p#/}" = "$p" ] && p=/root/work/$p; python3 /root/work/gold_pooled.py $p $2 2>&1 | grep -v Warning | cut -c1-300; done
pkill -f "status_pub"; pkill -f "status_pu[b].sh"; setsid nohup bash /root/status_pub.sh >> /proc/1/fd/1 2>&1 < /dev/null &
echo "--- log tail"; tail -4 /root/grpo_pool.log | cut -c1-220
echo "GP2"; echo "LAUNCH_DONE $(date -u)"
