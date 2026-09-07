# box C: (0) revive the ctl poll loop if it died (so later pushes to this file are picked up again),
#        (1) relaunch the 300-question evals of the new model if they are not running,
#        (2) publish a status file to HF every 5 min (pooler_distill/status.txt) so progress can be read without the log API.
#        (1b) fix the huggingface-hub 1.x import breakage that killed the 10:23 relaunch
# Safe to run repeatedly (ctl or by hand): nothing is duplicated.
cd /root/work
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
if ! pgrep -f "/root/ctl\.s[h]" >/dev/null; then
  if [ -f /root/start_ctl.sh ]; then setsid nohup bash /root/start_ctl.sh > /dev/null 2>&1 < /dev/null & else setsid nohup bash /root/ctl.sh > /dev/null 2>&1 < /dev/null & fi
  sleep 2; echo "ctl revived: $(pgrep -fc '/root/ctl\.s[h]')"
else
  echo "ctl alive"
fi
cat > /root/status_pub.sh <<'SP'
#!/bin/bash
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
while true; do
  { echo "=== $(date -u) ==="
    echo "ctl: $(pgrep -fc '/root/ctl\.s[h]')  evals: $(pgrep -fc 'pool_eva[l].py')  gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)  disk: $(df -h /root | awk 'NR==2{print $4}')"
    for m in all_c all_nc; do
      echo "[$m] rows=$(wc -l < /root/work/pooleval_$m.jsonl 2>/dev/null)  $(grep -o '\[[0-9]*/300\].*' /root/pooleval_$m.log 2>/dev/null | tail -1 | cut -c1-120)"
      grep -i "error\|Traceback\|Killed" /root/pooleval_$m.log 2>/dev/null | tail -2
    done
    t 2>/dev/null; tc 2>/dev/null
    echo "--- last eval log lines"; tail -2 /root/pooleval_all_c.log 2>/dev/null | cut -c1-200; tail -2 /root/pooleval_all_nc.log 2>/dev/null | cut -c1-200
  } > /root/work/status.txt 2>&1
  hf upload baya1116/hypernet-sp-distill /root/work/status.txt pooler_distill/status.txt >/dev/null 2>&1
  sleep 300
done
SP
chmod +x /root/status_pub.sh
# the HF upload pulled huggingface-hub 1.x, which transformers refuses at import -> pin it back (keeps the `hf` CLI, added in 0.34)
if ! python3 -c "import transformers.modeling_utils" 2>/dev/null; then
  pip install -q "huggingface_hub>=0.34,<1.0" 2>&1 | tail -1
  python3 -c "import huggingface_hub, transformers.modeling_utils; print('hub', huggingface_hub.__version__, 'import ok')" || echo "IMPORT_STILL_BROKEN"
else
  echo "import ok"
fi
if ! pgrep -f "pool_eva[l].py" >/dev/null; then
  export SP_BASE=/root/fft_hf SP_RANK=128 SP_NOSYS=1 SP_EPISODIC=1 SP_HOTPOT2=0 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  echo "=== RELAUNCH $(date -u) ===" >> /root/pooleval_all_c.log
  setsid nohup python3 /root/work/pool_eval.py /root/fft_new_all.safetensors /root/work/eval300.jsonl /root/work/pooleval_all_c.jsonl --n 300 --rw 768 --decode plain --tag "[all_c]" >> /root/pooleval_all_c.log 2>&1 < /dev/null &
  sleep 45
  echo "=== RELAUNCH $(date -u) ===" >> /root/pooleval_all_nc.log
  setsid nohup python3 /root/work/pool_eval.py /root/fft_new_all.safetensors /root/work/eval300.jsonl /root/work/pooleval_all_nc.jsonl --n 300 --rw 8000 --decode plain --tag "[all_nc]" >> /root/pooleval_all_nc.log 2>&1 < /dev/null &
  echo "evals launched"
else
  echo "evals already running"
fi
cat > /usr/local/bin/t <<'TT'
#!/bin/bash
echo "$(date -u +%H:%M)Z  eval $(pgrep -fc 'pool_eva[l].py')  ctl $(pgrep -fc '/root/ctl\.s[h]')"
for m in all_c all_nc; do
  if [ -s /root/work/pooleval_$m.jsonl ]; then python3 /root/work/paired.py /root/work/teacher600.jsonl /root/work/pooleval_$m.jsonl $m; else echo "$m 0"; fi
done
TT
chmod +x /usr/local/bin/t
curl -sS --retry 3 -o /root/work/cnc.py https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/cnc.py
cat > /usr/local/bin/tc <<'TC'
#!/bin/bash
python3 /root/work/cnc.py /root/work/teacher600.jsonl /root/work/pooleval_all_c.jsonl /root/work/pooleval_all_nc.jsonl
TC
chmod +x /usr/local/bin/tc
echo "--- CNC $(date -u +%H:%M) ---"; tc
for m in all_c all_nc; do
  echo "--- ANALYZE $m $(date -u +%H:%M) ---"
  python3 /root/work/analyze_pool.py /root/work/teacher600.jsonl /root/work/pooleval_$m.jsonl 768 2>/dev/null | grep "^=== ANALYZE n\|^buckets\|^student wrong\|gold->answer\|^gold-never-served\|^student searches\|^teacher searches\|stopped after\|mean gen tokens"
  python3 - /root/work/pooleval_$m.jsonl <<'PY'
import json,sys,collections
R=[json.loads(l) for l in open(sys.argv[1])]
more=collections.Counter(r["text"].count("<more/>") for r in R)
wrong=[r for r in R if not r["correct"]]
print("more/q all:", dict(sorted(more.items())), "| wrong rows more/q:", dict(sorted(collections.Counter(r["text"].count("<more/>") for r in wrong).items())))
print("mean info tokens served/q:", sum(r["text"].count("<information") for r in R)/max(1,len(R)))
PY
done
curl -sS --retry 3 -o /root/work/head_scan.py https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/head_scan.py
for m in all_c all_nc; do echo "--- HEAD_SCAN $m ---"; python3 /root/work/head_scan.py /root/work/pooleval_$m.jsonl 2>&1 | grep -v Warning | cut -c1-400; done
pkill -f "status_pu[b].sh"; setsid nohup bash /root/status_pub.sh > /dev/null 2>&1 < /dev/null &
echo "RELAUNCH_DONE $(date -u)"
