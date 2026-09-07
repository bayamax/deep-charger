# box C: paired comparison against the teacher's per-question results on the same held-out questions.
cd /root/work
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl
[ -s /root/work/teacher600.jsonl ] || curl -sS -L -o /root/work/teacher600.jsonl "https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/grpo_assets/mus_run/eval_step200/eval_heldout_300q_g2.jsonl"
curl -sS -L -o /root/work/paired.py "$RAW/paired.py"
echo "teacher rows: $(wc -l < /root/work/teacher600.jsonl)"
cat > /usr/local/bin/t <<'TT'
#!/bin/bash
echo "POOLEVAL-C $(date -u +%H:%M)Z  procs $(pgrep -fc 'pool_eva[l].py')  gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
for m in post pre; do
  echo "[$m] $(grep -o '\[[0-9]*/300\].*' /root/pooleval_$m.log 2>/dev/null | tail -1 | cut -c1-100)"
  [ -s /root/work/pooleval_$m.jsonl ] && python3 /root/work/paired.py /root/work/teacher600.jsonl /root/work/pooleval_$m.jsonl $m
  grep "EVAL_DONE" /root/pooleval_$m.log 2>/dev/null | tail -1
done
TT
chmod +x /usr/local/bin/t
cat > /root/status.sh <<'ST'
#!/bin/bash
echo "pooleval procs: $(pgrep -fc 'pool_eva[l].py') | gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
for m in post pre; do
  echo "[$m] $(grep -o '\[[0-9]*/300\].*' /root/pooleval_$m.log 2>/dev/null | tail -1 | cut -c1-110)"
  [ -s /root/work/pooleval_$m.jsonl ] && python3 /root/work/paired.py /root/work/teacher600.jsonl /root/work/pooleval_$m.jsonl $m
  grep "EVAL_DONE\|Error\|Traceback" /root/pooleval_$m.log 2>/dev/null | tail -1
done
ST
t
echo "PAIRED_ARMED $(date -u)"
