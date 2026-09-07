# box C: simplify t -> solved-so-far accuracy vs teacher on the same questions.
curl -sS -L -o /root/work/paired.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/paired.py"
cat > /usr/local/bin/t <<'TT'
#!/bin/bash
echo "$(date -u +%H:%M)Z  procs $(pgrep -fc 'pool_eva[l].py')"
for m in post pre; do
  if [ -s /root/work/pooleval_$m.jsonl ]; then python3 /root/work/paired.py /root/work/teacher600.jsonl /root/work/pooleval_$m.jsonl $m; else echo "$m 0/300"; fi
done
TT
chmod +x /usr/local/bin/t
t
echo "T_SIMPLE $(date -u)"
