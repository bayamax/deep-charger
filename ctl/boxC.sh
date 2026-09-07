# box C: extend both evals of the new model (fft_new_all) from 150 to 300 held-out questions.
# pool_eval resumes from its output file, so questions 1-150 are skipped and 151-300 are appended.
# The HF upload (nohup /root/hfup.sh) keeps running independently; it does not use the GPU.
cd /root/work
if pgrep -f "pool_eva[l].py" >/dev/null; then echo "EVAL ALREADY RUNNING"; pgrep -af "pool_eva[l].py" | cut -c1-90; exit 0; fi
export SP_BASE=/root/fft_hf SP_RANK=128 SP_NOSYS=1 SP_EPISODIC=1 SP_HOTPOT2=0 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== EXTEND $(date -u) ===" >> /root/pooleval_all_c.log
setsid nohup python3 /root/work/pool_eval.py /root/fft_new_all.safetensors /root/work/eval300.jsonl /root/work/pooleval_all_c.jsonl --n 300 --rw 768 --decode plain --tag "[all_c]" >> /root/pooleval_all_c.log 2>&1 < /dev/null &
sleep 60
echo "=== EXTEND $(date -u) ===" >> /root/pooleval_all_nc.log
setsid nohup python3 /root/work/pool_eval.py /root/fft_new_all.safetensors /root/work/eval300.jsonl /root/work/pooleval_all_nc.jsonl --n 300 --rw 8000 --decode plain --tag "[all_nc]" >> /root/pooleval_all_nc.log 2>&1 < /dev/null &
cat > /usr/local/bin/t <<'TT'
#!/bin/bash
echo "$(date -u +%H:%M)Z  eval $(pgrep -fc 'pool_eva[l].py')  hfup: $(tail -1 /root/hfup.log 2>/dev/null | cut -c1-40)"
for m in all_c all_nc; do
  if [ -s /root/work/pooleval_$m.jsonl ]; then python3 /root/work/paired.py /root/work/teacher600.jsonl /root/work/pooleval_$m.jsonl $m; else echo "$m 0"; fi
done
TT
chmod +x /usr/local/bin/t
cat > /root/status.sh <<'ST'
#!/bin/bash
t
grep -i "error\|Traceback" /root/pooleval_all_c.log /root/pooleval_all_nc.log 2>/dev/null | tail -2
ST
sleep 120; grep "eval\[\|/300\]" /root/pooleval_all_c.log | tail -2; grep "eval\[\|/300\]" /root/pooleval_all_nc.log | tail -2
echo "EXTEND_LAUNCHED $(date -u)"
