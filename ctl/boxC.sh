# box C: (1) analysis of post vs teacher on the questions done so far; (2) launch post WITHOUT compression on the first 100
# questions (same weights, rw=8000 so nothing is ever pooled) to split "compression cost" from "SFT quality".
cd /root/work
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl
curl -sS -L -o /root/work/analyze_pool.py "$RAW/analyze_pool.py"
python3 /root/work/analyze_pool.py /root/work/teacher600.jsonl /root/work/pooleval_post.jsonl 512 2>&1 | grep -v Warning
if ! pgrep -f "pooleval_post_nocom[p]" >/dev/null && ! grep -q EVAL_DONE /root/pooleval_post_nocomp.log 2>/dev/null; then
  export SP_BASE=/root/fft_hf SP_RANK=128 SP_NOSYS=1 SP_EPISODIC=1 SP_HOTPOT2=0 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  setsid nohup python3 /root/work/pool_eval.py /root/fft_new_stream.safetensors /root/work/eval300.jsonl /root/work/pooleval_post_nocomp.jsonl --n 100 --rw 8000 --tag "[nocomp]" > /root/pooleval_post_nocomp.log 2>&1 < /dev/null &
  echo "nocomp launched"
fi
cat > /usr/local/bin/t <<'TT'
#!/bin/bash
echo "$(date -u +%H:%M)Z  procs $(pgrep -fc 'pool_eva[l].py')"
for m in post post_nocomp pre; do
  if [ -s /root/work/pooleval_$m.jsonl ]; then python3 /root/work/paired.py /root/work/teacher600.jsonl /root/work/pooleval_$m.jsonl $m; else echo "$m 0/300"; fi
done
TT
sleep 120; head -8 /root/pooleval_post_nocomp.log | grep "load\|cfg\|eval\|1/"; grep -i "error\|Traceback" /root/pooleval_post_nocomp.log | head -3
echo "ANALYZE_DONE $(date -u)"
