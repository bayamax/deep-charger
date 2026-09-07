# box C: teacher-vs-post only -> stop the [pre] evaluation, keep [post] running alone (twice as fast).
pkill -f "pool_eval.py /root/hfdl/box_recover/fft_fina[l]" && sleep 3
echo "pre stopped; remaining: $(pgrep -af 'pool_eva[l].py' | cut -c1-80)"
cat > /usr/local/bin/t <<'TT'
#!/bin/bash
echo "POOLEVAL-C $(date -u +%H:%M)Z  procs $(pgrep -fc 'pool_eva[l].py')  gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
echo "[teacher ckpt_mus200, no compression] correct=41.5% (300 held-out)"
echo "[post] $(grep -o '\[[0-9]*/300\].*' /root/pooleval_post.log 2>/dev/null | tail -1 | cut -c1-100)"
grep "EVAL_DONE" /root/pooleval_post.log 2>/dev/null | tail -1
TT
chmod +x /usr/local/bin/t
cat > /root/status.sh <<'ST'
#!/bin/bash
echo "pooleval procs: $(pgrep -fc 'pool_eva[l].py') | gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
echo "[post] $(grep -o '\[[0-9]*/300\].*' /root/pooleval_post.log 2>/dev/null | tail -1 | cut -c1-110)"
grep "EVAL_DONE\|Error\|Traceback" /root/pooleval_post.log 2>/dev/null | tail -1
ST
echo "PRE_STOPPED $(date -u)"
