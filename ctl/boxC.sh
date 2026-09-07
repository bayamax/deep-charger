# box C: post runs alone; when it finishes, resume the pre eval from where it stopped (pool_eval resumes from its out file).
cat > /usr/local/bin/t <<'TT'
#!/bin/bash
echo "POOLEVAL-C $(date -u +%H:%M)Z  procs $(pgrep -fc 'pool_eva[l].py')  gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
echo "[teacher ckpt_mus200, no compression] correct=41.5% (300 held-out)"
for m in post pre; do echo "[$m] $(grep -o '\[[0-9]*/300\].*' /root/pooleval_$m.log 2>/dev/null | tail -1 | cut -c1-100)"; grep "EVAL_DONE" /root/pooleval_$m.log 2>/dev/null | tail -1; done
TT
chmod +x /usr/local/bin/t
cat > /root/status.sh <<'ST'
#!/bin/bash
echo "pooleval procs: $(pgrep -fc 'pool_eva[l].py') | gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
for m in post pre; do echo "[$m] $(grep -o '\[[0-9]*/300\].*' /root/pooleval_$m.log 2>/dev/null | tail -1 | cut -c1-110)"; grep "EVAL_DONE\|Error\|Traceback" /root/pooleval_$m.log 2>/dev/null | tail -1; done
ST
cat > /root/resume_pre.sh <<'RP'
#!/bin/bash
# wait for post to finish, then resume pre (appends to /root/work/pooleval_pre.jsonl, skipping questions already done)
while ! grep -q "EVAL_DONE" /root/pooleval_post.log 2>/dev/null; do sleep 60; done
pgrep -f "pool_eva[l].py" >/dev/null && exit 0
cd /root/work
export SP_BASE=/root/fft_hf SP_RANK=128 SP_NOSYS=1 SP_EPISODIC=1 SP_HOTPOT2=0 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== RESUME $(date -u) ===" >> /root/pooleval_pre.log
python3 /root/work/pool_eval.py /root/hfdl/box_recover/fft_final.safetensors /root/work/eval300.jsonl /root/work/pooleval_pre.jsonl --tag "[pre]" >> /root/pooleval_pre.log 2>&1 < /dev/null
RP
chmod +x /root/resume_pre.sh
pgrep -f "resume_pr[e].sh" >/dev/null || setsid nohup bash /root/resume_pre.sh > /root/resume_pre.out 2>&1 < /dev/null &
sleep 2; echo "pre done so far: $(wc -l < /root/work/pooleval_pre.jsonl) rows; resume watcher: $(pgrep -fc 'resume_pr[e].sh')"
echo "RESUME_ARMED $(date -u)"
