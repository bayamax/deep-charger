# box C: compressed (SP pooling) held-out evaluation, pre vs post self-distillation, run concurrently.
cd /root/work
if pgrep -f "pool_eva[l].py" >/dev/null; then echo "POOL EVAL ALREADY RUNNING"; exit 0; fi
[ -f /root/fft_new_stream.safetensors ] || { echo "no final checkpoint"; exit 0; }
grep -q "^\[fft\] DONE" /root/sft_stream.log || { echo "SFT not DONE yet"; exit 0; }
curl -sS -L -o /root/work/pool_eval.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/pool_eval.py" && python3 -m py_compile /root/work/pool_eval.py && echo "pool_eval ok"
export SP_BASE=/root/fft_hf SP_RANK=128 SP_NOSYS=1 SP_EPISODIC=1 SP_HOTPOT2=0 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
setsid nohup python3 /root/work/pool_eval.py /root/hfdl/box_recover/fft_final.safetensors /root/work/eval300.jsonl /root/work/pooleval_pre.jsonl --tag "[pre]" > /root/pooleval_pre.log 2>&1 < /dev/null &
sleep 60
setsid nohup python3 /root/work/pool_eval.py /root/fft_new_stream.safetensors /root/work/eval300.jsonl /root/work/pooleval_post.jsonl --tag "[post]" > /root/pooleval_post.log 2>&1 < /dev/null &
cat > /usr/local/bin/t <<'TT'
#!/bin/bash
echo "POOLEVAL-C $(date -u +%H:%M)Z  procs $(pgrep -fc 'pool_eva[l].py')  gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
for m in pre post; do echo "[$m] $(grep -o '\[[0-9]*/300\].*' /root/pooleval_$m.log 2>/dev/null | tail -1 | cut -c1-100)"; grep "EVAL_DONE" /root/pooleval_$m.log 2>/dev/null | tail -1; done
TT
chmod +x /usr/local/bin/t
cat > /root/status.sh <<'ST'
#!/bin/bash
echo "pooleval procs: $(pgrep -fc 'pool_eva[l].py') | gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
for m in pre post; do echo "[$m] $(grep -o '\[[0-9]*/300\].*' /root/pooleval_$m.log 2>/dev/null | tail -1 | cut -c1-110)"; grep "EVAL_DONE\|Error\|Traceback" /root/pooleval_$m.log 2>/dev/null | tail -1; done
ST
sleep 150; echo "--- pre head"; head -12 /root/pooleval_pre.log; echo "--- post head"; head -12 /root/pooleval_post.log; pgrep -af "pool_eva[l].py" | cut -c1-80
