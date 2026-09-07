# box C, steps 1-2-3 approved:
#  (0) RW sensitivity scan on the post rows done so far (answers "how big a raw window would keep the gold in view")
#  (1) stop the guard-decoded evals (post / nocomp / pre watcher) -> GPU free
#  (2) SFT "all": one pass over every landed teacher trajectory (failures included), from fft_final
#  (3) when SFT is DONE: compressed (rw from /root/rw.txt, default 512) and non-compressed (rw=8000) evals of the
#      new model on the first 150 held-out questions, plain (teacher-identical) decoding
cd /root/work
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl
for f in rw_scan.py pool_eval.py paired.py; do for i in 1 2 3 4 5; do curl -sS -L --retry 3 -o /root/work/$f "$RAW/$f" && python3 -m py_compile /root/work/$f && break; sleep 10; done; done
for i in 1 2 3 4 5; do curl -sS -L --retry 3 -o /root/do_sft_pool.sh "$RAW/do_sft_pool.sh" && bash -n /root/do_sft_pool.sh && break; sleep 10; done
grep -c "decode" /root/work/pool_eval.py; grep -c "harvest_all" /root/do_sft_pool.sh
python3 /root/work/rw_scan.py /root/work/teacher600.jsonl /root/work/pooleval_post.jsonl 2>&1 | grep -v Warning
# (1)
pkill -f "resume_pr[e].sh"; pkill -f "pool_eva[l].py"; sleep 8; pgrep -af "pool_eva[l].py" || echo "guard evals stopped (post $(wc -l < /root/work/pooleval_post.jsonl) rows, nocomp $(wc -l < /root/work/pooleval_post_nocomp.jsonl 2>/dev/null) rows kept)"
# (2)
[ -f /root/rw.txt ] || echo 512 > /root/rw.txt
rm -f /root/fft_new_all.safetensors /root/fft_seen_all.txt
FREEZE=0 GCKPT=0 bash /root/do_sft_pool.sh all 2>&1 | grep -v "^$" | tail -12
# (3) chain
cat > /root/after_sft.sh <<'AS'
#!/bin/bash
while ! grep -q "^\[fft\] DONE" /root/sft_all.log 2>/dev/null; do sleep 60; done
sleep 20; cd /root/work; RW=$(cat /root/rw.txt 2>/dev/null || echo 512)
export SP_BASE=/root/fft_hf SP_RANK=128 SP_NOSYS=1 SP_EPISODIC=1 SP_HOTPOT2=0 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
setsid nohup python3 /root/work/pool_eval.py /root/fft_new_all.safetensors /root/work/eval300.jsonl /root/work/pooleval_all_c.jsonl --n 150 --rw $RW --decode plain --tag "[all_c]" > /root/pooleval_all_c.log 2>&1 < /dev/null &
sleep 60
setsid nohup python3 /root/work/pool_eval.py /root/fft_new_all.safetensors /root/work/eval300.jsonl /root/work/pooleval_all_nc.jsonl --n 150 --rw 8000 --decode plain --tag "[all_nc]" > /root/pooleval_all_nc.log 2>&1 < /dev/null &
echo "EVALS_LAUNCHED rw=$RW $(date -u)"
AS
chmod +x /root/after_sft.sh
pgrep -f "after_sf[t].sh" >/dev/null || setsid nohup bash /root/after_sft.sh > /root/after_sft.out 2>&1 < /dev/null &
cat > /usr/local/bin/t <<'TT'
#!/bin/bash
echo "$(date -u +%H:%M)Z  sft $(pgrep -fc 'sft_pool_ru[n]')  eval $(pgrep -fc 'pool_eva[l].py')"
if pgrep -f "sft_pool_ru[n]" >/dev/null; then echo "SFT-all $(grep -o 'step [0-9]* ex [0-9]* loss [0-9.]* ema [0-9.]*' /root/sft_all.log | tail -1)"; fi
grep -q "^\[fft\] DONE" /root/sft_all.log 2>/dev/null && echo "SFT-all DONE"
for m in all_c all_nc post; do
  if [ -s /root/work/pooleval_$m.jsonl ]; then python3 /root/work/paired.py /root/work/teacher600.jsonl /root/work/pooleval_$m.jsonl $m; else echo "$m 0"; fi
done
TT
cat > /root/status.sh <<'ST'
#!/bin/bash
t
grep -i "error\|Traceback\|OOM" /root/sft_all.log /root/pooleval_all_c.log /root/pooleval_all_nc.log 2>/dev/null | tail -2
cat /root/after_sft.out 2>/dev/null | tail -1
ST
echo "STEP123_ARMED $(date -u)"
