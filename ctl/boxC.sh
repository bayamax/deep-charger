# box C: install `t`, then start the streaming SFT (waits for BOOT_DONE inside; self-guarded)
cat > /usr/local/bin/t <<'TT'
#!/bin/bash
M=stream
echo "SFT-C[$M] $(date -u +%H:%M)Z  $(pgrep -f 'sft_pool_ru[n]' >/dev/null && echo 'run ok' || echo 'run STOPPED')  harvest $(cat /root/harvest_sft/*.jsonl 2>/dev/null | wc -l) rows"
L=/root/sft_$M.log; LL=/root/sft_loss_$M.log; VL=/root/fft_val_$M.log
[ -f $LL ] && python3 - "$LL" "$VL" <<'PY' 2>/dev/null
import sys
rows=[l.split() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
if not rows: print("no steps yet"); raise SystemExit
step,ex,loss,ema,tok=rows[-1]
print("step %s  ex %s  loss %s  ema %s" % (step,ex,loss,ema))
ems=[float(r[3]) for r in rows]; m=lambda a: sum(a)/len(a)
if len(ems)>=10: print("ema  first10 %.3f  last10 %.3f" % (m(ems[:10]), m(ems[-10:])))
try:
    v=[l.split() for l in open(sys.argv[2]) if l.strip() and not l.startswith("#")]
    if v: print("val  step %s  all %s  multi %s  (%d evals)" % (v[-1][0], v[-1][2], v[-1][4], len(v)))
except Exception: pass
PY
grep -E "^\[fft\] (RESUMED|pooler|DONE|OOM|example failed|oversize|idle)" $L 2>/dev/null | tail -2 | cut -c1-96
ls -la /root/fft_new_$M.safetensors* 2>/dev/null | awk '{print "ckpt", $5/1e9 "GB", $6, $7, $8}' | tail -2
echo "gpu $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null)  disk $(df -h /root | tail -1 | awk '{print $4}') free"
TT
chmod +x /usr/local/bin/t
FREEZE=0 GCKPT=0 IDLE_EXIT=14400 bash /root/do_sft_pool.sh stream
echo "--- LOSSLOG $(date -u +%H:%M) v1788736032"; cat /root/sft_loss_stream.log; echo "--- END LOSSLOG"
