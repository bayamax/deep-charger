# box B: install richer `t` (loss / ema / val / progress). Does not (re)launch anything.
cat > /usr/local/bin/t <<'TT'
#!/bin/bash
# SFT-B status, phone-width. Read-only.
M=full; [ -f /root/sft_full.log ] || M=smoke
echo "SFT-B[$M] $(date -u +%H:%M)Z  $(pgrep -f 'sft_pool_ru[n]' >/dev/null && echo 'run ok' || echo 'run STOPPED')"
L=/root/sft_$M.log; LL=/root/sft_loss_$M.log; VL=/root/fft_val_$M.log
if [ -f $LL ]; then
  python3 - "$LL" "$VL" <<'PY' 2>/dev/null
import sys
rows=[l.split() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
if not rows: print("no steps yet"); raise SystemExit
step,ex,loss,ema,tok=rows[-1]
print("step %s  ex %s  loss %s  ema %s" % (step,ex,loss,ema))
ems=[float(r[3]) for r in rows]
def m(a): return sum(a)/len(a)
if len(ems)>=10: print("ema  first10 %.3f  last10 %.3f" % (m(ems[:10]), m(ems[-10:])))
try:
    v=[l.split() for l in open(sys.argv[2]) if l.strip() and not l.startswith("#")]
    if v: print("val  step %s  all %s  multi %s  (%d evals)" % (v[-1][0], v[-1][2], v[-1][4], len(v)))
except Exception: pass
PY
fi
grep -E "^\[fft\] (RESUMED|pooler|DONE|OOM|example failed|oversize|idle)" $L 2>/dev/null | tail -3 | cut -c1-96
ls -la /root/fft_new_$M.safetensors* 2>/dev/null | awk '{print "ckpt", $5/1e9 "GB", $6, $7, $8, $9}' | tail -2
echo "gpu $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null)  disk $(df -h /root | tail -1 | awk '{print $4}') free"
TT
chmod +x /usr/local/bin/t; echo "t installed"; t
