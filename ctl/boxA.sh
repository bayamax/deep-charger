# box A (generation). 1) install phone-width status command `t`  2) launch generation (self-guarded)
cat > /usr/local/bin/t <<'TT'
#!/bin/bash
# GEN status, phone-width. Read-only.
F=/root/work/eval_gen_mus200_p0of1.jsonl
echo "GEN-A $(date -u +%H:%M)Z  $(pgrep -f 'eval_heldout_paralle[l]' >/dev/null && echo 'run ok' || echo 'run STOPPED')"
grep -q BOOT_DONE /root/boot.log 2>/dev/null || { echo "boot: $(tail -1 /root/boot.log | cut -c1-70)"; exit 0; }
python3 - <<'PY' 2>/dev/null
import json,os,time
F="/root/work/eval_gen_mus200_p0of1.jsonl"
rows=[json.loads(l) for l in open(F)] if os.path.exists(F) else []
n=len(rows); TOT=2557
if n==0: print("rows 0/%d" % TOT); raise SystemExit
c=sum(1 for r in rows if r.get("correct")); g=sum(1 for r in rows if r.get("correct") and r.get("grounded"))
age=time.time()-os.path.getmtime(F)
t0=os.path.getmtime("/root/gen_mus200.log"); 
print("rows %d/%d  correct %.1f%%  keep(c&g) %d" % (n, TOT, 100*c/n, g))
print("last write %ds ago" % age)
PY
grep -o "\[[0-9]*/[0-9]*\].*eta=[0-9a-z]*" /root/gen_mus200.log 2>/dev/null | tail -1 | cut -c1-90
echo "gpu $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null)"
TT
chmod +x /usr/local/bin/t
bash /root/do_gen.sh
