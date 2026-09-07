# box C: `tl` = loss view for the SFT-all run (thirds, halves, last ema), phone width
cat > /usr/local/bin/tl <<'TL'
#!/bin/bash
L=/root/sft_loss_all.log
python3 - "$L" <<'PY'
import sys, statistics as st
rows=[l.split() for l in open(sys.argv[1]) if l.strip() and l.split()[0].isdigit()]
if not rows: print("no steps yet"); sys.exit()
loss=[float(r[2]) for r in rows]; ema=[float(r[3]) for r in rows]; n=len(loss); k=max(n//3,1); h=max(n//2,1)
print(f"SFT-all step {n}/~140  ema {ema[-1]:.3f}")
print(f"1st/2nd/3rd  {st.mean(loss[:k]):.3f} {st.mean(loss[k:2*k]) if n>=2*k and n>k else float('nan'):.3f} {st.mean(loss[2*k:]) if n>2*k else float('nan'):.3f}")
print(f"1st half {st.mean(loss[:h]):.3f}  2nd half {st.mean(loss[h:]) if n>h else float('nan'):.3f}")
print("last5 ema " + " ".join(f"{e:.2f}" for e in ema[-5:]))
PY
grep -q "^\[fft\] DONE" /root/sft_all.log 2>/dev/null && echo "DONE"
TL
chmod +x /usr/local/bin/tl; tl
echo "TL_INSTALLED $(date -u)"
