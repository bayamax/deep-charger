# box B (pooler SFT). 1) install `t`  2) smoke test (self-guarded)
cat > /usr/local/bin/t <<'TT'
#!/bin/bash
# SFT status, phone-width. Read-only.
echo "SFT-B $(date -u +%H:%M)Z  $(pgrep -f 'sft_pool_ru[n]' >/dev/null && echo 'run ok' || echo 'run STOPPED')"
grep -q BOOT_DONE /root/boot.log 2>/dev/null || { echo "boot: $(tail -1 /root/boot.log | cut -c1-70)"; exit 0; }
for m in smoke full; do L=/root/sft_$m.log; [ -f $L ] || continue; echo "[$m]"; grep -E "^\[fft\] (step|VAL|DONE|OOM|example failed|RESUMED|pooler)" $L | tail -4 | cut -c1-100; done
tail -1 /root/sft_loss_full.log 2>/dev/null
echo "gpu $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null)  disk $(df -h /root | tail -1 | awk '{print $4}') free"
TT
chmod +x /usr/local/bin/t
bash /root/do_sft_pool.sh smoke
