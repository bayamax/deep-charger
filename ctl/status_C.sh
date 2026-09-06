#!/bin/bash
echo "boot: $(grep -c BOOT_DONE /root/boot.log 2>/dev/null) | sft procs: $(pgrep -fc 'sft_pool_ru[n]') | harvest rows: $(cat /root/harvest_sft/*.jsonl 2>/dev/null | wc -l) | gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null) | disk: $(df -h /root | tail -1 | awk '{print $4}') free"
tail -1 /root/sft_loss_stream.log 2>/dev/null; grep -E "^\[fft\] (step|VAL|DONE|OOM|RESUMED)" /root/sft_stream.log 2>/dev/null | tail -2 | cut -c1-140
