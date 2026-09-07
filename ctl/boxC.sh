# box C: verify the post-SFT eval launch
cat /root/after_sft.out 2>/dev/null; pgrep -af "after_sf[t].sh\|pool_eva[l].py" | cut -c1-90
for m in all_c all_nc; do echo "--- $m"; grep "load\|cfg\|eval\[\|/150\]\|Error\|Traceback" /root/pooleval_$m.log 2>/dev/null | tail -4; done
ls -la /root/fft_new_all.safetensors; tail -2 /root/sft_all.log
echo "VERIFY_DONE $(date -u)"
