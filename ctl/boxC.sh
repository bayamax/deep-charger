# box C: the relaunch truncated the seen-list (launcher resets it) -> it started a 2nd pass over all rows.
# Redo cleanly: one single pass over the final 499 rows from fft_final, no waiting for data (~15 min).
pkill -f "sft_pool_ru[n].py"; sleep 8; pgrep -af "sft_pool_ru[n]" || echo "trainer stopped"
mv -f /root/fft_new_stream.safetensors /root/fft_stream_2pass_partial.safetensors 2>/dev/null
rm -f /root/fft_new_stream.safetensors.s* /root/fft_seen_stream.txt
mv -f /root/sft_stream.log /root/sft_stream_old.log; mv -f /root/sft_loss_stream.log /root/sft_loss_stream_old.log
echo "harvest rows: $(wc -l < /root/harvest_sft/gen_mus200.jsonl)"
FREEZE=0 GCKPT=0 IDLE_EXIT=240 bash /root/do_sft_pool.sh stream
for i in $(seq 1 80); do grep -q "^\[fft\] DONE" /root/sft_stream.log && break; sleep 30; done
grep "RESUMED\|^\[fft\] DONE" /root/sft_stream.log | tail -3; tail -2 /root/sft_loss_stream.log
ls -la /root/fft_new_stream.safetensors
echo "--- LOSSLOG clean-pass"; cat /root/sft_loss_stream.log; echo "--- END LOSSLOG"
echo "C_FINALIZE_DONE $(date -u)"
