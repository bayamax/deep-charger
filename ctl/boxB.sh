# box B: the 22GB recipe OOMs on 12GB (even len=391). Retry smoke with the 12GB recipe:
# freeze layers <14 and recompute activations (gradient checkpointing).
sed -i 's/FFT_FREEZE_BELOW=0 FFT_FREEZE_IO=1 FFT_GRADCKPT=0/FFT_FREEZE_BELOW=${FREEZE:-14} FFT_FREEZE_IO=1 FFT_GRADCKPT=${GCKPT:-1}/' /root/do_sft_pool.sh
grep -o "FFT_FREEZE_BELOW=[^ ]* FFT_FREEZE_IO=1 FFT_GRADCKPT=[^ ]*" /root/do_sft_pool.sh
rm -f /root/fft_new_smoke.safetensors
FREEZE=14 GCKPT=1 bash /root/do_sft_pool.sh smoke
