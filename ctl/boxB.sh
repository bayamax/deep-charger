# box B: diagnose missing files, repair, rerun smoke
echo "--- boot.log tail"; tail -12 /root/boot.log
echo "--- hfdl tree"; find /root/hfdl -maxdepth 3 -type f | head -30
echo "--- work"; ls /root/work
S=/root/hfdl/box_recover/scripts
if [ ! -f $S/sft_pool_run.py ]; then hf download baya1116/hypernet-sp-distill --include "box_recover/scripts/*" --include "box_recover/corpus.jsonl" --include "box_recover/fft_val_set.txt" --local-dir /root/hfdl 2>&1 | tail -2; fi
cp $S/*.py $S/*.sh /root/work/ 2>&1; cp /root/hfdl/box_recover/corpus.jsonl /root/work/corpus_box_final.jsonl 2>&1
cp /root/hfdl/grpo_assets/mus_run/eval_step200/eval_heldout_300q_g2.jsonl /root/work/eval300.jsonl 2>&1
ls -la /root/work | head -30
bash /root/do_sft_pool.sh smoke
