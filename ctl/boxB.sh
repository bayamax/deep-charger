# box B: fetch files the multi-include download skipped (scripts, corpus, pooler.pt), then smoke
R=baya1116/hypernet-sp-distill
hf download $R --include "box_recover/scripts/*" --local-dir /root/hfdl 2>&1 | tail -1
hf download $R --include "box_recover/corpus.jsonl" --local-dir /root/hfdl 2>&1 | tail -1
hf download $R --include "fft_out/pooler.pt" --local-dir /root/hfdl 2>&1 | tail -1
S=/root/hfdl/box_recover/scripts; cp $S/*.py $S/*.sh /root/work/; cp /root/hfdl/box_recover/corpus.jsonl /root/work/corpus_box_final.jsonl
ln -sfn /root/hfdl/fft_out/pooler.pt /root/work/fft_out/pooler.pt
ls -la /root/work /root/work/fft_out/ | head -30
python3 -c "import transformers,tokenizers,peft,bitsandbytes; print('versions', transformers.__version__, tokenizers.__version__, peft.__version__, bitsandbytes.__version__)"
bash /root/do_sft_pool.sh smoke
