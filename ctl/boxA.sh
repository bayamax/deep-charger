# box A: fix tokenizer incompat (newer tokenizers), fetch files the multi-include download skipped, launch gen
R=baya1116/hypernet-sp-distill
pip install -q -U "transformers>=4.51" "tokenizers>=0.21" "peft>=0.15" 2>&1 | tail -1
python3 -c "import transformers,tokenizers,peft; print('versions', transformers.__version__, tokenizers.__version__, peft.__version__)"
[ -f /root/hfdl/box_recover/corpus.jsonl ] || hf download $R --include "box_recover/corpus.jsonl" --local-dir /root/hfdl 2>&1 | tail -1
cp /root/hfdl/box_recover/corpus.jsonl /root/work/corpus_box_final.jsonl
ls -la /root/work/corpus_box_final.jsonl /root/work/eval300.jsonl /root/work/bf16_mus_pure/ /root/work/ckpt_mus200/ 2>&1 | head -20
python3 -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('/root/work/bf16_mus_pure'); print('tokenizer ok', len(t))"
cat > /usr/local/bin/t <<'TT'
#!/bin/bash
F=/root/work/eval_gen_mus200_p0of1.jsonl
echo "GEN-A $(date -u +%H:%M)Z  $(pgrep -f 'eval_heldout_paralle[l]' >/dev/null && echo 'run ok' || echo 'run STOPPED')"
python3 - <<'PY' 2>/dev/null
import json,os,time
F="/root/work/eval_gen_mus200_p0of1.jsonl"
rows=[json.loads(l) for l in open(F)] if os.path.exists(F) else []
n=len(rows); TOT=2557
if n==0: print("rows 0/%d" % TOT); raise SystemExit
c=sum(1 for r in rows if r.get("correct")); g=sum(1 for r in rows if r.get("correct") and r.get("grounded"))
print("rows %d/%d  correct %.1f%%  keep(c&g) %d" % (n, TOT, 100*c/n, g)); print("last write %ds ago" % (time.time()-os.path.getmtime(F)))
PY
grep -o "\[[0-9]*/[0-9]*\].*eta=[0-9a-z]*" /root/gen_mus200.log 2>/dev/null | tail -1 | cut -c1-90
echo "gpu $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null)"
TT
chmod +x /usr/local/bin/t
bash /root/do_gen.sh
