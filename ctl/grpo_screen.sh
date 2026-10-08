#!/bin/bash
# The GRPO screens for one copy, on box G2's base: single-turn shards 0-2 (100 each), br3 (two-turn, win), the Dolphin
# held-out hundred (nano-judged). Each part is reused when its output is already there (the mtg7 keeper's names), so a
# copy screened before costs nothing. Prints and writes /root/lines/<tag>.txt: "<tag> shard0 single300 br3_t1 br3_fu dolphin".
#   bash grpo_screen.sh <ckpt.safetensors> <tag>
CK=$1; T=$2; cd /root/work; mkdir -p /root/lines
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
cnt() { python3 -c "import json,sys;print(sum(bool(json.loads(l).get('correct')) for l in open(sys.argv[1]) if l.strip()))" "$1" 2>/dev/null || echo 0; }
full() { [ -s "$1" ] && [ "$(wc -l < "$1")" -ge "$2" ]; }
c=()
for i in 0 1 2; do
  f=/root/work/${T}st_$i.jsonl
  full $f 100 || env $ENV python3 /root/work/pool_eval.py $CK /root/work/ev_$i.jsonl $f --n 100 $EVARGS --tag "[${T}st$i]" > /root/${T}st_$i.log 2>&1
  c[$i]=$(cnt $f)
done
f=/root/work/br3_$T.jsonl
full $f 150 || env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl $f --multiturn /root/work/mt_eval_bridge3.jsonl --mt-mode win --n 1000 $EVARGS --tag "[br3-$T]" > /root/br3_$T.log 2>&1
read b1 b2 < <(python3 -c "
import json,sys; r=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(sum(x['correct'] for x in r if x['turn']==0), sum(x['correct'] for x in r if x['turn']==1))" $f 2>/dev/null || echo "0 0")
j=/root/work/dl_${T}_judged.jsonl
if ! full $j 100; then
  full /root/work/dl_$T.jsonl 100 || env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 /root/work/online_loop.py $CK /root/evalrun_dl_$T --questions /root/work/eval300.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl \
    --b 12 --gen 7000 --budget 2400 --temp 0.6 --maxsrch 7 --pooler none --lora-rank 16 --lora-layers all --stop eos --loop-break answer \
    --eval-file /root/work/dolphinq.jsonl --eval-out /root/work/dl_$T.jsonl > /root/dl_$T.log 2>&1
  OAI_KEY=$(cat /root/.oai 2>/dev/null) python3 /root/work/dl_judge.py /root/work/dl_$T.jsonl $T > /dev/null 2>&1
  rm -rf /root/evalrun_dl_$T
fi
dl=$(python3 -c "import json,sys;print(sum(int(json.loads(l).get('pass',0)) for l in open(sys.argv[1]) if l.strip()))" $j 2>/dev/null || echo 0)
echo "$T ${c[0]} $((c[0] + c[1] + c[2])) $b1 $b2 $dl" | tee /root/lines/$T.txt
