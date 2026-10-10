#!/bin/bash
# The conversation screens for one copy (2026-10-10, the user: a copy that answers the second and third turn better, and
# whose replies fit better, is worth taking even when single-turn accuracy stays where it is): br3 (97 two-turn dialogues,
# win), chain3 (60 three-turn dialogues from MuSiQue chains, win), and the nano judge's reply quality on single-turn shard 0,
# br3 and chain3 (reply_judge.py: correct and sound, natural, clean), and each br3 / chain3 conversation judged whole
# (conv_judge.py -> /root/lines/<tag>_conv.txt: "<tag> conv_br3 conv_ch3"). Each part is reused when its output is there.
# Prints and writes /root/lines/<tag>_mt.txt:
#   "<tag> br3_t1 br3_fu ch_t1 ch_t2 ch_t3 good_st0 good_br3 good_ch"
#   bash grpo_mtscreen.sh <ckpt.safetensors> <tag>
CK=$1; T=$2; cd /root/work; mkdir -p /root/lines
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
full() { [ -s "$1" ] && [ "$(wc -l < "$1")" -ge "$2" ]; }
turns() { python3 -c "
import json,sys; r=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(' '.join(str(sum(bool(x['correct']) for x in r if x['turn']==t)) for t in range(int(sys.argv[2]))))" "$1" "$2" 2>/dev/null || echo "0 0 0" | cut -d' ' -f1-$2; }
good() { [ -s "$1" ] || { echo 0; return; }; OAI_KEY=$(cat /root/.oai 2>/dev/null) python3 /root/work/reply_judge.py "$1" "$2" 2>/dev/null | tee -a /root/rq.log | sed -E 's/^RQ [^ ]+ good ([0-9]+)\/.*/\1/' | grep -E '^[0-9]+$' || echo 0; }
f=/root/work/br3_$T.jsonl
full $f 150 || env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl $f --multiturn /root/work/mt_eval_bridge3.jsonl --mt-mode win --n 1000 $EVARGS --tag "[br3-$T]" > /root/br3_$T.log 2>&1
read b1 b2 < <(turns $f 2)
g=/root/work/ch3_$T.jsonl; NCH=$(grep -c . /root/work/mt_eval_chain3.jsonl)
full $g $((3 * NCH)) || env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl $g --multiturn /root/work/mt_eval_chain3.jsonl --mt-mode win --n 1000 $EVARGS --tag "[ch3-$T]" > /root/ch3_$T.log 2>&1
read h1 h2 h3 < <(turns $g 3)
q0=$(good /root/work/${T}st_0.jsonl "${T}_st0"); q1=$(good $f "${T}_br3"); q2=$(good $g "${T}_ch3")
cq() { OAI_KEY=$(cat /root/.oai 2>/dev/null) python3 /root/work/conv_judge.py "$1" "$2" 2>/dev/null | tee -a /root/rq.log | sed -E 's/^CQ [^ ]+ good ([0-9]+)\/.*/\1/' | grep -E '^[0-9]+$' || echo 0; }
c1=$(cq $f "${T}_br3"); c2=$(cq $g "${T}_ch3")
echo "$T $c1 $c2" > /root/lines/${T}_conv.txt   # the whole conversations judged (conv_judge.py): kept apart so the 9-field line stays as rounds read it
echo "$T $b1 $b2 $h1 $h2 $h3 $q0 $q1 $q2" | tee /root/lines/${T}_mt.txt
