#!/bin/bash
# Re-run agent6 on a fresh machine (the local run's container went away): the data and the last saved model come back from
# the hub, training continues from model_best.pt (or agent5 when none was saved), then the same scoring as before.
#   HF_TOKEN=... bash ctl/charlm/a6_resume.sh WORKDIR
set -e; W=${1:-a6}; mkdir -p $W; cd $W; C=$(cd "$(dirname "$0")" && pwd)
HUB=https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/charlm
for f in mix_a6.jsonl goal_val.jsonl soda_val.jsonl help_val.jsonl npc_val.jsonl prompts_g.json; do [ -s $f ] || curl -sSfL -o $f $HUB/agent6/data/$f; done
[ -s base.pt ] || curl -sSfL -o base.pt $HUB/agent6/model_best.pt || curl -sSfL -o base.pt $HUB/agent5/model_final.pt
python3 -u $C/charlm_sft.py --base base.pt --data sftdata --out agent6 --shards 0 --jsonl mix_a6.jsonl --val-jsonl goal_val.jsonl --val-convs 500 --min-learned 2 \
  --prompts-json prompts_g.json --epochs 1 --lr 5e-5 --warmup 10 --micro 4 --eval-every 45 --hours 4 --hub baya1116/hypernet-sp-distill:charlm/agent6
python3 -u $C/chat_bits.py --models agent6/model_final.pt --sets soda_val.jsonl,help_val.jsonl,npc_val.jsonl --n 200 2>&1 | grep "^\[chat\]" | tee chat.txt
for m in final:200 4bit:100; do
  echo "[a6] agent6-${m%:*} playing made-up-named episodes:" | tee -a play6.txt
  python3 -u $C/goal_gen.py play --model agent6/model_${m%:*}.pt --n ${m#*:} --split test --seed 5 2>&1 | grep -E "^\[goal\]" | tee -a play6.txt
done
