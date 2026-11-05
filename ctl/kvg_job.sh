#!/bin/bash
# GRPO with raw-window guides (my call, the user left the timing to me): the second unfrozen DPO round did not generalise -
# chosen-ahead on its first, unseen pairs 48% / 56% (round 1: 80%); its pairs are mostly closed-but-wrong, a content difference
# DPO did not spread to new problems. Here the collectors are rollout workers (ctl/kvc_job.sh MODE=grpo): per problem the
# memory G=4 times with the newest weights + the raw window once (a right-and-closed raw draw joins the group as a guide with
# reward 1), uploaded to archive/sp3/dpo_col_g/. The trainer (this box): once NEWG new problems are in, GRPO on their groups
# (--grpo-groups: reward 1 = closed and right, advantage = reward - group mean, uniform groups dropped, 4 groups per update,
# the model unfrozen at lr 1e-6, hypernet 2e-5), publish (spKVF_latest.pt + kvf/VERSION, the same names the collectors read),
# the 60 problems at seed 1; while waiting it works its own shard as a rollout worker.
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
for _t in $(seq 60); do curl -sS -o sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q '"--opt-state"' sp_qwen3.py && break; sleep 120; done
LG=/root/w/sp_kvg.log; NEWG=${NEWG:-24}; NSH=${NSH:-4}; EVERY=${EVERY:-3}; GMIN=${GMIN:-12}
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
gt() {   # the GRPO flags, re-read every round: kvg_accum / kvg_ftlr / kvg_ftlayers in /root/w override the defaults without a restart
  echo "--grpo-groups --ft-layers $(cat kvg_ftlayers 2>/dev/null || echo 28) --opt8bit --ft-lr $(cat kvg_ftlr 2>/dev/null || echo ${FTLR:-1e-6}) --lr $(cat kvg_plr 2>/dev/null || echo 2e-5) --lora $(cat kvg_lora 2>/dev/null || echo 0) --lora-lr $(cat kvg_loralr 2>/dev/null || echo 1e-5) --epochs 1 --accum $(cat kvg_accum 2>/dev/null || echo ${ACCUM:-4}) --const-lr --grpo-wrong $(cat kvg_wrong 2>/dev/null || echo 0) --opt-state kvg_opt.pt"; }
GT=$(gt)
rd=$(cat kvg_round 2>/dev/null || echo 0); v0=$(cat VERSION 2>/dev/null || echo 0)
if [ ! -s kvg.pt ]; then   # the starting weights go out first, so the rollout workers sample from the policy being trained
  cp ${START:-kvf.pt} kvg.pt; cp kvg.pt kvf.pt; v0=$((v0 + 1)); echo $v0 > VERSION; echo $v0 > kvg_v0
  put kvg.pt spKVF_latest.pt && put VERSION kvf/VERSION; fi
v0=$(cat kvg_v0 2>/dev/null || echo $v0)
echo "[kg] ready $(date -u) | grpo-groups in the script: $(grep -c '"--grpo-groups"' sp_qwen3.py) | start ${START:-kvf.pt} (version $v0) | $GT"
ok60() { python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open(sys.argv[1])]; print(sum(correct(r) for r in rs), sum('</think>' in r['text'] for r in rs))" "$1"; }
curl -sS -o param_sketch.py "$RAW/ctl/param_sketch.py?nocache=$(date +%s)"
sk() {   # sk <ckpt> <version>: the weights' sketch (parameter-movement tracking), kept locally and on the hub
  [ -s sk_v$2.pt ] || { python3 param_sketch.py sketch $1 sk_v$2.pt && put sk_v$2.pt kvg/sk_v$2.pt > /dev/null; }; }
[ -s ${START:-kvf_round1.pt} ] && sk ${START:-kvf_round1.pt} $v0
[ -s kvg_prev.pt ] && [ -s kvg.pt ] && [ "$(cat VERSION 2>/dev/null)" -gt $((v0 + 1)) ] 2>/dev/null && sk kvg_prev.pt $(( $(cat VERSION) - 1 ))
[ -s kvg.pt ] && sk kvg.pt $(cat VERSION)
moves() { python3 param_sketch.py report $(ls sk_v*.pt | sort -t v -k2 -n) 2>&1 | sed 's/^/[kg] params /'; }
moves
touch used_groups.txt
while [ ! -e STOP_KG ]; do
  V0=$v0 GMIN=$GMIN CUR=$(cat VERSION) python3 - <<'PY' > newg_list.txt
import os, re
from huggingface_hub import HfApi
used = set(open('used_groups.txt').read().split()); v0 = int(os.environ['V0'])
fs = [f for f in HfApi().list_repo_files('baya1116/hypernet-sp-distill') if f.startswith('archive/sp3/dpo_col_g/') and f not in used]
gmin = int(os.environ.get('GMIN', '0'))
def g_(f):   # group size from the file name (g12_b..); the first G=4 files carry none
    m = re.search(r'/g(\d+)_b', f); return int(m.group(1)) if m else 4
lo = max(v0, int(os.environ.get('CUR', v0)) - 1)   # this run's weights, at most one version stale (asynchronous workers)
print('\n'.join(sorted(f for f in fs if int(re.search(r'_v(\d+)\.jsonl$', f).group(1)) >= lo and g_(f) >= gmin)))
PY
  : > newg.jsonl
  for f in $(cat newg_list.txt); do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" "$H/$f" >> newg.jsonl 2>/dev/null; done
  n=$(wc -l < newg.jsonl)
  NEWG=$(cat kvg_newg 2>/dev/null || echo $NEWG)
  if [ $n -lt $NEWG ]; then   # meanwhile one rollout block of its own shard
    curl -sS -o kvc_job.sh "$RAW/ctl/kvc_job.sh?nocache=$(date +%s)"
    MODE=grpo SHARD=$((NSH - 1)) NSHARD=$NSH ONE=1 RB=48 TB=60000 bash kvc_job.sh | grep "^\[kc\] grpo" | sed 's/^\[kc\]/[kg] own/' || sleep 600
    continue; fi
  rd=$((rd + 1)); cat newg_list.txt >> used_groups.txt; cp newg.jsonl kvg_groups_r$rd.jsonl
  cp kvg.pt kvg_prev.pt
  GT=$(gt); echo "[kg] round $rd flags: $GT"
  python3 sp_qwen3.py train --traces kvg_groups_r$rd.jsonl $M --max-len 12288 --ckpt kvg_prev.pt $GT --seed $((6000 + rd)) --out kvg_next.pt --log $LG > run_kg_tr.log 2>&1
  echo "[kg] round $rd: $n problems | $(grep '\[sp\] grpo groups' run_kg_tr.log | tail -1 | cut -d: -f2-) | $(grep -E '\[sp\] train .*grpo group' run_kg_tr.log | tail -1 | grep -oP 'mean reward.*') | $(grep -oP '\[sp\] LoRA.*' run_kg_tr.log | head -1)"
  grep -E "Error|Traceback|out of memory" run_kg_tr.log | tail -1 | sed "s/^/[kg] round $rd: /"
  [ -s kvg_next.pt ] && grep -q "^\[sp\] trained" run_kg_tr.log || { echo "[kg] round $rd: training FAILED"; break; }
  mv kvg_next.pt kvg.pt; echo $rd > kvg_round; v=$((v0 + rd)); echo $v > VERSION; cp kvg.pt kvf.pt
  put kvg.pt spKVF_latest.pt && put VERSION kvf/VERSION
  sk kvg.pt $v; moves
  { echo "round $rd -> v$v $(date -u +%H:%MZ) | $(grep '\[sp\] grpo groups' run_kg_tr.log | tail -1 | cut -d: -f2-) | $(grep -E '\[sp\] train .*grpo group' run_kg_tr.log | tail -1 | grep -oP 'mean reward [0-9.]+')"
    python3 param_sketch.py report $(ls sk_v*.pt | sort -t v -k2 -n) 2>&1 | grep "^\[lora\]" | sed 's/^/  /'; } >> kvg_rounds.txt
  put kvg_rounds.txt kvg/rounds.txt > /dev/null   # the container log drops lines; the round summaries live here too
  if [ $((rd % EVERY)) != 0 ]; then echo "[kg] round $rd (weights v$v) published"; continue; fi
  o=kvg_v${v}_s1.jsonl; rm -f $o   # (an older run's file of the same name was appended to once)
  python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $M --ckpt kvg.pt --max-len 12288 --roll-batch 48 --grpo-tok-budget 60000 --seed 1 --roll-out $o --log $LG > run_kggen.log 2>&1
  put $o kvg/$o; echo "[kg] round $rd (weights v$v) seed 1: $(ok60 $o | awk '{print $1 "/60 correct, closed " $2}')"; put $LG kvg/kvg.log
done
echo "[kg] JOB_DONE $(date -u)"
