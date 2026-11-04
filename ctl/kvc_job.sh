#!/bin/bash
# DPO data collector (the user: collect in parallel on cheaper boxes; the trainer takes the last shard one block at a time). SHARD of NSHARD: the problems of ctl/dpo_pool.jsonl from
# line 301 on (the first 300 went to iteration 1 of kvi) whose index % NSHARD == SHARD, in blocks of 24:
#  - the raw window 1024: always the ORIGINAL model (pooler-only checkpoint, memory unused: --no-sp);
#  - the memory: the trainer's newest weights (archive/sp3/kvf/VERSION -> spKVF_latest.pt, model + hypernet); before the
#    first one exists, the DPO reference (spKVR_latest.pt, --ctl-window segment 1);
#  - both sides redrawn where both failed; pairs both ways (dpo_pairs.py); every file to archive/sp3/dpo_col/s<SHARD>/, the
#    pairs as pairs_b<block>_v<weights version>.jsonl.
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); mkdir -p /root/w; cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && return 0; sleep 60; done; echo "[kc] upload $2 FAILED"; }
get() { for _i in 1 2 3; do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o "$2" "$H/$1" && return 0; sleep 10; done; return 1; }
for f in tools/sp_qwen3.py tools/sp_mathcheck.py ctl/dpo_pool.jsonl ctl/dpo_pairs.py; do curl -sS -o $(basename $f) "$RAW/$f?nocache=$(date +%s)"; done
python3 -c "import transformers" 2>/dev/null || python3 -m pip install -q "huggingface_hub[hf_xet]" transformers "math-verify[antlr4_13_2]" 2>&1 | grep -v -i warn | tail -1
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B')" > dl.log 2>&1
[ -s ref.pt ] || get archive/sp3/spKVR_latest.pt ref.pt || { echo "[kc] no reference weights"; exit 1; }
LG=/root/w/sp_kc.log; SHARD=${SHARD:-0}; NSHARD=${NSHARD:-3}; BS=24
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
RB=${RB:-12}; TB=${TB:-16000}   # streams / token budget per batch (12 GB cards: 24 / 30000 ran out of memory)
python3 -c "
import sys
L = open('dpo_pool.jsonl').readlines()[300:]
open('shard.jsonl', 'w').writelines(l for i, l in enumerate(L) if i % $NSHARD == $SHARD)"
NB=$(( ($(wc -l < shard.jsonl) + BS - 1) / BS ))
echo "[kc] ready $(date -u) | shard $SHARD/$NSHARD: $(wc -l < shard.jsonl) problems, $NB blocks of $BS | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
weights() {   # newest trainer weights -> mem.pt; echoes the version (0 = the reference)
  local v; v=$(curl -sSfL -H "Authorization: Bearer $HF_TOKEN" "$H/archive/sp3/kvf/VERSION" 2>/dev/null | tr -d '[:space:]')
  [ -n "$v" ] || v=0
  if [ "$v" != "$(cat mem_v 2>/dev/null)" ]; then
    if [ "$v" = 0 ]; then cp ref.pt mem.pt; else get archive/sp3/spKVF_latest.pt mem_new.pt && mv mem_new.pt mem.pt || v=$(cat mem_v 2>/dev/null || echo 0); fi
    echo $v > mem_v; fi
  cat mem_v; }
gen() { local p=$1 o=$2 sd=$3; shift 3; [ -s $o ] && return; local rb=$RB tb=$TB
  for _try in 1 2 3; do   # out of memory: halve the batch and go again
    python3 sp_qwen3.py rollout --traces $p $M "$@" --max-len 12288 --roll-batch $rb --grpo-tok-budget $tb --seed $sd --roll-out $o --log $LG > run_kc.log 2>&1
    [ "$(wc -l < $o 2>/dev/null || echo 0)" = "$(wc -l < $p)" ] && return
    echo "[kc] gen $o (batch $rb): $(grep -E 'Error|out of memory' run_kc.log | tail -1 | cut -c1-120)"; rm -f $o; rb=$((rb / 2)); tb=$((tb / 2)); done
  return 1; }
b=$(cat kc_block 2>/dev/null || echo 0)
if [ "$MODE" = grpo ]; then   # GRPO rollout worker: per block the memory G times per problem (GUIDE=1: the raw window once too)
  G=${G:-12}; GUIDE=${GUIDE:-0}; BS=${GBS:-6}; NB=$(( ($(wc -l < shard.jsonl) + BS - 1) / BS ))
  [ -s kcg_block ] || echo ${KCG_START:-$(( $(cat kc_block 2>/dev/null || echo 0) * 24 / BS ))} > kcg_block   # carry on after the problems used so far (KCG_START: an extra worker on the far half of a shard)
  b=$(cat kcg_block)
  while [ $b -lt $NB ] && [ ! -e STOP_KC ]; do t0=$(date +%s)
    sed -n "$(( b * BS + 1 )),$(( (b + 1) * BS ))p" shard.jsonl > p_$b.jsonl
    [ -s p_$b.jsonl ] || break
    python3 -c "
import sys; L = open('p_$b.jsonl').readlines(); open('pg_$b.jsonl', 'w').writelines(l for l in L for _ in range($G))"
    v=$(weights); sd=$(( 7000 + SHARD * 1000 + b * 4 ))
    if [ "$GUIDE" = 1 ]; then gen p_$b.jsonl graw_$b.jsonl $sd --ckpt ref.pt --no-sp; else : > graw_$b.jsonl; fi
    gen pg_$b.jsonl gmem_${b}_v$v.jsonl $((sd + 1)) --ckpt mem.pt
    s=$(python3 - $b $v <<'PY2'
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
b, v = sys.argv[1], sys.argv[2]; ok = lambda r: '</think>' in r['text'] and correct(r)
raw = {json.loads(l)['id']: json.loads(l) for l in open(f'graw_{b}.jsonl')}
grp = {}
for l in open(f'gmem_{b}_v{v}.jsonl'):
    r = json.loads(l); g = grp.setdefault(r['id'], {'id': r['id'], 'src': r.get('src', ''), 'gold': r['gold'], 'prompt': r['prompt'], 'samples': [], 'guide': None})
    g['samples'].append(r['text'])
for k, g in grp.items():
    if k in raw and ok(raw[k]): g['guide'] = raw[k]['text']
open(f'grp_b{b}_v{v}.jsonl', 'w').writelines(json.dumps(g) + '\n' for g in grp.values())
G = list(grp.values()); rs = [[ok({'text': t, 'gold': g['gold']}) for t in g['samples']] for g in G]
mixed = sum(0 < sum(x) < len(x) for x in rs); allbad = sum(sum(x) == 0 for x in rs)
print(f"{len(G)} problems, memory right {sum(map(sum, rs))}/{sum(map(len, rs))}, closed {sum('</think>' in t for g in G for t in g['samples'])} | "
      f"mixed groups {mixed}, all wrong {allbad} (with a right guide {sum(1 for g, x in zip(G, rs) if sum(x) == 0 and g['guide'])})")
PY2
)
    put grp_b${b}_v$v.jsonl dpo_col_g/s$SHARD/g${G}_b${b}_v$v.jsonl
    b=$((b + 1)); echo $b > kcg_block
    echo "[kc] grpo G$G block $b/$NB ($(( ($(date +%s) - t0) / 60 )) min, memory weights v$v): $s"
    [ -n "$ONE" ] && exit 0
  done
  echo "[kc] JOB_DONE $(date -u)"; exit 0
fi
while [ $b -lt $NB ] && [ ! -e STOP_KC ]; do t0=$(date +%s)
  sed -n "$(( b * BS + 1 )),$(( (b + 1) * BS ))p" shard.jsonl > p_$b.jsonl
  v=$(weights); sd=$(( 5000 + SHARD * 1000 + b * 4 ))
  gen p_$b.jsonl raw_$b.jsonl $sd --ckpt ref.pt --no-sp
  gen p_$b.jsonl mem_$b.jsonl $((sd + 1)) --ckpt mem.pt
  python3 - $b <<'PY'
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
b = sys.argv[1]; ok = lambda r: '</think>' in r['text'] and correct(r)
raw = {json.loads(l)['id']: json.loads(l) for l in open(f'raw_{b}.jsonl')}; mem = {json.loads(l)['id']: json.loads(l) for l in open(f'mem_{b}.jsonl')}
P = {json.loads(l)['id']: l for l in open(f'p_{b}.jsonl')}
open(f'todo_{b}.jsonl', 'w').writelines(P[k] for k in P if k in raw and k in mem and not ok(raw[k]) and not ok(mem[k]))
PY
  if [ -s todo_$b.jsonl ]; then gen todo_$b.jsonl raw2_$b.jsonl $((sd + 2)) --ckpt ref.pt --no-sp; gen todo_$b.jsonl mem2_$b.jsonl $((sd + 3)) --ckpt mem.pt; fi
  s=$(python3 dpo_pairs.py pairs_b${b}_v$v.jsonl raw_$b.jsonl $(ls raw2_$b.jsonl 2>/dev/null) -- mem_$b.jsonl $(ls mem2_$b.jsonl 2>/dev/null))
  for f in raw_$b mem_$b raw2_$b mem2_$b; do [ -s $f.jsonl ] && put $f.jsonl dpo_col/s$SHARD/$f.jsonl; done
  put pairs_b${b}_v$v.jsonl dpo_col/s$SHARD/pairs_b${b}_v$v.jsonl
  b=$((b + 1)); echo $b > kc_block
  echo "[kc] block $b/$NB ($(( ($(date +%s) - t0) / 60 )) min, memory weights v$v): $s"
  [ -n "$ONE" ] && exit 0   # the trainer runs one block at a time while it waits for pairs
done
echo "[kc] JOB_DONE $(date -u)"
