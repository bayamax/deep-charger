#!/bin/bash
# DPO data, round 2 (the user: run the planned 300 through first, then a second round). Round 1 gave 2 pairs in its first 50
# problems (raw window right & memory wrong); most problems are right for both or wrong for both. Round 2: the problems where
# the memory failed (unclosed or wrong) and the raw window failed too get one more raw-window generation at another seed -
# a raw success there makes a pair - and the memory is redrawn on them too (the user: redraw both), a memory success there
# makes a pair the other way round.
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
LG=/root/w/sp_kvd.log
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
[ -s dpo_r2_todo.jsonl ] || python3 - <<'PY'
import json, glob, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
ok = lambda r: '</think>' in r['text'] and correct(r)
raw = {r['id']: r for f in sorted(glob.glob('dpo_raw_b*.jsonl')) for l in open(f) for r in [json.loads(l)]}
mem = {r['id']: r for f in sorted(glob.glob('dpo_mem_b*.jsonl')) for l in open(f) for r in [json.loads(l)]}
P = {json.loads(l)['id']: l for l in open('dpo_prompts300.jsonl')}
todo = [k for k in mem if k in raw and not ok(mem[k]) and not ok(raw[k])]
n_pair = sum(1 for k in mem if k in raw and not ok(mem[k]) and ok(raw[k]))
open('dpo_r2_todo.jsonl', 'w').writelines(P[k] for k in todo)
print(f"[kd] round 1: {len(mem)} problems | raw right {sum(ok(raw[k]) for k in raw)}, memory right {sum(ok(mem[k]) for k in mem)} | pairs {n_pair} | round 2 redraws both sides on {len(todo)} problems")
PY
out=dpo_raw2_all.jsonl
if [ ! -s $out ]; then
  python3 sp_qwen3.py rollout --traces dpo_r2_todo.jsonl $M --ckpt dpo_ref.pt --no-sp --max-len 12288 --roll-batch 48 --grpo-tok-budget 60000 --seed 1011 --roll-out $out --log $LG > run_kd2gen.log 2>&1
  grep -E "Error|Traceback" run_kd2gen.log | tail -1 | sed "s/^/[kd] gen round 2: /"
fi
put $out dpo/$out
out2=dpo_mem2_all.jsonl
if [ ! -s $out2 ]; then
  python3 sp_qwen3.py rollout --traces dpo_r2_todo.jsonl $M --ckpt dpo_ref.pt --max-len 12288 --roll-batch 48 --grpo-tok-budget 60000 --seed 1012 --roll-out $out2 --log $LG > run_kd2gen.log 2>&1
  grep -E "Error|Traceback" run_kd2gen.log | tail -1 | sed "s/^/[kd] gen round 2 memory: /"
fi
put $out2 dpo/$out2
python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
for side, f in (('raw window', '$out'), ('memory', '$out2')):
    rs = [json.loads(l) for l in open(f)]
    print(f'[kd] round 2 {side}: {sum(\"</think>\" in r[\"text\"] and correct(r) for r in rs)}/{len(rs)} right and closed')"
date -u > DONE_KD2; put DONE_KD2 dpo/DONE_KD2; echo "[kd] ROUND2_DONE $(date -u)"
