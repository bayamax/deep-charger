#!/bin/bash
# One evaluation of the newest GRPO weights on a rollout worker (so the trainer keeps training): the 60 problems at seed $SEED
# with archive/sp3/spKVF_latest.pt (version from kvf/VERSION), uploaded as archive/sp3/kvg/eval_v<V>_s<SEED>.jsonl.
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && return 0; sleep 60; done; }
[ -s rl_eval60.jsonl ] || { curl -sS -o rl_eval120.jsonl "$RAW/ctl/rl_eval120.jsonl?nocache=$(date +%s)"; python3 -c "
rs = open('rl_eval120.jsonl').readlines(); open('rl_eval60.jsonl', 'w').writelines(rs[i] for i in range(len(rs)) if i % 24 < 12)"; }
v=$(curl -sSfL -H "Authorization: Bearer $HF_TOKEN" "$H/archive/sp3/kvf/VERSION" | tr -d '[:space:]')
[ "$v" = "$(cat mem_v 2>/dev/null)" ] && [ -s mem.pt ] || { curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o mem.pt "$H/archive/sp3/spKVF_latest.pt" && echo $v > mem_v; }
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
o=eval_v${v}_s$SEED.jsonl
python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $M --ckpt mem.pt --max-len 12288 --roll-batch 12 --grpo-tok-budget 16000 --seed $SEED --roll-out $o --log sp_kve.log > run_kve.log 2>&1
put $o kvg/$o
echo "[kc] eval v$v seed $SEED: $(python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open('$o')]; print(f'{sum(correct(r) for r in rs)}/{len(rs)} correct, closed {sum(\"</think>\" in r[\"text\"] for r in rs)}')") $(date -u)"
