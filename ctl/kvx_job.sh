#!/bin/bash
# One evaluation of an older GRPO version on a rollout worker (the user: did something go wrong at v19, when Adam's state
# started to carry across rounds?): the 60 problems at seed $SEED with archive/sp3/spKVF_latest.pt as of hub commit $REV
# (version $VN), into its own file (the worker's mem.pt / mem_v stay untouched), uploaded as archive/sp3/kvg/evalx_$VN_s$SEED.jsonl.
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && return 0; sleep 60; done; }
[ -s rl_eval60.jsonl ] || { curl -sS -o rl_eval120.jsonl "$RAW/ctl/rl_eval120.jsonl?nocache=$(date +%s)"; python3 -c "
rs = open('rl_eval120.jsonl').readlines(); open('rl_eval60.jsonl', 'w').writelines(rs[i] for i in range(len(rs)) if i % 24 < 12)"; }
[ -s evx_$VN.pt ] || python3 -c "
import shutil, sys; from huggingface_hub import hf_hub_download as d
shutil.copy(d('$R', 'archive/sp3/spKVF_latest.pt', revision='$REV', local_dir='evx_dl'), 'evx_$VN.pt'); shutil.rmtree('evx_dl')"
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
o=evalx_${VN}_s$SEED.jsonl
python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $M --ckpt evx_$VN.pt --max-len 12288 --roll-batch ${RB:-12} --grpo-tok-budget ${TB:-16000} --seed $SEED --roll-out $o --log sp_kvx.log > run_kvx.log 2>&1
put $o kvg/$o; rm -f evx_$VN.pt
echo "[kc] eval $VN seed $SEED: $(python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open('$o')]; print(f'{sum(correct(r) for r in rs)}/{len(rs)} correct, closed {sum(\"</think>\" in r[\"text\"] for r in rs)}')") $(date -u)"
