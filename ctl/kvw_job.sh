#!/bin/bash
# After the --ctl-para job: the same weights (its last segment) decoded with --para-window - each stream runs twice, with its
# memory and with the memory masked out, and the token opening a paragraph is sampled from the masked twin. --ctl-para
# segment 1 moved the unclosed count 18 -> 13 (raw window 8 / 10) at 40/60; this checks the decoding-time version on top.
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
for _t in $(seq 60); do curl -sS -o sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q '"--para-window"' sp_qwen3.py && break; sleep 120; done
LG=/root/w/sp_kvw.log
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
echo "[kw] ready $(date -u) | para-window in the script: $(grep -c '"--para-window"' sp_qwen3.py) | weights: ctl-para segment $(cat kvp_seg)"
score() { python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open(sys.argv[1])]
print('[kw] ' + sys.argv[2] + f': {sum(correct(r) for r in rs)}/{len(rs)} correct, closed {sum(\"</think>\" in r[\"text\"] for r in rs)}')" "$1" "$2"; }
for sd in 1 2; do
  [ -s kvw_s$sd.jsonl ] && continue
  python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $M --para-window --max-len 12288 --ckpt kvp.pt --roll-batch 24 --grpo-tok-budget 60000 --seed $sd --roll-out kvw_s$sd.jsonl --log $LG > run_kwgen.log 2>&1
  grep -E "Error|Traceback" run_kwgen.log | tail -1 | sed "s/^/[kw] gen s$sd: /"; grep "paragraph openings from" $LG | tail -1 | grep -oP "paragraph openings.*" | sed "s/^/[kw] s$sd /"
  score kvw_s$sd.jsonl "ctl-para segment $(cat kvp_seg) + para-window decoding seed $sd"; put kvw_s$sd.jsonl kvw/kvw_s$sd.jsonl
done
put $LG kvw/kvw.log; date -u > DONE_KW; put DONE_KW kvw/DONE_KW; echo "[kw] JOB_DONE $(date -u)"
