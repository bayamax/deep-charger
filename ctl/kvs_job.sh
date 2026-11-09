#!/bin/bash
# The eviction rules judged by KL, on paper (the user: KL, not attention coverage; then the learned selector, option 5).
# sp_qwen3.py kvsel: one full-context pass per trace, then at every k-th chunk boundary each rule keeps B older entries beside
# the prompt and the newest rw=1024, the next chunk is read on that cache and its KL to the full-context teacher is the score.
# Phase 1 (held-out, the longest math + competition traces): recent / random / cumulative / mean / EMA / last attention (each
# also per layer) and 'abl' = the exact block-ablation label (removing each block of 8 consecutive entries) - the headroom a
# KL-trained selector could reach. Phase 2: the training dump - the same labels (blocks of 16) on long pool traces, features
# per entry (attention history, age, position, token, key / value norms) saved for tools/kvsel_train.py. Both dumps go to the hub.
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && return 0; sleep 60; done; echo "[ks] upload of $2 FAILED"; }
for _t in $(seq 60); do curl -sS -o sp_ks.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q "def kvsel_trace" sp_ks.py && break; sleep 120; done
M="--rw 1024 --C 128 --maxd 0 --att-layers 8-19"
python3 -c "
import json
L = [l for f in ('held_math.jsonl', 'held_comp.jsonl') for l in open(f)]; L.sort(key=lambda l: -len(json.loads(l)['text']))
open('kvs_held.jsonl', 'w').writelines(L[:${NH:-8}])
rs = [l for l in open('rm_pool.jsonl') if json.loads(l).get('src', '?').split('-')[0] in ('math', 'comp')][:3000]
rs.sort(key=lambda l: -len(json.loads(l)['text'])); open('kvs_train.jsonl', 'w').writelines(rs[:${NT:-24}])
print('[ks] held-out', len(L[:${NH:-8}]), 'traces, training', len(rs[:${NT:-24}]))"
echo "[ks] ready $(date -u) | kvsel in the script: $(grep -c 'def kvsel_trace' sp_ks.py) | $M | budgets ${BS:-512,1024}"
if [ ! -s kvsel_held.pt ]; then t0=$(date +%s)
  python3 sp_ks.py kvsel --traces kvs_held.jsonl --max-len 8192 $M --kvsel-budgets ${BS:-512,1024} --kvsel-every 4 --kvsel-abl 8 \
    --kvsel-rules recent,random,cum,mean,ema0.9,ema0.98,last,abl,fut --kvsel-dump kvsel_held.pt --log sp_kvs.log > run_kvs.log 2>&1
  grep -E "^\[sp\] kvsel S|Error|Traceback" run_kvs.log | tail -2 | cut -c1-1600 | sed 's/^\[sp\]/[ks] held-out/'
  echo "[ks] phase 1 done ($(( $(date +%s) - t0 )) s) $(date -u)"; put kvsel_held.pt kvs/kvsel_held.pt; put sp_kvs.log kvs/kvs.log
fi
if [ ! -s kvsel_train.pt ]; then t0=$(date +%s)
  python3 sp_ks.py kvsel --traces kvs_train.jsonl --max-len 8192 $M --kvsel-budgets 512 --kvsel-every 6 --kvsel-abl 16 \
    --kvsel-rules recent,mean,ema0.9,abl,fut --kvsel-layerwise 0 --kvsel-dump kvsel_train.pt --log sp_kvs.log > run_kvs2.log 2>&1
  grep -E "^\[sp\] kvsel S|Error|Traceback" run_kvs2.log | tail -2 | cut -c1-600 | sed 's/^\[sp\]/[ks] training set/'
  echo "[ks] phase 2 done ($(( $(date +%s) - t0 )) s) $(date -u)"; put kvsel_train.pt kvs/kvsel_train.pt; put sp_kvs.log kvs/kvs.log
fi
echo "[ks] JOB_DONE $(date -u)"
