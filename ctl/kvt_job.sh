#!/bin/bash
# Option 5, the learned KV selector, end to end and unattended: once the KL harness (ctl/kvs_job.sh) has written its dumps,
# train the selector on the training dump (tools/kvsel_train.py: pairwise ranking on the exact ablation labels), score it on the
# held-out traces in the same KL harness beside the attention rules (rule mlp:kvsel_mlp.pt), then - whatever the outcome, the
# user wants the number - the 60 problems at seed 1 with --kv-keep 512 and the selector, with the harness's best attention
# rule (EMA 0.98, each layer its own choice) and with EMA 0.98 single-set (the equal-total window baselines run in ctl/kvk_job.sh). One GPU: the 60-problem runs wait for each other.
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && return 0; sleep 60; done; echo "[kt] upload of $2 FAILED"; }
score() { python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open(sys.argv[1])]
print(f'{sum(correct(r) for r in rs)}/{len(rs)} correct, closed {sum(\"</think>\" in r[\"text\"] for r in rs)}')" "$1"; }
for f in tools/kvsel_feat.py tools/kvsel_train.py; do curl -sS -o $(basename $f) "$RAW/$f?nocache=$(date +%s)"; done
for _t in $(seq 60); do curl -sS -o sp_kt.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q "kv-select" sp_kt.py && break; sleep 120; done
until [ -s kvsel_train.pt ] && grep -q "\[ks\] JOB_DONE" w.log; do sleep 120; done
M="--rw 1024 --C 128 --maxd 0 --att-layers 8-19"
for lb in fut abl; do [ -s kvsel_$lb.pt ] && continue; t0=$(date +%s)
  python3 kvsel_train.py kvsel_train.pt kvsel_$lb.pt --label $lb --epochs ${EP:-30} --B 512 --block 16 > run_kt_train_$lb.log 2>&1
  echo "[kt] selector ($lb label) trained ($(( $(date +%s) - t0 )) s): $(grep -E '^epoch|positions' run_kt_train_$lb.log | tail -2 | tr '\n' ' ' | cut -c1-400)"; put kvsel_$lb.pt kvs/kvsel_$lb.pt
done
if ! grep -q "\[kt\] held-out" w.log; then
  python3 sp_kt.py kvsel --traces kvs_held.jsonl --max-len 8192 $M --kvsel-budgets 512,1024 --kvsel-every 4 --kvsel-layerwise 0 \
    --kvsel-rules recent,mean,ema0.9,ema0.98,fut,mlp:kvsel_fut.pt,mlp:kvsel_abl.pt --log sp_kvt.log > run_kt_eval.log 2>&1
  echo "[kt] held-out $(grep -E '^\[sp\] kvsel S|Error|Traceback' run_kt_eval.log | tail -1 | sed 's/^\[sp\] kvsel [^:]*: //' | cut -c1-900)"
fi
best=$(grep "\[kt\] held-out" w.log | tail -1 | grep -oE "mlp:kvsel_(fut|abl).pt B=512 [0-9.]+" | sort -k3 -n | head -1 | grep -oE "kvsel_(fut|abl).pt"); best=${best:-kvsel_fut.pt}
# KT_SPECS picks the 60-problem runs (names: mlp, ema098L, ema098); RB the streams. Serial 109: the selector lost to EMA 0.98 on
# held-out KL (0.057 vs 0.050), so its 60-problem run is dropped - one negative is enough (the user) - and only ema098L runs.
spec_of() { case $1 in mlp) echo "--kv-select mlp:$best";; ema098L) echo "--kv-select ema0.98 --kv-layerwise";; ema098) echo "--kv-select ema0.98";; esac; }
for n in ${KT_SPECS:-ema098L}; do fl=$(spec_of $n)
  o=kvk_b512_${n}_s1.jsonl; [ -s $o ] && continue
  until ! pgrep -f "sp_qwen3.py rollout" > /dev/null && ! pgrep -f "sp_kt.py rollout" > /dev/null; do sleep 60; done
  t0=$(date +%s); rm -f $o.part
  python3 sp_kt.py rollout --traces rl_eval60.jsonl --no-sp --kv-evict --kv-keep 512 --kv-keep-path $fl --att-layers 8-19 --rw 1024 --C 128 --maxd 0 --max-len 12288 \
    --roll-batch ${RB:-24} --kv-budget 90000 --seed 1 --roll-out $o.part --log sp_kvt.log > run_kt_$n.log 2>&1
  if [ -s $o.part ]; then mv $o.part $o; put $o kvk/$o
    echo "[kt] B=512 $n rw 1024 seed 1: $(score $o) ($(( $(date +%s) - t0 )) s) | refs: mean rule 45, window alone 1024 43 / 1536 47 / 2048 48, raw window 42 / 44 $(date -u)"
  else echo "[kt] B=512 $n seed 1 FAILED: $(grep -E 'Error' run_kt_$n.log | tail -1 | cut -c1-200)"; fi
done
echo "[kt] JOB_DONE $(date -u)"
