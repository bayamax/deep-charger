#!/bin/bash
# The long-range eval of the bounded KV (the user: test it on varied problems, multi-turn too). ctl/mt_eval.jsonl
# (tools/mt_build.py, 160 rows): only a short system turn is pinned; every user turn is fed into the stream as forced tokens, so
# problem statements, documents and earlier turns can be evicted like the answer (the 60-problem eval pinned the problem, which is
# why nothing ever needed far memory there). Tasks: unpin 30 (math, statement not pinned), fact 30 (5 details, a long math turn,
# then one detail), chain 30 (two problems, then the sum of your two answers), code 30 (a function, a long math turn, then f(x)),
# docq 20 / docf 20 (a 100-entry log of ~3.1k tokens, the question after it / before it, the asked entry among the first 8).
# Conditions CONDS "name:rw:B[:flags]" (flags comma-separated, e.g. --kv-policy=struct,--kv-pool=7): full context (rw above the longest dialogue), the window alone, the window + B older entries by
# mean attention received - all in the same code path (--kv-evict --kv-keep B --kv-keep-path), seed 1. Scored by tools/mt_score.py.
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && return 0; sleep 60; done; echo "[mt] upload of $2 FAILED"; }
for _t in $(seq 60); do curl -sS -o sp_new.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q "${NEED:-multi-turn rows need}" sp_new.py && { cp sp_new.py sp_mt.py; break; }; sleep 120; done
for f in tools/sp_mathcheck.py tools/mt_score.py ctl/mt_eval.jsonl; do curl -sS -o $(basename $f) "$RAW/$f?nocache=$(date +%s)"; done
echo "[mt] ready $(date -u) | rows $(wc -l < mt_eval.jsonl) | forced turns in the script: $(grep -c 'multi-turn rows need' sp_mt.py) | conditions ${CONDS}"
for c in $CONDS; do IFS=: read n rw B fl <<< "$c"; fl=${fl//,/ }
  o=mt_${n}_s1.jsonl; [ -s $o ] && { echo "[mt] $n already: $(python3 mt_score.py $o)"; continue; }
  t0=$(date +%s); rm -f $o.part
  python3 sp_mt.py rollout --traces mt_eval.jsonl --no-sp --kv-evict --kv-keep $B --kv-keep-path $fl --att-layers 8-19 --rw $rw --C 128 --maxd 0 --max-len 13000 \
    --roll-batch ${RB:-24} --kv-budget ${KB:-110000} --seed 1 --roll-out $o.part --log sp_mt.log > run_mt.log 2>&1
  if [ -s $o.part ]; then mv $o.part $o; put $o mt/$o
    echo "[mt] $n (rw $rw, B $B${fl:+, $fl}; $(grep -oE "budget ~[0-9]+" sp_mt.log | tail -1)): $(python3 mt_score.py $o | cut -d: -f2-) ($(( $(date +%s) - t0 )) s) $(date -u)"
  else echo "[mt] $n FAILED: $(grep -E 'Error|error' run_mt.log | tail -1 | cut -c1-200)"; fi
  [ -e STOP_MT ] && break
done
echo "[mt] JOB_DONE $(date -u)"
