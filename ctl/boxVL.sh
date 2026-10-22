#!/bin/bash
# Box VL (3060, pytorch image): the NPC lesson applied to deep-charger - N sampled rollouts per question, the answer by
# vote. 2.1.0 (the adopted model: 2.0.0's 4-bit base + the mtg7 LoRA) on the search held-out (eval300, 150 questions)
# with the local store, 8 passes of pool_eval.py (each pass one rollout per question, temperature 0.6, the lineage's
# settings) in two processes; the passes go to the hub (archive/vote/), the vote is counted offline.
BOXVL_SERIAL=6
if [ -f /root/.boxvl_serial ] && [ "$(cat /root/.boxvl_serial)" -gt "$BOXVL_SERIAL" ] 2>/dev/null; then echo "BOXVL_STALE $BOXVL_SERIAL"; exit 0; fi
echo $BOXVL_SERIAL > /root/.boxvl_serial
if [ ! -e /root/.vl_pre ]; then touch /root/.vl_pre
  echo "[vl] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader)"
  echo "[vl] preflight hub: $(curl -sSL -o /dev/null -m 90 -w '%{size_download} bytes in %{time_total}s' https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/versions/2.0.0/base/model.safetensors -r 0-104857599 2>&1)"
fi
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
mkdir -p /root/work/localsearch /root/work/fft_out
for f in pool_eval.py web_search.py localsearch/search.py localsearch/store.py localsearch/ivf.py localsearch/terms.py localsearch/pq.py localsearch/embed.py; do
  for try in 1 2 3; do curl -sS -o /root/work/$f "$RAW/$f?nocache=$(date +%s)" && break; sleep 5; done
done
mkdir -p /root/work/runtime; cp /root/work/web_search.py /root/work/runtime/web_search.py 2>/dev/null; touch /root/work/runtime/__init__.py
[ -f /root/.bootstrapped ] || { python3 -m pip install -q "transformers>=4.51,<4.57" accelerate "huggingface_hub[hf_xet]>=0.34,<1.0" safetensors peft zstandard onnxruntime datasets certifi 2>&1 | tail -1; touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"; }
cat > /root/status.sh <<'ST'
grep -E "^\[(vl|eval|load|cfg|search|hub)|EVAL_DONE|Error|Traceback|out of memory" /root/vl.log 2>/dev/null | tail -n 4 | cut -c1-300
for f in /root/vl_p0.log /root/vl_p1.log /root/vl_t0.log /root/vl_t1.log; do [ -s $f ] && { echo "--- $(basename $f)"; tail -n 3 $f | cut -c1-300; }; done
pgrep -fc "pool_eval.py" | sed "s/^/pool_eval processes: /"
for i in 0 1; do f=/root/work/vote_p$i.jsonl; [ -s $f ] && echo "pass file $i: $(wc -l < $f) rows"; done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
python3 -c 'import datasets, peft, zstandard, onnxruntime' 2>/dev/null || python3 -m pip install -q datasets peft zstandard onnxruntime certifi 2>&1 | tail -1
# serial 2: job 2, after the vote passes - where the adopted model fails on TRAINING questions (the teacher's data is
# aimed there): 2.1.0 on the first 600 of the training items in teach_traj.py's order (corpus questions with a gold of
# at most six words, the held-out's 300 left out, shuffled with seed 0), one sampled rollout each, the local store.
pkill -f 'vljob2.sh' 2>/dev/null   # earlier serials' job-2 waiters, so only this one follows the passes
if [ ! -e /root/.vl_s6 ]; then touch /root/.vl_s6
  cat > /root/vljob2.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/work
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/vote/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
while [ ! -s /root/DONE ]; do sleep 60; done
echo "[vl] job 2: the adopted model on 600 training questions $(date -u)"
[ -s corpus.jsonl ] || curl -sSfL -o corpus.jsonl $H/box_recover/corpus.jsonl
python3 - <<'PY'
import json, random
rows = [json.loads(l) for l in open("corpus.jsonl") if l.strip()]; ex = set()
for l in open("eval300.jsonl"):
    try: ex.add(json.loads(l)["q"])
    except Exception: pass
items = [(r["q"], r["gold"].strip()) for r in rows if r.get("q") and (r.get("gold") or "").strip() and len(r["gold"].split()) <= 6 and r["q"] not in ex]
random.Random(0).shuffle(items)
with open("train600.jsonl", "w") as f:
    for q, g in items[:600]: f.write(json.dumps({"q": q, "gold": g}) + "\n")
print(f"[vl] {len(items)} training items, the first 600 written", flush=True)
PY
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 SP_LOCAL_STORE=/root/wiki_store SP_LOCAL_MODEL=/root/bge-small OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
( while sleep 600; do for i in 0 1; do [ -s /root/work/train600_$i.jsonl ] && put /root/work/train600_$i.jsonl train600_$i.jsonl > /dev/null; done; done ) & MIR=$!
python3 - <<'PY'
rows = open("train600.jsonl").readlines()
for i in range(2): open(f"train600_q$i.jsonl", "w").writelines(rows[i::2])
PY
for i in 0 1; do
  ( env $ENV python3 /root/work/pool_eval.py /root/mtg7_s40.safetensors /root/work/train600_q$i.jsonl /root/work/train600_$i.jsonl $EVARGS --n 300 --tag "[train600 $i]" 2>&1 | grep -E "^\[(eval|load|cfg|search)|EVAL_DONE|Error|Traceback|out of memory" | sed "s/^/[t$i] /" ) > /root/vl_t$i.log 2>&1 &
  sleep 90
done
wait $(jobs -p | grep -v $MIR) 2>/dev/null; kill $MIR 2>/dev/null
cat /root/vl_t0.log /root/vl_t1.log | grep -E "EVAL_DONE|Traceback|Error" | cut -c1-300
for i in 0 1; do put /root/work/train600_$i.jsonl train600_$i.jsonl; done
put /root/vl.log run2.log; date -u > /root/DONE2; put /root/DONE2 DONE2; echo "[vl] VLJOB2_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vljob2.sh >> /root/vl.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[vl] s2: job 2 (the model on 600 training questions) queued after the vote passes $(date -u)"
fi
if [ ! -e /root/.vljob6 ]; then touch /root/.vljob6
  cat > /root/vljob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/work
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/vote/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
get() { [ -s $2 ] || curl -sSfL -o $2 "$1" || { rm -f $2; echo "[vl] could not get $1"; return 1; }; }
for f in grpo_e2e_torch.py pooler_torch.py; do get $H/box_recover/scripts/$f /root/work/$f; done
python3 - <<'PY'   # transformers 4.56+ rejects a positive crop (keep N); the harness's crop_cache made version-proof
import re
p = "/root/work/grpo_e2e_torch.py"; s = open(p).read()
new = ('def crop_cache(past, length):\n    if hasattr(past, "crop"):\n        L = past.get_seq_length()\n        if L > length: past.crop(length - L)\n        return past\n')
s2 = re.sub(r'def crop_cache\(past, length\):\n(    .*\n)+?        past\.crop\(length\); return past\n', new, s, count=1)
print("[patch] crop_cache", "patched" if s2 != s else "NOT FOUND"); open(p, "w").write(s2)
PY
get $H/fft_out/pooler.pt /root/work/fft_out/pooler.pt
mkdir -p /root/gptq_hf_gq14; for f in config.json generation_config.json special_tokens_map.json tokenizer.json tokenizer_config.json model.safetensors; do get $H/versions/2.0.0/base/$f /root/gptq_hf_gq14/$f; done
get $H/versions/2.1.0/lora.safetensors /root/mtg7_s40.safetensors
get $H/grpo_assets/mus_run/eval_step200/eval_heldout_300q_g2.jsonl /root/work/eval300.jsonl
mkdir -p /root/wiki_store /root/bge-small
for f in blocks.idx titles.txt meta.json ivf_centroids.npy ivf_offsets.npy ivf_order.npy emb.bin emb_ivf.bin lex.sqlite docs.bin; do get $H/localsearch/wiki_en_20231101/$f /root/wiki_store/$f; done
for f in config.json model_int8.onnx special_tokens_map.json tokenizer.json tokenizer_config.json; do get $H/localsearch/bge-small-en-v1.5/$f /root/bge-small/$f; done
echo "[vl] assets in place $(date -u); store $(du -sh /root/wiki_store | cut -f1)"
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 SP_LOCAL_STORE=/root/wiki_store SP_LOCAL_MODEL=/root/bge-small OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600 --n 150"
CK=/root/mtg7_s40.safetensors
# two processes, four passes each; a pass = one sampled rollout of each of the 150 questions (pool_eval runs distinct
# questions once per output file, so each pass has its own file); passes mirrored to the hub as they fill
( while sleep 600; do for i in 0 1; do for k in 0 1 2 3; do f=/root/work/vote_p${i}_$k.jsonl; [ -s $f ] && put $f vote_p${i}_$k.jsonl > /dev/null; done; done; done ) & MIR=$!
for i in 0 1; do
  ( for k in 0 1 2 3; do f=/root/work/vote_p${i}_$k.jsonl; [ "$(wc -l < $f 2>/dev/null || echo 0)" -ge 150 ] && continue
      env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl $f $EVARGS --tag "[vote p$i k$k]" 2>&1 | grep -E "^\[(eval|load|cfg|search)|EVAL_DONE|Error|Traceback|out of memory" | sed "s/^/[p$i k$k] /"
    done ) > /root/vl_p$i.log 2>&1 &
  sleep 90
done
wait $(jobs -p | grep -v $MIR) 2>/dev/null; kill $MIR 2>/dev/null
cat /root/vl_p0.log /root/vl_p1.log | grep -E "EVAL_DONE|Traceback|Error" | cut -c1-300
for i in 0 1; do for k in 0 1 2 3; do put /root/work/vote_p${i}_$k.jsonl vote_p${i}_$k.jsonl; done; done
cat /root/vl_p0.log /root/vl_p1.log > /root/vl_runs.log; put /root/vl_runs.log vl_runs.log; put /root/vl.log run.log
[ "$(cat /root/work/vote_p*_*.jsonl 2>/dev/null | wc -l)" -ge 100 ] || { echo "[vl] no pass produced rows - not done"; exit 1; }
date -u > /root/DONE; put /root/DONE DONE; echo "[vl] VLJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vljob.sh >> /root/vl.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VLJOB_LAUNCHED $(date -u)"
fi
# CTL-END
