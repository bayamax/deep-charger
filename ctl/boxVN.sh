#!/bin/bash
# Box VN (3060, pytorch image): the NPC recipe on the search model - the teacher's trajectories (DeepSeek in the
# student's own search environment, the local store, rules-filtered: tools/teach_traj.py) learned into 2.1.0 by
# memfit.py --objective ce (the served pages masked, LoRA r16, mem10's settings), then the search held-out (eval300,
# 150 questions x 2) with the local store. The teacher rows arrive on the hub (archive/teach/traj_rows.jsonl) from the
# session's own machine; the job waits for them. Results to archive/teach/.
BOXVN_SERIAL=4
if [ -f /root/.boxvn_serial ] && [ "$(cat /root/.boxvn_serial)" -gt "$BOXVN_SERIAL" ] 2>/dev/null; then echo "BOXVN_STALE $BOXVN_SERIAL"; exit 0; fi
echo $BOXVN_SERIAL > /root/.boxvn_serial
if [ ! -e /root/.vn_pre ]; then touch /root/.vn_pre
  echo "[vn] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader)"
  echo "[vn] preflight hub: $(curl -sSL -o /dev/null -m 90 -w '%{size_download} bytes in %{time_total}s' https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/versions/2.0.0/base/model.safetensors -r 0-104857599 2>&1)"
fi
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
mkdir -p /root/work/localsearch /root/work/fft_out /root/work/runtime
for f in pool_eval.py memfit.py web_search.py localsearch/search.py localsearch/store.py localsearch/ivf.py localsearch/terms.py localsearch/pq.py localsearch/embed.py; do
  for try in 1 2 3; do curl -sS -o /root/work/$f "$RAW/$f?nocache=$(date +%s)" && break; sleep 5; done
done
cp /root/work/web_search.py /root/work/runtime/web_search.py 2>/dev/null; touch /root/work/runtime/__init__.py
[ -f /root/.bootstrapped ] || { python3 -m pip install -q "transformers>=4.51,<4.57" accelerate "huggingface_hub[hf_xet]>=0.34,<1.0" safetensors peft zstandard onnxruntime datasets certifi 2>&1 | tail -1; touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"; }
cat > /root/status.sh <<'ST'
grep -E "^\[(vn|eval|load|cfg|search|hub|memfit|pooler|lora|step)|MEMFIT_DONE|EVAL_DONE|Error|Traceback|out of memory" /root/vn.log 2>/dev/null | tail -n 6 | cut -c1-300
grep -E "^\[step|val" /root/memfit_teach1.log 2>/dev/null | tail -n 2 | cut -c1-200
for i in 0 1; do f=/root/work/teach1_ev$i.jsonl; [ -s $f ] && echo "eval file $i: $(wc -l < $f) rows"; done
for t in dc210 teach1; do f=/root/work/uc_$t.jsonl; [ -s $f ] && echo "uc file $t: $(wc -l < $f) rows"; done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
python3 -c 'import datasets, peft, zstandard, onnxruntime' 2>/dev/null || python3 -m pip install -q datasets peft zstandard onnxruntime certifi 2>&1 | tail -1
# serial 4: the use-case bench (real WildChat prompts, tools/uc) - job 2, after job 1: 2.1.0 and teach1 answer the 301
# prompts the way the app runs (search on, the local store, the reply after </think> up to 600 tokens), one process each;
# the replies go to archive/uc/ for the pairwise judge against 4o-mini.
if [ ! -e /root/.vn_s4 ]; then touch /root/.vn_s4
  cat > /root/vnjob2.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/work
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/uc/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
while [ ! -s /root/DONE ]; do sleep 60; done
curl -sSfL -o /root/work/uc_bench.jsonl "$H/archive/uc/uc_bench.jsonl?$(date +%s)"
echo "[vn] job 2: the use-case bench, $(wc -l < /root/work/uc_bench.jsonl) prompts, 2.1.0 and teach1 $(date -u)"
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 SP_LOCAL_STORE=/root/wiki_store SP_LOCAL_MODEL=/root/bge-small OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600 --n 400"
( while sleep 600; do for t in dc210 teach1; do [ -s /root/work/uc_$t.jsonl ] && put /root/work/uc_$t.jsonl uc_$t.jsonl > /dev/null; done; done ) & MIR=$!
for spec in dc210:/root/mtg7_s40.safetensors teach1:/root/teach1.safetensors; do t=${spec%%:*}; ck=${spec#*:}; [ -s $ck ] || continue
  ( env $ENV python3 /root/work/pool_eval.py $ck /root/work/uc_bench.jsonl /root/work/uc_$t.jsonl $EVARGS --tag "[uc $t]" 2>&1 | grep -E "^\[(eval|load|cfg)|EVAL_DONE|Error|Traceback|out of memory" | sed "s/^/[uc $t] /" ) > /root/vn_uc_$t.log 2>&1 &
  sleep 90
done
wait $(jobs -p | grep -v $MIR) 2>/dev/null; kill $MIR 2>/dev/null
cat /root/vn_uc_*.log | grep -E "EVAL_DONE|Traceback|Error" | cut -c1-300
for t in dc210 teach1; do [ -s /root/work/uc_$t.jsonl ] && put /root/work/uc_$t.jsonl uc_$t.jsonl; done
put /root/vn.log run_uc.log; date -u > /root/DONE_UC; put /root/DONE_UC DONE_UC; echo "[vn] VNJOB2_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vnjob2.sh >> /root/vn.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[vn] s4: job 2 (the use-case bench) queued after job 1 $(date -u)"
fi
if [ ! -e /root/.vnjob ]; then touch /root/.vnjob
  cat > /root/vnjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/work
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/teach/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
get() { [ -s $2 ] || curl -sSfL -o $2 "$1" || { rm -f $2; echo "[vn] could not get $1"; return 1; }; }
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
echo "[vn] assets in place $(date -u)"
until curl -sSfL -o /root/work/traj_rows.jsonl "$H/archive/teach/traj_rows.jsonl?$(date +%s)" && [ -s /root/work/traj_rows.jsonl ] && curl -sSfL -o /dev/null "$H/archive/teach/ROWS_READY"; do sleep 120; done
echo "[vn] teacher rows: $(wc -l < /root/work/traj_rows.jsonl) $(date -u)"
export SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 /root/work/memfit.py --objective ce --ce-native 1 --student full --ckpt /root/mtg7_s40.safetensors --data /root/work/traj_rows.jsonl --out /root/teach1.safetensors --log /root/memfit_teach1.log --steps 600 --val-every 100 --lr 0 --lr-lora 5e-6 > /root/memfit_teach1_run.log 2>&1
grep -E "MEMFIT_DONE|Traceback|Error" /root/memfit_teach1_run.log | tail -3; put /root/memfit_teach1.log memfit_teach1.log; put /root/memfit_teach1_run.log memfit_teach1_run.log
[ -s /root/teach1.safetensors ] || { echo "[vn] no checkpoint"; put /root/vn.log run.log; date -u > /root/DONE; put /root/DONE DONE; exit 1; }
put /root/teach1.safetensors teach1.safetensors
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 SP_LOCAL_STORE=/root/wiki_store SP_LOCAL_MODEL=/root/bge-small OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600 --n 150"
( while sleep 600; do for i in 0 1; do [ -s /root/work/teach1_ev$i.jsonl ] && put /root/work/teach1_ev$i.jsonl teach1_ev$i.jsonl > /dev/null; done; done ) & MIR=$!
for i in 0 1; do   # two sampled passes over the 150 questions, as two processes
  ( env $ENV python3 /root/work/pool_eval.py /root/teach1.safetensors /root/work/eval300.jsonl /root/work/teach1_ev$i.jsonl $EVARGS --tag "[teach1 ev$i]" 2>&1 | grep -E "^\[(eval|load|cfg|search)|EVAL_DONE|Error|Traceback|out of memory" | sed "s/^/[ev$i] /" ) > /root/vn_ev$i.log 2>&1 &
  sleep 90
done
wait $(jobs -p | grep -v $MIR) 2>/dev/null; kill $MIR 2>/dev/null
cat /root/vn_ev0.log /root/vn_ev1.log | grep -E "EVAL_DONE|Traceback|Error" | cut -c1-300
for i in 0 1; do put /root/work/teach1_ev$i.jsonl teach1_ev$i.jsonl; done
put /root/vn.log run.log; date -u > /root/DONE; put /root/DONE DONE; echo "[vn] VNJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vnjob.sh >> /root/vn.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VNJOB_LAUNCHED $(date -u)"
fi
# CTL-END
