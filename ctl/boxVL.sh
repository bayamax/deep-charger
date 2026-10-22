#!/bin/bash
# Box VL (3060, pytorch image): the NPC lesson applied to deep-charger - N sampled rollouts per question, the answer by
# vote. 2.1.0 (the adopted model: 2.0.0's 4-bit base + the mtg7 LoRA) on the search held-out (eval300, 150 questions)
# with the local store, 8 passes of pool_eval.py (each pass one rollout per question, temperature 0.6, the lineage's
# settings) in two processes; the passes go to the hub (archive/vote/), the vote is counted offline.
BOXVL_SERIAL=1
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
for f in pool_eval.py localsearch/search.py localsearch/store.py localsearch/ivf.py localsearch/terms.py localsearch/pq.py localsearch/embed.py; do
  for try in 1 2 3; do curl -sS -o /root/work/$f "$RAW/$f?nocache=$(date +%s)" && break; sleep 5; done
done
[ -f /root/.bootstrapped ] || { python3 -m pip install -q "transformers>=4.51,<4.57" accelerate "huggingface_hub[hf_xet]>=0.34,<1.0" safetensors peft zstandard onnxruntime 2>&1 | tail -1; touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"; }
cat > /root/status.sh <<'ST'
grep -E "^\[(vl|eval|load|cfg|search|hub)|EVAL_DONE|Error|Traceback|out of memory" /root/vl.log 2>/dev/null | tail -n 6 | cut -c1-300
for i in 0 1; do f=/root/work/vote_p$i.jsonl; [ -s $f ] && echo "pass file $i: $(wc -l < $f) rows"; done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.vljob ]; then touch /root/.vljob
  cat > /root/vljob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/work
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/vote/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
get() { [ -s $2 ] || curl -sSfL -o $2 "$1" || { rm -f $2; echo "[vl] could not get $1"; return 1; }; }
for f in grpo_e2e_torch.py pooler_torch.py; do get $H/box_recover/scripts/$f /root/work/$f; done
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
date -u > /root/DONE; put /root/DONE DONE; echo "[vl] VLJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vljob.sh >> /root/vl.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VLJOB_LAUNCHED $(date -u)"
fi
# CTL-END
