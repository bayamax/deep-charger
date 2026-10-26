#!/bin/bash
# Box GN (a cheap 3060, generation only): SPIN negatives. G (spG_longtraces_fft1024.pt) generates, under the deployment loop
# (SP + window 1024, C 128, maxD 384, Qwen3 thinking sampling), on 512 prompts whose full-KV generations already exist (the
# finished long teacher traces stage G trained on: MATH-train 260, open-domain reasoning 100, MMLU-Pro 60, instructions 60,
# programming 32; ctl/spin2_prompts.jsonl), to 8192 tokens - the length SPIN scores. 64 prompts at a time, each part uploaded
# (archive/sp3/spin2_neg.jsonl) so a restart resumes; DONE_GN at the end.
SER=1
if [ -f /root/.boxgn_serial ] && [ "$(cat /root/.boxgn_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxgn_serial
echo "[gn] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w
cat > /root/status.sh <<'ST'
grep -E "^\[(gn|hub)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-220
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.job ]; then touch /root/.job
  cat > /root/live.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
while [ ! -e DONE_GN ]; do
  { grep -E "^\[(gn|hub)\]" w.log | tail -n 20; tail -n 4 sp.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; date -u; } > gn_live.log 2>&1
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='gn_live.log', path_in_repo='archive/sp3/gn_live.log', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1
  sleep 600
done
JOB
  cat > /root/job.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
get() { for t in 1 2 3; do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o "$2" "$H/$1" && return 0; echo "[gn] get $1 attempt $t failed"; sleep 10; done; return 1; }
python3 -m pip install -q "huggingface_hub[hf_xet]" transformers 2>&1 | grep -v -i warn | tail -1
for t in 1 2 3; do
  x=""; [ $t -gt 1 ] && x="HF_HUB_DISABLE_XET=1"
  env $x timeout 900 python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-1.7B'))" > dl.log 2>&1 && break
  echo "[gn] model download attempt $t failed"
done
echo "[gn] model: $(tail -1 dl.log | cut -c1-120)"
curl -sS -o sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; curl -sS -o spin2_prompts.jsonl "$RAW/ctl/spin2_prompts.jsonl?nocache=$(date +%s)"
[ -s spG.pt ] || get archive/sp3/spG_longtraces_fft1024.pt spG.pt; echo "[gn] spG.pt $(ls -la spG.pt | awk '{print $5}') bytes, prompts $(wc -l < spin2_prompts.jsonl)"
get archive/sp3/spin2_neg.jsonl neg.jsonl > /dev/null 2>&1 || true; touch neg.jsonl
LG=/root/w/sp.log; F="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 384"
N=$(wc -l < spin2_prompts.jsonl)
for off in $(seq 0 64 $((N - 1))); do
  [ $(wc -l < neg.jsonl) -ge $((off + 64 < N ? off + 64 : N)) ] && continue
  echo "[gn] part $off $(date -u)"
  python3 sp_qwen3.py rollout --traces spin2_prompts.jsonl --skip $off --n 64 --max-len 8192 --roll-batch 32 $F --ckpt spG.pt --roll-out part.jsonl --log $LG > run_part.log 2>&1
  grep -E "Error|Traceback|out of memory" run_part.log | tail -2 | sed "s/^/[gn] /"
  cat part.jsonl >> neg.jsonl; rm -f part.jsonl; echo "[gn] $(wc -l < neg.jsonl) negatives"
  put neg.jsonl spin2_neg.jsonl
done
date -u > DONE_GN; put DONE_GN DONE_GN; echo "[gn] DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/live.sh' > /dev/null 2>&1 < /dev/null &
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[gn] job launched $(date -u)"
fi
# CTL-END
