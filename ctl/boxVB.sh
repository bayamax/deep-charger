#!/bin/bash
# Box VB (a 3090): the baselines with the same 3-seed treatment as G - the 96 problems (ctl/math96.jsonl, levels 2-4), 12288
# tokens, Qwen3 thinking sampling. Full KV with vLLM (seeds 2, 3 - plus seed 1 to compare vLLM with the HF-path run), the
# window alone (rw 1056, no SP) with sp_qwen3 (seeds 2, 3). Results to archive/sp3/gen96_*.jsonl, DONE_VB at the end.
SER=3
if [ -f /root/.boxvb_serial ] && [ "$(cat /root/.boxvb_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxvb_serial
echo "[vb] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w
cat > /root/status.sh <<'ST'
grep -E "^\[(vb|hub|gen)\]|Error|Traceback" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-200
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
# serial 3: real progress of the vLLM install (venv size, processes, a real bandwidth test)
if [ ! -e /root/.chk3 ]; then touch /root/.chk3
  echo "[vb] s3 venv: $(du -sh /root/vv 2>/dev/null | cut -f1) pkgs: $(ls /root/vv/lib/python3*/site-packages 2>/dev/null | wc -l) | procs: $(ps -eo etimes,args | grep -E 'pip|vllm|snapshot' | grep -v grep | head -3 | cut -c1-90 | tr '\n' '|')"
  echo "[vb] s3 net: $(curl -s -o /dev/null -w '%{speed_download}' -L https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/model-00002-of-00002.safetensors --max-time 20 | awk '{printf "%.0f MB/s", $1/1e6}') | pip log: $(tail -c 200 /root/w/vllm_install.log 2>/dev/null | tr '\n' ' ')"
fi
# serial 2: 25 min after launch no vLLM yet - what is it doing?
if [ ! -e /root/.chk2 ]; then touch /root/.chk2
  echo "[vb] s2: $(tail -c 300 /root/w/vllm_install.log 2>/dev/null | tr '\n' ' ')"
  echo "[vb] s2 procs: $(ps -eo etimes,args | grep -E 'pip|python|vllm' | grep -v grep | head -4 | cut -c1-120 | tr '\n' '|')"
  echo "[vb] s2 disk: $(df -h /root | tail -1 | awk '{print $4}') free, net: $(curl -s -o /dev/null -w '%{speed_download}' https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/config.json) B/s"
fi
if [ ! -e /root/.job ]; then touch /root/.job
  cat > /root/live.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
while [ ! -e DONE_VB ]; do
  { grep -E "^\[(vb|hub|gen)\]|Error|Traceback" w.log | tail -n 14; tail -n 2 sp.log 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; date -u; } > vb_live.log 2>&1
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='vb_live.log', path_in_repo='archive/sp3/vb_live.log', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1
  sleep 600
done
JOB
  cat > /root/job.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
python3 -m pip install -q "huggingface_hub[hf_xet]" transformers "math-verify[antlr4_13_2]" 2>&1 | grep -v -i warn | tail -1
python3 -m venv /root/vv && /root/vv/bin/pip install -q vllm > vllm_install.log 2>&1; /root/vv/bin/python -c "import vllm; print('[vb] vllm', vllm.__version__)" 2>&1 | tail -1
for f in tools/sp_qwen3.py tools/sp_mathcheck.py tools/gen_long_vllm.py ctl/math96.jsonl; do curl -sS -o $(basename $f) "$RAW/$f?nocache=$(date +%s)"; done
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-1.7B'))" > dl.log 2>&1; echo "[vb] model: $(tail -1 dl.log | cut -c1-100)"
for sd in 2 3 1; do
  echo "[vb] full KV (vLLM) seed $sd $(date -u)"
  /root/vv/bin/python gen_long_vllm.py math96.jsonl g96_full_v$sd.jsonl 12288 96 $sd > run_full_v$sd.log 2>&1
  grep -E "Error|Traceback" run_full_v$sd.log | tail -1 | sed "s/^/[vb] /"; python3 sp_mathcheck.py g96_full_v$sd.jsonl | sed "s/^/[vb] /"; put g96_full_v$sd.jsonl gen96_full_v$sd.jsonl
done
LG=/root/w/sp.log; G="--traces math96.jsonl --n 96 --max-len 12288 --S 32 --heads 8 --layers 3 --ffn 2048 --roll-batch 48 --no-sp --rw 1056 --C 128 --maxd 0 --log $LG"
for sd in 2 3; do
  echo "[vb] window alone seed $sd $(date -u)"
  python3 sp_qwen3.py rollout $G --seed $sd --roll-out g96_raw_s$sd.jsonl > run_raw_s$sd.log 2>&1
  grep -E "Error|Traceback|out of memory" run_raw_s$sd.log | tail -1 | sed "s/^/[vb] /"; python3 sp_mathcheck.py g96_raw_s$sd.jsonl | sed "s/^/[vb] /"; put g96_raw_s$sd.jsonl gen96_raw_s$sd.jsonl
done
date -u > DONE_VB; put DONE_VB DONE_VB; echo "[vb] DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/live.sh' > /dev/null 2>&1 < /dev/null &
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[vb] job launched $(date -u)"
fi
# CTL-END
