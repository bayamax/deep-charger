#!/bin/bash
# Box ME (3060, pytorch image, short): mid-run score of stage B (--par + all-layer FFT) - archive/sp3/spB_mid.pt with the
# deployment loop on the held-out MATH 64 / mixed 60 (default region + from 1792), vs stage A (0.079 / 0.066 default).
SER=2
if [ -f /root/.boxme_serial ] && [ "$(cat /root/.boxme_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxme_serial
echo "[me] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w; RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i"
curl -sS -o /root/w/sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
cat > /root/status.sh <<'ST'
grep -E "^\[(me|sp|hub|math|mix)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
# serial 2: after the scores, KL by window length on the same checkpoint (is stage B's flat training KL just the
# window-0 loss of the first positions, diluted by the token count?)
if [ ! -e /root/.prof ]; then touch /root/.prof
  curl -sS -o /root/w/sp_qwen3.py.new "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
  cat > /root/prof.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
while [ ! -s DONE_ME ]; do sleep 30; done
mv sp_qwen3.py.new sp_qwen3.py
F="--S 32 --rw 1024 --C 128 --maxd 384 --layers 1 --ffn 1024"
for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x="--n 32"; [ $n = mix ] && x="--n 30"
  python3 sp_qwen3.py profile --traces $f $x --max-len 4096 $F --ckpt spB_mid.pt --log prof.log 2>&1 | grep -E "^\[sp\] profile|Error|Traceback|out of memory" | sed "s/^/[$n] /"; done
python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='prof.log', path_in_repo='archive/sp3/spB_mid_profile.log', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1 && echo "[hub] spB_mid_profile.log"
echo "[me] PROFILE_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/prof.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[me] s2: profile queued after the scores $(date -u)"
fi
if [ ! -e /root/.job ]; then touch /root/.job
  cat > /root/job.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
python3 -m pip install -q "huggingface_hub[hf_xet]" transformers datasets 2>&1 | grep -v -i warn | tail -1
for t in 1 2 3; do x=""; [ $t -gt 1 ] && x="HF_HUB_DISABLE_XET=1"
  env $x timeout 600 python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-1.7B'))" > dl.log 2>&1 && break; echo "[me] model download attempt $t failed"; done
echo "[me] model: $(tail -1 dl.log | cut -c1-80)"
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o traces.jsonl "$H/archive/kvpool/traces.jsonl"
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o kv_full_p.jsonl "$H/archive/kvwin/q17_full_p.jsonl"
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o spB_mid.pt "$H/archive/sp3/spB_mid.pt"
echo "[me] ckpt: $(du -h spB_mid.pt | cut -f1)"
LG=/root/w/sp.log; F="--S 32 --rw 1024 --C 128 --maxd 384 --layers 1 --ffn 1024"
for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  for sf in 0 1792; do python3 sp_qwen3.py eval --traces $f $x --max-len 4096 --score-from $sf $F --ckpt spB_mid.pt --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback|out of memory" | sed "s/^/[$n] /"; done; done
put $LG spB_mid_eval.log; date -u > DONE_ME; put DONE_ME DONE_ME; echo "[me] JOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "JOB_LAUNCHED $(date -u)"
fi
# CTL-END
