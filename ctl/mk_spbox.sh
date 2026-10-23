#!/bin/bash
# mk_spbox.sh NAME "ARGS": writes ctl/boxNAME.sh - one stage-1 SP-pooler run of the numbers sweep (tools/sp_qwen3.py):
# train on 600 own traces (4 per update), then held-out KL on a COMMON region (answer tokens 1792..4096) so runs with
# different windows compare; the 1.0.2-numbers run (box VY) is re-scored the same way by box SA.
N=$1; ARGS=$2; L=$(echo $N | tr A-Z a-z)
cat > ctl/box$N.sh <<SCRIPT
#!/bin/bash
# Box $N (3060): SP-pooler numbers sweep, stage 1 (model frozen): $ARGS
SER=1
if [ -f /root/.box${L}_serial ] && [ "\$(cat /root/.box${L}_serial)" -gt "\$SER" ] 2>/dev/null; then echo "STALE \$SER"; exit 0; fi
echo \$SER > /root/.box${L}_serial
echo "[$L] preflight GPU: \$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:\$PATH
for f in \$(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv \$f \$f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "\$HF_TOKEN" ] && printf '%s' "\$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w; RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i"
curl -sS -o /root/w/sp_qwen3.py "\$RAW/tools/sp_qwen3.py?nocache=\$(date +%s)"
cat > /root/status.sh <<'ST'
grep -E "^\[($L|sp|hub|math|mix)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-200
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.job ]; then touch /root/.job
  cat > /root/job.sh <<'JOB'
export HF_TOKEN=\$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/\$R/resolve/main
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='\$R')" "\$1" "\$2" > /dev/null 2>&1 && echo "[hub] \$2" || echo "[hub] \$2 FAILED"; }
python3 -m pip install -q "huggingface_hub[hf_xet]" datasets 2>&1 | grep -v -i warn | tail -1
curl -sSfL -H "Authorization: Bearer \$HF_TOKEN" -o traces.jsonl "\$H/archive/kvpool/traces.jsonl"
curl -sSfL -H "Authorization: Bearer \$HF_TOKEN" -o kv_full.jsonl "\$H/archive/kvwin/q17_full.jsonl"
LG=/root/w/sp.log; F="$ARGS"
ev() { for e in kv_full.jsonl:math traces.jsonl:mix; do f=\${e%%:*}; n=\${e#*:}; x=""; [ \$n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces \$f \$x --max-len 4096 --score-from 1792 \$F "\$@" --log \$LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback|out of memory" | sed "s/^/[\$n] /"; done; }
PRE
python3 sp_qwen3.py train --traces traces.jsonl --skip 60 --n 600 --max-len 3072 \$F --accum 4 --lr 2e-4 --out sp.pt --log \$LG > run_train.log 2>&1
grep -E "^\[sp\] (train .*: [0-9]+ traces|trained)|Error|Traceback|out of memory" run_train.log | tail -3
put sp.pt sp_$N.pt; put \$LG sp_$N.log
ev --ckpt sp.pt
put \$LG sp_$N.log; put /root/w/w.log run_$N.log; date -u > DONE_$N; put DONE_$N DONE_$N; echo "[$L] JOB_DONE \$(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "JOB_LAUNCHED \$(date -u)"
fi
# CTL-END
SCRIPT
sed -i '/^PRE$/d' ctl/box$N.sh
bash -n ctl/box$N.sh && echo "ctl/box$N.sh ok"
