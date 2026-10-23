#!/bin/bash
# mk_genbox.sh I NSHARDS: ctl/boxG<I>.sh - one shard of the big trace set (tools/sp3/trace_gen2.py) on a cheap 3060
I=$1; NS=$2; N=G$I; L=g$I
cat > ctl/box$N.sh <<SCRIPT
#!/bin/bash
# Box $N (3060, vLLM): shard $I/$NS of the big single-pass training set - Qwen3-1.7B's own full-KV traces, 3200 tokens
SER=1
if [ -f /root/.box${L}_serial ] && [ "\$(cat /root/.box${L}_serial)" -gt "\$SER" ] 2>/dev/null; then echo "STALE \$SER"; exit 0; fi
echo \$SER > /root/.box${L}_serial
echo "[$L] preflight GPU: \$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:\$PATH
for f in \$(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv \$f \$f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "\$HF_TOKEN" ] && printf '%s' "\$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w; RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i"
curl -sS -o /root/w/trace_gen2.py "\$RAW/tools/sp3/trace_gen2.py?nocache=\$(date +%s)"
cat > /root/status.sh <<'ST'
grep -E "^\[($L|trace_gen2|hub)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.job ]; then touch /root/.job
  cat > /root/job.sh <<'JOB'
export HF_TOKEN=\$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/\$R/resolve/main
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='\$R')" "\$1" "\$2" > /dev/null 2>&1 && echo "[hub] \$2" || echo "[hub] \$2 FAILED"; }
python3 -m pip install -q "huggingface_hub[hf_xet]" 2>&1 | grep -v -i warn | tail -1
curl -sSfL -H "Authorization: Bearer \$HF_TOKEN" -o prompts_big.jsonl "\$H/archive/sp3/prompts_big.jsonl"
echo "[$L] prompts: \$(wc -l < prompts_big.jsonl)"
python3 trace_gen2.py --prompts prompts_big.jsonl --shard $I --nshards $NS --max-tokens 3200 --out traces_big_$I.jsonl 2>&1 | grep -E "^\[trace_gen2\]|Error|Traceback"
put traces_big_$I.jsonl traces_big_$I.jsonl
put /root/w/w.log run_$N.log; date -u > DONE_$N; put DONE_$N DONE_$N; echo "[$L] JOB_DONE \$(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "JOB_LAUNCHED \$(date -u)"
fi
# CTL-END
SCRIPT
bash -n ctl/box$N.sh && echo "ctl/box$N.sh ok"
