#!/bin/bash
# Box VC (3060, vLLM image): does Qwen3.5-9B (AWQ 4-bit, text only) fit a 12 GB card as the judge, and how well does it
# grade the reviewer's gold lines (github.com/bayamax/charlm-agent eval/) next to the 4B? Results to the hub (archive/charlm/judge/)
BOXVC_SERIAL=1
if [ -f /root/.boxvc_serial ] && [ "$(cat /root/.boxvc_serial)" -gt "$BOXVC_SERIAL" ] 2>/dev/null; then echo "BOXVC_STALE $BOXVC_SERIAL"; exit 0; fi
echo $BOXVC_SERIAL > /root/.boxvc_serial
if [ ! -e /root/.vc_pre ]; then touch /root/.vc_pre
  echo "[vc] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader)"
fi
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
for f in $(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv $f $f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
RAW="https://raw.githubusercontent.com/bayamax/charlm-agent/main"
mkdir -p /root/vc
for f in npc_gen.py judge_audit.py eval/gold_villager_lines.jsonl eval/gold_shots.jsonl; do curl -sS -o /root/vc/$(basename $f) "$RAW/$f?nocache=$(date +%s)"; done
[ -f /root/.bootstrapped ] || { python3 -m pip install -q "huggingface_hub[hf_xet]" 2>&1 | tail -1; touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"; }
cat > /root/status.sh <<'ST'
grep -E "^\[(vc|linejudge|gen|audit|hub)\]|Error|Traceback|out of memory" /root/vc/vc.log 2>/dev/null | tail -n 6 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.vcjob ]; then touch /root/.vcjob
  cat > /root/vcjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vc
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/charlm/judge/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
python3 -c "import json; open('gold_lab.jsonl','w').writelines(json.dumps({'id': json.loads(l)['id'], 'grade': json.loads(l)['grade']}) + '\n' for l in open('gold_villager_lines.jsonl'))"
: > judge_eval_9b.txt
for v in fewshot plain; do
  for cfg in "0.93 8 1" "0.95 4 1"; do read gm ms ea <<< "$cfg"
    fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3
    python3 -u npc_gen.py linejudge --variant $v --shots gold_shots.jsonl --inp gold_villager_lines.jsonl --out lj9_$v.jsonl \
      --model QuantTrio/Qwen3.5-9B-AWQ --fallback QuantTrio/Qwen3.5-9B-AWQ --max-len 6144 --gpu-mem $gm --max-seqs $ms --eager $ea 2>&1 | grep -vE "^INFO|^\(EngineCore" | tail -25
    [ -s lj9_$v.jsonl ] && { echo "[vc] 9B $v loaded with gpu-mem $gm, max-seqs $ms"; break; }
    echo "[vc] 9B $v did not run with gpu-mem $gm, max-seqs $ms"
  done
  [ -s lj9_$v.jsonl ] && { echo "== 9B $v (the 132 lines not used as examples)" | tee -a judge_eval_9b.txt; python3 judge_audit.py score --key lj9_$v.jsonl --labels gold_lab.jsonl | tee -a judge_eval_9b.txt; }
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
put judge_eval_9b.txt judge_eval_9b.txt; date -u > DONE9B; put DONE9B DONE9B; echo "[vc] VCJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vcjob.sh >> /root/vc/vc.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VCJOB_LAUNCHED $(date -u)"
fi
echo "BOXVC_OK serial $BOXVC_SERIAL $(date -u)"
# CTL-END
