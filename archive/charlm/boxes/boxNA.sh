#!/bin/bash
# box NA (2026-10-20, a 3060 - the user: how far does the character model go as a tiny agent? The game tells it who it is,
# its actions, the state and an event; it answers <say>...</say><act>...</act>). charlm/agent_gen.py makes the turns from
# rules; npc4 is fine-tuned on 20,000 conversations (agent1), then answers held-out ones with unseen characters, items
# and wordings - once with the training action names, once with new ones it must read from the card's list. The end is
# charlm/agent1/DONE on the hub, after which the box is let go.
BOXNA_SERIAL=1
if [ -f /root/.boxna_serial ] && [ "$(cat /root/.boxna_serial)" -gt "$BOXNA_SERIAL" ] 2>/dev/null; then echo "BOXNA_STALE $BOXNA_SERIAL"; exit 0; fi
echo $BOXNA_SERIAL > /root/.boxna_serial
if [ ! -e /root/.na_pre ]; then touch /root/.na_pre
  echo "[na] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader)"
  echo "[na] preflight hub: $(curl -sSL -o /dev/null -m 90 -w '%{size_download} bytes in %{time_total}s' https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/charlm/npc4/model_final.pt -r 0-104857599 2>&1)"
fi
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
for f in $(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv $f $f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
echo "[na] HF token: $([ -s /root/.hf_token ] && echo present || echo MISSING)"
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
mkdir -p /root/na
for f in charlm/agent_gen.py charlm/npc_bench.py charlm/charlm_sft.py charlm/charlm_fold.py q4.py; do
  for try in 1 2 3; do curl -sS -o /root/na/$(basename $f) "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/na/$(basename $f) && break; sleep 5; done
done
[ -f /root/.bootstrapped ] || { python3 -m pip install -q pyarrow "huggingface_hub[hf_xet]" 2>&1 | tail -1; touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"; }
cat > /root/status.sh <<'ST'
grep -E "^\[(na|agent|bench|sft|val|hub|gptq)\]|Error|Traceback" /root/na/na.log 2>/dev/null | tail -n 8 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh

if [ ! -e /root/.najob ]; then touch /root/.najob
  cat > /root/najob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/na
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='charlm/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
[ -s npc4.pt ] || curl -sSfL -o npc4.pt "https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/charlm/npc4/model_final.pt"
python3 agent_gen.py --n 20000 --split train --out agent_train.jsonl; python3 agent_gen.py --n 200 --split train --seed 7 --out agent_val.jsonl
python3 agent_gen.py --n 300 --split test --names train --out test_seen.jsonl; python3 agent_gen.py --n 300 --split test --out test_new.jsonl
python3 -c "
import json; r = [json.loads(l) for l in open('test_seen.jsonl')][:4]
json.dump([x['messages'][0]['content'] for x in r], open('prompts_a.json', 'w'))"
if [ ! -s agent1/model_final.pt ]; then
  python3 -u charlm_sft.py --base npc4.pt --data sftdata --out agent1 --shards 0 --jsonl agent_train.jsonl --val-jsonl agent_val.jsonl --val-convs 200 \
    --prompts-json prompts_a.json --epochs 1 --lr 1e-4 --warmup 30 --eval-every 4 --hours 1 --hub baya1116/hypernet-sp-distill:charlm/agent1 2>&1 | grep -vE "Warning|warn|^\[step"
fi
for m in final 4bit; do for t in seen new; do
  [ -s r_${m}_$t.jsonl ] || python3 -u npc_bench.py turns --model agent1/model_$m.pt --tag agent1-$m --bench test_$t.jsonl --out r_${m}_$t.jsonl --temp 0.05 --max-chars 200
  echo "[na] score agent1-$m, test with $t action names:" | tee -a score.txt; python3 agent_gen.py --score --inp r_${m}_$t.jsonl | tee -a score.txt
done; done
for f in score.txt r_final_seen.jsonl r_final_new.jsonl; do put $f agent1/$f; done
date -u > DONE; put DONE agent1/DONE; echo "[na] NAJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/najob.sh >> /root/na/na.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "NAJOB_LAUNCHED $(date -u)"
fi
echo "BOXNA_OK serial $BOXNA_SERIAL $(date -u)"
# CTL-END
