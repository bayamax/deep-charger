#!/bin/bash
# box NG (2026-10-20, a 3060 - the user: not many concepts but one pattern for any use: a goal on numbers, the state, and
# operations that change it; the model picks the operation that moves the state to the goal, the game applies it and
# tells it the new state). charlm/goal_gen.py makes 80,000 such episodes from rules (levels 1-4); npc4 is fine-tuned on
# them (agent2); then it plays held-out episodes in a closed loop - with the training variable/operation names and with
# made-up ones - and the share that reaches the goal is reported by level. The end is charlm/agent2/DONE on the hub.
BOXNG_SERIAL=1
if [ -f /root/.boxng_serial ] && [ "$(cat /root/.boxng_serial)" -gt "$BOXNG_SERIAL" ] 2>/dev/null; then echo "BOXNG_STALE $BOXNG_SERIAL"; exit 0; fi
echo $BOXNG_SERIAL > /root/.boxng_serial
if [ ! -e /root/.ng_pre ]; then touch /root/.ng_pre
  echo "[ng] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader)"
  echo "[ng] preflight hub: $(curl -sSL -o /dev/null -m 90 -w '%{size_download} bytes in %{time_total}s' https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/charlm/npc4/model_final.pt -r 0-104857599 2>&1)"
fi
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
echo "[ng] HF token: $([ -s /root/.hf_token ] && echo present || echo MISSING)"
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
mkdir -p /root/ng
for f in charlm/goal_gen.py charlm/charlm_run.py charlm/charlm_sft.py charlm/charlm_fold.py q4.py; do
  for try in 1 2 3; do curl -sS -o /root/ng/$(basename $f) "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/ng/$(basename $f) && break; sleep 5; done
done
[ -f /root/.bootstrapped ] || { python3 -m pip install -q pyarrow "huggingface_hub[hf_xet]" 2>&1 | tail -1; touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"; }
cat > /root/status.sh <<'ST'
grep -E "^\[(ng|goal|sft|val|hub|gptq)\]|Error|Traceback" /root/ng/ng.log 2>/dev/null | tail -n 8 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh

if [ ! -e /root/.ngjob ]; then touch /root/.ngjob
  cat > /root/ngjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/ng
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='charlm/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
[ -s npc4.pt ] || curl -sSfL -o npc4.pt "https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/charlm/npc4/model_final.pt"
python3 goal_gen.py data --n 80000 --split train --out goal_train.jsonl; python3 goal_gen.py data --n 500 --split train --seed 9 --out goal_val.jsonl
python3 -c "
import json; r = [json.loads(l) for l in open('goal_val.jsonl')][:4]
json.dump([x['messages'][0]['content'] for x in r], open('prompts_g.json', 'w'))"
if [ ! -s agent2/model_final.pt ]; then
  python3 -u charlm_sft.py --base npc4.pt --data sftdata --out agent2 --shards 0 --jsonl goal_train.jsonl --val-jsonl goal_val.jsonl --val-convs 500 \
    --prompts-json prompts_g.json --epochs 1 --lr 1e-4 --warmup 30 --eval-every 5 --hours 1 --hub baya1116/hypernet-sp-distill:charlm/agent2 2>&1 | grep -vE "Warning|warn|^\[step"
fi
for m in final 4bit; do for sp in test train; do
  echo "[ng] agent2-$m playing $sp-named episodes:" | tee -a play.txt
  python3 -u goal_gen.py play --model agent2/model_$m.pt --n 250 --split $sp --seed 5 2>&1 | grep -E "^\[goal\]" | tee -a play.txt
done; done
put play.txt agent2/play.txt; date -u > DONE; put DONE agent2/DONE; echo "[ng] NGJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/ngjob.sh >> /root/ng/ng.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "NGJOB_LAUNCHED $(date -u)"
fi
echo "BOXNG_OK serial $BOXNG_SERIAL $(date -u)"
# CTL-END
