#!/bin/bash
# box NI (2026-10-20, a 3060 - the user: go on in this direction, keep the chat). agent5 from agent4 (box NH: chat kept,
# goals 90/83/77/68% at levels 1-4): the trace in braces {...} (it began like an operation tag, "<t...", and turned into
# "<tazz"), the goal and operations sent again every turn (the trace copied them wrong from far back), 50,000 goal
# episodes mixed with the same 32,000 chat conversations, one pass. Measured both ways; the end is charlm/agent5/DONE.
BOXNI_SERIAL=1
if [ -f /root/.boxni_serial ] && [ "$(cat /root/.boxni_serial)" -gt "$BOXNI_SERIAL" ] 2>/dev/null; then echo "BOXNI_STALE $BOXNI_SERIAL"; exit 0; fi
echo $BOXNI_SERIAL > /root/.boxni_serial
if [ ! -e /root/.ni_pre ]; then touch /root/.ni_pre
  echo "[ni] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader)"
  echo "[ni] preflight hub: $(curl -sSL -o /dev/null -m 90 -w '%{size_download} bytes in %{time_total}s' https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/charlm/agent4/model_final.pt -r 0-104857599 2>&1)"
fi
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
echo "[ni] HF token: $([ -s /root/.hf_token ] && echo present || echo MISSING)"
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
mkdir -p /root/ni
for f in charlm/goal_gen.py charlm/charlm_run.py charlm/charlm_sft.py charlm/charlm_fold.py charlm/chat_bits.py charlm/soda_prep.py q4.py; do
  for try in 1 2 3; do curl -sS -o /root/ni/$(basename $f) "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/ni/$(basename $f) && break; sleep 5; done
done
[ -f /root/.bootstrapped ] || { python3 -m pip install -q pyarrow "huggingface_hub[hf_xet]" 2>&1 | tail -1; touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"; }
cat > /root/status.sh <<'ST'
grep -E "^\[(ni|goal|sft|val|hub|gptq|chat|soda)\]|Error|Traceback" /root/ni/ni.log 2>/dev/null | tail -n 8 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh

if [ ! -e /root/.nijob ]; then touch /root/.nijob
  cat > /root/nijob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/ni
HUB="https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/charlm"
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='charlm/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
get() { [ -s $2 ] || curl -sSfL -o $2 "$HUB/$1" || { rm -f $2; return 1; }; }
get agent4/model_final.pt agent4.pt; get npc_data/stage2_public.jsonl stage2_public.jsonl
for f in train_new.jsonl t3_0.jsonl t3_1.jsonl help_0.jsonl help_1.jsonl help_2.jsonl val_all.jsonl; do get npc_data2/$f $f; done
[ -s soda_0.parquet ] || curl -sSfL -o soda_0.parquet "https://huggingface.co/api/datasets/allenai/soda/parquet/default/train/0.parquet"
[ -s soda.jsonl ] || python3 -u soda_prep.py --parquet soda_0.parquet --n 15000 --val 300 --out soda.jsonl --val-out soda_val.jsonl
python3 goal_gen.py data --n 50000 --split train --seed 21 --out goal_train.jsonl; python3 goal_gen.py data --n 500 --split train --seed 9 --out goal_val.jsonl
python3 - <<'PY'
import json, random
random.seed(4); bad = lambda r: any("[user" in m["content"] or "[name]" in m["content"] for m in r["messages"])
help_ = [l for f in ("help_0.jsonl", "help_1.jsonl", "help_2.jsonl") for l in open(f)]; random.shuffle(help_)
open("help_val.jsonl", "w").writelines(help_[:300]); help_ = help_[300:]   # the same split as box NH's
open("npc_val.jsonl", "w").writelines(list(open("val_all.jsonl"))[:300])
npc = [l for f in ("train_new.jsonl", "t3_0.jsonl", "t3_1.jsonl") for l in open(f)]; random.shuffle(npc)
pub = [l for l in open("stage2_public.jsonl") if not bad(json.loads(l))]; random.shuffle(pub)
chat = list(open("soda.jsonl")) + help_[:3000] + npc[:10000] + pub[:4000]
mix = list(open("goal_train.jsonl")) + chat; random.shuffle(mix); open("mix_a5.jsonl", "w").writelines(mix)
json.dump([json.loads(l)["messages"][0]["content"] for l in list(open("goal_val.jsonl"))[:4]], open("prompts_g.json", "w"))
print(f"[ni] mix: 50000 goal episodes + {len(chat)} chat conversations", flush=True)
PY
if [ ! -s agent5/model_final.pt ]; then
  python3 -u charlm_sft.py --base agent4.pt --data sftdata --out agent5 --shards 0 --jsonl mix_a5.jsonl --val-jsonl goal_val.jsonl --val-convs 500 --min-learned 2 \
    --prompts-json prompts_g.json --epochs 1 --lr 5e-5 --warmup 30 --eval-every 10 --hours 1.5 --hub baya1116/hypernet-sp-distill:charlm/agent5 2>&1 | grep -vE "Warning|warn|^\[step"
fi
python3 -u chat_bits.py --models agent5/model_final.pt --sets soda_val.jsonl,help_val.jsonl,npc_val.jsonl --n 200 2>&1 | grep "^\[chat\]" | tee chat.txt
for m in final:200 4bit:100; do
  echo "[ni] agent5-${m%:*} playing made-up-named episodes:" | tee -a play5.txt
  python3 -u goal_gen.py play --model agent5/model_${m%:*}.pt --n ${m#*:} --split test --seed 5 2>&1 | grep -E "^\[goal\]" | tee -a play5.txt
done
put chat.txt agent5/chat.txt; put play5.txt agent5/play5.txt; date -u > DONE; put DONE agent5/DONE; echo "[ni] NIJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/nijob.sh >> /root/ni/ni.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "NIJOB_LAUNCHED $(date -u)"
fi
echo "BOXNI_OK serial $BOXNI_SERIAL $(date -u)"
# CTL-END
