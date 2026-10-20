#!/bin/bash
# box NH (2026-10-20, a 3060 - the user: go on in this direction, make it more complete). agent3: the answer first writes
# the rule's steps (<think>goal variable, value, which way; the operations that move it; needs; side effects</think><pick>),
# so the model compares neighbouring text instead of far-apart lines; 60,000 traced episodes (charlm/goal_gen.py) x 3
# passes from agent2 (box NG, picks only: reached 62/41/65/25% at levels 1-4 on made-up names). Then closed-loop play on
# made-up names. The end is charlm/agent3/DONE on the hub, after which the box is let go.
BOXNH_SERIAL=2
if [ -f /root/.boxnh_serial ] && [ "$(cat /root/.boxnh_serial)" -gt "$BOXNH_SERIAL" ] 2>/dev/null; then echo "BOXNH_STALE $BOXNH_SERIAL"; exit 0; fi
echo $BOXNH_SERIAL > /root/.boxnh_serial
if [ ! -e /root/.nh_pre ]; then touch /root/.nh_pre
  echo "[nh] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader)"
  echo "[nh] preflight hub: $(curl -sSL -o /dev/null -m 90 -w '%{size_download} bytes in %{time_total}s' https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/charlm/agent2/model_final.pt -r 0-104857599 2>&1)"
fi
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
echo "[nh] HF token: $([ -s /root/.hf_token ] && echo present || echo MISSING)"
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
mkdir -p /root/nh
for f in charlm/goal_gen.py charlm/charlm_run.py charlm/charlm_sft.py charlm/charlm_fold.py charlm/chat_bits.py charlm/soda_prep.py q4.py; do
  for try in 1 2 3; do curl -sS -o /root/nh/$(basename $f) "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/nh/$(basename $f) && break; sleep 5; done
done
[ -f /root/.bootstrapped ] || { python3 -m pip install -q pyarrow "huggingface_hub[hf_xet]" 2>&1 | tail -1; touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"; }
cat > /root/status.sh <<'ST'
grep -E "^\[(nh|nh4|goal|sft|val|hub|gptq|chat|soda)\]|Error|Traceback" /root/nh/nh.log 2>/dev/null | tail -n 8 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh

if [ ! -e /root/.nhjob ]; then touch /root/.nhjob
  cat > /root/nhjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/nh
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='charlm/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
[ -s agent2.pt ] || curl -sSfL -o agent2.pt "https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/charlm/agent2/model_final.pt"
python3 goal_gen.py data --n 60000 --split train --out goal_train.jsonl; python3 goal_gen.py data --n 500 --split train --seed 9 --out goal_val.jsonl
python3 -c "
import json; r = [json.loads(l) for l in open('goal_val.jsonl')][:4]
json.dump([x['messages'][0]['content'] for x in r], open('prompts_g.json', 'w'))"
if [ ! -s agent3/model_final.pt ]; then
  python3 -u charlm_sft.py --base agent2.pt --data sftdata --out agent3 --shards 0 --jsonl goal_train.jsonl --val-jsonl goal_val.jsonl --val-convs 500 --min-learned 2 \
    --prompts-json prompts_g.json --epochs 3 --lr 1e-4 --warmup 30 --eval-every 10 --hours 1.5 --hub baya1116/hypernet-sp-distill:charlm/agent3 2>&1 | grep -vE "Warning|warn|^\[step"
fi
for m in final:200 4bit:100; do
  echo "[nh] agent3-${m%:*} playing made-up-named episodes:" | tee -a play.txt
  python3 -u goal_gen.py play --model agent3/model_${m%:*}.pt --n ${m#*:} --split test --seed 5 2>&1 | grep -E "^\[goal\]" | tee -a play.txt
done
put play.txt agent3/play.txt; date -u > DONE; put DONE agent3/DONE; echo "[nh] NHJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/nhjob.sh >> /root/nh/nh.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "NHJOB_LAUNCHED $(date -u)"
fi
# ---- serial 2, the user: keep the chat ability too, by mixing (or alternating) the two kinds. After agent3: how much chat
# it lost (bits per assistant character on held-out SODA, help and NPC conversations, beside npc4); then agent4 from agent3
# on 30,000 goal episodes mixed with chat (SODA 15,000, help 3,000, NPC 10,000, public 4,000), one pass; measured both
# ways. The end is charlm/agent4/DONE ----
if [ ! -e /root/.nhjob4 ]; then touch /root/.nhjob4
  cat > /root/nhjob4.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/nh
HUB="https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/charlm"
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='charlm/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
get() { [ -s $2 ] || curl -sSfL -o $2 "$HUB/$1" || { rm -f $2; return 1; }; }
until grep -q NHJOB_DONE nh.log 2>/dev/null; do sleep 60; done
get npc4/model_final.pt npc4.pt; get npc_data/stage2_public.jsonl stage2_public.jsonl
for f in train_new.jsonl t3_0.jsonl t3_1.jsonl help_0.jsonl help_1.jsonl help_2.jsonl val_all.jsonl; do get npc_data2/$f $f; done
[ -s soda_0.parquet ] || curl -sSfL -o soda_0.parquet "https://huggingface.co/api/datasets/allenai/soda/parquet/default/train/0.parquet"
[ -s soda.jsonl ] || python3 -u soda_prep.py --parquet soda_0.parquet --n 15000 --val 300 --out soda.jsonl --val-out soda_val.jsonl
python3 - <<'PY'
import json, random
random.seed(4); bad = lambda r: any("[user" in m["content"] or "[name]" in m["content"] for m in r["messages"])
help_ = [l for f in ("help_0.jsonl", "help_1.jsonl", "help_2.jsonl") for l in open(f)]; random.shuffle(help_)
open("help_val.jsonl", "w").writelines(help_[:300]); help_ = help_[300:]
open("npc_val.jsonl", "w").writelines(list(open("val_all.jsonl"))[:300])
npc = [l for f in ("train_new.jsonl", "t3_0.jsonl", "t3_1.jsonl") for l in open(f)]; random.shuffle(npc)
pub = [l for l in open("stage2_public.jsonl") if not bad(json.loads(l))]; random.shuffle(pub)
goal = list(open("goal_train.jsonl")); random.shuffle(goal)
chat = list(open("soda.jsonl")) + help_[:3000] + npc[:10000] + pub[:4000]
mix = goal[:30000] + chat; random.shuffle(mix); open("mix_a4.jsonl", "w").writelines(mix)
print(f"[nh4] mix: {min(30000, len(goal))} goal episodes + {len(chat)} chat conversations", flush=True)
PY
echo "[nh4] chat before:" | tee chat.txt
python3 -u chat_bits.py --models npc4.pt,agent3/model_final.pt --sets soda_val.jsonl,help_val.jsonl,npc_val.jsonl --n 200 2>&1 | grep "^\[chat\]" | tee -a chat.txt
if [ ! -s agent4/model_final.pt ]; then
  python3 -u charlm_sft.py --base agent3/model_final.pt --data sftdata --out agent4 --shards 0 --jsonl mix_a4.jsonl --val-jsonl goal_val.jsonl --val-convs 500 --min-learned 2 \
    --prompts-json prompts_g.json --epochs 1 --lr 5e-5 --warmup 30 --eval-every 10 --hours 1.5 --hub baya1116/hypernet-sp-distill:charlm/agent4 2>&1 | grep -vE "Warning|warn|^\[step"
fi
echo "[nh4] chat after:" | tee -a chat.txt
python3 -u chat_bits.py --models agent4/model_final.pt --sets soda_val.jsonl,help_val.jsonl,npc_val.jsonl --n 200 2>&1 | grep "^\[chat\]" | tee -a chat.txt
for m in final:200 4bit:100; do
  echo "[nh4] agent4-${m%:*} playing made-up-named episodes:" | tee -a play4.txt
  python3 -u goal_gen.py play --model agent4/model_${m%:*}.pt --n ${m#*:} --split test --seed 5 2>&1 | grep -E "^\[goal\]" | tee -a play4.txt
done
put chat.txt agent4/chat.txt; put play4.txt agent4/play4.txt; date -u > DONE4; put DONE4 agent4/DONE; echo "[nh4] NHJOB4_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/nhjob4.sh >> /root/nh/nh.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "NHJOB4_QUEUED $(date -u)"
fi
echo "BOXNH_OK serial $BOXNH_SERIAL $(date -u)"
# CTL-END
