#!/bin/bash
# Box VE (3060, vLLM image), agent11 = agent10 with a short top-up: goal episodes (more two-step ones) and as much chat, so level 3 comes back and the chat stays; scored as agent10 was. Was: agent10 for github.com/bayamax/charlm-agent: (1) villager talks that ask facts, the teacher
# told facts must be right in any voice (--facts); (2) all villager talks through the rules (talk_rules.py) and the 4B judge
# shown the reviewer's graded lines (cut before a line graded 0); (3) agent10 from agent9 on them + agent8's replayed mix;
# (4) the same scoring as agent8/9 plus the rules on the bench replies. To the hub: archive/charlm/agent10/
BOXVE_SERIAL=1
if [ -f /root/.boxve_serial ] && [ "$(cat /root/.boxve_serial)" -gt "$BOXVE_SERIAL" ] 2>/dev/null; then echo "BOXVE_STALE $BOXVE_SERIAL"; exit 0; fi
echo $BOXVE_SERIAL > /root/.boxve_serial
if [ ! -e /root/.ve_pre ]; then touch /root/.ve_pre
  echo "[ve] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader)"
  echo "[ve] preflight hub: $(curl -sSL -o /dev/null -m 90 -w '%{size_download} bytes in %{time_total}s' https://huggingface.co/baya1116/charlm-agent/resolve/main/agent6/model_final.pt -r 0-104857599 2>&1)"
fi
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
for f in $(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv $f $f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
RAW="https://raw.githubusercontent.com/bayamax/charlm-agent/main"
mkdir -p /root/ve
for f in npc_gen.py npc_bench.py charlm_sft.py charlm_fold.py charlm_run.py chat_bits.py goal_gen.py talk_rules.py judge_audit.py q4.py eval/gold_shots.jsonl; do
  for try in 1 2 3; do curl -sS -o /root/ve/$(basename $f) "$RAW/$f?nocache=$(date +%s)" && break; sleep 5; done
done
[ -f /root/.bootstrapped ] || { python3 -m pip install -q pyarrow "huggingface_hub[hf_xet]" 2>&1 | tail -1; touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"; }
cat > /root/status.sh <<'ST'
grep -E "^\[(ve|gen|filter|rules|sft|val|gptq|chat|goal|judge|hub)\]|Error|Traceback|out of memory" /root/ve/ve.log 2>/dev/null | tail -n 6 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.vejob ]; then touch /root/.vejob
  cat > /root/vejob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/ve
H=https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/archive/charlm
J="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 3072 --gpu-mem 0.88 --max-seqs 48 --chunk 1000"
gpu_clear() { fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3; }
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/charlm/agent11/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
get() { [ -s $2 ] || curl -sSfL -o $2 "$1" || { rm -f $2; echo "[ve] could not get $1"; return 1; }; }
get $H/agent10/model_final.pt agent10.pt; get $H/agent10/villager_j.jsonl villager_j.jsonl
for f in vbench.jsonl vbench_gen.jsonl; do get $H/villager/$f $f; done
for f in mix8.jsonl mix8_select.jsonl mix8_villager_val.jsonl mix8_help_val.jsonl; do get $H/agent8/data/$f $f; done
for f in soda_val.jsonl npc_val.jsonl; do get $H/agent6/data/$f $f; done
get $H/agent10/judged10.jsonl judged10.jsonl; get $H/agent9/judged9.jsonl judged9.jsonl
[ -s goal_top.jsonl ] || python3 goal_gen.py data --n 6000 --levels 1,2,3,3,3,4,4 --split train --seed 61 --out goal_top.jsonl
python3 - <<'PY'
import json, random
key = lambda r: (r["messages"][0]["content"], r["messages"][1]["content"])
val = {key(json.loads(l)) for l in open("mix8_villager_val.jsonl")}
R = random.Random(11)
vil = [l for l in open("villager_j.jsonl") if key(json.loads(l)) not in val]; R.shuffle(vil)
rest = [l for l in open("mix8.jsonl") if not str(json.loads(l).get("src", "")).startswith("villager")]
chat = [l for l in rest if "GOAL:" not in json.loads(l)["messages"][0]["content"][:6] and not any("GOAL:" in m["content"][:6] for m in json.loads(l)["messages"][:2])]
R.shuffle(chat)
goal = list(open("goal_top.jsonl"))
mix = goal + vil[:3000] + chat[:4000]; R.shuffle(mix); open("mix11.jsonl", "w").writelines(mix)
ch = lambda ls: sum(len(m["content"]) for l in ls for m in json.loads(l)["messages"])
print(f"[ve] mix11: {len(goal)} goal episodes ({ch(goal) / 1e6:.1f}M chars), {min(3000, len(vil))} villager talks + {min(4000, len(chat))} other chats ({ch(vil[:3000] + chat[:4000]) / 1e6:.1f}M chars)", flush=True)
PY
put mix11.jsonl data/mix11.jsonl
python3 -c "import json; json.dump([json.loads(l)['messages'][0]['content'] + '\n\n' + json.loads(l)['messages'][1]['content'] for l in list(open('mix8_villager_val.jsonl'))[:2]] + [json.loads(l)['messages'][0]['content'] for l in list(open('goal_top.jsonl'))[:2]], open('prompts11.json', 'w'))"
gpu_clear
if [ ! -s agent11/model_final.pt ]; then
  python3 -u charlm_sft.py --base agent10.pt --data sftdata --out agent11 --shards 0 --jsonl mix11.jsonl --val-jsonl mix8_select.jsonl --val-convs 450 --min-learned 2 \
    --prompts-json prompts11.json --epochs 1 --lr 2e-5 --warmup 10 --eval-every 4 --hours 1 --ckpt-every 10 --hub baya1116/hypernet-sp-distill:archive/charlm/agent11 2>&1 | grep -vE "Warning|detach|backward|^\[step"
fi
python3 -u chat_bits.py --models agent10.pt,agent11/model_final.pt,agent11/model_4bit.pt --sets soda_val.jsonl,npc_val.jsonl,mix8_help_val.jsonl,mix8_villager_val.jsonl --n 300 2>&1 | grep "^\[chat\]" | tee chat11.txt
put chat11.txt chat11.txt
: > play11.txt
for m in agent11-4bit:agent11/model_4bit.pt agent11-final:agent11/model_final.pt; do IFS=: read t f <<< "$m"
  echo "[ve] $t:" | tee -a play11.txt; python3 -u goal_gen.py play --model $f --n 100 --split test --seed 5 2>&1 | grep -E "^\[goal\] level" | tee -a play11.txt
done
put play11.txt play11.txt
cat vbench.jsonl vbench_gen.jsonl > vbench_all.jsonl
python3 - <<'PY'
rows = open("vbench_all.jsonl").readlines()
for i in range(3): open(f"vb_part{i}.jsonl", "w").writelines(rows[i::3])
PY
for i in 0 1 2; do python3 -u npc_bench.py reply --model agent11/model_4bit.pt --tag agent11-4bit --bench vb_part$i.jsonl --out r11_$i.jsonl --n 4 > r11_$i.log 2>&1 & done; wait
cat r11_0.jsonl r11_1.jsonl r11_2.jsonl > r_agent11-4bit.jsonl; put r_agent11-4bit.jsonl r_agent11-4bit.jsonl
gpu_clear; python3 -u npc_gen.py judge --inp r_agent11-4bit.jsonl --out judged11.jsonl $J
cat judged9.jsonl judged10.jsonl judged11.jsonl > judged_all.jsonl
python3 npc_bench.py report --inp judged_all.jsonl > report11.txt; python3 talk_rules.py score --inp judged_all.jsonl > rules11.txt
sed 's/^/[report] /' report11.txt | grep -E "best +(ALL|G:)"; cat rules11.txt
for f in judged11.jsonl report11.txt rules11.txt; do put $f $f; done
date -u > DONE11; put DONE11 DONE; echo "[ve] VEJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vejob.sh >> /root/ve/ve.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VEJOB_LAUNCHED $(date -u)"
fi
echo "BOXVE_OK serial $BOXVE_SERIAL $(date -u)"
# CTL-END
