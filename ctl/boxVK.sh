#!/bin/bash
# Box VK (3060, vLLM image), agent13 for github.com/bayamax/charlm-agent: the bench's weak spots are facts in a strong
# voice - the dark-fantasy villager asked "do you remember my name?" answers in riddles without the name (27%), the gift
# it dislikes gets a warm thanks (56%). The training talks hold only ~40 dark name recalls. So: (1) the teacher writes
# fact-asking talks again (--facts: name, hours, gifts, what is sold must be right in any voice), dark and friends heavy;
# (2) the rules and the 4B judge filter them; (3) agent13 = agent12a on them (twice) + the replay; (4) the same scoring
# as agent12 beside agent12a's rows and agent12c's (the same weights scored again: the bench's noise). To archive/charlm/agent13/
BOXVK_SERIAL=1
if [ -f /root/.boxvk_serial ] && [ "$(cat /root/.boxvk_serial)" -gt "$BOXVK_SERIAL" ] 2>/dev/null; then echo "BOXVK_STALE $BOXVK_SERIAL"; exit 0; fi
echo $BOXVK_SERIAL > /root/.boxvk_serial
if [ ! -e /root/.vk_pre ]; then touch /root/.vk_pre
  echo "[vk] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader)"
  echo "[vk] preflight hub: $(curl -sSL -o /dev/null -m 90 -w '%{size_download} bytes in %{time_total}s' https://huggingface.co/baya1116/charlm-agent/resolve/main/agent6/model_final.pt -r 0-104857599 2>&1)"
fi
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
for f in $(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv $f $f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
RAW="https://raw.githubusercontent.com/bayamax/charlm-agent/main"
mkdir -p /root/vk
for f in npc_gen.py npc_bench.py charlm_sft.py charlm_fold.py charlm_run.py chat_bits.py goal_gen.py talk_rules.py q4.py eval/gold_shots.jsonl; do
  for try in 1 2 3; do curl -sS -o /root/vk/$(basename $f) "$RAW/$f?nocache=$(date +%s)" && break; sleep 5; done
done
[ -f /root/.bootstrapped ] || { python3 -m pip install -q pyarrow "huggingface_hub[hf_xet]" 2>&1 | tail -1; touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"; }
cat > /root/status.sh <<'ST'
grep -E "^\[(vk|gen|filter|rules|sft|val|gptq|chat|goal|judge|report|hub)\]|Error|Traceback|out of memory" /root/vk/vk.log 2>/dev/null | tail -n 6 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.vkjob ]; then touch /root/.vkjob
  cat > /root/vkjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vk
H=https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/archive/charlm
G="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 4096 --gpu-mem 0.88 --max-seqs 48"
JV="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 8192 --gpu-mem 0.88 --max-seqs 32 --chunk 1000"
J="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 3072 --gpu-mem 0.88 --max-seqs 48 --chunk 1000"
gpu_clear() { fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3; }
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/charlm/agent13/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
get() { [ -s $2 ] || curl -sSfL -o $2 "$1" || { rm -f $2; echo "[vk] could not get $1"; return 1; }; }
get $H/agent12/a/model_final.pt agent12a.pt; get $H/agent10/villager_j.jsonl villager_j.jsonl
for f in vbench.jsonl vbench_gen.jsonl; do get $H/villager/$f $f; done
for f in mix8.jsonl mix8_select.jsonl mix8_villager_val.jsonl; do get $H/agent8/data/$f $f; done
for f in soda_val.jsonl npc_val.jsonl; do get $H/agent6/data/$f $f; done
get $H/agent12/data/help_val.jsonl help_val.jsonl; get $H/agent12/judged12.jsonl judged12.jsonl; get $H/agent12/c/judged12c.jsonl judged12c.jsonl
# (1) fact-asking talks, dark and friends heavy (the teacher's rules: the name, the hours, the gift reaction must be right in any voice)
if [ ! -s villager_fact2.done ]; then
  ( while sleep 240; do put villager_fact2.jsonl villager_fact2.jsonl > /dev/null; done ) & MIR=$!
  gpu_clear; timeout 50m python3 -u npc_gen.py villager --facts 1 --genres dark,friends,cozy,dark,friends --n 3600 --seed 41 --out villager_fact2.jsonl --chunk 300 $G
  kill $MIR; put villager_fact2.jsonl villager_fact2.jsonl; date > villager_fact2.done
fi
# (2) the rules, then the judge shown the reviewer's graded lines
python3 talk_rules.py filter --inp villager_fact2.jsonl --out villager_fact2_r.jsonl
gpu_clear; [ -s villager_fact2_j.jsonl ] || python3 -u npc_gen.py vfilter --shots gold_shots.jsonl --min-grade 1 --inp villager_fact2_r.jsonl --out villager_fact2_j.jsonl $JV
put villager_fact2_j.jsonl villager_fact2_j.jsonl
[ -s goal_r5.jsonl ] || python3 goal_gen.py data --n 4000 --levels 1,2,3,3,3,4,4 --split train --seed 73 --out goal_r5.jsonl
python3 - <<'PY'
import json, random, re
key = lambda r: (r["messages"][0]["content"], r["messages"][1]["content"])
val = {key(json.loads(l)) for l in open("mix8_villager_val.jsonl")}
R = random.Random(13)
card = lambda l: json.loads(l)["messages"][0]["role"] == "system"
new = [l for l in open("villager_fact2_j.jsonl") if card(l) and key(json.loads(l)) not in val]
vil = [l for l in open("villager_j.jsonl") if card(l) and key(json.loads(l)) not in val]; R.shuffle(vil)
# the old talks that hold a name recall or a disliked gift, kept beside the new ones
def hard(l):
    m = json.loads(l)["messages"]; t = " ".join(x["content"] for x in m if x["role"] == "user")
    return bool(re.search(r"remember .*name|what'?s my name|who am I", t, re.I)) or bool(re.search(r"\bDislikes?:", m[0]["content"]) and re.search(r"gift|for you|take this|brought you", t, re.I))
old_hard = [l for l in vil if hard(l)]
rest = [l for l in open("mix8.jsonl") if not str(json.loads(l).get("src", "")).startswith("villager")]
goal = [l for l in rest if json.loads(l)["messages"][0]["content"].startswith("GOAL:")]; gs = set(goal); chat = [l for l in rest if l not in gs]; R.shuffle(chat)
mix = new * 2 + old_hard + list(open("goal_r5.jsonl"))[:3000] + chat[:3000] + vil[:1500]; R.shuffle(mix)
open("mix13.jsonl", "w").writelines(mix)
print(f"[vk] mix13: {len(new)} new fact talks (x2) + {len(old_hard)} old hard talks + 3000 goal + 3000 chat + 1500 villager = {len(mix)}", flush=True)
PY
python3 -c "import json; json.dump([json.loads(l)['messages'][0]['content'] + '\n\n' + json.loads(l)['messages'][1]['content'] for l in list(open('mix8_villager_val.jsonl'))[:3]], open('prompts13.json', 'w'))"
# (3) agent13 from agent12a
gpu_clear; mkdir -p agent13
( while sleep 300; do put agent13/train.log train.log > /dev/null; done ) & MIR=$!
[ -s agent13/model_4bit.pt ] || python3 -u charlm_sft.py --base agent12a.pt --data sftdata --out agent13 --shards 0 --jsonl mix13.jsonl --val-jsonl mix8_select.jsonl --val-convs 450 --min-learned 2 \
    --prompts-json prompts13.json --epochs 1 --lr 2e-5 --warmup 10 --eval-every 4 --hours 1 --ckpt-every 10 --hub baya1116/hypernet-sp-distill:archive/charlm/agent13 2>&1 | grep -vE "Warning|detach|backward|^\[step"
kill $MIR 2>/dev/null; put agent13/train.log train.log
[ -s agent13/model_4bit.pt ] || { echo "[vk] no 4-bit model"; date -u > DONE; put DONE DONE; exit 1; }
# (4) scoring
python3 -u chat_bits.py --models agent12a.pt,agent13/model_final.pt,agent13/model_4bit.pt --sets soda_val.jsonl,npc_val.jsonl,help_val.jsonl,mix8_villager_val.jsonl --n 300 2>&1 | grep "^\[chat\]" | tee chat13.txt
put chat13.txt chat13.txt
python3 -u goal_gen.py play --model agent13/model_4bit.pt --n 100 --split test --seed 5 2>&1 | grep -E "^\[goal\] level" | tee play13.txt; put play13.txt play13.txt
cat vbench.jsonl vbench_gen.jsonl > vbench_all.jsonl
python3 - <<'PY'
rows = open("vbench_all.jsonl").readlines()
for i in range(3): open(f"vb_part{i}.jsonl", "w").writelines(rows[i::3])
PY
for i in 0 1 2; do python3 -u npc_bench.py reply --model agent13/model_4bit.pt --tag agent13-4bit --bench vb_part$i.jsonl --out r13_$i.jsonl --n 4 > r13_$i.log 2>&1 & done; wait
cat r13_0.jsonl r13_1.jsonl r13_2.jsonl > r_agent13-4bit.jsonl; put r_agent13-4bit.jsonl r_agent13-4bit.jsonl
gpu_clear; python3 -u npc_gen.py judge --inp r_agent13-4bit.jsonl --out judged13.jsonl $J
cat judged12.jsonl judged12c.jsonl judged13.jsonl > judged_all13.jsonl
python3 npc_bench.py report --inp judged_all13.jsonl > report13.txt; python3 talk_rules.py score --inp judged_all13.jsonl > rules13.txt
sed 's/^/[report] /' report13.txt | grep -E "best +(ALL|G:|dark:recall|gift_dislike|memory |town )"; cat rules13.txt
for f in judged13.jsonl report13.txt rules13.txt; do put $f $f; done
put /root/vk/vk.log run.log; date -u > DONE; put DONE DONE; echo "[vk] VKJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vkjob.sh >> /root/vk/vk.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VKJOB_LAUNCHED $(date -u)"
fi
# CTL-END
