#!/bin/bash
# Box VP (3060, vLLM image), agent15 for github.com/bayamax/charlm-agent: agent14 (the judge-picked small talk) lost the
# reaction to a gift the villager dislikes (69% -> 49%; it thanks you for an old boot). No new teacher text: the gift
# exchanges already written for agent13b/13c (fact talks, rules- and judge-filtered) are pulled out and learned three
# times, the rest of the newest fact talks once, agent14's own judge-picked chat beside them so the charm stays, and the
# replay. agent15 from agent14; scored as agent14 was (beside agent13b's and agent14's rows). To archive/charlm/agent15/
BOXVP_SERIAL=1
if [ -f /root/.boxvp_serial ] && [ "$(cat /root/.boxvp_serial)" -gt "$BOXVP_SERIAL" ] 2>/dev/null; then echo "BOXVP_STALE $BOXVP_SERIAL"; exit 0; fi
echo $BOXVP_SERIAL > /root/.boxvp_serial
if [ ! -e /root/.vp_pre ]; then touch /root/.vp_pre
  echo "[vp] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader)"
  echo "[vp] preflight hub: $(curl -sSL -o /dev/null -m 90 -w '%{size_download} bytes in %{time_total}s' https://huggingface.co/baya1116/charlm-agent/resolve/main/agent6/model_final.pt -r 0-104857599 2>&1)"
fi
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
for f in $(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv $f $f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
RAW="https://raw.githubusercontent.com/bayamax/charlm-agent/main"
mkdir -p /root/vp
for f in npc_gen.py npc_bench.py charlm_sft.py charlm_fold.py charlm_run.py chat_bits.py goal_gen.py talk_rules.py q4.py eval/gold_shots.jsonl; do
  for try in 1 2 3; do curl -sS -o /root/vp/$(basename $f) "$RAW/$f?nocache=$(date +%s)" && break; sleep 5; done
done
[ -f /root/.bootstrapped ] || { python3 -m pip install -q pyarrow "huggingface_hub[hf_xet]" 2>&1 | tail -1; touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"; }
cat > /root/status.sh <<'ST'
grep -E "^\[(vp|gen|filter|rules|sft|val|gptq|chat|goal|judge|report|hub)\]|Error|Traceback|out of memory" /root/vp/vk.log 2>/dev/null | tail -n 6 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.vpjob ]; then touch /root/.vpjob
  cat > /root/vpjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vp
H=https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/archive/charlm
J="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 3072 --gpu-mem 0.88 --max-seqs 48 --chunk 1000"
gpu_clear() { fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3; }
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/charlm/agent15/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
get() { [ -s $2 ] || curl -sSfL -o $2 "$1" || { rm -f $2; echo "[vp] could not get $1"; return 1; }; }
get https://huggingface.co/baya1116/charlm-agent/resolve/main/agent14/model_final.pt agent14.pt
get $H/agent13/b/villager_fact3_j.jsonl villager_fact3_j.jsonl; get $H/agent13/c/villager_fact4_j.jsonl villager_fact4_j.jsonl
get $H/agent14/rft_chat.jsonl rft_chat.jsonl; get $H/agent10/villager_j.jsonl villager_j.jsonl
for f in vbench.jsonl vbench_gen.jsonl; do get $H/villager/$f $f; done
for f in mix8.jsonl mix8_select.jsonl mix8_villager_val.jsonl; do get $H/agent8/data/$f $f; done
for f in soda_val.jsonl npc_val.jsonl; do get $H/agent6/data/$f $f; done
get $H/agent12/data/help_val.jsonl help_val.jsonl; get $H/agent13/b/judged13b.jsonl judged13b.jsonl; get $H/agent14/judged14.jsonl judged14.jsonl
[ -s goal_r7.jsonl ] || python3 goal_gen.py data --n 4000 --levels 1,2,3,3,3,4,4 --split train --seed 75 --out goal_r7.jsonl
python3 -c "import json; json.dump([json.loads(l)['messages'][0]['content'] + '\n\n' + json.loads(l)['messages'][1]['content'] for l in list(open('mix8_villager_val.jsonl'))[:3]], open('prompts15.json', 'w'))"
cat vbench.jsonl vbench_gen.jsonl > vbench_all.jsonl
python3 - <<'PY'
rows = open("vbench_all.jsonl").readlines()
for i in range(3): open(f"vb_part{i}.jsonl", "w").writelines(rows[i::3])
PY
python3 - <<'PY'
import json, random, re, sys
sys.path.insert(0, "."); import talk_rules as TR
key = lambda r: (r["messages"][0]["content"], r["messages"][1]["content"])
val = {key(json.loads(l)) for l in open("mix8_villager_val.jsonl")}
R = random.Random(18); card = lambda l: json.loads(l)["messages"][0]["role"] == "system"
def gift_talk(l):   # the player brings up the very item the card says the villager dislikes
    m = json.loads(l)["messages"]; d = re.search(r"\bDislikes?:\s*([^.\n]+)", m[0]["content"])
    if not d: return False
    items = [x.strip().lower() for x in d.group(1).split(",") if x.strip()]
    return any(x["role"] == "user" and any(i and i in x["content"].lower() for i in items) for x in m)
facts = [l for f in ("villager_fact3_j.jsonl", "villager_fact4_j.jsonl") for l in open(f) if card(l) and key(json.loads(l)) not in val]
gifts = [l for l in facts if gift_talk(l)]; other4 = [l for l in open("villager_fact4_j.jsonl") if card(l) and key(json.loads(l)) not in val and not gift_talk(l)]
chat_tops = list(open("rft_chat.jsonl"))
vil = [l for l in open("villager_j.jsonl") if card(l) and key(json.loads(l)) not in val]; R.shuffle(vil)
rest = [l for l in open("mix8.jsonl") if not str(json.loads(l).get("src", "")).startswith("villager")]
goal = [l for l in rest if json.loads(l)["messages"][0]["content"].startswith("GOAL:")]; gs = set(goal); chat = [l for l in rest if l not in gs]; R.shuffle(chat)
mix = gifts * 3 + other4 + chat_tops + list(open("goal_r7.jsonl"))[:3000] + chat[:3000] + vil[:1500]; R.shuffle(mix)
open("mix15.jsonl", "w").writelines(mix)
print(f"[vp] mix15: {len(gifts)} disliked-item talks (x3) + {len(other4)} other fact talks + {len(chat_tops)} judge-picked chat + 3000 goal + 3000 chat + 1500 villager = {len(mix)}", flush=True)
assert len(gifts) >= 300, "too few disliked-item talks"
PY
[ $? = 0 ] || { echo "[vp] too few gift talks, stopping"; date -u > DONE; put DONE DONE; exit 1; }
gpu_clear; mkdir -p agent15
( while sleep 300; do put agent15/train.log train.log > /dev/null; done ) & MIR=$!
python3 -u charlm_sft.py --base agent14.pt --data sftdata --out agent15 --shards 0 --jsonl mix15.jsonl --val-jsonl mix8_select.jsonl --val-convs 450 --min-learned 2 \
    --prompts-json prompts15.json --epochs 1 --lr 2e-5 --warmup 10 --eval-every 4 --hours 1 --ckpt-every 10 --hub baya1116/hypernet-sp-distill:archive/charlm/agent15 2>&1 | grep -vE "Warning|detach|backward|^\[step"
kill $MIR 2>/dev/null; put agent15/train.log train.log
[ -s agent15/model_4bit.pt ] || { echo "[vp] no 4-bit model"; date -u > DONE; put DONE DONE; exit 1; }
python3 -u chat_bits.py --models agent14.pt,agent15/model_final.pt,agent15/model_4bit.pt --sets soda_val.jsonl,npc_val.jsonl,help_val.jsonl,mix8_villager_val.jsonl --n 300 2>&1 | grep "^\[chat\]" | tee chat15.txt
put chat15.txt chat15.txt
python3 -u goal_gen.py play --model agent15/model_4bit.pt --n 100 --split test --seed 5 2>&1 | grep -E "^\[goal\] level" | tee play15.txt; put play15.txt play15.txt
for i in 0 1 2; do python3 -u npc_bench.py reply --model agent15/model_4bit.pt --tag agent15-4bit --bench vb_part$i.jsonl --out r15_$i.jsonl --n 4 > r15_$i.log 2>&1 & done; wait
cat r15_0.jsonl r15_1.jsonl r15_2.jsonl > r_agent15-4bit.jsonl; put r_agent15-4bit.jsonl r_agent15-4bit.jsonl
gpu_clear; python3 -u npc_gen.py judge --inp r_agent15-4bit.jsonl --out judged15.jsonl $J
cat judged13b.jsonl judged14.jsonl judged15.jsonl > judged_all15.jsonl
python3 npc_bench.py report --inp judged_all15.jsonl > report15.txt; python3 talk_rules.py score --inp judged_all15.jsonl > rules15.txt
python3 - <<'PY'
import json, re, collections
FACT = re.compile(r"recall|town|gift_like|gift_dislike|:who|:offer|dark:memory")
M = collections.defaultdict(lambda: [0, 0, 0.0])
for l in open("judged_all15.jsonl"):
    r = json.loads(l)
    if r["pick"] != "best" or FACT.search(r["kind"]): continue
    j = r["judge"]; m = M[r["model"]]; m[0] += 1; m[1] += bool(j.get("usable")); m[2] += float(j.get("charm") or 0)
for k, (n, u, c) in M.items(): print(f"[report] chat kinds {k}: {n} probes, usable {100 * u / n:.1f}%, charm {c / n:.2f}", flush=True)
PY
sed 's/^/[report] /' report15.txt | grep -E "best +(ALL|G:|gift_dislike|gift_like|advice|feeling|town |dark:recall)"; cat rules15.txt
for f in judged15.jsonl report15.txt rules15.txt; do put $f $f; done
put /root/vp/vp.log run.log; date -u > DONE; put DONE DONE; echo "[vp] VPJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vpjob.sh >> /root/vp/vp.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VPJOB_LAUNCHED $(date -u)"
fi
# CTL-END
