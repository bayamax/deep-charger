#!/bin/bash
# Box VF (3060, vLLM image), agent12 = agent11 learning from its own replies: 6 per situation (villagers of three games + other chat), graded by the rules and the 4B judge shown the reviewer's examples; agent12a fine-tuned on the best of each (rejection sampling), agent12 = agent12a tuned on best-vs-worst pairs (DPO). Was: agent10 for github.com/bayamax/charlm-agent: (1) villager talks that ask facts, the teacher
# told facts must be right in any voice (--facts); (2) all villager talks through the rules (talk_rules.py) and the 4B judge
# shown the reviewer's graded lines (cut before a line graded 0); (3) agent10 from agent9 on them + agent8's replayed mix;
# (4) the same scoring as agent8/9 plus the rules on the bench replies. To the hub: archive/charlm/agent10/
BOXVF_SERIAL=1
if [ -f /root/.boxvf_serial ] && [ "$(cat /root/.boxvf_serial)" -gt "$BOXVF_SERIAL" ] 2>/dev/null; then echo "BOXVF_STALE $BOXVF_SERIAL"; exit 0; fi
echo $BOXVF_SERIAL > /root/.boxvf_serial
if [ ! -e /root/.vf_pre ]; then touch /root/.vf_pre
  echo "[vf] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader)"
  echo "[vf] preflight hub: $(curl -sSL -o /dev/null -m 90 -w '%{size_download} bytes in %{time_total}s' https://huggingface.co/baya1116/charlm-agent/resolve/main/agent6/model_final.pt -r 0-104857599 2>&1)"
fi
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
for f in $(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv $f $f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
RAW="https://raw.githubusercontent.com/bayamax/charlm-agent/main"
mkdir -p /root/vf
for f in npc_gen.py npc_bench.py charlm_sft.py charlm_fold.py charlm_run.py chat_bits.py goal_gen.py talk_rules.py judge_audit.py q4.py charlm_dpo.py rft_prep.py eval/gold_shots.jsonl; do
  for try in 1 2 3; do curl -sS -o /root/vf/$(basename $f) "$RAW/$f?nocache=$(date +%s)" && break; sleep 5; done
done
[ -f /root/.bootstrapped ] || { python3 -m pip install -q pyarrow "huggingface_hub[hf_xet]" 2>&1 | tail -1; touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"; }
cat > /root/status.sh <<'ST'
grep -E "^\[(vf|dpo|rft|linejudge|bench|gen|filter|rules|sft|val|gptq|chat|goal|judge|hub)\]|Error|Traceback|out of memory" /root/vf/vf.log 2>/dev/null | tail -n 6 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.vfjob ]; then touch /root/.vfjob
  cat > /root/vfjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vf
H=https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/archive/charlm
J="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 3072 --gpu-mem 0.88 --max-seqs 48 --chunk 1000"
JL="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 6144 --gpu-mem 0.88 --max-seqs 48 --chunk 2000"
gpu_clear() { fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3; }
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/charlm/agent12/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
get() { [ -s $2 ] || curl -sSfL -o $2 "$1" || { rm -f $2; echo "[vf] could not get $1"; return 1; }; }
get $H/agent11/model_final.pt agent11.pt; get $H/agent10/villager_j.jsonl villager_j.jsonl
for f in vbench.jsonl vbench_gen.jsonl; do get $H/villager/$f $f; done
for f in mix8.jsonl mix8_select.jsonl mix8_villager_val.jsonl; do get $H/agent8/data/$f $f; done
for f in soda_val.jsonl npc_val.jsonl; do get $H/agent6/data/$f $f; done
get $H/agent12/data/help_val.jsonl help_val.jsonl; get $H/agent11/judged11.jsonl judged11.jsonl
# situations (a card, the talk so far) from villager and other talks; agent8's held-out villagers left out; anchors and replay
[ -s goal_r.jsonl ] || python3 goal_gen.py data --n 4000 --levels 1,2,3,3,3,4,4 --split train --seed 71 --out goal_r.jsonl
python3 - <<'PY'
import json, random
key = lambda r: (r["messages"][0]["content"], r["messages"][1]["content"])
val = {key(json.loads(l)) for l in open("mix8_villager_val.jsonl")}
R = random.Random(12)
card = lambda l: json.loads(l)["messages"][0]["role"] == "system"
vil = [l for l in open("villager_j.jsonl") if card(l) and key(json.loads(l)) not in val]; R.shuffle(vil)
rest = [l for l in open("mix8.jsonl") if not str(json.loads(l).get("src", "")).startswith("villager")]
goal = [l for l in rest if json.loads(l)["messages"][0]["content"].startswith("GOAL:")]; gs = set(goal); chat = [l for l in rest if l not in gs]
R.shuffle(chat); cchat = [l for l in chat if card(l)]
sit = vil[:2000] + cchat[:600]; R.shuffle(sit)
for i in range(4): open(f"sit{i}.jsonl", "w").writelines(sit[i::4])
open("replay.jsonl", "w").writelines(list(open("goal_r.jsonl"))[:3000] + chat[600:3600] + vil[2000:3500])
open("anchor.jsonl", "w").writelines(list(open("goal_r.jsonl"))[3000:] + chat[3600:4600])
print(f"[vf] {len(sit)} situations ({min(2000, len(vil))} villager, {min(600, len(cchat))} other chat); replay {3000 + len(chat[600:3600]) + len(vil[2000:3500])}; anchors {1000 + len(chat[3600:4600])}", flush=True)
PY
# the model's own replies, 6 each, 4 at a time on the GPU
if [ ! -s samples.jsonl ]; then
  for i in 0 1 2 3; do python3 -u npc_bench.py sample --model agent11.pt --tag agent11 --bench sit$i.jsonl --situations 100000 --n 6 --temp 0.8 --seed $i --out samp$i.jsonl > samp$i.log 2>&1 & done; wait
  python3 - <<'PY'
import json
with open("samples.jsonl", "w") as f:
    for i in range(4):
        for l in open(f"samp{i}.jsonl"):
            r = json.loads(l); r["sid"] = r["sid"] * 4 + i; f.write(json.dumps(r) + "\n")
PY
fi
put samples.jsonl samples.jsonl
python3 rft_prep.py tojudge --samples samples.jsonl --out lj_in.jsonl
gpu_clear; [ -s lj_out.jsonl ] || python3 -u npc_gen.py linejudge --variant fewshot --shots gold_shots.jsonl --inp lj_in.jsonl --out lj_out.jsonl $JL
put lj_out.jsonl lj_out.jsonl
python3 rft_prep.py pick --samples samples.jsonl --grades lj_out.jsonl --rft rft.jsonl --pairs pairs.jsonl
for f in rft.jsonl pairs.jsonl; do put $f data/$f; done
cat rft.jsonl replay.jsonl > mix12a.jsonl
python3 -c "import json; json.dump([json.loads(l)['messages'][0]['content'] + '\n\n' + json.loads(l)['messages'][1]['content'] for l in list(open('mix8_villager_val.jsonl'))[:3]], open('prompts12.json', 'w'))"
gpu_clear
# agent12a: the best of each situation (rejection sampling) + replay
[ -s agent12a/model_final.pt ] || python3 -u charlm_sft.py --base agent11.pt --data sftdata --out agent12a --shards 0 --jsonl mix12a.jsonl --val-jsonl mix8_select.jsonl --val-convs 450 --min-learned 2 \
    --prompts-json prompts12.json --epochs 1 --lr 2e-5 --warmup 10 --eval-every 4 --hours 1 --ckpt-every 10 --hub baya1116/hypernet-sp-distill:archive/charlm/agent12/a 2>&1 | grep -vE "Warning|detach|backward|^\[step"
# agent12: agent12a on best-vs-worst pairs (DPO), the anchors beside; then its 4-bit file
[ -s agent12/model_final.pt ] || python3 -u charlm_dpo.py --base agent12a/model_final.pt --pairs pairs.jsonl --anchor anchor.jsonl --out agent12 --beta 0.1 --lr 5e-6 --epochs 2 --batch 8 --anchor-batch 4 2>&1 | grep -vE "Warning|detach"
[ -s agent12/model_4bit.pt ] || python3 -u charlm_sft.py --base agent12/model_final.pt --data sftdata --out agent12 --shards 0 --jsonl anchor.jsonl --val-jsonl mix8_select.jsonl --val-convs 450 --min-learned 2 \
    --epochs 0 --prompts-json prompts12.json --hub baya1116/hypernet-sp-distill:archive/charlm/agent12/b 2>&1 | grep -E "^\[(val|gptq|sft)\]"
# scoring: chat bits, goal play, the villager bench (the judge as before) and the rules
python3 -u chat_bits.py --models agent11.pt,agent12a/model_final.pt,agent12/model_final.pt,agent12a/model_4bit.pt,agent12/model_4bit.pt --sets soda_val.jsonl,npc_val.jsonl,help_val.jsonl,mix8_villager_val.jsonl --n 300 2>&1 | grep "^\[chat\]" | tee chat12.txt
put chat12.txt chat12.txt
: > play12.txt
for m in agent12a-4bit:agent12a/model_4bit.pt agent12-4bit:agent12/model_4bit.pt; do IFS=: read t f <<< "$m"
  echo "[vf] $t:" | tee -a play12.txt; python3 -u goal_gen.py play --model $f --n 100 --split test --seed 5 2>&1 | grep -E "^\[goal\] level" | tee -a play12.txt
done
put play12.txt play12.txt
cat vbench.jsonl vbench_gen.jsonl > vbench_all.jsonl
python3 - <<'PY'
rows = open("vbench_all.jsonl").readlines()
for i in range(3): open(f"vb_part{i}.jsonl", "w").writelines(rows[i::3])
PY
for m in agent12a-4bit:agent12a/model_4bit.pt agent12-4bit:agent12/model_4bit.pt; do IFS=: read t f <<< "$m"
  for i in 0 1 2; do python3 -u npc_bench.py reply --model $f --tag $t --bench vb_part$i.jsonl --out r_${t}_$i.jsonl --n 4 > r_${t}_$i.log 2>&1 & done; wait
  cat r_${t}_0.jsonl r_${t}_1.jsonl r_${t}_2.jsonl > r_$t.jsonl; put r_$t.jsonl r_$t.jsonl
done
gpu_clear; cat r_agent12a-4bit.jsonl r_agent12-4bit.jsonl > replies12.jsonl
python3 -u npc_gen.py judge --inp replies12.jsonl --out judged12.jsonl $J
cat judged11.jsonl judged12.jsonl > judged_all.jsonl
python3 npc_bench.py report --inp judged_all.jsonl > report12.txt; python3 talk_rules.py score --inp judged_all.jsonl > rules12.txt
sed 's/^/[report] /' report12.txt | grep -E "best +(ALL|G:)"; cat rules12.txt
for f in judged12.jsonl report12.txt rules12.txt; do put $f $f; done
date -u > DONE12; put DONE12 DONE; echo "[vf] VFJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vfjob.sh >> /root/vf/vf.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VFJOB_LAUNCHED $(date -u)"
fi
echo "BOXVF_OK serial $BOXVF_SERIAL $(date -u)"
# CTL-END
