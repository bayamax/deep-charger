#!/bin/bash
# Box VD (3060, vLLM image), agent10 for github.com/bayamax/charlm-agent: (1) villager talks that ask facts, the teacher
# told facts must be right in any voice (--facts); (2) all villager talks through the rules (talk_rules.py) and the 4B judge
# shown the reviewer's graded lines (cut before a line graded 0); (3) agent10 from agent9 on them + agent8's replayed mix;
# (4) the same scoring as agent8/9 plus the rules on the bench replies. To the hub: archive/charlm/agent10/
BOXVD_SERIAL=1
if [ -f /root/.boxvd_serial ] && [ "$(cat /root/.boxvd_serial)" -gt "$BOXVD_SERIAL" ] 2>/dev/null; then echo "BOXVD_STALE $BOXVD_SERIAL"; exit 0; fi
echo $BOXVD_SERIAL > /root/.boxvd_serial
if [ ! -e /root/.vd_pre ]; then touch /root/.vd_pre
  echo "[vd] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader)"
  echo "[vd] preflight hub: $(curl -sSL -o /dev/null -m 90 -w '%{size_download} bytes in %{time_total}s' https://huggingface.co/baya1116/charlm-agent/resolve/main/agent6/model_final.pt -r 0-104857599 2>&1)"
fi
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
for f in $(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv $f $f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
RAW="https://raw.githubusercontent.com/bayamax/charlm-agent/main"
mkdir -p /root/vd
for f in npc_gen.py npc_bench.py charlm_sft.py charlm_fold.py charlm_run.py chat_bits.py goal_gen.py talk_rules.py judge_audit.py q4.py eval/gold_shots.jsonl; do
  for try in 1 2 3; do curl -sS -o /root/vd/$(basename $f) "$RAW/$f?nocache=$(date +%s)" && break; sleep 5; done
done
[ -f /root/.bootstrapped ] || { python3 -m pip install -q pyarrow "huggingface_hub[hf_xet]" 2>&1 | tail -1; touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"; }
cat > /root/status.sh <<'ST'
grep -E "^\[(vd|gen|filter|rules|sft|val|gptq|chat|goal|judge|hub)\]|Error|Traceback|out of memory" /root/vd/vd.log 2>/dev/null | tail -n 6 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.vdjob ]; then touch /root/.vdjob
  cat > /root/vdjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vd
H=https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/archive/charlm
G="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 4096 --gpu-mem 0.88 --max-seqs 48"
J="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 8192 --gpu-mem 0.88 --max-seqs 32 --chunk 1000"
gpu_clear() { fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3; }
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/charlm/agent10/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
get() { [ -s $2 ] || curl -sSfL -o $2 "$1" || { rm -f $2; echo "[vd] could not get $1"; return 1; }; }
get $H/agent9/model_final.pt agent9.pt
for f in villager_cozy.jsonl villager_gen.jsonl vbench.jsonl vbench_gen.jsonl; do get $H/villager/$f $f; done
get $H/agent8/villager_more.jsonl villager_more.jsonl
for f in mix8.jsonl mix8_select.jsonl mix8_villager_val.jsonl; do get $H/agent8/data/$f $f; done
for f in soda_val.jsonl npc_val.jsonl; do get $H/agent6/data/$f $f; done
get $H/agent8/judged8.jsonl judged8.jsonl; get $H/agent9/judged9.jsonl judged9.jsonl
# (1) villager talks that ask facts
if [ ! -s villager_fact.done ]; then
  ( while sleep 240; do put villager_fact.jsonl villager_fact.jsonl > /dev/null; done ) & MIR=$!
  gpu_clear; timeout 40m python3 -u npc_gen.py villager --facts 1 --genres cozy,friends,dark --n 3600 --seed 21 --out villager_fact.jsonl --chunk 300 $G
  kill $MIR; put villager_fact.jsonl villager_fact.jsonl; date > villager_fact.done
fi
# (2) rules, then the judge with the reviewer's examples; agent8's held-out villagers left out
cat villager_cozy.jsonl villager_gen.jsonl villager_more.jsonl villager_fact.jsonl > villager_all.jsonl
python3 talk_rules.py filter --inp villager_all.jsonl --out villager_r.jsonl
gpu_clear; [ -s villager_j.jsonl ] || python3 -u npc_gen.py vfilter --shots gold_shots.jsonl --min-grade 1 --inp villager_r.jsonl --out villager_j.jsonl $J
put villager_j.jsonl villager_j.jsonl; put villager_j.jsonl.grades.jsonl villager_j_grades.jsonl
python3 - <<'PY'
import json, random
key = lambda r: (r["messages"][0]["content"], r["messages"][1]["content"])
val = {key(json.loads(l)) for l in open("mix8_villager_val.jsonl")}
vil = [l for l in open("villager_j.jsonl") if key(json.loads(l)) not in val]
rest = [l for l in open("mix8.jsonl") if not str(json.loads(l).get("src", "")).startswith("villager")]
random.seed(10); random.shuffle(rest); mix = vil + rest[:8000]; random.shuffle(mix); open("mix10.jsonl", "w").writelines(mix)
print(f"[vd] mix10: {len(vil)} villager talks (rules + judge), + {len(rest[:8000])} replayed from agent8's mix", flush=True)
PY
put mix10.jsonl data/mix10.jsonl
python3 -c "import json; json.dump([json.loads(l)['messages'][0]['content'] + '\n\n' + json.loads(l)['messages'][1]['content'] for l in list(open('mix8_villager_val.jsonl'))[:3]], open('prompts10.json', 'w'))"
gpu_clear
if [ ! -s agent10/model_final.pt ]; then
  python3 -u charlm_sft.py --base agent9.pt --data sftdata --out agent10 --shards 0 --jsonl mix10.jsonl --val-jsonl mix8_select.jsonl --val-convs 450 --min-learned 2 \
    --prompts-json prompts10.json --epochs 1 --lr 4e-5 --warmup 10 --eval-every 5 --hours 1.5 --ckpt-every 10 --hub baya1116/hypernet-sp-distill:archive/charlm/agent10 2>&1 | grep -vE "Warning|detach|backward|^\[step"
fi
# (4) scoring: chat bits, goal play, villager bench (judge as for agent8/9) and the rules on every model's replies
python3 -u chat_bits.py --models agent9.pt,agent10/model_final.pt,agent10/model_4bit.pt --sets soda_val.jsonl,npc_val.jsonl,mix8_villager_val.jsonl --n 300 2>&1 | grep "^\[chat\]" | tee chat10.txt
put chat10.txt chat10.txt
python3 -u goal_gen.py play --model agent10/model_4bit.pt --n 100 --split test --seed 5 2>&1 | grep -E "^\[goal\] level" | tee play10.txt; put play10.txt play10.txt
cat vbench.jsonl vbench_gen.jsonl > vbench_all.jsonl
python3 - <<'PY'
rows = open("vbench_all.jsonl").readlines()
for i in range(3): open(f"vb_part{i}.jsonl", "w").writelines(rows[i::3])
PY
for i in 0 1 2; do python3 -u npc_bench.py reply --model agent10/model_4bit.pt --tag agent10-4bit --bench vb_part$i.jsonl --out r10_$i.jsonl --n 4 > r10_$i.log 2>&1 & done; wait
cat r10_0.jsonl r10_1.jsonl r10_2.jsonl > r_agent10-4bit.jsonl; put r_agent10-4bit.jsonl r_agent10-4bit.jsonl
gpu_clear; python3 -u npc_gen.py judge --inp r_agent10-4bit.jsonl --out judged10.jsonl --model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 3072 --gpu-mem 0.88 --max-seqs 48 --chunk 1000
cat judged8.jsonl judged9.jsonl judged10.jsonl > judged_all.jsonl
python3 npc_bench.py report --inp judged_all.jsonl > report10.txt; python3 talk_rules.py score --inp judged_all.jsonl > rules10.txt
sed 's/^/[report] /' report10.txt | grep -E "best +(ALL|G:)"; cat rules10.txt
for f in judged10.jsonl report10.txt rules10.txt; do put $f $f; done
date -u > DONE10; put DONE10 DONE; echo "[vd] VDJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vdjob.sh >> /root/vd/vd.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VDJOB_LAUNCHED $(date -u)"
fi
echo "BOXVD_OK serial $BOXVD_SERIAL $(date -u)"
# CTL-END
