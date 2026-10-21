#!/bin/bash
# Box VB (3060, vLLM image), all for github.com/bayamax/charlm-agent: (1) more villager talks from Qwen3.5-4B for three
# kinds of games (cozy life-sim, friend-collection sim, dark fantasy), 50 minutes; (2) agent8 fine-tuned from agent6 on the
# GPU: those villagers + NPC/help/everyday chat + the new goal format (state in any order, the raise step with effects);
# (3) scoring: chat bits, goal play (agent6 and agent8 on the same shuffled test), and the villager bench judged by the 4B.
# Everything goes to the hub (baya1116/hypernet-sp-distill, archive/charlm/agent8/) as it is made.
BOXVB_SERIAL=7
if [ -f /root/.boxvb_serial ] && [ "$(cat /root/.boxvb_serial)" -gt "$BOXVB_SERIAL" ] 2>/dev/null; then echo "BOXVB_STALE $BOXVB_SERIAL"; exit 0; fi
echo $BOXVB_SERIAL > /root/.boxvb_serial
if [ ! -e /root/.vb_pre ]; then touch /root/.vb_pre
  echo "[vb] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader)"
  echo "[vb] preflight hub: $(curl -sSL -o /dev/null -m 90 -w '%{size_download} bytes in %{time_total}s' https://huggingface.co/baya1116/charlm-agent/resolve/main/agent6/model_final.pt -r 0-104857599 2>&1)"
fi
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
for f in $(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv $f $f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
echo "[vb] HF token: $([ -s /root/.hf_token ] && echo present || echo MISSING)"
RAW="https://raw.githubusercontent.com/bayamax/charlm-agent/main"
mkdir -p /root/vb
for f in npc_gen.py npc_bench.py charlm_sft.py charlm_fold.py charlm_run.py chat_bits.py soda_prep.py goal_gen.py card_pad.py mix_prep.py q4.py; do
  for try in 1 2 3; do curl -sS -o /root/vb/$f "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/vb/$f && break; sleep 5; done
done
if [ ! -f /root/.bootstrapped ]; then
  python3 -m pip install -q pyarrow "huggingface_hub[hf_xet]" 2>&1 | tail -1
  python3 -c "import vllm, torch; print('[vb] vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.get_device_name(0))" 2>&1 | tail -1
  touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"
fi
cat > /root/status.sh <<'ST'
grep -E "^\[(vb|gen|mix|sft|val|gptq|chat|goal|judge|hub|pad|soda|filter)\]|Error|Traceback" /root/vb/vb.log /root/vb/vb3.log 2>/dev/null | tail -n 8 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh

# serial 2, the user: the goal's variable first in STATE is fine as the game's protocol: goal_gen.py makes 75% of STATE
# lines that way (25% in another order, for games that do not), and play scores both orders; the job takes the new
# file when it gets to the goal data (not made yet: the villager run is still going)
if [ ! -e /root/.vb_s2 ]; then touch /root/.vb_s2
  curl -sS -o /root/vb/goal_gen.new "$RAW/goal_gen.py?nocache=$(date +%s)" && python3 -m py_compile /root/vb/goal_gen.new && mv /root/vb/goal_gen.new /root/vb/goal_gen.py
  [ -d /root/vb/agent8 ] || rm -f /root/vb/goal_train.jsonl /root/vb/goal_val.jsonl /root/vb/mix8.jsonl
  echo "[vb] s2: goal_gen.py with the protocol order $(grep -c 'SHUF = A.shuffle' /root/vb/goal_gen.py) $(date -u)"
fi
# serial 3, the user: grade the teacher's (Qwen's) own lines too: the job stopped (the villager run left to finish), and
# a new one waits for it, filters every villager talk with the judge (cut before the first bad line), then goes on
if [ ! -e /root/.vb_s3 ]; then touch /root/.vb_s3
  pkill -f "vbjo[b].sh"; sleep 2
  cat > /root/vbjob2.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vb
H=https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/archive/charlm
G="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 4096 --gpu-mem 0.88 --max-seqs 48"
J="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 3072 --gpu-mem 0.88 --max-seqs 48 --chunk 1000"
gpu_clear() { fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3; }
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/charlm/agent8/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
get() { [ -s $2 ] || curl -sSfL -o $2 "$1" || { rm -f $2; echo "[vb] could not get $1"; return 1; }; }
# what is already made, and the sources
get https://huggingface.co/baya1116/charlm-agent/resolve/main/agent6/model_final.pt agent6.pt
for f in villager_cozy.jsonl villager_gen.jsonl vbench.jsonl vbench_gen.jsonl; do get $H/villager/$f $f; done
for f in train_new.jsonl t3_0.jsonl t3_1.jsonl help_0.jsonl help_1.jsonl help_2.jsonl val_all.jsonl; do get $H/npc_data2/$f $f; done
get $H/npc_data/stage2_public.jsonl stage2_public.jsonl
for f in soda_val.jsonl npc_val.jsonl; do get $H/agent6/data/$f $f; done
# the villager run (left going when this job took over) ends first; then the judge grades the teacher's lines
while pgrep -f "npc_ge[n].py villager " > /dev/null; do sleep 30; done
put villager_more.jsonl villager_more.jsonl; date > villager_more.done
cat villager_cozy.jsonl villager_gen.jsonl villager_more.jsonl > villager_all.jsonl
gpu_clear; [ -s villager_f.jsonl ] || python3 -u npc_gen.py vfilter --inp villager_all.jsonl --out villager_f.jsonl $J
put villager_f.jsonl villager_f.jsonl
gpu_clear
# (2) the mix and agent8
[ -s soda_new.jsonl ] || { get "https://huggingface.co/api/datasets/allenai/soda/parquet/default/train/1.parquet" soda_1.parquet && python3 soda_prep.py --parquet soda_1.parquet --n 5000 --val 0 --seed 7 --out soda_new.jsonl --val-out /dev/null && rm -f soda_1.parquet; }
[ -s goal_train.jsonl ] || python3 goal_gen.py data --n 8000 --levels 1,2,2,3,3,4,4 --split train --seed 51 --out goal_train.jsonl
[ -s goal_val.jsonl ] || python3 goal_gen.py data --n 300 --split train --seed 9 --out goal_val.jsonl
[ -s mix8.jsonl ] || python3 mix_prep.py --villager villager_f.jsonl --npc train_new.jsonl,t3_0.jsonl,t3_1.jsonl \
  --helpdata help_0.jsonl,help_1.jsonl,help_2.jsonl --soda soda_new.jsonl --public stage2_public.jsonl --goal goal_train.jsonl \
  --npc-val npc_val.jsonl --soda-val soda_val.jsonl --goal-val goal_val.jsonl --out mix8.jsonl
for f in mix8.jsonl mix8_villager_val.jsonl mix8_select.jsonl; do put $f data/$f; done
python3 -c "import json; json.dump([json.loads(l)['messages'][0]['content'] + '\n\n' + json.loads(l)['messages'][1]['content'] for l in list(open('mix8_villager_val.jsonl'))[:3]] + [json.loads(l)['messages'][0]['content'] for l in list(open('goal_val.jsonl'))[:1]], open('prompts8.json', 'w'))"
if [ ! -s agent8/model_final.pt ]; then
  python3 -u charlm_sft.py --base agent6.pt --data sftdata --out agent8 --shards 0 --jsonl mix8.jsonl --val-jsonl mix8_select.jsonl --val-convs 450 --min-learned 2 \
    --prompts-json prompts8.json --epochs 1 --lr 5e-5 --warmup 20 --eval-every 10 --hours 2 --ckpt-every 10 --hub baya1116/hypernet-sp-distill:archive/charlm/agent8 2>&1 | grep -vE "Warning|detach|backward|^\[step"
fi
# (3) scoring
python3 -u chat_bits.py --models agent6.pt,agent8/model_final.pt,agent8/model_4bit.pt --sets soda_val.jsonl,npc_val.jsonl,mix8_help_val.jsonl,mix8_villager_val.jsonl --n 300 2>&1 | grep "^\[chat\]" | tee chat8.txt
put chat8.txt chat8.txt
: > play8.txt
for m in agent8-final:agent8/model_final.pt:200 agent8-4bit:agent8/model_4bit.pt:100; do IFS=: read t f n <<< "$m"   # agent6 on this test: run locally
  echo "[vb] $t, made-up names, state in any order:" | tee -a play8.txt
  python3 -u goal_gen.py play --model $f --n $n --split test --seed 5 --raise-fx $([ $t = agent6 ] && echo 0 || echo 1) 2>&1 | grep -E "^\[goal\]" | tee -a play8.txt
done
put play8.txt play8.txt
cat vbench.jsonl vbench_gen.jsonl > vbench_all.jsonl
get $H/agent8/r_agent6.jsonl r_agent6.jsonl   # agent6's replies are made on the local CPU meanwhile
[ -s r_agent8-4bit.jsonl ] || python3 -u npc_bench.py reply --model agent8/model_4bit.pt --tag agent8-4bit --bench vbench_all.jsonl --out r_agent8-4bit.jsonl --n 4
[ -s r_agent6.jsonl ] || python3 -u npc_bench.py reply --model agent6.pt --tag agent6 --bench vbench_all.jsonl --out r_agent6.jsonl --n 4
put r_agent8-4bit.jsonl r_agent8-4bit.jsonl; cat r_agent6.jsonl r_agent8-4bit.jsonl > replies8.jsonl
gpu_clear; python3 -u npc_gen.py judge --inp replies8.jsonl --out judged8.jsonl $J
python3 npc_bench.py report --inp judged8.jsonl > report8.txt; sed 's/^/[report] /' report8.txt | head -80
for f in judged8.jsonl report8.txt; do put $f $f; done
date -u > DONE; put DONE DONE; echo "[vb] VBJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vbjob2.sh >> /root/vb/vb.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[vb] s3: job 2 (with the filter) started $(date -u)"
fi
# serial 4, the user: raise the judge's bar and polish on good data only: a strict regrade of every villager talk runs now
# beside job 2 (its replies leave the GPU nearly free); once job 2 is done, agent9 = agent8 fine-tuned on the talks whose
# every line got 2 (agent8's held-out villagers left out) with some of agent8's mix replayed, then scored the same way
if [ ! -e /root/.vb_s5 ]; then touch /root/.vb_s5; rm -f /root/.vb_s4; pkill -f "vbjob[3].sh"; pkill -f "npc_ge[n].py vfilter --strict"; sleep 3; fi   # serial 5: job 3 again, now safe if job 2's GPU clearing cuts its regrade
if [ ! -e /root/.vb_s4 ]; then touch /root/.vb_s4
  for f in npc_gen.py npc_bench.py; do curl -sS -o /root/vb/$f.new "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/vb/$f.new && mv /root/vb/$f.new /root/vb/$f; done
  cat > /root/vbjob3.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vb
JS="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 3072 --gpu-mem 0.72 --max-seqs 32 --chunk 1000"
J="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 3072 --gpu-mem 0.88 --max-seqs 48 --chunk 1000"
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/charlm/agent9/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
[ -s villager_hq.jsonl ] || python3 -u npc_gen.py vfilter --strict 1 --min-grade 2 --inp villager_all.jsonl --out villager_hq.jsonl $JS
while [ ! -s DONE ]; do sleep 60; done   # job 2 (agent8's scoring) first; its GPU clearing may have cut the regrade short
[ -s villager_hq.jsonl ] || { fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3; python3 -u npc_gen.py vfilter --strict 1 --min-grade 2 --inp villager_all.jsonl --out villager_hq.jsonl $J; }
put villager_hq.jsonl villager_hq.jsonl; put villager_hq.jsonl.grades.jsonl villager_hq_grades.jsonl
python3 - <<'PY'
import json, random
key = lambda r: (r["messages"][0]["content"], r["messages"][1]["content"])
val = {key(json.loads(l)) for l in open("mix8_villager_val.jsonl")}
hq = [l for l in open("villager_hq.jsonl") if key(json.loads(l)) not in val]
rest = [l for l in open("mix8.jsonl") if not str(json.loads(l).get("src", "")).startswith("villager")]
random.seed(9); random.shuffle(rest); mix = hq + rest[:6000]; random.shuffle(mix); open("mix9.jsonl", "w").writelines(mix)
print(f"[vb] mix9: {len(hq)} villager talks with every line graded 2 (+ {len(rest[:6000])} replayed from agent8's mix)", flush=True)
PY
put mix9.jsonl data/mix9.jsonl
if [ ! -s agent9/model_final.pt ]; then
  python3 -u charlm_sft.py --base agent8/model_final.pt --data sftdata --out agent9 --shards 0 --jsonl mix9.jsonl --val-jsonl mix8_select.jsonl --val-convs 450 --min-learned 2 \
    --prompts-json prompts8.json --epochs 1 --lr 3e-5 --warmup 10 --eval-every 5 --hours 1 --ckpt-every 10 --hub baya1116/hypernet-sp-distill:archive/charlm/agent9 2>&1 | grep -vE "Warning|detach|backward|^\[step"
fi
python3 -u chat_bits.py --models agent8/model_final.pt,agent9/model_final.pt,agent9/model_4bit.pt --sets soda_val.jsonl,npc_val.jsonl,mix8_help_val.jsonl,mix8_villager_val.jsonl --n 300 2>&1 | grep "^\[chat\]" | tee chat9.txt
put chat9.txt chat9.txt
python3 -u goal_gen.py play --model agent9/model_4bit.pt --n 100 --split test --seed 5 2>&1 | grep -E "^\[goal\] level" | tee play9.txt; put play9.txt play9.txt
python3 - <<'PY'
rows = open("vbench_all.jsonl").readlines()
for i in range(3): open(f"vb_part{i}.jsonl", "w").writelines(rows[i::3])
PY
for i in 0 1 2; do python3 -u npc_bench.py reply --model agent9/model_4bit.pt --tag agent9-4bit --bench vb_part$i.jsonl --out r9_part$i.jsonl --n 4 > r9_part$i.log 2>&1 & done; wait
cat r9_part0.jsonl r9_part1.jsonl r9_part2.jsonl > r_agent9-4bit.jsonl; put r_agent9-4bit.jsonl r_agent9-4bit.jsonl
fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3
python3 -u npc_gen.py judge --inp r_agent9-4bit.jsonl --out judged9.jsonl $J
cat judged8.jsonl judged9.jsonl > judged89.jsonl; python3 npc_bench.py report --inp judged89.jsonl > report9.txt; sed 's/^/[report] /' report9.txt | head -60
for f in judged9.jsonl report9.txt; do put $f $f; done
date -u > DONE9; put DONE9 DONE; echo "[vb] VBJOB3_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vbjob3.sh >> /root/vb/vb3.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[vb] s4: job 3 (strict regrade, then agent9) started $(date -u)"
fi
# serial 6: agent8 lost facts (the player's name recalled 100% -> 78%, shop hours 57% -> 26%) while the judge still called
# such replies usable: job 3 stopped (its regrade had just begun) and job 4 runs it with rule checks on top: a talk is cut
# before a line that misses the player's name when asked to recall it, or a shop's opening time when asked; more of the
# fact-heavy NPC/help talks replayed; agent8's float replies scored too (is the loss the 4-bit's or the training's?)
if [ ! -e /root/.vb_s6 ]; then touch /root/.vb_s6
  pkill -f "vbjob[3].sh"; pkill -f "npc_ge[n].py vfilter"; sleep 3; fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3
  for f in npc_gen.py npc_bench.py; do curl -sS -o /root/vb/$f.new "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/vb/$f.new && mv /root/vb/$f.new /root/vb/$f; done
  cat > /root/vbjob4.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vb
J="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 3072 --gpu-mem 0.88 --max-seqs 48 --chunk 1000"
gpu_clear() { fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3; }
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/charlm/agent9/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
[ -s villager_hq.jsonl ] || { gpu_clear; python3 -u npc_gen.py vfilter --strict 1 --min-grade 2 --inp villager_all.jsonl --out villager_hq.jsonl $J; }
put villager_hq.jsonl villager_hq.jsonl; put villager_hq.jsonl.grades.jsonl villager_hq_grades.jsonl
python3 - <<'PY'
import json, random, re
key = lambda r: (r["messages"][0]["content"], r["messages"][1]["content"])
val = {key(json.loads(l)) for l in open("mix8_villager_val.jsonl")}
NAME = re.compile(r"\b(?:my name is|i'm|i am|call me)\s+([A-Z][a-z]+)", re.I)
def facts_ok(r):
    """cut before a line that misses the player's name when asked to recall it, or a shop's opening time when asked"""
    card = r["messages"][0]["content"] if r["messages"][0]["role"] == "system" else ""
    hours = {m.group(1).lower(): m.group(2) for m in re.finditer(r"([A-Za-z ]+?) is open (\d+ am) to", card)}
    out, name, prev = [], None, ""
    for m in r["messages"]:
        if m["role"] == "user":
            g = NAME.search(m["content"]); name = g.group(1) if g and g.group(1).lower() not in ("not", "so", "just", "here", "fine", "good", "sorry", "back") else name
        if m["role"] == "assistant":
            if name and re.search(r"\b(remember|know|what'?s|what is)\b.*\bname\b", prev, re.I) and name.lower() not in m["content"].lower(): break
            shop = next((k for k in hours if k.split()[-1] in prev.lower()), None)
            if shop and re.search(r"\b(open|hours|close)", prev, re.I) and hours[shop].split()[0] not in m["content"]: break
        out.append(m); prev = m["content"] if m["role"] == "user" else prev
    while out and out[-1]["role"] != "assistant": out.pop()
    return out if sum(x["role"] == "assistant" for x in out) >= 2 else None
hq, cut = [], 0
for l in open("villager_hq.jsonl"):
    r = json.loads(l)
    if key(r) in val: continue
    ms = facts_ok(r)
    if ms is None: cut += 1; continue
    cut += len(ms) < len(r["messages"]); r["messages"] = ms; hq.append(json.dumps(r) + "\n")
rest = [l for l in open("mix8.jsonl") if not str(json.loads(l).get("src", "")).startswith("villager")]
random.seed(9); random.shuffle(rest); mix = hq + rest[:8000]; random.shuffle(mix); open("mix9.jsonl", "w").writelines(mix)
print(f"[vb] mix9: {len(hq)} villager talks with every line graded 2 and the facts right ({cut} cut or dropped by the fact rules), + {len(rest[:8000])} replayed", flush=True)
PY
put mix9.jsonl data/mix9.jsonl
gpu_clear
if [ ! -s agent9/model_final.pt ]; then
  python3 -u charlm_sft.py --base agent8/model_final.pt --data sftdata --out agent9 --shards 0 --jsonl mix9.jsonl --val-jsonl mix8_select.jsonl --val-convs 450 --min-learned 2 \
    --prompts-json prompts8.json --epochs 1 --lr 3e-5 --warmup 10 --eval-every 5 --hours 1 --ckpt-every 10 --hub baya1116/hypernet-sp-distill:archive/charlm/agent9 2>&1 | grep -vE "Warning|detach|backward|^\[step"
fi
python3 -u chat_bits.py --models agent8/model_final.pt,agent9/model_final.pt,agent9/model_4bit.pt --sets soda_val.jsonl,npc_val.jsonl,mix8_help_val.jsonl,mix8_villager_val.jsonl --n 300 2>&1 | grep "^\[chat\]" | tee chat9.txt
put chat9.txt chat9.txt
python3 -u goal_gen.py play --model agent9/model_4bit.pt --n 100 --split test --seed 5 2>&1 | grep -E "^\[goal\] level" | tee play9.txt; put play9.txt play9.txt
python3 - <<'PY'
rows = open("vbench_all.jsonl").readlines()
for i in range(3): open(f"vb_part{i}.jsonl", "w").writelines(rows[i::3])
PY
for m in agent9-4bit:agent9/model_4bit.pt agent8-final:agent8/model_final.pt; do IFS=: read t f <<< "$m"
  for i in 0 1 2; do python3 -u npc_bench.py reply --model $f --tag $t --bench vb_part$i.jsonl --out r_${t}_$i.jsonl --n 4 > r_${t}_$i.log 2>&1 & done; wait
  cat r_${t}_0.jsonl r_${t}_1.jsonl r_${t}_2.jsonl > r_$t.jsonl; put r_$t.jsonl r_$t.jsonl
done
gpu_clear; cat r_agent9-4bit.jsonl r_agent8-final.jsonl > replies9.jsonl
python3 -u npc_gen.py judge --inp replies9.jsonl --out judged9.jsonl $J
cat judged8.jsonl judged9.jsonl > judged89.jsonl; python3 npc_bench.py report --inp judged89.jsonl > report9.txt; sed 's/^/[report] /' report9.txt | grep -E "ALL|G:" 
for f in judged9.jsonl report9.txt; do put $f $f; done
date -u > DONE9; put DONE9 DONE; echo "[vb] VBJOB4_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vbjob4.sh >> /root/vb/vb3.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[vb] s6: job 3 stopped; job 4 (strict regrade + fact rules, agent9, agent8-float scored too) started $(date -u)"
fi
# serial 7, the user: a quality check on the judge itself: once job 4 is done, the 4B grades the reviewer's 150 gold lines
# with three prompts (plain, a checklist of the failures the audit found, the checklist with 18 graded examples)
if [ ! -e /root/.vb_s7 ]; then touch /root/.vb_s7
  cat > /root/vbjob5.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vb
R2="https://raw.githubusercontent.com/bayamax/charlm-agent/main"
for f in npc_gen.py judge_audit.py eval/gold_villager_lines.jsonl eval/gold_shots.jsonl; do curl -sS -o $(basename $f) "$R2/$f?nocache=$(date +%s)"; done
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/charlm/judge/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
while [ ! -s DONE9 ]; do sleep 60; done
J="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 6144 --gpu-mem 0.88 --max-seqs 32"
python3 -c "import json; open('gold_lab.jsonl','w').writelines(json.dumps({'id': json.loads(l)['id'], 'grade': json.loads(l)['grade']}) + '\n' for l in open('gold_villager_lines.jsonl'))"
: > judge_eval.txt
for v in plain checklist fewshot; do
  fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3
  python3 -u npc_gen.py linejudge --variant $v --shots gold_shots.jsonl --inp gold_villager_lines.jsonl --out lj_$v.jsonl $J
  echo "== 4B $v (the 132 lines not used as examples)" | tee -a judge_eval.txt; python3 judge_audit.py score --key lj_$v.jsonl --labels gold_lab.jsonl | tee -a judge_eval.txt
done
put judge_eval.txt judge_eval_4b.txt; date -u > DONE_J; put DONE_J DONE; echo "[vb] VBJOB5_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vbjob5.sh >> /root/vb/vb3.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[vb] s7: job 5 (the judge on the gold lines) queued after job 4 $(date -u)"
fi
if [ ! -e /root/.vbjob ]; then touch /root/.vbjob
  cat > /root/vbjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vb
H=https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/archive/charlm
G="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 4096 --gpu-mem 0.88 --max-seqs 48"
J="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 3072 --gpu-mem 0.88 --max-seqs 48 --chunk 1000"
gpu_clear() { fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3; }
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/charlm/agent8/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
get() { [ -s $2 ] || curl -sSfL -o $2 "$1" || { rm -f $2; echo "[vb] could not get $1"; return 1; }; }
# what is already made, and the sources
get https://huggingface.co/baya1116/charlm-agent/resolve/main/agent6/model_final.pt agent6.pt
for f in villager_cozy.jsonl villager_gen.jsonl vbench.jsonl vbench_gen.jsonl; do get $H/villager/$f $f; done
for f in train_new.jsonl t3_0.jsonl t3_1.jsonl help_0.jsonl help_1.jsonl help_2.jsonl val_all.jsonl; do get $H/npc_data2/$f $f; done
get $H/npc_data/stage2_public.jsonl stage2_public.jsonl
for f in soda_val.jsonl npc_val.jsonl; do get $H/agent6/data/$f $f; done
# (1) more villagers, 50 minutes, mirrored every 4
if [ ! -s villager_more.done ]; then
  ( while sleep 240; do put villager_more.jsonl villager_more.jsonl > /dev/null; done ) & MIR=$!
  gpu_clear; timeout 50m python3 -u npc_gen.py villager --genres cozy,friends,dark,cozy --n 4000 --seed 11 --out villager_more.jsonl --chunk 250 $G
  kill $MIR; put villager_more.jsonl villager_more.jsonl; date > villager_more.done
fi
gpu_clear
# (2) the mix and agent8
[ -s soda_new.jsonl ] || { get "https://huggingface.co/api/datasets/allenai/soda/parquet/default/train/1.parquet" soda_1.parquet && python3 soda_prep.py --parquet soda_1.parquet --n 5000 --val 0 --seed 7 --out soda_new.jsonl --val-out /dev/null && rm -f soda_1.parquet; }
[ -s goal_train.jsonl ] || python3 goal_gen.py data --n 8000 --levels 1,2,2,3,3,4,4 --split train --seed 51 --out goal_train.jsonl
[ -s goal_val.jsonl ] || python3 goal_gen.py data --n 300 --split train --seed 9 --out goal_val.jsonl
[ -s mix8.jsonl ] || python3 mix_prep.py --villager villager_cozy.jsonl,villager_gen.jsonl,villager_more.jsonl --npc train_new.jsonl,t3_0.jsonl,t3_1.jsonl \
  --helpdata help_0.jsonl,help_1.jsonl,help_2.jsonl --soda soda_new.jsonl --public stage2_public.jsonl --goal goal_train.jsonl \
  --npc-val npc_val.jsonl --soda-val soda_val.jsonl --goal-val goal_val.jsonl --out mix8.jsonl
for f in mix8.jsonl mix8_villager_val.jsonl mix8_select.jsonl; do put $f data/$f; done
python3 -c "import json; json.dump([json.loads(l)['messages'][0]['content'] + '\n\n' + json.loads(l)['messages'][1]['content'] for l in list(open('mix8_villager_val.jsonl'))[:3]] + [json.loads(l)['messages'][0]['content'] for l in list(open('goal_val.jsonl'))[:1]], open('prompts8.json', 'w'))"
if [ ! -s agent8/model_final.pt ]; then
  python3 -u charlm_sft.py --base agent6.pt --data sftdata --out agent8 --shards 0 --jsonl mix8.jsonl --val-jsonl mix8_select.jsonl --val-convs 450 --min-learned 2 \
    --prompts-json prompts8.json --epochs 1 --lr 5e-5 --warmup 20 --eval-every 10 --hours 2 --ckpt-every 10 --hub baya1116/hypernet-sp-distill:archive/charlm/agent8 2>&1 | grep -vE "Warning|detach|backward|^\[step"
fi
# (3) scoring
python3 -u chat_bits.py --models agent6.pt,agent8/model_final.pt,agent8/model_4bit.pt --sets soda_val.jsonl,npc_val.jsonl,mix8_help_val.jsonl,mix8_villager_val.jsonl --n 300 2>&1 | grep "^\[chat\]" | tee chat8.txt
put chat8.txt chat8.txt
: > play8.txt
for m in agent8-final:agent8/model_final.pt:200 agent8-4bit:agent8/model_4bit.pt:100; do IFS=: read t f n <<< "$m"   # agent6 on this test: run locally
  echo "[vb] $t, made-up names, state in any order:" | tee -a play8.txt
  python3 -u goal_gen.py play --model $f --n $n --split test --seed 5 --raise-fx $([ $t = agent6 ] && echo 0 || echo 1) 2>&1 | grep -E "^\[goal\]" | tee -a play8.txt
done
put play8.txt play8.txt
cat vbench.jsonl vbench_gen.jsonl > vbench_all.jsonl
get $H/agent8/r_agent6.jsonl r_agent6.jsonl   # agent6's replies are made on the local CPU meanwhile
[ -s r_agent8-4bit.jsonl ] || python3 -u npc_bench.py reply --model agent8/model_4bit.pt --tag agent8-4bit --bench vbench_all.jsonl --out r_agent8-4bit.jsonl --n 4
[ -s r_agent6.jsonl ] || python3 -u npc_bench.py reply --model agent6.pt --tag agent6 --bench vbench_all.jsonl --out r_agent6.jsonl --n 4
put r_agent8-4bit.jsonl r_agent8-4bit.jsonl; cat r_agent6.jsonl r_agent8-4bit.jsonl > replies8.jsonl
gpu_clear; python3 -u npc_gen.py judge --inp replies8.jsonl --out judged8.jsonl $J
python3 npc_bench.py report --inp judged8.jsonl > report8.txt; sed 's/^/[report] /' report8.txt | head -80
for f in judged8.jsonl report8.txt; do put $f $f; done
date -u > DONE; put DONE DONE; echo "[vb] VBJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vbjob.sh >> /root/vb/vb.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VBJOB_LAUNCHED $(date -u)"
fi
echo "BOXVB_OK serial $BOXVB_SERIAL $(date -u)"
# CTL-END
