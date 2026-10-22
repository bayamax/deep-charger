#!/bin/bash
# Box VM (3060, vLLM image), agent14 for github.com/bayamax/charlm-agent: the small talk has no right answer, so the
# judge chooses. Fresh situations (villagers written by the teacher with the bench's own probe kinds, the fact kinds
# left out, plus card chats from the mix), 6 replies each from agent13b, every reply graded by the genre-aware judge
# (answers, consistent, in character, fluent, charm, usable) and the rules; the top reply of each situation (usable,
# charm and in-character at 2, the rules passed; ties by likelihood) learned back into agent13b with the fact talks
# and the replay. Scored as agent13b was; the chat kinds of the bench (usable, charm) decide. To archive/charlm/agent14/
BOXVM_SERIAL=1
if [ -f /root/.boxvm_serial ] && [ "$(cat /root/.boxvm_serial)" -gt "$BOXVM_SERIAL" ] 2>/dev/null; then echo "BOXVM_STALE $BOXVM_SERIAL"; exit 0; fi
echo $BOXVM_SERIAL > /root/.boxvm_serial
if [ ! -e /root/.vm_pre ]; then touch /root/.vm_pre
  echo "[vm] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader)"
  echo "[vm] preflight hub: $(curl -sSL -o /dev/null -m 90 -w '%{size_download} bytes in %{time_total}s' https://huggingface.co/baya1116/charlm-agent/resolve/main/agent13b/model_final.pt -r 0-104857599 2>&1)"
fi
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
for f in $(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv $f $f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
RAW="https://raw.githubusercontent.com/bayamax/charlm-agent/main"
mkdir -p /root/vm
for f in npc_gen.py npc_bench.py charlm_sft.py charlm_fold.py charlm_run.py chat_bits.py goal_gen.py talk_rules.py q4.py eval/gold_shots.jsonl; do
  for try in 1 2 3; do curl -sS -o /root/vm/$(basename $f) "$RAW/$f?nocache=$(date +%s)" && break; sleep 5; done
done
[ -f /root/.bootstrapped ] || { python3 -m pip install -q pyarrow "huggingface_hub[hf_xet]" 2>&1 | tail -1; touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"; }
cat > /root/status.sh <<'ST'
grep -E "^\[(vm|gen|bench|judge|pick|sft|val|gptq|chat|goal|report|hub)\]|Error|Traceback|out of memory" /root/vm/vm.log 2>/dev/null | tail -n 6 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.vmjob ]; then touch /root/.vmjob
  cat > /root/vmjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vm
H=https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/archive/charlm
G="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 4096 --gpu-mem 0.88 --max-seqs 48"
J="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 3072 --gpu-mem 0.88 --max-seqs 48 --chunk 1000"
gpu_clear() { fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3; }
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/charlm/agent14/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
get() { [ -s $2 ] || curl -sSfL -o $2 "$1" || { rm -f $2; echo "[vm] could not get $1"; return 1; }; }
get https://huggingface.co/baya1116/charlm-agent/resolve/main/agent13b/model_final.pt agent13b.pt
get $H/agent13/b/villager_fact3_j.jsonl villager_fact3_j.jsonl; get $H/agent10/villager_j.jsonl villager_j.jsonl
for f in vbench.jsonl vbench_gen.jsonl; do get $H/villager/$f $f; done
for f in mix8.jsonl mix8_select.jsonl mix8_villager_val.jsonl; do get $H/agent8/data/$f $f; done
for f in soda_val.jsonl npc_val.jsonl; do get $H/agent6/data/$f $f; done
get $H/agent12/data/help_val.jsonl help_val.jsonl; get $H/agent12/judged12.jsonl judged12.jsonl; get $H/agent12/c/judged12c.jsonl judged12c.jsonl; get $H/agent13/b/judged13b.jsonl judged13b.jsonl
# (1) fresh villagers with the bench's probe kinds (new names and seed), the fact kinds left out; card chats from the mix
if [ ! -s vsit.jsonl ]; then
  gpu_clear; timeout 40m python3 -u npc_gen.py villagerbench --genres cozy,friends,dark --n 360 --seed 101 --out vsit.jsonl --chunk 60 $G 2>&1 | grep -vE "^INFO|^WARNING|^\(EngineCore|it/s\]"
fi
put vsit.jsonl vsit.jsonl
[ -s goal_r6.jsonl ] || python3 goal_gen.py data --n 4000 --levels 1,2,3,3,3,4,4 --split train --seed 74 --out goal_r6.jsonl
python3 - <<'PY'
import json, random, re
FACT = re.compile(r"recall|town|gift_like|gift_dislike|:who|:offer|dark:memory")
key = lambda r: (r["messages"][0]["content"], r["messages"][1]["content"])
val = {key(json.loads(l)) for l in open("mix8_villager_val.jsonl")}
R = random.Random(16); card = lambda l: json.loads(l)["messages"][0]["role"] == "system"
sits = []
for l in open("vsit.jsonl"):
    r = json.loads(l)
    for p in r["probes"]:
        if FACT.search(p["kind"]) or len(p["turns"]) != 1: continue
        sits.append({"messages": [{"role": "system", "content": r["card"]}, {"role": "user", "content": p["turns"][0]}, {"role": "assistant", "content": "-"}], "genre": r.get("genre") or "cozy", "kind": p["kind"]})
rest = [l for l in open("mix8.jsonl") if not str(json.loads(l).get("src", "")).startswith("villager")]
goal = [l for l in rest if json.loads(l)["messages"][0]["content"].startswith("GOAL:")]; gs = set(goal); chat = [l for l in rest if l not in gs]; R.shuffle(chat)
cchat = [l for l in chat if card(l)]
for l in cchat[1200:2000]:
    d = json.loads(l); d["genre"] = None; d["kind"] = "chat"; sits.append(d)
R.shuffle(sits); open("sits.jsonl", "w").writelines(json.dumps(s) + "\n" for s in sits)
vil = [l for l in open("villager_j.jsonl") if card(l) and key(json.loads(l)) not in val]; R.shuffle(vil)
open("replay.jsonl", "w").writelines(list(open("goal_r6.jsonl"))[:3000] + chat[2000:5000] + vil[:1500])
print(f"[vm] situations: {len(sits)} ({sum(1 for s in sits if s['kind'] != 'chat')} villager probes, {sum(1 for s in sits if s['kind'] == 'chat')} card chats); replay {3000 + len(chat[2000:5000]) + len(vil[:1500])}", flush=True)
PY
# (2) six replies each from agent13b, judged one by one, the rules beside; the top of each situation kept
gpu_clear
if [ ! -s samples.jsonl ]; then
  python3 - <<'PY'
rows = open("sits.jsonl").readlines()
for i in range(4): open(f"sit{i}.jsonl", "w").writelines(rows[i::4])
PY
  for i in 0 1 2 3; do python3 -u npc_bench.py sample --model agent13b.pt --tag agent13b --bench sit$i.jsonl --situations 100000 --n 6 --temp 0.9 --seed $((20 + i)) --out samp$i.jsonl > samp$i.log 2>&1 & done; wait
  python3 - <<'PY'
import json, random
with open("samples.jsonl", "w") as f:
    for i in range(4):
        dl = [json.loads(l) for l in open(f"sit{i}.jsonl")]; random.Random(20 + i).shuffle(dl)   # the sampler's own shuffle: sid -> the situation
        for l in open(f"samp{i}.jsonl"):
            r = json.loads(l); s = dl[r["sid"]]; r["sid"] = r["sid"] * 4 + i; r["genre"] = s.get("genre"); r["kind"] = s.get("kind")
            if r["reply"].strip(): f.write(json.dumps(r) + "\n")
PY
fi
put samples.jsonl samples.jsonl; echo "[vm] $(wc -l < samples.jsonl) replies to judge"
gpu_clear; [ -s samples_j.jsonl ] || python3 -u npc_gen.py judge --inp samples.jsonl --out samples_j.jsonl $J
put samples_j.jsonl samples_j.jsonl
python3 - <<'PY'
import json, collections, sys
sys.path.insert(0, "."); import talk_rules as TR
S = collections.defaultdict(list); n = 0
for l in open("samples_j.jsonl"):
    r = json.loads(l); j = r.get("judge") or {}
    if not j.get("usable"): continue
    rule = TR.check(r["reply"], r["turns"], r["card"])
    if rule: continue
    sc = sum(int(j.get(k) or 0) for k in ("answers", "consistent", "in_character", "fluent", "charm"))
    S[r["sid"]].append((int(j.get("charm") or 0) == 2 and int(j.get("in_character") or 0) == 2, sc, r["lp"], r)); n += 1
kept = collections.Counter(); out = open("rft_chat.jsonl", "w"); k = 0
for sid, rs in S.items():
    rs.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True); top = rs[0]
    if not top[0]: continue
    r = top[3]; msgs = [{"role": "system", "content": r["card"]}] + [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(r["turns"] + [r["reply"]])]
    out.write(json.dumps({"messages": msgs, "src": "rft_chat"}) + "\n"); kept[r.get("kind") or "chat"] += 1; k += 1
print(f"[pick] {n} usable+clean replies in {len(S)} situations; {k} tops with charm 2 and in-character 2 -> rft_chat.jsonl; by kind {dict(kept.most_common(12))}", flush=True)
PY
put rft_chat.jsonl rft_chat.jsonl
python3 - <<'PY'
import json, random
R = random.Random(17); mix = list(open("rft_chat.jsonl")) * 2 + list(open("villager_fact3_j.jsonl")) + list(open("replay.jsonl")); R.shuffle(mix)
open("mix14.jsonl", "w").writelines(mix); print(f"[vm] mix14: {sum(1 for _ in open('rft_chat.jsonl'))} chat tops (x2) + fact talks + replay = {len(mix)}", flush=True)
PY
python3 -c "import json; json.dump([json.loads(l)['messages'][0]['content'] + '\n\n' + json.loads(l)['messages'][1]['content'] for l in list(open('mix8_villager_val.jsonl'))[:3]], open('prompts14.json', 'w'))"
# (3) agent14 from agent13b
gpu_clear; mkdir -p agent14
( while sleep 300; do put agent14/train.log train.log > /dev/null; done ) & MIR=$!
[ -s agent14/model_4bit.pt ] || python3 -u charlm_sft.py --base agent13b.pt --data sftdata --out agent14 --shards 0 --jsonl mix14.jsonl --val-jsonl mix8_select.jsonl --val-convs 450 --min-learned 2 \
    --prompts-json prompts14.json --epochs 1 --lr 2e-5 --warmup 10 --eval-every 4 --hours 1 --ckpt-every 10 --hub baya1116/hypernet-sp-distill:archive/charlm/agent14 2>&1 | grep -vE "Warning|detach|backward|^\[step"
kill $MIR 2>/dev/null; put agent14/train.log train.log
[ -s agent14/model_4bit.pt ] || { echo "[vm] no 4-bit model"; date -u > DONE; put DONE DONE; exit 1; }
# (4) scoring: the bench's chat kinds decide, the rest must not fall
python3 -u chat_bits.py --models agent13b.pt,agent14/model_final.pt,agent14/model_4bit.pt --sets soda_val.jsonl,npc_val.jsonl,help_val.jsonl,mix8_villager_val.jsonl --n 300 2>&1 | grep "^\[chat\]" | tee chat14.txt
put chat14.txt chat14.txt
python3 -u goal_gen.py play --model agent14/model_4bit.pt --n 100 --split test --seed 5 2>&1 | grep -E "^\[goal\] level" | tee play14.txt; put play14.txt play14.txt
cat vbench.jsonl vbench_gen.jsonl > vbench_all.jsonl
python3 - <<'PY'
rows = open("vbench_all.jsonl").readlines()
for i in range(3): open(f"vb_part{i}.jsonl", "w").writelines(rows[i::3])
PY
for i in 0 1 2; do python3 -u npc_bench.py reply --model agent14/model_4bit.pt --tag agent14-4bit --bench vb_part$i.jsonl --out r14_$i.jsonl --n 4 > r14_$i.log 2>&1 & done; wait
cat r14_0.jsonl r14_1.jsonl r14_2.jsonl > r_agent14-4bit.jsonl; put r_agent14-4bit.jsonl r_agent14-4bit.jsonl
gpu_clear; python3 -u npc_gen.py judge --inp r_agent14-4bit.jsonl --out judged14.jsonl $J
cat judged12.jsonl judged12c.jsonl judged13b.jsonl judged14.jsonl > judged_all14.jsonl
python3 npc_bench.py report --inp judged_all14.jsonl > report14.txt; python3 talk_rules.py score --inp judged_all14.jsonl > rules14.txt
python3 - <<'PY'
import json, re, collections
FACT = re.compile(r"recall|town|gift_like|gift_dislike|:who|:offer|dark:memory")
M = collections.defaultdict(lambda: [0, 0, 0.0])
for l in open("judged_all14.jsonl"):
    r = json.loads(l)
    if r["pick"] != "best" or FACT.search(r["kind"]): continue
    j = r["judge"]; m = M[r["model"]]; m[0] += 1; m[1] += bool(j.get("usable")); m[2] += float(j.get("charm") or 0)
for k, (n, u, c) in M.items(): print(f"[report] chat kinds {k}: {n} probes, usable {100 * u / n:.1f}%, charm {c / n:.2f}", flush=True)
PY
sed 's/^/[report] /' report14.txt | grep -E "best +(ALL|G:|advice|feeling|memory |dream|gossip|worry|tease |town |dark:recall)"; cat rules14.txt
for f in judged14.jsonl report14.txt rules14.txt; do put $f $f; done
put /root/vm/vm.log run.log; date -u > DONE; put DONE DONE; echo "[vm] VMJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vmjob.sh >> /root/vm/vm.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VMJOB_LAUNCHED $(date -u)"
fi
# CTL-END
