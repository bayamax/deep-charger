#!/bin/bash
# Box VH (3060, vLLM image): the DPO step that fell over on box VF, again: agent12 = agent12a tuned on agent11's own
# best-vs-worst pairs (charlm_dpo.py, micro-batched now, its log kept), then the 4-bit file and the same scoring as
# agent12a (chat bits, goal play, the villager bench judged, the rules). To the hub: archive/charlm/agent12/b/
BOXVH_SERIAL=3
if [ -f /root/.boxvh_serial ] && [ "$(cat /root/.boxvh_serial)" -gt "$BOXVH_SERIAL" ] 2>/dev/null; then echo "BOXVH_STALE $BOXVH_SERIAL"; exit 0; fi
echo $BOXVH_SERIAL > /root/.boxvh_serial
if [ ! -e /root/.vh_pre ]; then touch /root/.vh_pre
  echo "[vh] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader)"
  echo "[vh] preflight hub: $(curl -sSL -o /dev/null -m 90 -w '%{size_download} bytes in %{time_total}s' https://huggingface.co/baya1116/charlm-agent/resolve/main/agent6/model_final.pt -r 0-104857599 2>&1)"
fi
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
for f in $(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv $f $f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
RAW="https://raw.githubusercontent.com/bayamax/charlm-agent/main"
mkdir -p /root/vh
for f in npc_gen.py npc_bench.py charlm_sft.py charlm_fold.py charlm_run.py chat_bits.py goal_gen.py talk_rules.py charlm_dpo.py q4.py; do
  for try in 1 2 3; do curl -sS -o /root/vh/$f "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/vh/$f && break; sleep 5; done
done
[ -f /root/.bootstrapped ] || { python3 -m pip install -q pyarrow "huggingface_hub[hf_xet]" 2>&1 | tail -1; touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"; }
cat > /root/status.sh <<'ST'
grep -E "^\[(vh|dpo|oom|rft|linejudge|sample|sft|val|gptq|chat|goal|judge|bench|hub)\]|Error|Traceback|out of memory" /root/vh/vh.log 2>/dev/null | tail -n 6 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
# serial 3: the DPO rounds did not help (b: drifted, worse on the bench; c: gentle, no step beat the base), the
# rejection-sampling round did (agent11 -> agent12a). Job 3, after job 2: round 2 of that from agent12a's own replies on
# fresh situations (villager_j slice 2000-4000 + other card chats 600-1200, left out of round 1), the few-shot line judge
# + the rules pick the best of 6, learned from agent12a with the replay beside; 4-bit; the same scoring -> agent12/d/
if [ ! -e /root/.vh_s3 ]; then touch /root/.vh_s3
  cat > /root/vhjob3.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vh
H=https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/archive/charlm
J="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 3072 --gpu-mem 0.88 --max-seqs 48 --chunk 1000"
JL="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 6144 --gpu-mem 0.88 --max-seqs 48 --chunk 2000"
gpu_clear() { fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3; }
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/charlm/agent12/d/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
get() { [ -s $2 ] || curl -sSfL -o $2 "$1" || { rm -f $2; echo "[vh] could not get $1"; return 1; }; }
while [ ! -s DONE_C ]; do sleep 60; done
echo "[vh] job 3 (RFT round 2 from agent12a) starts $(date -u)"
RAW="https://raw.githubusercontent.com/bayamax/charlm-agent/main"
for f in rft_prep.py talk_rules.py eval/gold_shots.jsonl; do curl -sSfL -o $(basename $f) "$RAW/$f?nocache=$(date +%s)"; done
get $H/agent10/villager_j.jsonl villager_j.jsonl
[ -s goal_r4.jsonl ] || python3 goal_gen.py data --n 4000 --levels 1,2,3,3,3,4,4 --split train --seed 72 --out goal_r4.jsonl
python3 - <<'PY'
import json, random
key = lambda r: (r["messages"][0]["content"], r["messages"][1]["content"])
val = {key(json.loads(l)) for l in open("mix8_villager_val.jsonl")}
R = random.Random(12)   # the same shuffle as round 1 (box VF), so the slices after its are fresh
card = lambda l: json.loads(l)["messages"][0]["role"] == "system"
vil = [l for l in open("villager_j.jsonl") if card(l) and key(json.loads(l)) not in val]; R.shuffle(vil)
rest = [l for l in open("mix8.jsonl") if not str(json.loads(l).get("src", "")).startswith("villager")]
goal = [l for l in rest if json.loads(l)["messages"][0]["content"].startswith("GOAL:")]; gs = set(goal); chat = [l for l in rest if l not in gs]
R.shuffle(chat); cchat = [l for l in chat if card(l)]
sit = vil[2000:4000] + cchat[600:1200]; R2 = random.Random(13); R2.shuffle(sit)
for i in range(4): open(f"sit{i}.jsonl", "w").writelines(sit[i::4])
open("replay2.jsonl", "w").writelines(list(open("goal_r4.jsonl"))[:3000] + chat[1200:4200] + vil[:1500])
print(f"[vh] round 2: {len(sit)} situations ({len(vil[2000:4000])} villager, {len(cchat[600:1200])} other chat); replay {3000 + len(chat[1200:4200]) + len(vil[:1500])}", flush=True)
PY
gpu_clear
if [ ! -s samples2.jsonl ]; then
  for i in 0 1 2 3; do python3 -u npc_bench.py sample --model agent12a.pt --tag agent12a --bench sit$i.jsonl --situations 100000 --n 6 --temp 0.8 --seed $((10 + i)) --out samp$i.jsonl > samp$i.log 2>&1 & done; wait
  python3 - <<'PY'
import json
with open("samples2.jsonl", "w") as f:
    for i in range(4):
        for l in open(f"samp{i}.jsonl"):
            r = json.loads(l); r["sid"] = r["sid"] * 4 + i; f.write(json.dumps(r) + "\n")
PY
fi
put samples2.jsonl samples2.jsonl
python3 rft_prep.py tojudge --samples samples2.jsonl --out lj2_in.jsonl
gpu_clear; [ -s lj2_out.jsonl ] || python3 -u npc_gen.py linejudge --variant fewshot --shots gold_shots.jsonl --inp lj2_in.jsonl --out lj2_out.jsonl $JL
put lj2_out.jsonl lj2_out.jsonl
python3 rft_prep.py pick --samples samples2.jsonl --grades lj2_out.jsonl --rft rft2.jsonl --pairs pairs2.jsonl
for f in rft2.jsonl pairs2.jsonl; do put $f $f; done
cat rft2.jsonl replay2.jsonl > mix12d.jsonl
[ -s prompts12.json ] || python3 -c "import json; json.dump([json.loads(l)['messages'][0]['content'] + '\n\n' + json.loads(l)['messages'][1]['content'] for l in list(open('mix8_villager_val.jsonl'))[:3]], open('prompts12.json', 'w'))"
gpu_clear; mkdir -p agent12d
( while sleep 300; do put agent12d/train.log train.log > /dev/null; done ) & MIR=$!
[ -s agent12d/model_4bit.pt ] || python3 -u charlm_sft.py --base agent12a.pt --data sftdata --out agent12d --shards 0 --jsonl mix12d.jsonl --val-jsonl mix8_select.jsonl --val-convs 450 --min-learned 2 \
    --prompts-json prompts12.json --epochs 1 --lr 2e-5 --warmup 10 --eval-every 4 --hours 1 --ckpt-every 10 --hub baya1116/hypernet-sp-distill:archive/charlm/agent12/d 2>&1 | grep -vE "Warning|detach|backward|^\[step"
kill $MIR 2>/dev/null; put agent12d/train.log train.log
[ -s agent12d/model_4bit.pt ] || { echo "[vh] round 2 produced no 4-bit model"; date -u > DONE_D; put DONE_D DONE; exit 1; }
python3 -u chat_bits.py --models agent12a.pt,agent12d/model_final.pt,agent12d/model_4bit.pt --sets soda_val.jsonl,npc_val.jsonl,help_val.jsonl,mix8_villager_val.jsonl --n 300 2>&1 | grep "^\[chat\]" | tee chat12d.txt
put chat12d.txt chat12d.txt
python3 -u goal_gen.py play --model agent12d/model_4bit.pt --n 100 --split test --seed 5 2>&1 | grep -E "^\[goal\] level" | tee play12d.txt; put play12d.txt play12d.txt
for i in 0 1 2; do python3 -u npc_bench.py reply --model agent12d/model_4bit.pt --tag agent12d-4bit --bench vb_part$i.jsonl --out r12d_$i.jsonl --n 4 > r12d_$i.log 2>&1 & done; wait
cat r12d_0.jsonl r12d_1.jsonl r12d_2.jsonl > r_agent12d-4bit.jsonl; put r_agent12d-4bit.jsonl r_agent12d-4bit.jsonl
gpu_clear; python3 -u npc_gen.py judge --inp r_agent12d-4bit.jsonl --out judged12d.jsonl $J
cat judged12.jsonl judged12c.jsonl judged12d.jsonl > judged_alld.jsonl
python3 npc_bench.py report --inp judged_alld.jsonl > report12d.txt; python3 talk_rules.py score --inp judged_alld.jsonl > rules12d.txt
sed 's/^/[report] /' report12d.txt | grep -E "best +(ALL|G:)"; cat rules12d.txt
for f in judged12d.jsonl report12d.txt rules12d.txt; do put $f $f; done
put /root/vh/vh.log run.log; date -u > DONE_D; put DONE_D DONE; echo "[vh] VHJOB3_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vhjob3.sh >> /root/vh/vh.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[vh] s3: job 3 (RFT round 2 from agent12a) queued after job 2 $(date -u)"
fi
# serial 2: the first DPO drifted (chosen above rejected 92%, but held-out chat bits 0.82 -> 0.99): job 2, after job 1,
# runs it gently (beta 0.3, lr 2e-6, one epoch, more plain learning) with the held-out tracked every 40 steps and the best
# checkpoint kept, then the same 4-bit and scoring into agent12/c/
if [ ! -e /root/.vh_s2 ]; then touch /root/.vh_s2
  cat > /root/vhjob2.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vh
J="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 3072 --gpu-mem 0.88 --max-seqs 48 --chunk 1000"
gpu_clear() { fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3; }
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/charlm/agent12/c/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
while [ ! -s DONE ]; do sleep 60; done
curl -sS -o charlm_dpo.py "https://raw.githubusercontent.com/bayamax/charlm-agent/main/charlm_dpo.py?nocache=$(date +%s)"; python3 -m py_compile charlm_dpo.py
gpu_clear; mkdir -p agent12c
python3 -u charlm_dpo.py --base agent12a.pt --pairs pairs.jsonl --anchor anchor.jsonl --val mix8_select.jsonl --val-n 300 --val-every 40 --out agent12c \
  --beta 0.3 --lr 2e-6 --epochs 1 --batch 8 --micro 2 --anchor-batch 4 --sft-w 0.5 --anchor-w 1.0 --max-ctx 1400 2>&1 | grep -vE "Warning|detach"
put agent12c/train.log train.log
[ -s agent12c/model_best.pt ] || { echo "[vh] no step beat the base on the held-out: agent12c = agent12a"; cp agent12a.pt agent12c/model_best.pt; }
put agent12c/model_best.pt model_best.pt
python3 -u charlm_sft.py --base agent12c/model_best.pt --data sftdata --out agent12c --shards 0 --jsonl anchor.jsonl --val-jsonl mix8_select.jsonl --val-convs 450 --min-learned 2 \
    --epochs 0 --prompts-json prompts12.json 2>&1 | grep -E "^\[(val|gptq|sft)\]"
put agent12c/model_4bit.pt model_4bit.pt
python3 -u chat_bits.py --models agent12a.pt,agent12c/model_best.pt,agent12c/model_4bit.pt --sets soda_val.jsonl,npc_val.jsonl,help_val.jsonl,mix8_villager_val.jsonl --n 300 2>&1 | grep "^\[chat\]" | tee chat12c.txt
put chat12c.txt chat12c.txt
python3 -u goal_gen.py play --model agent12c/model_4bit.pt --n 100 --split test --seed 5 2>&1 | grep -E "^\[goal\] level" | tee play12c.txt; put play12c.txt play12c.txt
for i in 0 1 2; do python3 -u npc_bench.py reply --model agent12c/model_4bit.pt --tag agent12c-4bit --bench vb_part$i.jsonl --out r12c_$i.jsonl --n 4 > r12c_$i.log 2>&1 & done; wait
cat r12c_0.jsonl r12c_1.jsonl r12c_2.jsonl > r_agent12c-4bit.jsonl; put r_agent12c-4bit.jsonl r_agent12c-4bit.jsonl
gpu_clear; python3 -u npc_gen.py judge --inp r_agent12c-4bit.jsonl --out judged12c.jsonl $J
cat judged12.jsonl judged12c.jsonl > judged_allc.jsonl
python3 npc_bench.py report --inp judged_allc.jsonl > report12c.txt; python3 talk_rules.py score --inp judged_allc.jsonl > rules12c.txt
sed 's/^/[report] /' report12c.txt | grep -E "best +(ALL|G:)"; cat rules12c.txt
for f in judged12c.jsonl report12c.txt rules12c.txt; do put $f $f; done
put /root/vh/vh.log run.log; date -u > DONE_C; put DONE_C DONE; echo "[vh] VHJOB2_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vhjob2.sh >> /root/vh/vh.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[vh] s2: job 2 (the gentle DPO with the held-out tracked) queued after job 1 $(date -u)"
fi
if [ ! -e /root/.vhjob ]; then touch /root/.vhjob
  cat > /root/vhjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/vh
H=https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/archive/charlm
J="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 3072 --gpu-mem 0.88 --max-seqs 48 --chunk 1000"
gpu_clear() { fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3; }
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/charlm/agent12/b/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
get() { [ -s $2 ] || curl -sSfL -o $2 "$1" || { rm -f $2; echo "[vh] could not get $1"; return 1; }; }
get $H/agent12/a/model_final.pt agent12a.pt; get $H/agent12/data/pairs.jsonl pairs.jsonl
for f in mix8.jsonl mix8_select.jsonl mix8_villager_val.jsonl; do get $H/agent8/data/$f $f; done
for f in soda_val.jsonl npc_val.jsonl; do get $H/agent6/data/$f $f; done
get $H/agent12/data/help_val.jsonl help_val.jsonl; get $H/agent12/judged12.jsonl judged12.jsonl
for f in vbench.jsonl vbench_gen.jsonl; do get $H/villager/$f $f; done
[ -s goal_r.jsonl ] || python3 goal_gen.py data --n 1200 --levels 1,2,3,3,3,4,4 --split train --seed 71 --out goal_r.jsonl
python3 - <<'PY'
import json, random
R = random.Random(12); rest = [l for l in open("mix8.jsonl") if not str(json.loads(l).get("src", "")).startswith("villager")]
chat = [l for l in rest if not json.loads(l)["messages"][0]["content"].startswith("GOAL:")]; R.shuffle(chat)
open("anchor.jsonl", "w").writelines(list(open("goal_r.jsonl")) + chat[:1000]); print(f"[vh] anchors: 1200 goal episodes + 1000 chats", flush=True)
PY
( while sleep 300; do put agent12/train.log train.log > /dev/null; done ) & MIR=$!
[ -s agent12/model_final.pt ] || python3 -u charlm_dpo.py --base agent12a.pt --pairs pairs.jsonl --anchor anchor.jsonl --out agent12 --beta 0.1 --lr 5e-6 --epochs 2 --batch 8 --micro 2 --anchor-batch 4 --max-ctx 1400 2>&1 | grep -vE "Warning|detach"
kill $MIR; put agent12/train.log train.log
[ -s agent12/model_final.pt ] || { echo "[vh] DPO produced no model"; date -u > DONE; put DONE DONE; exit 1; }
put agent12/model_final.pt model_final.pt
python3 -c "import json; json.dump([json.loads(l)['messages'][0]['content'] + '\n\n' + json.loads(l)['messages'][1]['content'] for l in list(open('mix8_villager_val.jsonl'))[:3]], open('prompts12.json', 'w'))"
[ -s agent12/model_4bit.pt ] || python3 -u charlm_sft.py --base agent12/model_final.pt --data sftdata --out agent12 --shards 0 --jsonl anchor.jsonl --val-jsonl mix8_select.jsonl --val-convs 450 --min-learned 2 \
    --epochs 0 --prompts-json prompts12.json 2>&1 | grep -E "^\[(val|gptq|sft)\]"
put agent12/model_4bit.pt model_4bit.pt
python3 -u chat_bits.py --models agent12a.pt,agent12/model_final.pt,agent12/model_4bit.pt --sets soda_val.jsonl,npc_val.jsonl,help_val.jsonl,mix8_villager_val.jsonl --n 300 2>&1 | grep "^\[chat\]" | tee chat12b.txt
put chat12b.txt chat12b.txt
python3 -u goal_gen.py play --model agent12/model_4bit.pt --n 100 --split test --seed 5 2>&1 | grep -E "^\[goal\] level" | tee play12b.txt; put play12b.txt play12b.txt
cat vbench.jsonl vbench_gen.jsonl > vbench_all.jsonl
python3 - <<'PY'
rows = open("vbench_all.jsonl").readlines()
for i in range(3): open(f"vb_part{i}.jsonl", "w").writelines(rows[i::3])
PY
for i in 0 1 2; do python3 -u npc_bench.py reply --model agent12/model_4bit.pt --tag agent12-4bit --bench vb_part$i.jsonl --out r12_$i.jsonl --n 4 > r12_$i.log 2>&1 & done; wait
cat r12_0.jsonl r12_1.jsonl r12_2.jsonl > r_agent12-4bit.jsonl; put r_agent12-4bit.jsonl r_agent12-4bit.jsonl
gpu_clear; python3 -u npc_gen.py judge --inp r_agent12-4bit.jsonl --out judged12b.jsonl $J
cat judged12.jsonl judged12b.jsonl > judged_all.jsonl
python3 npc_bench.py report --inp judged_all.jsonl > report12b.txt; python3 talk_rules.py score --inp judged_all.jsonl > rules12b.txt
sed 's/^/[report] /' report12b.txt | grep -E "best +(ALL|G:)"; cat rules12b.txt
for f in judged12b.jsonl report12b.txt rules12b.txt; do put $f $f; done
put /root/vh/vh.log run.log; date -u > DONE; put DONE DONE; echo "[vh] VHJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vhjob.sh >> /root/vh/vh.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VHJOB_LAUNCHED $(date -u)"
fi
echo "BOXVH_OK serial $BOXVH_SERIAL $(date -u)"
# CTL-END
