#!/bin/bash
# box NR (2026-10-19, a 3060 - the user: not polished NPC lines but natural talk in a role: small talk, simple directions
# and explanations; keep the character model): Qwen3.5-4B (AWQ, Apache-2.0) writes such conversations for 3.5 h
# (npc_gen.py help) and a held-out set of probes (helpbench); npc5 is fine-tuned from npc4 on them with as many of the
# earlier NPC dialogues and some public chat (placeholder names dropped); npc4 and npc5 answer the probes and Qwen3.5-9B
# grades them. Pull-based as the others; the log goes to the hub every 10 minutes (charlm/npc5/boxlog_NR.txt); the end
# is charlm/npc5/DONE on the hub, after which the box is let go.
BOXNR_SERIAL=6
if [ -f /root/.boxnr_serial ] && [ "$(cat /root/.boxnr_serial)" -gt "$BOXNR_SERIAL" ] 2>/dev/null; then echo "BOXNR_STALE $BOXNR_SERIAL"; exit 0; fi
echo $BOXNR_SERIAL > /root/.boxnr_serial
if [ ! -e /root/.nr_pre ]; then touch /root/.nr_pre   # the host first (Vast has many bad ones)
  echo "[nr] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader)"
  echo "[nr] preflight hub: $(curl -sSL -o /dev/null -m 90 -w '%{size_download} bytes in %{time_total}s' https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/charlm/npc4/model_final.pt -r 0-104857599 2>&1)"
fi
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
for f in $(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv $f $f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
# serial 2: the 4B wrote poor probes (answers in place of questions): the job stopped (part 0 kept), the probes made again
# by the 9B with questions required; the conversations keep their 3.5 h from the first start (until 11:07 UTC)
if [ ! -e /root/.nr_s2 ]; then touch /root/.nr_s2
  pkill -f "nrjo[b].sh"; pkill -f "npc_ge[n].py"; sleep 3; fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3
  rm -f /root/.nrjob /root/nr/helpbench.jsonl /root/nr/*.tmp; echo "[nr] s2: job stopped, probes to be made by the 9B $(date -u)"
fi
# serial 3: npc_gen.py put the fallback first on every pre-Blackwell GPU (meant for NVFP4 only), so the "9B" probes were the
# 4B's again: fixed; the job stopped and started again, the probes by the 9B
if [ ! -e /root/.nr_s3 ]; then touch /root/.nr_s3
  pkill -f "nrjo[b].sh"; pkill -f "npc_ge[n].py"; sleep 3; fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3
  rm -f /root/.nrjob /root/nr/helpbench.jsonl /root/nr/*.tmp; echo "[nr] s3: job stopped, the model order fixed $(date -u)"
fi
# serial 4: npc_gen.py loads the Qwen3.5 models text only (their vision encoder left out), so the 9B may fit for the
# grading; this run only fetches the scripts again - the running job takes it from its next call
# serial 5, the user: train on SODA (allenai/soda, CC BY 4.0, everyday social conversations) too. The job script stopped
# (its running part left to finish and be picked up), and started again with SODA in the mix
if [ ! -e /root/.nr_s5 ]; then touch /root/.nr_s5
  pkill -f "nrjo[b].sh"; rm -f /root/.nrjob; echo "[nr] s5: job script stopped (the running part goes on), SODA added $(date -u)"
fi
# serial 6, the user: for now SODA alone, at once - the generation stopped (parts 0-2 kept for later), npc5 from npc4 on
# SODA's 40,000 only; npc4 and npc5 graded on the help probes
if [ ! -e /root/.nr_s6 ]; then touch /root/.nr_s6
  pkill -f "nrjo[b].sh"; pkill -f "npc_ge[n].py"; sleep 3; fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3
  rm -f /root/.nrjob /root/nr/*.tmp /root/nr/*.tmp.raw; echo "[nr] s6: generation stopped; npc5 on SODA alone $(date -u)"
fi
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
echo "[nr] HF token: $([ -s /root/.hf_token ] && echo present || echo MISSING)"
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
mkdir -p /root/nr
for f in charlm/npc_gen.py charlm/npc_bench.py charlm/charlm_sft.py charlm/charlm_fold.py charlm/soda_prep.py q4.py; do
  for try in 1 2 3; do curl -sS -o /root/nr/$(basename $f) "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/nr/$(basename $f) && break; sleep 5; done
done
if [ ! -f /root/.bootstrapped ]; then
  python3 -m pip install -q pyarrow "huggingface_hub[hf_xet]" 2>&1 | tail -1
  python3 -c "import vllm, torch; print('[nr] vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.get_device_name(0))" 2>&1 | tail -1
  touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"
fi
cat > /root/status.sh <<'ST'
grep -E "^\[(nr|gen|bench|sft|val|judge|report|hub)\]|Error|Traceback" /root/nr/nr.log 2>/dev/null | tail -n 6 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh

if ! pgrep -f "nrmirro[r].sh" >/dev/null; then
  cat > /root/nrmirror.sh <<'MR'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
while true; do
  { echo "=== boxNR $(date -u) ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; df -h /root | tail -1
    echo "--- ctl.log ---"; tail -n 10 /root/ctl.log 2>/dev/null | cut -c1-300
    echo "--- nr.log ---"; grep -E "^\[(nr|gen|bench|sft|val|judge|report|hub|reply|oom|gptq)|Error|Traceback|out of memory" /root/nr/nr.log 2>/dev/null | tail -n 120 | cut -c1-900
    echo "--- npc5 steps ---"; grep -E "^\[step" /root/nr/npc5/train.log 2>/dev/null | tail -n 20; } > /root/boxlog_NR.txt
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='/root/boxlog_NR.txt', path_in_repo='charlm/npc5/boxlog_NR.txt', repo_id='baya1116/hypernet-sp-distill')" >/dev/null 2>&1
  sleep 600
done
MR
  setsid nohup bash /root/nrmirror.sh > /dev/null 2>&1 < /dev/null &
fi

if [ ! -e /root/.nrjob ]; then touch /root/.nrjob
  cat > /root/nrjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/nr
HUB="https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/charlm"
G="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 6144 --gpu-mem 0.88 --max-seqs 48 --chunk 400"
GE="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 6144 --gpu-mem 0.9 --max-seqs 16 --eager 1 --chunk 400"
J="--model QuantTrio/Qwen3.5-9B-AWQ --fallback QuantTrio/Qwen3.5-4B-AWQ --max-len 3072 --gpu-mem 0.93 --max-seqs 16 --eager 1 --chunk 1000"
gpu_clear() { fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3; }
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='charlm/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
get() { [ -s $2 ] || curl -sSfL -o $2 "$HUB/$1" || { rm -f $2; return 1; }; }
gen() {   # gen MODE N SEED OUT: the light settings again if the first try made nothing
  [ -s $4 ] && return 0; gpu_clear; python3 -u npc_gen.py $1 --n $2 --seed $3 --out $4.tmp $G
  [ -s $4.tmp ] || { echo "[nr] $4: again with the light settings"; gpu_clear; python3 -u npc_gen.py $1 --n $2 --seed $3 --out $4.tmp $GE; }
  [ -s $4.tmp ] && mv $4.tmp $4
}
while pgrep -f "npc_ge[n].py" > /dev/null; do sleep 30; done   # a part left running by the stopped job: let it finish, keep it
for t in help_*.jsonl.tmp; do [ -s "$t" ] && mv "$t" "${t%.tmp}" && put "${t%.tmp}" npc_data2/$(basename "${t%.tmp}"); done
get npc4/model_final.pt npc4.pt; get npc_data/stage2_public.jsonl stage2_public.jsonl
for f in train_new.jsonl t3_0.jsonl t3_1.jsonl t3_2.jsonl val_all.jsonl; do get npc_data2/$f $f; done
ls -la *.pt *.jsonl | sed 's/^/[nr] /'
echo "[nr] probes $(date -u)"; [ -s helpbench.jsonl ] || { gpu_clear; python3 -u npc_gen.py helpbench --n 160 --seed 1 --out helpbench.tmp $J; [ -s helpbench.tmp ] && mv helpbench.tmp helpbench.jsonl && put helpbench.jsonl npc_data2/helpbench.jsonl; }
DL=0; c=0   # serial 6: no more generation   # 3.5 h of conversations from the first start, 1,000 requests a part
while [ $(date +%s) -lt $DL ] && [ $c -lt 20 ]; do
  [ -s help_$c.jsonl ] || { echo "[nr] conversations part $c $(date -u)"; gen help 1000 $((c + 101)) help_$c.jsonl && put help_$c.jsonl npc_data2/help_$c.jsonl; }; c=$((c + 1))
done
cat help_*.jsonl > help_all.jsonl; N=$(wc -l < help_all.jsonl); echo "[nr] conversations: $N"
[ -s soda_0.parquet ] || curl -sSfL -o soda_0.parquet "https://huggingface.co/api/datasets/allenai/soda/parquet/default/train/0.parquet"
[ -s soda.jsonl ] || python3 -u soda_prep.py --parquet soda_0.parquet --n 40000 --val 300 --out soda.jsonl --val-out soda_val.jsonl
python3 - <<'PY'
import json, random
random.seed(5); bad = lambda r: any("[user" in m["content"] or "[name]" in m["content"] for m in r["messages"])
help_ = [l for l in open("help_all.jsonl")]; random.shuffle(help_)
old = [l for f in ("train_new.jsonl", "t3_0.jsonl", "t3_1.jsonl", "t3_2.jsonl") for l in open(f)]; random.shuffle(old)
pub = [l for l in open("stage2_public.jsonl") if not bad(json.loads(l))]; random.shuffle(pub)
val, help_ = help_[:300], help_[300:]; soda = [l for l in open("soda.jsonl")]
mix = soda; random.shuffle(mix)   # serial 6: SODA alone
open("mix5.jsonl", "w").writelines(mix); open("val5.jsonl", "w").writelines(val + [l for l in open("soda_val.jsonl")] + [l for l in open("val_all.jsonl")])
print(f"[nr] mix: SODA alone, {len(soda)} conversations; held out {len(val)} new + 300 SODA + the NPC set", flush=True)
json.dump([json.loads(l)["card"] + "\n\n" + json.loads(l)["probes"][0]["turns"][0] for l in list(open("helpbench.jsonl"))[:6]], open("prompts5.json", "w"))
PY
if [ ! -s npc5/model_final.pt ]; then
  gpu_clear; python3 -u charlm_sft.py --base npc4.pt --data sftdata --out npc5 --shards 0 --jsonl mix5.jsonl --val-jsonl val5.jsonl --val-convs 1200 \
    --prompts-json prompts5.json --epochs 1.2 --lr 5e-5 --warmup 50 --eval-every 15 --hours 3.5 --hub baya1116/hypernet-sp-distill:charlm/npc5 2>&1 | grep -vE "Warning|warn|^\[step"
fi
for m in npc4:npc4.pt npc5-best:npc5/model_best.pt npc5-final:npc5/model_final.pt npc5-4bit:npc5/model_4bit.pt; do
  t=${m%%:*}; f=${m#*:}; [ -s $f ] && [ ! -s r_$t.jsonl ] && python3 -u npc_bench.py reply --model $f --tag $t --bench helpbench.jsonl --out r_$t.jsonl
done
cat r_*.jsonl > replies5.jsonl
[ -s judged5.jsonl ] || { gpu_clear; python3 -u npc_gen.py judge --inp replies5.jsonl --out judged5.jsonl $J; }
python3 npc_bench.py report --inp judged5.jsonl > report5.txt; sed 's/^/[report] /' report5.txt
put judged5.jsonl npc5/judged5.jsonl; put report5.txt npc5/report5.txt; put help_all.jsonl npc_data2/help_all.jsonl
date -u > DONE; put DONE npc5/DONE; echo "[nr] NRJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/nrjob.sh >> /root/nr/nr.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "NRJOB_LAUNCHED $(date -u)"
fi
echo "BOXNR_OK serial $BOXNR_SERIAL $(date -u)"
# CTL-END
