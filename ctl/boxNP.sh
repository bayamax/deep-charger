#!/bin/bash
# box NP (2026-10-18, an RTX 5090 - the user: train the NPC model until a game can use it, within the $5 left on Vast): an open
# model, Qwen3.6-35B-A3B (Apache-2.0, so its outputs may train a commercial model; NVFP4 under vLLM), writes character cards
# with concrete facts and dialogues whose player lines each have a set purpose (charlm/npc_gen.py), and a held-out set of
# probes; the character model (npc1) is fine-tuned on them with the earlier public NPC data (npc2), and both models' replies
# to the probes are graded by the same open model (charlm/npc_bench.py). Pull-based as the others; the log goes to the hub
# every 10 minutes (charlm/npc2/boxlog_NP.txt) and to the container log every 5 (status.sh).
BOXNP_SERIAL=2
if [ -f /root/.boxnp_serial ] && [ "$(cat /root/.boxnp_serial)" -gt "$BOXNP_SERIAL" ] 2>/dev/null; then echo "BOXNP_STALE $BOXNP_SERIAL"; exit 0; fi
echo $BOXNP_SERIAL > /root/.boxnp_serial
# serial 2: Vast's own vLLM image (theirs builds; the vllm-openai one did not): its Python is /venv/main, and its own server
# must not hold the GPU
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
echo "[np] HF token: $([ -s /root/.hf_token ] && echo present || echo MISSING)"
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
mkdir -p /root/np
for f in charlm/npc_gen.py charlm/npc_bench.py charlm/charlm_sft.py charlm/charlm_fold.py q4.py; do
  for try in 1 2 3; do curl -sS -o /root/np/$(basename $f) "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/np/$(basename $f) && break; sleep 5; done
done

if [ ! -f /root/.bootstrapped ]; then
  mkdir /root/.bootstrap_lock 2>/dev/null || { echo "bootstrap already running"; exit 0; }
  python3 -m pip install -q pyarrow "huggingface_hub[hf_xet]" 2>&1 | tail -1 || uv pip install --system -q pyarrow "huggingface_hub[hf_xet]" 2>&1 | tail -1
  python3 -c "import vllm, torch; print('[np] vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.get_device_name(0))"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; df -h /root | tail -1; free -g | head -2; nproc
  touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"
fi

cat > /root/status.sh <<'ST'
grep -vE "Warning|warn|it/s\]|^\s*$" /root/np/np.log 2>/dev/null | tail -n 6 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST

if ! pgrep -f "npmirro[r].sh" >/dev/null; then
  cat > /root/npmirror.sh <<'MR'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
while true; do
  { echo "=== boxNP $(date -u) ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; df -h /root | tail -1
    echo "--- ctl.log ---"; tail -n 15 /root/ctl.log 2>/dev/null | cut -c1-300
    echo "--- np.log ---"; grep -vE "Warning|warn|it/s\]" /root/np/np.log 2>/dev/null | tail -n 150 | cut -c1-900
    echo "--- npc2/train.log ---"; grep -vE "^\[step" /root/np/npc2/train.log 2>/dev/null | tail -n 40 | cut -c1-900
    grep -E "^\[step" /root/np/npc2/train.log 2>/dev/null | tail -n 30; } > /root/boxlog_NP.txt
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='/root/boxlog_NP.txt', path_in_repo='charlm/npc2/boxlog_NP.txt', repo_id='baya1116/hypernet-sp-distill')" >/dev/null 2>&1
  sleep 600
done
MR
  setsid nohup bash /root/npmirror.sh > /dev/null 2>&1 < /dev/null &
fi

# ---- the job: probes, dialogues (4 parts, each kept as made), npc1's replies, fine-tuning, npc2's replies, grading ----
if [ ! -e /root/.npjob ]; then touch /root/.npjob
  cat > /root/npjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/np
HUB="https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/charlm"
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='charlm/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
for f in npc1/model_final.pt npc_data/stage2_public.jsonl npc_data/stage2_val.jsonl; do
  [ -s $(basename $f) ] || curl -sSfL -o $(basename $f) "$HUB/$f"; done
mv -n model_final.pt npc1.pt 2>/dev/null; ls -la *.pt *.jsonl | sed 's/^/[np] /'
echo "[np] probes $(date -u)"
[ -s npc_bench.jsonl ] || { python3 -u npc_gen.py bench --n 150 --out npc_bench.tmp && mv npc_bench.tmp npc_bench.jsonl && put npc_bench.jsonl npc_data2/npc_bench.jsonl; }
for c in 0 1 2 3; do
  echo "[np] dialogues part $c $(date -u)"
  [ -s train_$c.jsonl ] || { python3 -u npc_gen.py train --n 2500 --seed $((c + 1)) --out train_$c.tmp && mv train_$c.tmp train_$c.jsonl && put train_$c.jsonl npc_data2/train_$c.jsonl; }
done
cat train_0.jsonl train_1.jsonl train_2.jsonl > train_new.jsonl; head -n -400 train_3.jsonl >> train_new.jsonl
tail -n 400 train_3.jsonl > val_new.jsonl; cat val_new.jsonl stage2_val.jsonl > val_all.jsonl
echo "[np] new dialogues: $(wc -l < train_new.jsonl) to train, $(wc -l < val_new.jsonl) held out"
python3 -c "
import json; r = [json.loads(l) for l in open('npc_bench.jsonl')][:6]
json.dump([x['card'] + '\n\n' + x['probes'][0]['turns'][0] for x in r], open('prompts.json', 'w'))"
[ -s replies_npc1.jsonl ] || python3 -u npc_bench.py reply --model npc1.pt --tag npc1 --bench npc_bench.jsonl --out replies_npc1.jsonl
if [ ! -s npc2/model_final.pt ]; then
  python3 -u charlm_sft.py --base npc1.pt --data sftdata --out npc2 --shards 0 --jsonl train_new.jsonl,stage2_public.jsonl --val-jsonl val_all.jsonl \
    --val-convs 1000 --prompts-json prompts.json --epochs 2 --lr 1e-4 --warmup 50 --eval-every 10 --hours 3 --stop-plateau 0.002 --stop-evals 3 \
    --hub baya1116/hypernet-sp-distill:charlm/npc2 2>&1 | grep -vE "Warning|warn|^\[step"
fi
for m in best final 4bit; do
  [ -s npc2/model_$m.pt ] && [ ! -s replies_npc2_$m.jsonl ] && python3 -u npc_bench.py reply --model npc2/model_$m.pt --tag npc2-$m --bench npc_bench.jsonl --out replies_npc2_$m.jsonl
done
cat replies_*.jsonl > replies_all.jsonl
[ -s judged_all.jsonl ] || python3 -u npc_gen.py judge --inp replies_all.jsonl --out judged_all.jsonl
python3 npc_bench.py report --inp judged_all.jsonl > report.txt; sed 's/^/[report] /' report.txt
put judged_all.jsonl npc2/judged_all.jsonl; put report.txt npc2/report.txt
echo "[np] NPJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/npjob.sh >> /root/np/np.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "NPJOB_LAUNCHED $(date -u)"
fi
echo "BOXNP_OK serial $BOXNP_SERIAL $(date -u)"
# CTL-END
