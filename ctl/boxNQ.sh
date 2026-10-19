#!/bin/bash
# box NQ (2026-10-18, a 3060 - the user: the NPC model is small, a 3060 will do): box NP (A100) made the dialogues with
# Qwen3.6-35B-A3B and graded npc1-npc3; this box takes over from round 3's dialogues. npc4 is fine-tuned from npc1 on all
# of them (+ round 2's picks + the public set); the probes' replies of npc1-npc4 are graded again, all by the same judge,
# Qwen3.5-9B (AWQ, Apache-2.0, fits 12 GB), so they compare; then round 4, a reranker for the game (npc_bench.py rank).
# Pull-based as the others; the log goes to the hub every 10 minutes (charlm/npc4/boxlog_NQ.txt).
BOXNQ_SERIAL=2
if [ -f /root/.boxnq_serial ] && [ "$(cat /root/.boxnq_serial)" -gt "$BOXNQ_SERIAL" ] 2>/dev/null; then echo "BOXNQ_STALE $BOXNQ_SERIAL"; exit 0; fi
echo $BOXNQ_SERIAL > /root/.boxnq_serial
if [ ! -e /root/.nq_pre ]; then touch /root/.nq_pre   # the host first (Vast has many bad ones)
  echo "[nq] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader)"
  echo "[nq] preflight hub: $(curl -sSL -o /dev/null -m 90 -w '%{size_download} bytes in %{time_total}s' https://huggingface.co/QuantTrio/Qwen3.5-9B-AWQ/resolve/main/model-00001-of-00002.safetensors -r 0-104857599 2>&1)"
fi
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
# serial 2: the 9B judge ran out of memory in vLLM's warm-up (256 sequences' logits at once on 12 GB) and hung: the job
# stopped, the GPU cleared, the job again with 64 sequences in flight (what is done - npc4, its replies - is kept)
if [ ! -e /root/.nq_s2 ]; then touch /root/.nq_s2
  pkill -f "nqjo[b].sh"; pkill -f "npc_ge[n].py"; sleep 3; fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 5
  rm -f /root/.nqjob /root/nq/judged_9b.jsonl; echo "[nq] s2: judge restarted with 64 sequences, GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) used $(date -u)"
fi
for f in $(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv $f $f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
echo "[nq] HF token: $([ -s /root/.hf_token ] && echo present || echo MISSING)"
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
mkdir -p /root/nq
for f in charlm/npc_gen.py charlm/npc_bench.py charlm/charlm_sft.py charlm/charlm_fold.py q4.py; do
  for try in 1 2 3; do curl -sS -o /root/nq/$(basename $f) "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/nq/$(basename $f) && break; sleep 5; done
done
if [ ! -f /root/.bootstrapped ]; then
  python3 -m pip install -q pyarrow "huggingface_hub[hf_xet]" 2>&1 | tail -1
  python3 -c "import vllm, torch; print('[nq] vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.get_device_name(0))" 2>&1 | tail -1
  df -h /root | tail -1; free -g | head -2
  touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"
fi
cat > /root/status.sh <<'ST'
grep -vE "Warning|warn|it/s\]|^\s*$|INFO|WARNING" /root/nq/nq.log 2>/dev/null | tail -n 6 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh

if ! pgrep -f "nqmirro[r].sh" >/dev/null; then
  cat > /root/nqmirror.sh <<'MR'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
while true; do
  { echo "=== boxNQ $(date -u) ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; df -h /root | tail -1
    echo "--- ctl.log ---"; tail -n 15 /root/ctl.log 2>/dev/null | cut -c1-300
    echo "--- nq.log ---"; grep -vE "Warning|warn|it/s\]|INFO|WARNING" /root/nq/nq.log 2>/dev/null | tail -n 150 | cut -c1-900
    echo "--- npc4/train.log ---"; grep -vE "^\[step" /root/nq/npc4/train.log 2>/dev/null | tail -n 40 | cut -c1-900
    grep -E "^\[step" /root/nq/npc4/train.log 2>/dev/null | tail -n 30; } > /root/boxlog_NQ.txt
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='/root/boxlog_NQ.txt', path_in_repo='charlm/npc4/boxlog_NQ.txt', repo_id='baya1116/hypernet-sp-distill')" >/dev/null 2>&1
  sleep 600
done
MR
  setsid nohup bash /root/nqmirror.sh > /dev/null 2>&1 < /dev/null &
fi

# ---- the job: wait for round 3's dialogues on the hub, fetch, fine-tune npc4, replies, grading by 9B, round 4 ----
if [ ! -e /root/.nqjob ]; then touch /root/.nqjob
  cat > /root/nqjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/nq
HUB="https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/charlm"; J="--model QuantTrio/Qwen3.5-9B-AWQ --fallback QuantTrio/Qwen3.5-9B-AWQ --max-len 4096 --gpu-mem 0.88 --max-seqs 64"
gpu_clear() { fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3; }
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='charlm/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
get() { [ -s $2 ] || curl -sSfL -o $2 "$HUB/$1" || { rm -f $2; return 1; }; }
until curl -sSf -o /dev/null "$HUB/npc_data2/train_new.jsonl" 2>/dev/null; do sleep 300; done   # box NP's last upload
echo "[nq] round 3's dialogues are on the hub $(date -u)"
get npc1/model_final.pt npc1.pt; get npc_data/stage2_public.jsonl stage2_public.jsonl
for f in train_new.jsonl val_all.jsonl prompts.json npc_bench.jsonl rft.jsonl t3_0.jsonl t3_1.jsonl t3_2.jsonl t3_3.jsonl t3_4.jsonl t3_5.jsonl; do get npc_data2/$f $f; done
get npc2/judged_all.jsonl judged_all.jsonl; get npc3/judged_r3.jsonl judged_r3.jsonl
cat t3_*.jsonl > t3_all.jsonl; ls -la *.pt *.jsonl | sed 's/^/[nq] /'
if [ ! -s npc4/model_final.pt ]; then
  python3 -u charlm_sft.py --base npc1.pt --data sftdata --out npc4 --shards 0 --jsonl train_new.jsonl,t3_all.jsonl,rft.jsonl,stage2_public.jsonl \
    --val-jsonl val_all.jsonl --val-convs 1000 --prompts-json prompts.json --epochs 1.5 --lr 1e-4 --warmup 50 --eval-every 20 --hours 4 \
    --hub baya1116/hypernet-sp-distill:charlm/npc4 2>&1 | grep -vE "Warning|warn|^\[step"
fi
for m in best final 4bit; do
  [ -s npc4/model_$m.pt ] && [ ! -s r4_$m.jsonl ] && python3 -u npc_bench.py reply --model npc4/model_$m.pt --tag npc4-$m --bench npc_bench.jsonl --out r4_$m.jsonl
done
# every model's replies graded again by the one judge here
python3 -c "
import json
with open('replies_9b.jsonl', 'w') as f:
    for fn in ('judged_all.jsonl', 'judged_r3.jsonl', 'r4_best.jsonl', 'r4_final.jsonl', 'r4_4bit.jsonl'):
        try:
            for l in open(fn): r = json.loads(l); r.pop('judge', None); f.write(json.dumps(r) + '\n')
        except FileNotFoundError: pass"
[ -s judged_9b.jsonl ] || { gpu_clear; python3 -u npc_gen.py judge $J --inp replies_9b.jsonl --out judged_9b.jsonl; }
python3 npc_bench.py report --inp judged_9b.jsonl > report_9b.txt; sed 's/^/[report9b] /' report_9b.txt
put judged_9b.jsonl npc4/judged_9b.jsonl; put report_9b.txt npc4/report_9b.txt
# round 4: the reranker for the better npc4 file by this judge
M=$(python3 -c "
import json, collections; u = collections.defaultdict(list)
for l in open('judged_9b.jsonl'):
    r = json.loads(l)
    if r['pick'] == 'first' and r['model'] in ('npc4-best', 'npc4-final'): u[r['model']].append(bool(r['judge'].get('usable')))
print(max(u, key=lambda k: sum(u[k]) / len(u[k])) if u else 'npc4-final')")
F=npc4/model_${M#*-}.pt; echo "[nq] round 4: the ranker for $F $(date -u)"
[ -s s4.jsonl ] || python3 -u npc_bench.py sample --model $F --tag $M --bench t3_all.jsonl --situations 1500 --n 8 --seed 4 --out s4.jsonl
[ -s s4_judged.jsonl ] || { gpu_clear; python3 -u npc_gen.py judge $J --inp s4.jsonl --out s4_judged.jsonl; }
python3 -u npc_bench.py rank --model $F --inp s4_judged.jsonl --out rank_$M.pt; put rank_$M.pt npc4/rank_$M.pt
[ -s r5.jsonl ] || python3 -u npc_bench.py reply --model $F --tag $M-n8 --bench npc_bench.jsonl --n 8 --ranker rank_$M.pt --out r5.jsonl
[ -s judged_r5.jsonl ] || { gpu_clear; python3 -u npc_gen.py judge $J --inp r5.jsonl --out judged_r5.jsonl; }
python3 npc_bench.py report --inp judged_r5.jsonl > report4.txt; sed 's/^/[report4] /' report4.txt
put judged_r5.jsonl npc4/judged_r5.jsonl; put report4.txt npc4/report4.txt; put s4_judged.jsonl npc_data2/s4_judged.jsonl
echo "[nq] NQJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/nqjob.sh >> /root/nq/nq.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "NQJOB_LAUNCHED $(date -u)"
fi
echo "BOXNQ_OK serial $BOXNQ_SERIAL $(date -u)"
# CTL-END
