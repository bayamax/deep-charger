# box D (A4000 16GB, cheaper successor of box C): resume the pooler-lineage GRPO from the checkpoint C uploaded to HF.
# The onstart already bootstrapped deps, fft_hf, fft_new_all.safetensors and /root/grpo_pool/{latest.safetensors,state.json,...}.
# Safe to re-run: never relaunches while a trainer is alive; a dead trainer is NOT auto-restarted (OOM rule) - look first.
# 03:40 UTC: OOM at step 44 on the 16GB card -> lm_head slice + gradient checkpointing in pg_backward; relaunch (resumes from step 40).
cd /root/work
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
python3 -c "import transformers.modeling_utils" 2>/dev/null || pip install -q "huggingface_hub>=0.34,<1.0" 2>&1 | tail -1
for i in 1 2 3 4 5 6; do curl -sS -o /root/work/grpo_pool.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/grpo_pool.py?nocache=$(date +%s)" && grep -q "def pg_backward" /root/work/grpo_pool.py && python3 -m py_compile /root/work/grpo_pool.py && break; sleep 5; done
for f in gold_pooled.py paired.py cnc.py strat.py analyze_pool.py pool_eval.py; do curl -sS -o /root/work/$f "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/$f?nocache=$(date +%s)"; done
echo "trainer fetched: $(wc -l < /root/work/grpo_pool.py) lines, v2=$(grep -c "def pg_backward" /root/work/grpo_pool.py)"
echo "gpu: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader) | disk: $(df -h /root | awk 'NR==2{print $4}') free"
echo "resume state: $(cut -c1-40 /root/grpo_pool/state.json 2>/dev/null)  latest: $(ls -la /root/grpo_pool/latest.safetensors 2>/dev/null | awk '{print $5}') bytes"
if [ ! -s /root/grpo_pool/latest.safetensors ] || [ ! -s /root/grpo_pool/state.json ] || [ ! -s /root/fft_new_all.safetensors ] || [ ! -f /root/fft_hf/model.safetensors ]; then
  echo "NOT READY: checkpoint or model missing, not launching"; exit 0
fi
# v4 (user request 08:00 UTC): loop guard OFF, phantom-sample advantage ON (--phantom 0.5 --phantom-scale 0.5). Resume from step 40.
if [ ! -f /root/grpo_pool/.v4 ]; then pkill -f "grpo_poo[l].py"; sleep 5; touch /root/grpo_pool/.v4; echo "v3 trainer stopped for the phantom run"; fi
if ! pgrep -f "grpo_poo[l].py" >/dev/null; then
  export SP_BASE=/root/fft_hf SP_RANK=128 SP_NOSYS=1 SP_EPISODIC=1 SP_HOTPOT2=0 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  echo "=== LAUNCH $(date -u) ===" >> /root/grpo_pool.log
  setsid nohup python3 /root/work/grpo_pool.py /root/fft_new_all.safetensors /root/grpo_pool --steps 200 --g 12 --rw 768 --maxd 384 --samepage 1 --gradckpt 1 --maxsrch 0 --phantom 0.5 --phantom-scale 0.5 >> /root/grpo_pool.log 2>&1 < /dev/null &
  echo "trainer launched (resume)"
else
  echo "trainer already running"
fi
cat > /usr/local/bin/t <<'TT'
#!/bin/bash
echo "$(date -u +%H:%M)Z  grpo $(pgrep -fc 'grpo_poo[l].py')  ctl $(pgrep -fc '/root/ctl\.s[h]')  gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
python3 - <<'PY' 2>/dev/null
import json,collections
by=collections.defaultdict(list); last=0
for l in open("/root/grpo_pool/rollouts.jsonl"):
    try: r=json.loads(l)
    except Exception: continue
    if r["step"]<=last and r["step"] in by and by[r["step"]] and by[r["step"]][-1] is not None and len(by[r["step"]])>=12:
        for k in [k for k in by if k>=r["step"]]: by.pop(k)      # a restart: drop the superseded rows
    by[r["step"]].append(r); last=max(last,r["step"]) if r["step"]>last else r["step"]
S=sorted(by); mid=len(S)//2
def acc(steps):
    rs=[r for s in steps for r in by[s]]
    return f"{100*sum(r['correct'] for r in rs)/max(len(rs),1):.0f}% srch {sum(len(r['queries']) for r in rs)/max(len(rs),1):.1f}"
print(f"steps {S[0]}-{S[-1]}  1st half {acc(S[:mid])}  |  2nd half {acc(S[mid:])}")
print(f"last25 {acc(S[-25:])}  gnd {100*sum(r['grounded'] for s in S[-25:] for r in by[s])/max(sum(len(by[s]) for s in S[-25:]),1):.0f}%   step {S[-1]}/200")
PY
grep "^\[step" /root/grpo_pool.log | tail -3 | sed 's/ landed=[0-9]*%//;s/ more=[0-9.]*//;s/ |grad|=[0-9.]*//;s/ skip=[01]//' | cut -c1-90
grep -i "error\|Traceback\|Killed\|GRPO_POOL_DONE" /root/grpo_pool.log | tail -2 | cut -c1-120
TT
chmod +x /usr/local/bin/t
cat > /root/status_pub.sh <<'SP'
#!/bin/bash
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
while true; do
  { echo "=== $(date -u) === box D"; t 2>/dev/null; echo "--- last log lines"; tail -3 /root/grpo_pool.log | cut -c1-200; } > /root/work/status.txt 2>&1
  echo "--- STATUS $(date -u +%H:%M) ---"; cat /root/work/status.txt
  hf upload baya1116/hypernet-sp-distill /root/work/status.txt pooler_distill/status.txt >/dev/null 2>&1
  # keep the latest checkpoint preserved off-box every time it changes (every 20 steps)
  if [ -f /root/grpo_pool/state.json ] && ! cmp -s /root/grpo_pool/state.json /root/.state_uploaded 2>/dev/null; then
    hf upload baya1116/hypernet-sp-distill /root/grpo_pool/latest.safetensors pooler_distill/grpo_pool/latest.safetensors >/dev/null 2>&1 \
    && hf upload baya1116/hypernet-sp-distill /root/grpo_pool/state.json pooler_distill/grpo_pool/state.json >/dev/null 2>&1 \
    && hf upload baya1116/hypernet-sp-distill /root/grpo_pool/rollouts.jsonl pooler_distill/grpo_pool/rollouts.jsonl >/dev/null 2>&1 \
    && cp /root/grpo_pool/state.json /root/.state_uploaded && echo "ckpt uploaded: $(cut -c1-30 /root/grpo_pool/state.json)"
  fi
  sleep 300
done
SP
chmod +x /root/status_pub.sh
pkill -f "status_pub"; setsid nohup bash /root/status_pub.sh >> /proc/1/fd/1 2>&1 < /dev/null &
sleep 60; tail -4 /root/grpo_pool.log | cut -c1-200; echo "LAUNCH_DONE $(date -u)"
