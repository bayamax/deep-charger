# box C: launch GRPO for the pooler lineage (teacher recipe + compression) from the distilled model.
#   trainer  : /root/work/grpo_pool.py  (see its docstring)   out: /root/grpo_pool/{grpo.log,rollouts.jsonl,latest.safetensors,state.json}
#   data     : the teacher's 2857-question pool (HF box_recover/corpus.jsonl) minus the held-out 300
# Safe to re-run: never relaunches while a trainer is alive; a dead trainer is NOT auto-restarted (OOM rule) - look first.
cd /root/work
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
if ! pgrep -f "/root/ctl\.s[h]" >/dev/null; then
  if [ -f /root/start_ctl.sh ]; then setsid nohup bash /root/start_ctl.sh > /dev/null 2>&1 < /dev/null & else setsid nohup bash /root/ctl.sh > /dev/null 2>&1 < /dev/null & fi
  sleep 2; echo "ctl revived: $(pgrep -fc '/root/ctl\.s[h]')"
fi
pkill -f "pool_eva[l].py" && echo "leftover evaluator killed"
python3 -c "import transformers.modeling_utils" 2>/dev/null || pip install -q "huggingface_hub>=0.34,<1.0" 2>&1 | tail -1
if [ ! -s /root/work/corpus_box_final.jsonl ]; then
  curl -sSL --retry 3 -o /root/work/corpus_box_final.jsonl "https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/box_recover/corpus.jsonl"
  echo "corpus downloaded: $(wc -l < /root/work/corpus_box_final.jsonl) rows"
fi
for i in 1 2 3 4 5 6; do curl -sS -o /root/work/grpo_pool.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/grpo_pool.py?nocache=$(date +%s)" && grep -q "def pg_backward" /root/work/grpo_pool.py && python3 -m py_compile /root/work/grpo_pool.py && break; sleep 5; done
echo "trainer fetched: $(wc -l < /root/work/grpo_pool.py) lines, v2=$(grep -c "def pg_backward" /root/work/grpo_pool.py)"
echo "gpu: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader) | disk: $(df -h /root | awk 'NR==2{print $4}') free"
# v1 -> v2: the gradient path now replays the recorded mass-evicted pooled sets; restart from step 0 (no checkpoint existed yet)
if [ ! -f /root/grpo_pool/.v2 ]; then
  pkill -f "grpo_poo[l].py"; sleep 5; [ -d /root/grpo_pool ] && mv /root/grpo_pool /root/grpo_pool_v1_$(date +%H%M); mkdir -p /root/grpo_pool; touch /root/grpo_pool/.v2
  echo "v1 trainer stopped, outdir rotated"
fi
mkdir -p /root/grpo_pool
if ! pgrep -f "grpo_poo[l].py" >/dev/null; then
  export SP_BASE=/root/fft_hf SP_RANK=128 SP_NOSYS=1 SP_EPISODIC=1 SP_HOTPOT2=0 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  echo "=== LAUNCH $(date -u) ===" >> /root/grpo_pool.log
  setsid nohup python3 /root/work/grpo_pool.py /root/fft_new_all.safetensors /root/grpo_pool --steps 200 --g 12 --rw 768 --maxd 384 >> /root/grpo_pool.log 2>&1 < /dev/null &
  echo "trainer launched"
else
  echo "trainer already running"
fi
cat > /usr/local/bin/t <<'TT'
#!/bin/bash
echo "$(date -u +%H:%M)Z  grpo $(pgrep -fc 'grpo_poo[l].py')  ctl $(pgrep -fc '/root/ctl\.s[h]')  gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
python3 - <<'PY' 2>/dev/null
import json,collections
by=collections.defaultdict(list)
for l in open("/root/grpo_pool/rollouts.jsonl"):
    try: r=json.loads(l); by[r["step"]].append(r)
    except Exception: pass
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
  { echo "=== $(date -u) ==="; t 2>/dev/null; echo "--- last log lines"; tail -3 /root/grpo_pool.log | cut -c1-200; } > /root/work/status.txt 2>&1
  echo "--- STATUS $(date -u +%H:%M) ---"; cat /root/work/status.txt
  hf upload baya1116/hypernet-sp-distill /root/work/status.txt pooler_distill/status.txt >/dev/null 2>&1
  sleep 300
done
SP
chmod +x /root/status_pub.sh
echo "--- UTIL $(date -u +%H:%M) ---"
for i in $(seq 1 12); do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; sleep 5; done | sort | uniq -c | sort -rn | head -6
echo "cache rows: $(wc -l < /root/work/pool_eval_cache.jsonl)  (file mtime $(date -u -r /root/work/pool_eval_cache.jsonl +%H:%M))"
python3 - <<'PY'
import json,collections
R=[json.loads(l) for l in open("/root/grpo_pool/rollouts.jsonl")]
by=collections.defaultdict(list)
for r in R: by[r["step"]].append(r)
S=sorted(by)[-8:]
tot=0; uniq=0
for s in S:
    qs=[q for r in by[s] for q in r["queries"][:5]]      # only the first 5 searches per roll are actually fetched
    tot+=len(qs); uniq+=len(set(qs))
print(f"last 8 steps: fetched searches {tot}, distinct queries {uniq} -> at most {uniq} wiki API round-trips (~{uniq*4*0.6/60:.1f} min of API sleep) vs step time ~13 min each")
PY
pkill -f "status_pub"; pkill -f "status_pu[b].sh"; setsid nohup bash /root/status_pub.sh >> /proc/1/fd/1 2>&1 < /dev/null &
echo "--- log tail"; tail -4 /root/grpo_pool.log | cut -c1-220
sleep 60; tail -3 /root/grpo_pool.log | cut -c1-200; echo "LAUNCH_DONE $(date -u)"
