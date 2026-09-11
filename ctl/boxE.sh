# box E (24GB, replaces the A4000 whose host had no free GPU left): measure the held-out set through
# the 4-bit grid the phone actually runs.
#
# The app loads an MLX affine 4-bit conversion, group 64. Its weight index says what that touches:
# the seven projections of every block plus embed_tokens and lm_head, scales and biases in fp16,
# and nothing else - the norms, the attention biases and the pooler stay in float. q4.py reproduces
# that grid in torch and pool_eval.py --q4 1 applies it after loading.
#
# The comparison is against the 39.7% the same evaluator produced on the same 150 questions, whose
# per-rollout output the onstart restores as ev_out_*.jsonl, so the difference can be paired over
# questions instead of being read off two independent means.
cd /root/work
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
MODE=qat
SHARDS=3
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
for f in pool_eval.py q4.py qat.py build_merged.py web_search.py; do
  for try in 1 2 3; do curl -sS -o /root/work/$f "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/work/$f && break; sleep 5; done
done
cp /root/work/web_search.py /root/work/runtime/web_search.py 2>/dev/null
echo "fetched: pool_eval $(wc -l < /root/work/pool_eval.py) lines, q4 $(wc -l < /root/work/q4.py) lines"

if [ "$MODE" = "qat" ]; then
  # Quantization-aware training. The held-out measurement is the before-number and stays on disk;
  # what runs now moves the weights so that the same 4-bit conversion stops costing accuracy.
  pkill -f "evalkee[p].sh"; pkill -f "pool_eval.p[y]"; sleep 8
  pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 2
  for i in 0 1 2; do hf upload baya1116/hypernet-sp-distill /root/work/q4_out_$i.jsonl pooler_distill/grpo_pool3_step200/q4_shard$i.jsonl >/dev/null 2>&1; done
  echo "before-number frozen at $(cat /root/work/q4_out_*.jsonl 2>/dev/null | wc -l) rollouts"
  # the model's own traces: on-distribution calibration input, and the only data this needs
  if [ ! -s /root/work/rollouts.jsonl ]; then
    for try in 1 2 3; do hf download baya1116/hypernet-sp-distill --include "pooler_distill/grpo_pool3/rollouts.jsonl" --local-dir /root/hfdl 2>&1 | tail -1 && break; sleep 10; done
    cp /root/hfdl/pooler_distill/grpo_pool3/rollouts.jsonl /root/work/rollouts.jsonl
  fi
  echo "traces: $(wc -l < /root/work/rollouts.jsonl) rollouts"
  RUNNING=0
  pgrep -f "qat.p[y]" >/dev/null && { RUNNING=1; echo "QAT already running: $(tail -1 /root/qat.log)"; }
  [ -f /root/qat_hf/model.safetensors ] && { RUNNING=1; echo "QAT already finished -> /root/qat_hf"; }
  if [ "$RUNNING" = "0" ]; then
  # Two steps first: they print the step-0 loss, which IS the quantization damage while the adapter
  # is still zero, and they prove the memory fits before an hour is committed to it. A crash here is
  # not retried, it is looked at.
  if [ ! -f /root/.qat_selftest_ok ]; then
    rm -f /root/qat.log
    cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/qat.py \
      --base /root/eval_hf200 --data /root/work/rollouts.jsonl --selftest 2 2>&1 | tail -22
    grep -q "^step 2 " /root/qat.log 2>/dev/null || { echo "QAT SELFTEST FAILED - not launching"; exit 0; }
    touch /root/.qat_selftest_ok
  fi
  cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/qat.py \
    --base /root/eval_hf200 --data /root/work/rollouts.jsonl --out /root/qat_hf \
    --rank 32 --alpha 64 --lr 1e-4 --steps ${QSTEPS:-1500} --accum 4 --len 640 --kpos 256 \
    >> /root/qat_run.log 2>&1 < /dev/null &
  sleep 20
  fi
  cat > /usr/local/bin/t <<'TTQ'
#!/bin/bash
echo "$(date -u +%H:%M)Z qat $(pgrep -fc 'qat.p[y]')本  $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
grep -E "^\[q4\]|^\[lora\]|^\[data\]" /root/qat_run.log 2>/dev/null | head -5
python3 - <<'PYT' 2>/dev/null
import re
rows=[]
for l in open("/root/qat.log"):
    m=re.match(r"step (\d+) kl=([\d.]+) mse=([\d.]+)", l)
    if m: rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
if rows:
    rows.sort()
    def mean(xs): return sum(xs)/len(xs)
    a, b = rows[:20], rows[-20:]
    print(f"  step {rows[-1][0]}  kl {mean([r[1] for r in a]):.4f} -> {mean([r[1] for r in b]):.4f}"
          f"   hidden mse {mean([r[2] for r in a]):.4f} -> {mean([r[2] for r in b]):.4f}"
          f"   ({100*(1-mean([r[1] for r in b])/max(mean([r[1] for r in a]),1e-9)):.0f}% of the rounding damage removed)")
PYT
tail -2 /root/qat.log 2>/dev/null
python3 - <<'PYQ' 2>/dev/null
import json,glob,collections,math
def summary(pat):
    rows=[]
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            try: rows.append(json.loads(line))
            except Exception: pass
    if not rows: return None
    by=collections.defaultdict(list)
    for r in rows: by[r.get("q","")].append(bool(r.get("correct")))
    return len(rows), 100*sum(1 for r in rows if r.get("correct"))/len(rows), {k:sum(v)/len(v) for k,v in by.items()}
F=summary("/root/work/ev_out_*.jsonl"); Q=summary("/root/work/q4_out_*.jsonl"); N=summary("/root/work/qa_out_*.jsonl")
for tag,S in (("bf16",F),("4bit before",Q),("4bit after",N)):
    if S: print(f"  {tag:12s} {S[1]:5.1f}%  ({S[0]} roll)")
for tag,S in (("before",Q),("after",N)):
    if F and S:
        c=sorted(set(F[2])&set(S[2]))
        if not c: continue
        d=[S[2][k]-F[2][k] for k in c]; m=sum(d)/len(d)
        sd=(sum((x-m)**2 for x in d)/len(d))**0.5
        print(f"  paired {tag:6s} vs bf16 {100*m:+.1f} pt ±{100*sd/math.sqrt(len(d)):.1f} over {len(c)} questions")
PYQ
TTQ
  chmod +x /usr/local/bin/t
  cat > /root/status.sh <<'STQ'
#!/bin/bash
t 2>/dev/null
STQ
  chmod +x /root/status.sh
  pkill -f "status_pu[b]"; sleep 1
  cat > /root/status_pub.sh <<'SPQ'
#!/bin/bash
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
while true; do
  { echo "=== $(date -u) === box E (qat)"; t 2>/dev/null; } > /root/work/status.txt 2>&1
  echo "--- STATUS $(date -u +%H:%M) ---"; cat /root/work/status.txt
  sleep 300
done
SPQ
  # When the training finishes cleanly, the same held-out measurement runs against the trained
  # weights, through the same evaluator, the same 150 questions and the same 4-bit grid. This is
  # sequenced rather than retried: it starts only on QAT_DONE plus a saved model, and a shard that
  # dies mid-run is restarted because pool_eval resumes from its own output.
  cat > /root/afterkeep.sh <<'AKQ'
#!/bin/bash
until [ -s /root/qat_hf/model.safetensors ] && grep -q QAT_DONE /root/qat_run.log 2>/dev/null; do sleep 60; done
pkill -f "qat.p[y]"; sleep 10
while :; do
  for i in 0 1 2; do
    want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
    have=$(wc -l < /root/work/qa_out_$i.jsonl 2>/dev/null || echo 0)
    [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
    pgrep -f "pool_eval.py .* /root/work/ev_$i.jsonl" >/dev/null && continue
    echo "[after $(date -u +%H:%M)] shard $i at $have/$want - starting"
    grep -viE "^\s*$" /root/qa_$i.log 2>/dev/null | tail -4 | cut -c1-200
    cd /root/work && SP_BASE=/root/qat_hf SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py \
      /root/pooler200.safetensors /root/work/ev_$i.jsonl /root/work/qa_out_$i.jsonl \
      --n 999 --rw 768 --maxd 384 --samepage 1 --decode plain --q4 1 --tag "[a$i]" \
      >> /root/qa_$i.log 2>&1 < /dev/null &
    sleep 60
  done
  sleep 120
done
AKQ
  chmod +x /root/afterkeep.sh
  pkill -f "afterkee[p].sh"; sleep 1
  setsid nohup bash /root/afterkeep.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  chmod +x /root/status_pub.sh
  setsid nohup bash /root/status_pub.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  sleep 40; t; echo "QAT_LAUNCH_DONE $(date -u)"
  exit 0
fi

# The control loop starts before the assets finish arriving, so wait here rather than exiting: ctl
# only re-runs this file when its content changes, and a one-line "not ready" would be the last thing
# the box ever said.
for w in $(seq 1 40); do
  [ -s /root/eval_hf200/model.safetensors ] && [ -s /root/pooler200.safetensors ] && break
  echo "[wait $w] downloading: $(du -sh /root/hfdl 2>/dev/null | cut -f1) in /root/hfdl, $(tail -1 /root/boot.log 2>/dev/null | cut -c1-100)"
  sleep 60
done
if [ ! -s /root/eval_hf200/model.safetensors ]; then echo "NOT READY: /root/eval_hf200 never arrived"; exit 0; fi
if [ ! -s /root/pooler200.safetensors ]; then echo "NOT READY: pooler never arrived"; exit 0; fi
if [ ! -s /root/work/eval300.jsonl ]; then echo "NOT READY: questions never arrived"; exit 0; fi

# thirty seconds of checking the quantizer is cheaper than two hours of measuring with a broken one
python3 /root/work/q4.py || { echo "Q4 SELFTEST FAILED - not launching"; exit 0; }

# the same first 300 lines every earlier number on this file came from
if [ ! -f /root/work/ev_0.jsonl ]; then
  python3 - <<'PY'
import json
qs=[l for l in open("/root/work/eval300.jsonl") if l.strip()][:300]
S=3
for i in range(S): open(f"/root/work/ev_{i}.jsonl","w").writelines(qs[i::S])
print(f"[shard] first 300 lines, {len({json.loads(l).get('q','') for l in qs})} distinct questions -> {[len(qs[i::S]) for i in range(S)]}")
PY
fi
[ -f /root/.q4_t0 ] || date +%s > /root/.q4_t0

# One-shot: the first launch did the quantization arithmetic on the GPU, which left about 9.5GB
# reserved per process and let only two of the three shards onto the card - the third OOMed and the
# watchdog kept retrying it. The quantizer now works on the CPU in row blocks, so restart the shards
# once to pick that up. pool_eval resumes from its own output, so only the question in flight is lost.
if [ ! -f /root/.q4_cpuquant ]; then
  touch /root/.q4_cpuquant
  pkill -f "evalkee[p].sh"; pkill -f "pool_eval.p[y]"; sleep 10
  pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 3
  echo "shards stopped; they come back on the CPU-side quantizer ($(cat /root/work/q4_out_*.jsonl 2>/dev/null | wc -l) rollouts kept)"
fi

# ctl only re-runs this file when its content changes, so a shard that dies would otherwise sit dead
# until the next edit. This watchdog restarts one when its process is gone and its output is short;
# pool_eval resumes by skipping the questions already in its own output.
cat > /root/evalkeep.sh <<'KG'
#!/bin/bash
while :; do
  for i in 0 1 2; do
    want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
    have=$(wc -l < /root/work/q4_out_$i.jsonl 2>/dev/null || echo 0)
    [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
    pgrep -f "pool_eval.py .* /root/work/ev_$i.jsonl" >/dev/null && continue
    echo "[keep $(date -u +%H:%M)] shard $i is dead at $have/$want - restarting"
    grep -viE "^\s*$" /root/q4_$i.log 2>/dev/null | tail -4 | cut -c1-200
    cd /root/work && SP_BASE=/root/eval_hf200 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py \
      /root/pooler200.safetensors /root/work/ev_$i.jsonl /root/work/q4_out_$i.jsonl \
      --n 999 --rw 768 --maxd 384 --samepage 1 --decode plain --q4 1 --tag "[q$i]" \
      >> /root/q4_$i.log 2>&1 < /dev/null &
    sleep 60          # stagger the model loads; three landing together is what killed a shard last time
  done
  sleep 120
done
KG
chmod +x /root/evalkeep.sh
pkill -f "evalkee[p].sh"; sleep 1
setsid nohup bash /root/evalkeep.sh >> /proc/1/fd/1 2>&1 < /dev/null &

cat > /usr/local/bin/t <<'TT'
#!/bin/bash
python3 - <<'PY'
import json,glob,os,time,math,collections

def read(pat):
    rows=[]
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows

def summary(rows):
    n=len(rows)
    if not n: return None
    by=collections.defaultdict(list)
    for r in rows: by[r.get("q","")].append(bool(r.get("correct")))
    qm={k:sum(v)/len(v) for k,v in by.items()}
    mu=sum(qm.values())/len(qm)
    sd=(sum((x-mu)**2 for x in qm.values())/len(qm))**0.5
    def per_q(key):
        d=collections.defaultdict(list)
        for r in rows: d[r.get("q","")].append(float(r.get(key) or 0))
        return {k:sum(v)/len(v) for k,v in d.items()}
    return dict(n=n, q=len(qm), qm=qm, gq=per_q("grounded"), sq=per_q("ns"),
                c=100*sum(1 for r in rows if r.get("correct"))/n,
                g=100*sum(1 for r in rows if r.get("grounded"))/n,
                l=100*sum(1 for r in rows if r.get("landed"))/n,
                s=sum(r.get("ns",0) for r in rows)/n,
                se=100*sd/math.sqrt(len(qm)))

alive=os.popen("pgrep -fc 'pool_eval.p[y]'").read().strip() or "0"
qh=os.popen("grep '^\\[q4\\]' /root/q4_0.log 2>/dev/null | head -2").read().rstrip()
t0=0
try: t0=int(open("/root/.q4_t0").read().strip())
except Exception: pass
Q=summary(read("/root/work/q4_out_*.jsonl")); F=summary(read("/root/work/ev_out_*.jsonl"))
nq=Q["n"] if Q else 0
el=max(time.time()-t0,1) if t0 else 0
eta=f"  残り~{(300-nq)/(nq/el)/60:.0f}分" if nq and el and nq<300 else ""
print(f"{time.strftime('%H:%M',time.gmtime())}Z q4eval {alive}本  {nq}/300{eta}")
if qh: print("  "+qh.replace("\n","\n  "))
for tag,S in (("bf16",F),("4bit",Q)):
    if S: print(f"  {tag}  correct {S['c']:5.1f}% ±{S['se']:.1f}  gnd {S['g']:3.0f}%  land {S['l']:3.0f}%  srch {S['s']:.1f}  ({S['n']} roll / {S['q']} q)")
if Q and F:
    common=sorted(set(Q["qm"]) & set(F["qm"]))
    if common:
        def pair(key, scale, unit):
            d=[Q[key][k]-F[key][k] for k in common]
            m=sum(d)/len(d); sd=(sum((x-m)**2 for x in d)/len(d))**0.5
            return f"{scale*m:+.1f}{unit} ±{scale*sd/math.sqrt(len(d)):.1f}"
        print(f"  paired 4bit-bf16 over {len(common)} shared questions: "
              f"correct {pair('qm',100,'pt')}  gnd {pair('gq',100,'pt')}  srch {pair('sq',1,'')}")
PY
TT
chmod +x /usr/local/bin/t
cat > /root/status.sh <<'ST'
#!/bin/bash
t 2>/dev/null
ST
chmod +x /root/status.sh
pkill -f "status_pu[b]"; sleep 1
cat > /root/status_pub.sh <<'SP2'
#!/bin/bash
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
while true; do
  { echo "=== $(date -u) === box E (4-bit held-out)"; t 2>/dev/null; } > /root/work/status.txt 2>&1
  echo "--- STATUS $(date -u +%H:%M) ---"; cat /root/work/status.txt
  for i in 0 1 2; do hf upload baya1116/hypernet-sp-distill /root/work/q4_out_$i.jsonl pooler_distill/grpo_pool3_step200/q4_shard$i.jsonl >/dev/null 2>&1; done
  sleep 300
done
SP2
chmod +x /root/status_pub.sh
setsid nohup bash /root/status_pub.sh >> /proc/1/fd/1 2>&1 < /dev/null &
sleep 30; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader; t
echo "Q4EVAL_LAUNCH_DONE $(date -u)"
