# box D (A4000 16GB, cheaper successor of box C): resume the pooler-lineage GRPO from the checkpoint C uploaded to HF.
# The onstart already bootstrapped deps, fft_hf, fft_new_all.safetensors and /root/grpo_pool/{latest.safetensors,state.json,...}.
# Safe to re-run: never relaunches while a trainer is alive; a dead trainer is NOT auto-restarted (OOM rule) - look first.
# 03:40 UTC: OOM at step 44 on the 16GB card -> lm_head slice + gradient checkpointing in pg_backward; relaunch (resumes from step 40).
# 08:50 UTC Sep 10: the first version of this relied on a ctl content change to relaunch, so the run sat idle
#   for 15 min after finishing step 200 - keepgoing.sh below is the standing watchdog that fixes it.
# 06:00 UTC Sep 10 (user request): TARGET raised past 200 so the run continues instead of stopping. --steps is the TOTAL
#   target; the trainer resumes from grpo_pool3/{latest.safetensors,state.json}. A dead trainer is relaunched ONLY after a
#   clean GRPO_POOL_DONE with no crash in the tail - a crashed trainer is still left alone (OOM rule).
cd /root/work
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
python3 -c "import transformers.modeling_utils" 2>/dev/null || pip install -q "huggingface_hub>=0.34,<1.0" 2>&1 | tail -1
for i in 1 2 3 4 5 6; do curl -sS -o /root/work/grpo_pool.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/grpo_pool.py?nocache=$(date +%s)" && grep -q "def pg_backward" /root/work/grpo_pool.py && python3 -m py_compile /root/work/grpo_pool.py && break; sleep 5; done
for f in gold_pooled.py paired.py cnc.py strat.py analyze_pool.py pool_eval.py; do curl -sS -o /root/work/$f "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/$f?nocache=$(date +%s)"; done
echo "trainer fetched: $(wc -l < /root/work/grpo_pool.py) lines, v2=$(grep -c "def pg_backward" /root/work/grpo_pool.py)"
echo "gpu: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader) | disk: $(df -h /root | awk 'NR==2{print $4}') free"
echo "resume state: $(cut -c1-40 /root/grpo_pool3/state.json 2>/dev/null)  latest: $(ls -la /root/grpo_pool3/latest.safetensors 2>/dev/null | awk '{print $5}') bytes"
# Preserve the milestone checkpoints off-box (user request 15:40 UTC Sep 10). status_pub only ever uploads
# latest.safetensors, which is overwritten every 20 steps; ckpt_step100/200 are written by the trainer at the
# 100-step marks and would otherwise be lost with the instance. Each file is uploaded once, tracked by a marker.
for C in /root/grpo_pool3/ckpt_step*.safetensors; do
  [ -e "$C" ] || continue
  B=$(basename "$C"); M="/root/.uploaded_$B"
  [ -f "$M" ] && { echo "already preserved: $B"; continue; }
  echo "preserving $B ($(du -h "$C" | cut -f1)) -> HF"
  if hf upload baya1116/hypernet-sp-distill "$C" "pooler_distill/grpo_pool3/$B" >/dev/null 2>&1; then
    touch "$M"; echo "  PRESERVED $B"
  else
    echo "  UPLOAD FAILED $B"
  fi
done
if [ ! -s /root/fft_new_all.safetensors ] || [ ! -f /root/fft_hf/model.safetensors ]; then
  echo "NOT READY: model missing, not launching"; exit 0
fi
# v6 (user request 01:00 UTC Sep 9): match the teacher's LoRA (r16, layers 20-27) and train only a small
# adapter on the pooler. The SFT checkpoint is rank-128-on-all-layers in peft naming, so its weights must be
# merged into a plain HF model first -- otherwise the layers without LoRA silently fall back to the stock base.
for f in build_merged.py grpo_pool.py; do
  for i in 1 2 3 4 5 6; do curl -sS -o /root/work/$f "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/$f?nocache=$(date +%s)" && python3 -m py_compile /root/work/$f && break; sleep 5; done
done
grep -q "PoolerAdapter" /root/work/grpo_pool.py && echo "trainer v6 fetched" || { echo "FETCH FAILED"; exit 0; }
if [ ! -f /root/fft_hf2/model.safetensors ]; then
  pkill -f "grpo_poo[l].py"; sleep 8
  echo "--- MERGE $(date -u +%H:%M) ---"
  cd /root/work && SP_BASE=/root/fft_hf python3 /root/work/build_merged.py /root/fft_new_all.safetensors /root/fft_hf2 /root/pooler_sft.safetensors 2>&1 | grep -v Warning | tail -6
fi
if [ ! -f /root/fft_hf2/model.safetensors ]; then echo "NOT READY: merge failed"; exit 0; fi
if [ ! -f /root/.v6_checked ]; then
  echo "--- SELFTEST $(date -u +%H:%M) ---"
  python3 - <<'PY'
import torch, re
src=open("/root/work/grpo_pool.py").read()
cls=src[src.index("class PoolerAdapter"):src.index('if A.pooler == "none":')]
ns={"torch":torch}; exec(cls, ns); PA=ns["PoolerAdapter"]
base={"query":torch.randn(32,16),"blocks.0.cross.in_proj_weight":torch.randn(48,16),
      "blocks.0.lnq1.weight":torch.ones(16),"out_scale":torch.tensor(3.0)}
for mode in ("ln","lora"):
    a=PA(base,mode,4,2.0)
    ok = all(torch.allclose(a[k],base[k]) for k in base) and set(a.keys())==set(base.keys())
    print(f"  PoolerAdapter[{mode}]: identity-at-init {ok}, keys {len(a.keys())}, trainable {sum(p.numel() for p in a.trainable())}")
    assert ok
PY
  echo "--- PARITY $(date -u +%H:%M) (merged model, 6 training questions) ---"
  cd /root/work && SP_BASE=/root/fft_hf2 SP_RANK=128 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 \
    python3 /root/work/pool_eval.py /root/pooler_sft.safetensors /root/work/corpus_box_final.jsonl /root/work/parity.jsonl \
    --n 6 --rw 768 --decode plain --tag "[parity]" 2>&1 | grep -E "^\[load\]|^\[cfg\]|^\[[0-9]+/|EVAL_DONE" | tail -8
  python3 -c "
import json
R=[json.loads(l) for l in open('/root/work/parity.jsonl')]
print('  parity rows', len(R), 'landed', sum(r['landed'] for r in R), 'correct', sum(r['correct'] for r in R), 'mean srch', sum(r['ns'] for r in R)/max(len(R),1))
for r in R[:2]: print('   q:', r['q'][:50], '| ans:', (r['answer'] or '(none)')[:45], '| q1:', (r['queries'] or [''])[0][:35])"
  touch /root/.v6_checked
fi
if [ ! -f /root/grpo_pool3/.v6 ]; then pkill -f "grpo_poo[l].py"; sleep 5; mkdir -p /root/grpo_pool3; touch /root/grpo_pool3/.v6; echo "v5 stopped; fresh v6 run"; fi
[ -f /root/.grpo_batch ] || echo 1 > /root/.grpo_batch     # rollouts decoded in lockstep; switch.sh raises it once the equivalence test passes
[ -f /root/.grpo_g ] || echo 12 > /root/.grpo_g            # rollouts per question. Bigger groups mean fewer all-wrong groups, which produce no gradient at all
[ -f /root/.grpo_bp ] || echo 12 > /root/.grpo_bp          # rollouts the gradient replays. Kept at 12 as the group grows: the extra rollouts buy a better group, not a bigger update
[ -f /root/.grpo_target ] || echo 5000 > /root/.grpo_target  # switch.sh owns this from here on; a later run must not clobber it
TARGET=$(cat /root/.grpo_target)
AT=$(python3 -c "import json;print(json.load(open('/root/grpo_pool3/state.json'))['step'])" 2>/dev/null); AT=${AT:-0}
if pgrep -f "grpo_poo[l].py" >/dev/null; then
  echo "trainer already running (step $AT, target $TARGET) - not touching it"
elif [ "$AT" -ge "$TARGET" ]; then
  echo "target $TARGET already reached (step $AT) - raise TARGET to continue"
else
  CLEAN=$(grep -c "GRPO_POOL_DONE" /root/grpo_pool3.log 2>/dev/null); CLEAN=${CLEAN:-0}
  CRASH=$(tail -60 /root/grpo_pool3.log 2>/dev/null | grep -c -E "Traceback|CUDA out of memory|Killed"); CRASH=${CRASH:-0}
  if [ "$AT" -gt 0 ] && [ "$CLEAN" -eq 0 ]; then
    echo "trainer is DOWN at step $AT with no GRPO_POOL_DONE - NOT auto-relaunching, look at the log first"
  elif [ "$CRASH" -gt 0 ]; then
    echo "crash markers in the log tail - NOT auto-relaunching, look first"
  else
    export SP_NOSYS=1 SP_EPISODIC=1 SP_HOTPOT2=0 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo "=== LAUNCH $(date -u) ===" >> /root/grpo_pool3.log
    BS=$(cat /root/.grpo_batch 2>/dev/null); BS=${BS:-1}; GS=$(cat /root/.grpo_g 2>/dev/null); GS=${GS:-12}; BP=$(cat /root/.grpo_bp 2>/dev/null); BP=${BP:-12}
    setsid nohup python3 /root/work/grpo_pool.py /root/fft_hf2 /root/grpo_pool3 --steps $TARGET --g $GS --rw 768 --maxd 384 \
      --lora-rank 16 --lora-layers 20-27 --pooler lora --pooler-rank 8 --pooler-init /root/pooler_sft.safetensors \
      --samepage 1 --gradckpt 1 --maxsrch 0 --phantom 0 --batch $BS --backprop $BP >> /root/grpo_pool3.log 2>&1 < /dev/null &
    echo "trainer launched (v6, resuming from step $AT, target $TARGET)"
  fi
fi
# The control loop only runs this script when its content changes, so the relaunch above fires at most
# once per edit. keepgoing.sh is the standing watchdog: it resumes the trainer whenever the target has been
# raised past the saved step, and only after a CLEAN finish - a crash is still left alone (OOM rule).
cat > /root/keepgoing.sh <<'KG'
#!/bin/bash
while true; do
  sleep 60
  T=$(cat /root/.grpo_target 2>/dev/null); T=${T:-0}
  pgrep -f "grpo_poo[l].py" >/dev/null && continue
  AT=$(python3 -c "import json;print(json.load(open('/root/grpo_pool3/state.json'))['step'])" 2>/dev/null); AT=${AT:-0}
  [ "$AT" -ge "$T" ] && continue
  TL=$(tail -40 /root/grpo_pool3.log 2>/dev/null)
  if echo "$TL" | grep -q -E "Traceback|CUDA out of memory|Killed"; then
    echo "[keepgoing] crash markers in the log tail at step $AT - standing down for an hour"; sleep 3600; continue
  fi
  if ! echo "$TL" | grep -q "GRPO_POOL_DONE"; then
    echo "[keepgoing] trainer down at step $AT without GRPO_POOL_DONE - standing down for an hour"; sleep 3600; continue
  fi
  export SP_NOSYS=1 SP_EPISODIC=1 SP_HOTPOT2=0 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  echo "=== LAUNCH $(date -u) (keepgoing $AT -> $T) ===" >> /root/grpo_pool3.log
  BS=$(cat /root/.grpo_batch 2>/dev/null); BS=${BS:-1}; GS=$(cat /root/.grpo_g 2>/dev/null); GS=${GS:-12}; BP=$(cat /root/.grpo_bp 2>/dev/null); BP=${BP:-12}
  setsid nohup python3 /root/work/grpo_pool.py /root/fft_hf2 /root/grpo_pool3 --steps $T --g $GS --rw 768 --maxd 384 \
    --lora-rank 16 --lora-layers 20-27 --pooler lora --pooler-rank 8 --pooler-init /root/pooler_sft.safetensors \
    --samepage 1 --gradckpt 1 --maxsrch 0 --phantom 0 --batch $BS --backprop $BP >> /root/grpo_pool3.log 2>&1 < /dev/null &
  echo "[keepgoing] relaunched from step $AT toward $T"
  sleep 600                                  # cooldown: never spin on a trainer that dies on startup
done
KG
chmod +x /root/keepgoing.sh
pkill -f "keepgoin[g].sh"; setsid nohup bash /root/keepgoing.sh >> /proc/1/fd/1 2>&1 < /dev/null &
echo "keepgoing watchdog started (target $TARGET)"
# One-shot changeover to the batched rollout. The current trainer is left to stop itself at TARGET=260 (a save
# boundary, so nothing is lost); this waits for that clean stop, checks the batched rollout reproduces the single
# one token for token under a greedy policy, and only then raises the batch size and the target. A failed test
# leaves the unbatched path in place.
cat > /root/switch.sh <<'SW'
#!/bin/bash
ST() { python3 -c "import json;print(json.load(open('/root/grpo_pool3/state.json'))['step'])" 2>/dev/null; }
AT0=$(ST); AT0=${AT0:-0}
echo "[switch] waiting for the next checkpoint save (now at step $AT0) before the changeover"
for i in $(seq 1 200); do
  sleep 60
  pgrep -f "grpo_poo[l].py" >/dev/null || { echo "[switch] trainer already stopped at $(ST)"; break; }
  AT=$(ST); AT=${AT:-0}
  if [ "$AT" -gt "$AT0" ]; then
    echo "[switch] fresh save at step $AT - stopping the trainer for the changeover (nothing lost)"
    pkill -f "grpo_poo[l].py"; sleep 25; break
  fi
done
pkill -f "grpo_poo[l].py"; sleep 5
cd /root/work
OK=0
for N in 48 24 12; do
  echo "=== BATCH SELFTEST N=$N $(date -u) step $(ST) ==="
  SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 /root/work/grpo_pool.py /root/fft_hf2 /root/grpo_pool3 --g $N --rw 768 --maxd 384 \
    --lora-rank 16 --lora-layers 20-27 --pooler lora --pooler-rank 8 --pooler-init /root/pooler_sft.safetensors \
    --samepage 1 --maxsrch 0 --phantom 0 --selftest-batch $N > /root/selftest.txt 2>&1
  grep -viE "warning|warn\(" /root/selftest.txt | tail -14
  if grep -q "BATCH_SELFTEST PASS" /root/selftest.txt; then
    echo $N > /root/.grpo_batch; echo $N > /root/.grpo_g; echo 12 > /root/.grpo_bp; echo 5000 > /root/.grpo_target; OK=1
    echo "[switch] PASS at N=$N -> group $N, batch $N, gradient replays 12, no step target"; break
  fi
  if grep -qiE "outofmemory|CUDA out of memory" /root/selftest.txt; then
    echo "[switch] N=$N did not fit - trying a smaller group"; continue
  fi
  echo "[switch] N=$N MISMATCH (not memory) - the batched path is wrong, keeping the single one"; break
done
if [ "$OK" != "1" ]; then echo 1 > /root/.grpo_batch; echo 12 > /root/.grpo_g; echo 12 > /root/.grpo_bp; echo 5000 > /root/.grpo_target; fi
BS=$(cat /root/.grpo_batch); T=$(cat /root/.grpo_target); GS=$(cat /root/.grpo_g); BP=$(cat /root/.grpo_bp)
export SP_NOSYS=1 SP_EPISODIC=1 SP_HOTPOT2=0 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== LAUNCH $(date -u) (switch batch=$BS -> $T) ===" >> /root/grpo_pool3.log
setsid nohup python3 /root/work/grpo_pool.py /root/fft_hf2 /root/grpo_pool3 --steps $T --g $GS --rw 768 --maxd 384 \
  --lora-rank 16 --lora-layers 20-27 --pooler lora --pooler-rank 8 --pooler-init /root/pooler_sft.safetensors \
  --samepage 1 --gradckpt 1 --maxsrch 0 --phantom 0 --batch $BS --backprop $BP >> /root/grpo_pool3.log 2>&1 < /dev/null &
echo "[switch] relaunched from step $(ST): group $GS, batch $BS, gradient replays $BP, target $T"
SW
chmod +x /root/switch.sh
if [ ! -f /root/.switch_done2 ]; then
  touch /root/.switch_done2
  pkill -f "switc[h].sh"; setsid nohup bash /root/switch.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  echo "switch armed: waits for the next save, tests the batched rollout, then resumes with it"
else
  echo "switch already armed earlier (rm /root/.switch_done2 to re-arm)"
fi
cat > /usr/local/bin/t <<'TT'
#!/bin/bash
echo "$(date -u +%H:%M)Z  grpo $(pgrep -fc 'grpo_poo[l].py')  ctl $(pgrep -fc '/root/ctl\.s[h]')  gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
python3 - <<'PY' 2>/dev/null
import json,collections
by=collections.defaultdict(list); last=0
for l in open("/root/grpo_pool3/rollouts.jsonl"):
    try: r=json.loads(l)
    except Exception: continue
    if r["step"]<last:                                   # steps only grow within a run: a smaller step means a restart
        for k in [k for k in by if k>=r["step"]]: by.pop(k)   # drop the superseded rows
    by[r["step"]].append(r); last=r["step"]
S=sorted(by); mid=len(S)//2
def acc(steps):
    rs=[r for s in steps for r in by[s]]
    return f"{100*sum(r['correct'] for r in rs)/max(len(rs),1):.0f}% srch {sum(len(r['queries']) for r in rs)/max(len(rs),1):.1f}"
print(f"steps {S[0]}-{S[-1]}  1st half {acc(S[:mid])}  |  2nd half {acc(S[mid:])}")
TGT=open("/root/.grpo_target").read().strip() if __import__("os").path.exists("/root/.grpo_target") else "?"
print(f"last25 {acc(S[-25:])}  gnd {100*sum(r['grounded'] for s in S[-25:] for r in by[s])/max(sum(len(by[s]) for s in S[-25:]),1):.0f}%   step {S[-1]}/{TGT}")
PY
grep "^\[step" /root/grpo_pool3.log | tail -3 | sed 's/ landed=[0-9]*%//;s/ more=[0-9.]*//;s/ |grad|=[0-9.]*//;s/ skip=[01]//' | cut -c1-90
grep -i "error\|Traceback\|Killed\|GRPO_POOL_DONE" /root/grpo_pool3.log | tail -2 | cut -c1-120
TT
chmod +x /usr/local/bin/t
cat > /root/status_pub.sh <<'SP'
#!/bin/bash
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
while true; do
  { echo "=== $(date -u) === box D"; t 2>/dev/null; echo "--- last log lines"; tail -3 /root/grpo_pool3.log | cut -c1-200; } > /root/work/status.txt 2>&1
  echo "--- STATUS $(date -u +%H:%M) ---"; cat /root/work/status.txt
  hf upload baya1116/hypernet-sp-distill /root/work/status.txt pooler_distill/status.txt >/dev/null 2>&1
  # keep the latest checkpoint preserved off-box every time it changes (every 20 steps)
  if [ -f /root/grpo_pool3/state.json ] && ! cmp -s /root/grpo_pool3/state.json /root/.state_uploaded 2>/dev/null; then
    hf upload baya1116/hypernet-sp-distill /root/grpo_pool3/latest.safetensors pooler_distill/grpo_pool3/latest.safetensors >/dev/null 2>&1 \
    && hf upload baya1116/hypernet-sp-distill /root/grpo_pool3/state.json pooler_distill/grpo_pool3/state.json >/dev/null 2>&1 \
    && hf upload baya1116/hypernet-sp-distill /root/grpo_pool3/rollouts.jsonl pooler_distill/grpo_pool3/rollouts.jsonl >/dev/null 2>&1 \
    && cp /root/grpo_pool3/state.json /root/.state_uploaded && echo "ckpt uploaded: $(cut -c1-30 /root/grpo_pool3/state.json)"
  fi
  sleep 300
done
SP
chmod +x /root/status_pub.sh
pkill -f "status_pub"; setsid nohup bash /root/status_pub.sh >> /proc/1/fd/1 2>&1 < /dev/null &
sleep 60; tail -4 /root/grpo_pool3.log | cut -c1-200; echo "LAUNCH_DONE $(date -u)"
