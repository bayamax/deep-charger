#!/bin/bash
# box G2 (2026-10-08 00:40 JST): the GRPO box moved off host 155385 - after the credit lapse its GPU stayed with another
# renter and the start sat in Vast's queue (the user: that is no good, move house). Only what the routine needs is here:
# the 4-bit base, s100, the evaluation sets, the data, the loop; then mtg7 and the rounds after it. Pull-based as the
# others (ctl.sh runs this file whenever it changes).
BOXG2_SERIAL=2
if [ -f /root/.boxg2_serial ] && [ "$(cat /root/.boxg2_serial)" -gt "$BOXG2_SERIAL" ] 2>/dev/null; then echo "BOXG2_STALE $BOXG2_SERIAL"; exit 0; fi
echo $BOXG2_SERIAL > /root/.boxg2_serial
mkdir -p /root/work/runtime /root/work/fft_out /root/hfdl; cd /root/work
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
[ -s /root/.dsk ] || { [ -n "$DSK_KEY" ] && printf '%s' "$DSK_KEY" > /root/.dsk && chmod 600 /root/.dsk; }
[ -s /root/.oai ] || { [ -n "$OAI_KEY" ] && printf '%s' "$OAI_KEY" > /root/.oai && chmod 600 /root/.oai; }
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"

# ---- the code, fresh on every control run ----
for f in online_loop.py pool_eval.py r1_traj.py tqa_items.py musique_items.py nq_items.py web_search.py dl_judge.py gptq.py q4.py packmlx.py checkmlx.py dequant_state.py build_merged.py memfit.py mt_sim.py; do
  for try in 1 2 3; do curl -sS -o /root/work/$f "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/work/$f && break; sleep 5; done
done
cp /root/work/web_search.py /root/work/runtime/web_search.py 2>/dev/null; touch /root/work/runtime/__init__.py

# ---- one-time bootstrap: libraries, the harness, the data ----
if [ ! -f /root/.bootstrapped ]; then
  mkdir /root/.bootstrap_lock 2>/dev/null || { echo "bootstrap already running"; exit 0; }
  echo "=== bootstrap $(date -u) ==="
  pip install -q "transformers==4.44.2" "peft==0.12.0" "safetensors==0.8.0" "huggingface_hub>=0.34,<1.0" accelerate certifi datasets pyarrow scipy 2>&1 | tail -1
  dl() { for try in 1 2 3 4 5 6; do hf download $R --include "$1" --local-dir /root/hfdl >/dev/null 2>&1 && return 0; sleep 20; done; echo "dl failed: $1"; return 1; }
  dl "box_recover/scripts/*"; dl "fft_out/pooler.pt"; dl "grpo_assets/mus_run/eval_step200/eval_heldout_300q_g2.jsonl"; dl "pooler_distill/pool_eval_cache.jsonl"
  dl "pooler_distill/chatsft/multiturn/mtg3_s100.safetensors"; dl "pooler_distill/chatsft/multiturn/mt_eval_bridge3.jsonl"; dl "pooler_distill/chatsft/multiturn/mtg2_items.jsonl"
  dl "pooler_distill/chatsft/dolphin_v1.jsonl"; dl "pooler_distill/chatsft/dolphin_v2.jsonl"; dl "pooler_distill/chatsft/replay_v1.jsonl"
  dl "pooler_distill/chatsft/teach/probe_s100.jsonl"; dl "pooler_distill/chatsft/teach/r1_traj.jsonl"; dl "pooler_distill/selfq_*.jsonl"
  dl "pooler_distill/chatsft/rollouts/gq14_*.jsonl"; dl "pooler_distill/chatsft/g14m_hf/*"
  SC=/root/hfdl/box_recover/scripts; cp $SC/*.py $SC/*.sh /root/work/ 2>/dev/null
  ln -sfn /root/hfdl/fft_out/pooler.pt /root/work/fft_out/pooler.pt
  cp /root/hfdl/grpo_assets/mus_run/eval_step200/eval_heldout_300q_g2.jsonl /root/work/eval300.jsonl
  cp /root/hfdl/pooler_distill/pool_eval_cache.jsonl /root/work/pool_eval_cache.jsonl
  cp /root/hfdl/pooler_distill/chatsft/multiturn/mtg3_s100.safetensors /root/mtg3_s100.safetensors
  cp /root/hfdl/pooler_distill/chatsft/multiturn/mt_eval_bridge3.jsonl /root/hfdl/pooler_distill/chatsft/multiturn/mtg2_items.jsonl /root/work/
  cp /root/hfdl/pooler_distill/chatsft/dolphin_v1.jsonl /root/hfdl/pooler_distill/chatsft/replay_v1.jsonl /root/work/
  cp /root/hfdl/pooler_distill/chatsft/teach/probe_s100.jsonl /root/work/probe_items.jsonl; cp /root/hfdl/pooler_distill/chatsft/teach/r1_traj.jsonl /root/work/r1_traj.jsonl
  cat /root/hfdl/pooler_distill/selfq_*.jsonl > /root/work/selfq_all.jsonl
  cat /root/hfdl/pooler_distill/chatsft/rollouts/gq14_*.jsonl > /root/work/qcal_g2.jsonl     # the lineage's own traces, as gq14's calibration was
  python3 - <<'PYB'
import json, random
qs = [l for l in open("/root/work/eval300.jsonl") if l.strip()][:300]
for i in range(3): open(f"/root/work/ev_{i}.jsonl", "w").writelines(qs[i::3])
# dolphin_v1 minus the fixed held-out hundred (seed 0, dolphin_v2's questions excluded from the draw) - the rule box G used
v1 = [json.loads(l) for l in open("/root/hfdl/pooler_distill/chatsft/dolphin_v1.jsonl") if l.strip()]
v1 = [r for r in v1 if r.get("q") and r.get("reply")]
v2q = set((json.loads(l).get("q") or "").strip() for l in open("/root/hfdl/pooler_distill/chatsft/dolphin_v2.jsonl") if l.strip())
cand = [r for r in v1 if r["q"].strip() not in v2q]; random.Random(0).shuffle(cand)
held = cand[:100]; hq = set(r["q"].strip() for r in held)
open("/root/work/dolphin_heldout100.jsonl", "w").write("".join(json.dumps({"q": r["q"], "ref": r["reply"]}, ensure_ascii=False) + "\n" for r in held))
open("/root/work/dolphinq.jsonl", "w").write("".join(json.dumps({"q": r["q"]}, ensure_ascii=False) + "\n" for r in held))
n = 0
with open("/root/work/dolphin_rft.jsonl", "w") as o:
    for r in v1:
        if r["q"].strip() in hq: continue
        o.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
print(f"[bootstrap] eval shards 3 x {len(qs)//3}; dolphin_rft {n} problems, 100 held out; selfq_all {sum(1 for _ in open('/root/work/selfq_all.jsonl'))}")
PYB
  touch /root/.bootstrapped; rmdir /root/.bootstrap_lock
  echo "=== bootstrap done $(date -u) ==="; df -h /root | tail -1; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# ---- the 4-bit base: box G's own (gq14) if it reaches the hub within the hour, else quantised here the same way ----
if [ ! -s /root/gptq_hf_gq14/model.safetensors ] && ! pgrep -f "basekee[p].sh" >/dev/null && ! grep -q "BASE_READY" /root/base.log 2>/dev/null; then
  cat > /root/basekeep.sh <<'BK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
while [ ! -f /root/.bootstrapped ]; do sleep 30; done
N=12; grep -q "BASE_FAILED" /root/base.log 2>/dev/null && N=1   # the hour's wait for box G's copy was already spent
for i in $(seq 1 $N); do
  if hf download $R --include "pooler_distill/chatsft/gq14_hf/*" --local-dir /root/hfdl >/dev/null 2>&1 && [ -s /root/hfdl/pooler_distill/chatsft/gq14_hf/model.safetensors ]; then
    rm -rf /root/gptq_hf_gq14; cp -r /root/hfdl/pooler_distill/chatsft/gq14_hf /root/gptq_hf_gq14; echo "[base] gq14 from the hub (box G's own) $(date -u +%H:%M)"; break; fi
  [ "$N" -gt 1 ] && sleep 300
done
if [ ! -s /root/gptq_hf_gq14/model.safetensors ]; then
  echo "[base] quantising g14 here from its own traces $(date -u +%H:%M)"
  cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/gptq.py --base /root/hfdl/pooler_distill/chatsft/g14m_hf --data /root/work/qcal_g2.jsonl \
    --out-hf /root/gptq_hf_gq14 --out-mlx /root/gptq_mlx4_gq14 --state /root/gptq_state_g2.pt > /root/gptq_g2.log 2>&1
  grep -E "^\[gptq\] 128|^\[out\]|GPTQ_DONE|Error|Traceback" /root/gptq_g2.log | tail -4
  [ -s /root/gptq_hf_gq14/model.safetensors ] && hf upload $R /root/gptq_hf_gq14 pooler_distill/chatsft/gq14_hf >/dev/null 2>&1 && echo "[base] uploaded as gq14_hf (rebuilt)"
fi
[ -s /root/gptq_hf_gq14/model.safetensors ] && echo "BASE_READY $(date -u)" || echo "BASE_FAILED $(date -u)"
BK
  setsid nohup bash -c 'bash /root/basekeep.sh 2>&1 | tee -a /root/base.log' >> /proc/1/fd/1 2>&1 < /dev/null &
fi

# ---- the mirror: what this box is doing, on the hub every ten minutes ----
if ! pgrep -f "mirrorkee[p].sh" >/dev/null; then
  cat > /root/mirrorkeep.sh <<'MK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
while :; do
  { echo "=== boxG2 $(date -u) ==="; echo "--- ctl.log ---"; tail -n 40 /root/ctl.log 2>/dev/null | cut -c1-300
    for f in /root/base.log /root/mtg7.log; do [ -e $f ] && { echo "--- $f ---"; tail -n 20 $f | cut -c1-300; }; done
    [ -s /root/mtg7_run.log ] && { echo "--- mtg7 teacher lines ---"; grep -E "^\[r1\]|^\[followup\]|^\[warn\]" /root/mtg7_run.log | tail -n 8 | cut -c1-250; }
    echo "--- r1 on the fly: why no trajectory ---"; tail -n 5 /root/online_mtg7/r1_fly_out.jsonl.why 2>/dev/null | cut -c1-300
    echo "--- mtg7 by source (fresh questions) ---"
    python3 - <<'PQ' 2>/dev/null
import json, collections
src = {}
for l in open("/root/work/mtg7_items.jsonl"):
    if l.strip(): r = json.loads(l); src[r["q"].strip()] = r.get("src", "?")
by = collections.defaultdict(lambda: [0, 0, 0, 0])   # samples, passing, questions, questions with a pass
seen = collections.defaultdict(list)
for l in open("/root/online_mtg7/rollouts.jsonl"):
    if not l.strip(): continue
    x = json.loads(l)
    if x.get("kind") != "search": continue
    seen[x["q"].strip()].append(x.get("reward", 0) >= 1.0)
for q, v in seen.items():
    s = src.get(q, "follow-up"); b = by[s]; b[0] += len(v); b[1] += sum(v); b[2] += 1; b[3] += any(v)
for s, b in sorted(by.items()): print(f"  {s:10s}: {b[2]:3d} questions, {b[3]:3d} with a pass, samples {b[1]}/{b[0]} = {100*b[1]/max(b[0],1):.0f}%")
PQ
    echo "--- gpu/disk ---"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader; df -h /root | tail -1
    echo "--- processes ---"; pgrep -fa "python3 /root/work/" | cut -c1-200; } > /root/boxlog.txt 2>&1
  hf upload $R /root/boxlog.txt pooler_distill/chatsft/audit/boxlog_G2.txt >/dev/null 2>&1
  sleep 600
done
MK
  setsid nohup bash /root/mirrorkeep.sh > /dev/null 2>&1 < /dev/null &
fi

# ---- mtg7: TriviaQA 400 + MuSiQue two-hop 200, from s100, the recipe of mtg6 (12 and 12, win, follow-ups and R1 on the fly,
# Dolphin's CoT on an all-miss reasoning group, the batch budget 3600 s). Starts once the base is ready. ----
if [ -f /root/.bootstrapped ] && grep -q "BASE_READY" /root/base.log 2>/dev/null && ! pgrep -f "mtg7kee[p].sh" >/dev/null && ! grep -q "MTG7_JOB_DONE" /root/mtg7.log 2>/dev/null && [ ! -e /root/.mtg7 ]; then touch /root/.mtg7
  python3 /root/work/tqa_items.py --n 600 --out /root/work/tqa_items_1.jsonl --pool /root/work/tqa_pool.jsonl \
    --exclude /root/work/eval300.jsonl /root/work/dolphin_v1.jsonl /root/work/dolphinq.jsonl /root/work/replay_v1.jsonl /root/work/selfq_all.jsonl /root/work/probe_items.jsonl /root/work/mt_eval*.jsonl /root/work/mtg*_items.jsonl 2>&1 | tail -1 >> /root/mtg7.log
  python3 /root/work/musique_items.py --n 200 --out /root/work/musique_items_1.jsonl --pool /root/work/musique_pool.jsonl \
    --exclude /root/work/eval300.jsonl /root/work/dolphin_v1.jsonl /root/work/dolphinq.jsonl /root/work/replay_v1.jsonl /root/work/selfq_all.jsonl /root/work/probe_items.jsonl /root/work/mt_eval*.jsonl /root/work/mtg*_items.jsonl 2>&1 | tail -1 >> /root/mtg7.log
  python3 - <<'PM' >> /root/mtg7.log
import json, random, os
t = [dict(json.loads(l), src="triviaqa") for l in open("/root/work/tqa_items_1.jsonl") if l.strip()][:400]
m = [json.loads(l) for l in open("/root/work/musique_items_1.jsonl") if l.strip()] if os.path.exists("/root/work/musique_items_1.jsonl") else []
it = t + m; random.Random(11).shuffle(it)
open("/root/work/mtg7_items.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in it))
print(f"[mtg7] items: {len(t)} TriviaQA + {len(m)} MuSiQue two-hop = {len(it)}")
PM
  cat > /root/mtg7keep.sh <<'M7'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; OUT=/root/online_mtg7
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
S0=/root/mtg3_s100.safetensors; B0=54
while pgrep -f "pool_eval.p[y]|online_loop.p[y]" >/dev/null; do sleep 30; done
[ -s /root/work/mtg7_items.jsonl ] || { echo "MTG7_JOB_DONE no items"; exit 1; }
echo "[mtg7] start from s100 on $(wc -l < /root/work/mtg7_items.jsonl) fresh questions (TriviaQA + MuSiQue two-hop) $(date -u +%H:%M); $(df -h /root | tail -1 | awk '{print $4}') free"
( last=0; while sleep 60; do s=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null || echo 0)
    if [ "$s" != "$last" ] && [ $((s % 40)) -eq 0 ] && [ "$s" -gt 0 ] && [ ! -s /root/mtg7_s$s.safetensors ]; then sleep 20; cp $OUT/latest.safetensors /root/mtg7_s$s.safetensors; hf upload $R /root/mtg7_s$s.safetensors pooler_distill/chatsft/multiturn/mtg7_s$s.safetensors >/dev/null 2>&1; echo "[mtg7] copy at step $s (on the hub)"; fi; last=$s; done ) & CP=$!
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OAI_KEY=$(cat /root/.oai 2>/dev/null) DSK_KEY=$(cat /root/.dsk 2>/dev/null) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /root/work/online_loop.py $S0 $OUT --pooler none --lora-layers all --lora-rank 16 --save-lora-only 1 \
  --mt-items /root/work/mtg7_items.jsonl --heldout /root/work/eval300.jsonl --reason /root/work/dolphin_rft.jsonl --reason-every 4 --reason-g 12 --search-g 12 \
  --demo-on-fail 0.5 --followup 0.5 --r1-on-fail 1 --cot-on-fail 0.5 \
  --steps 120 --save-every 40 --lr 5e-6 --search-lr 5e-6 --search-temp 0.9 --search-gen 2000 --temp 0.6 --gen 7000 --budget 3600 --maxsrch 7 --stop eos \
  --judge-api openai --judge-model gpt-5-nano --w-talk 0.5 --dolphin-min 0 --adv-std 1 --pg-norm mean --kl 0 \
  --guard 1 --guard-steps 20 > /root/mtg7_run.log 2>&1
sleep 90; kill $CP 2>/dev/null
grep -E "^\[data\]|^\[init\]|ONLINE_|Error|Traceback" /root/mtg7_run.log | tail -4 | cut -c1-250
S=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null); echo "[mtg7] stopped at step ${S:-0} $(date -u +%H:%M)"
[ -s $OUT/latest.safetensors ] && [ ! -s /root/mtg7_s$S.safetensors ] && cp $OUT/latest.safetensors /root/mtg7_s$S.safetensors
for f in rollouts.jsonl followups.jsonl r1_onfly.jsonl; do [ -s $OUT/$f ] && hf upload $R $OUT/$f pooler_distill/chatsft/multiturn/mtg7_$f >/dev/null 2>&1; done
python3 - <<'PYL'
import json
qs = {json.loads(l)["q"] for l in open("/root/online_mtg7/rollouts.jsonl") if json.loads(l).get("kind") == "search"}
with open("/root/work/trained_items.jsonl", "a") as f:
    for q in qs: f.write(json.dumps({"q": q, "run": "mtg7"}, ensure_ascii=False) + "\n")
print(f"[mtg7] {len(qs)} search questions used (follow-ups included) -> the ledger", flush=True)
PYL
hf upload $R /root/work/trained_items.jsonl pooler_distill/chatsft/multiturn/trained_items.jsonl >/dev/null 2>&1
rm -f "${OUT:?}"/*.pt "${OUT:?}"/good.safetensors
BEST=; BC=$B0
for CK in $(ls /root/mtg7_s*.safetensors 2>/dev/null | sort -V); do
  T=$(basename $CK .safetensors)
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/ev_0.jsonl /root/work/${T}st_0.jsonl --n 100 $EVARGS --tag "[${T}st0]" > /root/${T}st_0.log 2>&1
  c=$(python3 -c "import json;print(sum(bool(json.loads(l).get('correct')) for l in open('/root/work/${T}st_0.jsonl')))" 2>/dev/null || echo 0)
  echo "[mtg7] $T single shard 0: $c/100 (s100 54)"; [ "$c" -gt "$BC" ] && { BC=$c; BEST=$CK; }
done
[ -n "$BEST" ] || { echo "MTG7_JOB_DONE no copy above s100's shard 0 (54)"; exit 0; }
T=$(basename $BEST .safetensors); echo "[mtg7] full screens for $T"
SN=$BC; for i in 1 2; do env $ENV python3 /root/work/pool_eval.py $BEST /root/work/ev_$i.jsonl /root/work/${T}st_$i.jsonl --n 100 $EVARGS --tag "[${T}st$i]" > /root/${T}st_$i.log 2>&1
  c=$(python3 -c "import json;print(sum(bool(json.loads(l).get('correct')) for l in open('/root/work/${T}st_$i.jsonl')))" 2>/dev/null || echo 0); SN=$((SN + c)); echo "[mtg7] single shard $i: $c/100"; done
echo "[mtg7] $T single-turn: $SN/300 (s100 160)"
env $ENV python3 /root/work/pool_eval.py $BEST /root/work/eval300.jsonl /root/work/br3_$T.jsonl --multiturn /root/work/mt_eval_bridge3.jsonl --mt-mode win --n 1000 $EVARGS --tag "[br3-$T]" > /root/br3_$T.log 2>&1
echo "[mtg7] br3 (win): $(python3 -c "
import json; r=[json.loads(l) for l in open('/root/work/br3_$T.jsonl')]
print('turn 1', sum(x['correct'] for x in r if x['turn']==0), '/', sum(1 for x in r if x['turn']==0), ', follow-up', sum(x['correct'] for x in r if x['turn']==1), '/', sum(1 for x in r if x['turn']==1))" 2>&1 | tail -1) (s100 43/31)"
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $BEST /root/evalrun_dl_mtg7 \
  --questions /root/work/eval300.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl \
  --b 12 --gen 7000 --budget 2400 --temp 0.6 --maxsrch 7 --pooler none --lora-rank 16 --lora-layers all --stop eos --loop-break answer \
  --eval-file /root/work/dolphinq.jsonl --eval-out /root/work/dl_$T.jsonl > /root/dl_$T.log 2>&1
OAI_KEY=$(cat /root/.oai 2>/dev/null) python3 /root/work/dl_judge.py /root/work/dl_$T.jsonl $T | sed "s/\[mix0\]/[mtg7]/"
echo "[mtg7] (s100: Dolphin 56)"
echo "MTG7_JOB_DONE $(date -u)"
M7
  setsid nohup bash -c 'bash /root/mtg7keep.sh 2>&1 | tee -a /root/mtg7.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MTG7_LAUNCHED $(date -u)"
fi
echo "BOXG2_OK serial $BOXG2_SERIAL $(date -u)"
# CTL-END
