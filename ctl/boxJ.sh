# box J (RTX 3060 12GB): the pooler to 4 bits, by the method that worked for the model (GPTQ on the app's affine grid,
# group 64, fp16 scale and bias). The model is the shipped 4-bit g14 (release/g14-4bit-gptq-trained, dequantized back
# to the evaluator's form); the pooler is its float32 pooler. Measured two ways: the output cosine of the quantized
# pooler against the float one on held-out windows, and the single-turn search held-out (102 rollouts) with the
# float pooler, the GPTQ pooler and a plain round-to-nearest pooler (what a naive conversion would ship), paired.
cd /root/work 2>/dev/null || { mkdir -p /root/work; cd /root/work; }
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
R=baya1116/hypernet-sp-distill
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
BOXJ_SERIAL=10
if [ -f /root/.boxj_serial ] && [ "$(cat /root/.boxj_serial)" -gt "$BOXJ_SERIAL" ] 2>/dev/null; then echo "BOXJ_STALE $BOXJ_SERIAL"; exit 0; fi
echo $BOXJ_SERIAL > /root/.boxj_serial

# ---- one-time bootstrap: the evaluator's environment (box G's, minus what this job does not use) ----
if [ ! -f /root/.bootstrapped ]; then
  mkdir /root/.bootstrap_lock 2>/dev/null || { echo "bootstrap already running"; exit 0; }
  echo "=== bootstrap $(date -u) ==="
  mkdir -p /root/work/fft_out /root/work/runtime /root/work/dwq_calib /root/hfdl /root/pq
  touch /root/work/runtime/__init__.py
  pip install -q "transformers==4.44.2" "peft==0.12.0" "safetensors==0.8.0" "huggingface_hub>=0.34,<1.0" accelerate certifi datasets scipy 2>&1 | tail -1
  for inc in "box_recover/scripts/*" "fft_out/pooler.pt" "grpo_assets/mus_run/eval_step200/eval_heldout_300q_g2.jsonl" \
             "pooler_distill/pool_eval_cache.jsonl" "pooler_distill/dwq_calib/*" "release/g14-4bit-gptq-trained/*"; do
    for try in 1 2 3 4 5 6; do hf download $R --include "$inc" --local-dir /root/hfdl 2>&1 | tail -1 && break; sleep 20; done
    echo "dl $inc $(date -u +%H:%M)"
  done
  SC=/root/hfdl/box_recover/scripts; cp $SC/*.py $SC/*.sh /root/work/ 2>/dev/null
  ln -sfn /root/hfdl/fft_out/pooler.pt /root/work/fft_out/pooler.pt
  cp /root/hfdl/grpo_assets/mus_run/eval_step200/eval_heldout_300q_g2.jsonl /root/work/eval300.jsonl
  cp /root/hfdl/pooler_distill/pool_eval_cache.jsonl /root/work/pool_eval_cache.jsonl
  cp /root/hfdl/pooler_distill/dwq_calib/train.jsonl /root/work/dwq_calib/train.jsonl
  python3 - <<'PYB'
import json
qs=[l for l in open("/root/work/eval300.jsonl") if l.strip()][:300]
for i in range(3): open(f"/root/work/ev_{i}.jsonl","w").writelines(qs[i::3])
print(f"[shard] {len(qs)} questions in 3 shards")
PYB
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /root | tail -1; free -g | head -2
  ls -la /root/hfdl/release/g14-4bit-gptq-trained/ | tail -n +2 | awk '{print $5, $9}'
  [ -s /root/hfdl/release/g14-4bit-gptq-trained/model.safetensors ] && [ -s /root/work/grpo_e2e_torch.py ] && touch /root/.bootstrapped && echo "BOOTSTRAP_DONE $(date -u)"
  rmdir /root/.bootstrap_lock 2>/dev/null
fi
[ -f /root/.bootstrapped ] || { echo "bootstrap incomplete - stopping here"; exit 0; }

# the code, fresh from the branch on every control run
for f in pool_eval.py q4.py pooler_gptq.py mlx2hf.py web_search.py online_loop.py pooler_gridfit.py pooler4bit_release.py; do
  curl -sS -L -o /root/work/$f.new "$RAW/$f?$(date +%s)" && python3 -m py_compile /root/work/$f.new 2>/dev/null && mv /root/work/$f.new /root/work/$f || rm -f /root/work/$f.new
done

cp /root/work/web_search.py /root/work/runtime/web_search.py 2>/dev/null   # the evaluator imports it as runtime.web_search

if [ ! -e /root/.mirror_v6 ]; then touch /root/.mirror_v6; pkill -f "mirrorkee[p].sh"; sleep 1; echo "MIRROR_RESTART (pq3 lines) $(date -u)"; fi
# ---- the mirror: what this box is doing, on the hub every 10 minutes ----
if ! pgrep -f "mirrorkee[p].sh" >/dev/null; then
  cat > /root/mirrorkeep.sh <<'MK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
while :; do
  { echo "=== boxJ $(date -u) ==="; echo "--- ctl.log ---"; tail -n 40 /root/ctl.log 2>/dev/null | cut -c1-300
    echo "--- pq.log ---"; tail -n 8 /root/pq.log 2>/dev/null | cut -c1-300
    echo "--- pq7.log ---"; tail -n 12 /root/pq7.log 2>/dev/null | cut -c1-300; grep -E "^step|^val" /root/pq7_fit.log 2>/dev/null | tail -2
    echo "--- pq6.log ---"; tail -n 3 /root/pq6.log 2>/dev/null | cut -c1-300
    echo "--- pq5.log ---"; tail -n 4 /root/pq5.log 2>/dev/null | cut -c1-300
    echo "--- pq4.log ---"; tail -n 10 /root/pq4.log 2>/dev/null | cut -c1-300
    echo "--- pq3.log ---"; tail -n 30 /root/pq3.log 2>/dev/null | cut -c1-300; for f in /root/work/pq*_dolphin.jsonl; do [ -e $f ] && echo "$(basename $f) $(wc -l < $f)"; done; grep -h "^\[eval\]" /root/pq*_dolphin.log 2>/dev/null | tail -2
    echo "--- eval progress ---"; for f in /root/work/pq*_out_*.jsonl; do [ -e $f ] && echo "$(basename $f) $(wc -l < $f)"; done
    echo "--- gpu ---"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
    echo "--- processes ---"; pgrep -fa "python3 /root/work/" | cut -c1-160; } > /root/boxlog.txt 2>&1
  hf upload $R /root/boxlog.txt pooler_distill/pooler4bit/boxlog_J.txt >/dev/null 2>&1
  sleep 600
done
MK
  setsid nohup bash /root/mirrorkeep.sh > /dev/null 2>&1 < /dev/null &
  echo "MIRROR_LAUNCHED $(date -u)"
fi

# pq2 (the first run's evaluations all failed on a missing runtime.web_search; the quantization re-runs, deterministic,
# for the input-driven error measure)
# ---- pq1: dequantize the shipped model, GPTQ and RTN the pooler, then the 102-rollout held-out three ways ----
if false && ! pgrep -f "pqkee[p].sh" >/dev/null && ! grep -q "PQ2_JOB_DONE" /root/pq.log 2>/dev/null; then   # replaced by pq3
  cat > /root/pqkeep.sh <<'PK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
REL=/root/hfdl/release/g14-4bit-gptq-trained; M=/root/g14q_hf; POOL=$REL/pooler.safetensors
[ -s $M/model.safetensors ] || python3 /root/work/mlx2hf.py --mlx $REL --out $M || { echo "PQ_ABORT mlx2hf"; exit 1; }
for m in gptq rtn; do
  python3 /root/work/pooler_gptq.py --pooler $POOL --model $M --data /root/work/dwq_calib/train.jsonl --out /root/pq --method $m 2>&1 | grep -E "^\[pq\]|POOLER_Q_DONE|Error|Traceback"
done
[ -s /root/pq/pooler_gptq_dq.safetensors ] || { echo "PQ_ABORT gptq"; exit 1; }
for f in pooler_gptq_mlx pooler_rtn_mlx pooler_gptq_dq; do hf upload $R /root/pq/$f.safetensors pooler_distill/pooler4bit/$f.safetensors >/dev/null 2>&1; done
ENV="SP_BASE=$M SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
for arm in "pqg:/root/pq/pooler_gptq_dq.safetensors" "pqf:$POOL" "pqr:/root/pq/pooler_rtn_dq.safetensors"; do
  T=${arm%%:*}; CK=${arm#*:}
  for i in 0 1 2; do
    env $ENV python3 /root/work/pool_eval.py $CK /root/work/ev_$i.jsonl /root/work/${T}_out_$i.jsonl --n 34 $EVARGS --tag "[$T$i]" > /root/${T}_$i.log 2>&1
    echo "$(grep -E "EVAL_DONE|Error|Traceback" /root/${T}_$i.log | tail -1 | cut -c1-250)"
    hf upload $R /root/work/${T}_out_$i.jsonl pooler_distill/pooler4bit/${T}_out_$i.jsonl >/dev/null 2>&1
  done
done
echo "PQ2_JOB_DONE $(date -u)"
PK
  setsid nohup bash -c 'bash /root/pqkeep.sh 2>&1 | tee -a /root/pq.log' > /dev/null 2>&1 < /dev/null &
  echo "PQ_LAUNCHED $(date -u)"
fi
# ---- pq3 (2026-10-01, the user: the pooler compresses thinking too, so its quantization must hold for reasoning, not only
# search): calibration half search traces, half reasoning (dolphin_v1's R1 thinking and replies, the Dolphin held-out
# hundred left out), and every arm measured on both yardsticks - the Dolphin held-out (100, nano-judged on box G, which
# holds the key; g14's settings: temp 0.6, gen 7000, loop-break answer) and the search held-out (102). Arms: float
# pooler (pqf), GPTQ on the mixed calibration (pqm), round-to-nearest (pqr). pq2's search-only GPTQ arm is dropped.
if [ ! -e /root/.pq3_swap ]; then touch /root/.pq3_swap; pkill -f "pqkee[p].sh"; pkill -f "pool_eval.p[y]"; sleep 3; echo "PQ3_SWAP stopped pq2 $(date -u)"; fi
if false && ! pgrep -f "pq3kee[p].sh" >/dev/null && ! grep -q "PQ3_JOB_DONE" /root/pq3.log 2>/dev/null; then   # replaced by pq4
  cat > /root/pq3keep.sh <<'PK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
REL=/root/hfdl/release/g14-4bit-gptq-trained; M=/root/g14q_hf; POOL=$REL/pooler.safetensors
for f in dolphin_v1.jsonl dolphin_v2.jsonl; do [ -s /root/hfdl/pooler_distill/chatsft/$f ] || hf download $R --include "pooler_distill/chatsft/$f" --local-dir /root/hfdl >/dev/null 2>&1; done
cp /root/hfdl/pooler_distill/chatsft/dolphin_v1.jsonl /root/work/dolphin_v1.jsonl
python3 - <<'PD'
import json, random
v1 = [json.loads(l) for l in open("/root/hfdl/pooler_distill/chatsft/dolphin_v1.jsonl") if l.strip()]
v1 = [r for r in v1 if r.get("q") and r.get("reply")]
v2q = set((json.loads(l).get("q") or "").strip() for l in open("/root/hfdl/pooler_distill/chatsft/dolphin_v2.jsonl") if l.strip())
cand = [r for r in v1 if r["q"].strip() not in v2q]; random.Random(0).shuffle(cand)
held = cand[:100]; hq = set(r["q"].strip() for r in held)
with open("/root/work/dolphin_heldout100.jsonl", "w") as o:
    for r in held: o.write(json.dumps({"q": r["q"], "ref": r["reply"]}, ensure_ascii=False) + "\n")
with open("/root/work/dolphinq.jsonl", "w") as o:
    for r in held: o.write(json.dumps({"q": r["q"]}, ensure_ascii=False) + "\n")
n = 0
with open("/root/work/dolphin_calib.jsonl", "w") as o:
    for r in v1:
        if r["q"].strip() not in hq: o.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
print(f"[pq3] Dolphin held-out 100 (box G's construction), {n} reasoning records for calibration")
PD
for m in gptq rtn; do
  python3 /root/work/pooler_gptq.py --pooler $POOL --model $M --data /root/work/dwq_calib/train.jsonl --data2 /root/work/dolphin_calib.jsonl --mix 0.5 --tag _mix --out /root/pq --method $m 2>&1 | grep -E "^\[pq\]|POOLER_Q_DONE|Error|Traceback"
done
[ -s /root/pq/pooler_gptq_mix_dq.safetensors ] || { echo "PQ3_ABORT gptq"; exit 1; }
for f in pooler_gptq_mix_mlx pooler_gptq_mix_dq; do hf upload $R /root/pq/$f.safetensors pooler_distill/pooler4bit/$f.safetensors >/dev/null 2>&1; done
ARMS="pqf:$POOL pqm:/root/pq/pooler_gptq_mix_dq.safetensors pqr:/root/pq/pooler_rtn_mix_dq.safetensors"
for arm in $ARMS; do
  T=${arm%%:*}; CK=${arm#*:}
  OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $M /root/evalrun_$T \
    --questions /root/work/eval300.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl --pooler-init $CK \
    --b 12 --gen 7000 --budget 2400 --temp 0.6 --maxsrch 7 --pooler none --lora-rank 16 --lora-layers 20-27 --stop eos --loop-break answer \
    --eval-file /root/work/dolphinq.jsonl --eval-out /root/work/${T}_dolphin.jsonl > /root/${T}_dolphin.log 2>&1
  echo "[pq3] $T dolphin: $(wc -l < /root/work/${T}_dolphin.jsonl 2>/dev/null) replies, $(grep -E 'EVAL_DONE|Error' /root/${T}_dolphin.log | tail -1 | cut -c1-160) $(date -u +%H:%M)"
  hf upload $R /root/work/${T}_dolphin.jsonl pooler_distill/pooler4bit/${T}_dolphin.jsonl >/dev/null 2>&1
done
ENV="SP_BASE=$M SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
for arm in $ARMS; do
  T=${arm%%:*}; CK=${arm#*:}
  for i in 0 1 2; do
    env $ENV python3 /root/work/pool_eval.py $CK /root/work/ev_$i.jsonl /root/work/${T}_out_$i.jsonl --n 34 $EVARGS --tag "[$T$i]" > /root/${T}_$i.log 2>&1
    echo "[pq3] $(grep -E "EVAL_DONE|Error|Traceback" /root/${T}_$i.log | tail -1 | cut -c1-250)"
    hf upload $R /root/work/${T}_out_$i.jsonl pooler_distill/pooler4bit/${T}_out_$i.jsonl >/dev/null 2>&1
  done
done
echo "PQ3_JOB_DONE $(date -u)"
PK
  setsid nohup bash -c 'bash /root/pq3keep.sh 2>&1 | tee -a /root/pq3.log' > /dev/null 2>&1 < /dev/null &
  echo "PQ3_LAUNCHED $(date -u)"
fi
# ---- pq4 (the user: re-use what is already solved). The float pooler on search is already measured for the shipped model
# on these very questions (q14gx, 300 rollouts, 47.3%; paired per question offline), and round-to-nearest is out on the
# pooler error alone (14x GPTQ's); so only: the float pooler on the Dolphin held-out (running now - no 4-bit float run
# of it exists, g14's 53% was the 16-bit model), the mixed-calibration GPTQ pooler on the Dolphin held-out, and that
# GPTQ pooler on the search held-out (102). pq3's keeper stops; its running evaluation is left to finish and is waited for.
if [ ! -e /root/.pq4_swap ]; then touch /root/.pq4_swap; pkill -f "pq3kee[p].sh"; sleep 2; echo "PQ4_SWAP stopped the pq3 keeper (its running evaluation continues) $(date -u)"; fi
if ! pgrep -f "pq4kee[p].sh" >/dev/null && ! grep -q "PQ4_JOB_DONE" /root/pq4.log 2>/dev/null; then
  cat > /root/pq4keep.sh <<'PK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
M=/root/g14q_hf; GQ=/root/pq/pooler_gptq_mix_dq.safetensors
while pgrep -f "evalrun_pqf" >/dev/null; do sleep 60; done
echo "[pq4] pqf dolphin: $(wc -l < /root/work/pqf_dolphin.jsonl) replies $(date -u +%H:%M)"
hf upload $R /root/work/pqf_dolphin.jsonl pooler_distill/pooler4bit/pqf_dolphin.jsonl >/dev/null 2>&1
OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $M /root/evalrun_pqm \
  --questions /root/work/eval300.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl --pooler-init $GQ \
  --b 12 --gen 7000 --budget 2400 --temp 0.6 --maxsrch 7 --pooler none --lora-rank 16 --lora-layers 20-27 --stop eos --loop-break answer \
  --eval-file /root/work/dolphinq.jsonl --eval-out /root/work/pqm_dolphin.jsonl > /root/pqm_dolphin.log 2>&1
echo "[pq4] pqm dolphin: $(wc -l < /root/work/pqm_dolphin.jsonl) replies $(date -u +%H:%M)"
hf upload $R /root/work/pqm_dolphin.jsonl pooler_distill/pooler4bit/pqm_dolphin.jsonl >/dev/null 2>&1
ENV="SP_BASE=$M SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
for i in 0 1 2; do
  env $ENV python3 /root/work/pool_eval.py $GQ /root/work/ev_$i.jsonl /root/work/pqm_out_$i.jsonl --n 34 $EVARGS --tag "[pqm$i]" > /root/pqm_$i.log 2>&1
  echo "[pq4] $(grep -E "EVAL_DONE|Error|Traceback" /root/pqm_$i.log | tail -1 | cut -c1-250)"
  hf upload $R /root/work/pqm_out_$i.jsonl pooler_distill/pooler4bit/pqm_out_$i.jsonl >/dev/null 2>&1
done
for i in 0 1 2; do hf upload $R /root/work/pqg_out_$i.jsonl pooler_distill/pooler4bit/pqg_out_$i.jsonl >/dev/null 2>&1; done
echo "PQ4_JOB_DONE $(date -u)"
PK
  setsid nohup bash -c 'bash /root/pq4keep.sh 2>&1 | tee -a /root/pq4.log' > /dev/null 2>&1 < /dev/null &
  echo "PQ4_LAUNCHED $(date -u)"
fi
# ---- pq5: the Dolphin held-out read 54 (float) vs 47 (GPTQ) on one draw each - 17 vs 10 discordant, inside the noise but
# twice the unfinished (6 vs 12) and shorter thinking (median 790 vs 562 words). A second draw of both, same questions,
# so the comparison rests on 200 replies an arm.
if ! pgrep -f "pq5kee[p].sh" >/dev/null && ! grep -q "PQ5_JOB_DONE" /root/pq5.log 2>/dev/null; then
  cat > /root/pq5keep.sh <<'PK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
M=/root/g14q_hf; REL=/root/hfdl/release/g14-4bit-gptq-trained
until grep -q "PQ4_JOB_DONE" /root/pq4.log 2>/dev/null; do sleep 60; done
for arm in "pqf2:$REL/pooler.safetensors" "pqm2:/root/pq/pooler_gptq_mix_dq.safetensors"; do
  T=${arm%%:*}; CK=${arm#*:}
  OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $M /root/evalrun_$T \
    --questions /root/work/eval300.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl --pooler-init $CK \
    --b 12 --gen 7000 --budget 2400 --temp 0.6 --maxsrch 7 --pooler none --lora-rank 16 --lora-layers 20-27 --stop eos --loop-break answer \
    --eval-file /root/work/dolphinq.jsonl --eval-out /root/work/${T}_dolphin.jsonl > /root/${T}_dolphin.log 2>&1
  echo "[pq5] $T dolphin: $(wc -l < /root/work/${T}_dolphin.jsonl) replies $(date -u +%H:%M)"
  hf upload $R /root/work/${T}_dolphin.jsonl pooler_distill/pooler4bit/${T}_dolphin.jsonl >/dev/null 2>&1
done
echo "PQ5_JOB_DONE $(date -u)"
PK
  setsid nohup bash -c 'bash /root/pq5keep.sh 2>&1 | tee -a /root/pq5.log' > /dev/null 2>&1 < /dev/null &
  echo "PQ5_LAUNCHED $(date -u)"
fi
# ---- pq6 (the user's reading of the Dolphin split: the drop sits at 300-550 thinking words, where nothing is compressed
# yet; there the pooler's only effect is the constant 32-vector soft prompt it emits for an EMPTY past, off by 0.5% after
# quantization - and the model is tuned to the exact one). The GPTQ pooler with that empty-past prompt taken from the
# float pooler (SP_EMPTY_FROM; 98 KB extra for the app), on the Dolphin held-out.
if ! pgrep -f "pq6kee[p].sh" >/dev/null && ! grep -q "PQ6_JOB_DONE" /root/pq6.log 2>/dev/null; then
  cat > /root/pq6keep.sh <<'PK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
M=/root/g14q_hf; REL=/root/hfdl/release/g14-4bit-gptq-trained
until grep -q "PQ5_JOB_DONE" /root/pq5.log 2>/dev/null; do sleep 60; done
T=pqe
SP_EMPTY_FROM=$REL/pooler.safetensors OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $M /root/evalrun_$T \
  --questions /root/work/eval300.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl --pooler-init /root/pq/pooler_gptq_mix_dq.safetensors \
  --b 12 --gen 7000 --budget 2400 --temp 0.6 --maxsrch 7 --pooler none --lora-rank 16 --lora-layers 20-27 --stop eos --loop-break answer \
  --eval-file /root/work/dolphinq.jsonl --eval-out /root/work/${T}_dolphin.jsonl > /root/${T}_dolphin.log 2>&1
echo "[pq6] $(grep -m1 'empty-past' /root/${T}_dolphin.log | cut -c1-160)"
echo "[pq6] $T dolphin: $(wc -l < /root/work/${T}_dolphin.jsonl) replies $(date -u +%H:%M)"
hf upload $R /root/work/${T}_dolphin.jsonl pooler_distill/pooler4bit/${T}_dolphin.jsonl >/dev/null 2>&1
echo "PQ6_JOB_DONE $(date -u)"
PK
  setsid nohup bash -c 'bash /root/pq6keep.sh 2>&1 | tee -a /root/pq6.log' > /dev/null 2>&1 < /dev/null &
  echo "PQ6_LAUNCHED $(date -u)"
fi
# ---- pq7 (the user: if the reasoning gap holds, train the 4-bit pooler's grid as the model's was trained): GPTQ's codes
# kept, the fp16 scales and biases (and the float parts) moved so the model's next-token distribution under the 4-bit
# pooler matches it under the float one, on search and reasoning traces (pooler_gridfit.py). Prepared now, run after
# pq6 so it never shares the card with an evaluation: a 3-step selftest, the fit, then the Dolphin held-out (pqt).
if ! pgrep -f "pq7kee[p].sh" >/dev/null && ! grep -q "PQ7_JOB_DONE\|PQ7_ABORT" /root/pq7.log 2>/dev/null; then
  cat > /root/pq7keep.sh <<'PK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
M=/root/g14q_hf; REL=/root/hfdl/release/g14-4bit-gptq-trained
until grep -q "PQ6_JOB_DONE" /root/pq6.log 2>/dev/null; do sleep 60; done
GF="env SP_BASE=$M SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/pooler_gridfit.py --float $REL/pooler.safetensors --codes /root/pq/pooler_gptq_mix_mlx.safetensors --search /root/work/dwq_calib/train.jsonl --reason /root/work/dolphin_calib.jsonl --out /root/pq/pooler_gridfit"
$GF --selftest 3 > /root/pq7_selftest.log 2>&1
grep -E "^\[grid\]|^\[data\]|^step|^val|SELFTEST|Error|Traceback|memory" /root/pq7_selftest.log | tail -8 | cut -c1-250
grep -q GRIDFIT_SELFTEST_DONE /root/pq7_selftest.log || { echo "PQ7_ABORT selftest"; exit 1; }
$GF --steps 600 > /root/pq7_fit.log 2>&1
grep -E "^\[grid\]|^val|GRIDFIT_DONE|Error|Traceback" /root/pq7_fit.log | tail -16 | cut -c1-250
[ -s /root/pq/pooler_gridfit_dq.safetensors ] || { echo "PQ7_ABORT fit"; exit 1; }
for f in pooler_gridfit_mlx pooler_gridfit_dq; do hf upload $R /root/pq/$f.safetensors pooler_distill/pooler4bit/$f.safetensors >/dev/null 2>&1; done
T=pqt
OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $M /root/evalrun_$T \
  --questions /root/work/eval300.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl --pooler-init /root/pq/pooler_gridfit_dq.safetensors \
  --b 12 --gen 7000 --budget 2400 --temp 0.6 --maxsrch 7 --pooler none --lora-rank 16 --lora-layers 20-27 --stop eos --loop-break answer \
  --eval-file /root/work/dolphinq.jsonl --eval-out /root/work/${T}_dolphin.jsonl > /root/${T}_dolphin.log 2>&1
echo "[pq7] $T dolphin: $(wc -l < /root/work/${T}_dolphin.jsonl) replies $(date -u +%H:%M)"
hf upload $R /root/work/${T}_dolphin.jsonl pooler_distill/pooler4bit/${T}_dolphin.jsonl >/dev/null 2>&1
echo "PQ7_JOB_DONE $(date -u)"
PK
  setsid nohup bash -c 'bash /root/pq7keep.sh 2>&1 | tee -a /root/pq7.log' > /dev/null 2>&1 < /dev/null &
  echo "PQ7_LAUNCHED $(date -u)"
fi
# ---- release (2026-10-02, the user: "put the 4-bit pooler that matches the float one on Hugging Face, and add it to
# the app notes"): the GPTQ (search + reasoning calibration) pooler beside the app's model as pooler_4bit.safetensors,
# its float parts in float32 as evaluated (checked bit-exact before writing), and the release notes with section 1a.
if [ ! -e /root/.pq_release_v1 ]; then
  touch /root/.pq_release_v1
  python3 /root/work/pooler4bit_release.py /root/pq/pooler_gptq_mix_mlx.safetensors /root/pq/pooler_gptq_mix_dq.safetensors /root/pq/pooler_4bit.safetensors 2>&1 | tail -2
  if [ -s /root/pq/pooler_4bit.safetensors ]; then
    for t in 1 2 3; do hf upload $R /root/pq/pooler_4bit.safetensors release/g14-4bit-gptq-trained/pooler_4bit.safetensors 2>&1 | tail -1 && break; sleep 20; done
    for f in USAGE.md README.md; do curl -sSf -o /root/pq/release_$f "$RAW/release/$f?nocache=$(date +%s)" && grep -q "pooler_4bit" /root/pq/release_$f && hf upload $R /root/pq/release_$f release/$f 2>&1 | tail -1; done
    hf download $R release/g14-4bit-gptq-trained/pooler_4bit.safetensors --local-dir /root/pq/check >/dev/null 2>&1
    echo "PQ_RELEASE_DONE hub copy sha256 $(sha256sum /root/pq/check/release/g14-4bit-gptq-trained/pooler_4bit.safetensors 2>/dev/null | cut -c1-64) $(date -u)"
  else echo "PQ_RELEASE_ABORT"; fi
fi
# ---- final archive before the box is deleted (2026-10-02, the user's OK): logs, the held-out / calibration sets built
# here, and anything in /root/pq or /root/work not yet on the hub, under pooler_distill/pooler4bit/final/
if [ ! -e /root/.pq_archive_v1 ]; then
  touch /root/.pq_archive_v1; A=/root/pq_archive; rm -rf $A; mkdir -p $A/logs $A/data $A/pq $A/work
  cp /root/ctl.log /root/pq*.log $A/logs/ 2>/dev/null
  cp /root/work/dolphin_calib.jsonl /root/work/dolphin_heldout100.jsonl /root/work/dolphinq.jsonl $A/data/ 2>/dev/null
  for f in /root/pq/*.safetensors; do b=$(basename $f); [ "$b" = pooler_4bit.safetensors ] && continue
    curl -sfI "https://huggingface.co/$R/resolve/main/pooler_distill/pooler4bit/$b" >/dev/null || cp $f $A/pq/; done
  for f in /root/work/pq*_out_*.jsonl /root/work/pq*_dolphin*.jsonl; do [ -e $f ] || continue; b=$(basename $f)
    curl -sfI "https://huggingface.co/$R/resolve/main/pooler_distill/pooler4bit/$b" >/dev/null || cp $f $A/work/; done
  ls -R $A | head -60; du -sh $A
  for t in 1 2 3; do hf upload $R $A pooler_distill/pooler4bit/final 2>&1 | tail -1 && break; sleep 30; done
  echo "PQ_ARCHIVE_DONE $(find $A -type f | wc -l) files $(date -u)"
fi
echo "BOXJ_OK serial $BOXJ_SERIAL $(date -u)"
# CTL-END
