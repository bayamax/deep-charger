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
BOXJ_SERIAL=1
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
for f in pool_eval.py q4.py pooler_gptq.py mlx2hf.py; do
  curl -sS -L -o /root/work/$f.new "$RAW/$f?$(date +%s)" && python3 -m py_compile /root/work/$f.new 2>/dev/null && mv /root/work/$f.new /root/work/$f || rm -f /root/work/$f.new
done

# ---- the mirror: what this box is doing, on the hub every 10 minutes ----
if ! pgrep -f "mirrorkee[p].sh" >/dev/null; then
  cat > /root/mirrorkeep.sh <<'MK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
while :; do
  { echo "=== boxJ $(date -u) ==="; echo "--- ctl.log ---"; tail -n 40 /root/ctl.log 2>/dev/null | cut -c1-300
    echo "--- pq.log ---"; tail -n 30 /root/pq.log 2>/dev/null | cut -c1-300
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

# ---- pq1: dequantize the shipped model, GPTQ and RTN the pooler, then the 102-rollout held-out three ways ----
if ! pgrep -f "pqkee[p].sh" >/dev/null && ! grep -q "PQ_JOB_DONE" /root/pq.log 2>/dev/null; then
  cat > /root/pqkeep.sh <<'PK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
REL=/root/hfdl/release/g14-4bit-gptq-trained; M=/root/g14q_hf; POOL=$REL/pooler.safetensors
[ -s $M/model.safetensors ] || python3 /root/work/mlx2hf.py --mlx $REL --out $M || { echo "PQ_ABORT mlx2hf"; exit 1; }
for m in gptq rtn; do
  [ -s /root/pq/pooler_${m}_dq.safetensors ] || python3 /root/work/pooler_gptq.py --pooler $POOL --model $M --data /root/work/dwq_calib/train.jsonl --out /root/pq --method $m 2>&1 | grep -E "^\[pq\]|POOLER_Q_DONE|Error|Traceback"
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
echo "PQ_JOB_DONE $(date -u)"
PK
  setsid nohup bash -c 'bash /root/pqkeep.sh 2>&1 | tee -a /root/pq.log' > /dev/null 2>&1 < /dev/null &
  echo "PQ_LAUNCHED $(date -u)"
fi
echo "BOXJ_OK serial $BOXJ_SERIAL $(date -u)"
# CTL-END
