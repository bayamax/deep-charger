# box I (RTX 3060 12GB, the small first try the user asked for: cheap box, small data, small model): the sentence-sequence BART - each sentence of a document becomes one vector (a frozen
# sentence encoder), and a transformer is trained on the sequence of those vectors: an encoder that reads a
# corrupted sequence (spans of sentences masked) and a decoder that regenerates it one sentence vector at a time.
# The encoder's outputs are context-aware sentence / page vectors (search, hierarchy); the decoder generates in
# sentence-vector space. Separate from box G (the app model's multi-turn work); results go to the hub under sentbart/.
cd /root
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
R=baya1116/hypernet-sp-distill
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
BOXI_SERIAL=11
if [ -f /root/.boxi_serial ] && [ "$(cat /root/.boxi_serial)" -gt "$BOXI_SERIAL" ] 2>/dev/null; then echo "BOXI_STALE $BOXI_SERIAL"; exit 0; fi
echo $BOXI_SERIAL > /root/.boxi_serial
mkdir -p /root/sb /root/work

# ---- one-time bootstrap ----
if [ ! -f /root/.bootstrapped ]; then
  mkdir /root/.bootstrap_lock 2>/dev/null || { echo "bootstrap already running"; exit 0; }
  echo "=== bootstrap $(date -u) ==="
  pip install -q "transformers==4.44.2" "safetensors" "huggingface_hub>=0.34,<1.0" datasets pyarrow blingfire numpy 2>&1 | tail -1
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  df -h /root | tail -1; free -g | head -2
  touch /root/.bootstrapped; rmdir /root/.bootstrap_lock
  echo "=== bootstrap done $(date -u) ==="
fi

# the code, fresh from the branch on every control run
for f in prep.py embed.py train.py evalsb.py; do
  curl -sS -L -o /root/sb/$f.new "$RAW/sentbart/$f?$(date +%s)" && grep -q "^#!/usr/bin/env python3" /root/sb/$f.new && mv /root/sb/$f.new /root/sb/$f || rm -f /root/sb/$f.new
done
ls /root/sb

# ---- the mirror: what this box is doing, on the hub every 10 minutes ----
if ! pgrep -f "mirrorkee[p].sh" >/dev/null; then
  cat > /root/mirrorkeep.sh <<'MK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
while :; do
  { echo "=== boxI $(date -u) ==="; echo "--- ctl.log ---"; tail -n 60 /root/ctl.log 2>/dev/null | cut -c1-300
    for f in /root/sb_*.log; do [ -e $f ] && { echo "--- $f ---"; tail -n 25 $f | cut -c1-300; }; done
    echo "--- gpu ---"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
    echo "--- disk ---"; df -h /root | tail -1; du -sh /root/sb/data 2>/dev/null
    echo "--- processes ---"; pgrep -fa "python3 /root/sb/" | cut -c1-160; } > /root/boxlog.txt 2>&1
  hf upload $R /root/boxlog.txt sentbart/audit/boxlog_I.txt >/dev/null 2>&1
  sleep 600
done
MK
  setsid nohup bash /root/mirrorkeep.sh > /dev/null 2>&1 < /dev/null &
  echo "MIRROR_LAUNCHED $(date -u)"
fi
# ---- data: 2 of the 41 shards of English Wikipedia (about 240k articles, ~10M sentences) as sentence lists, then their vectors ----
if ! pgrep -f "datakee[p].sh" >/dev/null && ! grep -q "DATA_JOB_DONE" /root/sb_data.log 2>/dev/null; then
  cat > /root/datakeep.sh <<'DK'
cd /root/sb; export HF_HUB_ENABLE_HF_TRANSFER=0
python3 /root/sb/prep.py --out /root/sb/data/docs --shards 0-1 || { echo "DATA_ABORT prep"; exit 1; }
python3 /root/sb/embed.py --dir /root/sb/data/docs || { echo "DATA_ABORT embed"; exit 1; }
du -sh /root/sb/data/docs
echo "DATA_JOB_DONE $(date -u)"
DK
  setsid nohup bash -c 'bash /root/datakeep.sh 2>&1 | tee -a /root/sb_data.log' > /dev/null 2>&1 < /dev/null &
  echo "DATA_LAUNCHED $(date -u)"
fi
# ---- run1: small - shard 0 to train, shard 1 held out; d 512, 4+4 layers (~30M parameters), batch 32, 10k steps ----
if ! pgrep -f "run1kee[p].sh" >/dev/null && ! grep -q "RUN1_JOB_DONE" /root/sb_run1.log 2>/dev/null; then
  cat > /root/run1keep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
until grep -qE "DATA_JOB_DONE|DATA_ABORT" /root/sb_data.log 2>/dev/null; do sleep 120; done
grep -q DATA_ABORT /root/sb_data.log && { echo "RUN1_ABORT: data failed"; exit 1; }
echo "[run1] start $(date -u +%H:%M)"
( while sleep 1800; do hf upload $R /root/sb/run1/train.log sentbart/small1/train.log >/dev/null 2>&1; done ) &
UP=$!
python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/run1 --steps 10000 --batch 32 --seq 128 --d 512 --layers 4 --heads 8 --ffn 2048 --warmup 500 --eval-every 500 --save-every 1000
kill $UP 2>/dev/null
hf upload $R /root/sb/run1/train.log sentbart/small1/train.log >/dev/null 2>&1
hf upload $R /root/sb/run1/model_latest.pt sentbart/small1/model_latest.pt >/dev/null 2>&1
echo "RUN1_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/run1keep.sh 2>&1 | tee -a /root/sb_run1.log' > /dev/null 2>&1 < /dev/null &
  echo "RUN1_LAUNCHED $(date -u)"
fi
# ---- run2 / run3: run1 was still climbing when its learning rate reached zero (10k steps, 15 min), and its next-sentence
# metric was unfair to it (the baseline alone had the sentences already seen removed from its candidates; both now do).
# 6x longer, without (run2) and with (run3) the decoder starting from the previous sentence's vector ----
if ! pgrep -f "run23kee[p].sh" >/dev/null && ! grep -q "RUN23_JOB_DONE" /root/sb_run23.log 2>/dev/null; then
  cat > /root/run23keep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
while pgrep -f "python3 /root/sb/train.py" >/dev/null; do sleep 60; done
for cfg in "run2:0" "run3:1"; do
  N=${cfg%%:*}; PS=${cfg#*:}
  echo "[$N] start $(date -u +%H:%M) prev-skip $PS"
  python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/$N --steps 60000 --batch 32 --seq 128 --d 512 --layers 4 --heads 8 --ffn 2048 --warmup 1000 --eval-every 2000 --save-every 4000 --prev-skip $PS 2>&1 | grep -E "^\[eval|TRAIN_DONE|Error|Traceback"
  hf upload $R /root/sb/$N/train.log sentbart/small_$N/train.log >/dev/null 2>&1
  hf upload $R /root/sb/$N/model_latest.pt sentbart/small_$N/model_latest.pt >/dev/null 2>&1
done
echo "RUN23_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/run23keep.sh 2>&1 | tee -a /root/sb_run23.log' > /dev/null 2>&1 < /dev/null &
  echo "RUN23_LAUNCHED $(date -u)"
fi
# ---- run4, re-planned 2026-10-01 23:30 JST (the user: "if there is headroom, add data first"): run2 saw its ~120k
# training documents about 15 times over and flattened, and 60k steps take only 87 min on this card, so data is the
# limit, not compute. Vectors go int8 with a per-row scale (cosine to fp16 >= 0.9999, half the disk), the text of a
# shard is deleted once it is embedded (prep.py can rewrite it; only the held-out shard 001 keeps its text), and
# shards are added one at a time up to 20 while at least 6 GB stay free. Shard 001 stays held out, as in run2/run3,
# so the numbers line up. Then the page-token model, 200k steps.
if [ ! -e /root/.run4_swap ]; then touch /root/.run4_swap; pkill -f "run4kee[p].sh"; pkill -f "python3 /root/sb/embed.py"; pkill -f "python3 /root/sb/prep.py"; sleep 5; echo "RUN4_SWAPPED (old 6-shard run4 stopped) $(date -u)"; fi
if ! pgrep -f "run4bkee[p].sh" >/dev/null && ! grep -q "RUN4B_JOB_DONE\|RUN4B_ABORT" /root/sb_run4b.log 2>/dev/null; then
  cat > /root/run4bkeep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb
while pgrep -f "python3 /root/sb/" >/dev/null; do sleep 30; done
rm -f $D/*.tmp.npy $D/*.part
# fp16 shards already written -> int8 + scale
python3 - <<'PY2'
import glob, os, numpy as np
for vf in sorted(glob.glob("/root/sb/data/docs/vec_*.npy")):
    v = np.load(vf, mmap_mode="r")
    if v.dtype == np.int8: continue
    k = os.path.basename(vf)[4:-4]; q = np.empty(v.shape, np.int8); sc = np.empty(len(v), np.float16)
    for i in range(0, len(v), 1 << 20):
        x = np.asarray(v[i:i + (1 << 20)], np.float32); s = np.abs(x).max(1).clip(1e-6).astype(np.float16)
        sc[i:i + len(x)] = s; q[i:i + len(x)] = np.round(x / s.astype(np.float32)[:, None] * 127).clip(-127, 127)
    np.save(f"/root/sb/data/docs/scl_{k}.npy", sc); np.save(vf + ".tmp.npy", q); os.replace(vf + ".tmp.npy", vf)
    print(f"[int8] {k}: {len(v)} rows", flush=True)
PY2
dropt() { for f in $D/docs_*.jsonl; do k=$(basename $f .jsonl); k=${k#docs_}; [ "$k" = 001 ] && continue; [ -e $D/vec_$k.npy ] && rm -f $f; done; rm -rf /root/.cache/huggingface/hub/datasets--wikimedia--wikipedia; }
python3 /root/sb/embed.py --dir $D --int8 1 2>&1 | grep -E "^\[embed\]|Error|Traceback"; dropt
for si in $(seq 2 19); do
  k=$(printf %03d $si); [ -e $D/vec_$k.npy ] && continue
  fr=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9); [ "$fr" -lt 6 ] && { echo "[data] stop at shard $si: ${fr} GB free"; break; }
  python3 /root/sb/prep.py --out $D --shards $si-$si 2>&1 | grep -E "^\[prep\] shard|Error"
  python3 /root/sb/embed.py --dir $D --int8 1 2>&1 | grep -E "^\[embed\] [0-9]|Error|Traceback"; dropt
done
ls $D/vec_*.npy | wc -l; du -sh $D; df -h /root | tail -1
[ -e $D/vec_001.npy ] || { echo "RUN4B_ABORT no held-out shard"; exit 1; }
echo "[run4b] train start $(date -u +%H:%M)"
( while sleep 1800; do hf upload $R /root/sb/run4/train.log sentbart/small_run4/train.log >/dev/null 2>&1; done ) & UP=$!
python3 /root/sb/train.py --data $D --out /root/sb/run4 --eval-shard 001 --steps 200000 --batch 32 --seq 128 --d 512 --layers 4 --heads 8 --ffn 2048 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 2>&1 | grep -E "^\[data\]|^\[model\]|^\[eval|TRAIN_DONE|Error|Traceback"
kill $UP 2>/dev/null
hf upload $R /root/sb/run4/train.log sentbart/small_run4/train.log >/dev/null 2>&1
hf upload $R /root/sb/run4/model_latest.pt sentbart/small_run4/model_latest.pt >/dev/null 2>&1
echo "RUN4B_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/run4bkeep.sh 2>&1 | tee -a /root/sb_run4b.log' > /dev/null 2>&1 < /dev/null &
  echo "RUN4B_LAUNCHED $(date -u)"
fi
# ---- run5 (queued 2026-10-02 04:45 JST): run4's page vector (page token alone, in-batch negatives) is far below the
# plain mean of the visible sentences on held-out page retrieval (top-1 0.29 vs 0.57 at 60k). run5 starts the page
# vector AT that mean and learns a correction (--page-res, head zero-initialised, as --prev-skip did for the decoder),
# and makes the page loss harder with the last 64 batches' pages as extra negatives (--page-queue 64, ~2000 pages).
# Same data / held-out / size / steps as run4, after it.
if ! pgrep -f "run5kee[p].sh" >/dev/null && ! grep -q "RUN5_JOB_DONE\|RUN5_ABORT" /root/sb_run5.log 2>/dev/null; then
  cat > /root/run5keep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb
until grep -qE "RUN4B_JOB_DONE|RUN4B_ABORT" /root/sb_run4b.log 2>/dev/null; do sleep 120; done
while pgrep -f "python3 /root/sb/" >/dev/null; do sleep 30; done
echo "[run5] train start $(date -u +%H:%M)"
( while sleep 1800; do hf upload $R /root/sb/run5/train.log sentbart/small_run5/train.log >/dev/null 2>&1; done ) & UP=$!
python3 /root/sb/train.py --data $D --out /root/sb/run5 --eval-shard 001 --steps 200000 --batch 32 --seq 128 --d 512 --layers 4 --heads 8 --ffn 2048 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-res 1 --page-queue 64 2>&1 | grep -E "^\[data\]|^\[model\]|^\[eval|TRAIN_DONE|Error|Traceback"
kill $UP 2>/dev/null
hf upload $R /root/sb/run5/train.log sentbart/small_run5/train.log >/dev/null 2>&1
hf upload $R /root/sb/run5/model_latest.pt sentbart/small_run5/model_latest.pt >/dev/null 2>&1
echo "RUN5_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/run5keep.sh 2>&1 | tee -a /root/sb_run5.log' > /dev/null 2>&1 < /dev/null &
  echo "RUN5_LAUNCHED $(date -u)"
fi
# 2026-10-02 09:00 JST: run5 stops at its 40k+ checkpoint (it answered its question: the residual page beats the mean,
# top-10 81.8 vs 78.8) so the model-only page vectors (run6a/6b, the user's priority) start now instead of at 12:40.
if [ ! -e /root/.run5_stop ]; then touch /root/.run5_stop
  pkill -f "run5kee[p].sh"; pkill -f "python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/run5"; sleep 10
  hf upload $R /root/sb/run5/train.log sentbart/small_run5/train.log >/dev/null 2>&1
  hf upload $R /root/sb/run5/model_latest.pt sentbart/small_run5/model_latest.pt >/dev/null 2>&1
  echo "RUN5_ABORT stopped early for run6 (model-only page vectors first) $(date -u)" >> /root/sb_run5.log; echo "RUN5_STOPPED $(date -u)"
fi
# ---- run6a / run6b (queued 2026-10-02 09:10 JST, the user: a page vector without the input mean would be more general
# and the hidden states more essential): run5 changed two things at once (start from the mean, 64 batches of
# negatives), so the model-only page vector gets the harder negatives too. 6a: the page token alone + the queue
# (run4 + the queue: is the queue what run4 lacked?). 6b: the mean of the encoder's outputs (contextual states, not
# the input vectors) through page_head + the queue. 100k steps each, after run5, compared at equal steps.
if ! pgrep -f "run6kee[p].sh" >/dev/null && ! grep -q "RUN6_JOB_DONE\|RUN6_ABORT" /root/sb_run6.log 2>/dev/null; then
  cat > /root/run6keep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb
until grep -qE "RUN5_JOB_DONE|RUN5_ABORT" /root/sb_run5.log 2>/dev/null; do sleep 120; done
while pgrep -f "python3 /root/sb/" >/dev/null; do sleep 30; done
for v in "6a:token" "6b:mean"; do n=${v%%:*}; pp=${v#*:}
  echo "[run$n] train start (page-pool $pp) $(date -u +%H:%M)"
  ( while sleep 1800; do hf upload $R /root/sb/run$n/train.log sentbart/small_run$n/train.log >/dev/null 2>&1; done ) & UP=$!
  python3 /root/sb/train.py --data $D --out /root/sb/run$n --eval-shard 001 --steps 100000 --batch 32 --seq 128 --d 512 --layers 4 --heads 8 --ffn 2048 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-pool $pp --page-queue 64 2>&1 | grep -E "^\[data\]|^\[model\]|^\[eval|TRAIN_DONE|Error|Traceback"
  kill $UP 2>/dev/null
  hf upload $R /root/sb/run$n/train.log sentbart/small_run$n/train.log >/dev/null 2>&1
  hf upload $R /root/sb/run$n/model_latest.pt sentbart/small_run$n/model_latest.pt >/dev/null 2>&1
done
echo "RUN6_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/run6keep.sh 2>&1 | tee -a /root/sb_run6.log' > /dev/null 2>&1 < /dev/null &
  echo "RUN6_LAUNCHED $(date -u)"
fi
# ---- run4c / run4m (2026-10-02 09:10 JST, the user: "why not just continue the one that has been learning from the
# start?"): instead of run6a/6b from scratch, both start from run4's 200k weights (page top-1 39.0%, still rising when
# its schedule ran out) with a fresh schedule and the harder negatives (64 batches of pages). 4c: the page token, as
# run4. 4m: the mean of the encoder's outputs through page_head. No input mean in either. 100k steps each.
if [ ! -e /root/.run6_swap ]; then touch /root/.run6_swap
  pkill -f "run6kee[p].sh"; pkill -f "python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/run6"; sleep 10
  echo "RUN6_ABORT replaced by run4c/run4m (continue run4) $(date -u)" >> /root/sb_run6.log; echo "RUN6_REPLACED $(date -u)"
fi
if ! pgrep -f "run4ckee[p].sh" >/dev/null && ! grep -q "RUN4C_JOB_DONE\|RUN4C_ABORT" /root/sb_run4c.log 2>/dev/null; then
  cat > /root/run4ckeep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb
while pgrep -f "python3 /root/sb/" >/dev/null; do sleep 30; done
[ -s /root/sb/run4/model_latest.pt ] || { echo "RUN4C_ABORT no run4 weights"; exit 1; }
for v in "4c:token" "4m:mean"; do n=${v%%:*}; pp=${v#*:}
  echo "[run$n] train start from run4 200k (page-pool $pp) $(date -u +%H:%M)"
  ( while sleep 1800; do hf upload $R /root/sb/run$n/train.log sentbart/small_run$n/train.log >/dev/null 2>&1; done ) & UP=$!
  python3 /root/sb/train.py --data $D --out /root/sb/run$n --init /root/sb/run4/model_latest.pt --eval-shard 001 --steps 100000 --batch 32 --seq 128 --d 512 --layers 4 --heads 8 --ffn 2048 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-pool $pp --page-queue 64 2>&1 | grep -E "^\[data\]|^\[model\]|^\[init\]|^\[eval|TRAIN_DONE|Error|Traceback"
  kill $UP 2>/dev/null
  hf upload $R /root/sb/run$n/train.log sentbart/small_run$n/train.log >/dev/null 2>&1
  hf upload $R /root/sb/run$n/model_latest.pt sentbart/small_run$n/model_latest.pt >/dev/null 2>&1
done
echo "RUN4C_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/run4ckeep.sh 2>&1 | tee -a /root/sb_run4c.log' > /dev/null 2>&1 < /dev/null &
  echo "RUN4C_LAUNCHED $(date -u)"
fi
# ---- page recall curve (2026-10-02 09:20 JST, the user: "at which top-k does it reach 90%?"): run4 200k and run5 40k
# evaluated once on the same 2000 held-out pages, recall at 1..200 and the k that holds 90%, beside the training run
if [ ! -e /root/.curve_v1 ]; then touch /root/.curve_v1
  ( cd /root/sb; C="--data /root/sb/data/docs --eval-shard 001 --eval-only 1 --batch 32 --seq 128 --d 512 --layers 4 --heads 8 --ffn 2048 --page 1"
    python3 /root/sb/train.py $C --out /root/sb/ev4 --init /root/sb/run4/model_latest.pt 2>&1 | grep -E "^\[curve|^\[init|Error|Traceback" | sed 's/^/[run4 200k] /'
    python3 /root/sb/train.py $C --out /root/sb/ev5 --init /root/sb/run5/model_latest.pt --page-res 1 2>&1 | grep -E "^\[curve|^\[init|Error|Traceback" | sed 's/^/[run5 40k] /'
  ) > /root/sb_curve.log 2>&1
  cat /root/sb_curve.log; hf upload $R /root/sb_curve.log sentbart/audit/page_curve.log >/dev/null 2>&1; echo "CURVE_DONE $(date -u)"
fi
echo "BOXI_OK serial $BOXI_SERIAL $(date -u)"
# CTL-END
