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
BOXI_SERIAL=32
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
for f in prep.py embed.py train.py evalsb.py links.py; do
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
# ---- 2026-10-02 11:10 JST (the user: "train it? train what?"): the mean of the encoder's outputs can be measured on
# run4c's weights with no training at all. When run4c is done, run4m is stopped before it trains, and run4c is
# evaluated three ways with no new training: the page token (as trained), the mean of the encoder's outputs through
# page_head, and the mean of the encoder's own sentence reconstructions (enc_head, already in the sentence space).
# Then run4c simply continues (run4c2, page token, another 100k from its weights) to keep the card busy.
if ! pgrep -f "pool3kee[p].sh" >/dev/null && ! grep -q "POOL3_JOB_DONE" /root/sb_pool3.log 2>/dev/null; then
  cat > /root/pool3keep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb
until grep -q "\[run4m\] train start" /root/sb_run4c.log 2>/dev/null; do sleep 30; done
sleep 5; pkill -f "run4ckee[p].sh"; pkill -f "python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/run4m"; sleep 10
echo "[pool3] run4m stopped before training; run4c $(python3 -c "import torch;print(torch.load('/root/sb/run4c/model_latest.pt',map_location='cpu')['step'])") evaluated three ways $(date -u +%H:%M)"
C="--data $D --eval-shard 001 --eval-only 1 --batch 32 --seq 128 --d 512 --layers 4 --heads 8 --ffn 2048 --page 1 --init /root/sb/run4c/model_latest.pt"
for pp in token mean encmean; do python3 /root/sb/train.py $C --out /root/sb/ev_$pp --page-pool $pp 2>&1 | grep -E "^\[curve\]|Error|Traceback" | sed "s/^/[$pp] /"; done
hf upload $R /root/sb_pool3.log sentbart/audit/pool3.log >/dev/null 2>&1
echo "[run4c2] train start $(date -u +%H:%M)"
( while sleep 1800; do hf upload $R /root/sb/run4c2/train.log sentbart/small_run4c2/train.log >/dev/null 2>&1; done ) & UP=$!
python3 /root/sb/train.py --data $D --out /root/sb/run4c2 --init /root/sb/run4c/model_latest.pt --eval-shard 001 --steps 100000 --batch 32 --seq 128 --d 512 --layers 4 --heads 8 --ffn 2048 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-queue 64 2>&1 | grep -E "^\[init\]|^\[eval|^\[curve|TRAIN_DONE|Error|Traceback"
kill $UP 2>/dev/null; hf upload $R /root/sb/run4c2/train.log sentbart/small_run4c2/train.log >/dev/null 2>&1; hf upload $R /root/sb/run4c2/model_latest.pt sentbart/small_run4c2/model_latest.pt >/dev/null 2>&1
echo "POOL3_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/pool3keep.sh 2>&1 | tee -a /root/sb_pool3.log' > /dev/null 2>&1 < /dev/null &
  echo "POOL3_LAUNCHED $(date -u)"
fi
# ---- train until it stops improving (2026-10-02 12:00 JST, the user: "keep training until the gains run out"): after
# run4c2, 100k-step segments continue from the latest weights (fresh schedule, same settings) for as long as a
# segment adds at least 1 point of page top-1 over the segment before it; at most 6 segments. PLATEAU when it stops.
if ! pgrep -f "chainkee[p].sh" >/dev/null && ! grep -q "CHAIN_JOB_DONE" /root/sb_chain.log 2>/dev/null; then
  cat > /root/chainkeep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb
until grep -q "POOL3_JOB_DONE" /root/sb_pool3.log 2>/dev/null; do sleep 60; done
last() { grep "^\[eval" /root/sb/$1/train.log | tail -1 | sed -E 's/.*page_top1 ([0-9.]+).*/\1/'; }
prev=run4c2; base=0.493; cur=$(last run4c2); echo "[chain] run4c -> run4c2: page top-1 $base -> $cur"
for i in 3 4 5 6 7 8; do
  python3 -c "import sys; sys.exit(0 if float('$cur') - float('$base') >= 0.01 else 1)" || { echo "PLATEAU after $prev: top-1 $base -> $cur (< 1 point per 100k)"; break; }
  n=run4c$i; echo "[chain] $n from $prev (top-1 $cur) $(date -u +%H:%M)"
  ( while sleep 1800; do hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; done ) & UP=$!
  python3 /root/sb/train.py --data $D --out /root/sb/$n --init /root/sb/$prev/model_latest.pt --eval-shard 001 --steps 100000 --batch 32 --seq 128 --d 512 --layers 4 --heads 8 --ffn 2048 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-queue 64 2>&1 | grep -E "^\[eval|^\[curve|TRAIN_DONE|Error|Traceback" | tail -3
  kill $UP 2>/dev/null; hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; hf upload $R /root/sb/$n/model_latest.pt sentbart/small_$n/model_latest.pt >/dev/null 2>&1
  rm -f /root/sb/$prev/state.pt
  base=$cur; cur=$(last $n); prev=$n; echo "[chain] $n done: page top-1 $base -> $cur"
done
echo "CHAIN_JOB_DONE best $prev $(date -u)"
RK
  setsid nohup bash -c 'bash /root/chainkeep.sh 2>&1 | tee -a /root/sb_chain.log' > /dev/null 2>&1 < /dev/null &
  echo "CHAIN_LAUNCHED $(date -u)"
fi
# 2026-10-02 15:05 JST: the run4c keeper was relaunched by later control runs (its job marker never got written after
# pool3 stopped it) and started run4m from scratch, holding the card while chain2 waited. Mark it done, stop run4m.
if [ ! -e /root/.run4m_stop ]; then touch /root/.run4m_stop
  echo "RUN4C_ABORT run4m not wanted (pool3 measured the pooled variants without training) $(date -u)" >> /root/sb_run4c.log
  pkill -f "run4ckee[p].sh"; pkill -f "python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/run4m"; echo "RUN4M_STOPPED $(date -u)"
fi
# ---- chain2 (2026-10-02 14:25 JST): run4c2 diverged at ~40k (gradient norm 1 -> 50 -> 1e10; page top-1 48.3 -> 16.6),
# so "PLATEAU" above was a blow-up, not a plateau. Restart from run4c's final weights (top-1 49.3) at a third of the
# learning rate (1e-4: these weights have been through three warm restarts at 3e-4), skip any step whose gradient
# norm exceeds 5 (healthy steps sit at 0.8-1.2), keep the best weights by held-out top-1, and continue in 100k
# segments from the best weights while a segment adds at least a point; at most 6.
if ! pgrep -f "chain2kee[p].sh" >/dev/null && ! grep -q "CHAIN2_JOB_DONE" /root/sb_chain2.log 2>/dev/null; then
  cat > /root/chain2keep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb
while pgrep -f "python3 /root/sb/" >/dev/null; do sleep 30; done
best() { grep "^\[best\]" /root/sb/$1/train.log 2>/dev/null | tail -1 | sed -E 's/.*page_top1 ([0-9.]+).*/\1/'; }
prev=/root/sb/run4c/model_latest.pt; base=0.493
for i in 1 2 3 4 5 6; do
  n=run4d$i; echo "[chain2] $n from $(basename $(dirname $prev))/$(basename $prev) (top-1 $base) $(date -u +%H:%M)"
  ( while sleep 1800; do hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; done ) & UP=$!
  python3 /root/sb/train.py --data $D --out /root/sb/$n --init $prev --eval-shard 001 --steps 100000 --batch 32 --seq 128 --d 512 --layers 4 --heads 8 --ffn 2048 --lr 1e-4 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-queue 64 --skip-grad 5 2>&1 | grep -E "^\[best|TRAIN_DONE|Error|Traceback" | tail -4
  kill $UP 2>/dev/null; hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; hf upload $R /root/sb/$n/model_best.pt sentbart/small_$n/model_best.pt >/dev/null 2>&1
  rm -f /root/sb/$n/state.pt
  cur=$(best $n); cur=${cur:-0}; echo "[chain2] $n done: best page top-1 $cur (from $base)"
  python3 -c "import sys; sys.exit(0 if float('$cur') - float('$base') >= 0.01 else 1)" || { echo "PLATEAU2 at $n: best $cur vs $base"; break; }
  prev=/root/sb/$n/model_best.pt; base=$cur
done
echo "CHAIN2_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/chain2keep.sh 2>&1 | tee -a /root/sb_chain2.log' > /dev/null 2>&1 < /dev/null &
  echo "CHAIN2_LAUNCHED $(date -u)"
fi
# ---- article search (2026-10-02 17:00 JST, the user: measure article search before making it bigger): 500 held-out
# documents (shard 001), a 3-sentence passage from each (not the lead) goes to the hub; box G has DeepSeek write two
# questions per passage (natural, and one that avoids the article's name); then search over ALL ~118k documents of the
# shard: the page vector (best weights so far, whole documents, nothing masked) against the mean of the sentence
# vectors, the lead sentence, the best single sentence, and page + mean. Runs beside the training (small GPU share).
if ! pgrep -f "searchkee[p].sh" >/dev/null && ! grep -q "SEARCH_JOB_DONE" /root/sb_search.log 2>/dev/null; then
  cat > /root/searchkeep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb; mkdir -p /root/sb/se
python3 - <<'PY2'
import json, random
rows = []
for i, l in enumerate(open("/root/sb/data/docs/docs_001.jsonl")):
    d = json.loads(l); rows.append((i, d["title"], d["sents"]))
rng = random.Random(7); pick = rng.sample([r for r in rows if len(r[2]) >= 8], 500); out = []
for i, t, ss in pick:
    a = rng.randint(1, len(ss) - 3); out.append({"idx": i, "title": t, "passage": " ".join(ss[a:a + 3])})
open("/root/sb/se/docs500.jsonl", "w").write("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out)); print("[search] 500 passages from", len(rows), "held-out documents")
PY2
hf upload $R /root/sb/se/docs500.jsonl sentbart/searcheval/docs500.jsonl >/dev/null 2>&1
until hf download $R sentbart/searcheval/queries.jsonl --local-dir /root/sb/se/dl >/dev/null 2>&1 && [ -s /root/sb/se/dl/sentbart/searcheval/queries.jsonl ]; do sleep 120; done
echo "[search] $(wc -l < /root/sb/se/dl/sentbart/searcheval/queries.jsonl) queries $(date -u +%H:%M)"
CK=$(ls -t /root/sb/run4d*/model_best.pt 2>/dev/null | head -1); CK=${CK:-/root/sb/run4c/model_latest.pt}; echo "[search] weights $CK"
python3 /root/sb/train.py --data $D --out /root/sb/se/run --eval-shard 001 --batch 32 --seq 128 --d 512 --layers 4 --heads 8 --ffn 2048 --page 1 --init $CK \
  --search-eval /root/sb/se/dl/sentbart/searcheval/queries.jsonl --search-out /root/sb/se/result.json 2>&1 | grep -E "^\[search|^\[init|Error|Traceback"
hf upload $R /root/sb/se/result.json sentbart/searcheval/result.json >/dev/null 2>&1; hf upload $R /root/sb_search.log sentbart/searcheval/search.log >/dev/null 2>&1
echo "SEARCH_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/searchkeep.sh 2>&1 | tee -a /root/sb_search.log' > /dev/null 2>&1 < /dev/null &
  echo "SEARCH_LAUNCHED $(date -u)"
fi
# ---- search2 (2026-10-02 17:50 JST, the user: the real input is the query the app model itself writes in <search>):
# the held-out shard's titles go to the hub; box G collects (query the model wrote -> page it was served) pairs from its
# rollouts whose page is one of these documents; then the same search over all ~118k held-out documents.
if ! pgrep -f "search2kee[p].sh" >/dev/null && ! grep -q "SEARCH2_JOB_DONE" /root/sb_search2.log 2>/dev/null; then
  cat > /root/search2keep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb; mkdir -p /root/sb/se2
python3 -c "
import json
open('/root/sb/se2/titles001.txt','w').write(''.join(json.loads(l)['title'].replace('\n',' ')+'\n' for l in open('/root/sb/data/docs/docs_001.jsonl')))"
hf upload $R /root/sb/se2/titles001.txt sentbart/searcheval/titles001.txt >/dev/null 2>&1; echo "[search2] $(wc -l < /root/sb/se2/titles001.txt) titles up"
until hf download $R sentbart/searcheval/dcq.jsonl --local-dir /root/sb/se2/dl >/dev/null 2>&1 && [ -s /root/sb/se2/dl/sentbart/searcheval/dcq.jsonl ]; do sleep 120; done
echo "[search2] $(wc -l < /root/sb/se2/dl/sentbart/searcheval/dcq.jsonl) app-model queries $(date -u +%H:%M)"
while pgrep -f "search-eval" >/dev/null; do sleep 30; done
CK=$(ls -t /root/sb/run4d*/model_best.pt 2>/dev/null | head -1); echo "[search2] weights $CK"
python3 /root/sb/train.py --data $D --out /root/sb/se2/run --eval-shard 001 --batch 32 --seq 128 --d 512 --layers 4 --heads 8 --ffn 2048 --page 1 --init $CK \
  --search-eval /root/sb/se2/dl/sentbart/searcheval/dcq.jsonl --search-out /root/sb/se2/result.json 2>&1 | grep -E "^\[search|Error|Traceback"
hf upload $R /root/sb/se2/result.json sentbart/searcheval/result_dcq.json >/dev/null 2>&1; hf upload $R /root/sb_search2.log sentbart/searcheval/search2.log >/dev/null 2>&1
echo "SEARCH2_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/search2keep.sh 2>&1 | tee -a /root/sb_search2.log' > /dev/null 2>&1 < /dev/null &
  echo "SEARCH2_LAUNCHED $(date -u)"
fi
# ---- grow (2026-10-02 18:40 JST, the user: if it has topped out, add layers rather than start over): chain2 stops; the
# best weights so far (by held-out page top-1, any run4d*) grow from 4+4 to 8+8 layers - each old layer keeps its place,
# a new identity-initialised layer after each (the grown model computes exactly what the old one did, checked) - and
# training continues at lr 1e-4 with the skip guard, in 100k segments from the best weights while a segment adds a point.
if [ ! -e /root/.grow_v1 ]; then touch /root/.grow_v1
  echo "CHAIN2_JOB_DONE stopped for grow $(date -u)" >> /root/sb_chain2.log
  pkill -f "chain2kee[p].sh"; pkill -f "python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/run4d"; sleep 10; echo "CHAIN2_STOPPED $(date -u)"
fi
if ! pgrep -f "growkee[p].sh" >/dev/null && ! grep -q "GROW_JOB_DONE" /root/sb_grow.log 2>/dev/null; then
  cat > /root/growkeep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb
while pgrep -f "python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/run" >/dev/null; do sleep 30; done
bestof() { grep -h "^\[best\]" /root/sb/$1/train.log 2>/dev/null | tail -1 | sed -E 's/.*page_top1 ([0-9.]+).*/\1/'; }
src=""; base=0
for d in /root/sb/run4d*; do b=$(bestof $(basename $d)); [ -n "$b" ] && [ -s $d/model_best.pt ] && python3 -c "import sys; sys.exit(0 if float('$b') > float('$base') else 1)" && { src=$d/model_best.pt; base=$b; }; done
[ -n "$src" ] || { echo "GROW_JOB_DONE no source"; exit 1; }
for d in /root/sb/run4d*; do hf upload $R $d/train.log sentbart/small_$(basename $d)/train.log >/dev/null 2>&1; done
hf upload $R $src sentbart/small_grow_source.pt >/dev/null 2>&1
prev=""
for i in 1 2 3 4; do
  n=run4g$i; echo "[grow] $n $( [ -z "$prev" ] && echo "grown 4->8 from $src" || echo "from $prev" ) (top-1 $base) $(date -u +%H:%M)"
  SRCARG=$( [ -z "$prev" ] && echo "--grow $src" || echo "--init $prev" )
  ( while sleep 1800; do hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; done ) & UP=$!
  python3 /root/sb/train.py --data $D --out /root/sb/$n $SRCARG --eval-shard 001 --steps 100000 --batch 32 --seq 128 --d 512 --layers 8 --heads 8 --ffn 2048 --lr 1e-4 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-queue 64 --skip-grad 5 2>&1 | grep -E "^\[grow|^\[model|^\[best|TRAIN_DONE|Error|Traceback" | tail -6
  kill $UP 2>/dev/null; hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; hf upload $R /root/sb/$n/model_best.pt sentbart/small_$n/model_best.pt >/dev/null 2>&1
  rm -f /root/sb/$n/state.pt
  cur=$(bestof $n); cur=${cur:-0}; echo "[grow] $n done: best page top-1 $cur (from $base)"
  python3 -c "import sys; sys.exit(0 if float('$cur') - float('$base') >= 0.01 else 1)" || { echo "PLATEAU_GROW at $n: best $cur vs $base"; break; }
  prev=/root/sb/$n/model_best.pt; base=$cur
done
echo "GROW_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/growkeep.sh 2>&1 | tee -a /root/sb_grow.log' > /dev/null 2>&1 < /dev/null &
  echo "GROW_LAUNCHED $(date -u)"
fi
# ---- search3 (2026-10-02 18:50 JST, the user: candidates from this model x keyword search): the app model's own queries
# again, now with BM25 keyword search over the held-out shard (title + text) and the combinations - reciprocal-rank
# fusion of each vector method with BM25, and each vector method's top 100 ordered by BM25. 4+4 weights (run4d1 best).
if ! pgrep -f "search3kee[p].sh" >/dev/null && ! grep -q "SEARCH3_JOB_DONE" /root/sb_search3.log 2>/dev/null; then
  cat > /root/search3keep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb; mkdir -p /root/sb/se3
python3 /root/sb/train.py --data $D --out /root/sb/se3/run --eval-shard 001 --batch 32 --seq 128 --d 512 --layers 4 --heads 8 --ffn 2048 --page 1 --init /root/sb/run4d1/model_best.pt \
  --search-eval /root/sb/se2/dl/sentbart/searcheval/dcq.jsonl --search-text $D/docs_001.jsonl --search-out /root/sb/se3/result.json 2>&1 | grep -E "^\[search|Error|Traceback"
hf upload $R /root/sb/se3/result.json sentbart/searcheval/result_hybrid.json >/dev/null 2>&1; hf upload $R /root/sb_search3.log sentbart/searcheval/search3.log >/dev/null 2>&1
echo "SEARCH3_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/search3keep.sh 2>&1 | tee -a /root/sb_search3.log' > /dev/null 2>&1 < /dev/null &
  echo "SEARCH3_LAUNCHED $(date -u)"
fi
# ---- search4 (2026-10-02 19:00 JST): the same as search3 with the queries embedded WITHOUT bge's query instruction - the
# page vector was trained against plain sentence vectors, the instruction may have put the queries elsewhere
if ! pgrep -f "search4kee[p].sh" >/dev/null && ! grep -q "SEARCH4_JOB_DONE" /root/sb_search4.log 2>/dev/null; then
  cat > /root/search4keep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb; mkdir -p /root/sb/se4
until grep -q "SEARCH3_JOB_DONE" /root/sb_search3.log 2>/dev/null; do sleep 60; done
python3 /root/sb/train.py --data $D --out /root/sb/se4/run --eval-shard 001 --batch 32 --seq 128 --d 512 --layers 4 --heads 8 --ffn 2048 --page 1 --init /root/sb/run4d1/model_best.pt --q-prefix 0 \
  --search-eval /root/sb/se2/dl/sentbart/searcheval/dcq.jsonl --search-text $D/docs_001.jsonl --search-out /root/sb/se4/result.json 2>&1 | grep -E "^\[search|Error|Traceback"
hf upload $R /root/sb/se4/result.json sentbart/searcheval/result_hybrid_noprefix.json >/dev/null 2>&1; hf upload $R /root/sb_search4.log sentbart/searcheval/search4.log >/dev/null 2>&1
echo "SEARCH4_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/search4keep.sh 2>&1 | tee -a /root/sb_search4.log' > /dev/null 2>&1 < /dev/null &
  echo "SEARCH4_LAUNCHED $(date -u)"
fi
# ---- search5 (2026-10-02 19:15 JST): search3 printed nothing past the BM25 line (no error caught - the grown model's
# training holds 5.5 GB, the ~20 [queries x 118k] matrices did not fit beside it); the search now runs 64 queries at a
# time. With and without bge's query instruction, full output kept.
if [ ! -e /root/.search5_v1 ]; then touch /root/.search5_v1; pkill -f "search4kee[p].sh"; pkill -f "search-eval"; echo "SEARCH4_JOB_DONE replaced by search5" >> /root/sb_search4.log; fi
if ! pgrep -f "search5kee[p].sh" >/dev/null && ! grep -q "SEARCH5_JOB_DONE" /root/sb_search5.log 2>/dev/null; then
  cat > /root/search5keep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb; mkdir -p /root/sb/se5
for px in 1 0; do
  python3 /root/sb/train.py --data $D --out /root/sb/se5/run$px --eval-shard 001 --batch 32 --seq 128 --d 512 --layers 4 --heads 8 --ffn 2048 --page 1 --init /root/sb/run4d1/model_best.pt --q-prefix $px \
    --search-eval /root/sb/se2/dl/sentbart/searcheval/dcq.jsonl --search-text $D/docs_001.jsonl --search-out /root/sb/se5/result_p$px.json > /root/sb/se5/full_p$px.log 2>&1
  echo "== query instruction $px"; grep -E "^\[search" /root/sb/se5/full_p$px.log; tail -3 /root/sb/se5/full_p$px.log | grep -vE "^\[search" | cut -c1-300
  hf upload $R /root/sb/se5/result_p$px.json sentbart/searcheval/result_hybrid_p$px.json >/dev/null 2>&1
done
hf upload $R /root/sb_search5.log sentbart/searcheval/search5.log >/dev/null 2>&1
echo "SEARCH5_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/search5keep.sh 2>&1 | tee -a /root/sb_search5.log' > /dev/null 2>&1 < /dev/null &
  echo "SEARCH5_LAUNCHED $(date -u)"
fi
# ---- 2026-10-02 19:20 JST (the user: training first; the query instruction stays): search5 stops; the hybrid search runs
# once, with the instruction, on the CPU (SB_DEV=cpu) so the card stays with the grown model's training
if [ ! -e /root/.search6_v1 ]; then touch /root/.search6_v1; pkill -f "search5kee[p].sh"; pkill -f "search-eval"; echo "SEARCH5_JOB_DONE replaced by search6 (cpu)" >> /root/sb_search5.log; fi
if ! pgrep -f "search6kee[p].sh" >/dev/null && ! grep -q "SEARCH6_JOB_DONE" /root/sb_search6.log 2>/dev/null; then
  cat > /root/search6keep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb; mkdir -p /root/sb/se6
SB_DEV=cpu OMP_NUM_THREADS=4 nice -n 10 python3 /root/sb/train.py --data $D --out /root/sb/se6/run --eval-shard 001 --batch 32 --seq 128 --d 512 --layers 4 --heads 8 --ffn 2048 --page 1 --init /root/sb/run4d1/model_best.pt \
  --search-eval /root/sb/se2/dl/sentbart/searcheval/dcq.jsonl --search-text $D/docs_001.jsonl --search-out /root/sb/se6/result.json > /root/sb/se6/full.log 2>&1
grep -E "^\[search" /root/sb/se6/full.log; tail -3 /root/sb/se6/full.log | grep -vE "^\[search" | cut -c1-300
hf upload $R /root/sb/se6/result.json sentbart/searcheval/result_hybrid.json >/dev/null 2>&1; hf upload $R /root/sb_search6.log sentbart/searcheval/search6.log >/dev/null 2>&1
echo "SEARCH6_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/search6keep.sh 2>&1 | tee -a /root/sb_search6.log' > /dev/null 2>&1 < /dev/null &
  echo "SEARCH6_LAUNCHED $(date -u)"
fi
# 2026-10-02 19:25 JST (the user: the 2000-page measure is enough): the CPU search test is cancelled too
if [ ! -e /root/.search6_cancel ]; then touch /root/.search6_cancel; echo "SEARCH6_JOB_DONE cancelled" >> /root/sb_search6.log; pkill -f "search6kee[p].sh"; pkill -f "search-eval"; echo "SEARCH6_CANCELLED $(date -u)"; fi
# ---- the results so far on the hub (2026-10-02 19:35 JST, the user): run4d2's best weights and the README (docs/sentbart.md)
if [ ! -e /root/.hub_v1 ]; then touch /root/.hub_v1
  [ -s /root/sb/run4d2/model_best.pt ] && hf upload $R /root/sb/run4d2/model_best.pt sentbart/small_run4d2/model_best.pt >/dev/null 2>&1
  curl -sSf -o /root/sb/README_sentbart.md "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/docs/sentbart.md?$(date +%s)" && hf upload $R /root/sb/README_sentbart.md sentbart/README.md >/dev/null 2>&1
  echo "HUB_UP $(date -u)"
fi
# ---- links (2026-10-03 00:50 JST, the user: plain contrast pushes related pages out as hard as unrelated ones - use
# the link graph to set how near they should be). Now, on the CPU beside the training: the training shards' page ids
# again (prep.py --ids-only, same filter and order as their vectors), Wikipedia's ordered body links
# (cgscsystems/wikipedia-ordered-links, nlink_sequences: 18M pages; checked: 399/400 of our page ids are in it), the
# link graph between the training documents (links.py). When run4g2 ends (~04:00) the grow chain stops and training
# continues from the best 8+8 weights with link-paired batches and soft targets (a linked page --link-w 0.2, halved
# past an article's first 10 links); the negatives queue stays at 64 batches so only the links change.
if ! pgrep -f "linkkee[p].sh" >/dev/null && ! grep -q "LINK_JOB_DONE\|LINK_ABORT" /root/sb_link.log 2>/dev/null; then
  cat > /root/linkkeep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb
for k in $(ls $D/vec_*.npy | sed -E 's/.*vec_([0-9]+)\.npy/\1/'); do
  [ "$k" = 001 ] && continue; [ -s $D/ids_$k.npy ] && continue
  nice -n 10 python3 /root/sb/prep.py --out $D --shards $((10#$k))-$((10#$k)) --ids-only 1 2>&1 | grep -E "^\[prep\] shard|Error|Traceback"
done
rm -rf /root/.cache/huggingface/hub/datasets--wikimedia--wikipedia
python3 -c "
import numpy as np, glob, os
bad = [k for k in sorted(os.path.basename(f)[4:-4] for f in glob.glob('$D/vec_*.npy')) if k != '001' and len(np.load('$D/ids_%s.npy' % k)) != len(np.load('$D/off_%s.npy' % k)) - 1]
print('[link] id lists vs vectors:', 'all match' if not bad else 'MISMATCH ' + str(bad))" | tee /root/sb/ids_check.txt
grep -q "all match" /root/sb/ids_check.txt || { echo "LINK_ABORT ids"; exit 1; }
for t in 1 2 3; do hf download cgscsystems/wikipedia-ordered-links nlink_sequences.parquet --repo-type dataset --local-dir /root/sb/linkdata >/dev/null 2>&1 && break; sleep 30; done
nice -n 10 python3 /root/sb/links.py --data $D --eval-shard 001 --seq /root/sb/linkdata/nlink_sequences.parquet --out /root/sb/links.npz 2>&1 | grep -vE "row group [1-9][0-9]*/" | tail -4
[ -s /root/sb/links.npz ] || { echo "LINK_ABORT links"; exit 1; }
rm -rf /root/sb/linkdata; hf upload $R /root/sb_link.log sentbart/links/link.log >/dev/null 2>&1
echo "[link] waiting for run4g2 to end $(date -u +%H:%M)"
until grep -q "\[grow\] run4g2 done" /root/sb_grow.log 2>/dev/null; do sleep 20; done
pkill -f "growkee[p].sh"; sleep 2; pkill -f "python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/run4g3"; echo "GROW_JOB_DONE stopped for links" >> /root/sb_grow.log; sleep 10
bestof() { grep -h "^\[best\]" /root/sb/$1/train.log 2>/dev/null | tail -1 | sed -E 's/.*page_top1 ([0-9.]+).*/\1/'; }
src=""; base=0
for d in /root/sb/run4g1 /root/sb/run4g2; do b=$(bestof $(basename $d)); [ -n "$b" ] && [ -s $d/model_best.pt ] && python3 -c "import sys; sys.exit(0 if float('$b') > float('$base') else 1)" && { src=$d/model_best.pt; base=$b; }; done
hf upload $R /root/sb/run4g2/train.log sentbart/small_run4g2/train.log >/dev/null 2>&1; hf upload $R $src sentbart/small_link_source.pt >/dev/null 2>&1
prev=$src
for i in 1 2 3 4; do
  n=run4L$i; echo "[link] $n from $prev (top-1 $base) $(date -u +%H:%M)"
  ( while sleep 1800; do hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; done ) & UP=$!
  python3 /root/sb/train.py --data $D --out /root/sb/$n --init $prev --eval-shard 001 --steps 100000 --batch 32 --seq 128 --d 512 --layers 8 --heads 8 --ffn 2048 --lr 1e-4 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-queue 64 --skip-grad 5 --links /root/sb/links.npz --link-w 0.2 2>&1 | grep -E "^\[links\]|^\[best|TRAIN_DONE|Error|Traceback" | tail -6
  kill $UP 2>/dev/null; hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; hf upload $R /root/sb/$n/model_best.pt sentbart/small_$n/model_best.pt >/dev/null 2>&1
  rm -f /root/sb/$n/state.pt
  cur=$(bestof $n); cur=${cur:-0}; echo "[link] $n done: best page top-1 $cur (from $base)"
  python3 -c "import sys; sys.exit(0 if float('$cur') - float('$base') >= 0.01 else 1)" || { echo "PLATEAU_LINK at $n: best $cur vs $base"; break; }
  prev=/root/sb/$n/model_best.pt; base=$cur
done
echo "LINK_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/linkkeep.sh 2>&1 | tee -a /root/sb_link.log' > /dev/null 2>&1 < /dev/null &
  echo "LINK_LAUNCHED $(date -u)"
fi
# ---- 2026-10-03 04:50 JST (night check): run4L1 is coming apart - page top-1 54.5 -> 50.3 / 49.7 at 10k / 20k, gradient
# norm 1 -> 5-10 and 2000+ steps skipped by the guard by 20k (the run4c2 pattern). Linked articles share near-identical
# sentences, so half-linked batches also make the sentence-level losses fight over near-duplicates. Stop it; restart
# from the same start (run4g2 best, 54.5) gentler: lr 3e-5, a quarter of each batch linked, same soft targets (0.2).
if [ ! -e /root/.link_b ]; then touch /root/.link_b
  echo "LINK_JOB_DONE stopped run4L1 (unstable) $(date -u)" >> /root/sb_link.log
  pkill -f "linkkee[p].sh"; pkill -f "python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/run4L"; sleep 10
  hf upload $R /root/sb/run4L1/train.log sentbart/small_run4L1/train.log >/dev/null 2>&1; echo "RUN4L1_STOPPED $(date -u)"
fi
if ! pgrep -f "linkbkee[p].sh" >/dev/null && ! grep -q "LINKB_JOB_DONE" /root/sb_linkb.log 2>/dev/null; then
  cat > /root/linkbkeep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb
while pgrep -f "python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/run" >/dev/null; do sleep 20; done
n=run4L1b; echo "[linkb] $n from run4g2 best (top-1 0.545), lr 3e-5, link-frac 0.25 $(date -u +%H:%M)"
( while sleep 1800; do hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; done ) & UP=$!
python3 /root/sb/train.py --data $D --out /root/sb/$n --init /root/sb/run4g2/model_best.pt --eval-shard 001 --steps 100000 --batch 32 --seq 128 --d 512 --layers 8 --heads 8 --ffn 2048 --lr 3e-5 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-queue 64 --skip-grad 5 --links /root/sb/links.npz --link-w 0.2 --link-frac 0.25 2>&1 | grep -E "^\[links\]|^\[best|TRAIN_DONE|Error|Traceback" | tail -6
kill $UP 2>/dev/null; hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; hf upload $R /root/sb/$n/model_best.pt sentbart/small_$n/model_best.pt >/dev/null 2>&1
echo "LINKB_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/linkbkeep.sh 2>&1 | tee -a /root/sb_linkb.log' > /dev/null 2>&1 < /dev/null &
  echo "LINKB_LAUNCHED $(date -u)"
fi
# ---- 2026-10-03 07:45 JST (the user: links are a means, not the goal - go with plain negatives if they work better):
# run4L1b (soft link targets) held top-1 at 53.3-53.5 for 50k steps, below its 54.5 start; it stops. Plain negatives
# again from the best 8+8 weights (run4g2, 54.5), with 4x the queue (256 batches, ~8000 pages - the lever that gave
# +10 points before), lr 1e-4, skip guard, best kept; 100k segments while each adds a point.
if [ ! -e /root/.plainq ]; then touch /root/.plainq
  echo "LINKB_JOB_DONE stopped for plain negatives $(date -u)" >> /root/sb_linkb.log
  pkill -f "linkbkee[p].sh"; pkill -f "python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/run4L1b"; sleep 10
  hf upload $R /root/sb/run4L1b/train.log sentbart/small_run4L1b/train.log >/dev/null 2>&1; echo "RUN4L1B_STOPPED $(date -u)"
fi
if ! pgrep -f "plainqkee[p].sh" >/dev/null && ! grep -q "PLAINQ_JOB_DONE" /root/sb_plainq.log 2>/dev/null; then
  cat > /root/plainqkeep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb
while pgrep -f "python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/run" >/dev/null; do sleep 20; done
bestof() { grep -h "^\[best\]" /root/sb/$1/train.log 2>/dev/null | tail -1 | sed -E 's/.*page_top1 ([0-9.]+).*/\1/'; }
prev=/root/sb/run4g2/model_best.pt; base=0.545
for i in 1 2 3 4; do
  n=run4q$i; echo "[plainq] $n from $prev (top-1 $base), queue 256 $(date -u +%H:%M)"
  ( while sleep 1800; do hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; done ) & UP=$!
  python3 /root/sb/train.py --data $D --out /root/sb/$n --init $prev --eval-shard 001 --steps 100000 --batch 32 --seq 128 --d 512 --layers 8 --heads 8 --ffn 2048 --lr 1e-4 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-queue 256 --skip-grad 5 2>&1 | grep -E "^\[best|TRAIN_DONE|Error|Traceback" | tail -6
  kill $UP 2>/dev/null; hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; hf upload $R /root/sb/$n/model_best.pt sentbart/small_$n/model_best.pt >/dev/null 2>&1
  rm -f /root/sb/$n/state.pt
  cur=$(bestof $n); cur=${cur:-0}; echo "[plainq] $n done: best page top-1 $cur (from $base)"
  python3 -c "import sys; sys.exit(0 if float('$cur') - float('$base') >= 0.01 else 1)" || { echo "PLATEAU_PLAINQ at $n: best $cur vs $base"; break; }
  prev=/root/sb/$n/model_best.pt; base=$cur
done
echo "PLAINQ_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/plainqkeep.sh 2>&1 | tee -a /root/sb_plainq.log' > /dev/null 2>&1 < /dev/null &
  echo "PLAINQ_LAUNCHED $(date -u)"
fi
# ---- 2026-10-03 09:15 JST (the user: if it looks marginal at 30k, go back to 54.5 and keep doing what worked): the
# 8000-page queue went unstable (skipped steps 6 -> 827 between 22k and 33k, gradient ~7; top-1 52.5 at 30k). Stop
# it; continue from run4g2's best (54.5) exactly as run4g2 did (queue 64, lr 1e-4, skip guard), 100k segments while
# each adds a point.
if [ ! -e /root/.cont54 ]; then touch /root/.cont54
  echo "PLAINQ_JOB_DONE stopped (unstable) $(date -u)" >> /root/sb_plainq.log
  pkill -f "plainqkee[p].sh"; pkill -f "python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/run4q"; sleep 10
  hf upload $R /root/sb/run4q1/train.log sentbart/small_run4q1/train.log >/dev/null 2>&1; echo "RUN4Q1_STOPPED $(date -u)"
fi
if ! pgrep -f "cont54kee[p].sh" >/dev/null && ! grep -q "CONT54_JOB_DONE" /root/sb_cont54.log 2>/dev/null; then
  cat > /root/cont54keep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb
while pgrep -f "python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/run" >/dev/null; do sleep 20; done
bestof() { grep -h "^\[best\]" /root/sb/$1/train.log 2>/dev/null | tail -1 | sed -E 's/.*page_top1 ([0-9.]+).*/\1/'; }
prev=/root/sb/run4g2/model_best.pt; base=0.545
for i in 3 4 5 6; do
  n=run4g$i; rm -rf /root/sb/$n; echo "[cont54] $n from $prev (top-1 $base), queue 64 $(date -u +%H:%M)"
  ( while sleep 1800; do hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; done ) & UP=$!
  python3 /root/sb/train.py --data $D --out /root/sb/$n --init $prev --eval-shard 001 --steps 100000 --batch 32 --seq 128 --d 512 --layers 8 --heads 8 --ffn 2048 --lr 1e-4 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-queue 64 --skip-grad 5 2>&1 | grep -E "^\[best|TRAIN_DONE|Error|Traceback" | tail -6
  kill $UP 2>/dev/null; hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; hf upload $R /root/sb/$n/model_best.pt sentbart/small_$n/model_best.pt >/dev/null 2>&1
  rm -f /root/sb/$n/state.pt
  cur=$(bestof $n); cur=${cur:-0}; echo "[cont54] $n done: best page top-1 $cur (from $base)"
  python3 -c "import sys; sys.exit(0 if float('$cur') - float('$base') >= 0.01 else 1)" || { echo "PLATEAU_CONT54 at $n: best $cur vs $base"; break; }
  prev=/root/sb/$n/model_best.pt; base=$cur
done
echo "CONT54_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/cont54keep.sh 2>&1 | tee -a /root/sb_cont54.log' > /dev/null 2>&1 < /dev/null &
  echo "CONT54_LAUNCHED $(date -u)"
fi
# ---- 2026-10-03 11:50 JST: run4g3 (from 54.5 at lr 1e-4 again) never got back above 52.9 and went unstable after 25k
# (skipped steps 19 -> 10661 by 45k, gradient 8-20). run4g2's gains all came late, at low lr. Stop it; continue from
# run4g2's best at lr 3e-5 (queue 64, skip guard), 100k segments while each adds a point.
if [ ! -e /root/.cont54b ]; then touch /root/.cont54b
  echo "CONT54_JOB_DONE stopped (run4g3 unstable) $(date -u)" >> /root/sb_cont54.log
  pkill -f "cont54kee[p].sh"; pkill -f "python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/run4g3"; sleep 10
  hf upload $R /root/sb/run4g3/train.log sentbart/small_run4g3/train.log >/dev/null 2>&1; echo "RUN4G3_STOPPED $(date -u)"
fi
if ! pgrep -f "cont54bkee[p].sh" >/dev/null && ! grep -q "CONT54B_JOB_DONE" /root/sb_cont54b.log 2>/dev/null; then
  cat > /root/cont54bkeep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb
while pgrep -f "python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/run" >/dev/null; do sleep 20; done
bestof() { grep -h "^\[best\]" /root/sb/$1/train.log 2>/dev/null | tail -1 | sed -E 's/.*page_top1 ([0-9.]+).*/\1/'; }
prev=/root/sb/run4g2/model_best.pt; base=0.545
for i in 1 2 3 4; do
  n=run4h$i; rm -rf /root/sb/$n; echo "[cont54b] $n from $prev (top-1 $base), lr 3e-5 $(date -u +%H:%M)"
  ( while sleep 1800; do hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; done ) & UP=$!
  python3 /root/sb/train.py --data $D --out /root/sb/$n --init $prev --eval-shard 001 --steps 100000 --batch 32 --seq 128 --d 512 --layers 8 --heads 8 --ffn 2048 --lr 3e-5 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-queue 64 --skip-grad 5 2>&1 | grep -E "^\[best|TRAIN_DONE|Error|Traceback"
  kill $UP 2>/dev/null; hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; hf upload $R /root/sb/$n/model_best.pt sentbart/small_$n/model_best.pt >/dev/null 2>&1
  rm -f /root/sb/$n/state.pt
  cur=$(bestof $n); cur=${cur:-0}; echo "[cont54b] $n done: best page top-1 $cur (from $base)"
  python3 -c "import sys; sys.exit(0 if float('$cur') - float('$base') >= 0.01 else 1)" || { echo "PLATEAU_CONT54B at $n: best $cur vs $base"; break; }
  prev=/root/sb/$n/model_best.pt; base=$cur
done
echo "CONT54B_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/cont54bkeep.sh 2>&1 | tee -a /root/sb_cont54b.log' > /dev/null 2>&1 < /dev/null &
  echo "CONT54B_LAUNCHED $(date -u)"
fi
# ---- 2026-10-03 17:45 JST: run4h1 (8+8, lr 3e-5) ended at 55.1 (+0.6 over 54.5; the chain wants +1) and the card has
# been idle since. The last growth (4+4 -> 8+8) was worth ~3-4 points, and the user's server has room for ~120M: grow
# 8+8 -> 16+16 (identity-initialised layers in between) from run4h1's best. lr 5e-5 for the grown segment (1e-4
# blew up after 25k at 8+8), 3e-5 after; skip guard; chained while each 100k segment adds a point.
if ! pgrep -f "grow16kee[p].sh" >/dev/null && ! grep -q "GROW16_JOB_DONE" /root/sb_grow16.log 2>/dev/null; then
  cat > /root/grow16keep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb
while pgrep -f "python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/run" >/dev/null; do sleep 30; done
bestof() { grep -h "^\[best\]" /root/sb/$1/train.log 2>/dev/null | tail -1 | sed -E 's/.*page_top1 ([0-9.]+).*/\1/'; }
src=/root/sb/run4h1/model_best.pt; base=$(bestof run4h1); base=${base:-0.551}
[ -s $src ] || { echo "GROW16_JOB_DONE no source"; exit 1; }
prev=""
for i in 1 2 3 4; do
  n=run4k$i; LR=$( [ -z "$prev" ] && echo 5e-5 || echo 3e-5 )
  echo "[grow16] $n $( [ -z "$prev" ] && echo "grown 8->16 from $src" || echo "from $prev" ) (top-1 $base) lr $LR $(date -u +%H:%M)"
  SRCARG=$( [ -z "$prev" ] && echo "--grow $src" || echo "--init $prev" )
  rm -rf /root/sb/$n
  ( while sleep 1800; do hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; done ) & UP=$!
  python3 /root/sb/train.py --data $D --out /root/sb/$n $SRCARG --eval-shard 001 --steps 100000 --batch 32 --seq 128 --d 512 --layers 16 --heads 8 --ffn 2048 --lr $LR --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-queue 64 --skip-grad 5 2>&1 | grep -E "^\[grow|^\[model|^\[best|TRAIN_DONE|Error|Traceback"
  kill $UP 2>/dev/null; hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; hf upload $R /root/sb/$n/model_best.pt sentbart/small_$n/model_best.pt >/dev/null 2>&1
  rm -f /root/sb/$n/state.pt
  cur=$(bestof $n); cur=${cur:-0}; echo "[grow16] $n done: best page top-1 $cur (from $base)"
  python3 -c "import sys; sys.exit(0 if float('$cur') - float('$base') >= 0.01 else 1)" || { echo "PLATEAU_GROW16 at $n: best $cur vs $base"; break; }
  prev=/root/sb/$n/model_best.pt; base=$cur
done
echo "GROW16_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/grow16keep.sh 2>&1 | tee -a /root/sb_grow16.log' > /dev/null 2>&1 < /dev/null &
  echo "GROW16_LAUNCHED $(date -u)"
fi
# ---- 2026-10-03 18:05 JST (the user: save this result): run4h1's best (55.1) and the README on the hub again
if [ ! -e /root/.hub_v2 ]; then touch /root/.hub_v2
  [ -s /root/sb/run4h1/model_best.pt ] && hf upload $R /root/sb/run4h1/model_best.pt sentbart/small_run4h1/model_best.pt >/dev/null 2>&1 && echo "HUB_V2 weights up"
  curl -sSf -o /root/sb/README_sentbart.md "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/docs/sentbart.md?$(date +%s)" && hf upload $R /root/sb/README_sentbart.md sentbart/README.md >/dev/null 2>&1 && echo "HUB_V2 readme up"
  echo "HUB_V2 $(date -u)"
fi
# ---- 2026-10-03 23:15 JST: run4k1 (16+16) is stuck at ~54 with 10905 of 57800 steps skipped. The skip guard drops a
# step whose PRE-clip gradient norm is >= 5, but the step is clipped to norm 1 anyway; the norm drifts up to 5-11 as the
# contrast sharpens, so the guard ends up discarding most updates (run4g3 and run4q1 "went unstable" the same way: the
# losses never blew up). Restart the growth from run4h1's best with the guard only for real blow-ups (>= 100, nan/inf).
if [ ! -e /root/.grow16b ]; then touch /root/.grow16b
  echo "GROW16_JOB_DONE stopped (skip guard starved it) $(date -u)" >> /root/sb_grow16.log
  pkill -f "grow16kee[p].sh"; pkill -f "python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/run4k1"; sleep 10
  hf upload $R /root/sb/run4k1/train.log sentbart/small_run4k1/train.log >/dev/null 2>&1; echo "RUN4K1_STOPPED $(date -u)"
fi
if ! pgrep -f "grow16bkee[p].sh" >/dev/null && ! grep -q "GROW16B_JOB_DONE" /root/sb_grow16b.log 2>/dev/null; then
  sed -e 's/run4k\$i/run4n$i/; s/--skip-grad 5/--skip-grad 100/; s/\[grow16\]/[grow16b]/g; s/PLATEAU_GROW16/PLATEAU_GROW16B/; s/GROW16_JOB_DONE/GROW16B_JOB_DONE/g' /root/grow16keep.sh > /root/grow16bkeep.sh
  grep -q "run4n\$i" /root/grow16bkeep.sh && grep -q "skip-grad 100" /root/grow16bkeep.sh || { echo "GROW16B_JOB_DONE bad script" >> /root/sb_grow16b.log; }
  grep -q "GROW16B_JOB_DONE bad" /root/sb_grow16b.log 2>/dev/null || { setsid nohup bash -c 'bash /root/grow16bkeep.sh 2>&1 | tee -a /root/sb_grow16b.log' > /dev/null 2>&1 < /dev/null & echo "GROW16B_LAUNCHED $(date -u)"; }
fi
echo "BOXI_OK serial $BOXI_SERIAL $(date -u)"
# CTL-END
