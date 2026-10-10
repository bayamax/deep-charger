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
BOXI_SERIAL=80
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
for f in prep.py embed.py train.py train_e2e.py e2e_ladder.py search_cascade.py pick_docs.py evalsb.py links.py prep_dolphin.py; do
  curl -sS -L -o /root/sb/$f.new "$RAW/sentbart/$f?$(date +%s)" && grep -q "^#!/usr/bin/env python3" /root/sb/$f.new && mv /root/sb/$f.new /root/sb/$f || rm -f /root/sb/$f.new
done
ls /root/sb

# ---- the mirror: what this box is doing, on the hub every 10 minutes ----
if [ ! -e /root/.mirror_r2 ]; then touch /root/.mirror_r2; pkill -f "mirrorkee[p].sh"; sleep 1; fi
if [ ! -e /root/.mirror_r3 ]; then touch /root/.mirror_r3; pkill -f "mirrorkee[p].sh"; sleep 1; fi   # again for the pool fields
if [ ! -e /root/.mirror_r4 ]; then touch /root/.mirror_r4; pkill -f "mirrorkee[p].sh"; sleep 1; fi   # again for the rung records   # 2026-10-07: restarted once for the ladder2 lines
if ! pgrep -f "mirrorkee[p].sh" >/dev/null; then
  cat > /root/mirrorkeep.sh <<'MK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
while :; do
  { echo "=== boxI $(date -u) ==="; echo "--- ctl.log ---"; tail -n 60 /root/ctl.log 2>/dev/null | cut -c1-300
    for f in /root/sb_*.log; do [ -e $f ] && { echo "--- $f ---"; tail -n 25 $f | cut -c1-300; }; done
    echo "--- gpu ---"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
    echo "--- disk ---"; df -h /root | tail -1; du -sh /root/sb/data 2>/dev/null
    echo "--- ladder2 / dolphin ladder ---"; ls -t /root/sb/ladder2 /root/sb/dolphin_ladder 2>/dev/null | head -12 | tr '\n' ' '; echo
    f=$(ls -t /root/sb/ladder2/*.out /root/sb/dolphin_ladder/*.out 2>/dev/null | head -1); [ -n "$f" ] && { echo "== $f"; grep -E "^\[eval|^\[best|^\[plateau|^\[init\]|Error|Traceback" $f | tail -8 | cut -c1-300; echo "pool fields:"; grep -E "^\[eval" $f | tail -6 | sed -E 's/^(\[eval [0-9]+\]).*page_top1 ([0-9.]+).*(pool_top1.*)$/\1 2000: \2 | \3/' | cut -c1-200; grep "^\[step" $f | tail -1 | cut -c1-200; }
    [ -s /root/sb/dolphin_e2e/train.log ] && { echo "--- dolphin e2e (train.log) ---"; grep -E "^\[eval|^\[best|^\[plateau|^\[init\]|^\[e2e\]|Error|Traceback" /root/sb/dolphin_e2e/train.log | tail -8 | sed -E 's/^(\[eval [0-9]+\]).*page_top1 ([0-9.]+).*page_base_top1 ([0-9.]+).*(next_top1 [0-9.]+).*(pool_top1 [0-9.]+ pool_top10 [0-9.]+).*$/\1 2000: \2 (base \3) \4 \5/' | cut -c1-300; grep "^\[step" /root/sb/dolphin_e2e/train.log | tail -1 | cut -c1-200; }
    echo "--- ladder2 rungs (ladder.json) and dev guards ---"; python3 -c "
import json,glob
try:
    import os; L=json.load(open('/root/sb/dolphin_ladder/ladder.json' if os.path.exists('/root/sb/dolphin_ladder/ladder.json') else '/root/sb/ladder2/ladder.json'))
    for r in L.get('rungs',[]): print(' ', r.get('name'), 'select', r.get('pool_top1'), '2000:', r.get('page_top1'), 'step', r.get('step'), 'kept' if r.get('kept') else '', r.get('dev') or '')
    print('  cap:', json.dumps(L.get('cap',{}))[:400]); print('  cur:', json.dumps(L.get('cur'))[:300])
except Exception as e: print('  (no ladder.json yet)', e)
for f in sorted(glob.glob('/root/sb/ladder2/*_dev.json')): print(' ', f.split('/')[-1], open(f).read().strip()[:300])
" 2>&1
    echo "--- processes ---"; pgrep -fa "python3 /root/sb/" | cut -c1-300; } > /root/boxlog.txt 2>&1
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
# ---- meth (2026-10-04 08:40 JST): the 16+16 growth (run4n1, guard fixed) ended at 54.1 - below the 8+8 it started
# from (55.1): size is not the ceiling. The training method is: the page vector was learned from a document with 30%
# of its sentences hidden, but it is measured on one with a single sentence hidden; and each step saw 32 documents.
# run4p*: 8+8 from run4h1's best, --page-input one (a second encoder pass, one sentence hidden, that sentence queries),
# batch 64 (32 if the card runs out), lr 3e-5, guard 100; chained while a 100k segment adds a point over 55.1.
if ! pgrep -f "methkee[p].sh" >/dev/null && ! grep -q "METH_JOB_DONE" /root/sb_meth.log 2>/dev/null; then
  cat > /root/methkeep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb
while pgrep -f "python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/run" >/dev/null; do sleep 30; done
curl -sSf -o /root/sb/train.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/sentbart/train.py?$(date +%s)" || echo "[meth] train.py fetch failed, using the box copy"
grep -q "page-input" /root/sb/train.py || { echo "METH_JOB_DONE train.py has no --page-input"; exit 1; }
bestof() { grep -h "^\[best\]" /root/sb/$1/train.log 2>/dev/null | tail -1 | sed -E 's/.*page_top1 ([0-9.]+).*/\1/'; }
prev=/root/sb/run4h1/model_best.pt; base=0.551; BS=64
for i in 1 2 3 4; do
  n=run4p$i; rm -rf /root/sb/$n; echo "[meth] $n from $prev (top-1 $base), page-input one, batch $BS, lr 3e-5 $(date -u +%H:%M)"
  ( while sleep 1800; do hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; done ) & UP=$!
  python3 /root/sb/train.py --data $D --out /root/sb/$n --init $prev --eval-shard 001 --steps 100000 --batch $BS --seq 128 --d 512 --layers 8 --heads 8 --ffn 2048 --lr 3e-5 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-queue 64 --page-input one --skip-grad 100 2>&1 | grep -E "^\[model|^\[best|TRAIN_DONE|Error|Traceback|out of memory" | tee /root/sb/meth_$n.out
  kill $UP 2>/dev/null
  if grep -q "out of memory" /root/sb/meth_$n.out && [ $BS -gt 32 ]; then BS=32; echo "[meth] $n: out of memory at batch 64, again at 32"; rm -rf /root/sb/$n
    ( while sleep 1800; do hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; done ) & UP=$!
    python3 /root/sb/train.py --data $D --out /root/sb/$n --init $prev --eval-shard 001 --steps 100000 --batch $BS --seq 128 --d 512 --layers 8 --heads 8 --ffn 2048 --lr 3e-5 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-queue 64 --page-input one --skip-grad 100 2>&1 | grep -E "^\[model|^\[best|TRAIN_DONE|Error|Traceback|out of memory"
    kill $UP 2>/dev/null
  fi
  hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; hf upload $R /root/sb/$n/model_best.pt sentbart/small_$n/model_best.pt >/dev/null 2>&1
  rm -f /root/sb/$n/state.pt
  cur=$(bestof $n); cur=${cur:-0}; echo "[meth] $n done: best page top-1 $cur (from $base)"
  python3 -c "import sys; sys.exit(0 if float('$cur') - float('$base') >= 0.01 else 1)" || { echo "PLATEAU_METH at $n: best $cur vs $base"; break; }
  prev=/root/sb/$n/model_best.pt; base=$cur
done
echo "METH_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/methkeep.sh 2>&1 | tee -a /root/sb_meth.log' > /dev/null 2>&1 < /dev/null &
  echo "METH_LAUNCHED $(date -u)"
fi
# ---- abl (2026-10-04 09:40 JST, the user: review the training method, my call). Four 30k arms from run4h1's best
# (55.1), each one change, against run4h1's own first 30k at the same rate (53.7 / 54.0 / 54.3 at 10k/20k/30k):
#   one32   page vector from a one-sentence-hidden pass (the measured condition), batch 32
#   one64   the same at batch 64 (twice the fresh negatives)
#   lr1e4   lr 1e-4 with the guard at 100 (the earlier "instability" at 1e-4 was the guard at 5 starving the run)
#   one32w2 one32 with the page loss at double weight
# Then the best arm's recipe runs 100k from its own best, when it beat the control's 30k (54.3) by a point.
if [ ! -e /root/.abl_swap ]; then touch /root/.abl_swap
  echo "METH_JOB_DONE replaced by abl $(date -u)" >> /root/sb_meth.log
  pkill -f "methkee[p].sh"; pkill -f "python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/run4p"; sleep 10; echo "METH_STOPPED $(date -u)"
fi
if ! pgrep -f "ablkee[p].sh" >/dev/null && ! grep -q "ABL_JOB_DONE" /root/sb_abl.log 2>/dev/null; then
  cat > /root/ablkeep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb
while pgrep -f "python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/run" >/dev/null; do sleep 30; done
curl -sSf -o /root/sb/train.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/sentbart/train.py?$(date +%s)" || echo "[abl] train.py fetch failed, using the box copy"
grep -q "page-input" /root/sb/train.py || { echo "ABL_JOB_DONE train.py has no --page-input"; exit 1; }
bestof() { grep -h "^\[best\]" /root/sb/$1/train.log 2>/dev/null | tail -1 | sed -E 's/.*page_top1 ([0-9.]+).*/\1/'; }
evals() { grep -hoE "^\[eval [0-9]+\] page_top1 [0-9.]+" /root/sb/$1/train.log 2>/dev/null | sed -E 's/\[eval ([0-9]+)\] page_top1 ([0-9.]+)/\1:\2/' | tr '\n' ' '; }
COMMON="--data $D --init /root/sb/run4h1/model_best.pt --eval-shard 001 --seq 128 --d 512 --layers 8 --heads 8 --ffn 2048 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-queue 64 --skip-grad 100"
declare -A CFG=( [one32]="--batch 32 --lr 3e-5 --page-input one" [one64]="--batch 64 --lr 3e-5 --page-input one" [lr1e4]="--batch 32 --lr 1e-4" [one32w2]="--batch 32 --lr 3e-5 --page-input one --w-page 2" )
best=""; bestv=0
for a in one32 one64 lr1e4 one32w2; do
  n=abl_$a; rm -rf /root/sb/$n; echo "[abl] $a: ${CFG[$a]} $(date -u +%H:%M)"
  python3 /root/sb/train.py $COMMON --out /root/sb/$n --steps 30000 ${CFG[$a]} 2>&1 | grep -E "^\[model|TRAIN_DONE|Error|Traceback|out of memory"
  hf upload $R /root/sb/$n/train.log sentbart/abl/$a/train.log >/dev/null 2>&1; rm -f /root/sb/$n/state.pt
  cur=$(bestof $n); cur=${cur:-0}; echo "[abl] $a done: $(evals $n)| best $cur (control run4h1: 10000:0.537 20000:0.540 30000:0.543)"
  python3 -c "import sys; sys.exit(0 if float('$cur') > float('$bestv') else 1)" && { best=$a; bestv=$cur; }
done
echo "[abl] best arm $best at $bestv"
python3 -c "import sys; sys.exit(0 if float('$bestv') >= 0.553 else 1)" || { echo "ABL_JOB_DONE no arm beat the control by a point (best $best $bestv)"; exit 0; }
prev=/root/sb/abl_$best/model_best.pt; base=$bestv
for i in 1 2 3; do
  n=run4r$i; rm -rf /root/sb/$n; echo "[abl] $n: $best recipe 100k from $prev (top-1 $base) $(date -u +%H:%M)"
  ( while sleep 1800; do hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; done ) & UP=$!
  python3 /root/sb/train.py ${COMMON/--init \/root\/sb\/run4h1\/model_best.pt/--init $prev} --out /root/sb/$n --steps 100000 ${CFG[$best]} 2>&1 | grep -E "^\[model|^\[best|TRAIN_DONE|Error|Traceback|out of memory"
  kill $UP 2>/dev/null; hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; hf upload $R /root/sb/$n/model_best.pt sentbart/small_$n/model_best.pt >/dev/null 2>&1
  rm -f /root/sb/$n/state.pt
  cur=$(bestof $n); cur=${cur:-0}; echo "[abl] $n done: best page top-1 $cur (from $base)"
  python3 -c "import sys; sys.exit(0 if float('$cur') - float('$base') >= 0.01 else 1)" || { echo "PLATEAU_ABL at $n: best $cur vs $base"; break; }
  prev=/root/sb/$n/model_best.pt; base=$cur
done
echo "ABL_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/ablkeep.sh 2>&1 | tee -a /root/sb_abl.log' > /dev/null 2>&1 < /dev/null &
  echo "ABL_LAUNCHED $(date -u)"
fi
# ---- 2026-10-04 10:00 JST (the user: the next-sentence accuracy as cosine): the eval now also reports the cosine of the
# predicted next-sentence vector to the true one, with the previous sentence and a random sentence as the no-model
# readings. Measured once on the CPU beside the training, on the 55.1 model (run4h1) and the 4+4 (run4d1).
if [ ! -e /root/.cos1 ]; then touch /root/.cos1
  ( curl -sSf -o /root/sb/train_cos.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/sentbart/train.py?$(date +%s)"
    cd /root/sb; C="--data /root/sb/data/docs --eval-shard 001 --eval-only 1 --batch 32 --seq 128 --d 512 --heads 8 --ffn 2048 --page 1"
    SB_DEV=cpu OMP_NUM_THREADS=4 python3 /root/sb/train_cos.py $C --layers 8 --out /root/sb/evcos8 --init /root/sb/run4h1/model_best.pt 2>&1 | grep -E "^\[eval-only|Error|Traceback" | sed 's/^/[cos run4h1 8+8] /'
    SB_DEV=cpu OMP_NUM_THREADS=4 python3 /root/sb/train_cos.py $C --layers 4 --out /root/sb/evcos4 --init /root/sb/run4d1/model_best.pt 2>&1 | grep -E "^\[eval-only|Error|Traceback" | sed 's/^/[cos run4d1 4+4] /'
    echo "COS_DONE $(date -u)" ) > /root/sb_cos.log 2>&1 &
fi
# the cosine log in full (the mirror cuts long lines)
if [ ! -e /root/.cos1_up ] && grep -q COS_DONE /root/sb_cos.log 2>/dev/null; then touch /root/.cos1_up; hf upload $R /root/sb_cos.log sentbart/audit/cos.log >/dev/null 2>&1; echo "COS_UP $(date -u)"; fi
# ---- abl2 (2026-10-04 15:10 JST, the user: the page vector suffered because it was measured unlike it was trained, not
# because 30% hidden is wrong to train on - widen it step by step). Once abl's four arms are in ("best arm"), abl's
# 100k stage is stopped and two more 30k arms run from the same 55.1 start:
#   ramp     page pass: one sentence hidden at first, widening to 30% over 20k steps (every hidden sentence queries)
#   maskcur  one32 + the sentence losses' hidden share growing 15% -> 50% over 20k
# then the best of all six arms runs 100k from its own best weights, chained while a segment adds a point.
if ! pgrep -f "abl2kee[p].sh" >/dev/null && ! grep -q "ABL2_JOB_DONE" /root/sb_abl2.log 2>/dev/null; then
  cat > /root/abl2keep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb
until grep -qE "best arm|ABL_JOB_DONE" /root/sb_abl.log 2>/dev/null; do sleep 60; done
pkill -f "ablkee[p].sh"; pkill -f "python3 /root/sb/train.py --data /root/sb/data/docs --init /root/sb/abl_"; pkill -f "python3 /root/sb/train.py .*--out /root/sb/run4r"; sleep 10
echo "ABL_JOB_DONE stopped by abl2 before its 100k stage $(date -u)" >> /root/sb_abl.log
curl -sSf -o /root/sb/train.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/sentbart/train.py?$(date +%s)" || echo "[abl2] train.py fetch failed"
grep -q "page-ramp-max" /root/sb/train.py || { echo "ABL2_JOB_DONE train.py has no ramp"; exit 1; }
bestof() { grep -h "^\[best\]" /root/sb/$1/train.log 2>/dev/null | tail -1 | sed -E 's/.*page_top1 ([0-9.]+).*/\1/'; }
evals() { grep -hoE "^\[eval [0-9]+\] page_top1 [0-9.]+" /root/sb/$1/train.log 2>/dev/null | sed -E 's/\[eval ([0-9]+)\] page_top1 ([0-9.]+)/\1:\2/' | tr '\n' ' '; }
COMMON="--data $D --eval-shard 001 --seq 128 --d 512 --layers 8 --heads 8 --ffn 2048 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-queue 64 --skip-grad 100"
declare -A CFG=( [one32]="--batch 32 --lr 3e-5 --page-input one" [one64]="--batch 64 --lr 3e-5 --page-input one" [lr1e4]="--batch 32 --lr 1e-4" [one32w2]="--batch 32 --lr 3e-5 --page-input one --w-page 2" \
                 [ramp]="--batch 32 --lr 3e-5 --page-input ramp --page-ramp-max 0.3 --page-ramp-steps 20000" [maskcur]="--batch 32 --lr 3e-5 --page-input one --mask-start 0.15 --mask 0.5 --mask-ramp-steps 20000" )
for a in ramp maskcur; do
  n=abl_$a; rm -rf /root/sb/$n; echo "[abl2] $a: ${CFG[$a]} $(date -u +%H:%M)"
  python3 /root/sb/train.py $COMMON --init /root/sb/run4h1/model_best.pt --out /root/sb/$n --steps 30000 ${CFG[$a]} 2>&1 | grep -E "^\[model|TRAIN_DONE|Error|Traceback|out of memory"
  hf upload $R /root/sb/$n/train.log sentbart/abl/$a/train.log >/dev/null 2>&1; rm -f /root/sb/$n/state.pt
  echo "[abl2] $a done: $(evals $n)| best $(bestof $n) (control run4h1: 10000:0.537 20000:0.540 30000:0.543; one32 best 0.581)"
done
best=""; bestv=0
for a in one32 one64 lr1e4 one32w2 ramp maskcur; do v=$(bestof abl_$a); [ -n "$v" ] && [ -s /root/sb/abl_$a/model_best.pt ] && python3 -c "import sys; sys.exit(0 if float('$v') > float('$bestv') else 1)" && { best=$a; bestv=$v; }; done
echo "[abl2] best of six: $best at $bestv"
[ -n "$best" ] || { echo "ABL2_JOB_DONE no arm"; exit 1; }
prev=/root/sb/abl_$best/model_best.pt; base=$bestv
for i in 1 2 3; do
  n=run4s$i; rm -rf /root/sb/$n; echo "[abl2] $n: $best recipe 100k from $prev (top-1 $base) $(date -u +%H:%M)"
  ( while sleep 1800; do hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; done ) & UP=$!
  python3 /root/sb/train.py $COMMON --init $prev --out /root/sb/$n --steps 100000 ${CFG[$best]} 2>&1 | grep -E "^\[model|^\[best|TRAIN_DONE|Error|Traceback|out of memory"
  kill $UP 2>/dev/null; hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; hf upload $R /root/sb/$n/model_best.pt sentbart/small_$n/model_best.pt >/dev/null 2>&1
  rm -f /root/sb/$n/state.pt
  cur=$(bestof $n); cur=${cur:-0}; echo "[abl2] $n done: best page top-1 $cur (from $base)"
  python3 -c "import sys; sys.exit(0 if float('$cur') - float('$base') >= 0.01 else 1)" || { echo "PLATEAU_ABL2 at $n: best $cur vs $base"; break; }
  prev=/root/sb/$n/model_best.pt; base=$cur
done
echo "ABL2_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/abl2keep.sh 2>&1 | tee -a /root/sb_abl2.log' > /dev/null 2>&1 < /dev/null &
  echo "ABL2_LAUNCHED $(date -u)"
fi
# ---- 2026-10-05 07:20 JST: the disk filled (40G, 687M left) and run4s1 died writing its 20k checkpoint at 22:45; the
# card has been idle since. Free it: upload the two 58.1 arms' best weights, then delete the checkpoints of every
# finished run except the kept bests (run4h1, abl_one32, abl_maskcur - all on the hub). Then continue from abl_one32's
# best at lr 1e-5 (3e-5 restarts knocked every continuation back a point first), 100k, chained while +1 point.
if [ ! -e /root/.diskfree1 ]; then touch /root/.diskfree1
  echo "DISK_BEFORE $(df -h / | tail -1)"
  hf upload $R /root/sb/abl_one32/model_best.pt sentbart/abl/one32/model_best.pt >/dev/null 2>&1 && echo "UP one32 best"
  hf upload $R /root/sb/abl_maskcur/model_best.pt sentbart/abl/maskcur/model_best.pt >/dev/null 2>&1 && echo "UP maskcur best"
  for d in /root/sb/run* /root/sb/abl_* /root/sb/ev*; do [ -d "$d" ] || continue
    case "$(basename $d)" in run4h1|abl_one32|abl_maskcur) find "$d" -name "*.pt" ! -name "model_best.pt" -delete ;; *) find "$d" -name "*.pt" -delete ;; esac
  done
  rm -f /root/sb/train_cos.py
  echo "DISK_AFTER $(df -h / | tail -1)"; du -sh /root/sb/* 2>/dev/null | sort -rh | head -8
fi
if [ -e /root/.diskfree1 ] && ! pgrep -f "cont58kee[p].sh" >/dev/null && ! grep -q "CONT58_JOB_DONE" /root/sb_cont58.log 2>/dev/null; then
  cat > /root/cont58keep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb
while pgrep -f "python3 /root/sb/train.py --data /root/sb/data/docs" >/dev/null; do sleep 30; done
bestof() { grep -h "^\[best\]" /root/sb/$1/train.log 2>/dev/null | tail -1 | sed -E 's/.*page_top1 ([0-9.]+).*/\1/'; }
prev=/root/sb/abl_one32/model_best.pt; base=0.581
for i in 1 2 3; do
  n=run4t$i; rm -rf /root/sb/$n; echo "[cont58] $n from $prev (top-1 $base), one32 recipe, lr 1e-5 $(date -u +%H:%M); $(df -h / | tail -1 | awk '{print $4}') free"
  ( while sleep 1800; do hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; done ) & UP=$!
  python3 /root/sb/train.py --data $D --init $prev --eval-shard 001 --seq 128 --d 512 --layers 8 --heads 8 --ffn 2048 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-queue 64 --skip-grad 100 \
    --out /root/sb/$n --steps 100000 --batch 32 --lr 1e-5 --page-input one 2>&1 | grep -E "^\[model|^\[best|TRAIN_DONE|Error|Traceback|out of memory|write failed"
  kill $UP 2>/dev/null; hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; hf upload $R /root/sb/$n/model_best.pt sentbart/small_$n/model_best.pt >/dev/null 2>&1
  rm -f /root/sb/$n/state.pt /root/sb/$n/model_latest.pt
  cur=$(bestof $n); cur=${cur:-0}; echo "[cont58] $n done: best page top-1 $cur (from $base)"
  python3 -c "import sys; sys.exit(0 if float('$cur') - float('$base') >= 0.01 else 1)" || { echo "PLATEAU_CONT58 at $n: best $cur vs $base"; break; }
  prev=/root/sb/$n/model_best.pt; base=$cur
done
echo "CONT58_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/cont58keep.sh 2>&1 | tee -a /root/sb_cont58.log' > /dev/null 2>&1 < /dev/null &
  echo "CONT58_LAUNCHED $(date -u)"
fi
# ---- hn (2026-10-05 09:10 JST): continuing the 58.1 arm does not add (run4t1 at lr 1e-5: 57.5, 57.3 at 10k/20k), so
# stop it and try the next method lever: hard negatives. k-means (4096 clusters) over the training documents' mean
# sentence vectors; half of each batch from one cluster, so the page loss (and the sentence losses) see look-alike
# documents instead of 31 random ones. From run4h1 (55.1) with the one32 recipe, 30k, against one32 (58.1 at 20k):
# hn50 (half the batch one cluster), hn100 (the whole batch); the better one, if it beats 58.1 by half a point,
# runs 100k from its best.
if [ ! -e /root/.hn_swap ]; then touch /root/.hn_swap
  echo "CONT58_JOB_DONE stopped for hn (no gain) $(date -u)" >> /root/sb_cont58.log
  pkill -f "cont58kee[p].sh"; pkill -f "python3 /root/sb/train.py .*--out /root/sb/run4t1"; sleep 10
  hf upload $R /root/sb/run4t1/train.log sentbart/small_run4t1/train.log >/dev/null 2>&1; rm -f /root/sb/run4t1/*.pt; echo "RUN4T1_STOPPED $(date -u)"
fi
if ! pgrep -f "hnkee[p].sh" >/dev/null && ! grep -q "HN_JOB_DONE" /root/sb_hn.log 2>/dev/null; then
  cat > /root/hnkeep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb
while pgrep -f "python3 /root/sb/train.py --data /root/sb/data/docs" >/dev/null; do sleep 30; done
curl -sSf -o /root/sb/train.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/sentbart/train.py?$(date +%s)" || echo "[hn] train.py fetch failed"
grep -q "cluster-frac" /root/sb/train.py || { echo "HN_JOB_DONE train.py has no --cluster"; exit 1; }
bestof() { grep -h "^\[best\]" /root/sb/$1/train.log 2>/dev/null | tail -1 | sed -E 's/.*page_top1 ([0-9.]+).*/\1/'; }
evals() { grep -hoE "^\[eval [0-9]+\] page_top1 [0-9.]+" /root/sb/$1/train.log 2>/dev/null | sed -E 's/\[eval ([0-9]+)\] page_top1 ([0-9.]+)/\1:\2/' | tr '\n' ' '; }
COMMON="--data $D --eval-shard 001 --seq 128 --d 512 --layers 8 --heads 8 --ffn 2048 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-queue 64 --skip-grad 100 --batch 32 --lr 3e-5 --page-input one --cluster 4096"
best=""; bestv=0
for f in 0.5 1.0; do a=hn$(python3 -c "print(int($f*100))")
  n=abl_$a; rm -rf /root/sb/$n; echo "[hn] $a: cluster-frac $f $(date -u +%H:%M); $(df -h / | tail -1 | awk '{print $4}') free"
  python3 /root/sb/train.py $COMMON --cluster-frac $f --init /root/sb/run4h1/model_best.pt --out /root/sb/$n --steps 30000 2>&1 | grep -E "^\[model|^\[cluster|TRAIN_DONE|Error|Traceback|out of memory|write failed"
  hf upload $R /root/sb/$n/train.log sentbart/abl/$a/train.log >/dev/null 2>&1; rm -f /root/sb/$n/state.pt /root/sb/$n/model_latest.pt
  v=$(bestof $n); echo "[hn] $a done: $(evals $n)| best ${v:-?} (one32: 10000:0.574 20000:0.581 30000:0.579)"
  [ -n "$v" ] && python3 -c "import sys; sys.exit(0 if float('$v') > float('$bestv') else 1)" && { best=$a; bestv=$v; }
done
echo "[hn] best $best at $bestv"
python3 -c "import sys; sys.exit(0 if float('$bestv') >= 0.586 else 1)" || { echo "HN_JOB_DONE no gain over one32 (best $best $bestv)"; exit 0; }
F=$( [ "$best" = hn100 ] && echo 1.0 || echo 0.5 ); prev=/root/sb/abl_$best/model_best.pt; base=$bestv
hf upload $R $prev sentbart/abl/$best/model_best.pt >/dev/null 2>&1
for i in 1 2 3; do
  n=run4u$i; rm -rf /root/sb/$n; echo "[hn] $n: $best recipe 100k from $prev (top-1 $base) $(date -u +%H:%M)"
  ( while sleep 1800; do hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; done ) & UP=$!
  python3 /root/sb/train.py $COMMON --cluster-frac $F --init $prev --out /root/sb/$n --steps 100000 2>&1 | grep -E "^\[model|^\[best|TRAIN_DONE|Error|Traceback|out of memory|write failed"
  kill $UP 2>/dev/null; hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; hf upload $R /root/sb/$n/model_best.pt sentbart/small_$n/model_best.pt >/dev/null 2>&1
  rm -f /root/sb/$n/state.pt /root/sb/$n/model_latest.pt
  cur=$(bestof $n); cur=${cur:-0}; echo "[hn] $n done: best page top-1 $cur (from $base)"
  python3 -c "import sys; sys.exit(0 if float('$cur') - float('$base') >= 0.01 else 1)" || { echo "PLATEAU_HN at $n: best $cur vs $base"; break; }
  prev=/root/sb/$n/model_best.pt; base=$cur
done
echo "HN_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/hnkeep.sh 2>&1 | tee -a /root/sb_hn.log' > /dev/null 2>&1 < /dev/null &
  echo "HN_LAUNCHED $(date -u)"
fi
# ---- ph (2026-10-05 09:40 JST, the user: hard negatives belong to the page loss only - the sentence prediction is not
# there yet, its cosine barely beats "the sentence before"). Once hn's clustering has written its labels, hn is stopped
# and two arms run with RANDOM batches (sentence losses unchanged) and look-alike documents added to the page loss only
# (encoded without gradient, one sentence hidden, their page vectors as extra negatives): ph1 (one per batch document,
# 32 more), ph2 (two each, 64). From run4h1 (55.1), one32 recipe, 30k, against one32's 58.1.
if ! pgrep -f "phkee[p].sh" >/dev/null && ! grep -q "PH_JOB_DONE" /root/sb_ph.log 2>/dev/null; then
  cat > /root/phkeep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; D=/root/sb/data/docs; cd /root/sb
until ls /root/sb/doc_clusters/labels_4096_*.npy >/dev/null 2>&1 || grep -q "HN_JOB_DONE" /root/sb_hn.log 2>/dev/null; do sleep 60; done
echo "HN_JOB_DONE replaced by ph (page-only hard negatives) $(date -u)" >> /root/sb_hn.log
pkill -f "hnkee[p].sh"; pkill -f "python3 /root/sb/train.py .*--out /root/sb/abl_hn"; pkill -f "python3 /root/sb/train.py .*--out /root/sb/run4u"; sleep 15
curl -sSf -o /root/sb/train.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/sentbart/train.py?$(date +%s)" || echo "[ph] train.py fetch failed"
grep -q "page-hard" /root/sb/train.py || { echo "PH_JOB_DONE train.py has no --page-hard"; exit 1; }
bestof() { grep -h "^\[best\]" /root/sb/$1/train.log 2>/dev/null | tail -1 | sed -E 's/.*page_top1 ([0-9.]+).*/\1/'; }
evals() { grep -hoE "^\[eval [0-9]+\] page_top1 [0-9.]+" /root/sb/$1/train.log 2>/dev/null | sed -E 's/\[eval ([0-9]+)\] page_top1 ([0-9.]+)/\1:\2/' | tr '\n' ' '; }
COMMON="--data $D --eval-shard 001 --seq 128 --d 512 --layers 8 --heads 8 --ffn 2048 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-queue 64 --skip-grad 100 --batch 32 --lr 3e-5 --page-input one --cluster 4096 --cluster-frac 0"
best=""; bestv=0
for k in 1 2; do a=ph$k
  n=abl_$a; rm -rf /root/sb/$n; echo "[ph] $a: page-hard $k $(date -u +%H:%M); $(df -h / | tail -1 | awk '{print $4}') free"
  python3 /root/sb/train.py $COMMON --page-hard $k --init /root/sb/run4h1/model_best.pt --out /root/sb/$n --steps 30000 2>&1 | grep -E "^\[model|^\[cluster|TRAIN_DONE|Error|Traceback|out of memory|write failed"
  hf upload $R /root/sb/$n/train.log sentbart/abl/$a/train.log >/dev/null 2>&1; rm -f /root/sb/$n/state.pt /root/sb/$n/model_latest.pt
  v=$(bestof $n); echo "[ph] $a done: $(evals $n)| best ${v:-?} (one32: 10000:0.574 20000:0.581 30000:0.579)"
  [ -n "$v" ] && python3 -c "import sys; sys.exit(0 if float('$v') > float('$bestv') else 1)" && { best=$a; bestv=$v; }
done
echo "[ph] best $best at $bestv"
python3 -c "import sys; sys.exit(0 if float('$bestv') >= 0.586 else 1)" || { echo "PH_JOB_DONE no gain over one32 (best $best $bestv)"; exit 0; }
K=${best#ph}; prev=/root/sb/abl_$best/model_best.pt; base=$bestv
hf upload $R $prev sentbart/abl/$best/model_best.pt >/dev/null 2>&1
for i in 1 2 3; do
  n=run4v$i; rm -rf /root/sb/$n; echo "[ph] $n: $best recipe 100k from $prev (top-1 $base) $(date -u +%H:%M)"
  ( while sleep 1800; do hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; done ) & UP=$!
  python3 /root/sb/train.py $COMMON --page-hard $K --init $prev --out /root/sb/$n --steps 100000 2>&1 | grep -E "^\[model|^\[best|TRAIN_DONE|Error|Traceback|out of memory|write failed"
  kill $UP 2>/dev/null; hf upload $R /root/sb/$n/train.log sentbart/small_$n/train.log >/dev/null 2>&1; hf upload $R /root/sb/$n/model_best.pt sentbart/small_$n/model_best.pt >/dev/null 2>&1
  rm -f /root/sb/$n/state.pt /root/sb/$n/model_latest.pt
  cur=$(bestof $n); cur=${cur:-0}; echo "[ph] $n done: best page top-1 $cur (from $base)"
  python3 -c "import sys; sys.exit(0 if float('$cur') - float('$base') >= 0.01 else 1)" || { echo "PLATEAU_PH at $n: best $cur vs $base"; break; }
  prev=/root/sb/$n/model_best.pt; base=$cur
done
echo "PH_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/phkeep.sh 2>&1 | tee -a /root/sb_ph.log' > /dev/null 2>&1 < /dev/null &
  echo "PH_LAUNCHED $(date -u)"
fi
# ---- bgectl (2026-10-05 14:20 JST): the control for box K's BART on language-model sentence vectors - the same
# recipe from scratch (8+8, page vector from the one-sentence-hidden input, lr 1e-4, 100k) on the same Wikipedia
# shards (0, 2-6 for training, 1 held out: the same 2000 evaluation articles) with bge-small vectors. After ph.
if ! pgrep -f "bgectlkee[p].sh" >/dev/null && ! grep -q "BGECTL_JOB_DONE" /root/sb_bgectl.log 2>/dev/null; then
  cat > /root/bgectlkeep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/sb
until grep -q "PH_JOB_DONE" /root/sb_ph.log 2>/dev/null; do sleep 120; done
while pgrep -f "python3 /root/sb/train.py --data" >/dev/null; do sleep 30; done
D=/root/sb/data/docs7; mkdir -p $D
for k in 000 001 002 003 004 005 006; do for f in vec scl off; do [ -e /root/sb/data/docs/${f}_$k.npy ] && ln -sf /root/sb/data/docs/${f}_$k.npy $D/${f}_$k.npy; done; done
echo "[bgectl] shards: $(ls $D | tr '\n' ' ') $(date -u +%H:%M)"
n=bgectl1
( while sleep 1800; do hf upload $R /root/sb/$n/train.log sentbart/lmz/$n/train.log >/dev/null 2>&1; done ) & UP=$!
python3 /root/sb/train.py --data $D --out /root/sb/$n --eval-shard 001 --steps 100000 --batch 32 --seq 128 --d 512 --layers 8 --heads 8 --ffn 2048 \
  --lr 1e-4 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-queue 64 --page-input one --skip-grad 100 2>&1 | grep -E "^\[data|^\[model|^\[best|TRAIN_DONE|Error|Traceback|out of memory"
kill $UP 2>/dev/null; hf upload $R /root/sb/$n/train.log sentbart/lmz/$n/train.log >/dev/null 2>&1
rm -f /root/sb/$n/state.pt /root/sb/$n/model_latest.pt
echo "BGECTL_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/bgectlkeep.sh 2>&1 | tee -a /root/sb_bgectl.log' > /dev/null 2>&1 < /dev/null &
  echo "BGECTL_LAUNCHED $(date -u)"
fi
# ---- e2e (2026-10-05 18:40 JST, the user: the bge control is not needed - end to end instead). bgectl1 stops (its log
# stays). train_e2e.py: bge-small trained inside the BART's step (every BART loss reaches it; inputs and targets are
# its outputs), anchored to the stock bge by in-batch InfoNCE; from abl_one32 (58.1%) and stock bge, so step 0 is
# 58.1. Training text: shards 0, 2, 3 again as text (prep.py). Evaluated every 2000 steps on the same 2000 held-out
# documents, embedded by the current bge.
if [ ! -e /root/.bgectl_stop ]; then touch /root/.bgectl_stop
  pkill -f "bgectlkee[p].sh"; pkill -f "train.py --data /root/sb/data/docs7"; sleep 5
  hf upload baya1116/hypernet-sp-distill /root/sb/bgectl1/train.log sentbart/lmz/bgectl1/train.log >/dev/null 2>&1
  echo "BGECTL_JOB_DONE stopped by the user at $(grep -oE '^\[step [0-9]+' /root/sb/bgectl1/train.log | tail -1) $(date -u)" >> /root/sb_bgectl.log
  rm -f /root/sb/bgectl1/state.pt /root/sb/bgectl1/model_latest.pt
fi
if [ -f /root/sb/train_e2e.py ] && ! pgrep -f "e2ekee[p].sh" >/dev/null && ! grep -q "E2E_JOB_DONE" /root/sb_e2e.log 2>/dev/null; then
  cat > /root/e2ekeep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/sb
while pgrep -f "python3 /root/sb/train.py --data" >/dev/null; do sleep 30; done
T=/root/sb/data/e2e; mkdir -p $T
for r in 0-0 2-3; do python3 /root/sb/prep.py --out $T --shards $r 2>&1 | grep --line-buffered -E "^\[prep\] shard|Error|Traceback"; done
rm -rf /root/.cache/huggingface/hub/datasets--wikimedia--wikipedia
ls $T; df -h /root | tail -1
I0=/root/sb/abl_one32/model_best.pt; [ -s $I0 ] || hf download $R sentbart/abl/one32/model_best.pt --local-dir /root/sb/hfdl >/dev/null 2>&1 && [ -s $I0 ] || I0=/root/sb/hfdl/sentbart/abl/one32/model_best.pt
n=e2e1; echo "[e2e] $n from $I0, anchor 1.0, bge lr 1e-5, BART lr 3e-5, 30k $(date -u +%H:%M)"
( while sleep 1800; do hf upload $R /root/sb/$n/train.log sentbart/e2e/$n/train.log >/dev/null 2>&1; done ) & UP=$!
python3 /root/sb/train_e2e.py --vec /root/sb/data/docs --text $T --eval-text /root/sb/data/docs/docs_001.jsonl --init $I0 --out /root/sb/$n \
  --steps 30000 --w-anchor 1.0 --lr-enc 1e-5 --lr 3e-5 2>&1 | grep --line-buffered -E "^\[e2e\]|^\[model|^\[init|^\[eval|^\[best|^\[step [0-9]*000\]|TRAIN_DONE|Error|Traceback|out of memory|assert"
kill $UP 2>/dev/null; hf upload $R /root/sb/$n/train.log sentbart/e2e/$n/train.log >/dev/null 2>&1; hf upload $R /root/sb/$n/model_best.pt sentbart/e2e/$n/model_best.pt >/dev/null 2>&1
rm -f /root/sb/$n/state_e2e.pt
echo "E2E_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/e2ekeep.sh 2>&1 | tee -a /root/sb_e2e.log' > /dev/null 2>&1 < /dev/null &
  echo "E2E_LAUNCHED $(date -u)"
fi
# ---- cascade (2026-10-05 19:50 JST, the user: one vector per article picks ~10 candidates, the candidates are opened
# and searched sentence by sentence with bge). search_cascade.py with the app model's own queries over all 118,398
# held-out articles: the 58.1 model (abl_one32) now, beside e2e1; e2e1's best once its 10k evaluation is in (its
# own encoder embeds the shard and the queries).
if [ -f /root/sb/search_cascade.py ] && ! pgrep -f "cascadekee[p].sh" >/dev/null && ! grep -q "CASCADE_JOB_DONE" /root/sb_cascade.log 2>/dev/null; then
  cat > /root/cascadekeep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/sb; mkdir -p /root/sb/sc
QF=/root/sb/se2/dl/sentbart/searcheval/dcq.jsonl; [ -s $QF ] || hf download $R sentbart/searcheval/dcq.jsonl --local-dir /root/sb/se2/dl >/dev/null 2>&1
I0=/root/sb/abl_one32/model_best.pt; [ -s $I0 ] || I0=/root/sb/hfdl/sentbart/abl/one32/model_best.pt
echo "[cascade] one32 (58.1) $(date -u +%H:%M)"
python3 /root/sb/search_cascade.py --vec /root/sb/data/docs --text /root/sb/data/docs/docs_001.jsonl --ckpt $I0 --queries $QF --out /root/sb/sc/one32.json 2>&1 | grep --line-buffered -E "^\[cascade\]|CASCADE_DONE|Error|Traceback|out of memory"
hf upload $R /root/sb/sc/one32.json sentbart/searcheval/cascade_one32.json >/dev/null 2>&1
until grep -q "^\[eval 10000\]" /root/sb/e2e1/train.log 2>/dev/null || grep -q "E2E_JOB_DONE" /root/sb_e2e.log 2>/dev/null; do sleep 120; done
cp /root/sb/e2e1/model_best.pt /root/sb/sc/e2e1_best.pt; echo "[cascade] e2e1 best ($(grep '^\[best' /root/sb/e2e1/train.log | tail -1)) $(date -u +%H:%M)"
python3 /root/sb/search_cascade.py --vec /root/sb/data/docs --text /root/sb/data/docs/docs_001.jsonl --ckpt /root/sb/sc/e2e1_best.pt --queries $QF --out /root/sb/sc/e2e1.json 2>&1 | grep --line-buffered -E "^\[cascade\]|CASCADE_DONE|Error|Traceback|out of memory"
hf upload $R /root/sb/sc/e2e1.json sentbart/searcheval/cascade_e2e1.json >/dev/null 2>&1; rm -f /root/sb/sc/e2e1_best.pt
echo "CASCADE_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/cascadekeep.sh 2>&1 | tee -a /root/sb_cascade.log' > /dev/null 2>&1 < /dev/null &
  echo "CASCADE_LAUNCHED $(date -u)"
fi
# ---- cascadeH (2026-10-05 20:10 JST, the user: HyDE-style queries): the same two-stage search with the queries
# rewritten as one Wikipedia-style sentence each (box G, nano), embedded as plain sentences - the 58.1 model.
if [ -f /root/sb/search_cascade.py ] && ! pgrep -f "cascadeHkee[p].sh" >/dev/null && ! grep -q "CASCADEH_JOB_DONE" /root/sb_cascadeH.log 2>/dev/null; then
  cat > /root/cascadeHkeep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/sb; mkdir -p /root/sb/sc
until hf download $R sentbart/searcheval/dcq_hyde.jsonl --local-dir /root/sb/sc/dl >/dev/null 2>&1 && [ -s /root/sb/sc/dl/sentbart/searcheval/dcq_hyde.jsonl ]; do sleep 120; done
until [ -s /root/sb/sc/one32.json ]; do sleep 60; done
I0=/root/sb/abl_one32/model_best.pt; [ -s $I0 ] || I0=/root/sb/hfdl/sentbart/abl/one32/model_best.pt
echo "[cascadeH] one32 (58.1), HyDE queries $(date -u +%H:%M)"
python3 /root/sb/search_cascade.py --vec /root/sb/data/docs --text /root/sb/data/docs/docs_001.jsonl --ckpt $I0 --queries /root/sb/sc/dl/sentbart/searcheval/dcq_hyde.jsonl --out /root/sb/sc/one32_hyde.json 2>&1 | grep --line-buffered -E "^\[cascade\] (q_|h_)|CASCADE_DONE|Error|Traceback|out of memory"
hf upload $R /root/sb/sc/one32_hyde.json sentbart/searcheval/cascade_one32_hyde.json >/dev/null 2>&1
echo "CASCADEH_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/cascadeHkeep.sh 2>&1 | tee -a /root/sb_cascadeH.log' > /dev/null 2>&1 < /dev/null &
  echo "CASCADEH_LAUNCHED $(date -u)"
fi
# ---- 2026-10-05 22:45 JST, the user: 100 queries are enough. The keyword run of e2e1 on all 553 stops; the HyDE 100
# (both forms of the same 100 queries) is the one measurement from here on.
if [ ! -e /root/.cascade553_stop ]; then touch /root/.cascade553_stop
  pkill -f "cascadekee[p].sh"; pkill -f "search_cascade.py.*e2e1_best.pt"; pkill -f "cascadeEkee[p].sh"; sleep 3
  grep -q "CASCADE_JOB_DONE" /root/sb_cascade.log 2>/dev/null || echo "CASCADE_JOB_DONE e2e1 on 553 dropped (100 are enough) $(date -u)" >> /root/sb_cascade.log
  rm -f /root/sb/sc/e2e1_best.pt
fi
# ---- 2026-10-05 22:50 JST, the user: the time is the same either way - keep all 553 keyword queries. cascadeE now runs
# once on all of them, the HyDE 100 carried along (one re-embedding of the shard).
if [ ! -e /root/.cascadeE3 ]; then touch /root/.cascadeE3; pkill -f "cascadeEkee[p].sh"; pkill -f "search_cascade.py.*e2e1_bestH.pt"; sleep 3; fi
# ---- cascadeE (2026-10-05 20:15 JST): e2e1's best (from its 10k evaluation on) on the HyDE 100 as well - keyword and
# HyDE forms of the same queries, its own encoder; after the keyword run of the cascade job.
if [ -f /root/sb/search_cascade.py ] && ! pgrep -f "cascadeEkee[p].sh" >/dev/null && ! grep -q "CASCADEE_JOB_DONE" /root/sb_cascadeE3.log 2>/dev/null; then
  cat > /root/cascadeEkeep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/sb
until grep -q "CASCADE_JOB_DONE" /root/sb_cascade.log 2>/dev/null && grep -q "CASCADEH_JOB_DONE" /root/sb_cascadeH.log 2>/dev/null; do sleep 120; done
python3 - <<'PYM'
import json
H = {json.loads(l)["idx"]: json.loads(l) for l in open("/root/sb/sc/dl/sentbart/searcheval/dcq_hyde.jsonl")}
rows = []
for l in open("/root/sb/se2/dl/sentbart/searcheval/dcq.jsonl"):
    r = json.loads(l); h = H.get(r["idx"])
    if h and h.get("q_api") == r.get("q_api") and h.get("q_good") == r.get("q_good"):
        r.update({k: v for k, v in h.items() if k.startswith("h_")})
    rows.append(r)
open("/root/sb/sc/dcq_all.jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
print(f"[cascadeE] {len(rows)} keyword queries, {sum(1 for r in rows if r.get('h_api') or r.get('h_good'))} with a HyDE form", flush=True)
PYM
cp /root/sb/e2e1/model_best.pt /root/sb/sc/e2e1_bestH.pt; echo "[cascadeE] e2e1 best ($(grep '^\[best' /root/sb/e2e1/train.log | tail -1)), HyDE 100 $(date -u +%H:%M)"
python3 /root/sb/search_cascade.py --vec /root/sb/data/docs --text /root/sb/data/docs/docs_001.jsonl --ckpt /root/sb/sc/e2e1_bestH.pt --queries /root/sb/sc/dcq_all.jsonl --out /root/sb/sc/e2e1_hyde.json 2>&1 | grep --line-buffered -E "^\[cascade\] (q_|h_)|CASCADE_DONE|Error|Traceback|out of memory"
hf upload $R /root/sb/sc/e2e1_hyde.json sentbart/searcheval/cascade_e2e1_hyde.json >/dev/null 2>&1; rm -f /root/sb/sc/e2e1_bestH.pt
echo "CASCADEE_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/cascadeEkeep.sh 2>&1 | tee -a /root/sb_cascadeE3.log' > /dev/null 2>&1 < /dev/null &
  echo "CASCADEE_LAUNCHED $(date -u)"
fi
# ---- 2026-10-06 00:35 JST: the ladder restarted once on the fixed e2e_ladder.py (each rung's checkpoints removed after
# use - the disk holds ~13 GB); it resumes from ladder.json where there is one.
if [ ! -e /root/.ladder_r2 ] && [ -f /root/sb/e2e_ladder.py ] && grep -q '"model_best.pt")' /root/sb/e2e_ladder.py; then touch /root/.ladder_r2
  pkill -f "ladderkee[p].sh"; pkill -f "e2e_ladder.p[y]"; pkill -f "train_e2e.py.*/root/sb/ladder/"; pkill -f "search_cascade.py.*/root/sb/ladder/"; sleep 10
  echo "LADDER_RESTART $(date -u)" >> /root/sb_ladder.log
fi
# ---- ladder (2026-10-06 00:20 JST, the user, away tomorrow: raise the sentence BART as far as it goes - negatives ->
# bge layers -> BART layers, round and round, each step when the last has levelled off; the best kept per capacity).
# e2e1 is stopped: its sentence -> page score rose (58.1 -> 66.1) but on the app model's real queries its bge got WORSE
# (page recall@10 28.6 -> 21.2, the lead sentence 60.0 -> 43.4) - the query path drifted. The ladder starts again from
# the 58.1 model with the anchor also on query-form inputs, rungs selected on page top-1 among 20,000 held-out articles
# and kept only if a dev split of the real queries (300) does not fall. Record: sentbart/e2e/ladder/ladder.json.
if [ -f /root/sb/e2e_ladder.py ] && ! pgrep -f "ladderkee[p].sh" >/dev/null && ! grep -q "LADDER_JOB_DONE" /root/sb_ladder.log 2>/dev/null; then
  pkill -f "train_e2e.py.*--out /root/sb/e2e1"; sleep 10
  hf upload baya1116/hypernet-sp-distill /root/sb/e2e1/train.log sentbart/e2e/e2e1/train.log >/dev/null 2>&1
  hf upload baya1116/hypernet-sp-distill /root/sb/e2e1/model_best.pt sentbart/e2e/e2e1/model_best.pt >/dev/null 2>&1
  rm -f /root/sb/e2e1/state_e2e.pt /root/sb/e2e1/model_latest.pt
  cat > /root/ladderkeep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/sb
I0=/root/sb/abl_one32/model_best.pt; [ -s $I0 ] || I0=/root/sb/hfdl/sentbart/abl/one32/model_best.pt
QF=/root/sb/se2/dl/sentbart/searcheval/dcq.jsonl; [ -s $QF ] || hf download $R sentbart/searcheval/dcq.jsonl --local-dir /root/sb/se2/dl >/dev/null 2>&1
echo "[ladder] from $I0 $(date -u +%H:%M); $(df -h /root | tail -1 | awk '{print $4}') free"
python3 -u /root/sb/e2e_ladder.py --start $I0 --work /root/sb/ladder --queries $QF 2>&1 | grep --line-buffered -E "^\[ladder\]|LADDER_DONE|Error|Traceback"
echo "LADDER_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/ladderkeep.sh 2>&1 | tee -a /root/sb_ladder.log' > /dev/null 2>&1 < /dev/null &
  echo "LADDER_LAUNCHED $(date -u)"
fi
# ---- ladderlog (2026-10-06 03:50 JST): the running rung's training log on the hub every 20 minutes (the ladder's own
# log only speaks between rungs).
if [ ! -e /root/.ladderlog_r2 ]; then touch /root/.ladderlog_r2; pkill -f "ladderlogkee[p].sh"; sleep 1; fi   # restarted once for the Dolphin ladder's logs
if ! pgrep -f "ladderlogkee[p].sh" >/dev/null; then
  cat > /root/ladderlogkeep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
while :; do
  f=$(ls -t /root/sb/ladder/r*/train.log /root/sb/ladder2/r*/train.log /root/sb/dolphin_ladder/r*/train.log 2>/dev/null | head -1)
  [ -n "$f" ] && { echo "== $f $(date -u)"; grep -E "^\[eval|^\[best|^\[plateau|^\[e2e\]" $f | tail -20 | cut -c1-400; grep "^\[step" $f | tail -2; } > /root/ladder_current.txt && hf upload $R /root/ladder_current.txt sentbart/e2e/ladder/current.txt >/dev/null 2>&1
  sleep 1200
done
RK
  setsid nohup bash /root/ladderlogkeep.sh > /dev/null 2>&1 < /dev/null &
fi
# ---- cascadeR1 (2026-10-06 13:25 JST, the user: and with HyDE queries?): ladder r01's best (e2e + query anchor) on the
# keyword queries and the HyDE 100 (dcq_all), beside the running rung. The 58.1 model and e2e1 were measured on the same.
if [ -s /root/sb/ladder/cap_bge12_bart8/model_best.pt ] && [ -s /root/sb/sc/dcq_all.jsonl ] && ! pgrep -f "cascadeR1kee[p].sh" >/dev/null && ! grep -q "CASCADER1_JOB_DONE" /root/sb_cascadeR1.log 2>/dev/null; then
  cat > /root/cascadeR1keep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/sb
cp /root/sb/ladder/cap_bge12_bart8/model_best.pt /root/sb/sc/r01_best.pt; echo "[cascadeR1] r01 best $(date -u +%H:%M)"
python3 /root/sb/search_cascade.py --vec /root/sb/data/docs --text /root/sb/data/docs/docs_001.jsonl --ckpt /root/sb/sc/r01_best.pt --queries /root/sb/sc/dcq_all.jsonl --out /root/sb/sc/r01.json 2>&1 | grep --line-buffered -E "^\[cascade\] (q_api|h_api)|CASCADE_DONE|Error|Traceback|out of memory"
hf upload $R /root/sb/sc/r01.json sentbart/searcheval/cascade_r01.json >/dev/null 2>&1; rm -f /root/sb/sc/r01_best.pt
echo "CASCADER1_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/cascadeR1keep.sh 2>&1 | tee -a /root/sb_cascadeR1.log' > /dev/null 2>&1 < /dev/null &
  echo "CASCADER1_LAUNCHED $(date -u)"
fi
# ---- e2e2 (2026-10-06 23:30 JST, the user: the HyDE-style and search-query losses were never in - put them in). Step 1,
# here: the documents the query writer gets (pick_docs.py: 20,000 training documents + the 2000 evaluation ones, a passage
# each) to the hub; box G writes the queries with nano and puts them back. Step 2: once the queries are on the hub, the
# running ladder stops and a second ladder starts from the 58.1 model with train_e2e --queries (query -> article and
# query -> passage sentence losses, search-query and HyDE forms), selected on pool_qh_top1 (the written queries of the
# 2000 evaluation documents against 20,000 pages), the real-query dev guard as before. Record: sentbart/e2e2/ladder.json.
if [ -f /root/sb/pick_docs.py ] && [ ! -e /root/.e2e2_docs ]; then touch /root/.e2e2_docs
  ( export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); mkdir -p /root/sb/e2e2
    python3 /root/sb/pick_docs.py --vec /root/sb/data/docs --text /root/sb/data/e2e --eval-text /root/sb/data/docs/docs_001.jsonl --n 20000 --out /root/sb/e2e2/docs_for_qgen.jsonl 2>&1 | tail -2
    [ -s /root/sb/e2e2/docs_for_qgen.jsonl ] && hf upload baya1116/hypernet-sp-distill /root/sb/e2e2/docs_for_qgen.jsonl sentbart/e2e2/docs_for_qgen.jsonl >/dev/null 2>&1 && echo "E2E2_DOCS_UP $(date -u)"
  ) >> /root/sb_e2e2.log 2>&1 &
fi
# ---- 2026-10-07 02:20 JST: ladder2's waiting keeper replaced (starts on the partial queries at 10,000 rows, refreshes them).
if [ ! -e /root/.ladder2_v2 ]; then touch /root/.ladder2_v2; pkill -f "ladder2kee[p].sh"; sleep 2; fi
if [ ! -e /root/.ladder2_v3 ]; then touch /root/.ladder2_v3; pkill -f "ladder2kee[p].sh"; sleep 2; fi
if [ -f /root/sb/e2e_ladder.py ] && ! pgrep -f "ladder2kee[p].sh" >/dev/null && ! grep -q "LADDER2_JOB_DONE" /root/sb_ladder2.log 2>/dev/null; then
  cat > /root/ladder2keep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/sb; mkdir -p /root/sb/e2e2
# the writer takes hours: start once 10,000 rows are there (the 2000 evaluation documents' come first), and refresh the
# file from the hub every 15 minutes - each later rung reads it afresh at its start
fetchq() { best=""; bn=0   # whichever of the final and the partial file holds more rows (a killed writer once uploaded a short "final")
  for f in queries.jsonl queries_partial.jsonl; do hf download $R sentbart/e2e2/$f --local-dir /root/sb/e2e2/dl >/dev/null 2>&1 && [ -s /root/sb/e2e2/dl/sentbart/e2e2/$f ] && n=$(wc -l < /root/sb/e2e2/dl/sentbart/e2e2/$f) && [ "$n" -gt "$bn" ] && { bn=$n; best=/root/sb/e2e2/dl/sentbart/e2e2/$f; }; done
  [ -n "$best" ] || return 1; cp $best /root/sb/e2e2/queries.jsonl.new && mv /root/sb/e2e2/queries.jsonl.new /root/sb/e2e2/queries.jsonl; }
until fetchq && [ "$(wc -l < /root/sb/e2e2/queries.jsonl)" -ge 10000 ]; do sleep 300; done
echo "[ladder2] queries: $(wc -l < /root/sb/e2e2/queries.jsonl) rows $(date -u +%H:%M)"
( while sleep 900; do fetchq; done ) > /dev/null 2>&1 &
pkill -f "ladderkee[p].sh"; pkill -f "e2e_ladder.py --start"; pkill -f "train_e2e.py.*/root/sb/ladder/"; pkill -f "search_cascade.py.*/root/sb/ladder/"; sleep 15
echo "LADDER_JOB_DONE replaced by ladder2 $(date -u)" >> /root/sb_ladder.log
hf upload $R /root/sb/ladder/ladder.json sentbart/e2e/ladder/ladder.json >/dev/null 2>&1
for d in /root/sb/ladder/r*/; do rm -f $d/state_e2e.pt $d/model_latest.pt $d/model_best.pt; done
I0=/root/sb/abl_one32/model_best.pt; [ -s $I0 ] || I0=/root/sb/hfdl/sentbart/abl/one32/model_best.pt
QF=/root/sb/se2/dl/sentbart/searcheval/dcq.jsonl
echo "[ladder2] from $I0 with the query losses $(date -u +%H:%M); $(df -h /root | tail -1 | awk '{print $4}') free"
python3 -u /root/sb/e2e_ladder.py --start $I0 --work /root/sb/ladder2 --queries $QF --qfile /root/sb/e2e2/queries.jsonl --select pool_qh_top1 2>&1 | grep --line-buffered -E "^\[ladder\]|LADDER_DONE|Error|Traceback" | sed -u 's/^\[ladder\]/[ladder2]/'
echo "LADDER2_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/ladder2keep.sh 2>&1 | tee -a /root/sb_ladder2.log' > /dev/null 2>&1 < /dev/null &
  echo "LADDER2_LAUNCHED $(date -u)"
fi
# ---- 2026-10-07 19:10 JST (the user: the dev guard looks very low - is it the HyDE-form evaluation?): the guard runs the
# app's keyword queries over all 118,398 articles; the pool's HyDE numbers are nano-written sentences over 20,000. The
# missing figure - HyDE-form queries over the 118,398 - is measured here for ladder2's r01 model (the 100 dev queries
# that have HyDE forms, like cascade_one32_hyde.json for the start model). Runs beside the training as the earlier cascades did.
if [ ! -e /root/.casc_l2r01 ] && [ -s /root/sb/ladder2/cap_bge12_bart8/model_best.pt ]; then touch /root/.casc_l2r01
  ( export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; mkdir -p /root/sb/sc
    until hf download $R sentbart/searcheval/dcq_hyde.jsonl --local-dir /root/sb/sc/dl >/dev/null 2>&1 && [ -s /root/sb/sc/dl/sentbart/searcheval/dcq_hyde.jsonl ]; do sleep 120; done
    echo "[cascadeL2] r01 (cap_bge12_bart8) on the HyDE dev queries $(date -u +%H:%M)"
    python3 /root/sb/search_cascade.py --vec /root/sb/data/docs --text /root/sb/data/docs/docs_001.jsonl --ckpt /root/sb/ladder2/cap_bge12_bart8/model_best.pt \
      --queries /root/sb/sc/dl/sentbart/searcheval/dcq_hyde.jsonl --out /root/sb/sc/l2r01_hyde.json 2>&1 | grep -E "^\[cascade\]|Error|Traceback|out of memory" | cut -c1-300
    [ -s /root/sb/sc/l2r01_hyde.json ] && hf upload $R /root/sb/sc/l2r01_hyde.json sentbart/searcheval/cascade_ladder2_r01_hyde.json >/dev/null 2>&1 && echo "CASCADEL2_DONE $(date -u)"
  ) > /root/sb_cascadeL2.log 2>&1 &
fi
# ---- Dolphin context sequences, e2e (2026-10-07 20:30 JST; the user's last item: the BART before bge was touched (abl_one32) +
# stock bge, trained e2e on Dolphin conversations - the question's sentences read as context, never hidden or predicted).
# Data: 50,000 training + 22,000 held-out conversations of dolphin-r1's deepseek set (dolphin_v1's questions excluded),
# stock bge vectors for the layout, then train_e2e.py: pool retrieval among 20,000 conversations selects the best.
# DOLPHIN_GO=1 also stops ladder2 (the box has one GPU) - set once the user says so.
DOLPHIN_GO=1
if [ "$DOLPHIN_GO" = "1" ] && [ ! -e /root/.dolphin_e2e ] && [ -s /root/sb/prep_dolphin.py ]; then touch /root/.dolphin_e2e
  cat > /root/dolphinkeep.sh <<'DK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/sb; D=/root/sb/data/dolphin; OUT=/root/sb/dolphin_e2e
pkill -f "ladder2kee[p].sh"; pkill -f "e2e_ladder.py --start"; pkill -f "train_e2e.py.*/root/sb/ladder2/"; pkill -f "search_cascade.py.*/root/sb/ladder2/"; sleep 15
echo "LADDER2_JOB_DONE stopped for the Dolphin e2e $(date -u)" >> /root/sb_ladder2.log
hf upload $R /root/sb/ladder2/ladder.json sentbart/e2e/ladder2/ladder.json >/dev/null 2>&1
for d in /root/sb/ladder2/r*/; do rm -f $d/state_e2e.pt $d/model_latest.pt $d/model_best.pt; done
I0=/root/sb/abl_one32/model_best.pt; [ -s $I0 ] || { hf download $R sentbart/abl/one32/model_best.pt --local-dir /root/sb/hfdl >/dev/null 2>&1; I0=/root/sb/hfdl/sentbart/abl/one32/model_best.pt; }
[ -s $I0 ] || { echo "DOLPHIN_JOB_DONE no one32"; exit 1; }
mkdir -p /root/sb/dl; hf download $R pooler_distill/chatsft/dolphin_v1.jsonl --local-dir /root/sb/dl >/dev/null 2>&1
if [ ! -s $D/docs_001.jsonl ]; then
  echo "[dolphin] preparing the conversations $(date -u +%H:%M); $(df -h /root | tail -1 | awk '{print $4}') free"
  python3 /root/sb/prep_dolphin.py --out $D --n-train 50000 --n-eval 22000 --exclude /root/sb/dl/pooler_distill/chatsft/dolphin_v1.jsonl 2>&1 | grep -E "^\[prep_dolphin\]|PREP_DOLPHIN|Error|Traceback" | cut -c1-300
  head -c 600 $D/docs_001.jsonl; echo
fi
[ -s $D/docs_001.jsonl ] || { echo "DOLPHIN_JOB_DONE no documents"; exit 1; }
[ -s $D/vec_001.npy ] || { echo "[dolphin] stock bge vectors $(date -u +%H:%M)"; python3 /root/sb/embed.py --dir $D 2>&1 | grep -vE "Warning|warn" | tail -3 | cut -c1-200; }
[ -s $D/vec_001.npy ] || { echo "DOLPHIN_JOB_DONE no vectors"; exit 1; }
echo "[dolphin] e2e from $I0 $(date -u +%H:%M)"
python3 -u /root/sb/train_e2e.py --vec $D --text $D --eval-text $D/docs_001.jsonl --init $I0 --out $OUT --steps 40000 --batch 32 --page-queue 16   --pool 20000 --select pool_top1 --patience 4 --min-gain 0.002 --anchor-q 0 2>&1 | grep --line-buffered -E "^\[e2e\]|^\[data\]|^\[init\]|^\[best|^\[plateau|TRAIN_DONE|Error|Traceback|out of memory" | cut -c1-300
for f in model_best.pt train.log; do [ -s $OUT/$f ] && hf upload $R $OUT/$f sentbart/dolphin/e2e1/$f >/dev/null 2>&1; done
echo "DOLPHIN_JOB_DONE $(date -u)"
DK
  setsid nohup bash -c 'bash /root/dolphinkeep.sh 2>&1 | tee -a /root/sb_dolphin.log' > /dev/null 2>&1 < /dev/null &
  echo "DOLPHIN_LAUNCHED $(date -u)"
fi
# ---- 2026-10-07 21:40 JST (the user: if this works, a 1 GB-class model is fine - the capacity scheduler as before). The single
# e2e run is replaced by the ladder on the Dolphin conversations: negatives -> bge layers -> BART layers (doubling, up to 64),
# a rung kept on the pool gain (20,000 conversations), the best per capacity on the hub under sentbart/dolphin. No real-query
# guard (not Wikipedia). The stock vectors are re-stored as int8 first (the disk: 4 GB free, large checkpoints to come).
if [ ! -e /root/.dolphin_ladder ] && [ -e /root/.dolphin_e2e ] && grep -q -- "--guard" /root/sb/e2e_ladder.py; then touch /root/.dolphin_ladder
  cat > /root/dolphinladderkeep.sh <<'DL'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/sb; D=/root/sb/data/dolphin
pkill -f "dolphinkee[p].sh"; pkill -f "train_e2e.py.*/root/sb/dolphin_e2e"; sleep 15
echo "DOLPHIN_JOB_DONE replaced by the ladder $(date -u)" >> /root/sb_dolphin.log
until [ -s $D/vec_001.npy ] && ! pgrep -f "embed.py --dir $D" >/dev/null; do sleep 60; done
if [ ! -s $D/scl_000.npy ]; then
  echo "[dolphinladder] int8 vectors $(date -u +%H:%M); $(df -h /root | tail -1 | awk '{print $4}') free"
  mkdir -p $D/q8; ln -sf $D/docs_000.jsonl $D/q8/docs_000.jsonl; ln -sf $D/docs_001.jsonl $D/q8/docs_001.jsonl
  python3 /root/sb/embed.py --dir $D/q8 --int8 1 2>&1 | grep -E "^\[embed\]|EMBED_DONE|Error" | cut -c1-200
  [ -s $D/q8/scl_001.npy ] && { rm -f $D/vec_000.npy $D/vec_001.npy $D/off_000.npy $D/off_001.npy; mv $D/q8/vec_*.npy $D/q8/scl_*.npy $D/q8/off_*.npy $D/; rm -rf $D/q8; }
fi
[ -s $D/scl_001.npy ] || { echo "DOLPHINLADDER_JOB_DONE no int8 vectors"; exit 1; }
rm -rf /root/sb/dolphin_e2e
I0=/root/sb/abl_one32/model_best.pt; [ -s $I0 ] || I0=/root/sb/hfdl/sentbart/abl/one32/model_best.pt
echo "[dolphinladder] from $I0 $(date -u +%H:%M); $(df -h /root | tail -1 | awk '{print $4}') free"
python3 -u /root/sb/e2e_ladder.py --start $I0 --work /root/sb/dolphin_ladder --vec $D --text $D --eval-text $D/docs_001.jsonl --pool 20000 \
  --guard 0 --anchor-q 0 --hub sentbart/dolphin --max-layers 64 --max-enc 20 --select pool_top1 2>&1 | grep --line-buffered -E "^\[ladder\]|LADDER_DONE|Error|Traceback|out of memory" | sed -u 's/^\[ladder\]/[dolphinladder]/'
echo "DOLPHINLADDER_JOB_DONE $(date -u)"
DL
  setsid nohup bash -c 'bash /root/dolphinladderkeep.sh 2>&1 | tee -a /root/sb_dolphinladder.log' > /dev/null 2>&1 < /dev/null &
  echo "DOLPHINLADDER_LAUNCHED $(date -u)"
fi
# ---- 2026-10-07 22:10 JST (the user: why store vectors at all - e2e embeds on the fly): train.py now takes a shard's layout alone
# (off_XXX.npy). Once the ladder's first training process is up (its mmap keeps what it opened), the stored vectors go.
if [ ! -e /root/.dolphin_novec ] && [ -e /root/.dolphin_ladder ] && grep -q "class NoVec" /root/sb/train.py; then touch /root/.dolphin_novec
  ( D=/root/sb/data/dolphin; until pgrep -f "train_e2e.py.*/root/sb/dolphin_ladder/" >/dev/null; do sleep 60; done; sleep 120
    rm -f $D/vec_000.npy $D/vec_001.npy $D/scl_000.npy $D/scl_001.npy; rm -rf $D/q8
    echo "[dolphinladder] stored vectors removed $(date -u +%H:%M); $(df -h /root | tail -1 | awk '{print $4}') free" >> /root/sb_dolphinladder.log ) > /dev/null 2>&1 &
fi
# ---- 2026-10-07 22:25 JST: the int8 re-embedding failed (the disk); with train.py taking the layout alone, the vectors are simply
# dropped and the ladder launched on the text + off files.
if [ ! -e /root/.dolphin_ladder2 ] && [ -e /root/.dolphin_ladder ] && grep -q "class NoVec" /root/sb/train.py; then touch /root/.dolphin_ladder2
  cat > /root/dolphinladderkeep.sh <<'DL'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/sb; D=/root/sb/data/dolphin
pkill -f "dolphinkee[p].sh"; pkill -f "train_e2e.py.*/root/sb/dolphin_e2e"; pkill -f "embed.py --dir $D"; sleep 10
rm -f $D/vec_000.npy $D/vec_001.npy $D/scl_000.npy $D/scl_001.npy; rm -rf $D/q8 /root/sb/dolphin_e2e
[ -s $D/off_000.npy ] && [ -s $D/off_001.npy ] || python3 - <<'PO'
import json, numpy as np
for k in ("000", "001"):
    n = [len(json.loads(l)["sents"]) for l in open(f"/root/sb/data/dolphin/docs_{k}.jsonl") if l.strip()]
    np.save(f"/root/sb/data/dolphin/off_{k}.npy", np.cumsum([0] + n).astype(np.int64))
PO
[ -s $D/off_001.npy ] || { echo "DOLPHINLADDER_JOB_DONE no layout"; exit 1; }
I0=/root/sb/abl_one32/model_best.pt; [ -s $I0 ] || I0=/root/sb/hfdl/sentbart/abl/one32/model_best.pt
echo "[dolphinladder] from $I0 (text + layout, no stored vectors) $(date -u +%H:%M); $(df -h /root | tail -1 | awk '{print $4}') free"
python3 -u /root/sb/e2e_ladder.py --start $I0 --work /root/sb/dolphin_ladder --vec $D --text $D --eval-text $D/docs_001.jsonl --pool 20000 \
  --guard 0 --anchor-q 0 --hub sentbart/dolphin --max-layers 64 --max-enc 20 --select pool_top1 2>&1 | grep --line-buffered -E "^\[ladder\]|LADDER_DONE|Error|Traceback|out of memory" | sed -u 's/^\[ladder\]/[dolphinladder]/'
echo "DOLPHINLADDER_JOB_DONE $(date -u)"
DL
  setsid nohup bash -c 'bash /root/dolphinladderkeep.sh 2>&1 | tee -a /root/sb_dolphinladder.log' > /dev/null 2>&1 < /dev/null &
  echo "DOLPHINLADDER_RELAUNCHED $(date -u)"
fi
# ---- 2026-10-08 00:00 JST: the Vast credit ran out and the box stopped six minutes into r01. After the restart the ladder keeper
# is relaunched (e2e_ladder.py resumes from ladder.json: r00_ref again, briefly, then r01 from the start).
if [ -s /root/dolphinladderkeep.sh ] && ! pgrep -f "dolphinladderkee[p].sh" >/dev/null && grep -q "text + layout" /root/sb_dolphinladder.log 2>/dev/null \
   && ! grep -q "DOLPHINLADDER_JOB_DONE [A-Z][a-z][a-z] " /root/sb_dolphinladder.log 2>/dev/null; then
  echo "[dolphinladder] relaunched after the box stopped $(date -u +%H:%M)" >> /root/sb_dolphinladder.log
  setsid nohup bash -c 'bash /root/dolphinladderkeep.sh 2>&1 | tee -a /root/sb_dolphinladder.log' > /dev/null 2>&1 < /dev/null &
fi
# ---- 2026-10-08 12:40 JST: e2e_ladder.py retries a rung that runs out of GPU memory with half the batch (the BART-doubling rungs
# on this 12 GB card). The running ladder loaded the old file; it is restarted once at the first rung boundary after r01
# (ladder.json then holds r01, and a rung resumes from its own state), losing minutes at most.
if [ ! -e /root/.dolphin_ladder_oom ] && grep -q "def run1" /root/sb/e2e_ladder.py; then touch /root/.dolphin_ladder_oom
  ( until grep -qE "^\[dolphinladder\] r0[2-9]_" /root/sb_dolphinladder.log 2>/dev/null; do sleep 120; done; sleep 60
    pkill -f "dolphinladderkee[p].sh"; pkill -f "e2e_ladder.py --start"; pkill -f "train_e2e.py.*/root/sb/dolphin_ladder/"; sleep 20
    echo "[dolphinladder] restarted for the out-of-memory retry $(date -u +%H:%M)" >> /root/sb_dolphinladder.log
    setsid nohup bash -c 'bash /root/dolphinladderkeep.sh 2>&1 | tee -a /root/sb_dolphinladder.log' > /dev/null 2>&1 < /dev/null & ) > /dev/null 2>&1 &
fi
# ---- 2026-10-08 17:30 JST (the user: what matters is the reconstruction of the context-vector sequence, not finding the
# conversation a sentence came from). The Dolphin ladder selects and judges rungs on recon_top1 = the mean of the masked-
# sentence (encoder) and hidden-tail (decoder) top-1 among the eval batch's sentences. Restarted now: r01 resumes from its
# step-16000 state, r00_ref is measured again on the new metric.
if [ ! -e /root/.dolphin_recon ] && grep -q "recon_top1" /root/sb/train_e2e.py; then touch /root/.dolphin_recon
  pkill -f "dolphinladderkee[p].sh"; pkill -f "e2e_ladder.py --start"; pkill -f "train_e2e.py.*/root/sb/dolphin_ladder/"; sleep 20
  sed -i 's/--select pool_top1/--select recon_top1/' /root/dolphinladderkeep.sh
  grep -q -- "--select recon_top1" /root/dolphinladderkeep.sh && echo "[dolphinladder] restarted to select on the reconstruction (recon_top1) $(date -u +%H:%M)" >> /root/sb_dolphinladder.log
  setsid nohup bash -c 'bash /root/dolphinladderkeep.sh 2>&1 | tee -a /root/sb_dolphinladder.log' > /dev/null 2>&1 < /dev/null &
fi
# ---- 2026-10-08 19:40 JST (the user: in use it is the continuation that matters). The Dolphin ladder selects and judges rungs
# on dec_top1 - each hidden tail sentence predicted from the true ones before it, named among the eval batch's sentences.
# Restarted just after r01's step-20000 save, so nothing is lost.
if [ ! -e /root/.dolphin_dec ] && grep -q -- "--select recon_top1" /root/dolphinladderkeep.sh 2>/dev/null; then touch /root/.dolphin_dec
  ( until grep -qE "^\[step 20[1-9]00\]" /root/sb/dolphin_ladder/r01_e2e/train.log 2>/dev/null || grep -qE "^\[dolphinladder\] r0[2-9]_" /root/sb_dolphinladder.log; do sleep 60; done
    pkill -f "dolphinladderkee[p].sh"; pkill -f "e2e_ladder.py --start"; pkill -f "train_e2e.py.*/root/sb/dolphin_ladder/"; sleep 20
    sed -i 's/--select recon_top1/--select dec_top1/' /root/dolphinladderkeep.sh
    grep -q -- "--select dec_top1" /root/dolphinladderkeep.sh && echo "[dolphinladder] restarted to select on the continuation (dec_top1) $(date -u +%H:%M)" >> /root/sb_dolphinladder.log
    setsid nohup bash -c 'bash /root/dolphinladderkeep.sh 2>&1 | tee -a /root/sb_dolphinladder.log' > /dev/null 2>&1 < /dev/null & ) > /dev/null 2>&1 &
fi
# ---- 2026-10-09 00:30 JST (the user: double the layers first). The Dolphin ladder's turn becomes BART x2 -> bge +2 -> negatives.
# The running ladder is left alone: the keeper is edited, and the restart already armed for the first rung after r01 (the
# out-of-memory retry) relaunches it with the new code and order - its next rung is then the BART doubling.
if [ ! -e /root/.dolphin_order ] && grep -q -- "--order" /root/sb/e2e_ladder.py; then touch /root/.dolphin_order
  sed -i 's/--select dec_top1 /--select dec_top1 --order bart,bge,neg /' /root/dolphinladderkeep.sh
  grep -q -- "--order bart,bge,neg" /root/dolphinladderkeep.sh && echo "[dolphinladder] the next rungs: BART x2 first, then bge +2, then negatives $(date -u +%H:%M)" >> /root/sb_dolphinladder.log
fi
# ---- 2026-10-09 05:30 JST: r02_bart ran out of memory at batch 32 and was retried at 16 - and its evaluation shrank with it
# (16 documents, 1,624 candidates instead of 3,223), so its step 0 read 16.6% against r01's 16.0% with nothing learnt.
# The evaluation batch is now fixed at 32 (train.py --eval-batch); r02 starts again on the fixed yardstick.
if [ ! -e /root/.dolphin_evalbatch ] && grep -q -- "--eval-batch" /root/sb/train_e2e.py; then touch /root/.dolphin_evalbatch
  pkill -f "dolphinladderkee[p].sh"; pkill -f "e2e_ladder.py --start"; pkill -f "train_e2e.py.*/root/sb/dolphin_ladder/"; sleep 20
  rm -rf /root/sb/dolphin_ladder/r02_bart /root/sb/dolphin_ladder/r02_bart.out
  echo "[dolphinladder] restarted: the evaluation batch fixed at 32 (r02 had measured on 16) $(date -u +%H:%M)" >> /root/sb_dolphinladder.log
  setsid nohup bash -c 'bash /root/dolphinladderkeep.sh 2>&1 | tee -a /root/sb_dolphinladder.log' > /dev/null 2>&1 < /dev/null &
fi
# ---- 2026-10-10 18:00 JST: the Dolphin held-out shard's text on the hub, once (the GPT-2 rerank estimate - how far the
# context BART has to go when GPT-2 writes each sentence from its vector - runs off the box, on CPU, from it)
if [ ! -e /root/.dl001_up ] && [ -s /root/sb/data/dolphin/docs_001.jsonl ]; then touch /root/.dl001_up
  ( hf upload $R /root/sb/data/dolphin/docs_001.jsonl sentbart/dolphin/data/docs_001.jsonl >/dev/null 2>&1 && echo "DL001_UP $(ls -la /root/sb/data/dolphin/docs_001.jsonl | awk '{print $5}') $(date -u)" ) >> /root/boxI_extra.log 2>&1 &
fi
# ---- 2026-10-11 00:30 JST: the disk ran out under r05 (BART 32+32: its state file and checkpoints at step 4000), which
# died writing model_best.pt and was scored as "not kept". The capacity copies already on the hub (cap_bge12_bart8,
# cap_bge12_bart16, cap_bge14_bart16 = cur.pt) are dropped locally, and what holds the rest of the disk goes to the hub.
if [ ! -e /root/.disk_r05 ]; then touch /root/.disk_r05
  for c in cap_bge12_bart8 cap_bge12_bart16 cap_bge14_bart16; do rm -f /root/sb/dolphin_ladder/$c/model_best.pt; done
  { date -u; df -h /root | tail -1; du -xsh /root/* /root/sb/* /root/sb/data/* /root/sb/dolphin_ladder/* 2>/dev/null | sort -h | tail -40; } > /root/du_I.txt 2>&1
  hf upload $R /root/du_I.txt sentbart/audit/du_I.txt >/dev/null 2>&1 &
fi
if [ ! -e /root/.disk_r05b ]; then touch /root/.disk_r05b   # du counts a path once across its arguments: one call per level
  { date -u; df -h /root | tail -1; for d in /root/sb /root/sb/data /root/sb/dolphin_ladder; do echo "== $d"; du -xsh $d/* 2>/dev/null | sort -h | tail -15; done; } > /root/du_I.txt 2>&1
  hf upload $R /root/du_I.txt sentbart/audit/du_I.txt >/dev/null 2>&1 &
fi
# ---- 2026-10-11 00:40 JST: what the Dolphin ladder left behind: r01's checkpoints (its best is on the hub as cap_bge12_bart8),
# r00_ref's (the start's own reading) and r05's broken write (BART 32+32 died at step 4000 on the full disk) - about 3.6 GB.
if [ ! -e /root/.disk_r05c ]; then touch /root/.disk_r05c
  rm -f /root/sb/dolphin_ladder/r01_e2e/*.pt /root/sb/dolphin_ladder/r00_ref/*.pt /root/sb/dolphin_ladder/r05_bart/*.pt
  { date -u; df -h /root | tail -1; } >> /root/du_I.txt; hf upload $R /root/du_I.txt sentbart/audit/du_I.txt >/dev/null 2>&1 &
fi
# ---- 2026-10-11 00:50 JST (the user: delete the Wikipedia data). The Wikipedia shards' text and vectors (data/docs, 22 GB)
# and the e2e text (data/e2e) go; the Dolphin ladder reads data/dolphin only. The keepers that built them are guarded by
# their logs (DATA_JOB_DONE ...), so nothing rebuilds them.
if [ ! -e /root/.wiki_rm ] && ! pgrep -f "sb/data/docs|sb/data/e2e" >/dev/null; then touch /root/.wiki_rm
  rm -rf /root/sb/data/docs /root/sb/data/e2e
  { date -u; echo "wiki data removed"; df -h /root | tail -1; } >> /root/du_I.txt; hf upload $R /root/du_I.txt sentbart/audit/du_I.txt >/dev/null 2>&1 &
fi
# ---- 2026-10-11 01:00 JST (the user: run the BART doubling again). r05 (BART 32+32) died at step 4000 on the full disk while
# still rising (22.1 -> 22.7) and was scored "not kept"; the disk now has 31 GB free. r06 (bge 16, a few hundred steps in)
# is stopped, ladder.json's turn is put back on the BART doubling with r05's failure undone, and the ladder resumes from
# cur.pt (r03/r04's 16+16) - the next rung is BART 32+32 from scratch.
if [ ! -e /root/.dolphin_redo_bart ] && [ -s /root/sb/dolphin_ladder/ladder.json ]; then touch /root/.dolphin_redo_bart
  pkill -f "dolphinladderkee[p].sh"; pkill -f "e2e_ladder.py --start"; pkill -f "train_e2e.py.*/root/sb/dolphin_ladder/"; sleep 20
  python3 - <<'PJ' >> /root/sb_dolphinladder.log 2>&1
import json
p = "/root/sb/dolphin_ladder/ladder.json"; rec = json.load(open(p)); cur = rec["cur"]
order = ["bart", "bge", "neg"]
while order[cur["turn"] % 3] != "bart": cur["turn"] -= 1
cur["fails"] = max(0, cur["fails"] - 1)
json.dump(rec, open(p, "w"), indent=1)
print(f"[dolphinladder] turn put back on the BART doubling (turn {cur['turn']}, fails {cur['fails']}, from {cur['name']} at {cur['pool_top1']:.3f})", flush=True)
PJ
  rm -rf /root/sb/dolphin_ladder/r06_bge /root/sb/dolphin_ladder/r06_bge.out /root/sb/dolphin_ladder/cap_bge14_bart32
  echo "[dolphinladder] restarted to run the BART doubling again (r05 died on the full disk) $(date -u +%H:%M); $(df -h /root | tail -1 | awk '{print $4}') free" >> /root/sb_dolphinladder.log
  setsid nohup bash -c 'bash /root/dolphinladderkeep.sh 2>&1 | tee -a /root/sb_dolphinladder.log' > /dev/null 2>&1 < /dev/null &
fi
echo "BOXI_OK serial $BOXI_SERIAL $(date -u)"
# CTL-END

