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
BOXI_SERIAL=5
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
# ---- run4 (the goal is the page vector the hidden states give - a search system if it carries enough): a page token
# in front of the encoder, trained so each hidden sentence finds its own document's vector (--page 1), and measured as
# page retrieval over 2000 held-out documents (one sentence hidden in each, against the mean of the others' vectors).
# Data 2 -> 6 shards (5 to train, shard 5 held out), 100k steps.
if ! pgrep -f "run4kee[p].sh" >/dev/null && ! grep -q "RUN4_JOB_DONE\|RUN4_ABORT" /root/sb_run4.log 2>/dev/null; then
  cat > /root/run4keep.sh <<'RK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/sb
while pgrep -f "python3 /root/sb/" >/dev/null; do sleep 60; done
python3 /root/sb/prep.py --out /root/sb/data/docs --shards 2-5 > /root/sb/prep4.log 2>&1; grep -E "^\[prep\]" /root/sb/prep4.log
grep -q PREP_DONE /root/sb/prep4.log || { tail -5 /root/sb/prep4.log; echo "RUN4_ABORT prep"; exit 1; }
python3 /root/sb/embed.py --dir /root/sb/data/docs > /root/sb/embed4.log 2>&1; grep -E "^\[embed\]" /root/sb/embed4.log
grep -q EMBED_DONE /root/sb/embed4.log || { tail -5 /root/sb/embed4.log; echo "RUN4_ABORT embed"; exit 1; }
df -h /root | tail -1
echo "[run4] train start $(date -u +%H:%M)"
( while sleep 1800; do hf upload $R /root/sb/run4/train.log sentbart/small_run4/train.log >/dev/null 2>&1; done ) & UP=$!
python3 /root/sb/train.py --data /root/sb/data/docs --out /root/sb/run4 --steps 100000 --batch 32 --seq 128 --d 512 --layers 4 --heads 8 --ffn 2048 --warmup 1000 --eval-every 5000 --save-every 5000 --page 1 2>&1 | grep -E "^\[data\]|^\[model\]|^\[eval|TRAIN_DONE|Error|Traceback"
kill $UP 2>/dev/null
hf upload $R /root/sb/run4/train.log sentbart/small_run4/train.log >/dev/null 2>&1
hf upload $R /root/sb/run4/model_latest.pt sentbart/small_run4/model_latest.pt >/dev/null 2>&1
echo "RUN4_JOB_DONE $(date -u)"
RK
  setsid nohup bash -c 'bash /root/run4keep.sh 2>&1 | tee -a /root/sb_run4.log' > /dev/null 2>&1 < /dev/null &
  echo "RUN4_LAUNCHED $(date -u)"
fi
echo "BOXI_OK serial $BOXI_SERIAL $(date -u)"
# CTL-END
