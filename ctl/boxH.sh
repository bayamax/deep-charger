# box H (RTX 3090 24GB): the sentence-sequence BART - each sentence of a document becomes one vector (a frozen
# sentence encoder), and a transformer is trained on the sequence of those vectors: an encoder that reads a
# corrupted sequence (spans of sentences masked) and a decoder that regenerates it one sentence vector at a time.
# The encoder's outputs are context-aware sentence / page vectors (search, hierarchy); the decoder generates in
# sentence-vector space. Separate from box G (the app model's multi-turn work); results go to the hub under sentbart/.
cd /root
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
R=baya1116/hypernet-sp-distill
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
BOXH_SERIAL=2
if [ -f /root/.boxh_serial ] && [ "$(cat /root/.boxh_serial)" -gt "$BOXH_SERIAL" ] 2>/dev/null; then echo "BOXH_STALE $BOXH_SERIAL"; exit 0; fi
echo $BOXH_SERIAL > /root/.boxh_serial
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
  { echo "=== boxH $(date -u) ==="; echo "--- ctl.log ---"; tail -n 60 /root/ctl.log 2>/dev/null | cut -c1-300
    for f in /root/sb_*.log; do [ -e $f ] && { echo "--- $f ---"; tail -n 25 $f | cut -c1-300; }; done
    echo "--- gpu ---"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
    echo "--- disk ---"; df -h /root | tail -1; du -sh /root/sb/data 2>/dev/null
    echo "--- processes ---"; pgrep -fa "python3 /root/sb/" | cut -c1-160; } > /root/boxlog.txt 2>&1
  hf upload $R /root/boxlog.txt sentbart/audit/boxlog_H.txt >/dev/null 2>&1
  sleep 600
done
MK
  setsid nohup bash /root/mirrorkeep.sh > /dev/null 2>&1 < /dev/null &
  echo "MIRROR_LAUNCHED $(date -u)"
fi
# ---- data: 10 of the 41 shards of English Wikipedia (about 1.5M articles) as sentence lists, then their vectors ----
if ! pgrep -f "datakee[p].sh" >/dev/null && ! grep -q "DATA_JOB_DONE" /root/sb_data.log 2>/dev/null; then
  cat > /root/datakeep.sh <<'DK'
cd /root/sb; export HF_HUB_ENABLE_HF_TRANSFER=0
python3 /root/sb/prep.py --out /root/sb/data/docs --shards 0-9 || { echo "DATA_ABORT prep"; exit 1; }
python3 /root/sb/embed.py --dir /root/sb/data/docs || { echo "DATA_ABORT embed"; exit 1; }
du -sh /root/sb/data/docs
echo "DATA_JOB_DONE $(date -u)"
DK
  setsid nohup bash -c 'bash /root/datakeep.sh 2>&1 | tee -a /root/sb_data.log' > /dev/null 2>&1 < /dev/null &
  echo "DATA_LAUNCHED $(date -u)"
fi
echo "BOXH_OK serial $BOXH_SERIAL $(date -u)"
# CTL-END
