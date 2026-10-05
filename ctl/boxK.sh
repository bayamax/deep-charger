# box K (RTX 3060 12GB, rented 2026-10-05, the user: "lossless neural compression of Wikipedia, then the sentence BART
# on that language model's sentence-end states"). A small causal LM (SmolLM2-135M, Apache-2.0) compresses articles
# losslessly (arithmetic coding of its next-token probabilities, round trip checked), and the same LM's final-layer
# state at each sentence's last token (the sentence alone, standardised, L2-normalised) replaces bge-small as the
# sentence vector the sentence-sequence BART is trained on. Results go to the hub under sentbart/lmz/.
cd /root
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
R=baya1116/hypernet-sp-distill
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
BOXK_SERIAL=5
if [ -f /root/.boxk_serial ] && [ "$(cat /root/.boxk_serial)" -gt "$BOXK_SERIAL" ] 2>/dev/null; then echo "BOXK_STALE $BOXK_SERIAL"; exit 0; fi
echo $BOXK_SERIAL > /root/.boxk_serial
mkdir -p /root/lz/docs

# ---- one-time bootstrap ----
if [ ! -f /root/.bootstrapped ]; then
  mkdir /root/.bootstrap_lock 2>/dev/null || { echo "bootstrap already running"; exit 0; }
  echo "=== bootstrap $(date -u) ==="
  pip install -q "transformers==4.44.2" "safetensors" "huggingface_hub>=0.34,<1.0" pyarrow blingfire numpy 2>&1 | tail -1
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  df -h /root | tail -1; free -g | head -2
  touch /root/.bootstrapped; rmdir /root/.bootstrap_lock
  echo "=== bootstrap done $(date -u) ==="
fi

# the code, fresh from the branch on every control run
for f in lmz/lmz.py lmz/artret.py sentbart/prep.py sentbart/train.py; do
  b=$(basename $f); curl -sS -L -o /root/lz/$b.new "$RAW/$f?$(date +%s)" && grep -q "^#!/usr/bin/env python3" /root/lz/$b.new && mv /root/lz/$b.new /root/lz/$b || rm -f /root/lz/$b.new
done
ls /root/lz

# ---- the mirror: what this box is doing, on the hub every 10 minutes ----
if ! pgrep -f "mirrorkee[p].sh" >/dev/null; then
  cat > /root/mirrorkeep.sh <<'MK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
while :; do
  { echo "=== boxK $(date -u) ==="; echo "--- ctl.log ---"; tail -n 40 /root/ctl.log 2>/dev/null | cut -c1-300
    for f in /root/lz_*.log; do [ -e $f ] && { echo "--- $f ---"; tail -n 25 $f | cut -c1-300; }; done
    echo "--- gpu ---"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
    echo "--- disk ---"; df -h /root | tail -1; du -sh /root/lz/docs 2>/dev/null
    echo "--- processes ---"; pgrep -fa "python3 /root/lz/" | cut -c1-160; } > /root/boxlog.txt 2>&1
  hf upload $R /root/boxlog.txt sentbart/audit/boxlog_K.txt >/dev/null 2>&1
  sleep 600
done
MK
  setsid nohup bash /root/mirrorkeep.sh > /dev/null 2>&1 < /dev/null &
  echo "MIRROR_LAUNCHED $(date -u)"
fi

# ---- the pipeline: Wikipedia shards 0-6 as sentence lists (shard 1 held out, the same 2000 evaluation articles box I
# uses), compression measured on held-out articles, sentence vectors from the LM, then the BART on them ----
if [ -f /root/.bootstrapped ] && ! pgrep -f "lzkee[p].sh" >/dev/null && ! grep -q "LZ_JOB_DONE" /root/lz_main.log 2>/dev/null; then
  cat > /root/lzkeep.sh <<'LK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/lz
echo "[lz] prep $(date -u +%H:%M)"
python3 /root/lz/prep.py --out /root/lz/docs --shards 0-6 2>&1 | grep -E "^\[prep\]|Error|Traceback" | tail -8
ls /root/lz/docs/docs_001.jsonl >/dev/null || { echo "LZ_JOB_DONE prep failed"; exit 1; }
echo "[lz] compression on held-out articles $(date -u +%H:%M)"
python3 /root/lz/lmz.py bpc --docs /root/lz/docs/docs_001.jsonl --n 300 2>&1 | grep -E "^\[bpc\]|BPC_DONE|Error|Traceback" | tee /root/lz/bpc.txt
python3 /root/lz/lmz.py roundtrip --docs /root/lz/docs/docs_001.jsonl --n 10 2>&1 | grep -E "^\[roundtrip\]|ROUNDTRIP_DONE|Error|Traceback" | tee /root/lz/roundtrip.txt
hf upload $R /root/lz/bpc.txt sentbart/lmz/bpc.txt >/dev/null 2>&1; hf upload $R /root/lz/roundtrip.txt sentbart/lmz/roundtrip.txt >/dev/null 2>&1
echo "[lz] sentence vectors $(date -u +%H:%M)"
python3 /root/lz/lmz.py vec --dir /root/lz/docs --shards 001,000,002,003,004,005,006 2>&1 | grep -E "^\[vec\]|VEC_DONE|Error|Traceback|out of memory"
ls /root/lz/docs/vec_006.npy >/dev/null || { echo "LZ_JOB_DONE vec failed"; exit 1; }
rm -f /root/lz/docs/docs_00[02-6].jsonl   # the vectors are what the BART reads; the held-out shard's text is kept
df -h /root | tail -1
echo "[lz] BART on the LM sentence vectors $(date -u +%H:%M)"
n=lmrun1
( while sleep 1800; do hf upload $R /root/lz/$n/train.log sentbart/lmz/$n/train.log >/dev/null 2>&1; done ) & UP=$!
python3 /root/lz/train.py --data /root/lz/docs --out /root/lz/$n --eval-shard 001 --steps 100000 --batch 32 --seq 128 --d 512 --layers 8 --heads 8 --ffn 2048 \
  --lr 1e-4 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-queue 64 --page-input one --skip-grad 100 2>&1 | grep -E "^\[data|^\[model|^\[best|TRAIN_DONE|Error|Traceback|out of memory"
kill $UP 2>/dev/null; hf upload $R /root/lz/$n/train.log sentbart/lmz/$n/train.log >/dev/null 2>&1; hf upload $R /root/lz/$n/model_best.pt sentbart/lmz/$n/model_best.pt >/dev/null 2>&1
rm -f /root/lz/$n/state.pt
echo "LZ_JOB_DONE $(date -u)"
LK
  setsid nohup bash -c 'bash /root/lzkeep.sh 2>&1 | tee -a /root/lz_main.log' > /dev/null 2>&1 < /dev/null &
  echo "LZ_LAUNCHED $(date -u)"
fi
# ---- lz2 (2026-10-05 14:30 JST): compression done (held-out 300 articles: 0.955 bits/byte = 11.9% of raw, zlib 44.0%,
# lzma 45.5%; 10/10 articles back byte for byte). The vector pass died of CUDA OOM - it asked the LM for 49k-wide
# logits it never uses; fixed (the body only). Vectors and the BART again.
if [ -f /root/.bootstrapped ] && grep -q "LZ_JOB_DONE vec failed" /root/lz_main.log 2>/dev/null && ! pgrep -f "lz2kee[p].sh" >/dev/null && ! grep -q "LZ2_JOB_DONE" /root/lz_main.log 2>/dev/null; then
  cat > /root/lz2keep.sh <<'LK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/lz
grep -q "lm.model(input_ids" /root/lz/lmz.py || { echo "LZ2_JOB_DONE stale lmz.py"; exit 1; }
rm -f /root/lz/docs/vec_*.npy /root/lz/docs/scl_*.npy /root/lz/docs/off_*.npy /root/lz/docs/lm_stats.npz
echo "[lz2] sentence vectors $(date -u +%H:%M)"
python3 /root/lz/lmz.py vec --dir /root/lz/docs --shards 001,000,002,003,004,005,006 --batch 512 2>&1 | grep -E "^\[vec\]|VEC_DONE|Error|Traceback|out of memory"
ls /root/lz/docs/vec_006.npy >/dev/null || { echo "LZ2_JOB_DONE vec failed"; exit 1; }
rm -f /root/lz/docs/docs_00[02-6].jsonl
df -h /root | tail -1
echo "[lz2] BART on the LM sentence vectors $(date -u +%H:%M)"
n=lmrun1
( while sleep 1800; do hf upload $R /root/lz/$n/train.log sentbart/lmz/$n/train.log >/dev/null 2>&1; done ) & UP=$!
python3 /root/lz/train.py --data /root/lz/docs --out /root/lz/$n --eval-shard 001 --steps 100000 --batch 32 --seq 128 --d 512 --layers 8 --heads 8 --ffn 2048 \
  --lr 1e-4 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-queue 64 --page-input one --skip-grad 100 2>&1 | grep -E "^\[data|^\[model|^\[best|TRAIN_DONE|Error|Traceback|out of memory"
kill $UP 2>/dev/null; hf upload $R /root/lz/$n/train.log sentbart/lmz/$n/train.log >/dev/null 2>&1; hf upload $R /root/lz/$n/model_best.pt sentbart/lmz/$n/model_best.pt >/dev/null 2>&1
rm -f /root/lz/$n/state.pt
echo "LZ2_JOB_DONE $(date -u)"
LK
  setsid nohup bash -c 'bash /root/lz2keep.sh 2>&1 | tee -a /root/lz_main.log' > /dev/null 2>&1 < /dev/null &
  echo "LZ2_LAUNCHED $(date -u)"
fi
# ---- artret (2026-10-05 16:20 JST, the user: an index of one hidden state per article, at its end): Natural Questions
# test questions against 30k articles of shards 0-6 (the gold ones in them + random others); the LM's article-end state
# (plain, with a closing prompt, mean) vs bge-small. Beside the vector pass (small, ~3 GB of GPU).
if [ -f /root/.bootstrapped ] && [ -f /root/lz/artret.py ] && ! pgrep -f "artretkee[p].sh" >/dev/null && ! grep -q "ARTRET_JOB_DONE" /root/lz_artret.log 2>/dev/null; then
  cat > /root/artretkeep.sh <<'LK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/lz
ls /root/lz/docs/docs_006.jsonl >/dev/null 2>&1 || [ -s /root/lz/artret/pool.jsonl ] || { echo "ARTRET_JOB_DONE no documents"; exit 1; }
echo "[artret] start $(date -u +%H:%M)"
python3 /root/lz/artret.py --docs /root/lz/docs --pool 30000 --out /root/lz/artret 2>&1 | grep -E "^\[artret\]|ARTRET_DONE|Error|Traceback|out of memory"
hf upload $R /root/lz/artret/result.txt sentbart/lmz/artret_result.txt >/dev/null 2>&1
echo "ARTRET_JOB_DONE $(date -u)"
LK
  setsid nohup bash -c 'bash /root/artretkeep.sh 2>&1 | tee -a /root/lz_artret.log' > /dev/null 2>&1 < /dev/null &
  echo "ARTRET_LAUNCHED $(date -u)"
fi
# ---- lz3 (2026-10-05 19:05 JST, the user: a first read does not need all seven shards). The vectors stop once shard
# 002 is written (001 held out, 000 + 002 for training: ~230k documents); the BART runs bgectl1's recipe and schedule
# (lr 1e-4, cosine over 100k) but is stopped after its 30k evaluation, so its 10k / 20k / 30k compare with bgectl1's
# at the same steps (bge, 6 training shards: 26.1 / 33.2 / 38.3% page top-1).
if [ -f /root/.bootstrapped ] && pgrep -f "lz2kee[p].sh" >/dev/null && ! pgrep -f "lz3kee[p].sh" >/dev/null && ! grep -q "LZ3_JOB_DONE" /root/lz_main.log 2>/dev/null; then
  cat > /root/lz3keep.sh <<'LK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/lz
until [ -s /root/lz/docs/vec_002.npy ] && [ -s /root/lz/docs/off_002.npy ]; do sleep 60; done; sleep 30
pkill -f "lz2kee[p].sh"; pkill -f "lmz.py vec"; sleep 10
echo "[lz3] vectors stopped after shard 002 $(date -u +%H:%M): $(ls /root/lz/docs | grep -E '^vec_' | tr '\n' ' ')"
n=lmrun1; rm -rf /root/lz/$n
( while sleep 1800; do hf upload $R /root/lz/$n/train.log sentbart/lmz/$n/train.log >/dev/null 2>&1; done ) & UP=$!
python3 /root/lz/train.py --data /root/lz/docs --out /root/lz/$n --eval-shard 001 --steps 100000 --batch 32 --seq 128 --d 512 --layers 8 --heads 8 --ffn 2048 \
  --lr 1e-4 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-queue 64 --page-input one --skip-grad 100 > /root/lz/$n.out 2>&1 &
TP=$!
while kill -0 $TP 2>/dev/null; do
  grep -q "^\[eval 30000\]" /root/lz/$n/train.log 2>/dev/null && { sleep 5; kill $TP; break; }; sleep 60
done
grep -hE "^\[data|^\[model|Error|Traceback|out of memory" /root/lz/$n.out | tail -4
grep -hE "^\[eval" /root/lz/$n/train.log | sed -E 's/(page_top1 [0-9.]+ page_top10 [0-9.]+).*(next_top1 [0-9.]+ next_top10 [0-9.]+).*/\1 \2/' | sed 's/^/[lz3] /'
kill $UP 2>/dev/null; hf upload $R /root/lz/$n/train.log sentbart/lmz/$n/train.log >/dev/null 2>&1; hf upload $R /root/lz/$n/model_best.pt sentbart/lmz/$n/model_best.pt >/dev/null 2>&1
rm -f /root/lz/$n/state.pt
echo "LZ3_JOB_DONE $(date -u)"
LK
  setsid nohup bash -c 'bash /root/lz3keep.sh 2>&1 | tee -a /root/lz_main.log' > /dev/null 2>&1 < /dev/null &
  echo "LZ3_LAUNCHED $(date -u)"
fi
# ---- lz4 (2026-10-05 22:00 JST, the user: the sentence vectors were cut short - scale it up and train properly). The
# LM-vector BART beat the bge one at every matched step on a third of the data (30k: 40.0 vs 38.3 page top-1). Now
# shards 0-14 (bge's main run used 15): prep 7-14, LM sentence vectors for 003-014 (~1.8 h a shard), then the BART from
# lmrun1's best on all 14 training shards, 200k steps at lr 1e-4 (cosine), evaluated every 10k on shard 001's 2000.
if [ -f /root/.bootstrapped ] && grep -q "LZ3_JOB_DONE" /root/lz_main.log 2>/dev/null && ! pgrep -f "lz4kee[p].sh" >/dev/null && ! grep -q "LZ4_JOB_DONE" /root/lz_main.log 2>/dev/null; then
  cat > /root/lz4keep.sh <<'LK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/lz
python3 /root/lz/prep.py --out /root/lz/docs --shards 7-14 2>&1 | grep --line-buffered -E "^\[prep\] shard|Error|Traceback"
rm -rf /root/.cache/huggingface/hub/datasets--wikimedia--wikipedia
SH=$(ls /root/lz/docs | grep -oE '^docs_[0-9]+' | sed 's/docs_//' | sort | tr '\n' ',' | sed 's/,$//')
echo "[lz4] sentence vectors for $SH $(date -u +%H:%M); $(df -h /root | tail -1 | awk '{print $4}') free"
python3 -u /root/lz/lmz.py vec --dir /root/lz/docs --shards $SH --batch 512 2>&1 | grep --line-buffered -E "^\[vec\]|VEC_DONE|Error|Traceback|out of memory"
ls /root/lz/docs/vec_014.npy >/dev/null || { echo "LZ4_JOB_DONE vec failed"; exit 1; }
n=lmrun2; echo "[lz4] BART $n from lmrun1 best, $(ls /root/lz/docs | grep -c '^vec_') shards $(date -u +%H:%M)"
( while sleep 1800; do hf upload $R /root/lz/$n/train.log sentbart/lmz/$n/train.log >/dev/null 2>&1; done ) & UP=$!
python3 /root/lz/train.py --data /root/lz/docs --out /root/lz/$n --init /root/lz/lmrun1/model_best.pt --eval-shard 001 --steps 200000 --batch 32 --seq 128 --d 512 --layers 8 --heads 8 --ffn 2048 \
  --lr 1e-4 --warmup 1000 --eval-every 10000 --save-every 10000 --page 1 --page-queue 64 --page-input one --skip-grad 100 2>&1 | grep --line-buffered -E "^\[data|^\[model|^\[init|^\[eval|^\[best|TRAIN_DONE|Error|Traceback|out of memory" | sed -E 's/(page_top1 [0-9.]+ page_top10 [0-9.]+).*(next_top1 [0-9.]+ next_top10 [0-9.]+).*/\1 \2/'
kill $UP 2>/dev/null; hf upload $R /root/lz/$n/train.log sentbart/lmz/$n/train.log >/dev/null 2>&1; hf upload $R /root/lz/$n/model_best.pt sentbart/lmz/$n/model_best.pt >/dev/null 2>&1
rm -f /root/lz/$n/state.pt
echo "LZ4_JOB_DONE $(date -u)"
LK
  setsid nohup bash -c 'bash /root/lz4keep.sh 2>&1 | tee -a /root/lz_main.log' > /dev/null 2>&1 < /dev/null &
  echo "LZ4_LAUNCHED $(date -u)"
fi
echo "BOXK_OK serial $BOXK_SERIAL $(date -u)"
# CTL-END
