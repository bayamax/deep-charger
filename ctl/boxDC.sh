#!/bin/bash
# box DC (2026-10-16, a second 3060 beside box CH): the context BART's decoder. The BART (run 7) reads a question and the
# sentences so far and predicts the next sentence's vector; an MLP turns that vector into GPT-2 prefix embeddings and GPT-2
# writes the sentence (sentbart/dec_pairs.py, dec_train.py). Two decoders side by side on the one card: A the vector alone,
# B the vector plus the previous sentence's raw text (in case names do not survive the vector). Pull-based as the others;
# the log goes to the hub every 10 minutes (boxlog_DC.txt), the decoders to sentbart/dec/.
BOXDC_SERIAL=3
if [ -f /root/.boxdc_serial ] && [ "$(cat /root/.boxdc_serial)" -gt "$BOXDC_SERIAL" ] 2>/dev/null; then echo "BOXDC_STALE $BOXDC_SERIAL"; exit 0; fi
echo $BOXDC_SERIAL > /root/.boxdc_serial
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
mkdir -p /root/dec /root/hfdl
for f in sentbart/train.py sentbart/prep_dolphin.py sentbart/dec_pairs.py sentbart/dec_train.py; do
  for try in 1 2 3; do curl -sS -o /root/dec/$(basename $f) "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/dec/$(basename $f) && break; sleep 5; done
done

if [ ! -f /root/.bootstrapped ]; then
  mkdir /root/.bootstrap_lock 2>/dev/null || { echo "bootstrap already running"; exit 0; }
  pip install -q "transformers>=4.51,<4.57" "huggingface_hub>=0.34,<1.0" safetensors blingfire zstandard 2>&1 | tail -1
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /root | tail -1; free -g | head -2; nproc
  touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"
fi

if [ ! -e /root/.dcmirror_r3 ]; then touch /root/.dcmirror_r3; pkill -f "dcmirro[r].sh"; sleep 1; fi
if ! pgrep -f "dcmirro[r].sh" >/dev/null; then
  cat > /root/dcmirror.sh <<'MR'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
while true; do
  { echo "=== boxDC $(date -u) ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; df -h /root | tail -1; free -g | sed -n 2p
    echo "--- ctl.log ---"; tail -n 12 /root/ctl.log 2>/dev/null | cut -c1-300
    echo "--- job.log ---"; grep -vE "Warning|warn|it/s\]" /root/dec/job.log 2>/dev/null | tail -n 25 | cut -c1-400
    echo "--- processes ---"; ps -eo pid,etimes,pcpu,rss,args | grep "[d]ec_train.py" | cut -c1-160; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
    for r in A B; do echo "--- run$r.out (tail) ---"; grep -vE "Warning|warn|it/s\]" /root/dec/run$r.out 2>/dev/null | tail -n 6 | cut -c1-300; done
    for r in A B; do echo "--- run$r/train.log ---"; grep -vE "^\[step" /root/dec/run$r/train.log 2>/dev/null | tail -n 40 | cut -c1-500; grep -E "^\[step" /root/dec/run$r/train.log 2>/dev/null | tail -n 2; done; } > /root/boxlog_DC.txt
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='/root/boxlog_DC.txt', path_in_repo='pooler_distill/chatsft/audit/boxlog_DC.txt', repo_id='baya1116/hypernet-sp-distill')" >/dev/null 2>&1
  sleep 600
done
MR
  setsid nohup bash /root/dcmirror.sh > /dev/null 2>&1 < /dev/null &
fi

if [ ! -e /root/.dcjob ]; then touch /root/.dcjob
  cat > /root/dcjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/dec; mkdir -p data pairs
dl() { for try in 1 2 3 4; do [ -s $1 ] && [ "$(stat -c %s $1)" -ge $3 ] && return 0; curl -sSfL -C - -o $1 $2 || sleep 60; done; [ "$(stat -c %s $1 2>/dev/null || echo 0)" -ge $3 ]; }
[ -s data/model_best.pt ] || { hf download $R sentbart/dolphin/once7/model_best.pt sentbart/dolphin/heldout/am_held_docs_005.jsonl sentbart/dolphin/heldout/dolphin768full_docs_001.jsonl sentbart/dolphin/data/docs_001.jsonl --local-dir /root/hfdl >/dev/null 2>&1
  mv /root/hfdl/sentbart/dolphin/once7/model_best.pt data/; mv /root/hfdl/sentbart/dolphin/heldout/am_held_docs_005.jsonl data/am_held.jsonl; mv /root/hfdl/sentbart/dolphin/heldout/dolphin768full_docs_001.jsonl data/dolphin_held.jsonl; mv /root/hfdl/sentbart/dolphin/data/docs_001.jsonl data/dolphin_docs_001.jsonl; }
echo "[dcjob] BART and held-outs: $(ls -la data | wc -l) files $(date -u)"
if [ ! -s data/docs_007.jsonl ]; then
  Z=data/am_0.9M.jsonl.zst
  dl $Z https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M/resolve/main/am_0.9M.jsonl.zst 3200000000 || { echo "DCJOB am download failed"; exit 1; }
  python3 -u prep_dolphin.py --out data --schema am --all 1 --shard 007 --id-prefix am1 --max-sents 768 --max-q 24 --max-docs 20000 \
    --keep-eval data/dolphin_docs_001.jsonl --exclude-questions data/dolphin_docs_001.jsonl data/am_held.jsonl --jsonl $Z 2>&1 | grep -E "PREP_DOLPHIN|Error|Traceback" | cut -c1-300
  rm -f $Z
fi
echo "[dcjob] training conversations: $(wc -l < data/docs_007.jsonl) $(date -u)"
[ -s pairs/pairs_train.npz ] || python3 -u dec_pairs.py --ckpt data/model_best.pt --docs data/docs_007.jsonl --name train --out pairs --max-t 96 --batch 8 2>&1 | grep -E "^\[pairs\]|DONE|Error|Traceback" | grep -vE "/[0-9]+ conversations" | cut -c1-300
[ -s pairs/pairs_am_held.npz ] || python3 -u dec_pairs.py --ckpt data/model_best.pt --docs data/am_held.jsonl --name am_held --out pairs --max-docs 600 --max-t 96 2>&1 | grep -E "^\[pairs\] am_held: [0-9]+ pairs|Error|Traceback"
[ -s pairs/pairs_dolphin_held.npz ] || python3 -u dec_pairs.py --ckpt data/model_best.pt --docs data/dolphin_held.jsonl --name dolphin_held --out pairs --max-docs 600 --max-t 96 2>&1 | grep -E "^\[pairs\] dolphin_held: [0-9]+ pairs|Error|Traceback"
echo "[dcjob] pairs ready $(date -u)"
python3 -u dec_train.py --pairs pairs --train train --eval am_held,dolphin_held --ckpt data/model_best.pt --out runA --vec mix --raw-prev 0 --epochs 2 --hours 6 --hub $R:sentbart/dec/runA > runA.out 2>&1 &
sleep 120
python3 -u dec_train.py --pairs pairs --train train --eval am_held,dolphin_held --ckpt data/model_best.pt --out runB --vec mix --raw-prev 1 --epochs 2 --hours 6 --hub $R:sentbart/dec/runB > runB.out 2>&1 &
wait
grep -h "Error\|Traceback" runA.out runB.out | tail -5
echo "DCJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/dcjob.sh >> /root/dec/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "DCJOB_LAUNCHED $(date -u)"
fi
# ---- serial 2: both decoders ran out of the 12 GB card side by side (the vocabulary logits of every position, prefix and
# previous sentence included). dec_train.py now projects only the target positions; the pairs stay; the two runs again. ----
if [ ! -e /root/.dctrain2 ] && grep -q "pairs ready" /root/dec/job.log 2>/dev/null; then touch /root/.dctrain2
  cat > /root/dctrain2.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/dec
mkdir -p failed_oom; mv runA failed_oom/runA 2>/dev/null; mv runB failed_oom/runB 2>/dev/null
python3 -u dec_train.py --pairs pairs --train train --eval am_held,dolphin_held --ckpt data/model_best.pt --out runA --vec mix --raw-prev 0 --epochs 2 --hours 6 --hub $R:sentbart/dec/runA > runA.out 2>&1 &
sleep 240
python3 -u dec_train.py --pairs pairs --train train --eval am_held,dolphin_held --ckpt data/model_best.pt --out runB --vec mix --raw-prev 1 --epochs 2 --hours 6 --hub $R:sentbart/dec/runB > runB.out 2>&1 &
sleep 600; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed 's/^/[dctrain2] both running: GPU /'
wait
grep -h "Error\|Traceback" runA.out runB.out | tail -5
echo "DCTRAIN2_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/dctrain2.sh >> /root/dec/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[dctrain2] launched $(date -u)" | tee -a /root/dec/job.log
fi
echo "BOXDC_OK serial $BOXDC_SERIAL $(date -u)"
# CTL-END
