#!/bin/bash
# box DC (2026-10-16, a second 3060 beside box CH): the context BART's decoder. The BART (run 7) reads a question and the
# sentences so far and predicts the next sentence's vector; an MLP turns that vector into GPT-2 prefix embeddings and GPT-2
# writes the sentence (sentbart/dec_pairs.py, dec_train.py). Two decoders side by side on the one card: A the vector alone,
# B the vector plus the previous sentence's raw text (in case names do not survive the vector). Pull-based as the others;
# the log goes to the hub every 10 minutes (boxlog_DC.txt), the decoders to sentbart/dec/.
BOXDC_SERIAL=8
if [ -f /root/.boxdc_serial ] && [ "$(cat /root/.boxdc_serial)" -gt "$BOXDC_SERIAL" ] 2>/dev/null; then echo "BOXDC_STALE $BOXDC_SERIAL"; exit 0; fi
echo $BOXDC_SERIAL > /root/.boxdc_serial
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
mkdir -p /root/dec /root/hfdl
for f in sentbart/train.py sentbart/prep_dolphin.py sentbart/dec_pairs.py sentbart/dec_train.py sentbart/dec_e2e.py; do
  for try in 1 2 3; do curl -sS -o /root/dec/$(basename $f) "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/dec/$(basename $f) && break; sleep 5; done
done

if [ ! -f /root/.bootstrapped ]; then
  mkdir /root/.bootstrap_lock 2>/dev/null || { echo "bootstrap already running"; exit 0; }
  pip install -q "transformers>=4.51,<4.57" "huggingface_hub>=0.34,<1.0" safetensors blingfire zstandard 2>&1 | tail -1
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /root | tail -1; free -g | head -2; nproc
  touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"
fi

if [ ! -e /root/.dcmirror_r3 ]; then touch /root/.dcmirror_r3; pkill -f "dcmirro[r].sh"; sleep 1; fi
if [ ! -e /root/.dcmirror_r5 ]; then touch /root/.dcmirror_r5; pkill -f "dcmirro[r].sh"; sleep 1; fi
if [ ! -e /root/.dcmirror_r7 ]; then touch /root/.dcmirror_r7; pkill -f "dcmirro[r].sh"; sleep 1; fi
if [ ! -e /root/.dcmirror_r8 ]; then touch /root/.dcmirror_r8; pkill -f "dcmirro[r].sh"; sleep 1; fi
if ! pgrep -f "dcmirro[r].sh" >/dev/null; then
  cat > /root/dcmirror.sh <<'MR'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
while true; do
  { echo "=== boxDC $(date -u) ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; df -h /root | tail -1; free -g | sed -n 2p
    echo "--- ctl.log ---"; tail -n 12 /root/ctl.log 2>/dev/null | cut -c1-300
    echo "--- job.log ---"; grep -vE "Warning|warn|it/s\]" /root/dec/job.log 2>/dev/null | tail -n 25 | cut -c1-400
    echo "--- processes ---"; ps -eo pid,etimes,pcpu,rss,args | grep "[d]ec_train.py" | cut -c1-160; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
    for r in A B; do echo "--- run$r.out (tail) ---"; grep -vE "Warning|warn|it/s\]" /root/dec/run$r.out 2>/dev/null | tail -n 6 | cut -c1-300; done
    echo "--- e2e3/train.log ---"; grep -vE "^\[step" /root/dec/e2e3/train.log 2>/dev/null | tail -n 40 | cut -c1-500; grep -E "^\[step" /root/dec/e2e3/train.log 2>/dev/null | tail -n 2
    echo "--- e2e3.out (tail) ---"; grep -vE "Warning|warn|it/s\]" /root/dec/e2e3.out 2>/dev/null | tail -n 5 | cut -c1-300
    echo "--- e2e2/train.log ---"; grep -vE "^\[step" /root/dec/e2e2/train.log 2>/dev/null | tail -n 40 | cut -c1-500; grep -E "^\[step" /root/dec/e2e2/train.log 2>/dev/null | tail -n 2
    echo "--- e2e2.out (tail) ---"; grep -vE "Warning|warn|it/s\]" /root/dec/e2e2.out 2>/dev/null | tail -n 5 | cut -c1-300
    echo "--- e2e1/train.log ---"; grep -vE "^\[step" /root/dec/e2e1/train.log 2>/dev/null | tail -n 40 | cut -c1-500; grep -E "^\[step" /root/dec/e2e1/train.log 2>/dev/null | tail -n 2
    echo "--- e2e1.out (tail) ---"; grep -vE "Warning|warn|it/s\]" /root/dec/e2e1.out 2>/dev/null | tail -n 5 | cut -c1-300
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
# ---- serial 4: decoder B ran out of memory beside A (A alone holds ~10 GB and keeps the card at 100%, so two at once
# would not be faster anyway). One at a time: A stops 3.5 h after its start (its best weights are on the hub at every
# evaluation), then B (the previous sentence's raw text as well) for 3.5 h, so the two meet at the same number of steps. ----
if [ ! -e /root/.dcseq4 ]; then touch /root/.dcseq4
  cat > /root/dcseq4.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/dec
p=$(pgrep -f "dec_train.py.*--out runA" | head -1)
if [ -n "$p" ]; then
  while [ "$(ps -o etimes= -p $p 2>/dev/null | tr -d ' ')" -lt 12600 ] 2>/dev/null && kill -0 $p 2>/dev/null; do sleep 60; done
  kill $p 2>/dev/null; sleep 20; echo "[dcseq4] A stopped at 3.5 h: $(grep -E '^\[eval step' runA/train.log | tail -4 | cut -c1-120 | tr '\n' ' ') $(date -u)"
fi
mkdir -p failed_oom; mv runB failed_oom/runB_oom 2>/dev/null; mv runB.out failed_oom/runB_oom.out 2>/dev/null
python3 -u dec_train.py --pairs pairs --train train --eval am_held,dolphin_held --ckpt data/model_best.pt --out runB --vec mix --raw-prev 1 --epochs 2 --hours 3.5 --hub $R:sentbart/dec/runB > runB.out 2>&1
echo "DCSEQ4_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/dcseq4.sh >> /root/dec/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[dcseq4] A runs to 3.5 h, then B alone $(date -u)" | tee -a /root/dec/job.log
fi
# ---- serial 5, the user: with the BART frozen the decoder is capped by its vector (predicted-vector word F1 flat at
# 0.11-0.16 while the true vector's climbs to 0.33-0.49) - train them end to end. A stops now (its best decoder is in
# runA/dec_best), B is called off, and dec_e2e.py trains the BART, the MLP and GPT-2 together from those. ----
if [ ! -e /root/.dce2e5 ]; then touch /root/.dce2e5
  pkill -f "dcseq[4].sh"; pkill -f "dec_train.py.*--out run"; sleep 20
  echo "[dce2e5] A stopped, B called off; A's last evaluations: $(grep -E '^\[eval step' /root/dec/runA/train.log | tail -4 | cut -c1-110 | tr '\n' ' ')" | tee -a /root/dec/job.log
  cat > /root/dce2e5.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/dec
python3 -u dec_e2e.py --ckpt data/model_best.pt --dec runA/dec_best --docs data/docs_007.jsonl --held am_held=data/am_held.jsonl,dolphin_held=data/dolphin_held.jsonl \
  --out e2e1 --hours 5 --hub $R:sentbart/dec/e2e1 > e2e1.out 2>&1
grep -h "Error\|Traceback" e2e1.out | tail -3
echo "DCE2E5_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/dce2e5.sh >> /root/dec/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[dce2e5] end-to-end launched $(date -u)" | tee -a /root/dec/job.log
fi
# ---- serial 6: the first end-to-end update ran out of memory (BART + GPT-2 + MLP weights, grads and AdamW ~4 GB, then
# GPT-2's activations for 64 sentences). GPT-2 with gradient checkpointing, 12 sentences per conversation (48 per update),
# expandable allocator segments. ----
if [ ! -e /root/.dce2e6 ]; then touch /root/.dce2e6
  pkill -f "dce2e[5].sh"; pkill -f "dec_e2e.py"; sleep 10; cd /root/dec; mkdir -p failed_oom; mv e2e1 failed_oom/e2e1_oom 2>/dev/null; mv e2e1.out failed_oom/e2e1_oom.out 2>/dev/null
  cat > /root/dce2e6.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/dec
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -u dec_e2e.py --ckpt data/model_best.pt --dec runA/dec_best --docs data/docs_007.jsonl --held am_held=data/am_held.jsonl,dolphin_held=data/dolphin_held.jsonl \
  --out e2e1 --hours 5 --per-doc 12 --hub $R:sentbart/dec/e2e1 > e2e1.out 2>&1
grep -h "Error\|Traceback" e2e1.out | tail -3
echo "DCE2E6_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/dce2e6.sh >> /root/dec/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[dce2e6] end-to-end relaunched with GPT-2 checkpointing, 48 sentences per update $(date -u)" | tee -a /root/dec/job.log
fi
# ---- serial 7, the user: grow the BART inside the end-to-end run (1.2e8 parameters against the 1.5e9 of the model it is
# meant to serve). 60k more AM conversations are prepared while e2e1 goes on (CPU), then e2e1 stops and e2e2 resumes from
# its best state with the BART grown 16+16 -> 32+32 (identity-initialised: the function unchanged at the growth), on all
# 80k conversations, for 5 h. ----
if [ ! -e /root/.dce2e7 ]; then touch /root/.dce2e7
  cat > /root/dce2e7.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/dec
dl() { for try in 1 2 3 4; do [ -s $1 ] && [ "$(stat -c %s $1)" -ge $3 ] && return 0; curl -sSfL -C - -o $1 $2 || sleep 60; done; [ "$(stat -c %s $1 2>/dev/null || echo 0)" -ge $3 ]; }
if [ ! -s data/docs_008.jsonl ]; then
  Z=data/am_0.9M.jsonl.zst
  dl $Z https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M/resolve/main/am_0.9M.jsonl.zst 3200000000 || { echo "DCE2E7 am download failed"; exit 1; }
  python3 -u prep_dolphin.py --out data --schema am --all 1 --shard 008 --id-prefix am1 --max-sents 768 --max-q 24 --skip-rows 20172 --max-docs 60000 \
    --keep-eval data/dolphin_docs_001.jsonl --exclude-questions data/dolphin_docs_001.jsonl data/am_held.jsonl --jsonl $Z 2>&1 | grep -E "PREP_DOLPHIN_DONE|Error|Traceback" | cut -c1-300
  rm -f $Z
fi
echo "[dce2e7] 60k more conversations: $(wc -l < data/docs_008.jsonl) $(date -u)"
pkill -f "dce2e[6].sh"; pkill -f "dec_e2e.py.*--out e2e1"; sleep 20
RES=e2e1/e2e_best; [ -s $RES/bart.pt ] || RES=""
echo "[dce2e7] e2e1 stopped: $(grep -E '^\[eval step' e2e1/train.log | tail -4 | cut -c1-120 | tr '\n' ' '); resuming from ${RES:-the BART checkpoint and runA/dec_best} $(date -u)"
if [ -n "$RES" ]; then SRC="--resume $RES"; else SRC="--dec runA/dec_best"; fi
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -u dec_e2e.py --ckpt data/model_best.pt $SRC --grow-to 32 --docs data/docs_007.jsonl,data/docs_008.jsonl \
  --held am_held=data/am_held.jsonl,dolphin_held=data/dolphin_held.jsonl --out e2e2 --hours 5 --per-doc 12 --hub $R:sentbart/dec/e2e2 > e2e2.out 2>&1
grep -h "Error\|Traceback" e2e2.out | tail -3
echo "DCE2E7_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/dce2e7.sh >> /root/dec/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[dce2e7] launched: prep, then e2e2 with the BART at 32+32 $(date -u)" | tee -a /root/dec/job.log
fi
# ---- serial 8, the user: not straight to 32+32 - the usual way: the 16+16 BART goes on end to end and doubles its depth
# when two evaluations in a row gain under 0.01 (mean predicted-vector nll); the corpus may go round again, the held-out
# evaluations watching. Serial 7's job stops (its prep finishes on its own); e2e3 resumes from e2e1's best state at 16+16
# on the 80k conversations, growth on plateau up to 32+32, 6 h. ----
if [ ! -e /root/.dce2e8 ]; then touch /root/.dce2e8
  pkill -f "dce2e[7].sh"; pkill -f "dec_e2e.py.*--out e2e2"; sleep 5
  cat > /root/dce2e8.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/dec
while pgrep -f "prep_dolphin.py.*--shard 008" >/dev/null; do sleep 30; done
if ! grep -q "PREP_DOLPHIN_DONE" <(tail -c 2000 /root/dec/job.log) && [ "$(wc -l < data/docs_008.jsonl 2>/dev/null || echo 0)" -lt 59000 ]; then
  Z=data/am_0.9M.jsonl.zst
  for try in 1 2 3; do [ "$(stat -c %s $Z 2>/dev/null || echo 0)" -ge 3200000000 ] && break; curl -sSfL -C - -o $Z https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M/resolve/main/am_0.9M.jsonl.zst || sleep 60; done
  python3 -u prep_dolphin.py --out data --schema am --all 1 --shard 008 --id-prefix am1 --max-sents 768 --max-q 24 --skip-rows 20172 --max-docs 60000 \
    --keep-eval data/dolphin_docs_001.jsonl --exclude-questions data/dolphin_docs_001.jsonl data/am_held.jsonl --jsonl $Z 2>&1 | grep -E "PREP_DOLPHIN_DONE|Error|Traceback" | cut -c1-300
fi
rm -f data/am_0.9M.jsonl.zst
echo "[dce2e8] conversations: $(wc -l < data/docs_007.jsonl) + $(wc -l < data/docs_008.jsonl) $(date -u)"
pkill -f "dec_e2e.py"; sleep 20
RES=e2e1/e2e_best; [ -s $RES/bart.pt ] && SRC="--resume $RES" || SRC="--dec runA/dec_best"
echo "[dce2e8] e2e1's last: $(grep -E '^\[eval step' e2e1/train.log | tail -4 | cut -c1-110 | tr '\n' ' '); e2e3 from ${SRC} $(date -u)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -u dec_e2e.py --ckpt data/model_best.pt $SRC --docs data/docs_007.jsonl,data/docs_008.jsonl \
  --held am_held=data/am_held.jsonl,dolphin_held=data/dolphin_held.jsonl --out e2e3 --hours 6 --per-doc 12 --grow-plateau 0.01 --grow-max 32 --hub $R:sentbart/dec/e2e3 > e2e3.out 2>&1
grep -h "Error\|Traceback" e2e3.out | tail -3
echo "DCE2E8_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/dce2e8.sh >> /root/dec/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[dce2e8] launched: 16+16 on 80k, growth on plateau $(date -u)" | tee -a /root/dec/job.log
fi
echo "BOXDC_OK serial $BOXDC_SERIAL $(date -u)"
# CTL-END
