#!/bin/bash
# box DW (2026-10-17, a 3060): the context BART's decoder with a raw window. The BART stays frozen (end-to-end run e2e3's,
# 32+32); its predicted vector still goes in front of GPT-2 as 8 prefix embeddings, and now the last W raw tokens of the
# conversation so far sit between the prefix and the sentence, so GPT-2 continues the text. W starts at 0 and widens
# (16, 32, 64, 128, 256 tokens) each time the predicted-vector nll levels off: how wide a window the decoder needs.
# Each evaluation also scores another sentence's vector, so what the vector still adds next to the window shows.
# Pull-based as the others; the log goes to the hub every 10 minutes (boxlog_DW.txt), the decoders to sentbart/dec/win1.
BOXDW_SERIAL=2
if [ -f /root/.boxdw_serial ] && [ "$(cat /root/.boxdw_serial)" -gt "$BOXDW_SERIAL" ] 2>/dev/null; then echo "BOXDW_STALE $BOXDW_SERIAL"; exit 0; fi
echo $BOXDW_SERIAL > /root/.boxdw_serial
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
mkdir -p /root/dw /root/hfdl
for f in sentbart/train.py sentbart/prep_dolphin.py sentbart/dec_pairs.py sentbart/dec_train.py; do
  for try in 1 2 3; do curl -sS -o /root/dw/$(basename $f) "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/dw/$(basename $f) && break; sleep 5; done
done

if [ ! -f /root/.bootstrapped ]; then
  mkdir /root/.bootstrap_lock 2>/dev/null || { echo "bootstrap already running"; exit 0; }
  pip install -q "transformers>=4.51,<4.57" "huggingface_hub>=0.34,<1.0" safetensors blingfire zstandard 2>&1 | tail -1
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /root | tail -1; free -g | head -2; nproc
  touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"
fi

if ! pgrep -f "dwmirro[r].sh" >/dev/null; then
  cat > /root/dwmirror.sh <<'MR'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
while true; do
  { echo "=== boxDW $(date -u) ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; df -h /root | tail -1; free -g | sed -n 2p
    echo "--- ctl.log ---"; tail -n 12 /root/ctl.log 2>/dev/null | cut -c1-300
    echo "--- job.log ---"; grep -vE "Warning|warn|it/s\]" /root/dw/job.log 2>/dev/null | tail -n 25 | cut -c1-400
    echo "--- processes ---"; ps -eo pid,etimes,pcpu,rss,args | grep -E "[d]ec_train.py|[d]ec_pairs.py|[p]rep_dolphin.py" | cut -c1-160
    echo "--- win1/train.log ---"; grep -vE "^\[step" /root/dw/win1/train.log 2>/dev/null | tail -n 60 | cut -c1-500; grep -E "^\[step" /root/dw/win1/train.log 2>/dev/null | tail -n 2
    echo "--- win1.out (tail) ---"; grep -vE "Warning|warn|it/s\]" /root/dw/win1.out 2>/dev/null | tail -n 5 | cut -c1-300; } > /root/boxlog_DW.txt
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='/root/boxlog_DW.txt', path_in_repo='pooler_distill/chatsft/audit/boxlog_DW.txt', repo_id='baya1116/hypernet-sp-distill')" >/dev/null 2>&1
  sleep 600
done
MR
  setsid nohup bash /root/dwmirror.sh > /dev/null 2>&1 < /dev/null &
fi

if [ ! -e /root/.dwjob ]; then touch /root/.dwjob
  cat > /root/dwjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/dw; mkdir -p data pairs
dl() { for try in 1 2 3 4; do [ -s $1 ] && [ "$(stat -c %s $1)" -ge $3 ] && return 0; curl -sSfL -C - -o $1 $2 || sleep 60; done; [ "$(stat -c %s $1 2>/dev/null || echo 0)" -ge $3 ]; }
# the frozen BART and the decoder to start from: end-to-end run e2e3's best (BART 32+32, the MLP, GPT-2)
[ -s e2e_best/bart.pt ] || { hf download $R --include "sentbart/dec/e2e3/e2e_best/*" --local-dir /root/hfdl >/dev/null 2>&1; mv /root/hfdl/sentbart/dec/e2e3/e2e_best e2e_best; }
[ -s data/am_held.jsonl ] || { hf download $R sentbart/dolphin/heldout/am_held_docs_005.jsonl sentbart/dolphin/heldout/dolphin768full_docs_001.jsonl sentbart/dolphin/data/docs_001.jsonl --local-dir /root/hfdl >/dev/null 2>&1
  mv /root/hfdl/sentbart/dolphin/heldout/am_held_docs_005.jsonl data/am_held.jsonl; mv /root/hfdl/sentbart/dolphin/heldout/dolphin768full_docs_001.jsonl data/dolphin_held.jsonl; mv /root/hfdl/sentbart/dolphin/data/docs_001.jsonl data/dolphin_docs_001.jsonl; }
echo "[dwjob] BART $(ls -la e2e_best/bart.pt 2>/dev/null | awk '{print $5}') bytes; held-outs $(ls data | tr '\n' ' ') $(date -u)"
[ -s e2e_best/bart.pt ] || { echo "DWJOB no BART"; exit 1; }
# the same 20k AM conversations box DC trained decoder A on (shard 007)
if [ ! -s data/docs_007.jsonl ]; then
  Z=data/am_0.9M.jsonl.zst
  dl $Z https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M/resolve/main/am_0.9M.jsonl.zst 3200000000 || { echo "DWJOB am download failed"; exit 1; }
  python3 -u prep_dolphin.py --out data --schema am --all 1 --shard 007 --id-prefix am1 --max-sents 768 --max-q 24 --max-docs 20000 \
    --keep-eval data/dolphin_docs_001.jsonl --exclude-questions data/dolphin_docs_001.jsonl data/am_held.jsonl --jsonl $Z 2>&1 | grep -E "PREP_DOLPHIN|Error|Traceback" | cut -c1-300
  rm -f $Z
fi
echo "[dwjob] training conversations: $(wc -l < data/docs_007.jsonl) $(date -u)"
# pairs from the frozen BART, each with the raw text before its sentence (the last 1600 characters)
[ -s pairs/pairs_train.npz ] || python3 -u dec_pairs.py --ckpt e2e_best/bart.pt --docs data/docs_007.jsonl --name train --out pairs --max-t 96 --batch 8 --ctx-chars 1600 2>&1 | grep -E "^\[pairs\]|DONE|Error|Traceback" | grep -vE "/[0-9]+ conversations" | cut -c1-300
[ -s pairs/pairs_am_held.npz ] || python3 -u dec_pairs.py --ckpt e2e_best/bart.pt --docs data/am_held.jsonl --name am_held --out pairs --max-docs 600 --max-t 96 --ctx-chars 1600 2>&1 | grep -E "^\[pairs\] am_held: [0-9]+ pairs|Error|Traceback"
[ -s pairs/pairs_dolphin_held.npz ] || python3 -u dec_pairs.py --ckpt e2e_best/bart.pt --docs data/dolphin_held.jsonl --name dolphin_held --out pairs --max-docs 600 --max-t 96 --ctx-chars 1600 2>&1 | grep -E "^\[pairs\] dolphin_held: [0-9]+ pairs|Error|Traceback"
echo "[dwjob] pairs ready $(date -u)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -u dec_train.py --pairs pairs --train train --eval am_held,dolphin_held --ckpt e2e_best/bart.pt --init e2e_best \
  --out win1 --vec pred --win 0,16,32,64,128,256 --win-plateau 0.01 --win-min-evals 2 --win-max-min 90 --eval-every 20 --hours 10 --epochs 3 \
  --lr 5e-5 --lr-mlp 5e-4 --warmup 100 --ckpt-lm 1 --hub $R:sentbart/dec/win1 > win1.out 2>&1
grep -h "Error\|Traceback" win1.out | tail -3
echo "DWJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/dwjob.sh >> /root/dw/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "DWJOB_LAUNCHED $(date -u)"
fi
# ---- serial 2: the 20k conversations of shard 007 are ones the decoder had already trained on (box DC's e2e runs saw
# 007 and 008, ~80k rows of the AM file): at window 0 the training nll sat at 2.24 while the held-out one rose (AM 3.257 ->
# 3.332). Fresh conversations instead - 30k from row 90000 on, past everything the e2e runs read - and the run again. ----
if [ ! -e /root/.dw_fresh2 ]; then touch /root/.dw_fresh2
  pkill -f "dwjo[b].sh"; pkill -f "dec_train.py --pairs pairs"; sleep 15
  cd /root/dw && mkdir -p seen_data && mv win1 seen_data/win1_on_007 2>/dev/null; mv win1.out seen_data/ 2>/dev/null
  cat > /root/dwjob2.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/dw
dl() { for try in 1 2 3 4; do [ -s $1 ] && [ "$(stat -c %s $1)" -ge $3 ] && return 0; curl -sSfL -C - -o $1 $2 || sleep 60; done; [ "$(stat -c %s $1 2>/dev/null || echo 0)" -ge $3 ]; }
if [ ! -s data/docs_009.jsonl ]; then
  Z=data/am_0.9M.jsonl.zst
  dl $Z https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M/resolve/main/am_0.9M.jsonl.zst 3200000000 || { echo "DWJOB2 am download failed"; exit 1; }
  python3 -u prep_dolphin.py --out data --schema am --all 1 --shard 009 --id-prefix am1 --max-sents 768 --max-q 24 --skip-rows 90000 --max-docs 30000 \
    --keep-eval data/dolphin_docs_001.jsonl --exclude-questions data/dolphin_docs_001.jsonl data/am_held.jsonl --jsonl $Z 2>&1 | grep -E "PREP_DOLPHIN_DONE|Error|Traceback" | cut -c1-300
  rm -f $Z
fi
echo "[dwjob2] fresh conversations: $(wc -l < data/docs_009.jsonl) $(date -u)"
[ -s pairs/pairs_fresh.npz ] || python3 -u dec_pairs.py --ckpt e2e_best/bart.pt --docs data/docs_009.jsonl --name fresh --out pairs --max-t 96 --batch 8 --ctx-chars 1600 2>&1 | grep -E "^\[pairs\]|DONE|Error|Traceback" | grep -vE "/[0-9]+ conversations" | cut -c1-300
[ -s pairs/pairs_fresh.npz ] && rm -f pairs/pairs_train.npz pairs/pairs_train.jsonl
echo "[dwjob2] pairs ready $(date -u)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -u dec_train.py --pairs pairs --train fresh --eval am_held,dolphin_held --ckpt e2e_best/bart.pt --init e2e_best \
  --out win1 --vec pred --win 0,16,32,64,128,256 --win-plateau 0.01 --win-min-evals 2 --win-max-min 90 --eval-every 20 --hours 10 --epochs 3 \
  --lr 5e-5 --lr-mlp 5e-4 --warmup 100 --ckpt-lm 1 --hub $R:sentbart/dec/win1 > win1.out 2>&1
grep -h "Error\|Traceback" win1.out | tail -3
echo "DWJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/dwjob2.sh >> /root/dw/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[dwjob2] launched: fresh conversations from row 90000 $(date -u)" | tee -a /root/dw/job.log
fi
echo "BOXDW_OK serial $BOXDW_SERIAL $(date -u)"
# CTL-END
