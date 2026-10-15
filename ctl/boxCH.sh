#!/bin/bash
# box CH (2026-10-16, a 3060 - the user: the cheap one): the character-level LM made in earnest - 4096-character context, WikiText-103 + FineWeb-Edu,
# grown 4 -> 8 -> 12 -> 16 layers at d 512 (charlm/charlm2.py), then tested against GPT-2 small on the same documents and
# quantized to 4 bits by GPTQ. Pull-based as the others; the log goes to the hub every 10 minutes (boxlog_CH.txt), the
# checkpoints to charlm/run2/ as they are made.
BOXCH_SERIAL=3
if [ -f /root/.boxch_serial ] && [ "$(cat /root/.boxch_serial)" -gt "$BOXCH_SERIAL" ] 2>/dev/null; then echo "BOXCH_STALE $BOXCH_SERIAL"; exit 0; fi
echo $BOXCH_SERIAL > /root/.boxch_serial
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
mkdir -p /root/charlm/data /root/charlm/run2
for f in charlm/charlm2.py charlm/charlm_prep.py charlm/charlm_fold.py charlm/charlm_sft.py q4.py; do
  for try in 1 2 3; do curl -sS -o /root/charlm/$(basename $f) "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/charlm/$(basename $f) && break; sleep 5; done
done

if [ ! -f /root/.bootstrapped ]; then
  mkdir /root/.bootstrap_lock 2>/dev/null || { echo "bootstrap already running"; exit 0; }
  pip install -q "transformers>=4.51,<4.57" "huggingface_hub>=0.34,<1.0" pyarrow safetensors 2>&1 | tail -1
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /root | tail -1; free -g | head -2; nproc
  touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"
fi

# ---- the mirror (serial 2: it reads run2/train.log, which the job's grep pipe only flushes in 4 kB blocks) ----
if [ ! -e /root/.chmirror_r2 ]; then touch /root/.chmirror_r2; pkill -f "chmirro[r].sh"; sleep 1; fi
if [ ! -e /root/.chmirror_r3 ]; then touch /root/.chmirror_r3; pkill -f "chmirro[r].sh"; sleep 1; fi
if ! pgrep -f "chmirro[r].sh" >/dev/null; then
  cat > /root/chmirror.sh <<'MR'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
while true; do
  { echo "=== boxCH $(date -u) ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; df -h /root | tail -1
    echo "--- ctl.log ---"; tail -n 15 /root/ctl.log 2>/dev/null | cut -c1-300
    echo "--- prep.log ---"; tail -n 20 /root/charlm/prep.log 2>/dev/null | cut -c1-300
    echo "--- job.log ---"; grep -vE "Warning|warn" /root/charlm/job.log 2>/dev/null | tail -n 20 | cut -c1-600
    echo "--- run2/train.log ---"; grep -E "^\[val\]|^\[grow\]|^\[oom\]|^\[charlm2\]|^\[test\]|^\[gptq\]|^\[sample\]|^\[hub\]|CHARLM2_DONE" /root/charlm/run2/train.log 2>/dev/null | tail -n 40 | cut -c1-600
    grep -E "^\[step" /root/charlm/run2/train.log 2>/dev/null | tail -n 3 | cut -c1-300
    echo "--- sft1/train.log ---"; grep -vE "Warning|warn|^\[step" /root/charlm/sft1/train.log 2>/dev/null | tail -n 30 | cut -c1-900
    grep -E "^\[step" /root/charlm/sft1/train.log 2>/dev/null | tail -n 2 | cut -c1-300; } > /root/boxlog_CH.txt
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='/root/boxlog_CH.txt', path_in_repo='pooler_distill/chatsft/audit/boxlog_CH.txt', repo_id='baya1116/hypernet-sp-distill')" >/dev/null 2>&1
  sleep 600
done
MR
  setsid nohup bash /root/chmirror.sh > /dev/null 2>&1 < /dev/null &
fi

# ---- the job: prep (once), the speed of each depth, then training (resumes from its latest checkpoint after a restart) ----
if [ ! -e /root/.chjob ]; then touch /root/.chjob
  cat > /root/chjob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/charlm
[ -e data/PREP_DONE ] || python3 -u charlm_prep.py --out data --fineweb 2 > prep.log 2>&1
[ -e data/PREP_DONE ] || { echo "CHJOB prep failed $(date -u)"; exit 1; }
[ -e bench.done ] || { python3 -u charlm2.py --data data --out bench --bench 1 --d 512 --heads 8 --layers 4,8,12,16 --micro 16 2>&1 | grep -E "^\[bench\]|^\[charlm2\]|Error|Traceback"; touch bench.done; }
for try in 1 2 3 4 5; do
  RES=""; [ -s run2/model_latest.pt ] && RES="--resume run2/model_latest.pt"
  python3 -u charlm2.py --data data --out run2 --d 512 --heads 8 --layers 4,8,12,16 --seq 4096 --chars-per-step 131072 --micro 16 \
    --hours 30 --eval-every 30 --hub baya1116/hypernet-sp-distill:charlm/run2 $RES 2>&1 | grep -vE "Warning|warn"
  grep -q CHARLM2_DONE run2/train.log && break
  echo "CHJOB training stopped without finishing (try $try) - again in 2 min $(date -u)"; sleep 120
done
echo "CHJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/chjob.sh >> /root/charlm/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "CHJOB_LAUNCHED $(date -u)"
fi
# ---- 2026-10-16, the user: then fine-tune it on chat. Once the pre-training has written CHARLM2_DONE, SmolTalk's
# conversational parts (4 of its 9 train shards) for up to 3 h on its final weights (charlm/charlm_sft.py). ----
if [ ! -e /root/.chsft ]; then touch /root/.chsft
  cat > /root/chsft.sh <<'SFT'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/charlm
until grep -q CHARLM2_DONE run2/train.log 2>/dev/null; do sleep 300; done
mkdir -p sftdata
for try in 1 2 3; do
  python3 -u charlm_sft.py --base run2/model_latest.pt --data sftdata --out sft1 --shards 4 --hours 3 --hub baya1116/hypernet-sp-distill:charlm/sft1 2>&1 | grep -vE "Warning|warn"
  grep -q CHARLM_SFT_DONE sft1/train.log && break; echo "CHSFT stopped (try $try) $(date -u)"; sleep 120
done
echo "CHSFT_DONE $(date -u)"
SFT
  setsid nohup bash -c 'bash /root/chsft.sh >> /root/charlm/sft.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "CHSFT_LAUNCHED (waits for the pre-training) $(date -u)"
fi
echo "BOXCH_OK serial $BOXCH_SERIAL $(date -u)"
# CTL-END
