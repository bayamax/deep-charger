#!/bin/bash
# box CH (2026-10-16, a 3060 - the user: the cheap one): the character-level LM made in earnest - 4096-character context, WikiText-103 + FineWeb-Edu,
# grown 4 -> 8 -> 12 -> 16 layers at d 512 (charlm/charlm2.py), then tested against GPT-2 small on the same documents and
# quantized to 4 bits by GPTQ. Pull-based as the others; the log goes to the hub every 10 minutes (boxlog_CH.txt), the
# checkpoints to charlm/run2/ as they are made.
BOXCH_SERIAL=14
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
if [ ! -e /root/.chmirror_r5 ]; then touch /root/.chmirror_r5; pkill -f "chmirro[r].sh"; sleep 1; fi
if [ ! -e /root/.chmirror_r10 ]; then touch /root/.chmirror_r10; pkill -f "chmirro[r].sh"; sleep 1; fi   # serial 10: the chat lines too
if [ ! -e /root/.chmirror_r12 ]; then touch /root/.chmirror_r12; pkill -f "chmirro[r].sh"; sleep 1; fi   # serial 12: 4 h of training-loss lines
if [ ! -e /root/.chmirror_r14 ]; then touch /root/.chmirror_r14; pkill -f "chmirro[r].sh"; sleep 1; fi   # serial 14: the mix14 lines
if ! pgrep -f "chmirro[r].sh" >/dev/null; then
  cat > /root/chmirror.sh <<'MR'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
while true; do
  { echo "=== boxCH $(date -u) ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; df -h /root | tail -1
    echo "--- ctl.log ---"; tail -n 15 /root/ctl.log 2>/dev/null | cut -c1-300
    echo "--- prep.log ---"; tail -n 20 /root/charlm/prep.log 2>/dev/null | cut -c1-300
    echo "--- job.log ---"; grep -vE "Warning|warn" /root/charlm/job.log 2>/dev/null | tail -n 20 | cut -c1-600
    echo "--- run2/train.log ---"; grep -E "^\[val\]|^\[chat|^\[cool\]|^\[ctl\]|^\[plateau\]|^\[more7\]|^\[target9\]|^\[mix14\]|^\[ctl11\]|^\[grow\]|^\[oom\]|^\[diverge\]|^\[restart|^\[charlm2\]|^\[test\]|^\[gptq\]|^\[sample\]|^\[hub\]|CHARLM2_DONE" /root/charlm/run2/train.log 2>/dev/null | tail -n 50 | cut -c1-600
    grep -E "^\[step" /root/charlm/run2/train.log 2>/dev/null | tail -n 48 | cut -c1-300
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
    --hours 26 --eval-every 30 --lr 8e-4 --qknorm 1 --hub baya1116/hypernet-sp-distill:charlm/run2 $RES 2>&1 | grep -vE "Warning|warn"
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
# ---- 2026-10-16 (serial 4): at lr 1.5e-3 the run diverged after its 30-minute validation (1.716 -> 3.127 -> 3.206 bpc).
# The job stops; the diverged weights are set aside; it resumes from the best ones (step 1845) at lr 6e-4, and
# charlm2.py now goes back to its best weights at half the learning rate by itself if a validation is 0.1 worse. ----
if [ ! -e /root/.ch_restart4 ]; then touch /root/.ch_restart4
  pkill -f "chjo[b].sh"; pkill -f "charlm2.py --data data --out run2"; sleep 15
  cd /root/charlm && mv run2/model_latest.pt run2/model_diverged.pt && cp run2/model_best.pt run2/model_latest.pt
  sed -i 's/--hours 30 --eval-every 30 --hub/--hours 26 --eval-every 30 --lr 8e-4 --qknorm 1 --hub/' /root/chjob.sh
  grep -c "lr 6e-4" /root/chjob.sh | sed 's/^/[restart4] lr lines: /'
  setsid nohup bash -c 'bash /root/chjob.sh >> /root/charlm/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[restart4] resumed from the best weights (step 1845, 1.716 bpc) at lr 6e-4 $(date -u)" | tee -a /root/charlm/run2/train.log
fi
# ---- 2026-10-16 (serial 5): lr 6e-4 diverged too (1.716 -> 3.416 by the next validation): the attention logits over
# 4096 characters, not the step size alone. A fresh run with LayerNorm on each head's queries and keys (--qknorm 1) at
# lr 8e-4; the failed runs kept in run2_failed/. 26 h, so pre-training, chat tuning and the 4-bit steps fit the credit. ----
if [ ! -e /root/.ch_restart5 ]; then touch /root/.ch_restart5
  pkill -f "chjo[b].sh"; pkill -f "charlm2.py --data data --out run2"; sleep 15
  cd /root/charlm && mkdir -p run2_failed && mv run2/model_diverged.pt run2_failed/model_lr15e-4_diverged.pt 2>/dev/null; mv run2/train.log run2_failed/train_lr15e-4_and_6e-4.log; mv run2/model_best.pt run2_failed/model_best_step1845.pt; rm -f run2/model_latest.pt
  sed -i 's/--hours 26 --eval-every 30 --lr 8e-4 --qknorm 1 --hub/--hours 26 --eval-every 30 --lr 8e-4 --qknorm 1 --hub/' /root/chjob.sh
  grep -c "qknorm 1" /root/chjob.sh | sed 's/^/[restart5] qknorm lines: /'
  setsid nohup bash -c 'bash /root/chjob.sh >> /root/charlm/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[restart5] a fresh run with query/key LayerNorm at lr 8e-4, 26 h $(date -u)" | tee -a /root/charlm/run2/train.log
fi
# ---- serial 6: the growth 8 -> 12 dropped the last two blocks (an identity after each block until the count was reached,
# the rest cut off; 4 -> 8 was exact, so it had not shown): validation 1.208 -> 2.242 at growth. charlm2.py now spreads
# the identity blocks evenly and keeps every block (8 -> 12 -> 16 checked equal to the output). The job stops; the run
# resumes from the best weights (8 layers, 1.208, saved just before the growth) and grows to 12 at once (--grow-now). ----
if [ ! -e /root/.ch_restart6 ]; then touch /root/.ch_restart6
  pkill -f "chjo[b].sh"; pkill -f "charlm2.py --data data --out run2"; sleep 15
  cd /root/charlm && mv run2/model_latest.pt run2_failed/model_grow12_broken.pt 2>/dev/null; cp run2/model_best.pt run2/model_latest.pt
  python3 -c "import torch; c=torch.load('run2/model_latest.pt', map_location='cpu', weights_only=False); print('[restart6] resuming from', c['layers'], 'layers, step', c['step'], 'best', round(c['best']['bpc'], 3))" | tee -a run2/train.log
  sed -i 's/^for try in 1 2 3 4 5; do$/GN="--grow-now 1"\nfor try in 1 2 3 4 5; do/; s/ \$RES 2>&1 | grep -vE "Warning|warn"$/ $RES $GN 2>\&1 | grep -vE "Warning|warn"; GN=""/' /root/chjob.sh
  grep -c 'GN=' /root/chjob.sh | sed 's/^/[restart6] grow-now lines: /'
  setsid nohup bash -c 'bash /root/chjob.sh >> /root/charlm/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[restart6] relaunched: 8 layers from the best weights, grown to 12 at once with every block kept $(date -u)" | tee -a /root/charlm/run2/train.log
fi
# ---- serial 7, the user: more data rather than more passes - Wikipedia had been read ~2.5 times at p_wiki 0.3. Two more
# FineWeb-Edu sample-10BT shards (002, 003; ~6.8B characters, none of the held-out documents, which stay shard 001's) are
# prepared beside the run; at the next validation after that the run stops, the new text is appended to train_fw.bin, and
# it resumes with 53 h in all (23 more) at p_wiki 0.05 - Wikipedia ends near 3 passes, FineWeb near 0.7. ----
if [ ! -e /root/.ch_more7 ]; then touch /root/.ch_more7
  cat > /root/chmore7.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/charlm
L=run2/train.log
echo "[more7] preparing FineWeb shards 002, 003 beside the run $(date -u)" | tee -a $L
python3 -u charlm_prep.py --out data --more 2,3 > more7.log 2>&1
grep -E "^\[prep\]|Error|Traceback|CHARLM_MORE_DONE" more7.log | tail -3 | sed 's/^/[more7] /' | tee -a $L
grep -q CHARLM_MORE_DONE more7.log && [ -s data/train_fw_more.bin ] || { echo "[more7] prep failed - the run goes on unchanged $(date -u)" | tee -a $L; exit 1; }
n0=$(grep -c "^\[val\]" $L); while [ "$(grep -c '^\[val\]' $L)" -le "$n0" ]; do grep -q CHARLM2_DONE $L && break; sleep 30; done; sleep 60   # just after a validation (model_latest saved)
pkill -f "chjo[b].sh"; pkill -f "charlm2.py --data data --out run2"; sleep 20
b0=$(stat -c %s data/train_fw.bin); cat data/train_fw_more.bin >> data/train_fw.bin && rm -f data/train_fw_more.bin
echo "[more7] train_fw.bin $((b0 / 1000000))M -> $(( $(stat -c %s data/train_fw.bin) / 1000000 ))M chars $(date -u)" | tee -a $L
sed -i 's/^GN="--grow-now 1"$/GN=""/; s/--hours [0-9][0-9]* /--hours 53 --p-wiki 0.05 /' /root/chjob.sh
grep -o -- "--hours [0-9]* --p-wiki [0-9.]*" /root/chjob.sh | sed 's/^/[more7] job now: /' | tee -a $L; grep -c '^GN="--grow' /root/chjob.sh | sed 's/^/[more7] grow-now lines left: /' | tee -a $L
setsid nohup bash -c 'bash /root/chjob.sh >> /root/charlm/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
echo "[more7] resumed from model_latest with the larger FineWeb, 53 h in all, p_wiki 0.05 $(date -u)" | tee -a $L
JOB
  setsid nohup bash -c 'bash /root/chmore7.sh >> /root/charlm/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[more7] launched $(date -u)"
fi
# ---- serial 8, the user: the goal is a model that chats well enough, not the planned hours. A chat yardstick no training
# uses (the first 600 conversations of SmolTalk's test split, as 'User:'/'Assistant:' text) is scored at every validation
# with a reply to a chat prompt; once that score gains < 0.004 bpc over 4 validations (2 h) the learning rate falls to a
# tenth over 3 h and pre-training ends (53 h stays only the cap). The chat fine-tuning then runs until its own held-out
# score (the same test conversations) stops improving (< 0.005 over 3 evaluations, 1 h), with replies at every evaluation,
# 10 h the cap. The flags reach the run at its restart with the larger FineWeb (serial 7). ----
if [ ! -e /root/.ch_chat8 ]; then touch /root/.ch_chat8
  cd /root/charlm
  [ -s data/val_chat.bin ] || python3 -u charlm_prep.py --out data --chat-val 600 2>&1 | grep -E "^\[prep\]|DONE|Error|Traceback" | sed 's/^/[chat8] /' | tee -a run2/train.log
  if [ -s data/val_chat.bin ]; then
    grep -q -- "--chat-val" /root/chjob.sh || sed -i 's/ --eval-every 30 / --chat-val data\/val_chat.bin --cool-plateau 0.004 --cool-evals 4 --cool-hours 3 --eval-every 30 /' /root/chjob.sh
    grep -o -- "--chat-val [^ ]* --cool-plateau [0-9.]* --cool-evals [0-9]* --cool-hours [0-9.]*" /root/chjob.sh | sed 's/^/[chat8] job now: /' | tee -a run2/train.log
  fi
  pkill -f "chsf[t].sh"; sleep 2
  sed -i 's/ --hours 3 --hub / --hours 10 --stop-plateau 0.005 --stop-evals 3 --hub /' /root/chsft.sh
  grep -o -- "--hours [0-9]* --stop-plateau [0-9.]* --stop-evals [0-9]*" /root/chsft.sh | sed 's/^/[chat8] fine-tuning now: /' | tee -a run2/train.log
  setsid nohup bash -c 'bash /root/chsft.sh >> /root/charlm/sft.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[chat8] chat yardstick ready; the fine-tuning waits for pre-training again $(date -u)" | tee -a run2/train.log
fi
# ---- serial 9, the user: on to the chat fine-tuning once the chat perplexity is on a par with GPT-2's. GPT-2 small on the
# same 600 SmolTalk test conversations: token perplexity 11.38 = 0.845 bits per character. The target is within 5% of
# that (0.887 bpc), read from data/ctl.json at every validation (changeable without a restart); a plateau before it is
# only reported. The whole 2.3M-character yardstick is scored (as GPT-2's was). The run restarts once, just after a
# validation, for the new code. ----
if [ ! -e /root/.ch_target9 ]; then touch /root/.ch_target9
  echo '{"cool_target": 0.887, "plateau_cools": 0}' > /root/charlm/data/ctl.json
  cat > /root/chtarget9.sh <<'JOB'
cd /root/charlm; L=run2/train.log
grep -q -- "--ctl " /root/chjob.sh || sed -i 's/ --chat-val data\/val_chat.bin / --chat-val data\/val_chat.bin --chat-val-chars 2400000 --ctl data\/ctl.json /' /root/chjob.sh
grep -o -- "--chat-val [^ ]* --chat-val-chars [0-9]* --ctl [^ ]*" /root/chjob.sh | sed 's/^/[target9] job now: /' | tee -a $L
n0=$(grep -c "^\[val\]" $L); while [ "$(grep -c '^\[val\]' $L)" -le "$n0" ]; do grep -q CHARLM2_DONE $L && exit 0; sleep 30; done; sleep 150   # after the validation and its chat score
pkill -f "chjo[b].sh"; pkill -f "charlm2.py --data data --out run2"; sleep 20
setsid nohup bash -c 'bash /root/chjob.sh >> /root/charlm/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
echo "[target9] resumed: the chat target 0.887 bpc (GPT-2 small 0.845 + 5%) decides the cool-down $(date -u)" | tee -a $L
JOB
  setsid nohup bash -c 'bash /root/chtarget9.sh >> /root/charlm/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[target9] launched $(date -u)"
fi
# ---- serial 11, the user: one pass over more data, not passes over the same. Wikipedia has been read ~2.5 times: no more
# of it (p_wiki 0, read live from data/ctl.json at the next validation); FineWeb-Edu alone from here (13.8B characters,
# ~0.7 of a pass by the 53 h cap; more shards before it would come round again). ----
if [ ! -e /root/.ch_nowiki11 ]; then touch /root/.ch_nowiki11
  echo '{"cool_target": 0.887, "plateau_cools": 0, "p_wiki": 0.0}' > /root/charlm/data/ctl.json
  echo "[ctl11] data/ctl.json: $(cat /root/charlm/data/ctl.json) $(date -u)" | tee -a /root/charlm/run2/train.log
fi
# ---- serial 13: with Wikipedia no longer read, WikiText's validation climbs (1.123 -> 1.197) and drags the wiki+FineWeb
# mean up; the divergence guard (mean 0.1 above its best -> back to old weights at half the rate) would fire on that alone
# within hours. Off (diverge 9) through the control file; FineWeb and the chat score are what is watched now. ----
if [ ! -e /root/.ch_nodiv13 ]; then touch /root/.ch_nodiv13
  echo '{"cool_target": 0.887, "plateau_cools": 0, "p_wiki": 0.0, "diverge": 9.0}' > /root/charlm/data/ctl.json
  echo "[ctl13] data/ctl.json: $(cat /root/charlm/data/ctl.json) $(date -u)" | tee -a /root/charlm/run2/train.log
fi
# ---- serial 14, the user: chat in the pre-training mix too - GPT-2 read conversation-like web text, and chat is where this
# model is headed. SmolTalk train shards 4-8 (of 9) as the same plain 'User:'/'Assistant:' text (0-3 stay for the chat
# fine-tuning; the test split stays the unseen yardstick), a quarter of the windows from it. One restart for the new code
# (the last without the optimizer state: the running code predates it). ----
if [ ! -e /root/.ch_chatmix14 ]; then touch /root/.ch_chatmix14
  cat > /root/chmix14.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/charlm; L=run2/train.log
echo "[mix14] preparing SmolTalk train shards 4-8 as plain chat text beside the run $(date -u)" | tee -a $L
python3 -u charlm_prep.py --out data --chat-train 4,5,6,7,8 > mix14.log 2>&1
grep -E "^\[prep\]|Error|Traceback|CHARLM_CHATTRAIN_DONE" mix14.log | tail -3 | sed 's/^/[mix14] /' | tee -a $L
grep -q CHARLM_CHATTRAIN_DONE mix14.log && [ -s data/train_chat.bin ] || { echo "[mix14] prep failed - the run goes on unchanged $(date -u)" | tee -a $L; exit 1; }
sed -i 's/ --p-wiki [0-9.]* / --p-wiki 0.0 --p-chat 0.25 /' /root/chjob.sh
grep -o -- "--p-wiki [0-9.]* --p-chat [0-9.]*" /root/chjob.sh | sed 's/^/[mix14] job now: /' | tee -a $L
echo '{"cool_target": 0.887, "plateau_cools": 0, "p_wiki": 0.0, "diverge": 9.0, "p_chat": 0.25}' > data/ctl.json
n0=$(grep -c "^\[val\]" $L); while [ "$(grep -c '^\[val\]' $L)" -le "$n0" ]; do grep -q CHARLM2_DONE $L && exit 0; sleep 30; done; sleep 150
pkill -f "chjo[b].sh"; pkill -f "charlm2.py --data data --out run2"; sleep 20
setsid nohup bash -c 'bash /root/chjob.sh >> /root/charlm/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
echo "[mix14] resumed with a quarter of the windows from chat text $(date -u)" | tee -a $L
JOB
  setsid nohup bash -c 'bash /root/chmix14.sh >> /root/charlm/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[mix14] launched $(date -u)"
fi
echo "BOXCH_OK serial $BOXCH_SERIAL $(date -u)"
# CTL-END
