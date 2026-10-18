#!/bin/bash
# box CH (2026-10-16, a 3060 - the user: the cheap one): the character-level LM made in earnest - 4096-character context, WikiText-103 + FineWeb-Edu,
# grown 4 -> 8 -> 12 -> 16 layers at d 512 (charlm/charlm2.py), then tested against GPT-2 small on the same documents and
# quantized to 4 bits by GPTQ. Pull-based as the others; the log goes to the hub every 10 minutes (boxlog_CH.txt), the
# checkpoints to charlm/run2/ as they are made.
BOXCH_SERIAL=23
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
if [ ! -e /root/.chmirror_r15 ]; then touch /root/.chmirror_r15; pkill -f "chmirro[r].sh"; sleep 1; fi   # serial 15: the push15 lines
if [ ! -e /root/.chmirror_r17 ]; then touch /root/.chmirror_r17; pkill -f "chmirro[r].sh"; sleep 1; fi   # serial 17: the sft17 lines
if [ ! -e /root/.chmirror_r19 ]; then touch /root/.chmirror_r19; pkill -f "chmirro[r].sh"; sleep 1; fi   # serial 19: the sft19 lines
if [ ! -e /root/.chmirror_r20 ]; then touch /root/.chmirror_r20; pkill -f "chmirro[r].sh"; sleep 1; fi   # serial 20: the npc20 lines and npc1
if [ ! -e /root/.chmirror_r21 ]; then touch /root/.chmirror_r21; pkill -f "chmirro[r].sh"; sleep 1; fi   # serial 21: the npc21 lines
if [ ! -e /root/.chmirror_r23 ]; then touch /root/.chmirror_r23; pkill -f "chmirro[r].sh"; sleep 1; fi   # serial 23: 4 h of fine-tuning loss lines, one section per run
if ! pgrep -f "chmirro[r].sh" >/dev/null; then
  cat > /root/chmirror.sh <<'MR'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
while true; do
  { echo "=== boxCH $(date -u) ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; df -h /root | tail -1
    echo "--- ctl.log ---"; tail -n 15 /root/ctl.log 2>/dev/null | cut -c1-300
    echo "--- prep.log ---"; tail -n 20 /root/charlm/prep.log 2>/dev/null | cut -c1-300
    echo "--- job.log ---"; grep -vE "Warning|warn" /root/charlm/job.log 2>/dev/null | tail -n 20 | cut -c1-600
    echo "--- run2/train.log ---"; grep -E "^\[val\]|^\[chat|^\[cool\]|^\[ctl\]|^\[plateau\]|^\[more7\]|^\[target9\]|^\[mix14\]|^\[push15\]|^\[sft17\]|^\[sft19\]|^\[npc20\]|^\[npc21\]|^\[ctl11\]|^\[grow\]|^\[oom\]|^\[diverge\]|^\[restart|^\[charlm2\]|^\[test\]|^\[gptq\]|^\[sample\]|^\[hub\]|CHARLM2_DONE" /root/charlm/run2/train.log 2>/dev/null | tail -n 50 | cut -c1-600
    grep -E "^\[step" /root/charlm/run2/train.log 2>/dev/null | tail -n 48 | cut -c1-300
    for r in sft1 npc1; do echo "--- $r/train.log ---"; grep -vE "Warning|warn|^\[step" /root/charlm/$r/train.log 2>/dev/null | tail -n 30 | cut -c1-900
      echo "--- $r steps ---"; grep -E "^\[step" /root/charlm/$r/train.log 2>/dev/null | tail -n 48 | cut -c1-300; done; } > /root/boxlog_CH.txt
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
# ---- serial 15, the user: push on while the chat score falls this fast (1.146 -> 0.901 -> 0.864 within an hour of the chat
# mix). The cool-down the 0.887 target started (minute 1861) is called off; no fixed target - the cool-down now starts when
# the chat score gains < 0.01 over 4 validations (2 h); 40% of the windows from chat text (2.2B characters: still under one
# pass by the cap). One restart for the new code (the optimizer state now comes back with it). ----
if [ ! -e /root/.ch_push15 ]; then touch /root/.ch_push15
  echo '{"cool_target": 0.0, "plateau_cools": 1, "cool_plateau": 0.01, "cool_evals": 4, "p_wiki": 0.0, "diverge": 9.0, "p_chat": 0.4, "cool_cancel_before_min": 1900}' > /root/charlm/data/ctl.json
  cat > /root/chpush15.sh <<'JOB'
cd /root/charlm; L=run2/train.log
sed -i 's/ --p-chat [0-9.]* / --p-chat 0.4 /' /root/chjob.sh
echo "[push15] control file: $(cat data/ctl.json); job: $(grep -o -- '--p-chat [0-9.]*' /root/chjob.sh) $(date -u)" | tee -a $L
n0=$(grep -c "^\[val\]" $L); while [ "$(grep -c '^\[val\]' $L)" -le "$n0" ]; do grep -q CHARLM2_DONE $L && exit 0; sleep 30; done; sleep 150
pkill -f "chjo[b].sh"; pkill -f "charlm2.py --data data --out run2"; sleep 20
setsid nohup bash -c 'bash /root/chjob.sh >> /root/charlm/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
echo "[push15] resumed: the cool-down called off, the chat plateau decides, 40% chat $(date -u)" | tee -a $L
JOB
  setsid nohup bash -c 'bash /root/chpush15.sh >> /root/charlm/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[push15] launched $(date -u)"
fi
# ---- serial 16, the user: the sample's question needed knowledge (photosynthesis), the thing a 50M-parameter model has least
# of; an everyday one shows whether it can hold a conversation. The prompt changes through the control file. ----
if [ ! -e /root/.ch_prompt16 ]; then touch /root/.ch_prompt16
  cat > /root/charlm/data/ctl.json <<'CTLJ'
{"cool_target": 0.0, "plateau_cools": 1, "cool_plateau": 0.01, "cool_evals": 4, "p_wiki": 0.0, "diverge": 9.0, "p_chat": 0.4, "cool_cancel_before_min": 1900, "chat_prompt": "User: Hi! I'm going to the beach this weekend. Any tips?\nAssistant:"}
CTLJ
  echo "[ctl16] data/ctl.json: $(cat /root/charlm/data/ctl.json) $(date -u)" | tee -a /root/charlm/run2/train.log
fi
# ---- serial 17: the chat fine-tuning restarts, minutes in: one pass over its 315k conversations (it was set to two - one
# pass over more data, the user's rule), and GPTQ that no longer fails (pre-training's 4-bit pass stopped at a Cholesky of a
# not-positive-definite Hessian: now float64, the damping raised until it factorises). ----
if [ ! -e /root/.ch_sft17 ]; then touch /root/.ch_sft17
  pkill -f "chsf[t].sh"; pkill -f "charlm_sft.py"; sleep 10
  cd /root/charlm && mkdir -p sft_old && mv sft1 sft_old/sft1_2epochs 2>/dev/null
  sed -i 's/ --hours 10 --stop-plateau / --hours 10 --epochs 1 --stop-plateau /' /root/chsft.sh
  grep -o -- "--hours [0-9]* --epochs [0-9]* --stop-plateau [0-9.]* --stop-evals [0-9]*" /root/chsft.sh | sed 's/^/[sft17] fine-tuning now: /' | tee -a run2/train.log
  setsid nohup bash -c 'bash /root/chsft.sh >> /root/charlm/sft.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[sft17] the chat fine-tuning relaunched (one pass, robust GPTQ) $(date -u)" | tee -a run2/train.log
fi
# ---- serial 18: the chat fine-tuning stops right after its replies before training, three tries running, the GPU idle; its
# error goes to sft.log, which the mirror does not carry. Into sft1/train.log (which it does): sft.log's tail, memory, the
# kernel's OOM lines. ----
if [ ! -e /root/.ch_diag18 ]; then touch /root/.ch_diag18
  cd /root/charlm; mkdir -p sft1
  { echo "[diag18] $(date -u) - sft.log tail:"; grep -vE "Warning|warn" sft.log | tail -n 30 | cut -c1-300 | sed 's/^/[diag18] /'
    echo "[diag18] memory: $(free -g | sed -n 2p)"; dmesg 2>/dev/null | grep -iE "killed process|out of memory" | tail -n 3 | sed 's/^/[diag18] /'
    ps -eo pid,etimes,args | grep -E "[c]harlm_sft|[c]hsft" | cut -c1-150 | sed 's/^/[diag18] /'; } >> sft1/train.log
fi
# ---- serial 19: the fine-tuning held its 315k conversations as Python int lists (and packed them into more) - tens of GB,
# killed without a traceback right after its first replies. Now one byte a symbol (uint8 arrays). Relaunched. ----
if [ ! -e /root/.ch_sft19 ]; then touch /root/.ch_sft19
  pkill -f "chsf[t].sh"; pkill -f "charlm_sft.py"; sleep 5
  cd /root/charlm && mkdir -p sft_old && mv sft1 sft_old/sft1_oom 2>/dev/null
  setsid nohup bash -c 'bash /root/chsft.sh >> /root/charlm/sft.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[sft19] the chat fine-tuning relaunched with uint8 conversations $(date -u)" | tee -a run2/train.log
fi
# ---- serial 20, the user: the model is for indie-game NPCs (distributed, commercial): it need not say hard things, but must
# answer what it is asked. A second fine-tuning, on commercially usable conversations only (17,763: SmolTalk everyday and
# systemchats with short replies and no AI talk - Apache-2.0; Synthetic-Persona-Chat with its persona as a character card -
# CC BY 4.0), from the first fine-tuning's final weights, one pass; held out: their test splits (299); sample replies to NPC
# cards. Waits for the first fine-tuning to finish. ----
if [ ! -e /root/.ch_npc20 ]; then touch /root/.ch_npc20
  cat > /root/chnpc20.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/charlm; mkdir -p npcdata
until grep -q CHSFT_DONE sft.log 2>/dev/null; do sleep 300; done
for f in stage2_public.jsonl stage2_val.jsonl stage2_prompts.json; do
  [ -s npcdata/$f ] || curl -sSfL -o npcdata/$f "https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/charlm/npc_data/$f"; done
B=sft1/model_final.pt; [ -s $B ] || B=sft1/model_best.pt
echo "[npc20] the NPC fine-tuning from $B: $(wc -l < npcdata/stage2_public.jsonl) conversations $(date -u)" | tee -a run2/train.log
python3 -u charlm_sft.py --base $B --data sftdata --out npc1 --shards 0 --jsonl npcdata/stage2_public.jsonl --val-jsonl npcdata/stage2_val.jsonl \
  --prompts-json npcdata/stage2_prompts.json --epochs 1 --lr 1e-4 --warmup 50 --eval-every 5 --hours 3 --hub baya1116/hypernet-sp-distill:charlm/npc1 2>&1 | grep -vE "Warning|warn" > npc1.out
grep -h "Error\|Traceback" npc1.out | tail -3 | sed 's/^/[npc20] /' | tee -a run2/train.log
echo "[npc20] NPC_DONE $(date -u)" | tee -a run2/train.log
JOB
  setsid nohup bash -c 'bash /root/chnpc20.sh >> /root/charlm/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[npc20] queued after the first fine-tuning $(date -u)" | tee -a /root/charlm/run2/train.log
fi
# ---- serial 21: serial 20's wait took the CHSFT_DONE left in sft.log by the night's out-of-memory tries for the end of the
# running fine-tuning: the NPC run started at once beside it and ran out of GPU memory (the first went on). It now waits for
# this run's own CHARLM_SFT_DONE in sft1/train.log with no charlm_sft.py left running. ----
if [ ! -e /root/.ch_npc21 ]; then touch /root/.ch_npc21
  pkill -f "chnpc2[0].sh"; cd /root/charlm && mkdir -p sft_old && mv npc1 sft_old/npc1_oom 2>/dev/null; mv npc1.out sft_old/npc1_oom.out 2>/dev/null
  sed -i 's/^until grep -q CHSFT_DONE sft.log 2>\/dev\/null; do sleep 300; done$/until grep -q CHARLM_SFT_DONE sft1\/train.log 2>\/dev\/null \&\& ! pgrep -f "charlm_sf[t].py" >\/dev\/null; do sleep 300; done/' /root/chnpc20.sh
  grep -n "^until" /root/chnpc20.sh | sed 's/^/[npc21] wait: /' | tee -a run2/train.log
  setsid nohup bash -c 'bash /root/chnpc20.sh >> /root/charlm/job.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[npc21] the NPC fine-tuning re-queued after the running one $(date -u)" | tee -a run2/train.log
fi
# ---- serial 22: the early start may have fetched the NPC data before the 1,894 NPC dialogues were added to it: the copies
# go, so the queued run fetches the current files. ----
if [ ! -e /root/.ch_npcdata22 ]; then touch /root/.ch_npcdata22
  rm -f /root/charlm/npcdata/stage2_public.jsonl /root/charlm/npcdata/stage2_val.jsonl
  echo "[npc21] stale NPC data removed; fetched afresh when the run starts $(date -u)" | tee -a /root/charlm/run2/train.log
fi
echo "BOXCH_OK serial $BOXCH_SERIAL $(date -u)"
# CTL-END
