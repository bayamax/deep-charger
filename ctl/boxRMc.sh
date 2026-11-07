#!/bin/bash
# Candidate box for the RMT run (the user: start-up check, GPU check, then keep the cheapest that works): GPU empty?,
# bf16 matmul speed, model download time. Prints one [pr] RESULT line and waits; the winner gets the training serial.
# serial 2: this box passed (GPU empty, 68 TFLOPS bf16, model in 35 s) and is the cheapest that did - the RMT run moves here
# from box RM after its segment 2: ctl/rm_job.sh carries on from the hub's spRM_latest.pt.
# serial 3: the held-out mean is flat (0.0656 / 0.0660 / 0.0651 / 0.0646 over segments 10-13) - stop the training now (segment
# 14 is killed, segment 13's weights stay) and go straight to the 60 problems with and without the memory.
# serial 4: the frozen-reader run scored 38/60 with the memory (seed 1; the raw window alone 42-44) - the paper-faithful phase
# (ctl/rmf_job.sh: shared read/write weights, copy writes, all layers tuned, from segment 13) starts once the memory run at
# seed 2 is done; the no-memory runs of the frozen-reader job are dropped (that is the untouched model's raw window).
# serial 5: the status lines show the fine-tune phase's own log (sp_f.log).
# serial 6: the fine-tune at lr 1e-5 did not move off the floor in two segments (held-out mean 0.0938, 0.0947; chat worse than
# the floor - the model drifting from the original) - the user chose to go on at a lower lr: 2e-6, from segment 2.
# serial 7: the user - KV-level memory, made by the light separate hypernet (ctl/kvm_job.sh): the RMT fine-tune stops after
# its segment 5 (kept on the hub as spRMF_latest.pt) and the KV-memory pooler trains from scratch on the frozen model.
# serial 8: no need to wait for the RMT fine-tune's segment 5 (the user) - switch to the KV-memory hypernet now.
# serial 9: the account ran out of credit and the box was stopped mid segment 8 - after the restart the KV-memory job is
# launched again (it carries on from its local kvm.pt = segment 7).
# serial 10: enough (the user) - stop the KV-memory training after segment 8 and go to the 60 problems (seeds 1-2).
# serial 11: the user is asleep and asked me to keep trying. KV memory scored 35 / 37 (raw window 42 / 44); its generations fail
# to finish (17 unclosed vs 8-10) with ~60% more "Wait" / "Alternatively" - the hesitation drift again. ctl/kvr_job.sh, once the
# KV job is done: stopping / hesitation / entropy statistics with and without the memory, then 3 segments with a half reverse-KL
# objective from segment 8, then the 60 problems (seeds 1-2).
# serial 12: serial 11 reached the box before the switch to --ctl-window and started the half reverse-KL version of
# kvr_job.sh - once its statistics are done (kr_diag_done), stop it, drop anything it trained, and run the current kvr_job.sh
# (it skips the statistics and goes to the --ctl-window training).
# serial 13: the first statistics eval has sat at 0% GPU (5.6 GB) for 40 minutes - print what the processes are doing.
# serial 14: the eval spins one CPU core at 100% with the GPU idle (py-spy not allowed) - GPU check, stop it, and rerun the
# same eval on 2 traces with a faulthandler traceback after 4 minutes to see where it hangs.
# serial 15: the same eval on 2 traces runs fine and the raw-window one on all 65 took minutes - one held-out trace makes the
# memory eval spin. Watchdog: until the statistics are done, an eval older than 45 min is stopped (the job moves on); the
# current kvr_job.sh re-measures the memory per domain and puts a timeout on every eval.
# serial 16: --ctl-window segment 1 scored 40/60 with 3 closed-but-wrong, but 18 unclosed - the drift sits in what paragraphs
# open with. Stop the --ctl-window job now (its segment 2 is dropped; segment 1 stays as kvr.pt / spKVR_latest.pt) and start
# ctl/kvp_job.sh (--ctl-para) from segment 1.
# serial 17: --ctl-para segment 1: 40/60, unclosed 18 -> 13 (raw window 8 / 10). Once the --ctl-para job is done, decode its
# last weights with --para-window as well (ctl/kvw_job.sh, seeds 1-2).
# serial 18: para-window decoding seed 1: 37/60 (plain 35). Once that job is done, two more decodings of the same weights with
# the memory-masked twin (ctl/kvt_job.sh): the lower-entropy side per token, and a half/half logit mix.
# serial 19: the user - DPO with the raw-window trajectories as the ideal. The twin decodings of serial 18 are dropped; once the
# para-window job is done, generate the DPO data (ctl/kvd_job.sh: 300 training problems, raw window and memory).
# serial 20: once the DPO data is done, the DPO run itself (ctl/kvo_job.sh: pairs, training of the memory hypernet with the
# sequence log-probs under the KV-memory loop, the 60 problems at seeds 1-2).
# serial 21: the user - round 1 first (all 300), then a second round: the raw window redrawn on the problems both sides
# failed (ctl/kvd2_job.sh), then DPO on the pairs from both rounds. The serial-20 waiter is replaced.
# serial 22: the user - grow the DPO corpus step by step towards the pairs it needs (~200-300): after the first DPO run
# (kvo), ctl/kvi_job.sh iterates: 300 new problems, raw window + memory with the current weights, redraws, pairs both ways, DPO,
# the 60 problems and the held-out KL - up to 6 iterations, stopping early if it falls 4+ below the start twice running.
# serial 23: the status lines show the newest sp_*.log (the DPO run's own log was not visible).
# serial 24: DPO with the hypernet alone did nothing on the 60 problems (39 / 35) - the user: unfreeze the model, no KL anchor,
# and collect the data in parallel on cheaper boxes. The iterative hypernet-only loop (kvi) stops; ctl/kvf_job.sh trains all 28
# layers + the hypernet on the 53 pairs, publishes the weights for the collectors, then trains on their pairs in rounds and
# collects its own shard while it waits.
# serial 25: DPO round 2 did not generalise (48% / 56% chosen-ahead on its unseen pairs) and its 60 problems fell to 37 (closed
# 53 - it learned to stop, not to be right). Switch to GRPO with raw-window guides (ctl/kvg_job.sh) from the round-1 weights
# (kvf_prev.pt: 42 / 39), published first as the next version; the collectors become rollout workers.
# serial 26: the user - plain GRPO, groups of 12 at least, as many updates as possible. Restart the trainer with the current
# kvg_job.sh (24 new problems per round, groups of 12 only, the 60 problems every 3rd round); it carries on from its round 1.
# serial 27: 60 problems at v9: 36 (seed 1) / 35 (seed 2), closed 59 - the GRPO so far taught closing, not correctness
# (training-problem accuracy flat at ~56-58% over v6-v9, repeating tails up). Likely cause: 4 groups per update is a tiny,
# noisy batch (DeepScaleR-style runs use ~128 prompts per update) - only the consistent signal (close) gets learned. After the
# running round the trainer restarts with the current kvg_job.sh at 16 groups per update (flags re-read every round).
# serial 29: write kvg_newg (48 problems per round) even if serial 28 already ran.
# serial 28: the GRPO is coming apart - training-problem accuracy v8 58.6% -> v9 55.2% -> v10 49.2% (100+ problems each),
# answers shorter, the 60 problems at v9 37 (avg of 3). Stop the running round now and start over from v3 (the round-1 DPO
# weights, 42 / 39) at 16 groups per update and model lr 5e-7, with the parameter-movement tracking from the start; v3 goes
# out as the next version so the workers sample from it, older groups are not used.
# serial 30: CPU-only diagnosis while the GRPO restart runs: how far did the first GRPO run (v3 -> v10) move the model vs
# the hypernet (exact relative change, ctl/param_diff.py), also v3 -> v9 if kept.
# serial 31: the first GRPO run learned to guess - wrong answers went from running to the cap to closing early (wrong-but-
# closed 46% -> 95% of the wrong ones, their median length 29.7k -> 13.2k chars, distinct answers per 4 draws 1.27 -> 2.24):
# with 0 for unclosed and for wrong alike, any answer beats thinking on. Closed-but-wrong now scores -0.2 (guessing pays only
# above 1/6). The restarted run has not trained yet: restart its trainer with the current kvg_job.sh / sp_qwen3.py (state kept).
# serial 32: the user - the guessing comes from the training settings (tiny noisy updates, high lr), not the reward; change
# one thing at a time. Closed-but-wrong back to 0 (the standard 1 / 0 reward) from the next round on (kvg_wrong is re-read).
# serial 33: diagnosis of run 1 - the model's layers moved 5-9e-5 (relative), the size of ~40 Adam steps of lr 1e-6 in
# uncorrelated directions (sqrt(40) x 1e-6): a random walk, not a direction. The hypernet ran at lr 2e-5 (20-40x the model's)
# and so walked 20-40x as far - set its lr to the model's 5e-7 (kvg_plr, re-read each round). Also: the totals of the
# v3 -> v9 / v10 diff (the top lines were lost from the log), and what is running.
# serial 34: the cause of run 1 (and of the DPO rounds' faint effect): the model's weights are bf16, one bf16 step is 4e-3
# relative, and the fp32 masters moved 5-9e-5 in the whole run - the updates rounded away on the copy back; only the fp32
# hypernet moved (at 20x the lr) and drifted. Run 2: the model through fp32 LoRA (rank 32 on q/k/v/o and the MLP of all
# 28 layers, lr 1e-5), the hypernet frozen (lr 0), 16 groups per update, 48 problems per round, reward 1 / 0, from v3
# (kvf_round1.pt). Published first as the next version (B = 0: the same policy), groups more than one version old are skipped.
# serial 35: the user - restart from a model close to the base that did well: dpo_ref.pt (= kvr.pt, the --ctl-window
# segment-1 hypernet on the UNTOUCHED base model; 40/60, 3 closed-but-wrong) instead of v3 (whose hypernet carries a DPO
# round). Same run-2 settings (fp32 LoRA rank 32 lr 1e-5, hypernet frozen, 16 groups per update, 48 problems per round).
# serial 36: the container log drops lines (the round summaries went missing) - a side loop uploads the [kg]/[kc] lines of
# w.log to the hub (archive/sp3/kvg/wlog.txt) every 10 min; no restart of the trainer.
# serial 37: (a) the round-3 eval appended to run 1's kvg_r3_s1.jsonl (73/60 in the log) - run 2's rows separated offline:
# 39/60; old eval files moved away, evals now named by version. (b) the LoRA steps of rounds 2 and 3 are uncorrelated
# (cos +0.06): every round starts a fresh optimizer, so its single update is lr x sign(grad) for every parameter and noise
# moves as far as signal. The optimizer state now persists across rounds (--opt-state kvg_opt.pt): Adam's momentum averages
# the rounds. The trainer restarts after its running round (STOP_KG, state kept).
# serial 38: round 5 trained on 9 informative groups (48 problems, 19% mixed) - a round now also waits for 16 informative
# groups (kvg_newinf), so every update is a full one. Trainer restarts after its running round (state and optimizer kept).
# serial 39: the cosine of consecutive optimizer steps stopped meaning anything once Adam's state carried across rounds
# (momentum alone gives +0.6 for pure-noise gradients; v19 read +0.55). The training now sketches the LoRA gradient itself
# (--grad-sketch): even vs odd groups of a round, and each round against the last, in the round line and rounds.txt.
# The trainer restarts after its running round (state and optimizer kept).
# serial 40: the hypernet's gradient (computed even at lr 0) is sketched too - its even/odd agreement says whether unfreezing
# it has a signal to follow; both in the round line and rounds.txt. The trainer restarts after its running round.
# serial 41: round 11 - the hypernet's gradient agrees across groups far more than the LoRA's (even/odd cos +0.61 against +0.12):
# a direction worth following. The hypernet is unfrozen at lr 1e-6 (kvg_plr; Adam's state for it has been accumulating
# at lr 0) from the next round; the weights before it are kept as kvg_before_hyper.pt (hub history has them too).
# serial 42: v25 (the first round with the hypernet unfrozen) read 36/60 with 11 closed-but-wrong (5-7 before) - the guessing
# of run 1? The hypernet is frozen again from the next round (kvg_plr 0) while v25 at seed 2 and v24 at seed 1 settle it.
# serial 43: v24 (last frozen-hypernet weights) 40/60; v25 36 and 38. With v18-v24 at 37-41 (mean 39.2) the unfrozen hypernet
# looks ~2 lower, math most (23-24 of 36 against 25-28). After its running round the trainer stops, its weights get
# v24's hypernet back (the LoRA of the rounds since is kept), and it starts again (hypernet still frozen).
# serial 44: s43 found the hypernet of v26 equal to v24's (largest change 0.0): the "unfrozen" rounds 12-13 never moved it.
# torch's load_state_dict brings back the learning rates saved with the optimizer state, so since round 6 (the first
# carried state, saved with hypernet lr 0) every lr flag was ignored; sp_qwen3.py now puts this run's lrs back. v25's
# 36/38 was LoRA-only noise. The hypernet is unfrozen at lr 1e-6 for real (weights before it kept as kvg_before_hyper2.pt).
# serial 45: s44 left the fixed sp_qwen3.py as sp_new.py (round 15 was training) and round 16 started straight after - it
# likely runs the old code (hypernet lr still 0). The fixed file goes in as soon as no training runs.
# serial 46: the user - train hardest where the memory fails (raw window right, memory wrong), hypernet unfrozen. Workers
# give the problems whose first 4 memory draws all failed one raw-window draw as a guide (GUIDE=2); the gate now counts
# a group made informative by its guide. The trainer restarts after its running round (state, lrs and hypernet lr kept).
# serial 47: the credit is nearly out (the user tops it up later, maybe) - the trainer's state goes to the hub
# (archive/sp3/kvg/state/: weights, optimizer state, round / version counters, used group files, hypernet lr) so the
# run can carry on from it; copies first, so a training round that writes meanwhile does not tear them.
# serial 48: why no gain - in 24 updates the LoRA moved ~5e-4 of the weights (a bf16 step is 4e-3) and the 60-problem answers of v30
# differ from v18's no more than two seeds of one version do (7-9 flips against 7-10): the direction was there, the distance
# was not. LoRA lr 1e-5 -> 5e-5, 8 groups per update instead of 16 (about 10x the movement per round for the same data),
# hypernet lr 1e-6 -> 5e-6; read every round (kvg_loralr / kvg_accum / kvg_plr).
# serial 49: collapse. v34 read 22/60 with 37 closed-but-wrong and 1 unclosed (v31: 38 / 11 / 12); in training v33 already
# closed 89% with 32% closed-but-wrong (v32: 69% / 14%) - the "close now" direction (gradient halves +0.93) won at LoRA lr
# 5e-5 with momentum building. Now: the running training is stopped, the weights go back to v32 (hub commit 6631be0426)
# and are published as v35, the optimizer state is dropped (its momentum points to the collapse), every group drawn so
# far is marked used (v33 / v34 drew from the collapsing policy), LoRA lr 2e-5, hypernet lr 1e-6, 8 groups per update.
# serial 50: the workers finish their running blocks with v34 (the collapsed weights): for the next 3 hours every group file
# drawn with v33 or v34 that reaches the hub is marked used as it arrives.
# serial 51: the user sleeps; the cheap 3090 must last until the morning (~02:00Z), the credit (2.34$ at 17:31Z) does not pay
# for the six 3060s that long. At 18:50Z the workers are deleted (from the controller) and this box carries on alone: its own
# rollouts on the weak-area pool (shard 5, 12 problems a block, raw-window guides), a round at 8 informative groups (one
# update), the 60 problems after every round. The trainer restarts at 18:50Z (its own rollout killed; a training round is
# never interrupted).
# serial 52: v37 read 41/60 (the best so far, with v22) - its state goes to the hub (archive/sp3/kvg/state/) so the run can
# carry on from it after the credit; the trainer is not disturbed.
SER=52
if [ -f /root/.boxrmx_serial ] && [ "$(cat /root/.boxrmx_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxrmx_serial
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w
cat > /root/status.sh <<'ST'
grep -E "^\[(pr|rm|kv|kr|kp|kw|kt|kd|ko|ki|kf|kg|hub)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 6 | cut -c1-600
tail -n 1 $(ls -t /root/w/sp_*.log 2>/dev/null | head -1) 2>/dev/null | cut -c1-250
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.probe ]; then touch /root/.probe
  cat > /root/probe.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
U0=$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)
echo "[pr] preflight GPU: $U0"
python3 -m pip install -q "huggingface_hub[hf_xet]" transformers "math-verify[antlr4_13_2]" 2>&1 | grep -v -i warn | tail -1
t0=$(date +%s); timeout 1500 python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B')" > dl.log 2>&1; dl=$(( $(date +%s) - t0 ))
tf=$(python3 -c "
import torch, time
x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16); y = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
for _ in range(5): x @ y
torch.cuda.synchronize(); t = time.time()
for _ in range(50): x @ y
torch.cuda.synchronize(); print(round(50 * 2 * 4096**3 / (time.time() - t) / 1e12, 1))" 2>&1 | tail -1)
echo "[pr] RESULT gpu ($U0) | model download ${dl}s | bf16 matmul ${tf} TFLOPS | $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/probe.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.rmjob ]; then touch /root/.rmjob
  curl -sS -o /root/rm_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/rm_job.sh?nocache=$(date +%s)"
  setsid nohup bash -c 'bash /root/rm_job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[rm] RM job launched here $(date -u)"
fi
if [ ! -e /root/.s3stop ]; then touch /root/.s3stop
  touch /root/w/STOP_RM; pkill -f "sp_qwen3.py train"; echo "[rm] s3: training stopped after segment $(cat /root/w/rm_seg) - on to the 60 problems $(date -u)"
fi
if [ ! -e /root/.s4f ]; then touch /root/.s4f
  cat > /root/s4.sh <<'JOB'
while ! grep -q "segment 13 mem seed 2" /root/w/w.log; do sleep 30; done
for J in $(pgrep -f "^bash /root/rm_job.sh"); do pkill -P $J; kill $J; done; sleep 3; pkill -f "sp_qwen3.py"; sleep 3
curl -sS -o /root/rmf_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/rmf_job.sh?nocache=$(date +%s)"
echo "[rm] s4: frozen-reader job stopped after the seed-2 memory run; paper-faithful phase starting $(date -u)"
exec bash /root/rmf_job.sh
JOB
  setsid nohup bash -c 'bash /root/s4.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[rm] s4: waiting for the seed-2 memory run $(date -u)"
fi
if [ ! -e /root/.s6lr ]; then touch /root/.s6lr
  for J in $(pgrep -f "^bash /root/rmf_job.sh") $(pgrep -f "^bash /root/s4.sh"); do pkill -P $J; kill $J; done; sleep 3; pkill -f "sp_qwen3.py"; sleep 3
  curl -sS -o /root/rmf_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/rmf_job.sh?nocache=$(date +%s)"
  setsid nohup bash -c 'FTLR=2e-6 bash /root/rmf_job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[rm] s6: fine-tune restarted at lr 2e-6 from segment $(cat /root/w/rmf_seg) $(date -u)"
fi
if [ ! -e /root/.s10stop ]; then touch /root/.s10stop
  touch /root/w/STOP_KV; pkill -f "sp_qwen3.py train"; echo "[kv] s10: training stopped after segment $(cat /root/w/kvm_seg) - on to the 60 problems $(date -u)"
fi
if [ ! -e /root/.s9re ] && ! pgrep -f "^bash /root/kvm_job.sh" > /dev/null; then touch /root/.s9re
  [ -s /root/kvm_job.sh ] || curl -sS -o /root/kvm_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvm_job.sh?nocache=$(date +%s)"
  setsid nohup bash -c 'bash /root/kvm_job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[kv] s9: KV-memory job relaunched after the restart, from segment $(cat /root/w/kvm_seg 2>/dev/null) $(date -u)"
fi
if [ ! -e /root/.s8kv ]; then touch /root/.s8kv; touch /root/.s7kv
  for J in $(pgrep -f "^bash /root/s7.sh") $(pgrep -f "^bash /root/rmf_job.sh"); do pkill -P $J; kill $J; done; sleep 3; pkill -f "sp_qwen3.py"; sleep 3
  curl -sS -o /root/kvm_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvm_job.sh?nocache=$(date +%s)"
  setsid nohup bash -c 'bash /root/kvm_job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[kv] s8: RMT fine-tune stopped now (segment 4 on the hub); KV-memory hypernet started $(date -u)"
fi
if [ ! -e /root/.s7kv ]; then touch /root/.s7kv
  cat > /root/s7.sh <<'JOB'
while ! grep -q "^\[rm\] F segment 5 " /root/w/w.log; do sleep 30; done
for J in $(pgrep -f "^bash /root/rmf_job.sh"); do pkill -P $J; kill $J; done; sleep 3; pkill -f "sp_qwen3.py"; sleep 3
curl -sS -o /root/kvm_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvm_job.sh?nocache=$(date +%s)"
echo "[kv] s7: RMT fine-tune stopped after its segment 5; KV-memory hypernet starting $(date -u)"
exec bash /root/kvm_job.sh
JOB
  setsid nohup bash -c 'bash /root/s7.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[kv] s7: waiting for the RMT fine-tune's segment 5 $(date -u)"
fi
if [ ! -e /root/.s11kr ]; then touch /root/.s11kr
  cat > /root/s11.sh <<'JOB'
while [ ! -e /root/w/DONE_KV ]; do sleep 30; done; sleep 60
curl -sS -o /root/kvr_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvr_job.sh?nocache=$(date +%s)"
echo "[kr] s11: KV job done - hesitation statistics, then half reverse-KL training $(date -u)"
exec bash /root/kvr_job.sh
JOB
  setsid nohup bash -c 'bash /root/s11.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[kr] s11: waiting for the KV job to finish $(date -u)"
fi
if [ ! -e /root/.s12ctl ]; then touch /root/.s12ctl
  cat > /root/s12.sh <<'JOB'
while [ ! -e /root/w/kr_diag_done ]; do sleep 20; done
for J in $(pgrep -f "^bash /root/kvr_job.sh") $(pgrep -f "^bash /root/s11.sh"); do pkill -P $J; kill $J; done; sleep 3; pkill -f "sp_qwen3.py"; sleep 5
cd /root/w && rm -f kvr.pt kvr_next.pt kvr_seg
for _t in $(seq 60); do curl -sS -o /root/kvr_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvr_job.sh?nocache=$(date +%s)"; grep -q -- "--ctl-window --ckpt" /root/kvr_job.sh && break; sleep 120; done
echo "[kr] s12: half reverse-KL version stopped after the statistics; --ctl-window job ($(grep -c ctl-window /root/kvr_job.sh) mentions) $(date -u)"
exec bash /root/kvr_job.sh
JOB
  setsid nohup bash -c 'bash /root/s12.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[kr] s12: waiting for the statistics to finish $(date -u)"
fi
if [ ! -e /root/.s13dbg ]; then touch /root/.s13dbg
  echo "[kr] s13 debug $(date -u)"; ps aux | grep -E "sp_qwen3|kvr_job|s1[12].sh" | grep -v grep | cut -c1-250 | sed 's/^/[kr] ps: /'
  tail -n 5 /root/w/run_kr.log 2>/dev/null | cut -c1-300 | sed 's/^/[kr] run_kr: /'; free -g | sed 's/^/[kr] mem: /'
  python3 -m pip install -q py-spy > /dev/null 2>&1; for P in $(pgrep -f "sp_qwen3.py eval"); do timeout 60 py-spy dump --pid $P 2>&1 | head -n 30 | sed 's/^/[kr] spy: /'; done
fi
if [ ! -e /root/.s14dbg ]; then touch /root/.s14dbg
  cat > /root/s14.sh <<'JOB'
cd /root/w; echo "[kr] s14 start $(date -u)"
timeout 120 python3 -c "
import torch, time; x = torch.randn(4096, 4096, device='cuda'); torch.cuda.synchronize(); t = time.time(); [x @ x for _ in range(20)]; torch.cuda.synchronize(); print('[kr] s14 gpu matmul ok', round(time.time() - t, 3), 's')" 2>&1 | tail -1
pkill -f "sp_qwen3.py eval --traces rm_held.jsonl"; sleep 5
head -n 2 rm_held.jsonl > dbg2.jsonl
timeout 600 python3 -X faulthandler -c "
import faulthandler, sys, runpy; faulthandler.dump_traceback_later(240, exit=True)
sys.argv = ['sp_qwen3.py', 'eval', '--traces', 'dbg2.jsonl', '--max-len', '6144'] + '--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512 --end-stat --ckpt kvm.pt --log dbg.log'.split()
runpy.run_path('sp_qwen3.py', run_name='__main__')" > dbg.out 2>&1
grep -E "\[sp\] (eval|end-stat)|File|Error|line" dbg.out | grep -v "site-packages/tqdm" | tail -n 25 | cut -c1-250 | sed 's/^/[kr] dbg: /'
echo "[kr] s14 done $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/s14.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s15wd ]; then touch /root/.s15wd
  cat > /root/s15.sh <<'JOB'
while [ ! -e /root/w/kr_diag_done ]; do
  for P in $(pgrep -f "sp_qwen3.py eval"); do [ "$(ps -o etimes= -p $P | tr -d ' ')" -gt 2700 ] 2>/dev/null && { kill $P; echo "[kr] s15: stopped eval $P after 45 min $(date -u)"; }; done; sleep 60; done
JOB
  setsid nohup bash -c 'bash /root/s15.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s16kp ]; then touch /root/.s16kp
  for J in $(pgrep -f "^bash /root/kvr_job.sh") $(pgrep -f "bash /root/s12.sh"); do pkill -P $J; kill $J; done; sleep 3; pkill -f "sp_qwen3.py"; sleep 5
  cat > /root/s16.sh <<'JOB'
for _t in $(seq 60); do curl -sS -o /root/kvp_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvp_job.sh?nocache=$(date +%s)"; grep -q -- "--ctl-para --ckpt" /root/kvp_job.sh && break; sleep 120; done
echo "[kp] s16: --ctl-window job stopped (kept segment $(cat /root/w/kvr_seg)); --ctl-para job starting $(date -u)"
exec bash /root/kvp_job.sh
JOB
  setsid nohup bash -c 'bash /root/s16.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s17kw ]; then touch /root/.s17kw
  cat > /root/s17.sh <<'JOB'
while [ ! -e /root/w/DONE_KP ]; do sleep 30; done; sleep 30
for _t in $(seq 60); do curl -sS -o /root/kvw_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvw_job.sh?nocache=$(date +%s)"; grep -q -- "--para-window --max-len" /root/kvw_job.sh && break; sleep 120; done
echo "[kw] s17: --ctl-para job done - para-window decoding $(date -u)"
exec bash /root/kvw_job.sh
JOB
  setsid nohup bash -c 'bash /root/s17.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[kw] s17: waiting for the --ctl-para job $(date -u)"
fi
if [ ! -e /root/.s18kt ]; then touch /root/.s18kt
  cat > /root/s18.sh <<'JOB'
while [ ! -e /root/w/DONE_KW ]; do sleep 30; done; sleep 30
for _t in $(seq 60); do curl -sS -o /root/kvt_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvt_job.sh?nocache=$(date +%s)"; grep -q -- "twin-minent" /root/kvt_job.sh && break; sleep 120; done
echo "[kt] s18: para-window job done - twin-minent / twin-mix decoding $(date -u)"
exec bash /root/kvt_job.sh
JOB
  setsid nohup bash -c 'bash /root/s18.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s19kd ]; then touch /root/.s19kd
  for J in $(pgrep -f "bash /root/s18.sh") $(pgrep -f "^bash /root/kvt_job.sh"); do pkill -P $J; kill $J; done; pkill -f "kvt_.*jsonl"
  cat > /root/s19.sh <<'JOB'
while [ ! -e /root/w/DONE_KW ]; do sleep 30; done; sleep 30
for _t in $(seq 60); do curl -sS -o /root/kvd_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvd_job.sh?nocache=$(date +%s)"; grep -q -- "dpo_prompts300" /root/kvd_job.sh && break; sleep 120; done
echo "[kd] s19: para-window job done - DPO data generation $(date -u)"
exec bash /root/kvd_job.sh
JOB
  setsid nohup bash -c 'bash /root/s19.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[kd] s19: twin decodings dropped; waiting for the para-window job $(date -u)"
fi
if [ ! -e /root/.s20ko ]; then touch /root/.s20ko
  cat > /root/s20.sh <<'JOB'
while [ ! -e /root/w/DONE_KD ]; do sleep 30; done; sleep 30
for _t in $(seq 60); do curl -sS -o /root/kvo_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvo_job.sh?nocache=$(date +%s)"; grep -q -- "dpo_pairs" /root/kvo_job.sh && break; sleep 120; done
echo "[ko] s20: DPO data done - DPO training $(date -u)"
exec bash /root/kvo_job.sh
JOB
  setsid nohup bash -c 'bash /root/s20.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[ko] s20: waiting for the DPO data $(date -u)"
fi
if [ ! -e /root/.s21kd2 ]; then touch /root/.s21kd2
  for J in $(pgrep -f "bash /root/s20.sh"); do pkill -P $J; kill $J; done
  cat > /root/s21.sh <<'JOB'
while [ ! -e /root/w/DONE_KD ]; do sleep 30; done; sleep 30
for _t in $(seq 60); do curl -sS -o /root/kvd2_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvd2_job.sh?nocache=$(date +%s)"; grep -q -- "dpo_r2_todo" /root/kvd2_job.sh && break; sleep 120; done
echo "[kd] s21: round 1 done - round 2 $(date -u)"; bash /root/kvd2_job.sh
for _t in $(seq 60); do curl -sS -o /root/kvo_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvo_job.sh?nocache=$(date +%s)"; grep -q -- "dpo_raw2_" /root/kvo_job.sh && break; sleep 120; done
echo "[ko] s21: round 2 done - DPO training $(date -u)"
exec bash /root/kvo_job.sh
JOB
  setsid nohup bash -c 'bash /root/s21.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[kd] s21: waiting for round 1 $(date -u)"
fi
if [ ! -e /root/.s22ki ]; then touch /root/.s22ki
  cat > /root/s22.sh <<'JOB'
while [ ! -e /root/w/DONE_KO ]; do sleep 60; done; sleep 30
for _t in $(seq 60); do curl -sS -o /root/kvi_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvi_job.sh?nocache=$(date +%s)"; grep -q -- "dpo_pairs.py" /root/kvi_job.sh && break; sleep 120; done
echo "[ki] s22: first DPO run done - iterative DPO $(date -u)"
exec bash /root/kvi_job.sh
JOB
  setsid nohup bash -c 'bash /root/s22.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[ki] s22: waiting for the first DPO run $(date -u)"
fi
if [ ! -e /root/.s24kf ]; then touch /root/.s24kf
  for J in $(pgrep -f "bash /root/s22.sh") $(pgrep -f "^bash /root/kvi_job.sh"); do pkill -P $J; kill $J; done; sleep 3; pkill -f "sp_qwen3.py"; sleep 5
  cat > /root/s24.sh <<'JOB'
for _t in $(seq 60); do curl -sS -o /root/kvf_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvf_job.sh?nocache=$(date +%s)"; grep -q -- "NSH" /root/kvf_job.sh && break; sleep 120; done
echo "[kf] s24: iterative hypernet-only DPO stopped - DPO with the model unfrozen $(date -u)"
exec bash /root/kvf_job.sh
JOB
  setsid nohup bash -c 'bash /root/s24.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s25kg ]; then touch /root/.s25kg
  for J in $(pgrep -f "bash /root/s24.sh") $(pgrep -f "^bash /root/kvf_job.sh") $(pgrep -f "bash kvc_job.sh"); do pkill -P $J; kill $J; done; sleep 3; pkill -f "sp_qwen3.py"; sleep 5
  cp /root/w/kvf_prev.pt /root/w/kvf_round1.pt
  cat > /root/s25.sh <<'JOB'
for _t in $(seq 60); do curl -sS -o /root/kvg_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvg_job.sh?nocache=$(date +%s)"; grep -q -- "kvg_v0" /root/kvg_job.sh && break; sleep 120; done
echo "[kg] s25: DPO stopped - GRPO with guides from the round-1 weights $(date -u)"
START=kvf_round1.pt exec bash /root/kvg_job.sh
JOB
  setsid nohup bash -c 'bash /root/s25.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s26kg ]; then touch /root/.s26kg
  for J in $(pgrep -f "bash /root/s25.sh") $(pgrep -f "^bash /root/kvg_job.sh") $(pgrep -f "bash kvc_job.sh"); do pkill -P $J; kill $J; done; sleep 3; pkill -f "sp_qwen3.py"; sleep 5
  cat > /root/s26.sh <<'JOB'
for _t in $(seq 60); do curl -sS -o /root/kvg_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvg_job.sh?nocache=$(date +%s)"; grep -q -- "GMIN" /root/kvg_job.sh && break; sleep 120; done
echo "[kg] s26: plain GRPO, groups of 12 $(date -u)"
exec bash /root/kvg_job.sh
JOB
  setsid nohup bash -c 'bash /root/s26.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s27kg ]; then touch /root/.s27kg
  echo 16 > /root/w/kvg_accum; touch /root/w/STOP_KG
  cat > /root/s27.sh <<'JOB'
while pgrep -f "^bash /root/kvg_job.sh" > /dev/null; do sleep 30; done
rm -f /root/w/STOP_KG
for _t in $(seq 60); do curl -sS -o /root/kvg_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvg_job.sh?nocache=$(date +%s)"; grep -q -- "kvg_accum" /root/kvg_job.sh && break; sleep 120; done
echo "[kg] s27: 16 groups per update from here $(date -u)"
exec bash /root/kvg_job.sh
JOB
  setsid nohup bash -c 'bash /root/s27.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s28kg ]; then touch /root/.s28kg
  for J in $(pgrep -f "bash /root/s27.sh") $(pgrep -f "bash /root/s26.sh") $(pgrep -f "^bash /root/kvg_job.sh") $(pgrep -f "bash kvc_job.sh"); do pkill -P $J; kill $J; done; sleep 3; pkill -f "sp_qwen3.py"; sleep 5
  cd /root/w && mkdir -p grpo_run1 && mv -f kvg.pt kvg_prev.pt kvg_round kvg_v0 sk_v*.pt grpo_run1/ 2>/dev/null; rm -f STOP_KG
  echo 16 > kvg_accum; echo 5e-7 > kvg_ftlr; echo 48 > kvg_newg   # ~16 informative groups = one full update per round
  cat > /root/s28.sh <<'JOB'
for _t in $(seq 60); do curl -sS -o /root/kvg_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvg_job.sh?nocache=$(date +%s)"; grep -q -- "param_sketch" /root/kvg_job.sh && break; sleep 120; done
echo "[kg] s28: GRPO over from v3, 16 groups per update, model lr 5e-7 $(date -u)"
cd /root/w && START=kvf_round1.pt exec bash /root/kvg_job.sh
JOB
  setsid nohup bash -c 'bash /root/s28.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
[ -e /root/w/kvg_newg ] || echo 48 > /root/w/kvg_newg
if [ ! -e /root/.s30diff ]; then touch /root/.s30diff
  cat > /root/s30.sh <<'JOB'
cd /root/w; curl -sS -o param_diff.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/param_diff.py?nocache=$(date +%s)"
for c in grpo_run1/kvg.pt grpo_run1/kvg_prev.pt; do [ -s $c ] && nice -n 19 python3 param_diff.py kvf_round1.pt $c 2>&1 | sed "s|^|[kg] diff v3 -> $c: |"; done
JOB
  setsid nohup bash -c 'bash /root/s30.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s31kg ]; then touch /root/.s31kg
  echo -0.2 > /root/w/kvg_wrong
  for J in $(pgrep -f "bash /root/s28.sh") $(pgrep -f "^bash /root/kvg_job.sh") $(pgrep -f "bash kvc_job.sh"); do pkill -P $J; kill $J; done; sleep 3; pkill -f "sp_qwen3.py rollout"; pkill -f "sp_qwen3.py train"; sleep 5
  cat > /root/s31.sh <<'JOB'
for _t in $(seq 60); do curl -sS -o /root/kvg_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvg_job.sh?nocache=$(date +%s)"; grep -q -- "kvg_wrong" /root/kvg_job.sh && break; sleep 120; done
echo "[kg] s31: closed-but-wrong = $(cat /root/w/kvg_wrong) $(date -u)"
cd /root/w && START=kvf_round1.pt exec bash /root/kvg_job.sh
JOB
  setsid nohup bash -c 'bash /root/s31.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
[ -e /root/.s32w ] || { touch /root/.s32w; echo 0 > /root/w/kvg_wrong; echo "[kg] s32: closed-but-wrong back to 0 $(date -u)"; }
if [ ! -e /root/.s33 ]; then touch /root/.s33
  echo 5e-7 > /root/w/kvg_plr; echo "[kg] s33: hypernet lr 5e-7 from the next round $(date -u)"
  ps aux | grep -E "sp_qwen3.py|kvg_job|kvc_job|s31.sh|s28.sh" | grep -v grep | awk '{print "[kg] ps:", $11, $12, $13, $14, $15, $16}' | cut -c1-150
  cat > /root/s33.sh <<'JOB'
cd /root/w; for c in grpo_run1/kvg.pt grpo_run1/kvg_prev.pt; do [ -s $c ] && nice -n 19 python3 param_diff.py kvf_round1.pt $c 2>&1 | head -n 6 | sed "s|^|[kg] diff2 v3 -> $c: |"; done
JOB
  setsid nohup bash -c 'bash /root/s33.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s34kg ]; then touch /root/.s34kg
  for J in $(pgrep -f "bash /root/s31.sh") $(pgrep -f "bash /root/s28.sh") $(pgrep -f "^bash /root/kvg_job.sh") $(pgrep -f "bash kvc_job.sh"); do pkill -P $J; kill $J; done; sleep 3; pkill -f "sp_qwen3.py rollout"; pkill -f "sp_qwen3.py train"; sleep 5
  cd /root/w && mkdir -p grpo_run2_pre && mv -f kvg.pt kvg_prev.pt kvg_round kvg_v0 sk_v*.pt grpo_run2_pre/ 2>/dev/null; rm -f STOP_KG
  echo 32 > kvg_lora; echo 1e-5 > kvg_loralr; echo 0 > kvg_ftlayers; echo 0 > kvg_plr; echo 16 > kvg_accum; echo 48 > kvg_newg; echo 0 > kvg_wrong
  cat > /root/s34.sh <<'JOB'
for _t in $(seq 60); do curl -sS -o /root/kvg_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvg_job.sh?nocache=$(date +%s)"; grep -q -- "kvg_loralr" /root/kvg_job.sh && break; sleep 120; done
echo "[kg] s34: run 2 - fp32 LoRA rank 32 lr 1e-5, hypernet frozen, 16 groups per update, from v3 $(date -u)"
cd /root/w && START=kvf_round1.pt exec bash /root/kvg_job.sh
JOB
  setsid nohup bash -c 'bash /root/s34.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s35kg ]; then touch /root/.s35kg
  for J in $(pgrep -f "bash /root/s34.sh") $(pgrep -f "bash /root/s31.sh") $(pgrep -f "^bash /root/kvg_job.sh") $(pgrep -f "bash kvc_job.sh"); do pkill -P $J; kill $J; done; sleep 3; pkill -f "sp_qwen3.py rollout"; pkill -f "sp_qwen3.py train"; sleep 5
  cd /root/w && mkdir -p grpo_run2_v3start && mv -f kvg.pt kvg_prev.pt kvg_round kvg_v0 sk_v*.pt grpo_run2_v3start/ 2>/dev/null; rm -f STOP_KG
  [ -s dpo_ref.pt ] || curl -sSfL -H "Authorization: Bearer $(tr -d '[:space:]' < /root/.hf_token)" -o dpo_ref.pt "https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/archive/sp3/spKVR_latest.pt"
  echo 32 > kvg_lora; echo 1e-5 > kvg_loralr; echo 0 > kvg_ftlayers; echo 0 > kvg_plr; echo 16 > kvg_accum; echo 48 > kvg_newg; echo 0 > kvg_wrong
  cat > /root/s35.sh <<'JOB'
for _t in $(seq 60); do curl -sS -o /root/kvg_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvg_job.sh?nocache=$(date +%s)"; grep -q -- "kvg_loralr" /root/kvg_job.sh && break; sleep 120; done
echo "[kg] s35: run 2 from the base model + the segment-1 hypernet (dpo_ref.pt) $(date -u)"
cd /root/w && START=dpo_ref.pt exec bash /root/kvg_job.sh
JOB
  setsid nohup bash -c 'bash /root/s35.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s36wlog ]; then touch /root/.s36wlog
  cat > /root/wlog.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token); cd /root/w
while true; do grep -E "^\[k[gc]\]|^\[sp\] (grpo groups|LoRA|trained)" w.log | tail -n 400 > wlog.txt
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='wlog.txt', path_in_repo='archive/sp3/kvg/wlog.txt', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1; sleep 600; done
JOB
  setsid nohup bash -c 'bash /root/wlog.sh > /dev/null 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[kg] s36: w.log mirror to the hub started $(date -u)"
fi
if [ ! -e /root/.s37opt ]; then touch /root/.s37opt
  cd /root/w && mkdir -p grpo_run1 && mv -f kvg_r[0-9]*_s1.jsonl grpo_run1/ 2>/dev/null; touch STOP_KG
  cat > /root/s37.sh <<'JOB'
while pgrep -f "^bash /root/kvg_job.sh" > /dev/null; do sleep 30; done
rm -f /root/w/STOP_KG
for _t in $(seq 60); do curl -sS -o /root/kvg_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvg_job.sh?nocache=$(date +%s)"; grep -q -- "opt-state" /root/kvg_job.sh && break; sleep 120; done
echo "[kg] s37: optimizer state across rounds from here $(date -u)"
cd /root/w && START=dpo_ref.pt exec bash /root/kvg_job.sh
JOB
  setsid nohup bash -c 'bash /root/s37.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s38inf ]; then touch /root/.s38inf
  echo 16 > /root/w/kvg_newinf; touch /root/w/STOP_KG
  cat > /root/s38.sh <<'JOB'
while pgrep -f "^bash /root/kvg_job.sh" > /dev/null; do sleep 30; done
rm -f /root/w/STOP_KG
for _t in $(seq 60); do curl -sS -o /root/kvg_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvg_job.sh?nocache=$(date +%s)"; grep -q -- "kvg_newinf" /root/kvg_job.sh && break; sleep 120; done
echo "[kg] s38: rounds gated on 16 informative groups from here $(date -u)"
cd /root/w && START=dpo_ref.pt exec bash /root/kvg_job.sh
JOB
  setsid nohup bash -c 'bash /root/s38.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s39gsk ]; then touch /root/.s39gsk
  touch /root/w/STOP_KG
  cat > /root/s39.sh <<'JOB'
while pgrep -f "^bash /root/kvg_job.sh" > /dev/null; do sleep 30; done
rm -f /root/w/STOP_KG
for _t in $(seq 60); do curl -sS -o /root/kvg_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvg_job.sh?nocache=$(date +%s)"; grep -q -- "grad-sketch" /root/kvg_job.sh && break; sleep 120; done
echo "[kg] s39: gradient sketches per round from here $(date -u)"
cd /root/w && START=dpo_ref.pt exec bash /root/kvg_job.sh
JOB
  setsid nohup bash -c 'bash /root/s39.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s40gskh ]; then touch /root/.s40gskh
  touch /root/w/STOP_KG
  cat > /root/s40.sh <<'JOB'
while pgrep -f "^bash /root/kvg_job.sh" > /dev/null; do sleep 30; done
rm -f /root/w/STOP_KG
for _t in $(seq 60); do curl -sS -o /root/kvg_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvg_job.sh?nocache=$(date +%s)"; grep -q -- "even_h" /root/kvg_job.sh && break; sleep 120; done
cd /root/w; for _t in $(seq 60); do curl -sS -o sp_qwen3.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q 'GSK_P' sp_qwen3.py && break; sleep 120; done
echo "[kg] s40: hypernet gradient sketched too from here $(date -u)"
START=dpo_ref.pt exec bash /root/kvg_job.sh
JOB
  setsid nohup bash -c 'bash /root/s40.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s41hyper ]; then touch /root/.s41hyper
  cd /root/w && cp kvg.pt kvg_before_hyper.pt && cat VERSION > kvg_before_hyper.version && echo 1e-6 > kvg_plr
  echo "[kg] s41: hypernet lr 1e-6 from the next round (weights before it: v$(cat VERSION)) $(date -u)" >> /root/w/w.log
fi
if [ ! -e /root/.s42freeze ]; then touch /root/.s42freeze
  echo 0 > /root/w/kvg_plr; echo "[kg] s42: hypernet frozen again from the next round $(date -u)" >> /root/w/w.log
fi
if [ ! -e /root/.s43pool ]; then touch /root/.s43pool
  touch /root/w/STOP_KG
  cat > /root/s43.sh <<'JOB'
while pgrep -f "^bash /root/kvg_job.sh" > /dev/null; do sleep 30; done
cd /root/w && rm -f STOP_KG && cp kvg.pt kvg_before_s43.pt
python3 -c "
import torch
a = torch.load('kvg.pt', map_location='cpu', weights_only=False); b = torch.load('kvg_before_hyper.pt', map_location='cpu', weights_only=False)
d = max((a['pooler'][k].float() - b['pooler'][k].float()).abs().max().item() for k in b['pooler'])
a['pooler'] = b['pooler']; torch.save(a, 'kvg.pt'); print('[kg] s43: hypernet set back to v24 (largest change undone', d, ')')" >> /root/w/w.log 2>&1
echo 0 > kvg_plr
START=dpo_ref.pt exec bash /root/kvg_job.sh
JOB
  setsid nohup bash -c 'bash /root/s43.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s44hyper ]; then touch /root/.s44hyper
  cd /root/w; for _t in $(seq 60); do curl -sS -o sp_new.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q "keep this run's" sp_new.py && break; sleep 120; done
  pgrep -f "sp_qwen3.py train" > /dev/null || mv sp_new.py sp_qwen3.py   # a running training keeps its copy; the next own block refreshes it anyway
  cp kvg.pt kvg_before_hyper2.pt && cat VERSION > kvg_before_hyper2.version && echo 1e-6 > kvg_plr
  echo "[kg] s44: lrs now kept across the optimizer state; hypernet lr 1e-6 from the next round (weights before: v$(cat VERSION)) $(date -u)" >> /root/w/w.log
fi
if [ ! -e /root/.s45code ]; then touch /root/.s45code
  cat > /root/s45.sh <<'JOB'
cd /root/w; grep -q "keep this run's" sp_qwen3.py && { echo "[kg] s45: sp_qwen3.py already has the lr fix"; exit 0; }
while pgrep -f "sp_qwen3.py train" > /dev/null; do sleep 20; done
grep -q "keep this run's" sp_new.py 2>/dev/null || curl -sS -o sp_new.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/tools/sp_qwen3.py?nocache=$(date +%s)"
grep -q "keep this run's" sp_new.py && mv sp_new.py sp_qwen3.py && echo "[kg] s45: sp_qwen3.py with the lr fix in place $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/s45.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s46guide ]; then touch /root/.s46guide
  touch /root/w/STOP_KG
  cat > /root/s46.sh <<'JOB'
while pgrep -f "^bash /root/kvg_job.sh" > /dev/null; do sleep 30; done
rm -f /root/w/STOP_KG
for _t in $(seq 60); do curl -sS -o /root/kvg_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvg_job.sh?nocache=$(date +%s)"; grep -q -- "gd = g.get('guide')" /root/kvg_job.sh && break; sleep 120; done
echo "[kg] s46: guide-aware gate from here (hypernet lr $(cat /root/w/kvg_plr)) $(date -u)"
cd /root/w && START=dpo_ref.pt exec bash /root/kvg_job.sh
JOB
  setsid nohup bash -c 'bash /root/s46.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s47state ]; then touch /root/.s47state
  cat > /root/s47.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token); cd /root/w; mkdir -p state_up
for f in kvg.pt kvg_opt.pt kvg_round kvg_v0 VERSION used_groups.txt kvg_plr kvg_rounds.txt; do [ -e $f ] && cp $f state_up/; done
python3 -c "
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path='state_up', path_in_repo='archive/sp3/kvg/state', repo_id='baya1116/hypernet-sp-distill')
print('[kg] s47: trainer state on the hub (round', open('state_up/kvg_round').read().strip(), ', version', open('state_up/VERSION').read().strip(), ')')"
JOB
  setsid nohup bash -c 'bash /root/s47.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s48move ]; then touch /root/.s48move
  cd /root/w && echo 5e-5 > kvg_loralr && echo 8 > kvg_accum && echo 5e-6 > kvg_plr
  echo "[kg] s48: LoRA lr 5e-5, 8 groups per update, hypernet lr 5e-6 from the next round $(date -u)" >> /root/w/w.log
fi
if [ ! -e /root/.s49revert ]; then touch /root/.s49revert
  touch /root/w/STOP_KG; pkill -f "bash /root/kvg_job.sh"; sleep 2; pkill -f "sp_qwen3.py"; sleep 5
  cat > /root/s49.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token); cd /root/w
R=baya1116/hypernet-sp-distill
mkdir -p collapse_v34 && cp kvg.pt collapse_v34/ 2>/dev/null; mv kvg_opt.pt collapse_v34/ 2>/dev/null
python3 -c "
import shutil; from huggingface_hub import hf_hub_download as d
shutil.copy(d('$R', 'archive/sp3/spKVF_latest.pt', revision='6631be0426', local_dir='rv_dl'), 'kvg.pt'); shutil.rmtree('rv_dl')" || { echo "[kg] s49: download of v32 FAILED"; exit 1; }
cp kvg.pt kvg_prev.pt; cp kvg.pt kvf.pt; echo 22 > kvg_round; echo 35 > VERSION
python3 -c "
from huggingface_hub import HfApi
fs = [f for f in HfApi().list_repo_files('$R') if f.startswith('archive/sp3/dpo_col_g/')]
open('used_groups.txt', 'a').write('\n'.join(fs) + '\n'); print(len(fs))" > s49_used.txt
for _i in 1 2 3; do python3 -c "from huggingface_hub import HfApi; a=HfApi(); a.upload_file(path_or_fileobj='kvg.pt', path_in_repo='archive/sp3/spKVF_latest.pt', repo_id='$R'); a.upload_file(path_or_fileobj='VERSION', path_in_repo='archive/sp3/kvf/VERSION', repo_id='$R')" && break; sleep 30; done
echo 2e-5 > kvg_loralr; echo 1e-6 > kvg_plr; echo 8 > kvg_accum; rm -f STOP_KG
echo "[kg] s49: back to v32's weights, published as v35; optimizer state dropped; $(cat s49_used.txt) group files marked used; LoRA lr 2e-5, hypernet lr 1e-6 $(date -u)"
START=dpo_ref.pt exec bash /root/kvg_job.sh
JOB
  setsid nohup bash -c 'bash /root/s49.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s50skip ]; then touch /root/.s50skip
  cat > /root/s50.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token); cd /root/w
for _i in $(seq 90); do
  python3 -c "
import re; from huggingface_hub import HfApi
used = set(open('used_groups.txt').read().split())
new = [f for f in HfApi().list_repo_files('baya1116/hypernet-sp-distill') if f.startswith('archive/sp3/dpo_col_g/') and re.search(r'_v3[34]\.jsonl$', f) and f not in used]
if new: open('used_groups.txt', 'a').write('\n'.join(new) + '\n'); print('[kg] s50: skipped', len(new), 'group files drawn with v33/v34')" 2>/dev/null
  sleep 120; done
JOB
  setsid nohup bash -c 'bash /root/s50.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s51alone ]; then touch /root/.s51alone
  cat > /root/s51.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token); cd /root/w
while [ $(date -u +%s) -lt $(date -u -d "2026-11-07 18:50:00" +%s) ]; do sleep 60; done
st=$(python3 -c "
import re; from huggingface_hub import HfApi
b = [int(re.search(r'/g12_b(\d+)_v', f).group(1)) for f in HfApi().list_repo_files('baya1116/hypernet-sp-distill') if f.startswith('archive/sp3/dpo_col_g/n3g12/s5/')]
print(max(b) + 2 if b else 0)" 2>/dev/null); [ -n "$st" ] && echo $st > kcg_n3g12_block
echo "POOL=grpo_pool3.jsonl TAG=n3g12 GBS=12 GUIDE=2 SHARD=5 NSHARD=6 RB=40 TB=60000" > kvg_own
echo 8 > kvg_newinf; echo 24 > kvg_newg; echo 1 > kvg_every
touch STOP_KG; pkill -f "sp_qwen3.py rollout"
echo "[kg] s51: alone from here - own rollouts on grpo_pool3 shard 5 from block $(cat kcg_n3g12_block 2>/dev/null), a round at 8 informative groups, the 60 problems every round $(date -u)"
while pgrep -f "^bash /root/kvg_job.sh" > /dev/null; do sleep 30; done
rm -f STOP_KG
for _t in $(seq 60); do curl -sS -o /root/kvg_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvg_job.sh?nocache=$(date +%s)"; grep -q -- "kvg_own" /root/kvg_job.sh && break; sleep 120; done
START=dpo_ref.pt exec bash /root/kvg_job.sh
JOB
  setsid nohup bash -c 'bash /root/s51.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s52state ]; then touch /root/.s52state
  cat > /root/s52.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token); cd /root/w; rm -rf state_up; mkdir -p state_up
for f in kvg.pt kvg_opt.pt kvg_round kvg_v0 VERSION used_groups.txt kvg_plr kvg_loralr kvg_accum kvg_newinf kvg_newg kvg_every kvg_own kvg_rounds.txt kcg_n3g12_block; do [ -e $f ] && cp $f state_up/; done
python3 -c "
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path='state_up', path_in_repo='archive/sp3/kvg/state', repo_id='baya1116/hypernet-sp-distill')
print('[kg] s52: trainer state on the hub (round', open('state_up/kvg_round').read().strip(), ', version', open('state_up/VERSION').read().strip(), ')')"
JOB
  setsid nohup bash -c 'bash /root/s52.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
# CTL-END
