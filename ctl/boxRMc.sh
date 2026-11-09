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
# serial 53: v38 read 44/60 - above the raw window's 42 (seed 1). Its state goes to the hub (archive/sp3/kvg/state/).
# serial 54: the last top-up (the user). From v38 (44/60): LoRA lr 1.5e-5 (a little below the 2e-5 that got here, the
# push toward closing is strong), hypernet 1e-6, 8 groups per update, rounds at 8 informative groups, the 60 problems
# every round; four 3060 workers on shards 0-3 of the weak-area pool, this box keeps shard 5. Read every round.
# serial 55: the closing push is building again - training answers closed 75% / 81% / 81% and closed-but-wrong 17% / 21% / 26%
# for v38 / v39 / v40 (v33, before the collapse: 87% / 30%). v39 at seed 2 read 40/60 (seed 1 44). LoRA lr 1.5e-5 -> 1e-5;
# a revert to v38 / v39 follows if v41 keeps climbing.
# serial 56: the slide - 60 problems v38 44, v39 44 / 40, v40 43 / 39, v41 40 (seed 1 / seed 2), training answers closed 81% and
# closed-but-wrong 21% from v39 on (v38: 75% / 17%). Back to v38 (hub commit c6cb73969e), published as v42; the optimizer
# state is dropped; the groups drawn with v40 / v41 are marked used; LoRA lr 1e-5, hypernet 1e-6, 8 groups per update.
# serial 57: the lr follows the accuracy (the user): kvg_job's auto_lr after every round's eval - a new best is kept, a slide
# (E <= best-4 or closed-but-wrong > 28%) goes back to the best with the lr halved, a drift (> 22% / closed > 82%) halves
# the lr, a clean round (< 18%, < 78%, E >= best-1) raises it x1.25; 2.5e-6..2e-5. The best starts as v42 (= v38, 44/60).
# serial 58: steady steps from v38 (the user): auto_lr raises by x1.1 at most and never above 1e-5, and judges the eval on the
# two-seed mean when the workers' seed-2 parts are on the hub (they evaluate every new version, 15 problems each). The
# trainer restarts after its running round to take the new kvg_job.sh.
# serial 59: s57's restart (10:22Z) removed the STOP_KG that s58 had set, so the trainer runs s57's kvg_job (lr x1.25, cap 2e-5,
# one seed) and s58's waiter is still waiting. STOP_KG again: after round 30 the s58 waiter takes over with the steady
# kvg_job; the lr file is put back to 1e-5 at that moment.
# serial 60: auto_lr reverted on one low eval (v45 40/60, no drift in its answers: closed-but-wrong 21%) and halved the lr - too
# twitchy for 60 problems (sd ~3.6), and the workers' seed 2 arrived after its 40-minute wait. Now the judgement is made
# when the next round is about to start (seed 2 has had time), one low eval alone is "watch", a revert needs W > 28% or a low
# eval with drift or two low evals in a row. The lr goes back to 1e-5; the trainer restarts after its running round.
# serial 61: alone on this box, the LLM frozen (the user). The memory loses where the reasoning is long (the raw window's
# right answers there: 16.9k chars against 5.7k elsewhere) and loses by answering wrong (MATH 4: closed-but-wrong 2.5/12
# against the raw window's 0): the pooled past misleads, and the hypernet has moved 3e-4 of itself at lr 1e-6. After the
# running round: state + best to the hub; v38 with --fifo 2048 on the 60 (does dropping the far past help?); then from v38
# with LoRA lr 0, hypernet lr 1e-5, closed-but-wrong rewarded -0.1, the 60 problems every 2nd round, the controller
# steering the hypernet lr.
# serial 62: the --fifo 2048 eval wrote nothing (0/60) - its log tail goes to the container log to see why.
# serial 63: the --fifo eval ran out of memory - something else held 22 GB (the restarting trainer). kvg_job now runs a one-off
# job from kvg_extra between rounds, with the GPU free: the --fifo 2048 eval of v38 goes there. Restart after the running round.
# serial 64: eviction by attention mass (the user: what the pooler does not attend to can go), not FIFO: the one-off eval of
# v38 uses --maxd 2048 (the 2048 distant tokens with the most pooling mass are kept, the rest dropped).
# serial 65: how many distant tokens to keep is measured, not guessed (the user): sp_qwen3 massprof reports where the
# pooler's attention mass sits over long rollouts (share on the top-k, tokens for 50/80/90/95%); the one-off job runs it on
# the v38 / v47 eval texts, takes the median tokens-for-90% at a distant past of 4096 (rounded up to 256, within 512..4096)
# as --maxd, and evaluates v38 with it.
# serial 66: the mass profile says half of the pooling mass sits on ~20 tokens of the distant past, 70% on the top 256,
# whatever its length - the pooler attends to a handful of tokens. Which ones? massprof now names them; a one-off job
# runs it (seconds) between rounds.
# serial 67: the sink test joins the pending one-off job: v38 on the 60 problems with the 24 highest-mass distant tokens
# dropped before pooling (--drop-top 24). If the mass sits on sinks, this should not hurt; if it helps, the pooler was
# looking at the wrong tokens.
# serial 68: by token type the mass goes to spaces, commas and digits (6% / 3.8% / 3.7%, over 80-140 occurrences each) and
# to single tokens such as <think> (3.4% on one position): the heaviest positions still need naming. massprof now lists the
# tokens at the 20 heaviest positions of each text and their position; queued with the drop-top eval.
# serial 69: the model's own attention decides where the pooler looks (the user's idea): --att-prior records, per generated
# token, the attention it received from later tokens while in the window (layers 8-19) and adds log(share) to the pooler's
# attention logits over the distant buffer; the first two generated tokens (<think>, newline) count as the model's own
# sinks and are zeroed. One-off job: v38 on the 60 with --att-prior 1.0 (no retraining yet - does the LLM already read
# the re-aimed memory?).
# serial 70: round 37 failed loading a .pt ("unexpected pos": a file cut short) and the loop exited, leaving the GPU idle
# with three one-off jobs queued. Report disk space and which file is broken, mend it (a bad optimizer state is dropped,
# bad weights come back from the hub), restart the loop: the queued jobs run first, then the rounds.
# serial 71: the disk is full (456K of 50G): that is what cut kvg_opt.pt short. Report what fills it, remove what the hub
# already holds (gradient sketches, downloaded copies, the state upload dirs, caches, cores, logs over 300 MB), report again.
# serial 72: 3.6G free is not enough for a round's saves. The old experiments' leftovers go (the GRPO runs before the kvg
# lineage, the DPO round-1 full model that dpo_ref.pt supersedes, the RMT final that the hub holds as spRMF_latest.pt,
# the v34/v41 analysis dirs whose conclusions are in the notes). Also: is the KV-memory distillation data still here?
# serial 73: the pivot (the user: if the pooler sits in the attention-sink valley, leave the current RL and retrain the pooler
# from the base with the model's attention put into its attention mass). Once the queued one-off jobs are done, the RL loop
# (v49 lineage, hypernet-only) is stopped and ctl/kvp_job.sh starts: pooler retraining from v38 with --att-prior 1.0, LLM
# frozen. The hub w.log copy carries [kp] lines too.
# serial 74: kvp_job started 'from 2 segments' - an older job (the ctl-para segments) left kvp_seg, kvp_hist and maybe kvp.pt
# behind, so the start may not be v38 and the plateau rule would read old numbers. Check kvp.pt against kvg_best.pt; reset
# the counters (and the weights and baselines if they differ) and restart from segment 1.
# serial 75: the baseline massprof says the prior at A=1 does not move v38's pooler - its own logits keep half the effective
# mass on 17 tokens (the model's importance itself is flat: 616 tokens for half at distant 2048). Rather than fight the collapsed
# logits, the user's plan as stated: a new pooler from the base model with the prior (FROM=base), read against the kvm curve.
# serial 76: the segment-2 run of the 60 problems never ran - ev60's file name held the label's space ('kvp_segment 2_s1.jsonl'
# split into two arguments; the score step then opened 'kvp_segment'). The job's loop body is already parsed by bash, so
# the fix needs a restart: the current checkpoint is copied aside, STOP_KP ends the loop after the running segment, and the
# fixed script starts again (PRE_EV: the copied checkpoint's 60 problems at seed 1 first, then segment 4 onwards as before).
# serial 77: segment 4 collapsed again under the prior alone (effective mass 50% on 46 tokens, 31/60; segment 2 was 349 tokens,
# 39/60; KL on kvm's curve both times) - the pooler's own logits are now bounded (--att-cap 1.0) and the run restarts from the
# segment-2 weights (kvp_seg2.pt), the 60 problems after every segment. Segments 3-4 without the cap are parked in kvp_nocap/.
# serial 78: the user - the held-out KL is teacher-forced (each of a chunk's 128 tokens sees the true text before it), so it
# never sees the student's own drift; score the chunk the student WRITES itself (sp_qwen3.py eval --gen-kl). First check the
# measure against the 60 problems on the states whose score is known: the raw window (42 / 44), v38 (44 / 41), the new pooler
# at segment 2 (39) and segment 4 without the cap (31), kvm (35) - math, competition and MMLU-Pro held-out traces, 3 chunks
# each. It runs beside the training (another copy of the script), one state at a time, only while a segment has >= 250
# traces to go, so it never meets the 60-problem run.
# serial 79: the user - score the whole text the student wrote, in one pass: each state's own 60-problem generations are read
# by the full-context teacher and by that state (its memory rebuilt chunk by chunk from its own text, as when it wrote it),
# reverse KL(student || teacher) over the full length. Seven generation files with known scores: v38 44, the new pooler at
# segment 2 39 and segment 4 31, kvm 35 and 37, the raw window 1024 alone 36 and 37. After serial 78, with the same guard.
# serial 80: the user - make the whole-text measure the loss: on-policy distillation (ctl/kvq_job.sh). Once the capped run's
# segment 3 has its 60 problems, the distillation on the teacher's texts stops and the pooler trains on its own texts
# (48 fresh problems per round, loss 0.5 KL(teacher || student) + 0.5 KL(student || teacher) on the student's path), from the
# capped segment 3 if it kept 39 or more, else from segment 2 (39); the cap stays on.
# serial 81: the status lines and the hub's wlog copy did not show the on-policy job's [kq] lines - both include them now.
# serial 82: the user - an oracle seeded with the pooler's output: at 2 positions of each of 12 held-out math / competition
# traces, the 32 slots themselves are optimised through the frozen model on the chunk's KL, starting from what the pooler
# emits (and, as the reference, from random slots). How low the KL goes says what the frozen model can read out of 32
# slots; how far the slots move says whether a pooler could reproduce that optimum. The on-policy job pauses after its
# round (STOP_KQ), the oracle runs (~40 min), the job resumes.
# serial 83: the on-policy job's 'correct' read 0/48 - the pool rows carry no gold answer. The job now scores agreement with the
# teacher's own boxed answer and rebuilds its prompt file (same order); the box's copy of the job is refreshed before s82's resume.
# serial 84: the side checks (serials 78-79) waited on the distillation job's trace counter, which the on-policy job never
# advances - they now run whenever no sp_qwen3 / oracle process holds the GPU (the on-policy job waits for them before each of
# its steps), skipping the states already measured; the whole-text check reads 30 of each state's 60 texts.
# serial 85: the user - the oracle first. The running on-policy round is killed (its generation so far is lost, ~40 min), the
# oracle runs now, the side checks after it, the on-policy job resumes after those (round 2 from kvq.pt = round 1's weights).
# serial 86: the user - generation on cheap boxes. The weights after round 1 and the prompt list go to the hub now (no GPU
# needed), three ctl/boxKQ<i>.sh workers write round 2's texts meanwhile; the on-policy job resumes with WORKERS=3 and only trains.
# serial 87: the user - within one chain of thought, how do the pooler-seeded optima of consecutive positions relate (cosine
# between neighbours, the direction of the correction, the previous chunk's optimum on the next chunk)? If they are close, the
# target is a smooth function a pooler can learn and the rest is distillation. Runs right after the first oracle (6 consecutive
# chunk starts in each of 2 math + 2 competition traces), then the side checks, then the on-policy job resumes.
# serial 88: the chunk-sized oracle overfits - its slots reach KL 0.0016 on the fitted chunk and are WORSE than the pooler on the
# next one (0.069 vs 0.059), as far from the pooler's output as a random optimum (1.17 vs 1.37, cos 0.31). So the question is
# asked properly: the slots fitted on the next 4 chunks together (--oracle-horizon 4), scored on the 2 chunks after those, 4
# consecutive positions in each of 4 chains, neighbour statistics included. After serial 87's run, before the side checks.
# serial 89: serial 85 removed serial 84's marker before running it, so the next ctl run started a second copy of the side
# checks beside the oracle run (GPU shared, the oracle slowed). The stray copy is stopped; serial 88 runs the checks later.
# (serial 88 shrunk before it ran: 2 chains, pooler-seeded only, 120 steps - about 10 minutes instead of an hour.)
# serial 92: serial 88's body was written before it was shrunk (the marker kept the old text), so the long run started at 15:24 -
# killed; the one-chain horizon oracle (~5 min) runs instead, then the side checks, then the on-policy job resumes. The hub
# listing bug (kvq/ prefix) is fixed in the job and the workers.
SER=92
if [ -f /root/.boxrmx_serial ] && [ "$(cat /root/.boxrmx_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxrmx_serial
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w
cat > /root/status.sh <<'ST'
grep -E "^\[(pr|rm|kv|kr|kp|kq|kw|kt|kd|ko|ki|kf|kg|hub)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 6 | cut -c1-600
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
if [ ! -e /root/.s53state ]; then touch /root/.s53state
  sed 's/s52:/s53:/' /root/s52.sh > /root/s53.sh
  setsid nohup bash -c 'bash /root/s53.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s54last ]; then touch /root/.s54last
  cd /root/w && echo 1.5e-5 > kvg_loralr && echo 1e-6 > kvg_plr && echo 8 > kvg_accum && echo 8 > kvg_newinf && echo 24 > kvg_newg && echo 1 > kvg_every
  echo "[kg] s54: LoRA lr 1.5e-5 from the next round; 4 workers coming $(date -u)" >> /root/w/w.log
fi
if [ ! -e /root/.s55lr ]; then touch /root/.s55lr
  echo 1e-5 > /root/w/kvg_loralr; echo "[kg] s55: LoRA lr 1e-5 from the next round $(date -u)" >> /root/w/w.log
fi
if [ ! -e /root/.s56revert ]; then touch /root/.s56revert
  touch /root/w/STOP_KG; pkill -f "bash /root/kvg_job.sh"; sleep 2; pkill -f "sp_qwen3.py"; sleep 5
  cat > /root/s56.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token); cd /root/w
R=baya1116/hypernet-sp-distill
mkdir -p slide_v41 && cp kvg.pt slide_v41/ 2>/dev/null; mv kvg_opt.pt slide_v41/ 2>/dev/null
python3 -c "
import shutil; from huggingface_hub import hf_hub_download as d
shutil.copy(d('$R', 'archive/sp3/spKVF_latest.pt', revision='c6cb73969e', local_dir='rv_dl'), 'kvg.pt'); shutil.rmtree('rv_dl')" || { echo "[kg] s56: download of v38 FAILED"; exit 1; }
cp kvg.pt kvg_prev.pt; cp kvg.pt kvf.pt; echo 29 > kvg_round; echo 42 > VERSION
python3 -c "
import re; from huggingface_hub import HfApi
used = set(open('used_groups.txt').read().split())
fs = [f for f in HfApi().list_repo_files('$R') if f.startswith('archive/sp3/dpo_col_g/') and re.search(r'_v4[01]\.jsonl$', f) and f not in used]
open('used_groups.txt', 'a').write('\n'.join(fs) + '\n'); print(len(fs))" > s56_used.txt
for _i in 1 2 3; do python3 -c "from huggingface_hub import HfApi; a=HfApi(); a.upload_file(path_or_fileobj='kvg.pt', path_in_repo='archive/sp3/spKVF_latest.pt', repo_id='$R'); a.upload_file(path_or_fileobj='VERSION', path_in_repo='archive/sp3/kvf/VERSION', repo_id='$R')" && break; sleep 30; done
echo 1e-5 > kvg_loralr; echo 1e-6 > kvg_plr; echo 8 > kvg_accum; rm -f STOP_KG
echo "[kg] s56: back to v38's weights, published as v42; optimizer state dropped; $(cat s56_used.txt) group files (v40/v41) marked used; LoRA lr 1e-5 $(date -u)"
START=dpo_ref.pt exec bash /root/kvg_job.sh
JOB
  setsid nohup bash -c 'bash /root/s56.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  cat > /root/s56b.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token); cd /root/w
for _i in $(seq 60); do
  python3 -c "
import re; from huggingface_hub import HfApi
used = set(open('used_groups.txt').read().split())
new = [f for f in HfApi().list_repo_files('baya1116/hypernet-sp-distill') if f.startswith('archive/sp3/dpo_col_g/') and re.search(r'_v4[01]\.jsonl$', f) and f not in used]
if new: open('used_groups.txt', 'a').write('\n'.join(new) + '\n'); print('[kg] s56: skipped', len(new), 'group files drawn with v40/v41')" 2>/dev/null
  sleep 120; done
JOB
  setsid nohup bash -c 'bash /root/s56b.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s57auto ]; then touch /root/.s57auto
  cd /root/w && cp kvg.pt kvg_best.pt && echo 44 > kvg_best_score && echo 42 > kvg_best_v && touch kvg_auto && touch STOP_KG
  cat > /root/s57.sh <<'JOB'
while pgrep -f "^bash /root/kvg_job.sh" > /dev/null; do sleep 30; done
cd /root/w; rm -f STOP_KG
for _t in $(seq 60); do curl -sS -o /root/kvg_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvg_job.sh?nocache=$(date +%s)"; grep -q -- "auto_lr" /root/kvg_job.sh && break; sleep 120; done
echo "[kg] s57: the lr follows the accuracy from here (best v42 = v38, 44/60) $(date -u)"
START=dpo_ref.pt exec bash /root/kvg_job.sh
JOB
  setsid nohup bash -c 'bash /root/s57.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s58steady ]; then touch /root/.s58steady
  touch /root/w/STOP_KG
  cat > /root/s58.sh <<'JOB'
while pgrep -f "^bash /root/kvg_job.sh" > /dev/null; do sleep 30; done
cd /root/w; rm -f STOP_KG; [ -s kvg_best.pt ] || { cp kvg.pt kvg_best.pt; echo 44 > kvg_best_score; echo 42 > kvg_best_v; }; touch kvg_auto
for _t in $(seq 60); do curl -sS -o /root/kvg_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvg_job.sh?nocache=$(date +%s)"; grep -q -- "two-seed mean" /root/kvg_job.sh && break; sleep 120; done
echo "[kg] s58: steady auto lr (x1.1, cap 1e-5, two-seed eval) from here $(date -u)"
START=dpo_ref.pt exec bash /root/kvg_job.sh
JOB
  setsid nohup bash -c 'bash /root/s58.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s59stop ]; then touch /root/.s59stop
  touch /root/w/STOP_KG
  cat > /root/s59.sh <<'JOB'
while pgrep -f "^bash /root/kvg_job.sh" > /dev/null; do sleep 15; done
echo 1e-5 > /root/w/kvg_loralr; echo "[kg] s59: trainer stopped after its round; lr 1e-5; the steady kvg_job (s58) takes over $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/s59.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s60soft ]; then touch /root/.s60soft
  touch /root/w/STOP_KG
  cat > /root/s60.sh <<'JOB'
while pgrep -f "^bash /root/kvg_job.sh" > /dev/null; do sleep 30; done
cd /root/w; rm -f STOP_KG kvg_pending_auto; echo 1e-5 > kvg_loralr
for _t in $(seq 60); do curl -sS -o /root/kvg_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvg_job.sh?nocache=$(date +%s)"; grep -q -- "kvg_pending_auto" /root/kvg_job.sh && break; sleep 120; done
echo "[kg] s60: softer auto lr (judged at the next round's start, one low eval = watch), lr 1e-5 $(date -u)"
START=dpo_ref.pt exec bash /root/kvg_job.sh
JOB
  setsid nohup bash -c 'bash /root/s60.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s61hyper ]; then touch /root/.s61hyper
  touch /root/w/STOP_KG
  cat > /root/s61.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token); cd /root/w; R=baya1116/hypernet-sp-distill
while pgrep -f "^bash /root/kvg_job.sh" > /dev/null; do sleep 30; done
rm -f STOP_KG kvg_pending_auto; rm -rf state_up; mkdir -p state_up
for f in kvg.pt kvg_opt.pt kvg_round kvg_v0 VERSION used_groups.txt kvg_plr kvg_loralr kvg_accum kvg_newinf kvg_newg kvg_every kvg_own kvg_rounds.txt kvg_best.pt kvg_best_score kvg_best_v kvg_auto.txt kvg_skip_versions kvg_last_E; do [ -e $f ] && cp $f state_up/; done
python3 -c "
from huggingface_hub import HfApi; a = HfApi()
a.upload_folder(folder_path='state_up', path_in_repo='archive/sp3/kvg/state', repo_id='$R')
a.upload_file(path_or_fileobj='kvg_best.pt', path_in_repo='archive/sp3/final/spKVF_best_v38.pt', repo_id='$R')
print('[kg] s61: state (round', open('state_up/kvg_round').read().strip(), ', v' + open('state_up/VERSION').read().strip() + ') and the best (v38) on the hub')"
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
o=evalx_v38_fifo2048_s1.jsonl; rm -f $o
python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $M --fifo 2048 --ckpt kvg_best.pt --max-len 12288 --roll-batch 48 --grpo-tok-budget 60000 --seed 1 --roll-out $o --log sp_kvg.log > run_fifo.log 2>&1
python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='$o', path_in_repo='archive/sp3/kvg/$o', repo_id='$R')"
echo "[kg] eval v38 fifo2048 seed 1: $(python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open('$o')]; print(f'{sum(correct(r) for r in rs)}/60 correct, closed {sum(\"</think>\" in r[\"text\"] for r in rs)}')") (v38 plain: 44 / 41) $(date -u)"
cp kvg_best.pt kvg.pt; cp kvg.pt kvf.pt; rm -f kvg_opt.pt; cp kvg.pt kvg_prev.pt
rd=$(cat kvg_round); v=$(cat VERSION); bv=$(cat kvg_best_v); seq $((bv + 1)) $v | tr '\n' ' ' >> kvg_skip_versions; echo >> kvg_skip_versions
rd=$((rd + 1)); echo $rd > kvg_round; v=$(( $(cat kvg_v0) + rd )); echo $v > VERSION
for _i in 1 2 3; do python3 -c "from huggingface_hub import HfApi; a=HfApi(); a.upload_file(path_or_fileobj='kvg.pt', path_in_repo='archive/sp3/spKVF_latest.pt', repo_id='$R'); a.upload_file(path_or_fileobj='VERSION', path_in_repo='archive/sp3/kvf/VERSION', repo_id='$R')" && break; sleep 30; done
echo 0 > kvg_loralr; echo 1e-5 > kvg_plr; echo -0.1 > kvg_wrong; echo 2 > kvg_every; echo 8 > kvg_newinf; echo 24 > kvg_newg; echo $v > kvg_best_v
for _t in $(seq 60); do curl -sS -o /root/kvg_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvg_job.sh?nocache=$(date +%s)"; grep -q -- "steers the hypernet" /root/kvg_job.sh && break; sleep 120; done
echo "[kg] s61: from v38 as v$v - LoRA frozen, hypernet lr 1e-5, closed-but-wrong -0.1, eval every 2nd round $(date -u)"
START=dpo_ref.pt exec bash /root/kvg_job.sh
JOB
  setsid nohup bash -c 'bash /root/s61.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s62log ]; then touch /root/.s62log
  { echo "[kg] s62: run_fifo.log tail:"; grep -E "Error|Traceback|error" /root/w/run_fifo.log | head -3 | cut -c1-300; tail -n 4 /root/w/run_fifo.log | cut -c1-300; } >> /root/w/w.log 2>&1
fi
if [ ! -e /root/.s63fifo ]; then touch /root/.s63fifo
  cat > /root/w/kvg_extra <<'EX'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token); cd /root/w; R=baya1116/hypernet-sp-distill
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
o=evalx_v38_fifo2048_s1.jsonl; rm -f $o
python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $M --fifo 2048 --ckpt kvg_best.pt --max-len 12288 --roll-batch 48 --grpo-tok-budget 60000 --seed 1 --roll-out $o --log sp_kvg.log > run_fifo.log 2>&1
grep -E "Error|Traceback" run_fifo.log | tail -1 | cut -c1-200 | sed 's/^/[kg] fifo eval: /'
python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='$o', path_in_repo='archive/sp3/kvg/$o', repo_id='$R')"
echo "[kg] eval v38 fifo2048 seed 1: $(python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open('$o')]; print(f'{sum(correct(r) for r in rs)}/{len(rs)} correct, closed {sum(\"</think>\" in r[\"text\"] for r in rs)}')") (v38 plain: 44 / 41) $(date -u)"
EX
  touch /root/w/STOP_KG
  cat > /root/s63.sh <<'JOB'
while pgrep -f "^bash /root/kvg_job.sh" > /dev/null; do sleep 30; done
cd /root/w; rm -f STOP_KG kvg_pending_auto
for _t in $(seq 60); do curl -sS -o /root/kvg_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvg_job.sh?nocache=$(date +%s)"; grep -q -- "kvg_extra" /root/kvg_job.sh && break; sleep 120; done
echo "[kg] s63: restarting with the one-off job hook (the fifo eval first) $(date -u)"
START=dpo_ref.pt exec bash /root/kvg_job.sh
JOB
  setsid nohup bash -c 'bash /root/s63.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s64mass ]; then touch /root/.s64mass
  [ -e /root/w/kvg_extra.running ] || sed -i 's/--fifo 2048/--maxd 2048/; s/fifo2048/maxd2048/g; s/fifo eval/maxd eval/' /root/w/kvg_extra 2>/dev/null
  echo "[kg] s64: the one-off eval of v38 now with --maxd 2048 (attention-mass eviction) $(grep -c maxd2048 /root/w/kvg_extra 2>/dev/null)" >> /root/w/w.log
fi
if [ ! -e /root/.s65mass ]; then touch /root/.s65mass
  [ -e /root/w/kvg_extra.running ] || cat > /root/w/kvg_extra <<'EX'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token); cd /root/w; R=baya1116/hypernet-sp-distill
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
curl -sS -o sp_new.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q '"massprof"' sp_new.py && mv sp_new.py sp_qwen3.py
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
cat kvg_v38_s1.jsonl kvg_v47_s1.jsonl > mp_in.jsonl
python3 sp_qwen3.py massprof --traces mp_in.jsonl $M --ckpt kvg_best.pt --max-len 12288 --log sp_kvg.log > run_mp.log 2>&1
grep -E "massprof|Error|Traceback" run_mp.log | tail -5 | sed 's/^\[sp\]/[kg]/'
k=$(grep "massprof distant 4096" run_mp.log | grep -oP '90%: \K[0-9]+' | head -1); [ -n "$k" ] || k=$(grep "massprof distant 2048" run_mp.log | grep -oP '90%: \K[0-9]+' | head -1); [ -n "$k" ] || k=2048
k=$(( (k + 255) / 256 * 256 )); [ $k -lt 512 ] && k=512; [ $k -gt 4096 ] && k=4096
echo "[kg] eviction cap from the mass profile: --maxd $k"
o=evalx_v38_maxd${k}_s1.jsonl; rm -f $o
python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $M --ckpt kvg_best.pt --max-len 12288 --roll-batch 48 --grpo-tok-budget 60000 --seed 1 --roll-out $o --log sp_kvg.log --maxd $k > run_maxd.log 2>&1
grep -E "Error|Traceback" run_maxd.log | tail -1 | cut -c1-200 | sed 's/^/[kg] maxd eval: /'
python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='$o', path_in_repo='archive/sp3/kvg/$o', repo_id='$R')"
echo "[kg] eval v38 maxd$k seed 1: $(python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open('$o')]; print(f'{sum(correct(r) for r in rs)}/{len(rs)} correct, closed {sum(\"</think>\" in r[\"text\"] for r in rs)}')") (v38 plain: 44 / 41) $(date -u)"
EX
  echo "[kg] s65: the one-off job now measures the mass profile first and picks --maxd from it $(date -u)" >> /root/w/w.log
fi
if [ ! -e /root/.s66tok ]; then touch /root/.s66tok
  [ -e /root/w/kvg_extra.running ] || cat > /root/w/kvg_extra <<'EX'
cd /root/w; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
curl -sS -o sp_new.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q 'top tokens by mass' sp_new.py && mv sp_new.py sp_qwen3.py
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
python3 sp_qwen3.py massprof --traces mp_in.jsonl $M --ckpt kvg_best.pt --max-len 12288 --log sp_kvg.log > run_mp2.log 2>&1
grep -E "top tokens|Error|Traceback" run_mp2.log | tail -2 | sed 's/^\[sp\]/[kg]/' | cut -c1-900
EX
  echo "[kg] s66: one-off job - which tokens carry the pooling mass $(date -u)" >> /root/w/w.log
fi
if [ ! -e /root/.s67drop ]; then touch /root/.s67drop
  [ -e /root/w/kvg_extra.running ] && F=/root/w/kvg_extra_next || F=/root/w/kvg_extra
  cat >> $F <<'EX'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token); cd /root/w; R=baya1116/hypernet-sp-distill
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
curl -sS -o sp_new.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q '"--drop-top"' sp_new.py && mv sp_new.py sp_qwen3.py
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
o=evalx_v38_droptop24_s1.jsonl; rm -f $o
python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $M --drop-top 24 --ckpt kvg_best.pt --max-len 12288 --roll-batch 48 --grpo-tok-budget 60000 --seed 1 --roll-out $o --log sp_kvg.log > run_drop.log 2>&1
grep -E "Error|Traceback" run_drop.log | tail -1 | cut -c1-200 | sed 's/^/[kg] drop-top eval: /'
python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='$o', path_in_repo='archive/sp3/kvg/$o', repo_id='$R')"
echo "[kg] eval v38 drop-top24 seed 1: $(python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open('$o')]; print(f'{sum(correct(r) for r in rs)}/{len(rs)} correct, closed {sum(\"</think>\" in r[\"text\"] for r in rs)}')") (v38 plain: 44 / 41, maxd2816: 44) $(date -u)"
EX
  echo "[kg] s67: the --drop-top 24 eval of v38 queued in $F $(date -u)" >> /root/w/w.log
fi
if [ ! -e /root/.s68top ]; then touch /root/.s68top
  [ -e /root/w/kvg_extra.running ] && F=/root/w/kvg_extra_next || F=/root/w/kvg_extra
  cat >> $F <<'EX'
cd /root/w; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
curl -sS -o sp_new.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q 'heaviest positions' sp_new.py && mv sp_new.py sp_qwen3.py
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
python3 sp_qwen3.py massprof --traces mp_in.jsonl $M --ckpt kvg_best.pt --max-len 12288 --log sp_kvg.log > run_mp3.log 2>&1
grep -E "by position|heaviest|Error|Traceback" run_mp3.log | tail -3 | sed 's/^\[sp\]/[kg]/' | cut -c1-900
EX
  echo "[kg] s68: massprof (heaviest positions) queued in $F $(date -u)" >> /root/w/w.log
fi
if [ ! -e /root/.s69prior ]; then touch /root/.s69prior
  [ -e /root/w/kvg_extra.running ] && F=/root/w/kvg_extra_next || F=/root/w/kvg_extra
  cat >> $F <<'EX'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token); cd /root/w; R=baya1116/hypernet-sp-distill
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
curl -sS -o sp_new.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q '"--att-prior"' sp_new.py && mv sp_new.py sp_qwen3.py
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
o=evalx_v38_attprior1_s1.jsonl; rm -f $o
python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $M --att-prior 1.0 --ckpt kvg_best.pt --max-len 12288 --roll-batch 48 --grpo-tok-budget 60000 --seed 1 --roll-out $o --log sp_kvg.log > run_ap.log 2>&1
grep -E "att-prior|Error|Traceback" run_ap.log | tail -2 | cut -c1-300 | sed 's/^\[sp\]/[kg]/'
python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='$o', path_in_repo='archive/sp3/kvg/$o', repo_id='$R')"
echo "[kg] eval v38 att-prior1 seed 1: $(python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open('$o')]; print(f'{sum(correct(r) for r in rs)}/{len(rs)} correct, closed {sum(\"</think>\" in r[\"text\"] for r in rs)}')") (v38 plain: 44 / 41) $(date -u)"
EX
  echo "[kg] s69: the --att-prior 1.0 eval of v38 queued in $F $(date -u)" >> /root/w/w.log
fi
if [ ! -e /root/.s70mend ]; then touch /root/.s70mend
  cat > /root/s70.sh <<'JOB'
cd /root/w; export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token)
echo "[kg] s70: disk $(df -h /root | tail -1 | awk '{print $4" free of "$2}') | traceback: $(grep -B3 "unexpected pos" run_kg_tr.log | grep -oE 'File "[^"]+", line [0-9]+|torch.load[^)]*' | tail -2 | tr '\n' ' ')"
python3 - <<'PYM' | sed 's/^/[kg] s70: /'
import os, torch, shutil
from huggingface_hub import hf_hub_download as d
R = 'baya1116/hypernet-sp-distill'
for f in ['kvg.pt', 'kvg_prev.pt', 'kvg_opt.pt', 'kvg_best.pt', 'dpo_ref.pt']:
    if not os.path.exists(f): print(f, 'missing'); continue
    try: torch.load(f, map_location='cpu', weights_only=False); print(f, os.path.getsize(f) >> 20, 'MB ok')
    except Exception as e:
        print(f, os.path.getsize(f) >> 20, 'MB BROKEN:', str(e)[:80])
        if f == 'kvg_opt.pt': os.remove(f); print('kvg_opt.pt dropped (Adam restarts for the hypernet)')
        if f in ('kvg.pt', 'kvg_prev.pt'): shutil.copy(d(R, 'archive/sp3/spKVF_latest.pt', local_dir='mend_dl'), f); print(f, 'restored from the hub')
        if f == 'kvg_best.pt': shutil.copy(d(R, 'archive/sp3/final/spKVF_best_v38.pt', local_dir='mend_dl'), f); print(f, 'restored from the hub')
shutil.rmtree('mend_dl', ignore_errors=True)
PYM
rm -f STOP_KG
curl -sS -o /root/kvg_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvg_job.sh?nocache=$(date +%s)"
echo "[kg] s70: loop restarted, queued one-off jobs first: $(grep -c "^o=" kvg_extra 2>/dev/null) $(date -u)"
START=dpo_ref.pt exec bash /root/kvg_job.sh
JOB
  pgrep -f "^bash /root/kvg_job.sh" > /dev/null || setsid nohup bash -c 'bash /root/s70.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s71disk ]; then touch /root/.s71disk
  cd /root/w
  echo "[kg] s71: before: $(df -h /root | tail -1 | awk '{print $4" free"}') | $(du -xsh /root/w /root/.cache /tmp /venv 2>/dev/null | awk '{printf "%s %s, ", $2, $1}')"
  echo "[kg] s71: biggest in /root/w: $(du -xsk /root/w/* 2>/dev/null | sort -rn | head -12 | awk '{printf "%s %.1fG, ", $2, $1/1048576}')"
  rm -f kvg_gsk_r*.pt evx_*.pt kvg_next.pt core core.* /root/w/mend_dl -r; rm -rf state_up mend_dl evx_dl /root/.cache/pip
  rm -rf /root/.cache/huggingface/hub/models--baya1116--* /root/.cache/huggingface/hub/.locks/models--baya1116--* 2>/dev/null
  for f in $(find /root/w /root -maxdepth 2 -name "*.log" -size +300M 2>/dev/null); do tail -c 50000000 $f > $f.tmp && mv $f.tmp $f; done
  echo "[kg] s71: after: $(df -h /root | tail -1 | awk '{print $4" free"}') $(date -u)"
fi
if [ ! -e /root/.s72disk ]; then touch /root/.s72disk
  cd /root/w; rm -rf grpo_run1 grpo_run2_pre grpo_run2_v3start kvf_round1.pt kvf_prev.pt rmf_next.pt rmf.pt slide_v41 collapse_v34 state_up
  echo "[kg] s72: $(df -h /root | tail -1 | awk '{print $4" free"}') | distillation data: $(ls -la rm_pool.jsonl rm_held.jsonl kvm.pt 2>&1 | awk '{print $NF, $5}' | tr '\n' ' ') | one-offs left: $(grep -c "^o=" kvg_extra 2>/dev/null) running: $(ls kvg_extra.running 2>/dev/null) $(date -u)"
fi
if [ ! -e /root/.s73pivot ]; then touch /root/.s73pivot
  sed -i 's/\^\\\[k\[gc\]\\\]/^\\[k[gcp]\\]/' /root/wlog.sh; pkill -f "bash /root/wlog.sh"
  setsid nohup bash -c 'bash /root/wlog.sh > /dev/null 2>&1' > /dev/null 2>&1 < /dev/null &
  cat > /root/s73.sh <<'JOB'
cd /root/w; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
while [ -e kvg_extra.running ] || [ -s kvg_extra ]; do sleep 60; done   # the one-off jobs (drop-top eval, massprof, att-prior eval) finish first
touch STOP_KG
for J in $(pgrep -f "^bash /root/kvg_job.sh") $(pgrep -f "bash /root/s70.sh") $(pgrep -f "bash kvc_job.sh"); do pkill -P $J; kill $J; done; sleep 3
pkill -f "sp_qwen3.py rollout"; pkill -f "sp_qwen3.py train"; sleep 10
echo "[kp] s73: the RL loop stopped (round $(cat kvg_round), v$(cat VERSION)) - on to the pooler retraining with the model's attention as the prior $(date -u)"
for _t in $(seq 60); do curl -sS -o /root/kvp_job.sh "$RAW/ctl/kvp_job.sh?nocache=$(date +%s)"; grep -q "mprof" /root/kvp_job.sh && break; sleep 120; done
exec bash /root/kvp_job.sh
JOB
  pgrep -f "bash /root/s73.sh" > /dev/null || setsid nohup bash -c 'bash /root/s73.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[kp] s73: waiting for the one-off jobs, then the pivot $(date -u)" >> /root/w/w.log
fi
if [ ! -e /root/.s74reset ]; then touch /root/.s74reset
  cat > /root/s74.sh <<'JOB'
cd /root/w; touch STOP_KP
pkill -f "bash /root/kvp_job.sh"; sleep 2; pkill -f "sp_qwen3.py train"; pkill -f "sp_qwen3.py eval"; pkill -f "sp_qwen3.py massprof"; pkill -f "sp_qwen3.py rollout"; sleep 10
a=$(md5sum < kvp.pt); b=$(md5sum < kvg_best.pt)
if [ "$a" = "$b" ]; then echo "[kp] s74: kvp.pt is v38 (same md5) - counters reset (old kvp_seg $(cat kvp_seg), hist $(tr '\n' ' ' < kvp_hist 2>/dev/null)), baselines kept, restart from segment 1 $(date -u)"
else echo "[kp] s74: kvp.pt was NOT v38 (an older job's) - weights, counters and baselines reset, restart from segment 1 $(date -u)"; rm -f kvp.pt kvp_base_done; fi
rm -f kvp_seg kvp_hist kvp_next.pt STOP_KP
exec bash /root/kvp_job.sh
JOB
  setsid nohup bash -c 'bash /root/s74.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s75base ]; then touch /root/.s75base
  cat > /root/s75.sh <<'JOB'
cd /root/w; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i; touch STOP_KP
pkill -f "bash /root/kvp_job.sh"; pkill -f "bash /root/s74.sh"; sleep 2; pkill -f "sp_qwen3.py train"; pkill -f "sp_qwen3.py eval"; pkill -f "sp_qwen3.py massprof"; pkill -f "sp_qwen3.py rollout"; sleep 10
mkdir -p kvp_v38start; mv kvp.pt kvp_seg kvp_hist kvp_base_done kvp_next.pt kvp_v38start/ 2>/dev/null; rm -f STOP_KP
for _t in $(seq 60); do curl -sS -o /root/kvp_job.sh "$RAW/ctl/kvp_job.sh?nocache=$(date +%s)"; grep -q "FROM=" /root/kvp_job.sh && break; sleep 120; done
echo "[kp] s75: a new pooler from the base model with the prior (kvm's recipe), v38-start files parked in kvp_v38start/ $(date -u)"
FROM=base exec bash /root/kvp_job.sh
JOB
  setsid nohup bash -c 'bash /root/s75.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s76evfix ]; then touch /root/.s76evfix
  cat > /root/s76.sh <<'JOB'
cd /root/w; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
n=$(cat kvp_seg 2>/dev/null || echo 0); cp kvp.pt kvp_seg$n.pt; touch STOP_KP
echo "[kp] s76: 60-problem run fixed (file-name bug); kvp.pt = segment $n copied to kvp_seg$n.pt, the loop stops after the running segment and restarts with the fix $(date -u)"
while pgrep -f "bash /root/kvp_job.sh" > /dev/null; do sleep 60; done
for _t in $(seq 60); do curl -sS -o /root/kvp_job.sh "$RAW/ctl/kvp_job.sh?nocache=$(date +%s)"; grep -q "PRE_EV" /root/kvp_job.sh && break; sleep 120; done
rm -f STOP_KP; echo "[kp] s76: restart from $(cat kvp_seg) segments, first the 60 problems of segment $n $(date -u)"
PRE_EV=kvp_seg$n.pt:segment$n FROM=base exec bash /root/kvp_job.sh
JOB
  setsid nohup bash -c 'bash /root/s76.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s77cap ]; then touch /root/.s77cap
  cat > /root/s77.sh <<'JOB'
cd /root/w; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i; touch STOP_KP
pkill -f "bash /root/kvp_job.sh"; pkill -f "bash /root/s76.sh"; sleep 2; pkill -f "sp_qwen3.py train"; pkill -f "sp_qwen3.py eval"; pkill -f "sp_qwen3.py massprof"; pkill -f "sp_qwen3.py rollout"; sleep 10
mkdir -p kvp_nocap; mv kvp.pt kvp_nocap/kvp_seg$(cat kvp_seg).pt; cp kvp_hist kvp_nocap/kvp_hist; cp kvp_seg2.pt kvp.pt; echo 2 > kvp_seg; head -n 2 kvp_nocap/kvp_hist > kvp_hist; rm -f kvp_next.pt STOP_KP
for _t in $(seq 60); do curl -sS -o /root/kvp_job.sh "$RAW/ctl/kvp_job.sh?nocache=$(date +%s)"; grep -q "PRE_MP" /root/kvp_job.sh && break; sleep 120; done
echo "[kp] s77: pooler logits capped at 1.0 under the prior; restart from the segment-2 weights (segments 3-4 without the cap parked in kvp_nocap/), the 60 problems after every segment $(date -u)"
CAP=1.0 EVERY=1 PRE_MP="kvp.pt:segment 2 cap1" FROM=base exec bash /root/kvp_job.sh
JOB
  setsid nohup bash -c 'bash /root/s77.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s78gk ]; then touch /root/.s78gk
  cat > /root/s78.sh <<'JOB'
cd /root/w; export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i; R=baya1116/hypernet-sp-distill
for _t in $(seq 60); do curl -sS -o sp_gk.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q "def gen_chunk" sp_gk.py && break; sleep 120; done
cat held_math.jsonl held_comp.jsonl held_mc.jsonl > gk_held.jsonl
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"; AP="--att-prior 1.0 --att-layers 8-19"
echo "[kp] s78: the student writes its chunk (--gen-kl 3, 128 tokens) on $(wc -l < gk_held.jsonl) held-out traces, beside the training $(date -u)"
ok() { pgrep -f "sp_qwen3.py train" > /dev/null || return 1; n=$(grep -oP "trace \K[0-9]+(?=/600)" run_kp.log 2>/dev/null | tail -1); [ "${n:-0}" -le 350 ]; }
for spec in "raw window (60 problems 42 / 44)|--no-sp" "kvm segment 8 (35)|--ckpt kvm.pt" "new pooler segment 2 (39)|--ckpt kvp_seg2.pt $AP" \
            "new pooler segment 4, no cap (31)|--ckpt kvp_nocap/kvp_seg4.pt $AP" "v38 (44 / 41)|--ckpt kvg_best.pt"; do
  lb=${spec%%|*}; fl=${spec#*|}
  until ok; do sleep 60; done
  python3 sp_gk.py eval --traces gk_held.jsonl --max-len 6144 $M $fl --gen-kl 3 --gen-len 128 --seed 7 --log sp_gk.log > run_gk.log 2>&1
  echo "[kp] gk $lb: $(grep -E "\[sp\] eval|Error" run_gk.log | tail -1 | sed 's/^\[sp\] eval [^:]*: //' | cut -c1-420) $(date -u)"
done
grep "\[kp\] gk" /root/w/w.log > gk.txt; python3 -c "
from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='gk.txt', path_in_repo='archive/sp3/kvp/gk.txt', repo_id='$R')" > /dev/null 2>&1
echo "[kp] s78 done $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/s78.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s79own ]; then touch /root/.s79own
  cat > /root/s79.sh <<'JOB'
cd /root/w; export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
while pgrep -f "bash /root/s78.sh" > /dev/null; do sleep 60; done
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"; AP="--att-prior 1.0 --att-layers 8-19"
mkdir -p own; for f in kvg/kvg_v38_s1 kvp/kvp_segment2_s1 kvp/kvp_segment4_s1 kvm/kvm_mem_s1 kvm/kvm_mem_s2 dg/gen_raw1024 dg/gen_raw1024_s3; do
  [ -s own/${f#*/}.jsonl ] || curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o own/${f#*/}.jsonl "$H/archive/sp3/$f.jsonl"; done
echo "[kp] s79: each state reads its own 60-problem texts (full length), teacher vs student $(date -u)"
ok() { pgrep -f "sp_qwen3.py train" > /dev/null || return 1; n=$(grep -oP "trace \K[0-9]+(?=/600)" run_kp.log 2>/dev/null | tail -1); [ "${n:-0}" -le 350 ]; }
for spec in "v38 (44)|kvg_v38_s1|--ckpt kvg_best.pt" "new pooler segment 2 (39)|kvp_segment2_s1|--ckpt kvp_seg2.pt $AP" \
            "new pooler segment 4, no cap (31)|kvp_segment4_s1|--ckpt kvp_nocap/kvp_seg4.pt $AP" "kvm seed 1 (35)|kvm_mem_s1|--ckpt kvm.pt" \
            "kvm seed 2 (37)|kvm_mem_s2|--ckpt kvm.pt" "raw window (36)|gen_raw1024|--no-sp" "raw window seed 3 (37)|gen_raw1024_s3|--no-sp"; do
  lb=${spec%%|*}; r=${spec#*|}; fn=${r%%|*}; fl=${r#*|}
  until ok; do sleep 60; done
  python3 sp_gk.py eval --traces own/$fn.jsonl --max-len 12288 $M $fl --log sp_gk.log > run_own.log 2>&1
  echo "[kp] own $lb: $(grep -E "\[sp\] eval|Error" run_own.log | tail -1 | sed 's/^\[sp\] eval [^:]*: //' | cut -c1-300) $(date -u)"
done
grep "\[kp\] \(gk\|own\)" /root/w/w.log > gk.txt; python3 -c "
from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='gk.txt', path_in_repo='archive/sp3/kvp/gk.txt', repo_id='$R')" > /dev/null 2>&1
echo "[kp] s79 done $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/s79.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s80kq ]; then touch /root/.s80kq
  cat > /root/s80.sh <<'JOB'
cd /root/w; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
for _t in $(seq 240); do n=$(grep -oP "^\[kp\] segment 3 seed 1: \K[0-9]+(?=/60)" w.log | tail -1); [ -n "$n" ] && break; sleep 60; done
st=kvp_seg2.pt; [ -n "$n" ] && [ "$n" -ge 39 ] && { cp kvp.pt kvp_cap_seg3.pt; st=kvp_cap_seg3.pt; }
touch STOP_KP; pkill -f "bash /root/kvp_job.sh"; pkill -f "bash /root/s77.sh"; sleep 2; pkill -f "sp_qwen3.py train"; pkill -f "sp_qwen3.py eval"; pkill -f "sp_qwen3.py massprof"; pkill -f "sp_qwen3.py rollout"; sleep 10
for _t in $(seq 60); do curl -sS -o /root/kvq_job.sh "$RAW/ctl/kvq_job.sh?nocache=$(date +%s)"; grep -q "kvq_round" /root/kvq_job.sh && break; sleep 120; done
echo "[kq] s80: the capped segment 3 scored ${n:-?}/60 - on-policy distillation from $st $(date -u)"
START=$st CAP=1.0 W=0.5 exec bash /root/kvq_job.sh
JOB
  setsid nohup bash -c 'bash /root/s80.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s81kq ]; then touch /root/.s81kq
  sed -i 's/\^\\\[k\[gcp\]\\\]/^\\[k[gcpq]\\]/' /root/wlog.sh; pkill -f "bash /root/wlog.sh"
  setsid nohup bash -c 'bash /root/wlog.sh > /dev/null 2>&1' > /dev/null 2>&1 < /dev/null &
  grep -E "^\[kq\]" /root/w/w.log | tail -n 5 | sed 's/^/[kp] s81 sees: /'
fi
if [ ! -e /root/.s82oracle ]; then touch /root/.s82oracle
  cat > /root/s82.sh <<'JOB'
cd /root/w; export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i; R=baya1116/hypernet-sp-distill
touch STOP_KQ; echo "[kp] s82: oracle seeded with the pooler's output - the on-policy job pauses after its round $(date -u)"
while pgrep -f "bash /root/kvq_job.sh" > /dev/null; do sleep 60; done
while pgrep -f "sp_gk.py" > /dev/null; do sleep 60; done
for _t in $(seq 60); do curl -sS -o sp_or.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q "def oracle_trace" sp_or.py && break; sleep 120; done
head -n 6 held_math.jsonl > or_in.jsonl; head -n 6 held_comp.jsonl >> or_in.jsonl
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"; AP="--att-prior 1.0 --att-layers 8-19 --att-cap 1.0"
python3 sp_or.py oracle --traces or_in.jsonl --max-len 6144 $M $AP --ckpt kvp_seg2.pt --oracle-steps 200 --oracle-lr 1e-3 --log sp_or.log > run_or.log 2>&1
grep -E "^\[sp\] oracle|Error|Traceback" run_or.log | cut -c1-600 | sed 's/^\[sp\]/[kp]/'
python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='sp_or.log', path_in_repo='archive/sp3/kvp/oracle.log', repo_id='$R')" > /dev/null 2>&1
rm -f STOP_KQ; echo "[kp] s82 done - the on-policy job resumes $(date -u)"
exec bash /root/kvq_job.sh
JOB
  setsid nohup bash -c 'bash /root/s82.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s83gold ]; then touch /root/.s83gold
  cat > /root/s83.sh <<'JOB'
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
for _t in $(seq 60); do curl -sS -o /root/kvq_new.sh "$RAW/ctl/kvq_job.sh?nocache=$(date +%s)"; grep -q "teacher's answer" /root/kvq_new.sh && { mv /root/kvq_new.sh /root/kvq_job.sh; break; }; sleep 120; done
echo "[kp] s83: the on-policy job scores agreement with the teacher's answer from its next start (the 0/48 was the missing gold) $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/s83.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s84idle ]; then touch /root/.s84idle
  pkill -f "bash /root/s78.sh"; pkill -f "bash /root/s79.sh"
  cat > /root/s84.sh <<'JOB'
cd /root/w; export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i; R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
for _t in $(seq 60); do curl -sS -o /root/kvq_new.sh "$RAW/ctl/kvq_job.sh?nocache=$(date +%s)"; grep -q "quiet before training" /root/kvq_new.sh && { mv /root/kvq_new.sh /root/kvq_job.sh; break; }; sleep 120; done
[ -s sp_gk.py ] && grep -q "def gen_chunk" sp_gk.py || curl -sS -o sp_gk.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
[ -s gk_held.jsonl ] || cat held_math.jsonl held_comp.jsonl held_mc.jsonl > gk_held.jsonl
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"; AP="--att-prior 1.0 --att-layers 8-19"
idle() { ! pgrep -f "sp_qwen3.py" > /dev/null && ! pgrep -f "sp_or.py" > /dev/null; }
echo "[kp] s84: side checks run when the GPU is idle $(date -u)"
for spec in "raw window (60 problems 42 / 44)|--no-sp" "kvm segment 8 (35)|--ckpt kvm.pt" "new pooler segment 2 (39)|--ckpt kvp_seg2.pt $AP" \
            "new pooler segment 4, no cap (31)|--ckpt kvp_nocap/kvp_seg4.pt $AP" "v38 (44 / 41)|--ckpt kvg_best.pt"; do
  lb=${spec%%|*}; fl=${spec#*|}; grep -qF "[kp] gk $lb:" w.log && continue
  until idle; do sleep 45; done
  python3 sp_gk.py eval --traces gk_held.jsonl --max-len 6144 $M $fl --gen-kl 3 --gen-len 128 --seed 7 --log sp_gk.log > run_gk.log 2>&1
  echo "[kp] gk $lb: $(grep -E "\[sp\] eval|Error" run_gk.log | tail -1 | sed 's/^\[sp\] eval [^:]*: //' | cut -c1-420) $(date -u)"
done
mkdir -p own; for f in kvg/kvg_v38_s1 kvp/kvp_segment2_s1 kvp/kvp_segment4_s1 kvm/kvm_mem_s1 kvm/kvm_mem_s2 dg/gen_raw1024 dg/gen_raw1024_s3; do
  [ -s own/${f#*/}.jsonl ] || curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o own/${f#*/}.jsonl "$H/archive/sp3/$f.jsonl"
  [ -s own/${f#*/}_30.jsonl ] || awk 'NR % 2 == 1' own/${f#*/}.jsonl > own/${f#*/}_30.jsonl; done
for spec in "v38 (44)|kvg_v38_s1|--ckpt kvg_best.pt" "new pooler segment 2 (39)|kvp_segment2_s1|--ckpt kvp_seg2.pt $AP" \
            "new pooler segment 4, no cap (31)|kvp_segment4_s1|--ckpt kvp_nocap/kvp_seg4.pt $AP" "kvm seed 1 (35)|kvm_mem_s1|--ckpt kvm.pt" \
            "kvm seed 2 (37)|kvm_mem_s2|--ckpt kvm.pt" "raw window (36)|gen_raw1024|--no-sp" "raw window seed 3 (37)|gen_raw1024_s3|--no-sp"; do
  lb=${spec%%|*}; r=${spec#*|}; fn=${r%%|*}; fl=${r#*|}; grep -qF "[kp] own $lb:" w.log && continue
  until idle; do sleep 45; done
  python3 sp_gk.py eval --traces own/${fn}_30.jsonl --max-len 12288 $M $fl --log sp_gk.log > run_own.log 2>&1
  echo "[kp] own $lb: $(grep -E "\[sp\] eval|Error" run_own.log | tail -1 | sed 's/^\[sp\] eval [^:]*: //' | cut -c1-300) $(date -u)"
done
grep "\[kp\] \(gk\|own\)" w.log > gk.txt; python3 -c "
from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='gk.txt', path_in_repo='archive/sp3/kvp/gk.txt', repo_id='$R')" > /dev/null 2>&1
echo "[kp] s84 done $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/s84.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s85now ]; then touch /root/.s85now
  cat > /root/s85.sh <<'JOB'
cd /root/w; export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i; R=baya1116/hypernet-sp-distill
touch STOP_KQ; pkill -f "bash /root/s82.sh"; pkill -f "bash /root/s84.sh"; pkill -f "bash /root/kvq_job.sh"; sleep 2
pkill -f "sp_qwen3.py rollout"; pkill -f "sp_qwen3.py train"; pkill -f "sp_gk.py"; sleep 10
echo "[kp] s85: oracle first - the on-policy round in progress is dropped $(date -u)"
for _t in $(seq 60); do curl -sS -o sp_or.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q "def oracle_trace" sp_or.py && break; sleep 120; done
[ -s or_in.jsonl ] || { head -n 6 held_math.jsonl > or_in.jsonl; head -n 6 held_comp.jsonl >> or_in.jsonl; }
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"; AP="--att-prior 1.0 --att-layers 8-19 --att-cap 1.0"
python3 sp_or.py oracle --traces or_in.jsonl --max-len 6144 $M $AP --ckpt kvp_seg2.pt --oracle-steps 200 --oracle-lr 1e-3 --log sp_or.log > run_or.log 2>&1
grep -E "^\[sp\] oracle|Error|Traceback" run_or.log | cut -c1-600 | sed 's/^\[sp\]/[kp]/'
python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='sp_or.log', path_in_repo='archive/sp3/kvp/oracle.log', repo_id='$R')" > /dev/null 2>&1
echo "[kp] s85: oracle done - the side checks, then the on-policy job $(date -u)"
rm -f /root/.s84idle; bash /root/s84.sh; rm -f STOP_KQ
exec bash /root/kvq_job.sh
JOB
  setsid nohup bash -c 'bash /root/s85.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s86workers ]; then touch /root/.s86workers
  cat > /root/s86.sh <<'JOB'
cd /root/w; export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i; R=baya1116/hypernet-sp-distill
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && return 0; sleep 60; done; echo "[kq] upload of $2 FAILED"; }
for _t in $(seq 60); do curl -sS -o /root/kvq_new.sh "$RAW/ctl/kvq_job.sh?nocache=$(date +%s)"; grep -q "WORKERS" /root/kvq_new.sh && { mv /root/kvq_new.sh /root/kvq_job.sh; break; }; sleep 120; done
echo 3 > kvq_workers; rd=$(cat kvq_round 2>/dev/null || echo 0)
awk '/^if \[ ! -s kvq_prompts.jsonl \]/,/^print.*fi$/' /root/kvq_job.sh > /root/kq_prompts.sh; bash /root/kq_prompts.sh > /dev/null   # the prompt list with the teacher answers
put kvq_prompts.jsonl kvq/prompts.jsonl; put kvq.pt kvq/kvq_v$rd.pt; echo $rd > VERSION; put VERSION kvq/VERSION
echo "[kq] s86: version $rd and $(wc -l < kvq_prompts.jsonl) prompts published for 3 workers; the job resumes with WORKERS=3 after the oracle and the side checks $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/s86.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s87run ]; then touch /root/.s87run
  cat > /root/s87.sh <<'JOB'
cd /root/w; export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i; R=baya1116/hypernet-sp-distill
while pgrep -f "sp_or.py oracle" > /dev/null; do sleep 30; done   # the first oracle finishes
pkill -f "bash /root/s85.sh"; pkill -f "bash /root/s84.sh"; pkill -f "bash /root/kvq_job.sh"; sleep 2; pkill -f "sp_gk.py"; pkill -f "sp_qwen3.py"; sleep 5; touch STOP_KQ
for _t in $(seq 60); do curl -sS -o sp_or2.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q "oracle-run" sp_or2.py && break; sleep 120; done
head -n 2 held_math.jsonl > orun_in.jsonl; head -n 2 held_comp.jsonl >> orun_in.jsonl
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"; AP="--att-prior 1.0 --att-layers 8-19 --att-cap 1.0"
echo "[kp] s87: oracle run - 6 consecutive positions in each of 4 chains of thought $(date -u)"
python3 sp_or2.py oracle --traces orun_in.jsonl --max-len 6144 $M $AP --ckpt kvp_seg2.pt --oracle-steps 200 --oracle-lr 1e-3 --oracle-run 6 --log sp_orun.log > run_orun.log 2>&1
grep -E "^\[sp\] oracle|Error|Traceback" run_orun.log | cut -c1-700 | sed 's/^\[sp\]/[kp]/'
python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='sp_orun.log', path_in_repo='archive/sp3/kvp/oracle_run.log', repo_id='$R')" > /dev/null 2>&1
echo "[kp] s87 done - the side checks, then the on-policy job (workers) $(date -u)"
bash /root/s84.sh; rm -f STOP_KQ
exec bash /root/kvq_job.sh
JOB
  setsid nohup bash -c 'bash /root/s87.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s88hz ]; then touch /root/.s88hz
  cat > /root/s88.sh <<'JOB'
cd /root/w; export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i; R=baya1116/hypernet-sp-distill
while pgrep -f "sp_or2.py oracle" > /dev/null || pgrep -f "sp_or.py oracle" > /dev/null; do sleep 30; done
pkill -f "bash /root/s87.sh"; pkill -f "bash /root/s85.sh"; pkill -f "bash /root/s84.sh"; pkill -f "bash /root/kvq_job.sh"; sleep 2; pkill -f "sp_gk.py"; pkill -f "sp_qwen3.py"; sleep 5; touch STOP_KQ
for _t in $(seq 60); do curl -sS -o sp_or3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q "oracle-horizon" sp_or3.py && break; sleep 120; done
[ -s orun_in.jsonl ] || { head -n 2 held_math.jsonl > orun_in.jsonl; head -n 2 held_comp.jsonl >> orun_in.jsonl; }
head -n 1 held_math.jsonl > oh_in.jsonl   # one chain (the user: if one fails, it is impossible; a pass proves nothing, but that is the bet)
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"; AP="--att-prior 1.0 --att-layers 8-19 --att-cap 1.0"
echo "[kp] s88: oracle with a horizon - slots fitted on the next 4 chunks, scored on the 2 after, 4 positions in one chain, pooler-seeded only, 120 steps (~5 min) $(date -u)"
python3 sp_or3.py oracle --traces oh_in.jsonl --max-len 6144 $M $AP --ckpt kvp_seg2.pt --oracle-steps 120 --oracle-lr 1e-3 --oracle-horizon 4 --oracle-run 4 --oracle-random 0 --log sp_oh.log > run_oh.log 2>&1
grep -E "^\[sp\] oracle|Error|Traceback" run_oh.log | cut -c1-900 | sed 's/^\[sp\]/[kp]/'
python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='sp_oh.log', path_in_repo='archive/sp3/kvp/oracle_h4.log', repo_id='$R')" > /dev/null 2>&1
echo "[kp] s88 done - the side checks, then the on-policy job (workers) $(date -u)"
bash /root/s84.sh; rm -f STOP_KQ
exec bash /root/kvq_job.sh
JOB
  setsid nohup bash -c 'bash /root/s88.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.s89stray ]; then touch /root/.s89stray /root/.s84idle
  pkill -f "bash /root/s84.sh"; sleep 1; pkill -f "sp_gk.py"; echo "[kp] s89: the stray side-check copy stopped; the oracle run has the GPU $(date -u)"
fi
if [ ! -e /root/.s92one ]; then touch /root/.s92one
  cat > /root/s92.sh <<'JOB'
cd /root/w; export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i; R=baya1116/hypernet-sp-distill
pkill -f "bash /root/s88.sh"; pkill -f "bash /root/s87.sh"; pkill -f "bash /root/s84.sh"; pkill -f "bash /root/kvq_job.sh"; sleep 2; pkill -f "sp_or3.py"; pkill -f "sp_gk.py"; pkill -f "sp_qwen3.py"; sleep 5; touch STOP_KQ
for _t in $(seq 60); do curl -sS -o sp_or3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q "oracle-random" sp_or3.py && break; sleep 120; done
for _t in $(seq 60); do curl -sS -o /root/kvq_new.sh "$RAW/ctl/kvq_job.sh?nocache=$(date +%s)"; grep -q "f\[16:\]" /root/kvq_new.sh && { mv /root/kvq_new.sh /root/kvq_job.sh; break; }; sleep 120; done
head -n 1 held_math.jsonl > oh_in.jsonl
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"; AP="--att-prior 1.0 --att-layers 8-19 --att-cap 1.0"
echo "[kp] s92: one-chain horizon oracle (fit on the next 4 chunks, scored on the 2 after, 4 consecutive positions, pooler-seeded, 120 steps) $(date -u)"
python3 sp_or3.py oracle --traces oh_in.jsonl --max-len 6144 $M $AP --ckpt kvp_seg2.pt --oracle-steps 120 --oracle-lr 1e-3 --oracle-horizon 4 --oracle-run 4 --oracle-random 0 --log sp_oh1.log > run_oh1.log 2>&1
grep -E "^\[sp\] oracle|Error|Traceback" run_oh1.log | cut -c1-900 | sed 's/^\[sp\]/[kp]/'
python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='sp_oh1.log', path_in_repo='archive/sp3/kvp/oracle_h4_one.log', repo_id='$R')" > /dev/null 2>&1
echo "[kp] s92 done - the side checks, then the on-policy job (workers) $(date -u)"
bash /root/s84.sh; rm -f STOP_KQ
exec bash /root/kvq_job.sh
JOB
  setsid nohup bash -c 'bash /root/s92.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
# CTL-END
