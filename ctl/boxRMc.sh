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
SER=32
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
# CTL-END
