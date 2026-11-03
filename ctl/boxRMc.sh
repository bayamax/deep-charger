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
SER=12
if [ -f /root/.boxrmx_serial ] && [ "$(cat /root/.boxrmx_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxrmx_serial
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w
cat > /root/status.sh <<'ST'
grep -E "^\[(pr|rm|kv|kr|hub)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 6 | cut -c1-600
tail -n 1 /root/w/sp_kvr.log 2>/dev/null | cut -c1-200
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
# CTL-END
