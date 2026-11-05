#!/bin/bash
# Collector Ce (replaces Cb, which stopped taking its control script and stalled 3.5 h into a block) (the user: parallel DPO data collection on cheaper boxes; start-up check, GPU check, keep the cheapest
# that work). Serial 1: GPU empty?, bf16 matmul speed, model download time - one [pr] RESULT line. COLLECT=1 (a later serial)
# starts ctl/kvc_job.sh with this box's shard.
# serial 2: GRPO with guides - restart as a rollout worker (MODE=grpo: the memory x4 + the raw window per problem).
# serial 3: plain GRPO (the user): groups of 12 memory draws, no raw-window guide, blocks of 6 problems.
# serial 4: two-stage sampling (4 draws, then 8 more only where the 4 were mixed) - the running block finishes first
# (STOP_KC), then the current kvc_job.sh carries on.
# serial 5: new code (the trainer's weights now carry fp32 LoRA adapters; the rollout loads them from the checkpoint) -
# after the running block, fetch sp_qwen3.py / kvc_job.sh again and carry on.
# serial 6: pass 1 of the pool is nearly used up (Cg finished its range and sat idle) - after the running block, pass 2:
# the whole pool reshuffled, 6 shards, this box shard 2.
SER=6; COLLECT=1; SHARD=0; NSHARD=4; MODE=grpo
if [ -f /root/.boxc_serial ] && [ "$(cat /root/.boxc_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxc_serial
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w
cat > /root/status.sh <<'ST'
grep -E "^\[(pr|kc|hub)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 4 | cut -c1-300
tail -n 1 $(ls -t /root/w/sp_*.log 2>/dev/null | head -1) 2>/dev/null | cut -c1-200
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
if [ "$COLLECT" = 1 ] && [ ! -e /root/.collect_g12 ]; then touch /root/.collect_g12 /root/.collect_g /root/.collect3 /root/.collect
  for J in $(pgrep -f "bash /root/kvc_job.sh"); do pkill -P $J; kill $J; done; sleep 3; pkill -f "sp_qwen3.py"; sleep 3
  curl -sS -o /root/kvc_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvc_job.sh?nocache=$(date +%s)"
  setsid nohup bash -c "MODE=$MODE SHARD=$SHARD NSHARD=$NSHARD bash /root/kvc_job.sh >> /root/w/w.log 2>&1" > /dev/null 2>&1 < /dev/null &
  echo "[kc] collector launched: shard $SHARD of $NSHARD $(date -u)"
fi
if [ ! -e /root/.graceful2 ] && [ -e /root/.collect_g12 -o -e /root/.collect_g ]; then touch /root/.graceful2
  touch /root/w/STOP_KC
  cat > /root/regen.sh <<'JOB'
while pgrep -f "bash /root/kvc_job.sh" > /dev/null; do sleep 30; done
rm -f /root/w/STOP_KC
curl -sS -o /root/kvc_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvc_job.sh?nocache=$(date +%s)"
echo "[kc] two-stage sampling from here $(date -u) ($(grep -c 'G1=' /root/kvc_job.sh))"
exec env MODE=grpo SHARD=0 NSHARD=4 bash /root/kvc_job.sh
JOB
  setsid nohup bash -c 'bash /root/regen.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.graceful3 ]; then touch /root/.graceful3
  touch /root/w/STOP_KC
  cat > /root/regen3.sh <<'JOB'
while pgrep -f "bash /root/kvc_job.sh" > /dev/null; do sleep 30; done
rm -f /root/w/STOP_KC; cd /root/w
for _t in $(seq 60); do curl -sS -o sp_new.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q '"--lora-lr"' sp_new.py && break; sleep 120; done; mv sp_new.py sp_qwen3.py
curl -sS -o /root/kvc_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvc_job.sh?nocache=$(date +%s)"
echo "[kc] LoRA-aware code from here $(date -u)"
exec env MODE=grpo SHARD=0 NSHARD=4 bash /root/kvc_job.sh
JOB
  setsid nohup bash -c 'bash /root/regen3.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.pass2 ]; then touch /root/.pass2
  touch /root/w/STOP_KC
  cat > /root/pass2.sh <<'JOB'
while pgrep -f "bash /root/kvc_job.sh" > /dev/null; do sleep 30; done
rm -f /root/w/STOP_KC; cd /root/w
for _t in $(seq 60); do curl -sS -o /root/kvc_job.sh "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/kvc_job.sh?nocache=$(date +%s)"; grep -q -- "PASS=" /root/kvc_job.sh && break; sleep 120; done
echo "[kc] pass 2, shard 2 of 6 from here $(date -u)"
exec env PASS=2 MODE=grpo SHARD=2 NSHARD=6 bash /root/kvc_job.sh
JOB
  setsid nohup bash -c 'bash /root/pass2.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
# CTL-END
