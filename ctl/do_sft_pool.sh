#!/bin/bash
# Pooler-lineage SFT launcher (box B). Steps: build fft_hf from student.pt, convert generated
# trajectories to the harness's corpus format, seed the checkpoint from fft_final (pooler included),
# then start sft_pool_run.py once. Guards against double launch.
cd /root/work
if pgrep -f "sft_pool_ru[n].py" >/dev/null; then echo "SFT ALREADY RUNNING"; exit 0; fi
for i in $(seq 1 180); do grep -q BOOT_DONE /root/boot.log 2>/dev/null && break; [ $i -eq 1 ] && echo "waiting for BOOT_DONE"; sleep 60; done
grep -q BOOT_DONE /root/boot.log || { echo "BOOT NOT DONE after 3h"; tail -3 /root/boot.log; exit 0; }
MODE=${1:-smoke}     # smoke: 24 old-corpus rows | full: convert /root/work/gen_mus200.jsonl | stream: rows arrive in /root/harvest_sft/
# 1) fft_hf (HF dir for SP_BASE) from student.pt + R1 config
if [ ! -f /root/fft_hf/config.json ]; then
  python3 - <<'PY'
import torch, time, shutil, os
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
B="/root/hfdl/r1base"; t0=time.time()
cfg=AutoConfig.from_pretrained(B); m=AutoModelForCausalLM.from_config(cfg).half()
sd=torch.load("/root/hfdl/fft_out/student.pt", map_location="cpu"); m.load_state_dict(sd, strict=True); del sd
m.save_pretrained("/root/fft_hf", safe_serialization=True); AutoTokenizer.from_pretrained(B).save_pretrained("/root/fft_hf")
print("FFT_HF_DONE %.0fs" % (time.time()-t0), flush=True)
PY
fi
ls /root/fft_hf | head -3
# 2) data
mkdir -p /root/harvest_sft
if [ "$MODE" = "smoke" ]; then
  head -n 24 /root/work/corpus_box_final.jsonl > /root/harvest_sft/smoke.jsonl; IDLE=120
elif [ "$MODE" = "stream" ]; then
  rm -f /root/harvest_sft/smoke.jsonl; IDLE=${IDLE_EXIT:-14400}
else
  python3 - <<'PY'
import json
n=0; keep=0
with open("/root/harvest_sft/gen_mus200.jsonl","w") as fh:
    for l in open("/root/work/gen_mus200.jsonl"):
        r=json.loads(l); n+=1
        if not (r.get("correct") and r.get("grounded")): continue
        keep+=1
        fh.write(json.dumps({"q":r["q"],"gold":r.get("gold",""),"ok":True,"gnd":True,"srch":r.get("ns"),"traj":r["text"]}, ensure_ascii=False)+"\n")
print("gen rows", n, "kept correct+grounded", keep)
PY
  IDLE=900
fi
# 3) checkpoint seed (resume from the pooler SFT final weights)
OUTF=/root/fft_new_${MODE}.safetensors
[ -f $OUTF ] || cp /root/hfdl/box_recover/fft_final.safetensors $OUTF
: > /root/fft_seen_${MODE}.txt
# 4) launch (same recipe as the box pipeline; harness prefix expects /root/work layout)
SP_BASE=/root/fft_hf SP_RANK=128 SP_NOSYS=1 SP_EPISODIC=1 SP_RERANK=0 SP_HOTPOT2=0 SP_MAXD=4096 SP_LEN_NORM=0 SP_TRAIN_POOLER=1 \
FFT_LR=1e-5 SP_POOLER_LR=1e-5 FFT_FREEZE_BELOW=0 FFT_FREEZE_IO=1 FFT_GRADCKPT=0 FFT_CHUNK=128 \
SFT_ACC=8 SFT_ONE_RATIO=0.8 SFT_IDLE_EXIT=$IDLE FFT_VAL_EVERY=25 FFT_VAL_PIN=/root/hfdl/box_recover/fft_val_set.txt \
SFT_HARVEST="/root/harvest_sft/*.jsonl" FFT_SEEN=/root/fft_seen_${MODE}.txt SFT_LOSSLOG=/root/sft_loss_${MODE}.log FFT_VALLOG=/root/fft_val_${MODE}.log \
SFT_SAVE_EVERY=25 SFT_SNAP_EVERY=40 SFT_SNAP_KEEP=3 SFT_OUT=$OUTF \
OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
setsid nohup python3 /root/work/sft_pool_run.py > /root/sft_${MODE}.log 2>&1 < /dev/null &
sleep 240; echo "---log---"; grep -v "^$" /root/sft_${MODE}.log | tail -25; echo "---procs---"; pgrep -af "sft_pool_ru[n]"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
