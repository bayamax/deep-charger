#!/bin/bash
# Box AP (3090 24 GB, pytorch image): the ap32_large situation, exactly - the ORIGINAL scripts (HF squash b5989f09 archive/:
# train_attnpool.py, diag_qwen_ablation.py, train_qwen_distill.py, rw_eval.py, data_prep.py), the original model
# (DeepSeek-R1-Distill-Qwen-1.5B, frozen) and data (Dolphin-R1 reasoning-deepseek, answers >= 3500 chars, 768 tokens).
# (1) score the real ap32_large.pt with the original rw_eval (record: deep KL 0.124 at rw 128, raw160 ~0.19) and with
# tools/sp_qwen3.py on the same 24 contexts (does our evaluator agree?); (2) retrain from scratch with the original trainer
# (S 32, rw 128, C 64, 4000 contexts once, lr 3e-4) and score it the same way.
SER=1
if [ -f /root/.boxap_serial ] && [ "$(cat /root/.boxap_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxap_serial
echo "[ap] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w; RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i"
curl -sS -o /root/w/sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
cat > /root/status.sh <<'ST'
grep -E "^\[(ap|sp|hub|math|mix)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-220
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.job ]; then touch /root/.job
  cat > /root/live.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
while [ ! -e DONE_AP ]; do
  { grep -E "^\[(ap|hub|math|mix)\]" w.log | tail -n 20; tail -n 12 /root/w/ap.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; df -h / | tail -1; date -u; } > ap_live.log 2>&1
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='ap_live.log', path_in_repo='archive/sp3/ap_live.log', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1
  sleep 600
done
JOB
  cat > /root/job.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; echo "[ap] put $2 attempt $t: $(tail -1 put.err | cut -c1-160)"; sleep 90; done; echo "[hub] $2 FAILED"; }
get() { for t in 1 2 3; do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o "$2" "$H/$1" && return 0; echo "[ap] get $1 attempt $t failed"; sleep 10; done; return 1; }
python3 -m pip install -q "huggingface_hub[hf_xet]" transformers datasets 2>&1 | grep -v -i warn | tail -1
M=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
for t in 1 2 3; do
  x=""; [ $t -gt 1 ] && x="HF_HUB_DISABLE_XET=1"
  env $x timeout 600 python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$M'))" > dl.log 2>&1 && break
  echo "[ap] model download attempt $t failed"
done
echo "[ap] model: $(tail -1 dl.log | cut -c1-120)"
REV=b5989f09356eee0060ee7460dd249a83054622f7; HR=https://huggingface.co/$R/resolve/$REV
mkdir -p /workspace/attnpool; cd /workspace
for f in train_attnpool.py diag_qwen_ablation.py train_qwen_distill.py rw_eval.py data_prep.py; do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o $f "$HR/archive/$f" || echo "[ap] get $f FAILED"; done
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o /workspace/attnpool/ap32_large.pt "$HR/checkpoints/ap32_large.pt"; echo "[ap] ap32_large $(stat -c %s /workspace/attnpool/ap32_large.pt) bytes"
sed -i 's/^N = 50000/N = 12000/' data_prep.py
python3 data_prep.py > /root/w/prep.log 2>&1; echo "[ap] data: $(tail -1 /root/w/prep.log | cut -c1-120)"
cd /root/w; cp /workspace/*.py /root/w/ 2>/dev/null
RAWB=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/tools
curl -sS -o sp_qwen3.py "$RAWB/sp_qwen3.py?nocache=$(date +%s)"; curl -sS -o ap_convert.py "$RAWB/ap_convert.py?nocache=$(date +%s)"
LG=/root/w/ap.log
echo "[ap] (1) the real ap32_large, original rw_eval $(date -u)"
cd /workspace && python3 rw_eval.py --ckpt /workspace/attnpool/ap32_large.pt --rw_list 128,160,256,512 >> $LG 2>&1; cd /root/w
grep -E "^ *[0-9]+ +[0-9]+ +[0-9.]+|Error|Traceback" $LG | tail -5 | sed "s/^/[ap] rw_eval ap32: /"
python3 ap_convert.py /workspace/dolphin_subset.jsonl ap_eval.jsonl --skip 200 --n 24 --model $M >> $LG 2>&1
Q="--model $M --traces ap_eval.jsonl --max-len 768 --S 32 --heads 8 --layers 3 --ffn 2048 --C 64 --maxd 0 --log $LG"
python3 sp_qwen3.py eval $Q --rw 128 --ckpt /workspace/attnpool/ap32_large.pt 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[ap] ours: /"
python3 sp_qwen3.py eval $Q --rw 160 --no-sp 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[ap] ours: /"
python3 sp_qwen3.py eval $Q --rw 512 --ckpt /workspace/attnpool/ap32_large.pt 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[ap] ours: /"
put $LG ap.log
echo "[ap] (2) retrain from scratch with the original trainer $(date -u)"
cd /workspace && python3 train_attnpool.py --n_sp 32 --raw_window 128 --chunk_size 64 --max_ans_len 768 --n_total 4016 --n_eval 16 --epochs 1 --eval_every_steps 500 --lr 3e-4 --out /workspace/attnpool/ap32_repro.pt >> $LG 2>&1; cd /root/w
grep -E "deep\(>160\)|DONE_ATTNPOOL|Error|Traceback|out of memory" $LG | tail -4 | sed "s/^/[ap] train: /"
put /workspace/attnpool/ap32_repro.pt ap32_repro_r1distill.pt
cd /workspace && python3 rw_eval.py --ckpt /workspace/attnpool/ap32_repro.pt --rw_list 128,160,256,512 >> $LG 2>&1; cd /root/w
grep -E "^ *[0-9]+ +[0-9]+ +[0-9.]+|Error|Traceback" $LG | tail -4 | sed "s/^/[ap] rw_eval repro: /"
put $LG ap.log; date -u > DONE_AP; put DONE_AP DONE_AP; echo "[ap] DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/live.sh' > /dev/null 2>&1 < /dev/null &
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[ap] job launched $(date -u)"
fi
# CTL-END
