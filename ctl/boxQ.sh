#!/bin/bash
# Fresh box for the 4-bit weight quantization only (box 60106534 could not restart: its host's GPU was taken). Setup, then
# boxMT2 serial 11: tools/wq.py (rtn / gptq / dwq on the training conversations), then the 1024 fixed KV with the dwq and the gptq
# weights on ctl/mt_docc.jsonl and ctl/mt_eval.jsonl.
SER=1
if [ -f /root/.boxq_serial ] && [ "$(cat /root/.boxq_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxq_serial
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w
cat > /root/status.sh <<'ST'
grep -E "^\[mt\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 8 | cut -c1-600
tail -n 1 /root/w/sp_mt.log 2>/dev/null | cut -c1-250
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.n1 ]; then touch /root/.n1
  cat > /root/n1.sh <<'JOB'
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
cd /root/w; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
echo "[mt] box up: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader) $(date -u)"
python3 -m pip install -q "huggingface_hub[hf_xet]" transformers "math-verify[antlr4_13_2]" 2>&1 | grep -v -i warn | tail -1
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
timeout 1500 python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B')" > dl.log 2>&1
echo "[mt] setup done: transformers $(python3 -c 'import transformers; print(transformers.__version__)' 2>&1 | tail -1) $(date -u)"
for _t in $(seq 60); do curl -sS -o /root/mt_job.sh "$RAW/ctl/mt_job.sh?nocache=$(date +%s)"; grep -q "PFX" /root/mt_job.sh && break; sleep 120; done
touch /root/w/setup_done
JOB
  setsid nohup bash -c 'bash /root/n1.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.mt29 ]; then touch /root/.mt29
  cat > /root/mt29.sh <<'JOB'
cd /root/w; sleep 20; until [ -e /root/w/setup_done ]; do sleep 30; done
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
curl -sS -o wq.py "$RAW/tools/wq.py?nocache=$(date +%s)"
python3 -c "from huggingface_hub import hf_hub_download as d; [print(d('baya1116/hypernet-sp-distill', 'archive/sp3/mt/' + f, local_dir='.')) for f in ('tr_lg1024_s1.jsonl', 'tr2_lg1024_s1.jsonl')]"
echo "[mt] wq start $(date -u)"
[ -s wq_dwq.pt ] || python3 wq.py --calib archive/sp3/mt/tr_lg1024_s1.jsonl,archive/sp3/mt/tr2_lg1024_s1.jsonl --methods rtn,gptq,dwq --out wq > wq.log 2>&1
grep "^\[wq\]" wq.log | grep -v "dwq step\|gptq layer" | sed 's/^/[mt] /'; grep -E "Error|error" wq.log | tail -2 | sed 's/^/[mt] wq /'
for m in dwq gptq; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/wq/'+sys.argv[1], repo_id='baya1116/hypernet-sp-distill')" wq_$m.pt && echo "[mt] wq_$m.pt uploaded"; done
L="--kv-policy=struct2,--kv-pool=7,--kv-lines=32,--mem-prompt,--kv-lmk,--lmk-thr=0.004,--lmk-nb=0,--lmk-query=own+q,--lmk-eol,--lmk-turn"
NEED="dequantised 4-bit" TR=mt_docc.jsonl PFX=docc RB=32 CONDS="q4d1024:512:512:$L,--wq=/root/w/wq_dwq.pt" bash /root/mt_job.sh
NEED="dequantised 4-bit" RB=32 CONDS="q4d1024:512:512:$L,--wq=/root/w/wq_dwq.pt" bash /root/mt_job.sh
NEED="dequantised 4-bit" TR=mt_docc.jsonl PFX=docc RB=32 CONDS="q4g1024:512:512:$L,--wq=/root/w/wq_gptq.pt" bash /root/mt_job.sh
NEED="dequantised 4-bit" RB=32 CONDS="q4g1024:512:512:$L,--wq=/root/w/wq_gptq.pt" bash /root/mt_job.sh
JOB
  setsid nohup bash -c 'bash /root/mt29.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.wqlog ]; then touch /root/.wqlog; grep "^\[wq\]" /root/w/wq.log | grep -v "dwq step\|gptq layer"; grep "dwq step" /root/w/wq.log | tail -2; fi
if [ ! -e /root/.mt30 ]; then touch /root/.mt30
  cat > /root/mt30.sh <<'JOB'
cd /root/w; sleep 20; while pgrep -f "bash /root/mt29.sh" > /dev/null || pgrep -f "bash /root/mt_job.sh" > /dev/null; do sleep 60; done
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
curl -sS -o wq.py "$RAW/tools/wq.py?nocache=$(date +%s)"
echo "[mt] wq3 start $(date -u)"
[ -s wq3_dwq.pt ] || python3 wq.py --calib archive/sp3/mt/tr_lg1024_s1.jsonl,archive/sp3/mt/tr2_lg1024_s1.jsonl --methods dwq --init wq_gptq.pt --dwq-epochs 3 --out wq3 > wq3.log 2>&1
grep "^\[wq\]" wq3.log | grep -v "dwq step" | sed 's/^/[mt] /'; grep "dwq step" wq3.log | tail -1 | sed 's/^/[mt] /'; grep -E "Error|error" wq3.log | tail -2 | sed 's/^/[mt] wq3 /'
for f in wq3_dwq.pt wq3_summary.json wq_summary.json; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/wq/'+sys.argv[1], repo_id='baya1116/hypernet-sp-distill')" $f && echo "[mt] $f uploaded"; done
L="--kv-policy=struct2,--kv-pool=7,--kv-lines=32,--mem-prompt,--kv-lmk,--lmk-thr=0.004,--lmk-nb=0,--lmk-query=own+q,--lmk-eol,--lmk-turn"
NEED="dequantised 4-bit" TR=mt_docc.jsonl PFX=docc RB=32 CONDS="q4e1024:512:512:$L,--wq=/root/w/wq3_dwq.pt" bash /root/mt_job.sh
NEED="dequantised 4-bit" RB=32 CONDS="q4e1024:512:512:$L,--wq=/root/w/wq3_dwq.pt" bash /root/mt_job.sh
JOB
  setsid nohup bash -c 'bash /root/mt30.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.mt31 ]; then touch /root/.mt31
  cat > /root/mt31.sh <<'JOB'
cd /root/w; sleep 20; while pgrep -f "bash /root/mt30.sh" > /dev/null || pgrep -f "bash /root/mt_job.sh" > /dev/null; do sleep 60; done
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
curl -sS -o wq_kl.py "$RAW/tools/wq_kl.py?nocache=$(date +%s)"
python3 -c "from huggingface_hub import hf_hub_download as d; [d('baya1116/hypernet-sp-distill', 'archive/sp3/mt/' + f, local_dir='.') for f in ('mt_full_s1.jsonl', 'docc_full_s1.jsonl')]"
python3 wq_kl.py archive/sp3/mt/mt_full_s1.jsonl,archive/sp3/mt/docc_full_s1.jsonl wq_rtn.pt wq_gptq.pt wq_dwq.pt wq3_dwq.pt > wq_kl.log 2>&1
grep "^\[wq\]" wq_kl.log | sed 's/^/[mt] /'; grep -E "Error|error" wq_kl.log | tail -2 | sed 's/^/[mt] wq_kl /'
echo "[mt] ALL_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/mt31.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
# CTL-END
