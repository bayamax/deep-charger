#!/bin/bash
# Box for the on-policy distillation of the 4-bit weights (tools/wq_opd.py from the DWQ + math weights wq4_dwq.pt), then the
# eval-set KL and the main eval with the 1024 fixed KV.
SER=1
if [ -f /root/.boxo_serial ] && [ "$(cat /root/.boxo_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxo_serial
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
if [ ! -e /root/.o1 ]; then touch /root/.o1
  cat > /root/o1.sh <<'JOB'
cd /root/w; sleep 20; until [ -e /root/w/setup_done ]; do sleep 30; done
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
for f in wq_opd.py wq_kl.py; do curl -sS -o $f "$RAW/tools/$f?nocache=$(date +%s)"; done
python3 -c "from huggingface_hub import hf_hub_download as d; [d('baya1116/hypernet-sp-distill', 'archive/sp3/' + f, local_dir='.') for f in ('wq/wq4_dwq.pt', 'mt/tr_lg1024_s1.jsonl', 'mt/tr2_lg1024_s1.jsonl', 'dag2_fullkv_teacher.jsonl', 'mt/mt_full_s1.jsonl', 'mt/docc_full_s1.jsonl')]"
A=archive/sp3; echo "[mt] opd start $(date -u)"
[ -s wq5_dwq.pt ] || python3 wq_opd.py --init $A/wq/wq4_dwq.pt --calib $A/mt/tr_lg1024_s1.jsonl,$A/mt/tr2_lg1024_s1.jsonl,$A/dag2_fullkv_teacher.jsonl --evalc $A/mt/mt_full_s1.jsonl,$A/mt/docc_full_s1.jsonl --minutes 30 --out wq5 > opd.log 2>&1
grep "^\[wq\]" opd.log | grep -v "on-policy step" | sed 's/^/[mt] /'; grep "on-policy step" opd.log | tail -2 | sed 's/^/[mt] /'; grep -E "Error|error" opd.log | tail -2 | sed 's/^/[mt] opd /'
python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/wq/'+sys.argv[1], repo_id='baya1116/hypernet-sp-distill')" wq5_dwq.pt && echo "[mt] wq5_dwq.pt uploaded"
python3 wq_kl.py $A/mt/mt_full_s1.jsonl,$A/mt/docc_full_s1.jsonl wq5_dwq.pt > wq_kl5.log 2>&1
grep "^\[wq\]" wq_kl5.log | sed 's/^/[mt] /'; grep -E "Error|error" wq_kl5.log | tail -2 | sed 's/^/[mt] wq_kl /'
L="--kv-policy=struct2,--kv-pool=7,--kv-lines=32,--mem-prompt,--kv-lmk,--lmk-thr=0.004,--lmk-nb=0,--lmk-query=own+q,--lmk-eol,--lmk-turn"
NEED="dequantised 4-bit" RB=32 CONDS="q4o1024:512:512:$L,--wq=/root/w/wq5_dwq.pt" bash /root/mt_job.sh
echo "[mt] ALL5_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/o1.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
# CTL-END
