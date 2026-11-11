#!/bin/bash
# Second box for the fixed-KV work (the 3090 59977293 could not restart again: its host's GPU was taken). Setup, then only the
# current work: serial 1 = the calibration pass of --kv-probe (see ctl/boxMT.sh serial 19).
# serial 2: the yes/no self-report is anti-correlated with need (AUC 0.10: 'yes' while solving a fresh problem, 'no' while
# answering a log question). The user's original idea instead: recall when attention goes out of the cache (--kv-lmk: one
# landmark key per evicted line). Calibration on ctl/mt_cal.jsonl with the threshold above 1 (never acts), every check logged
# (hub mt/lmk_cal.jsonl).
# serial 3: out-of-cache attention separates need from no need (AUC 0.85; threshold 0.004: all 20 log questions fire, 0 fires
# while solving an unrelated problem), but the landmarks point at the right line only 7/20; the model's own last 3 lines find
# it 13/20, with the user's question line 20/20 (offline, calibration set). The main eval at 512 + 512: attention decides when,
# own+q / own decide what; then the log in pieces with own+q.
SER=3
if [ -f /root/.boxmt2_serial ] && [ "$(cat /root/.boxmt2_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxmt2_serial
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
if [ ! -e /root/.mt19 ]; then touch /root/.mt19
  cat > /root/mt19.sh <<'JOB'
cd /root/w; until [ -e /root/w/setup_done ]; do sleep 30; done
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
L="--kv-policy=struct2,--kv-pool=7,--kv-lines=32,--mem-prompt,--kv-probe,--probe-thr=1.01,--probe-log=probe_cal.jsonl"
rm -f probe_cal.jsonl
NEED="mem-prompt" TR=mt_cal.jsonl PFX=cal RB=32 CONDS="p0:512:512:$L" bash /root/mt_job.sh
grep -E "kv-probe:" sp_mt.log | tail -1 | sed 's/^/[mt] /'
python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='probe_cal.jsonl', path_in_repo='archive/sp3/mt/probe_cal.jsonl', repo_id='baya1116/hypernet-sp-distill')" && echo "[mt] probe_cal.jsonl uploaded $(wc -l < probe_cal.jsonl) lines"
JOB
  setsid nohup bash -c 'bash /root/mt19.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.mt20 ]; then touch /root/.mt20
  cat > /root/mt20.sh <<'JOB'
cd /root/w; sleep 20; while pgrep -f "bash /root/mt_job.sh" > /dev/null; do sleep 60; done
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
L="--kv-policy=struct2,--kv-pool=7,--kv-lines=32,--mem-prompt,--kv-lmk,--lmk-thr=2,--probe-log=lmk_cal.jsonl"
rm -f lmk_cal.jsonl
NEED="kv-lmk" TR=mt_cal.jsonl PFX=cal RB=32 CONDS="l0:512:512:$L" bash /root/mt_job.sh
grep -E "kv-lmk:" sp_mt.log | tail -1 | sed 's/^/[mt] /'
python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='lmk_cal.jsonl', path_in_repo='archive/sp3/mt/lmk_cal.jsonl', repo_id='baya1116/hypernet-sp-distill')" && echo "[mt] lmk_cal.jsonl uploaded $(wc -l < lmk_cal.jsonl) lines"
JOB
  setsid nohup bash -c 'bash /root/mt20.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.mt21 ]; then touch /root/.mt21
  cat > /root/mt21.sh <<'JOB'
cd /root/w; sleep 20; while pgrep -f "bash /root/mt20.sh" > /dev/null || pgrep -f "bash /root/mt_job.sh" > /dev/null; do sleep 60; done
L="--kv-policy=struct2,--kv-pool=7,--kv-lines=32,--mem-prompt,--kv-lmk,--lmk-thr=0.004"
NEED="lmk-query" RB=32 CONDS="la1024:512:512:$L,--lmk-query=own+q lb1024:512:512:$L,--lmk-query=own" bash /root/mt_job.sh
NEED="lmk-query" TR=mt_docc.jsonl PFX=docc RB=32 CONDS="la1024:512:512:$L,--lmk-query=own+q" bash /root/mt_job.sh
JOB
  setsid nohup bash -c 'bash /root/mt21.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.mt22 ]; then touch /root/.mt22
  pkill -f "bash /root/mt21.sh"; pkill -f "bash /root/mt_job.sh"; pkill -f sp_qwen3.py; sleep 15
  cat > /root/mt22.sh <<'JOB'
cd /root/w; sleep 20; while pgrep -f "bash /root/mt_job.sh" > /dev/null; do sleep 60; done
L="--kv-policy=struct2,--kv-pool=7,--kv-lines=32,--mem-prompt,--kv-lmk,--lmk-thr=0.004,--lmk-nb=0"
NEED="lmk-nb" RB=32 CONDS="lc1024:512:512:$L,--lmk-query=own+q ld1024:512:512:$L,--lmk-query=own" bash /root/mt_job.sh
NEED="lmk-nb" TR=mt_docc.jsonl PFX=docc RB=32 CONDS="lc1024:512:512:$L,--lmk-query=own+q" bash /root/mt_job.sh
JOB
  setsid nohup bash -c 'bash /root/mt22.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.mt23 ]; then touch /root/.mt23
  pkill -f "bash /root/mt22.sh"; pkill -f "bash /root/mt_job.sh"; pkill -f "sp_mt.py rollout"; sleep 20; pkill -9 -f "sp_mt.py rollout"; sleep 5
  rm -f /root/w/mt_lc1024_s1.jsonl /root/w/mt_ld1024_s1.jsonl /root/w/docc_lc1024_s1.jsonl
  sed 's/mt22/mt23/' /root/mt22.sh > /root/mt23.sh
  setsid nohup bash -c 'bash /root/mt23.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.mt24 ]; then touch /root/.mt24
  cat > /root/mt24.sh <<'JOB'
cd /root/w; sleep 20; while pgrep -f "bash /root/mt23.sh" > /dev/null || pgrep -f "bash /root/mt_job.sh" > /dev/null; do sleep 60; done
L="--kv-policy=struct2,--kv-pool=7,--kv-lines=32,--mem-prompt,--kv-lmk,--lmk-thr=0.004,--lmk-nb=0,--lmk-query=own+q,--lmk-eol"
NEED="lmk-eol" RB=32 CONDS="le1024:512:512:$L lf1024:512:512:$L,--kv-frame" bash /root/mt_job.sh
JOB
  setsid nohup bash -c 'bash /root/mt24.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.mt25 ]; then touch /root/.mt25
  cat > /root/mt25.sh <<'JOB'
cd /root/w; sleep 30; while pgrep -f "bash /root/mt24.sh" > /dev/null || pgrep -f "bash /root/mt_job.sh" > /dev/null; do sleep 60; done
L="--kv-policy=struct2,--kv-pool=7,--kv-lines=32,--mem-prompt,--kv-lmk,--lmk-thr=0.004,--lmk-nb=0,--lmk-query=own+q,--lmk-eol,--lmk-turn"
NEED="lmk-turn" TR=mt_docc.jsonl PFX=docc RB=32 CONDS="lg1024:512:512:$L" bash /root/mt_job.sh
NEED="lmk-turn" RB=32 CONDS="lg1024:512:512:$L" bash /root/mt_job.sh
JOB
  setsid nohup bash -c 'bash /root/mt25.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.mt26 ]; then touch /root/.mt26
  cat > /root/mt26.sh <<'JOB'
cd /root/w; sleep 20; while pgrep -f "bash /root/mt_job.sh" > /dev/null; do sleep 60; done
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
L="--kv-policy=struct2,--kv-pool=7,--kv-lines=32,--mem-prompt,--kv-lmk,--lmk-thr=0.004,--lmk-nb=0,--lmk-query=own+q,--lmk-eol,--lmk-turn"
rm -f lmk_train.pt
NEED="lmk-dump" TR=mt_train.jsonl PFX=tr RB=32 CONDS="lg1024:512:512:$L,--lmk-dump=lmk_train.pt" bash /root/mt_job.sh
grep -E "lmk-dump:" sp_mt.log | tail -1 | sed 's/^/[mt] /'
python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='lmk_train.pt', path_in_repo='archive/sp3/mt/lmk_train.pt', repo_id='baya1116/hypernet-sp-distill')" && echo "[mt] lmk_train.pt uploaded $(du -m lmk_train.pt | cut -f1) MB"
JOB
  setsid nohup bash -c 'bash /root/mt26.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.mt27 ]; then touch /root/.mt27
  cat > /root/mt27.sh <<'JOB'
cd /root/w; sleep 20; while pgrep -f "bash /root/mt26.sh" > /dev/null || pgrep -f "bash /root/mt_job.sh" > /dev/null; do sleep 60; done
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
L="--kv-policy=struct2,--kv-pool=7,--kv-lines=32,--mem-prompt,--kv-lmk,--lmk-thr=0.004,--lmk-nb=0,--lmk-query=own+q,--lmk-eol,--lmk-turn"
rm -f lmk_train2.pt
NEED="labseg" TR=mt_train2.jsonl PFX=tr2 RB=32 CONDS="lg1024:512:512:$L,--lmk-dump=lmk_train2.pt" bash /root/mt_job.sh
grep -E "lmk-dump:" sp_mt.log | tail -1 | sed 's/^/[mt] /'
python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='lmk_train2.pt', path_in_repo='archive/sp3/mt/lmk_train2.pt', repo_id='baya1116/hypernet-sp-distill')" && echo "[mt] lmk_train2.pt uploaded $(du -m lmk_train2.pt | cut -f1) MB"
JOB
  setsid nohup bash -c 'bash /root/mt27.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.mt28 ]; then touch /root/.mt28
  cat > /root/mt28.sh <<'JOB'
cd /root/w; sleep 20; while pgrep -f "bash /root/mt27.sh" > /dev/null || pgrep -f "bash /root/mt_job.sh" > /dev/null; do sleep 60; done
L="--kv-policy=struct2,--kv-pool=7,--kv-lines=32,--mem-prompt,--kv-lmk,--lmk-thr=0.004,--lmk-nb=0,--lmk-query=rhead+q,--lmk-eol,--lmk-turn"
NEED="lmk-bw" TR=mt_docc.jsonl PFX=docc RB=32 CONDS="lr1024:512:512:$L" bash /root/mt_job.sh
NEED="lmk-bw" RB=32 CONDS="lr1024:512:512:$L" bash /root/mt_job.sh
JOB
  setsid nohup bash -c 'bash /root/mt28.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.mt29 ]; then touch /root/.mt29
  cat > /root/mt29.sh <<'JOB'
cd /root/w; sleep 20; while pgrep -f "bash /root/mt28.sh" > /dev/null || pgrep -f "bash /root/mt_job.sh" > /dev/null; do sleep 60; done
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
# CTL-END
