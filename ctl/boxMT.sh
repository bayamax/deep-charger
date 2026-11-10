#!/bin/bash
# A fresh box for the long-range eval (ctl/mt_job.sh), because the stopped 3090 (58853599) could not restart: its host's GPU
# was taken ("resources_unavailable"). Only this job runs here.
# serial 2: the user - work out why it failed and find how small a fixed KV can be without an extra model. After the running
# condition (window 1024 + B=512) the chain switches to the new options, 32 streams: the window alone 1536; struct + pool;
# full context; struct + pool + 4-bit older entries (~1500 fp16 entries); the window alone 1024; 2-bit (~1600); 4-bit with a
# 512 window (~1000); pooling alone.
# serial 3: struct + pool (window 1024 + 512) read 104/160 against attention selection 55 and the window alone 38 at the same
# budget (facts and code 30/30). Queued after serial 2's chain: smaller budgets - struct at ~1024 (window 512 + 512) and ~512
# (256 + 256), and struct + 2-bit older entries at ~1000 (512 + 2560) and ~500 (256 + 1280).
# serial 4: full context ran out of memory (8 streams of ~13k tokens) - it runs again with half the KV allowance (4 streams),
# right after serial 2's chain and before serial 3's smaller budgets (serial 3's waiting script is replaced by this one).
# serial 5: the 4-bit run ran out of memory too (the simulation holds quantised entries at 16 bits: 32 streams of 2.7k
# entries). Each condition now carries its own KV allowance (--kv-budget, in entries, sets the stream count). After the running
# window-1024 condition: 4-bit, 2-bit, the smaller budgets, pooling alone, then full context last (4 streams, slow).
# serial 6: 4-bit older entries read 116/160 (struct alone 104); the log misses are near-misses (LL-3440 for LP-3440, SV-934 for
# SV-8934): pieces of a line survive. --kv-lines 32 keeps or drops whole lines. After the running 2-bit condition: 4-bit +
# whole lines; the smaller budgets then use whole lines if they beat plain 4-bit; pooling alone; full context last.
SER=6
if [ -f /root/.boxmt_serial ] && [ "$(cat /root/.boxmt_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxmt_serial
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w
cat > /root/status.sh <<'ST'
grep -E "^\[mt\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 8 | cut -c1-600
tail -n 1 /root/w/sp_mt.log 2>/dev/null | cut -c1-250
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.mt1 ]; then touch /root/.mt1
  cat > /root/mt1.sh <<'JOB'
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
cd /root/w; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
echo "[mt] box up: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader) $(date -u)"
python3 -m pip install -q "huggingface_hub[hf_xet]" transformers "math-verify[antlr4_13_2]" 2>&1 | grep -v -i warn | tail -1
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
timeout 1500 python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B')" > dl.log 2>&1
echo "[mt] setup done: transformers $(python3 -c 'import transformers; print(transformers.__version__)' 2>&1 | tail -1) $(date -u)"
for _t in $(seq 60); do curl -sS -o /root/mt_job.sh "$RAW/ctl/mt_job.sh?nocache=$(date +%s)"; grep -q "CONDS" /root/mt_job.sh && break; sleep 120; done
CONDS="k512:1024:512 w1536:1536:0 full:16384:0 w1024:1024:0 k1024:1024:1024 w2048:2048:0" exec bash /root/mt_job.sh
JOB
  setsid nohup bash -c 'bash /root/mt1.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.mt2 ]; then touch /root/.mt2
  cat > /root/mt2.sh <<'JOB'
cd /root/w; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
touch STOP_MT; while pgrep -f "bash /root/mt_job.sh" > /dev/null; do sleep 60; done; rm -f STOP_MT
for _t in $(seq 60); do curl -sS -o /root/mt_job.sh "$RAW/ctl/mt_job.sh?nocache=$(date +%s)"; grep -q "NEED" /root/mt_job.sh && break; sleep 120; done
echo "[mt] serial 2: the new options from here $(date -u)"
S="--kv-policy=struct,--kv-pool=7"
NEED="kv-policy" RB=32 CONDS="w1536:1536:0 st512:1024:512:$S full:16384:0 sq4:1024:1536:$S,--kv-quant=4 w1024:1024:0 sq2:1024:3072:$S,--kv-quant=2 sq4w512:512:1536:$S,--kv-quant=4 p512:1024:512:--kv-pool=7" exec bash /root/mt_job.sh
JOB
  setsid nohup bash -c 'bash /root/mt2.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.mt3 ]; then touch /root/.mt3
  cat > /root/mt3.sh <<'JOB'
cd /root/w; sleep 30; while pgrep -f "bash /root/mt2.sh" > /dev/null || pgrep -f "bash /root/mt_job.sh" > /dev/null; do sleep 60; done
echo "[mt] serial 3: smaller budgets $(date -u)"
S="--kv-policy=struct,--kv-pool=7"
NEED="kv-policy" RB=32 CONDS="s1024:512:512:$S sq2w512:512:2560:$S,--kv-quant=2 s512:256:256:$S sq2w256:256:1280:$S,--kv-quant=2" exec bash /root/mt_job.sh
JOB
  setsid nohup bash -c 'bash /root/mt3.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.mt4 ]; then touch /root/.mt4
  pkill -f "bash /root/mt3.sh"
  cat > /root/mt4.sh <<'JOB'
cd /root/w; sleep 30; while pgrep -f "bash /root/mt2.sh" > /dev/null || pgrep -f "bash /root/mt_job.sh" > /dev/null; do sleep 60; done
echo "[mt] serial 4: full context at 4 streams, then the smaller budgets $(date -u)"
S="--kv-policy=struct,--kv-pool=7"
NEED="kv-policy" RB=32 KB=56000 CONDS="full:16384:0" bash /root/mt_job.sh
NEED="kv-policy" RB=32 CONDS="s1024:512:512:$S sq2w512:512:2560:$S,--kv-quant=2 s512:256:256:$S sq2w256:256:1280:$S,--kv-quant=2" exec bash /root/mt_job.sh
JOB
  setsid nohup bash -c 'bash /root/mt4.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.mt5 ]; then touch /root/.mt5
  pkill -f "bash /root/mt4.sh"; touch /root/w/STOP_MT
  cat > /root/mt5.sh <<'JOB'
cd /root/w; while pgrep -f "bash /root/mt_job.sh" > /dev/null; do sleep 60; done; rm -f STOP_MT
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
for _t in $(seq 60); do curl -sS -o /root/mt_job.sh "$RAW/ctl/mt_job.sh?nocache=$(date +%s)"; grep -q "they come last" /root/mt_job.sh && break; sleep 120; done
echo "[mt] serial 5: per-condition KV allowance $(date -u)"
S="--kv-policy=struct,--kv-pool=7"; K5="--kv-budget=50000"
NEED="kv-policy" RB=32 CONDS="sq4:1024:1536:$S,--kv-quant=4,$K5 sq2:1024:3072:$S,--kv-quant=2,$K5 s1024:512:512:$S sq4w512:512:1536:$S,--kv-quant=4,$K5 sq2w512:512:2560:$S,--kv-quant=2,$K5 s512:256:256:$S sq2w256:256:1280:$S,--kv-quant=2,$K5 p512:1024:512:--kv-pool=7 full:16384:0:--kv-budget=54000" exec bash /root/mt_job.sh
JOB
  setsid nohup bash -c 'bash /root/mt5.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.mt6 ]; then touch /root/.mt6
  pkill -f "bash /root/mt5.sh"; touch /root/w/STOP_MT
  cat > /root/mt6.sh <<'JOB'
cd /root/w; while pgrep -f "bash /root/mt_job.sh" > /dev/null; do sleep 60; done; rm -f STOP_MT
echo "[mt] serial 6: whole lines $(date -u)"
S="--kv-policy=struct,--kv-pool=7"; K5="--kv-budget=50000"
NEED="kv-lines" RB=32 CONDS="sq4L:1024:1536:$S,--kv-quant=4,--kv-lines=32,$K5" bash /root/mt_job.sh
tot() { python3 mt_score.py $1 2>/dev/null | grep -oE "total [0-9]+" | grep -oE "[0-9]+"; }
L=""; [ -s mt_sq4L_s1.jsonl ] && [ "$(tot mt_sq4L_s1.jsonl)" -gt "$(tot mt_sq4_s1.jsonl)" ] && L=",--kv-lines=32"
echo "[mt] smaller budgets with '${L:-no whole lines}' $(date -u)"
NEED="kv-lines" RB=32 CONDS="s1024:512:512:$S$L sq4w512:512:1536:$S,--kv-quant=4,$K5$L sq2w512:512:2560:$S,--kv-quant=2,$K5$L s512:256:256:$S$L sq2w256:256:1280:$S,--kv-quant=2,$K5$L p512:1024:512:--kv-pool=7 full:16384:0:--kv-budget=54000" exec bash /root/mt_job.sh
JOB
  setsid nohup bash -c 'bash /root/mt6.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
fi
# CTL-END
