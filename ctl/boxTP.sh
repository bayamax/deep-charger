#!/bin/bash
# Box TP (3090 24 GB, pytorch image): the deep-charger recipe that made ap32_large (HF squash b5989f09: archive/train_attnpool.py,
# archive/HANDOFF.md), numbers moved to Qwen3-1.7B: model FROZEN, a 3-layer AttnPoolSP (S 32, 8 heads, ffn 2048) reads ALL past
# raw-token embeddings (no eviction), output KL on every chunk, raw window 128, chunk 64, 768-token contexts, 4000 contexts once,
# lr 3e-4. Two poolers side by side: the original (raw sinusoids, oldest-first) and pe-rev + pe-norm. Then held-out deep KL vs
# the equal-budget raw window (160) and the plain 128, the inference-time wider window (512 on 4096-token traces), and a
# generation check: 32 MATH problems, full KV vs SP + window 512 vs raw window 544 alone (accuracy, </think>, repetition).
SER=27
if [ -f /root/.boxtp_serial ] && [ "$(cat /root/.boxtp_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxtp_serial
echo "[tp] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w; RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i"
curl -sS -o /root/w/sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"
cat > /root/status.sh <<'ST'
grep -E "^\[(tp|sp|hub|math|mix)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 3 | cut -c1-300
tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-220
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
# serial 27: progress of the diverse long-trace run (groups of 150)
if [ ! -e /root/.chk27 ]; then touch /root/.chk27
  echo "[tp] s27: $(grep -E '^\[gen\]|Error|Traceback' /root/w/run_gendiv.log 2>/dev/null | tail -2 | tr '\n' ' ') $(wc -l < /root/w/divtr.jsonl 2>/dev/null) rows $(date -u)"
fi
# serial 26: the math run is at 493/900 after 1.6 h (~620 output tok/s: KV-bound concurrency) and vLLM returns only at the
# end - the diverse run uses the grouped generator (each group of 150 appended as it finishes, a rerun resumes)
if [ ! -e /root/.chk26 ]; then touch /root/.chk26
  curl -sS -o /root/w/gen_long_vllm.py "$RAW/tools/gen_long_vllm.py?nocache=$(date +%s)"; grep -q "a rerun skips" /root/w/gen_long_vllm.py && echo "[tp] s26: grouped generator in place" || echo "[tp] s26: WARNING generator not updated"
fi
# serial 25: watch vLLM's KV pressure on the long generation (preemption / throughput / errors), once
if [ ! -e /root/.chk25 ] || [ -s /root/w/run_gendiv.log ] && [ ! -e /root/.chk25b ]; then touch /root/.chk25; [ -s /root/w/run_gendiv.log ] && touch /root/.chk25b
  for f in /root/w/run_genlong.log /root/w/run_gendiv.log; do [ -s $f ] || continue
    echo "[tp] s25 $(basename $f): $(tr '\r' '\n' < $f | grep -E 'KV cache|GPU KV|Maximum concurrency|preempt|Processed prompts|Error|it/s' | tail -4 | cut -c1-200 | tr '\n' ' ')"; done
  echo "[tp] s25: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) $(date -u)"
fi
# serial 24: the long teacher traces must be diverse, not math only. When the math run ends, stop serial 22 before it trains;
# run the same full-KV generation on 740 diverse prompts (ctl/divlong.jsonl: open-domain reasoning 260, MMLU-Pro 160,
# programming 160, general instructions 160), then stage G on the finished traces of both, shuffled.
if [ ! -e /root/.job24 ]; then touch /root/.job24
  cat > /root/job24.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 90; done; echo "[hub] $2 FAILED"; }
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
curl -sS -o divlong.jsonl "$RAW/ctl/divlong.jsonl?nocache=$(date +%s)"; echo "[tp] s24: divlong $(wc -l < divlong.jsonl) prompts"
while pgrep -f "gen_long_vllm.py mathlong" > /dev/null || { [ ! -s longtr.jsonl ] && pgrep -f "^bash /root/job22.sh" > /dev/null; }; do sleep 1; done
J=$(pgrep -f "^bash /root/job22.sh"); [ -n "$J" ] && { pkill -P $J; kill $J; }; sleep 3; pkill -f "sp_qwen3.py train --traces trainG"; true
echo "[tp] s24: math long traces done, serial 22 stopped $(date -u)"
grep -E "^\[gen\]|Error|Traceback" run_genlong.log | tail -2 | sed "s/^/[tp] math: /"
put longtr.jsonl longtr_math_train_16k.jsonl
/root/vv/bin/python gen_long_vllm.py divlong.jsonl divtr.jsonl 16384 > run_gendiv.log 2>&1
grep -E "^\[gen\]|Error|Traceback" run_gendiv.log | tail -2 | sed "s/^/[tp] diverse: /"
put divtr.jsonl longtr_diverse_16k.jsonl
python3 -c "
import json, random
rs = [json.loads(l) for f in ('longtr.jsonl', 'divtr.jsonl') for l in open(f)]
ok = [r for r in rs if r.get('finish') == 'stop' and '</think>' in r['text']]; random.seed(0); random.shuffle(ok)
open('trainG.jsonl', 'w').writelines(json.dumps(r) + '\n' for r in ok)
import collections; print('[tp] s24: trainG', len(ok), 'finished traces', dict(collections.Counter(r['src'].split('-')[0] + '-' + r['src'].split('-')[1] for r in ok)))"
LG=/root/w/sp.log; F="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 384"
echo "[tp] stage G: end-aware FFT from E on long diverse traces $(date -u)"
python3 sp_qwen3.py train --traces trainG.jsonl --max-len 16384 $F --ckpt spE.pt --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 1e-4 --par --end-train --end-weight 4 --chunks-per-trace 8 --accum 8 --out spG.pt --log $LG > run_trainG.log 2>&1
grep -E "^\[sp\] (trained)|Error|Traceback" run_trainG.log | tail -3 | sed "s/^/[tp] G: /"; echo "[tp] G: $(grep -c 'out of memory' run_trainG.log) traces skipped (memory)"
put spG.pt spG_longtraces_fft1024.pt
for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 $F --ckpt spG.pt --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] G stageA-cond: /"; done
python3 sp_qwen3.py eval --traces kv_full_p.jsonl --max-len 16384 --end-stat $F --ckpt spG.pt --log $LG 2>&1 | grep -E "^\[sp\] (eval|end-stat)|Error|Traceback" | sed "s/^/[tp] stop-drive SP_G: /"
put $LG spP.log
G="--traces math96.jsonl --n 96 --max-len 12288 --S 32 --heads 8 --layers 3 --ffn 2048 --log $LG"
python3 sp_qwen3.py rollout $G --roll-batch 48 --rw 1024 --C 128 --maxd 384 --ckpt spG.pt --roll-out g96_G.jsonl > run_g96_G.log 2>&1
grep -E "Error|Traceback|out of memory" run_g96_G.log | tail -2 | sed "s/^/[tp] G: /"
python3 sp_mathcheck.py g96_G.jsonl g96_E.jsonl g96_full.jsonl g96_raw.jsonl | sed "s/^/[tp] /"; put g96_G.jsonl gen96_G.jsonl
put $LG spP.log; date -u > DONE_TF3; put DONE_TF3 DONE_TF3; echo "[tp] DONE_TF3 $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job24.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s24: diverse long traces + stage G on both queued $(date -u)"
fi
# serial 23: training first - stop the generation-setting search now (the -1.0 hesitation-bias run included); serial 22's
# stage G starts as soon as job21 is gone.
if [ ! -e /root/.job23 ]; then touch /root/.job23
  J=$(pgrep -f "^bash /root/job21.sh"); [ -n "$J" ] && { pkill -P $J; kill $J; echo "[tp] s23: generation-setting search stopped ($J)"; }
  sleep 3; pkill -f "roll-out g96_E_hb1.jsonl"; true
fi
# serial 22: full KV 80/96 (E 70, the window alone 77): of the 11 full-only problems E leaves 8 unfinished - all long thoughts
# (full KV needs 3k-11k tokens there), E loops in the tail. Every teacher trace in training was cut at 3200 tokens: long thinking
# (many evictions, a full buffer, positions far out) was never trained. Stage G, after the two hesitation-bias runs: 900
# MATH-train problems (levels 5/4/3: 500/300/100, ctl/mathlong.jsonl) through the full-KV model with vLLM to 16384 tokens,
# then end-aware FFT from E on the finished ones (whole traces, 8 blocks + start + end each), scores, and the 96-problem check.
# Replaces serial 20's stage F (same capped data).
if [ ! -e /root/.job22 ]; then touch /root/.job22
  J=$(pgrep -f "^bash /root/job20.sh"); [ -n "$J" ] && { pkill -P $J; kill $J; echo "[tp] s22: dropped the queued stage F"; }
  cat > /root/job22.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 90; done; echo "[hub] $2 FAILED"; }
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
curl -sS -o mathlong.jsonl "$RAW/ctl/mathlong.jsonl?nocache=$(date +%s)"; curl -sS -o gen_long_vllm.py "$RAW/tools/gen_long_vllm.py?nocache=$(date +%s)"
rm -f cur.pt best.pt nxt.pt spR.pt spD1.pt spQ.pt spP_norm.pt; echo "[tp] s22: disk $(df -h / | tail -1 | awk '{print $4}') free"
[ -x /root/vv/bin/python ] || { python3 -m venv /root/vv && /root/vv/bin/pip install -q vllm > vllm_install.log 2>&1; }
/root/vv/bin/python -c "import vllm; print('[tp] s22: vllm', vllm.__version__)" 2>&1 | tail -1
while [ ! -e run_g96_E_C32.log ] && pgrep -f "^bash /root/job21.sh" > /dev/null; do sleep 5; done
J=$(pgrep -f "^bash /root/job21.sh"); [ -n "$J" ] && { pkill -P $J; kill $J; }; sleep 3; pkill -f "roll-out g96_E_C32.jsonl"; sleep 5; rm -f g96_E_C32.jsonl
echo "[tp] s22: search stopped after the hesitation-bias runs; long teacher traces $(date -u)"
/root/vv/bin/python gen_long_vllm.py mathlong.jsonl longtr.jsonl 16384 > run_genlong.log 2>&1
grep -E "^\[gen\]|Error|Traceback" run_genlong.log | tail -3 | sed "s/^/[tp] /"
[ -s longtr.jsonl ] || { echo "[tp] s22: no long traces, stop"; exit 1; }
put longtr.jsonl longtr_math_train_16k.jsonl
python3 -c "
import json; rs=[json.loads(l) for l in open('longtr.jsonl')]; ok=[r for r in rs if r.get('finish')=='stop' and '</think>' in r['text']]
open('trainG.jsonl','w').writelines(json.dumps(r)+'\n' for r in ok); print('[tp] s22: trainG', len(ok), 'finished traces')"
LG=/root/w/sp.log; F="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 384"
echo "[tp] stage G: end-aware FFT from E on long traces $(date -u)"
python3 sp_qwen3.py train --traces trainG.jsonl --max-len 16384 $F --ckpt spE.pt --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 1e-4 --par --end-train --end-weight 4 --chunks-per-trace 8 --accum 8 --out spG.pt --log $LG > run_trainG.log 2>&1
grep -E "^\[sp\] (trained)|Error|Traceback|out of memory" run_trainG.log | tail -4 | sed "s/^/[tp] G: /"
put spG.pt spG_longtraces_fft1024.pt
for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 $F --ckpt spG.pt --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] G stageA-cond: /"; done
python3 sp_qwen3.py eval --traces kv_full_p.jsonl --max-len 16384 --end-stat $F --ckpt spG.pt --log $LG 2>&1 | grep -E "^\[sp\] (eval|end-stat)|Error|Traceback" | sed "s/^/[tp] stop-drive SP_G: /"
put $LG spP.log
G="--traces math96.jsonl --n 96 --max-len 12288 --S 32 --heads 8 --layers 3 --ffn 2048 --log $LG"; SP="--rw 1024 --C 128 --maxd 384 --ckpt spG.pt"
run() { n=$1; shift; echo "[tp] s22 run $n: $* $(date -u)"
  python3 sp_qwen3.py rollout $G "$@" --roll-out g96_$n.jsonl > run_g96_$n.log 2>&1
  grep -E "Error|Traceback|out of memory" run_g96_$n.log | tail -2 | sed "s/^/[tp] $n: /"
  python3 sp_mathcheck.py g96_$n.jsonl | sed "s/^/[tp] /"; put g96_$n.jsonl gen96_$n.jsonl; put $LG spP.log; }
run G       --roll-batch 48 $SP
python3 sp_mathcheck.py g96_*.jsonl | sed "s/^/[tp] final /"
put $LG spP.log; date -u > DONE_TF3; put DONE_TF3 DONE_TF3; echo "[tp] DONE_TF3 $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job22.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s22: stage G (long teacher traces) queued after the hesitation-bias runs $(date -u)"
fi
# serial 21: the </think> bias changes nothing (S: all 96 generations identical - </think> is outside top-20 while thinking)
# - skip E_eb1: start serial 19's work (own-drive, hesitation bias, knobs) as soon as full KV is done.
if [ ! -e /root/.job21 ]; then touch /root/.job21
  sed -e 's/run_g96_E_C32.log/run_g96_E_eb1.log/; s/roll-out g96_E_C32.jsonl/roll-out g96_E_eb1.jsonl/; s/rm -f g96_E_C32.jsonl/rm -f g96_E_eb1.jsonl/; s/job18 stopped after E_eb1/job18 stopped after full/' /root/job19.sh > /root/job21.sh
  grep -q "run_g96_E_eb1.log" /root/job21.sh && { J=$(pgrep -f "^bash /root/job19.sh"); [ -n "$J" ] && { pkill -P $J; kill $J; }
    setsid nohup bash -c 'bash /root/job21.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
    echo "[tp] s21: hesitation-bias runs moved before E_eb1 $(date -u)"; } || echo "[tp] s21: WARNING job19 not patched"
fi
# serial 20: stage E samples blocks (start, end, 4 random) - the accelerated version. Once the generation-setting search is
# done (DONE_TG16): stage F = the full pass, every block of every trace (--chunks-per-trace 0), --end-train / --end-weight 4,
# whole traces, from E on the rest of the unseen teacher traces (trainE from line 1201); scores, stop/hesitation drive, and
# the 96-problem check (default sampling and the -0.3 hesitation bias).
if [ ! -e /root/.job20 ]; then touch /root/.job20
  cat > /root/job20.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 90; done; echo "[hub] $2 FAILED"; }
while [ ! -e DONE_TG16 ] || pgrep -f "^bash /root/job1[89].sh" > /dev/null; do sleep 30; done
RAWB=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/tools
curl -sS -o sp_qwen3.py "$RAWB/sp_qwen3.py?nocache=$(date +%s)"
LG=/root/w/sp.log; F="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 384"
tail -n +1201 trainE.jsonl > trainF.jsonl; echo "[tp] stage F: full-pass FFT from E on $(wc -l < trainF.jsonl) traces $(date -u)"
python3 sp_qwen3.py train --traces trainF.jsonl --max-len 16384 $F --ckpt spE.pt --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 1e-4 --par --end-train --end-weight 4 --chunks-per-trace 0 --accum 8 --out spF.pt --log $LG > run_trainF.log 2>&1
grep -E "^\[sp\] (trained)|Error|Traceback|out of memory" run_trainF.log | tail -3 | sed "s/^/[tp] F: /"
put spF.pt spF_fullpass_fft1024.pt
for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 $F --ckpt spF.pt --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] F stageA-cond: /"; done
python3 sp_qwen3.py eval --traces kv_full_p.jsonl --max-len 16384 --end-stat $F --ckpt spF.pt --log $LG 2>&1 | grep -E "^\[sp\] (eval|end-stat)|Error|Traceback" | sed "s/^/[tp] stop-drive SP_F: /"
put $LG spP.log
G="--traces math96.jsonl --n 96 --max-len 12288 --S 32 --heads 8 --layers 3 --ffn 2048 --log $LG"; SP="--rw 1024 --C 128 --maxd 384 --ckpt spF.pt"
run() { n=$1; shift; echo "[tp] s20 run $n: $* $(date -u)"
  python3 sp_qwen3.py rollout $G "$@" --roll-out g96_$n.jsonl > run_g96_$n.log 2>&1
  grep -E "Error|Traceback|out of memory" run_g96_$n.log | tail -2 | sed "s/^/[tp] $n: /"
  python3 sp_mathcheck.py g96_$n.jsonl | sed "s/^/[tp] /"; put g96_$n.jsonl gen96_$n.jsonl; put $LG spP.log; }
run F       --roll-batch 48 $SP
run F_hb03  --roll-batch 48 $SP --hes-bias -0.3
python3 sp_mathcheck.py g96_*.jsonl | sed "s/^/[tp] final /"
put $LG spP.log; date -u > DONE_TF3; put DONE_TF3 DONE_TF3; echo "[tp] DONE_TF3 $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job20.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s20: stage F (full pass) queued after the generation-setting search $(date -u)"
fi
# serial 19: the stop drive is not the gap - on the teacher traces p(</think>) matches (1.514e-4 both, ends certain) but the
# "keep checking" openers (Wait/But/Alternatively/Hmm) do: teacher 0.0110, SP 0.0114 (+4%), window alone 0.0100 (-9%) - the
# order of the generation lengths (SP 9.5k > window 5.9k chars). After E's core runs (E, rawFT_E, full, E_eb1): the same
# numbers on the models' own generations, then a hesitation bias while thinking (-0.3, -1.0), then the remaining knobs.
if [ ! -e /root/.job19 ]; then touch /root/.job19
  cat > /root/job19.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 90; done; echo "[hub] $2 FAILED"; }
while [ ! -e run_g96_E_C32.log ] && [ ! -e DONE_TG16 ]; do sleep 5; done
J=$(pgrep -f "^bash /root/job18.sh"); [ -n "$J" ] && { pkill -P $J; kill $J; }; sleep 3; pkill -f "roll-out g96_E_C32.jsonl"; sleep 5; rm -f g96_E_C32.jsonl DONE_TG16
echo "[tp] s19: job18 stopped after E_eb1 $(date -u)"
RAWB=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/tools
curl -sS -o sp_qwen3.py "$RAWB/sp_qwen3.py?nocache=$(date +%s)"; grep -q "hes_bias" sp_qwen3.py || echo "[tp] s19: WARNING trainer without --hes-bias"
LG=/root/w/sp.log; F="--S 32 --heads 8 --layers 3 --ffn 2048"
for m in "SP@g96_SP.jsonl:--rw 1024 --C 128 --maxd 384 --ckpt spS.pt" "E@g96_E.jsonl:--rw 1024 --C 128 --maxd 384 --ckpt spE.pt" "raw@g96_raw.jsonl:--no-sp --rw 1056 --C 128 --maxd 0"; do
  k=${m%%:*}; g=${k#*@}; [ -s $g ] || continue
  python3 sp_qwen3.py eval --traces $g --max-len 12288 --end-stat $F ${m#*:} --log $LG 2>&1 | grep -E "^\[sp\] end-stat|Error|Traceback" | sed "s/^/[tp] own-drive ${k}: /"; done
put $LG spP.log
G="--traces math96.jsonl --n 96 --max-len 12288 $F --log $LG"; SP="--rw 1024 --C 128 --maxd 384 --ckpt spE.pt"
run() { n=$1; shift; echo "[tp] s19 run $n: $* $(date -u)"
  python3 sp_qwen3.py rollout $G "$@" --roll-out g96_$n.jsonl > run_g96_$n.log 2>&1
  grep -E "Error|Traceback|out of memory" run_g96_$n.log | tail -2 | sed "s/^/[tp] $n: /"
  python3 sp_mathcheck.py g96_$n.jsonl | sed "s/^/[tp] /"; put g96_$n.jsonl gen96_$n.jsonl; put $LG spP.log; }
run E_hb03   --roll-batch 48 $SP --hes-bias -0.3
run E_hb1    --roll-batch 48 $SP --hes-bias -1.0
run E_C32    --roll-batch 48 --rw 1024 --C 32 --maxd 384 --ckpt spE.pt
run E_all    --roll-batch 48 --rw 1024 --C 128 --maxd 0 --ckpt spE.pt
run E_w2k    --roll-batch 32 --rw 2048 --C 128 --maxd 384 --ckpt spE.pt
run E_pp1    --roll-batch 48 $SP --presence 1.0
run E_t08    --roll-batch 48 $SP --temp 0.8
python3 sp_mathcheck.py g96_*.jsonl | sed "s/^/[tp] final /"
put $LG spP.log; date -u > DONE_TG16; put DONE_TG16 DONE_TG16; echo "[tp] DONE_TG16 $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job19.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s19: own-drive + hesitation bias queued after E's core runs $(date -u)"
fi
# serial 18: the --par data never showed an ending: traces under rw+C+C tokens were skipped, the rest cut at 3072, only whole
# blocks scored, and the plain window from the start (the deployment's first 1152 tokens, FFT weights, no SP) never trained -
# and the window alone with the FFT weights already writes ~1.5x longer than with the base weights. Stage E, after the
# stop-drive numbers and the +1 end-bias run: from S on unseen teacher traces, whole traces (16384), --end-train (short
# traces, tails, one block holding </think>, the plain block from the start) and --end-weight 4 around the end; then the
# scores, the stop drive, and the generation check; the remaining knobs run on E.
if [ ! -e /root/.job18 ]; then touch /root/.job18
  cat > /root/job18.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 90; done; echo "[hub] $2 FAILED"; }
while [ ! -e run_g96_SP_eb2.log ] && [ ! -e DONE_TG16 ]; do sleep 5; done
J=$(pgrep -f "^bash /root/job17.sh"); [ -n "$J" ] && { pkill -P $J; kill $J; }; sleep 3; pkill -f "roll-out g96_SP_eb2.jsonl"; sleep 5; rm -f g96_SP_eb2.jsonl
echo "[tp] s18: job17 stopped after SP_eb1 $(date -u)"
RAWB=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/tools
curl -sS -o sp_qwen3.py "$RAWB/sp_qwen3.py?nocache=$(date +%s)"; grep -q "end_train" sp_qwen3.py || echo "[tp] s18: WARNING trainer without --end-train"
tail -n +2401 trainS.jsonl > trainE.jsonl
python3 - <<'PY' | sed "s/^/[tp] s18: /"
import json
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B"); rs = [json.loads(l) for l in open("trainE.jsonl")][:1200]
L = sorted(len(tk.encode(r.get("text", ""), add_special_tokens=False)) for r in rs); e = sum("</think>" in r.get("text", "") for r in rs)
print(f"trainE first 1200: {e} with </think>, tokens median {L[len(L)//2]} p90 {L[int(len(L)*.9)]} max {L[-1]}, under 1280: {sum(x < 1280 for x in L)}")
PY
LG=/root/w/sp.log; F="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 384"
echo "[tp] stage E: end-aware FFT from S $(date -u)"
python3 sp_qwen3.py train --traces trainE.jsonl --n 1200 --max-len 16384 $F --ckpt spS.pt --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 1e-4 --par --end-train --end-weight 4 --chunks-per-trace 4 --accum 8 --out spE.pt --log $LG > run_trainE.log 2>&1
grep -E "^\[sp\] (trained)|Error|Traceback|out of memory" run_trainE.log | tail -3 | sed "s/^/[tp] E: /"
put spE.pt spE_endaware_fft1024.pt
for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 $F --ckpt spE.pt --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] E stageA-cond: /"; done
for m in "SP_E:$F --ckpt spE.pt" "rawFT_E:--no-sp --rw 1056 --C 128 --maxd 0 --S 32 --heads 8 --layers 3 --ffn 2048 --ckpt spE.pt"; do
  python3 sp_qwen3.py eval --traces kv_full_p.jsonl --max-len 16384 --end-stat ${m#*:} --log $LG 2>&1 | grep -E "^\[sp\] (eval|end-stat)|Error|Traceback" | sed "s/^/[tp] stop-drive ${m%%:*}: /"; done
put $LG spP.log
G="--traces math96.jsonl --n 96 --max-len 12288 --S 32 --heads 8 --layers 3 --ffn 2048 --log $LG"; SP="--rw 1024 --C 128 --maxd 384 --ckpt spE.pt"
run() { n=$1; shift; echo "[tp] s18 run $n: $* $(date -u)"
  python3 sp_qwen3.py rollout $G "$@" --roll-out g96_$n.jsonl > run_g96_$n.log 2>&1
  grep -E "Error|Traceback|out of memory" run_g96_$n.log | tail -2 | sed "s/^/[tp] $n: /"
  python3 sp_mathcheck.py g96_$n.jsonl | sed "s/^/[tp] /"; put g96_$n.jsonl gen96_$n.jsonl; put $LG spP.log; }
run E        --roll-batch 48 $SP
run rawFT_E  --roll-batch 48 --no-sp --rw 1056 --C 128 --maxd 0 --ckpt spE.pt
run full     --roll-batch 8 --no-sp --rw 20000 --C 12288 --maxd 0
run E_eb1    --roll-batch 48 $SP --end-bias 1.0
run E_C32    --roll-batch 48 --rw 1024 --C 32 --maxd 384 --ckpt spE.pt
run E_all    --roll-batch 48 --rw 1024 --C 128 --maxd 0 --ckpt spE.pt
run E_w2k    --roll-batch 32 --rw 2048 --C 128 --maxd 384 --ckpt spE.pt
run E_pp1    --roll-batch 48 $SP --presence 1.0
run E_t08    --roll-batch 48 $SP --temp 0.8
python3 sp_mathcheck.py g96_*.jsonl | sed "s/^/[tp] final /"
put $LG spP.log; date -u > DONE_TG16; put DONE_TG16 DONE_TG16; echo "[tp] DONE_TG16 $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job18.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s18: stage E queued after the SP_eb1 run $(date -u)"
fi
# serial 17: on 96 problems SP 65/96 vs the window alone 77/96 (paired 1 vs 13): SP thoughts do not end - 25 left open at
# 12288, 19 of them closed by the window alone; the texts loop on re-verification ("all three roots are valid. But just to
# make sure..."). Mean KL hides it: stopping is a handful of tokens. Re-order the search once the rawFT run is done: first
# measure the stopping drive (--end-stat: mean p(</think>), log p at the real ends, mean p of Wait/But/Alternatively/Hmm,
# student vs teacher, on the MATH teacher traces to 16384), then SP with a </think> logit bias (+1, +2), then full KV and the
# remaining knobs.
if [ ! -e /root/.job17 ]; then touch /root/.job17
  cat > /root/job17.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 90; done; echo "[hub] $2 FAILED"; }
while [ ! -e run_g96_full.log ] && [ ! -e DONE_TG16 ]; do sleep 5; done
J=$(pgrep -f "^bash /root/job16.sh"); [ -n "$J" ] && { pkill -P $J; kill $J; }; sleep 3; pkill -f "roll-out g96_full.jsonl"; sleep 5; rm -f g96_full.jsonl
echo "[tp] s17: job16 stopped after rawFT $(date -u)"
RAWB=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/tools
curl -sS -o sp_qwen3.py "$RAWB/sp_qwen3.py?nocache=$(date +%s)"; grep -q "end_stat" sp_qwen3.py || echo "[tp] s17: WARNING trainer without --end-stat"
LG=/root/w/sp.log; F="--S 32 --heads 8 --layers 3 --ffn 2048"
for m in "SP:--rw 1024 --C 128 --maxd 384 --ckpt spS.pt" "raw:--no-sp --rw 1056 --C 128 --maxd 0" "rawFT:--no-sp --rw 1056 --C 128 --maxd 0 --ckpt spS.pt"; do
  python3 sp_qwen3.py eval --traces kv_full_p.jsonl --max-len 16384 --end-stat $F ${m#*:} --log $LG 2>&1 | grep -E "^\[sp\] (eval|end-stat)|Error|Traceback" | sed "s/^/[tp] stop-drive ${m%%:*}: /"; done
put $LG spP.log
G="--traces math96.jsonl --n 96 --max-len 12288 $F --log $LG"; SP="--rw 1024 --C 128 --maxd 384 --ckpt spS.pt"
run() { n=$1; shift; echo "[tp] s17 run $n: $* $(date -u)"
  python3 sp_qwen3.py rollout $G "$@" --roll-out g96_$n.jsonl > run_g96_$n.log 2>&1
  grep -E "Error|Traceback|out of memory" run_g96_$n.log | tail -2 | sed "s/^/[tp] $n: /"
  python3 sp_mathcheck.py g96_$n.jsonl | sed "s/^/[tp] /"; put g96_$n.jsonl gen96_$n.jsonl; put $LG spP.log; }
run SP_eb1  --roll-batch 48 $SP --end-bias 1.0
run SP_eb2  --roll-batch 48 $SP --end-bias 2.0
run full    --roll-batch 8 --no-sp --rw 20000 --C 12288 --maxd 0
run SP_C32  --roll-batch 48 --rw 1024 --C 32 --maxd 384 --ckpt spS.pt
run SP_all  --roll-batch 48 --rw 1024 --C 128 --maxd 0 --ckpt spS.pt
run SP_w2k  --roll-batch 32 --rw 2048 --C 128 --maxd 384 --ckpt spS.pt
run SP_pp1  --roll-batch 48 $SP --presence 1.0
run SP_t08  --roll-batch 48 $SP --temp 0.8
python3 sp_mathcheck.py g96_*.jsonl | sed "s/^/[tp] final /"
put $LG spP.log; date -u > DONE_TG16; put DONE_TG16 DONE_TG16; echo "[tp] DONE_TG16 $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job17.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s17: re-ordered search queued after the rawFT run $(date -u)"
fi
# serial 16: generation-setting search for the hypernet (replaces serial 15's check), after the FFT-to-plateau run: 96 MATH-500
# problems (levels 2-4), 12288 tokens, one sample each, paired per problem. Core first: SP (best checkpoint, deployment setting),
# the window alone (base weights), the window alone with the FFT weights (does the tuned model itself behave differently?),
# full KV. Then one knob at a time on SP: refresh every 32 tokens, no eviction (the pooler reads everything, like ap32),
# window 2048, presence penalty 1.0, temperature 0.8.
if [ ! -e /root/.job16 ]; then touch /root/.job16
  J=$(pgrep -f "^bash /root/job15.sh"); [ -n "$J" ] && { pkill -P $J; kill $J; echo "[tp] s16: replaced the queued serial-15 check"; }
  curl -sS -o /root/w/math96.jsonl "$RAW/ctl/math96.jsonl?nocache=$(date +%s)"
  cat > /root/job16.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 90; done; echo "[hub] $2 FAILED"; }
while [ ! -e DONE_TS ]; do sleep 2; done
J=$(pgrep -f "^bash /root/job13.sh"); [ -n "$J" ] && { pkill -P $J; kill $J; }; sleep 3; pkill -f "roll-out g16_"; sleep 3
RAWB=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/tools
curl -sS -o sp_qwen3.py "$RAWB/sp_qwen3.py?nocache=$(date +%s)"; grep -q "a.presence" sp_qwen3.py || echo "[tp] s16: WARNING trainer without --presence"
echo "[tp] s16: spS.pt $(ls -la spS.pt | awk '{print $5}') bytes $(date -u)"; put spS.pt spS_fft_plateau.pt; put DONE_TS DONE_TS
LG=/root/w/sp.log; G="--traces math96.jsonl --n 96 --max-len 12288 --S 32 --heads 8 --layers 3 --ffn 2048 --log $LG"
SP="--rw 1024 --C 128 --maxd 384 --ckpt spS.pt"
run() { n=$1; shift; echo "[tp] s16 run $n: $* $(date -u)"
  python3 sp_qwen3.py rollout $G "$@" --roll-out g96_$n.jsonl > run_g96_$n.log 2>&1
  grep -E "Error|Traceback|out of memory" run_g96_$n.log | tail -2 | sed "s/^/[tp] $n: /"
  python3 sp_mathcheck.py g96_$n.jsonl | sed "s/^/[tp] /"; put g96_$n.jsonl gen96_$n.jsonl; put $LG spP.log; }
run SP      --roll-batch 48 $SP
run raw     --roll-batch 48 --no-sp --rw 1056 --C 128 --maxd 0
run rawFT   --roll-batch 48 --no-sp --rw 1056 --C 128 --maxd 0 --ckpt spS.pt
run full    --roll-batch 8 --no-sp --rw 20000 --C 12288 --maxd 0
run SP_C32  --roll-batch 48 --rw 1024 --C 32 --maxd 384 --ckpt spS.pt
run SP_all  --roll-batch 48 --rw 1024 --C 128 --maxd 0 --ckpt spS.pt
run SP_w2k  --roll-batch 32 --rw 2048 --C 128 --maxd 384 --ckpt spS.pt
run SP_pp1  --roll-batch 48 $SP --presence 1.0
run SP_t08  --roll-batch 48 $SP --temp 0.8
python3 sp_mathcheck.py g96_*.jsonl | sed "s/^/[tp] final /"
put $LG spP.log; date -u > DONE_TG16; put DONE_TG16 DONE_TG16; echo "[tp] DONE_TG16 $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job16.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s16: generation-setting search queued after DONE_TS $(date -u)"
fi
# serial 15: 32 level-4/5 problems at one sample each cannot separate the methods (binomial spread ~3 problems, and even full
# KV leaves a third open at 8192). The check after the FFT-to-plateau run is replaced: 96 MATH-500 problems, 32 each of levels
# 2/3/4 (ctl/math96.jsonl), 16384 tokens, paired per problem: SP (best checkpoint), the window alone, full KV (plain
# incremental decoding: one 16384-token chunk, no re-prefill).
if [ ! -e /root/.job15 ] && [ ! -e /root/.job16 ]; then touch /root/.job15
  curl -sS -o /root/w/math96.jsonl "$RAW/ctl/math96.jsonl?nocache=$(date +%s)"; echo "[tp] s15: math96 $(wc -l < /root/w/math96.jsonl) rows"
  cat > /root/job15.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 90; done; echo "[hub] $2 FAILED"; }
while [ ! -e DONE_TS ]; do sleep 2; done
J=$(pgrep -f "^bash /root/job13.sh"); [ -n "$J" ] && { pkill -P $J; kill $J; }; sleep 3; pkill -f "roll-out g16_" ; sleep 3
echo "[tp] s15: old 16k check stopped; spS.pt $(ls -la spS.pt | awk '{print $5}') bytes $(date -u)"
put spS.pt spS_fft_plateau.pt; put DONE_TS DONE_TS
LG=/root/w/sp.log; G="--traces math96.jsonl --n 96 --max-len 16384 --S 32 --heads 8 --layers 3 --ffn 2048 --log $LG"
echo "[tp] s15: generation check, 96 problems, 16384 tokens $(date -u)"
python3 sp_qwen3.py rollout $G --roll-batch 16 --rw 1024 --C 128 --maxd 384 --ckpt spS.pt --roll-out g96_S.jsonl > run_g96.log 2>&1
python3 sp_mathcheck.py g96_S.jsonl | sed "s/^/[tp] /"; put g96_S.jsonl gen96_S.jsonl
python3 sp_qwen3.py rollout $G --roll-batch 16 --no-sp --rw 1056 --C 128 --maxd 0 --roll-out g96_raw1056.jsonl >> run_g96.log 2>&1
python3 sp_mathcheck.py g96_raw1056.jsonl | sed "s/^/[tp] /"; put g96_raw1056.jsonl gen96_raw1056.jsonl
python3 sp_qwen3.py rollout $G --roll-batch 6 --no-sp --rw 20000 --C 16384 --maxd 0 --roll-out g96_full.jsonl >> run_g96.log 2>&1
grep -E "Error|Traceback|out of memory" run_g96.log | tail -3 | sed "s/^/[tp] /"
python3 sp_mathcheck.py g96_full.jsonl g96_S.jsonl g96_raw1056.jsonl | sed "s/^/[tp] /"
put g96_full.jsonl gen96_full.jsonl; put run_g96.log gen96_run.log; put $LG spP.log; date -u > DONE_TG16; put DONE_TG16 DONE_TG16; echo "[tp] DONE_TG16 $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job15.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s15: 96-problem generation check queued after DONE_TS $(date -u)"
fi
# serial 14: serial 12's 16k rollout had started (01:26) before serial 13 stopped its shell - stop that rollout if it survived,
# and list what runs on the GPU
if [ ! -e /root/.chk14 ]; then touch /root/.chk14
  pkill -f "roll-out g16_D1.jsonl" && echo "[tp] s14: stopped the stray serial-12 rollout"
  sleep 5; ps -eo pid,etimes,args | grep "sp_qwen3.py" | grep -v grep | cut -c1-200 | sed "s/^/[tp] s14 proc: /"
fi
# serial 13: keep FFT going until the loss stops falling. From D1 (DAgger) on unseen teacher traces (trainR from line 2385,
# ~4800), 800 per segment, held-out stage-A-condition KL (MATH 64 + mixed 60) after each; stop after two segments without a
# 1% gain on the best (or when the traces run out). The 16k generation check (serial 12) then runs on the best checkpoint.
if [ ! -e /root/.job13 ]; then touch /root/.job13
  J=$(pgrep -f "^bash /root/job12.sh"); [ -n "$J" ] && { pkill -P $J; kill $J; echo "[tp] s13: stopped the queued 16k check ($J)"; }
  cat > /root/job13.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 90; done; echo "[hub] $2 FAILED"; }
while [ ! -e DONE_TD1 ]; do sleep 60; done
LG=/root/w/sp.log; F="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 384"
score() { m=$(python3 sp_qwen3.py eval --traces kv_full_p.jsonl --max-len 4096 $F --ckpt $1 --log $LG 2>&1 | grep -oP "KL \K[0-9.]+" | tail -1)
  x=$(python3 sp_qwen3.py eval --traces traces.jsonl --n 60 --max-len 4096 $F --ckpt $1 --log $LG 2>&1 | grep -oP "KL \K[0-9.]+" | tail -1)
  echo "$m $x"; }
tail -n +2385 trainR.jsonl > trainS.jsonl; NT=$(wc -l < trainS.jsonl); echo "[tp] s13: $NT unseen teacher traces"
cp spD1.pt cur.pt; best=$(python3 -c "print((0.0699+0.0613)/2)"); cp spD1.pt best.pt; bad=0; seg=0; off=0
while [ $off -lt $NT ] && [ $bad -lt 2 ]; do seg=$((seg+1))
  python3 sp_qwen3.py train --traces trainS.jsonl --skip $off --n 800 --max-len 3072 $F --ckpt cur.pt --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 1e-4 --par --chunks-per-trace 4 --accum 8 --out nxt.pt --log $LG > run_trainS$seg.log 2>&1
  grep -E "Error|Traceback" run_trainS$seg.log | tail -2 | sed "s/^/[tp] S$seg: /"
  [ -s nxt.pt ] || { echo "[tp] S$seg: no checkpoint, stop"; break; }
  mv nxt.pt cur.pt; off=$((off+800)); read m x < <(score cur.pt); v=$(python3 -c "print(($m+$x)/2)")
  echo "[tp] S$seg (traces $off): MATH $m mixed $x mean $v (best $best)"
  if python3 -c "import sys; sys.exit(0 if $v < $best * 0.99 else 1)"; then best=$v; cp cur.pt best.pt; bad=0; echo "[tp] S$seg: new best"
  else bad=$((bad+1)); python3 -c "import sys; sys.exit(0 if $v < $best else 1)" && { best=$v; cp cur.pt best.pt; }; fi
  put $LG spP.log
done
echo "[tp] s13: stopped after $seg segments, best mean $best"
cp best.pt spS.pt; put spS.pt spS_fft_plateau.pt
python3 sp_qwen3.py profile --traces kv_full_p.jsonl --n 16 --max-len 4096 $F --ckpt spS.pt --log $LG 2>&1 | grep -E "^\[sp\] profile|Error|Traceback" | tail -7 | sed "s/^/[math] S /"
python3 sp_qwen3.py eval --traces g8_R2.jsonl --max-len 8192 --score-from 1184 $F --ckpt spS.pt --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[tp] own-KL on g8_R2.jsonl: S: /"
put $LG spP.log; date -u > DONE_TS; put DONE_TS DONE_TS; echo "[tp] DONE_TS $(date -u)"
G="--traces kv_full_p.jsonl --n 32 --max-len 16384 --S 32 --heads 8 --layers 3 --ffn 2048 --log $LG"
echo "[tp] s13: generation check at 16384 tokens $(date -u)"
python3 sp_qwen3.py rollout $G --roll-batch 16 --rw 1024 --C 128 --maxd 384 --ckpt spS.pt --roll-out g16_S.jsonl > run_g16.log 2>&1
python3 sp_mathcheck.py g16_S.jsonl | sed "s/^/[tp] /"; put g16_S.jsonl ap32gen16_S.jsonl
python3 sp_qwen3.py rollout $G --roll-batch 16 --no-sp --rw 1056 --C 128 --maxd 0 --roll-out g16_raw1056.jsonl >> run_g16.log 2>&1
python3 sp_mathcheck.py g16_raw1056.jsonl | sed "s/^/[tp] /"; put g16_raw1056.jsonl ap32gen16_raw1056.jsonl
python3 sp_qwen3.py rollout $G --roll-batch 4 --no-sp --rw 20000 --C 512 --maxd 0 --roll-out g16_full.jsonl >> run_g16.log 2>&1
grep -E "Error|Traceback|out of memory" run_g16.log | tail -3 | sed "s/^/[tp] /"
python3 sp_mathcheck.py g16_full.jsonl g16_S.jsonl g16_raw1056.jsonl | sed "s/^/[tp] /"
put g16_full.jsonl ap32gen16_full.jsonl; put run_g16.log gen16_run.log; put $LG spP.log; date -u > DONE_TG16; put DONE_TG16 DONE_TG16; echo "[tp] DONE_TG16 $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job13.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s13: FFT-to-plateau launched (waits for DONE_TD1) $(date -u)"
fi
# serial 12: at 8192 tokens even full KV leaves 12/32 thoughts open (18/32 correct); the window alone "wins" (20/32) by
# writing half as much (median 9.5k vs 16.7k chars) and SP loses mostly by finishing later (closed-answer accuracy 87% vs
# 90%). After DAgger, the check at 16384 tokens: full KV, D1 (SP at the deployment setting) and the window alone.
if [ ! -e /root/.job12 ] && [ ! -e /root/.job13 ]; then touch /root/.job12
  cat > /root/job12.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 90; done; echo "[hub] $2 FAILED"; }
while [ ! -e DONE_TD1 ]; do sleep 60; done
LG=/root/w/sp.log; G="--traces kv_full_p.jsonl --n 32 --max-len 16384 --S 32 --heads 8 --layers 3 --ffn 2048 --log $LG"
echo "[tp] s12: generation check at 16384 tokens $(date -u)"
python3 sp_qwen3.py rollout $G --roll-batch 16 --rw 1024 --C 128 --maxd 384 --ckpt spD1.pt --roll-out g16_D1.jsonl > run_g16.log 2>&1
python3 sp_mathcheck.py g16_D1.jsonl | sed "s/^/[tp] /"; put g16_D1.jsonl ap32gen16_D1.jsonl
python3 sp_qwen3.py rollout $G --roll-batch 16 --no-sp --rw 1056 --C 128 --maxd 0 --roll-out g16_raw1056.jsonl >> run_g16.log 2>&1
python3 sp_mathcheck.py g16_raw1056.jsonl | sed "s/^/[tp] /"; put g16_raw1056.jsonl ap32gen16_raw1056.jsonl
python3 sp_qwen3.py rollout $G --roll-batch 4 --no-sp --rw 20000 --C 512 --maxd 0 --roll-out g16_full.jsonl >> run_g16.log 2>&1
grep -E "Error|Traceback|out of memory" run_g16.log | tail -3 | sed "s/^/[tp] /"
python3 sp_mathcheck.py g16_full.jsonl g16_D1.jsonl g16_raw1056.jsonl | sed "s/^/[tp] /"
put g16_full.jsonl ap32gen16_full.jsonl; put run_g16.log gen16_run.log; put $LG spP.log; date -u > DONE_TG16; put DONE_TG16 DONE_TG16; echo "[tp] DONE_TG16 $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job12.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  sed -e 's/DONE_TR /DONE_TG16 /' /root/live8.sh > /root/live12.sh; pkill -f live11.sh; setsid nohup bash -c 'bash /root/live12.sh' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s12: 16k generation check queued after DONE_TD1 $(date -u)"
fi
# serial 11: R2 beats every KL so far (0.0669 / 0.0585) yet generates worse than the raw window alone (13/32 vs 20/32, SP
# answers ~2x longer, fewer closed thoughts) - the teacher-forced KL misses what happens on the student's own path. (a) KL on
# the rollouts themselves: R2 and the window alone, each on its own and on the other's generations (all scored from 1184).
# (b) DAgger, the step the ap32 path took next: R2 generates on unseen training prompts (4096 tokens), the frozen full-context
# model labels those states, --par FFT on them; then the same 8192-token generation check.
if [ ! -e /root/.job11 ]; then touch /root/.job11
  cat > /root/job11.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 90; done; echo "[hub] $2 FAILED"; }
H=https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main
RAWB=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/tools
curl -sS -o sp_qwen3.py "$RAWB/sp_qwen3.py?nocache=$(date +%s)"
[ -s g8_raw1056.jsonl ] || curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o g8_raw1056.jsonl "$H/archive/sp3/ap32gen8_raw1056.jsonl"
LG=/root/w/sp.log; F="--S 32 --heads 8 --layers 3 --ffn 2048 --C 128 --maxd 384"; E="--max-len 8192 --score-from 1184"
echo "[tp] s11: KL on the rollouts $(date -u)"
for g in g8_R2.jsonl g8_raw1056.jsonl; do
  python3 sp_qwen3.py eval --traces $g $E $F --rw 1024 --ckpt spR2.pt --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[tp] own-KL on $g: R2: /"
  python3 sp_qwen3.py eval --traces $g $E $F --rw 1056 --no-sp --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[tp] own-KL on $g: raw1056: /"
done
put $LG spP.log
echo "[tp] s11: DAgger rollouts $(date -u)"
tail -n +2001 trainR.jsonl > dag_p.jsonl; n=$(wc -l < dag_p.jsonl); echo "[tp] s11: $n unseen trainR prompts"
[ $n -lt 384 ] && head -n $((384 - n)) trainR.jsonl >> dag_p.jsonl
python3 sp_qwen3.py rollout --traces dag_p.jsonl --n 384 --max-len 4096 --roll-batch 16 $F --rw 1024 --ckpt spR2.pt --roll-out dag1.jsonl --log $LG > run_dag.log 2>&1
grep -E "Error|Traceback|out of memory" run_dag.log | tail -2 | sed "s/^/[tp] /"; echo "[tp] s11: $(wc -l < dag1.jsonl) DAgger traces"
put dag1.jsonl dag1_R2rollouts.jsonl
python3 sp_qwen3.py train --traces dag1.jsonl --max-len 4096 $F --rw 1024 --ckpt spR2.pt --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 1e-4 --par --chunks-per-trace 4 --accum 8 --out spD1.pt --log $LG > run_trainD1.log 2>&1
grep -E "^\[sp\] (trained)|Error|Traceback|out of memory" run_trainD1.log | tail -3 | sed "s/^/[tp] D1: /"
put spD1.pt spD1_dagger_fft1024.pt
for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 $F --rw 1024 --ckpt spD1.pt --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] D1 stageA-cond: /"; done
python3 sp_qwen3.py eval --traces g8_R2.jsonl $E $F --rw 1024 --ckpt spD1.pt --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[tp] own-KL on g8_R2.jsonl: D1: /"
put $LG spP.log
python3 sp_qwen3.py rollout --traces kv_full_p.jsonl --n 32 --max-len 8192 --roll-batch 16 $F --rw 1024 --ckpt spD1.pt --roll-out g8_D1.jsonl --log $LG > run_gD1.log 2>&1
grep -E "Error|Traceback|out of memory" run_gD1.log | tail -2 | sed "s/^/[tp] /"
python3 sp_mathcheck.py g8_D1.jsonl g8_R2.jsonl g8_raw1056.jsonl | sed "s/^/[tp] /"
put g8_D1.jsonl ap32gen8_D1.jsonl; put $LG spP.log; date -u > DONE_TD1; put DONE_TD1 DONE_TD1; echo "[tp] DONE_TD1 $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job11.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  sed -e 's/DONE_TR /DONE_TD1 /' /root/live8.sh > /root/live11.sh; pkill -f live10.sh; setsid nohup bash -c 'bash /root/live11.sh' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s11: rollout KL + DAgger launched $(date -u)"
fi
# serial 10: stage R ran out of memory at ~850 traces (its 800-trace checkpoint already scores 0.0706 / 0.0621 under the stage-A
# conditions - level with the best so far). The trainer now skips a trace that does not fit. R2 = continue from spR.pt on the
# rest (trainR from 851, 1150 traces), then the same scores and the generation check (8192 tokens, rollout without the
# empty-buffer SP) at rw 1024.
if [ ! -e /root/.job10 ]; then touch /root/.job10
  cat > /root/job10.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 90; done; echo "[hub] $2 FAILED"; }
RAWB=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/tools
curl -sS -o sp_qwen3.py "$RAWB/sp_qwen3.py?nocache=$(date +%s)"
grep -q "out of memory, skipped" sp_qwen3.py && grep -q "sp_empty" sp_qwen3.py && echo "[tp] s10: trainer has the OOM skip + rollout fix" || echo "[tp] s10: WARNING old trainer"
LG=/root/w/sp.log; F="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 384"
echo "[tp] stage R2: continue from spR.pt $(date -u)"
python3 sp_qwen3.py train --traces trainR.jsonl --skip 850 --n 1150 --max-len 3072 $F --ckpt spR.pt --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 1e-4 --par --chunks-per-trace 4 --accum 8 --out spR2.pt --log $LG > run_trainR2.log 2>&1
grep -E "^\[sp\] (trained)|Error|Traceback|out of memory" run_trainR2.log | tail -3 | sed "s/^/[tp] R2: /"
put spR2.pt spR2_ap32recipe_fft1024.pt
for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 $F --ckpt spR2.pt --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] R2 stageA-cond: /"; done
put $LG spP.log
G="--traces kv_full_p.jsonl --n 32 --max-len 8192 --roll-batch 16 --S 32 --heads 8 --layers 3 --ffn 2048 --log $LG"
python3 sp_qwen3.py rollout $G --rw 1024 --C 128 --maxd 384 --ckpt spR2.pt --roll-out g8_R2.jsonl > run_gR2.log 2>&1
grep -E "Error|Traceback|out of memory" run_gR2.log | tail -2 | sed "s/^/[tp] /"
python3 sp_mathcheck.py g8_R2.jsonl | sed "s/^/[tp] /"
put g8_R2.jsonl ap32gen8_R2.jsonl; put $LG spP.log; date -u > DONE_TR2; put DONE_TR2 DONE_TR2; echo "[tp] DONE_TR2 $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job10.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  sed -e 's/DONE_TR /DONE_TR2 /' /root/live8.sh > /root/live10.sh; setsid nohup bash -c 'bash /root/live10.sh' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s10: stage R2 launched $(date -u)"
fi
# serial 9: two copies of the 8k generation job are running (the serial-6 one was not stopped before serial 7 relaunched it)
# - stop the newer copy and its current rollout; the shared output files keep both copies' rows (extra samples)
if [ ! -e /root/.dup9 ]; then touch /root/.dup9
  J=$(pgrep -f "^bash /root/job6.sh" | sort -n); echo "[tp] s9: job6 copies: $(echo $J | tr '\n' ' ')"
  if [ $(echo "$J" | wc -w) -ge 2 ]; then P=$(echo "$J" | tail -1); pkill -P $P; kill $P; echo "[tp] s9: stopped copy $P"; fi
fi
# serial 8: stage Q (FFT at rw 128) improved rw 128 slightly but HURT the deployment window (stage-A conditions 0.084 / 0.071
# -> 0.096 / 0.088): the model specialised to the narrow window. Stage R = the frozen recipe pooler (spP_orig) + all-layer FFT
# AT the deployment window, the way stage B did it (--par: every window 0..rw+C at 4 boundaries per trace, rw 1024, C 128,
# maxD 384, 8 traces per update), on traces nobody has seen (trainQ from line 1501), after the 8k generation check.
if [ ! -e /root/.job8 ]; then touch /root/.job8
  cat > /root/job8.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 90; done; echo "[hub] $2 FAILED"; }
while [ ! -e DONE_TG8 ]; do sleep 30; done
LG=/root/w/sp.log; F="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 384"
echo "[tp] stage R: recipe pooler + FFT at rw 1024 (--par) $(date -u)"
tail -n +1501 trainQ.jsonl > trainR.jsonl
python3 sp_qwen3.py train --traces trainR.jsonl --n 2000 --max-len 3072 $F --ckpt spP_orig.pt --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 1e-4 --par --chunks-per-trace 4 --accum 8 --out spR.pt --log $LG > run_trainR.log 2>&1
grep -E "^\[sp\] (trained)|Error|Traceback|out of memory" run_trainR.log | tail -3 | sed "s/^/[tp] R: /"
put spR.pt spR_ap32recipe_fft1024.pt
for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 $F --ckpt spR.pt --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] R stageA-cond: /"; done
python3 sp_qwen3.py profile --traces kv_full_p.jsonl --n 16 --max-len 4096 $F --ckpt spR.pt --log $LG 2>&1 | grep -E "^\[sp\] profile|Error|Traceback" | tail -7 | sed "s/^/[math] R /"
put $LG spP.log; date -u > DONE_TR; put DONE_TR DONE_TR; echo "[tp] DONE_TR $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job8.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  cat > /root/live8.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
while [ ! -e DONE_TR ]; do
  { grep -E "^\[(tp|hub|math|mix)\]" w.log | tail -n 24; tail -n 8 sp.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; date -u; } > tp_live.log 2>&1
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='tp_live.log', path_in_repo='archive/sp3/tp_live.log', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1
  sleep 600
done
JOB
  pkill -f live5.sh; setsid nohup bash -c 'bash /root/live8.sh' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s8: stage R queued after the 8k generation check; live log until DONE_TR $(date -u)"
fi
# serial 7: relaunch the (still waiting) 8k generation job with the full-KV run in batches of 8
if [ ! -e /root/.job7 ]; then touch /root/.job7
  pkill -f job6.sh; sleep 2
  sed -i 's#python3 sp_qwen3.py rollout $G --no-sp --rw 16384#python3 sp_qwen3.py rollout $G --roll-batch 8 --no-sp --rw 16384#' /root/job6.sh
  grep -q -- "--roll-batch 8 --no-sp --rw 16384" /root/job6.sh && echo "[tp] s7: job6 patched" || echo "[tp] s7: WARNING job6 not patched"
  setsid nohup bash -c 'bash /root/job6.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s7: 8k generation job relaunched $(date -u)"
fi
# serial 6: the 4096-token generation check is not usable (full KV left 19/32 thoughts unfinished; the window-only path
# stops early and "wins") - after stage Q, generate to 8192 tokens: full KV, SP (stage Q weights) at the deployment setting
# (rw 1024, C 128, maxD 384), the frozen recipe pooler the same way, and the equal-budget window alone (1056).
if [ ! -e /root/.job6 ]; then touch /root/.job6
  cat > /root/job6.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 90; done; echo "[hub] $2 FAILED"; }
while [ ! -e DONE_TQ ]; do sleep 30; done
LG=/root/w/sp.log; G="--traces kv_full_p.jsonl --n 32 --max-len 8192 --roll-batch 16 --S 32 --heads 8 --layers 3 --ffn 2048 --log $LG"
echo "[tp] generation check at 8192 tokens $(date -u)"
python3 sp_qwen3.py rollout $G --roll-batch 8 --no-sp --rw 16384 --C 512 --maxd 0 --roll-out g8_full.jsonl > run_g8.log 2>&1
python3 sp_qwen3.py rollout $G --rw 1024 --C 128 --maxd 384 --ckpt spQ.pt --roll-out g8_Q1024.jsonl >> run_g8.log 2>&1
python3 sp_qwen3.py rollout $G --rw 1024 --C 128 --maxd 384 --ckpt spP_orig.pt --roll-out g8_P1024.jsonl >> run_g8.log 2>&1
python3 sp_qwen3.py rollout $G --no-sp --rw 1056 --C 128 --maxd 0 --roll-out g8_raw1056.jsonl >> run_g8.log 2>&1
grep -E "Error|Traceback|out of memory" run_g8.log | tail -2
python3 sp_mathcheck.py g8_full.jsonl g8_Q1024.jsonl g8_P1024.jsonl g8_raw1056.jsonl | sed "s/^/[tp] /"
for g in full Q1024 P1024 raw1056; do put g8_$g.jsonl ap32gen8_$g.jsonl; done
put $LG spP.log; date -u > DONE_TG8; put DONE_TG8 DONE_TG8; echo "[tp] DONE_TG8 $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job6.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  sed -i 's/while \[ ! -e DONE_TQ \]/while [ ! -e DONE_TG8 ]/' /root/live5.sh 2>/dev/null
  echo "[tp] s6: 8192-token generation check queued $(date -u)"
fi
# serial 5: the live log stopped at DONE_TP - restart it until stage Q is done, following the FFT run's log
if [ ! -e /root/.live5 ]; then touch /root/.live5
  sed -e 's/while \[ ! -e DONE_TP \]/while [ ! -e DONE_TQ ]/' -e 's#tail -n 4 spP_orig.log; tail -n 4 spP_norm.log; tail -n 8 sp.log;#tail -n 12 sp.log;#' /root/live.sh > /root/live5.sh
  setsid nohup bash -c 'bash /root/live5.sh' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s5: live log restarted until DONE_TQ"
fi
# serial 4 (the user: if the loss does not get low enough with the model frozen, unfreeze it): after the frozen-model
# results (DONE_TP2), stage Q = the better recipe pooler + all-layer FFT (fp32 master, 8-bit AdamW), the same rw 128 / C 64,
# 3072-token contexts (distant up to ~2.9k, cap 4096 = no eviction in practice), 8 chunks per trace, 1500 traces nobody has
# seen (shards 2-5), teacher = the original model; then the same scores (deep KL at rw 128, rw 512, stage-A conditions) and
# the generation check (MATH 32: SP + rw 512 / 1024 vs full KV).
if [ ! -e /root/.job4 ]; then touch /root/.job4
  cat > /root/job4.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 90; done; echo "[hub] $2 FAILED"; }
while [ ! -e DONE_TP2 ]; do sleep 30; done
LG=/root/w/sp.log; L3="--S 32 --heads 8 --layers 3 --ffn 2048"
B=$(python3 - <<'PY'
import re
L=open("/root/w/sp.log").read()
def k(c):
    m=re.findall(r"%s: 64 traces \d+ tokens  KL ([0-9.]+)" % c, L); return float(m[0]) if m else 9
print("spP_orig.pt" if k("spP_orig.pt") <= k("spP_norm.pt") else "spP_norm.pt")
PY
)
PE=""; [ "$B" = spP_norm.pt ] && PE="--pe-rev --pe-norm"
echo "[tp] stage Q: FFT + pooler from $B $PE $(date -u)"
cat traces_big_2.jsonl traces_big_3.jsonl traces_big_4.jsonl traces_big_5.jsonl > trainQ.jsonl
python3 sp_qwen3.py train --traces trainQ.jsonl --n 1500 --max-len 3072 $L3 --rw 128 --C 64 --maxd 4096 --ckpt $B $PE --ft-layers 28 --opt8bit --ft-lr 1e-5 --lr 1e-4 --chunks-per-trace 8 --accum 1 --out spQ.pt --log $LG > run_trainQ.log 2>&1
grep -E "^\[sp\] (trained)|Error|Traceback|out of memory" run_trainQ.log | tail -3 | sed "s/^/[tp] Q: /"
put spQ.pt spQ_ap32recipe_fft.pt; put $LG spP.log
for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces $f $x --max-len 768 --score-from 224 --rw 128 $L3 --C 64 --maxd 0 --ckpt spQ.pt --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] Q rw128: /"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 --score-from 608 --rw 512 $L3 --C 64 --maxd 4096 --ckpt spQ.pt --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] Q rw512: /"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 --rw 1024 $L3 --C 128 --maxd 384 --ckpt spQ.pt --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] Q stageA-cond: /"; done
put $LG spP.log
G="--traces kv_full_p.jsonl --n 32 --max-len 4096 --roll-batch 16 $L3 --log $LG"
python3 sp_qwen3.py rollout $G --rw 512 --C 64 --maxd 4096 --ckpt spQ.pt --roll-out genQ_sp512.jsonl > run_gq.log 2>&1
python3 sp_qwen3.py rollout $G --rw 1024 --C 128 --maxd 384 --ckpt spQ.pt --roll-out genQ_sp1024.jsonl >> run_gq.log 2>&1
grep -E "Error|Traceback|out of memory" run_gq.log | tail -2
python3 sp_mathcheck.py gen_full.jsonl genQ_sp512.jsonl genQ_sp1024.jsonl | sed "s/^/[tp] /"
for g in sp512 sp1024; do put genQ_$g.jsonl ap32genQ_$g.jsonl; done
put $LG spP.log; date -u > DONE_TQ; put DONE_TQ DONE_TQ; echo "[tp] DONE_TQ $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job4.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s4: stage Q (FFT) queued after the frozen-model results $(date -u)"
fi
# serial 3 (the user: score them under the same conditions as the earlier 0.06-0.08 numbers): after the pipeline, the deployment
# loop exactly as stages A / B / W3 were scored - rw 1024, C 128, maxD 384, 4096-token traces, MATH 64 / mixed 60 - for both
# poolers, plus the no-SP equal-budget window (1056).
if [ ! -e /root/.job3 ]; then touch /root/.job3
  cat > /root/job3.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 90; done; echo "[hub] $2 FAILED"; }
while [ ! -e DONE_TP ]; do sleep 30; done
LG=/root/w/sp.log; Q="--S 32 --heads 8 --layers 3 --ffn 2048 --C 128 --maxd 384 --max-len 4096"
echo "[tp] same conditions as stage A (rw 1024, C 128, maxD 384, 4096 tokens) $(date -u)"
for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  for c in spP_orig.pt spP_norm.pt; do python3 sp_qwen3.py eval --traces $f $x --rw 1024 $Q --ckpt $c --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] /"; done
  python3 sp_qwen3.py eval --traces $f $x --rw 1056 $Q --no-sp --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] /"; done
put $LG spP.log; date -u > DONE_TP2; put DONE_TP2 DONE_TP2; echo "[tp] DONE_TP2 $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job3.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s3: same-condition scoring queued $(date -u)"
fi
# serial 2: the two trainers write their own logs - show both in the status line and the live log
if [ ! -e /root/.st2 ]; then touch /root/.st2
  sed -i 's#tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-220#for v in orig norm; do tail -n 1 /root/w/spP_$v.log 2>/dev/null | cut -c1-200; done; tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-200#' /root/status.sh
  sed -i 's#tail -n 12 sp.log;#tail -n 4 spP_orig.log; tail -n 4 spP_norm.log; tail -n 8 sp.log;#' /root/live.sh
  pkill -f live.sh; setsid nohup bash -c 'bash /root/live.sh' > /dev/null 2>&1 < /dev/null &
  echo "[tp] s2: status/live follow both trainer logs"
fi
if [ ! -e /root/.job ]; then touch /root/.job
  cat > /root/live.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
while [ ! -e DONE_TP ]; do
  { grep -E "^\[(tp|hub|math|mix)\]" w.log | tail -n 20; tail -n 12 sp.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; df -h / | tail -1; date -u; } > tp_live.log 2>&1
  python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='tp_live.log', path_in_repo='archive/sp3/tp_live.log', repo_id='baya1116/hypernet-sp-distill')" > /dev/null 2>&1
  sleep 600
done
JOB
  cat > /root/job.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; echo "[tp] put $2 attempt $t: $(tail -1 put.err | cut -c1-160)"; sleep 90; done; echo "[hub] $2 FAILED"; }
get() { for t in 1 2 3; do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o "$2" "$H/$1" && return 0; echo "[tp] get $1 attempt $t failed"; sleep 10; done; return 1; }
python3 -m pip install -q "huggingface_hub[hf_xet]" transformers bitsandbytes datasets 2>&1 | grep -v -i warn | tail -1
for t in 1 2 3; do
  x=""; [ $t -gt 1 ] && x="HF_HUB_DISABLE_XET=1"
  env $x timeout 600 python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-1.7B'))" > dl.log 2>&1 && break
  echo "[tp] model download attempt $t failed"
done
echo "[tp] model: $(tail -1 dl.log | cut -c1-120)"
get archive/kvpool/traces.jsonl traces.jsonl; get archive/kvwin/q17_full_p.jsonl kv_full_p.jsonl
for i in 0 1 2 3 4 5; do get archive/sp3/traces_big_$i.jsonl traces_big_$i.jsonl; done
curl -sS -o sp_mathcheck.py "$RAW/tools/sp_mathcheck.py?nocache=$(date +%s)" || curl -sS -o sp_mathcheck.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/tools/sp_mathcheck.py"
LG=/root/w/sp.log
P="--S 32 --heads 8 --layers 3 --ffn 2048 --C 64 --maxd 0"
echo "[tp] training two poolers (orig PE | pe-rev+pe-norm) $(date -u)"
python3 sp_qwen3.py train --traces traces_big_0.jsonl,traces_big_1.jsonl --n 4000 --max-len 768 --rw 128 $P --lr 3e-4 --accum 1 --out spP_orig.pt --log spP_orig.log > run_orig.log 2>&1 &
sleep 120
python3 sp_qwen3.py train --traces traces_big_0.jsonl,traces_big_1.jsonl --n 4000 --max-len 768 --rw 128 $P --pe-rev --pe-norm --lr 3e-4 --accum 1 --out spP_norm.pt --log spP_norm.log > run_norm.log 2>&1 &
wait
for v in orig norm; do grep -E "^\[sp\] trained|Error|Traceback|out of memory" run_$v.log | tail -2 | sed "s/^/[tp] $v: /"; done
put spP_orig.pt spP_orig_ap32recipe.pt; put spP_norm.pt spP_norm_ap32recipe.pt; cat spP_orig.log spP_norm.log >> $LG; put $LG spP.log
echo "[tp] deep KL on 768-token held-out traces (scored from 224) $(date -u)"
dk() { for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces $f $x --max-len 768 --score-from 224 "$@" --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] /"; done; }
dk --rw 160 $P --no-sp
dk --rw 128 $P --no-sp
dk --rw 128 $P --ckpt spP_orig.pt
dk --rw 128 $P --ckpt spP_norm.pt
put $LG spP.log
B=$(python3 - <<'PY'
import re
L=open("/root/w/sp.log").read()
def k(c):
    m=re.findall(r"rw128 C64 maxd0 L3 ffn2048[^\n]*%s: 64 traces \d+ tokens  KL ([0-9.]+)" % c, L); return float(m[-1]) if m else 9
print("spP_orig.pt" if k("spP_orig.pt") <= k("spP_norm.pt") else "spP_norm.pt")
PY
)
echo "[tp] better pooler: $B"
echo "[tp] wider window at inference: 4096-token traces, window 512, buffer cap 4096 $(date -u)"
for e in kv_full_p.jsonl:math traces.jsonl:mix; do f=${e%%:*}; n=${e#*:}; x=""; [ $n = mix ] && x="--n 60"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 --score-from 608 --rw 512 --S 32 --heads 8 --layers 3 --ffn 2048 --C 64 --maxd 4096 --ckpt $B --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] /"
  python3 sp_qwen3.py eval --traces $f $x --max-len 4096 --score-from 608 --rw 544 $P --no-sp --log $LG 2>&1 | grep -E "^\[sp\] eval|Error|Traceback" | sed "s/^/[$n] /"; done
put $LG spP.log
echo "[tp] generation check: 32 MATH problems, 4096 tokens $(date -u)"
G="--traces kv_full_p.jsonl --n 32 --max-len 4096 --roll-batch 16 --S 32 --heads 8 --layers 3 --ffn 2048 --log $LG"
python3 sp_qwen3.py rollout $G --no-sp --rw 8192 --C 512 --maxd 0 --roll-out gen_full.jsonl > run_gfull.log 2>&1
python3 sp_qwen3.py rollout $G --rw 512 --C 64 --maxd 4096 --ckpt $B --roll-out gen_sp512.jsonl > run_gsp.log 2>&1
python3 sp_qwen3.py rollout $G --no-sp --rw 544 --C 64 --maxd 0 --roll-out gen_raw544.jsonl > run_graw.log 2>&1
grep -h -E "Error|Traceback|out of memory" run_gfull.log run_gsp.log run_graw.log | tail -3
python3 sp_mathcheck.py gen_full.jsonl gen_sp512.jsonl gen_raw544.jsonl | sed "s/^/[tp] /"
for g in full sp512 raw544; do put gen_$g.jsonl ap32gen_$g.jsonl; done
put $LG spP.log; date -u > DONE_TP; put DONE_TP DONE_TP; echo "[tp] DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/live.sh' > /dev/null 2>&1 < /dev/null &
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[tp] job launched $(date -u)"
fi
# CTL-END
