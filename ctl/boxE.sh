# box E (24GB, replaces the A4000 whose host had no free GPU left): measure the held-out set through
# the 4-bit grid the phone actually runs.
#
# The app loads an MLX affine 4-bit conversion, group 64. Its weight index says what that touches:
# the seven projections of every block plus embed_tokens and lm_head, scales and biases in fp16,
# and nothing else - the norms, the attention biases and the pooler stay in float. q4.py reproduces
# that grid in torch and pool_eval.py --q4 1 applies it after loading.
#
# The comparison is against the 39.7% the same evaluator produced on the same 150 questions, whose
# per-rollout output the onstart restores as ev_out_*.jsonl, so the difference can be paired over
# questions instead of being read off two independent means.
cd /root/work
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
DRUN=d4
DPOL=1
DFOCUS=1
MODE=dwq
SHARDS=3
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
for f in pool_eval.py q4.py qat.py dwq.py checkmlx.py build_merged.py web_search.py; do
  for try in 1 2 3; do curl -sS -o /root/work/$f "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/work/$f && break; sleep 5; done
done
cp /root/work/web_search.py /root/work/runtime/web_search.py 2>/dev/null
echo "fetched: pool_eval $(wc -l < /root/work/pool_eval.py) lines, q4 $(wc -l < /root/work/q4.py) lines"

# The status command is the same whatever mode this file is in, so it is installed once,
# before the modes - a probe that forgets to reinstall it would otherwise report a stale table.
cat > /usr/local/bin/t <<'TTD'
#!/bin/bash
echo "$(date -u +%H:%M)Z dwq $(pgrep -fc 'dwq.p[y]')本  eval $(pgrep -fc 'pool_eval.p[y]')本  $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
for f in /root/dwq_run_*.log; do grep -hE "^\[dwq\]" "$f" 2>/dev/null | tail -3; done
python3 - <<'PYD' 2>/dev/null
import re,glob,os
for f in sorted(glob.glob("/root/dwq_d*.log")):
  tr=[];va=[]
  for l in open(f):
      m=re.match(r"step (\d+) kl=([\d.]+)",l)
      if m: tr.append((int(m.group(1)),float(m.group(2))))
      m=re.match(r"val (\d+) kl=([\d.]+)",l)
      if m: va.append((int(m.group(1)),float(m.group(2))))
  if not tr and not va: continue
  tag=os.path.basename(f)[4:-4]
  mean=lambda xs: sum(xs)/len(xs)
  s=f"  {tag}: "
  if tr: s+=f"step {tr[-1][0]} train kl {mean([r[1] for r in tr[:20]]):.4f} -> {mean([r[1] for r in tr[-20:]]):.4f}"
  if va: s+=f"   val {va[0][1]:.4f} -> {va[-1][1]:.4f} (best {min(v for _,v in va):.4f} @ {min(va,key=lambda x:x[1])[0]})"
  print(s)
PYD
python3 - <<'PYE' 2>/dev/null
import json,glob,collections,math,os
KEYS=("correct","grounded","landed","ns")
def load(pat):
  rows=[]
  for f in sorted(glob.glob(pat)):
      for line in open(f):
          try: rows.append(json.loads(line))
          except Exception: pass
  if not rows: return None
  per={}
  for k in KEYS:
      d=collections.defaultdict(list)
      for r in rows: d[r.get("q","")].append(float(r.get(k) or 0))
      per[k]={q:sum(v)/len(v) for q,v in d.items()}
  m={k:sum(float(r.get(k) or 0) for r in rows)/len(rows) for k in KEYS}
  return len(rows), m, per
runs=[("bf16","/root/work/ev_out_*.jsonl"),("4bit plain","/root/work/q4_out_*.jsonl"),
    ("4bit +STE-lora","/root/work/qa_out_*.jsonl"),("4bit +dwq d1","/root/work/dw_out_*.jsonl")]
for p in sorted(glob.glob("/root/work/d[0-9]_out_0.jsonl")):
  t=os.path.basename(p).split("_")[0]
  runs.append((f"4bit +dwq {t}", f"/root/work/{t}_out_*.jsonl"))
for p in sorted(glob.glob("/root/work/g[0-9]*_out_0.jsonl")):
  t=os.path.basename(p).split("_")[0]
  runs.append((f"4bit {t} plain", f"/root/work/{t}_out_*.jsonl"))
L={n:load(p) for n,p in runs}
for n,_ in runs:
  S=L[n]
  if S: print(f"  {n:16s} {100*S[1]['correct']:5.1f}%  gnd {100*S[1]['grounded']:3.0f}%  "
              f"land {100*S[1]['landed']:3.0f}%  srch {S[1]['ns']:.1f}  ({S[0]} roll)")
F=L["bf16"]
for n,_ in runs[1:]:
  S=L[n]
  if not (F and S): continue
  c=sorted(set(F[2]["correct"])&set(S[2]["correct"]))
  if not c: continue
  out=[]
  for k,sc,u in (("correct",100,"pt"),("grounded",100,"pt"),("landed",100,"pt"),("ns",1,"")):
      d=[S[2][k][q]-F[2][k][q] for q in c]; m=sum(d)/len(d)
      sd=(sum((x-m)**2 for x in d)/len(d))**0.5
      out.append(f"{k[:4]} {sc*m:+.1f}{u}±{sc*sd/math.sqrt(len(d)):.1f}")
  print(f"  vs bf16 {n:16s} " + "  ".join(out) + f"  ({len(c)} q)")
PYE
TTD
chmod +x /usr/local/bin/t
cat > /root/status.sh <<'STD'
#!/bin/bash
t 2>/dev/null
STD
chmod +x /root/status.sh
pkill -f "status_pu[b]"; sleep 1
cat > /root/status_pub.sh <<'SPD'
#!/bin/bash
while true; do
{ echo "=== $(date -u) === box E (dwq)"; t 2>/dev/null; } > /root/work/status.txt 2>&1
echo "--- STATUS $(date -u +%H:%M) ---"; cat /root/work/status.txt
sleep 300
done
SPD
chmod +x /root/status_pub.sh
setsid nohup bash /root/status_pub.sh >> /proc/1/fd/1 2>&1 < /dev/null &

if [ "$MODE" = "probe" ]; then
  # Three training attempts have each closed most of the KL gap to bf16 and left the score where it
  # was. Before a fourth, ask a different question: is the format itself the binding constraint?
  # Group 32 halves how many weights share a scale. It needs one line changed in the app and a
  # reconversion, so it is a recommendation rather than a drop-in - but if it recovers the gap, that
  # is the answer, and if it does not, no amount of training the group-64 grid will help either.
  PRUN=${PRUN:-g32}; PG=${PG:-32}
  pkill -f "afterkee[p].sh"; pkill -f "evalkee[p].sh"; pkill -f "dwqkee[p].sh"; pkill -f "qat.p[y]"; sleep 5
  R=baya1116/hypernet-sp-distill; D=pooler_distill/grpo_pool3_step200_q4
  if [ ! -f /root/.d3_published ] && [ -s /root/dwq_mlx4_d3/model.safetensors ]; then
    export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
    echo "--- publishing d3 ($(du -shL /root/dwq_mlx4_d3 | cut -f1)) ---"
    hf upload $R /root/dwq_mlx4_d3 $D/dwq_d3_mlx4 2>&1 | tail -2
    hf upload $R /root/dwq/d3.pt $D/dwq_d3_params.pt 2>&1 | tail -1
    hf upload $R /root/dwq_d3.log $D/dwq_d3.log >/dev/null 2>&1
    for i in 0 1 2; do hf upload $R /root/work/d3_out_$i.jsonl $D/d3_shard$i.jsonl >/dev/null 2>&1; done
    touch /root/.d3_published; echo "d3 published"
  fi
  cat > /root/probekeep.sh <<PKQ
#!/bin/bash
PRUN=$PRUN; PG=$PG
PKQ
  cat >> /root/probekeep.sh <<'PKQ2'
while :; do
  for i in 0 1 2; do
    want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
    have=$(wc -l < /root/work/${PRUN}_out_$i.jsonl 2>/dev/null || echo 0)
    [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
    pgrep -f "pool_eval.py .* /root/work/ev_$i.jsonl" >/dev/null && continue
    echo "[$PRUN-probe $(date -u +%H:%M)] shard $i at $have/$want - starting (group $PG)"
    grep -viE "^\s*$" /root/${PRUN}_$i.log 2>/dev/null | tail -4 | cut -c1-200
    cd /root/work && SP_BASE=/root/eval_hf200 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py \
      /root/pooler200.safetensors /root/work/ev_$i.jsonl /root/work/${PRUN}_out_$i.jsonl \
      --n 999 --rw 768 --maxd 384 --samepage 1 --decode plain --q4 1 --q4group $PG --tag "[$PRUN$i]" \
      >> /root/${PRUN}_$i.log 2>&1 < /dev/null &
    sleep 60
  done
  sleep 120
done
PKQ2
  chmod +x /root/probekeep.sh
  pkill -f "probekee[p].sh"; sleep 1
  setsid nohup bash /root/probekeep.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  sleep 40; t; echo "PROBE_LAUNCH_DONE $PRUN group $PG $(date -u)"
  exit 0
fi

if [ "$MODE" = "dwq" ]; then
  # Train the quantization parameters rather than the weights under them, starting from the bf16
  # model that scored 39.7%. RUN names the attempt, so a second one does not overwrite the first.
  RUN=${DRUN:-d1}
  pkill -f "afterkee[p].sh"; pkill -f "evalkee[p].sh"; pkill -f "dwqkee[p].sh"; pkill -f "qat.p[y]"; sleep 5
  [ -s /root/work/dwq_calib/train.jsonl ] || { echo "NOT READY: no calibration set"; exit 0; }
  echo "=== DWQ run $RUN $(date -u) === calibration $(wc -l < /root/work/dwq_calib/train.jsonl) sequences"
  HF=/root/dwq_hf_$RUN; MLX=/root/dwq_mlx4_$RUN; LOG=/root/dwq_$RUN.log
  # a run under a different tag is a superseded configuration, not something to wait for
  if pgrep -af "dwq.p[y]" | grep -qv -- "--ckpt /root/dwq/$RUN.pt"; then
    echo "stopping the run that is not $RUN: $(pgrep -af 'dwq.p[y]' | head -1 | cut -c1-120)"
    pkill -f "dwq.p[y]"; sleep 8; pkill -9 -f "dwq.p[y]" 2>/dev/null; sleep 2
  fi
  RUNNING=0
  pgrep -f "dwq.p[y]" >/dev/null && { RUNNING=1; echo "DWQ already running: $(tail -1 $LOG)"; }
  [ -f $HF/model.safetensors ] && { RUNNING=1; echo "DWQ $RUN already finished"; }
  if [ "$RUNNING" = "0" ]; then
    # Only a launch needs the card to itself. Re-running this file while a measurement is in flight
    # must not kill it - that is how the status command gets fixed without losing an hour of work.
    pkill -f "pool_eval.p[y]"; sleep 8; pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 2
    # The quantizer's own checks first: they cost seconds and they gate an hour of GPU.
    python3 /root/work/q4.py || { echo "Q4 SELFTEST FAILED - not launching"; exit 0; }
    cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/dwq.py \
      --base /root/eval_hf200 --data /root/work/dwq_calib/train.jsonl \
      --out-hf $HF --out-mlx $MLX --ckpt /root/dwq/$RUN.pt --log $LOG \
      --clip-search ${DCLIP:-1} --lr ${DLR:-2e-6} --steps ${DSTEPS:-4000} \
      --accum 4 --len 1024 --kpos 256 --temp ${DTEMP:-1.0} --val 48 --val-every ${DVAL:-100} \
      --policy-only ${DPOL:-0} --focus ${DFOCUS:-0} \
      >> /root/dwq_run_$RUN.log 2>&1 < /dev/null &
    sleep 20
  fi
  # The measurement needs no --q4: the directory already holds the 4-bit values, dequantized.
  cat > /root/dwqkeep.sh <<DKQ
#!/bin/bash
RUN=$RUN; HF=$HF
DKQ
  cat >> /root/dwqkeep.sh <<'DKQ2'
until [ -s $HF/model.safetensors ] && grep -q DWQ_DONE /root/dwq_run_$RUN.log 2>/dev/null; do sleep 60; done
pkill -f "dwq.p[y]"; sleep 10
python3 /root/work/checkmlx.py ${HF/_hf_/_mlx4_} $HF 2>&1 | tail -8
while :; do
  for i in 0 1 2; do
    want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
    have=$(wc -l < /root/work/${RUN}_out_$i.jsonl 2>/dev/null || echo 0)
    [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
    pgrep -f "pool_eval.py .* /root/work/ev_$i.jsonl" >/dev/null && continue
    echo "[$RUN-eval $(date -u +%H:%M)] shard $i at $have/$want - starting"
    grep -viE "^\s*$" /root/${RUN}_$i.log 2>/dev/null | tail -4 | cut -c1-200
    cd /root/work && SP_BASE=$HF SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py \
      /root/pooler200.safetensors /root/work/ev_$i.jsonl /root/work/${RUN}_out_$i.jsonl \
      --n 999 --rw 768 --maxd 384 --samepage 1 --decode plain --tag "[$RUN$i]" \
      >> /root/${RUN}_$i.log 2>&1 < /dev/null &
    sleep 60
  done
  sleep 120
done
DKQ2
  chmod +x /root/dwqkeep.sh
  setsid nohup bash /root/dwqkeep.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  sleep 40; t; echo "DWQ_LAUNCH_DONE $RUN $(date -u)"
  exit 0
fi

if [ "$MODE" = "publish" ]; then
  # Everything worth keeping leaves the box before it is touched: this host's ssh tunnel never came
  # up, so the next box is a new one, and a stopped instance is not guaranteed to start again.
  pkill -f "afterkee[p].sh"; pkill -f "evalkee[p].sh"; pkill -f "pool_eval.p[y]"; sleep 8
  pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 2
  R=baya1116/hypernet-sp-distill; D=pooler_distill/grpo_pool3_step200_q4
  python3 - <<'PYP' > /root/work/qat_metrics.json
import json, glob, collections, math

def load(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            try: rows.append(json.loads(line))
            except Exception: pass
    if not rows: return None
    by = collections.defaultdict(list)
    for r in rows: by[r.get("q", "")].append(bool(r.get("correct")))
    return {"rollouts": len(rows), "questions": len(by),
            "correct": round(100*sum(1 for r in rows if r.get("correct"))/len(rows), 1),
            "grounded": round(100*sum(1 for r in rows if r.get("grounded"))/len(rows), 1),
            "landed": round(100*sum(1 for r in rows if r.get("landed"))/len(rows), 1),
            "searches_per_rollout": round(sum(r.get("ns", 0) for r in rows)/len(rows), 2),
            "_per_q": {k: sum(v)/len(v) for k, v in by.items()}}

def paired(X, Y):
    if not (X and Y): return None
    c = sorted(set(X["_per_q"]) & set(Y["_per_q"]))
    if not c: return None
    d = [Y["_per_q"][k] - X["_per_q"][k] for k in c]
    m = sum(d)/len(d); sd = (sum((x-m)**2 for x in d)/len(d))**0.5
    return {"delta_pt": round(100*m, 1), "se_pt": round(100*sd/math.sqrt(len(c)), 1), "questions": len(c)}

F, Q, N = load("/root/work/ev_out_*.jsonl"), load("/root/work/q4_out_*.jsonl"), load("/root/work/qa_out_*.jsonl")
out = {
  "what": "Does the phone's 4-bit conversion cost accuracy, and does training against it help?",
  "quantization": {"scheme": "MLX affine, group 64, 4 bits, scales and biases fp16",
                   "modules": "7 projections x 28 blocks + embed_tokens + lm_head = 198, 1776.9M weights",
                   "weight_relative_error": {"projections": 9.51, "embed_tokens": 10.19, "lm_head": 10.35}},
  "training": {"method": "straight-through on the merged weight: forward uses W + (Q(W)-W).detach(), W = base + BA",
               "trainable": "LoRA r32 alpha64 on the 196 projections, 36.9M parameters",
               "objective": "KL to the model's own bf16 self at sampled positions + normalised hidden-state MSE",
               "data": "4344 of the model's own GRPO traces", "steps": 1500, "seconds_per_step": 1.7,
               "kl": {"first_20_steps": 0.0713, "last_20_steps": 0.0229},
               "hidden_mse": {"first_20_steps": 0.0260, "last_20_steps": 0.0110}},
  "heldout": {"file": "eval300.jsonl first 300 lines, 150 distinct questions x 2",
              "settings": {"rw": 768, "maxd": 384, "samepage": 1, "temp": 0.9, "gen": 1500, "maxs": 5, "decode": "plain"},
              "bf16": F and {k: v for k, v in F.items() if k != "_per_q"},
              "4bit_before": Q and {k: v for k, v in Q.items() if k != "_per_q"},
              "4bit_after": N and {k: v for k, v in N.items() if k != "_per_q"}},
  "paired": {"4bit_before_vs_bf16": paired(F, Q), "4bit_after_vs_bf16": paired(F, N),
             "4bit_after_vs_before": paired(Q, N)},
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PYP
  cat /root/work/qat_metrics.json
  echo "--- uploading $(du -shL /root/qat_hf 2>/dev/null | cut -f1) ---"
  hf upload $R /root/qat_hf $D/model 2>&1 | tail -2
  hf upload $R /root/work/qat_metrics.json $D/metrics.json 2>&1 | tail -1
  hf upload $R /root/qat/latest.pt $D/qat_lora.pt 2>&1 | tail -1
  hf upload $R /root/qat.log $D/qat.log >/dev/null 2>&1
  for i in 0 1 2; do
    hf upload $R /root/work/q4_out_$i.jsonl $D/before_shard$i.jsonl >/dev/null 2>&1
    hf upload $R /root/work/qa_out_$i.jsonl $D/after_shard$i.jsonl >/dev/null 2>&1
  done
  # Calibration text for DWQ, in the shape the model actually sees: the harness prompt, the <think>
  # opener, and the model's own trace. mlx-lm reads a folder of jsonl with a "text" field.
  mkdir -p /root/work/dwq_calib
  python3 - <<'PYC'
import json, os
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/root/eval_hf200")
held = set()
for line in open("/root/work/eval300.jsonl"):
    try: held.add(json.loads(line).get("q", ""))
    except Exception: pass
seen, n = set(), 0
with open("/root/work/dwq_calib/train.jsonl", "w") as out:
    for line in open("/root/work/rollouts.jsonl"):
        try: d = json.loads(line)
        except Exception: continue
        q, t = d.get("q", ""), d.get("text", "")
        if not q or not t or q in held:     # the held-out questions never enter any training input
            continue
        head = tok.apply_chat_template([{"role": "user", "content": q}],
                                       add_generation_prompt=True, tokenize=False) + "<think>\n"
        out.write(json.dumps({"text": head + t}, ensure_ascii=False) + "\n"); n += 1
        seen.add(q)
print(f"[calib] {n} sequences over {len(seen)} questions, {len(held)} held-out questions excluded")
PYC
  hf upload $R /root/work/dwq_calib/train.jsonl pooler_distill/dwq_calib/train.jsonl 2>&1 | tail -1
  echo "PUBLISH_DONE $(date -u)"
  exit 0
fi

if [ "$MODE" = "qat" ]; then
  # Quantization-aware training. The held-out measurement is the before-number and stays on disk;
  # what runs now moves the weights so that the same 4-bit conversion stops costing accuracy.
  pkill -f "evalkee[p].sh"; pkill -f "pool_eval.p[y]"; sleep 8
  pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 2
  for i in 0 1 2; do hf upload baya1116/hypernet-sp-distill /root/work/q4_out_$i.jsonl pooler_distill/grpo_pool3_step200/q4_shard$i.jsonl >/dev/null 2>&1; done
  echo "before-number frozen at $(cat /root/work/q4_out_*.jsonl 2>/dev/null | wc -l) rollouts"
  # the model's own traces: on-distribution calibration input, and the only data this needs
  if [ ! -s /root/work/rollouts.jsonl ]; then
    for try in 1 2 3; do hf download baya1116/hypernet-sp-distill --include "pooler_distill/grpo_pool3/rollouts.jsonl" --local-dir /root/hfdl 2>&1 | tail -1 && break; sleep 10; done
    cp /root/hfdl/pooler_distill/grpo_pool3/rollouts.jsonl /root/work/rollouts.jsonl
  fi
  echo "traces: $(wc -l < /root/work/rollouts.jsonl) rollouts"
  RUNNING=0
  pgrep -f "qat.p[y]" >/dev/null && { RUNNING=1; echo "QAT already running: $(tail -1 /root/qat.log)"; }
  [ -f /root/qat_hf/model.safetensors ] && { RUNNING=1; echo "QAT already finished -> /root/qat_hf"; }
  if [ "$RUNNING" = "0" ]; then
  # Two steps first: they print the step-0 loss, which IS the quantization damage while the adapter
  # is still zero, and they prove the memory fits before an hour is committed to it. A crash here is
  # not retried, it is looked at.
  if [ ! -f /root/.qat_selftest_ok ]; then
    rm -f /root/qat.log
    cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/qat.py \
      --base /root/eval_hf200 --data /root/work/rollouts.jsonl --selftest 2 2>&1 | tail -22
    grep -q "^step 2 " /root/qat.log 2>/dev/null || { echo "QAT SELFTEST FAILED - not launching"; exit 0; }
    touch /root/.qat_selftest_ok
  fi
  cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/qat.py \
    --base /root/eval_hf200 --data /root/work/rollouts.jsonl --out /root/qat_hf \
    --rank 32 --alpha 64 --lr 1e-4 --steps ${QSTEPS:-1500} --accum 4 --len 640 --kpos 256 \
    >> /root/qat_run.log 2>&1 < /dev/null &
  sleep 20
  fi
  cat > /usr/local/bin/t <<'TTQ'
#!/bin/bash
echo "$(date -u +%H:%M)Z qat $(pgrep -fc 'qat.p[y]')本  $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
grep -E "^\[q4\]|^\[lora\]|^\[data\]" /root/qat_run.log 2>/dev/null | head -5
python3 - <<'PYT' 2>/dev/null
import re
rows=[]
for l in open("/root/qat.log"):
    m=re.match(r"step (\d+) kl=([\d.]+) mse=([\d.]+)", l)
    if m: rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
if rows:
    rows.sort()
    def mean(xs): return sum(xs)/len(xs)
    a, b = rows[:20], rows[-20:]
    print(f"  step {rows[-1][0]}  kl {mean([r[1] for r in a]):.4f} -> {mean([r[1] for r in b]):.4f}"
          f"   hidden mse {mean([r[2] for r in a]):.4f} -> {mean([r[2] for r in b]):.4f}"
          f"   ({100*(1-mean([r[1] for r in b])/max(mean([r[1] for r in a]),1e-9)):.0f}% of the rounding damage removed)")
PYT
tail -2 /root/qat.log 2>/dev/null
python3 - <<'PYQ' 2>/dev/null
import json,glob,collections,math
def summary(pat):
    rows=[]
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            try: rows.append(json.loads(line))
            except Exception: pass
    if not rows: return None
    by=collections.defaultdict(list)
    for r in rows: by[r.get("q","")].append(bool(r.get("correct")))
    return len(rows), 100*sum(1 for r in rows if r.get("correct"))/len(rows), {k:sum(v)/len(v) for k,v in by.items()}
F=summary("/root/work/ev_out_*.jsonl"); Q=summary("/root/work/q4_out_*.jsonl"); N=summary("/root/work/qa_out_*.jsonl")
for tag,S in (("bf16",F),("4bit before",Q),("4bit after",N)):
    if S: print(f"  {tag:12s} {S[1]:5.1f}%  ({S[0]} roll)")
def pair(tag,X,Y):
    if not (X and Y): return
    c=sorted(set(X[2])&set(Y[2]))
    if not c: return
    d=[Y[2][k]-X[2][k] for k in c]; m=sum(d)/len(d)
    sd=(sum((x-m)**2 for x in d)/len(d))**0.5
    print(f"  paired {tag:22s} {100*m:+.1f} pt ±{100*sd/math.sqrt(len(d)):.1f} over {len(c)} questions")
pair("4bit before vs bf16",F,Q); pair("4bit after vs bf16",F,N); pair("4bit after vs before",Q,N)
PYQ
TTQ
  chmod +x /usr/local/bin/t
  cat > /root/status.sh <<'STQ'
#!/bin/bash
t 2>/dev/null
STQ
  chmod +x /root/status.sh
  pkill -f "status_pu[b]"; sleep 1
  cat > /root/status_pub.sh <<'SPQ'
#!/bin/bash
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
while true; do
  { echo "=== $(date -u) === box E (qat)"; t 2>/dev/null; } > /root/work/status.txt 2>&1
  echo "--- STATUS $(date -u +%H:%M) ---"; cat /root/work/status.txt
  sleep 300
done
SPQ
  # When the training finishes cleanly, the same held-out measurement runs against the trained
  # weights, through the same evaluator, the same 150 questions and the same 4-bit grid. This is
  # sequenced rather than retried: it starts only on QAT_DONE plus a saved model, and a shard that
  # dies mid-run is restarted because pool_eval resumes from its own output.
  cat > /root/afterkeep.sh <<'AKQ'
#!/bin/bash
until [ -s /root/qat_hf/model.safetensors ] && grep -q QAT_DONE /root/qat_run.log 2>/dev/null; do sleep 60; done
pkill -f "qat.p[y]"; sleep 10
while :; do
  for i in 0 1 2; do
    want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
    have=$(wc -l < /root/work/qa_out_$i.jsonl 2>/dev/null || echo 0)
    [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
    pgrep -f "pool_eval.py .* /root/work/ev_$i.jsonl" >/dev/null && continue
    echo "[after $(date -u +%H:%M)] shard $i at $have/$want - starting"
    grep -viE "^\s*$" /root/qa_$i.log 2>/dev/null | tail -4 | cut -c1-200
    cd /root/work && SP_BASE=/root/qat_hf SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py \
      /root/pooler200.safetensors /root/work/ev_$i.jsonl /root/work/qa_out_$i.jsonl \
      --n 999 --rw 768 --maxd 384 --samepage 1 --decode plain --q4 1 --tag "[a$i]" \
      >> /root/qa_$i.log 2>&1 < /dev/null &
    sleep 60
  done
  sleep 120
done
AKQ
  chmod +x /root/afterkeep.sh
  pkill -f "afterkee[p].sh"; sleep 1
  setsid nohup bash /root/afterkeep.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  chmod +x /root/status_pub.sh
  setsid nohup bash /root/status_pub.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  sleep 40; t; echo "QAT_LAUNCH_DONE $(date -u)"
  exit 0
fi

# The control loop starts before the assets finish arriving, so wait here rather than exiting: ctl
# only re-runs this file when its content changes, and a one-line "not ready" would be the last thing
# the box ever said.
for w in $(seq 1 40); do
  [ -s /root/eval_hf200/model.safetensors ] && [ -s /root/pooler200.safetensors ] && break
  echo "[wait $w] downloading: $(du -sh /root/hfdl 2>/dev/null | cut -f1) in /root/hfdl, $(tail -1 /root/boot.log 2>/dev/null | cut -c1-100)"
  sleep 60
done
if [ ! -s /root/eval_hf200/model.safetensors ]; then echo "NOT READY: /root/eval_hf200 never arrived"; exit 0; fi
if [ ! -s /root/pooler200.safetensors ]; then echo "NOT READY: pooler never arrived"; exit 0; fi
if [ ! -s /root/work/eval300.jsonl ]; then echo "NOT READY: questions never arrived"; exit 0; fi

# thirty seconds of checking the quantizer is cheaper than two hours of measuring with a broken one
python3 /root/work/q4.py || { echo "Q4 SELFTEST FAILED - not launching"; exit 0; }

# the same first 300 lines every earlier number on this file came from
if [ ! -f /root/work/ev_0.jsonl ]; then
  python3 - <<'PY'
import json
qs=[l for l in open("/root/work/eval300.jsonl") if l.strip()][:300]
S=3
for i in range(S): open(f"/root/work/ev_{i}.jsonl","w").writelines(qs[i::S])
print(f"[shard] first 300 lines, {len({json.loads(l).get('q','') for l in qs})} distinct questions -> {[len(qs[i::S]) for i in range(S)]}")
PY
fi
[ -f /root/.q4_t0 ] || date +%s > /root/.q4_t0

# One-shot: the first launch did the quantization arithmetic on the GPU, which left about 9.5GB
# reserved per process and let only two of the three shards onto the card - the third OOMed and the
# watchdog kept retrying it. The quantizer now works on the CPU in row blocks, so restart the shards
# once to pick that up. pool_eval resumes from its own output, so only the question in flight is lost.
if [ ! -f /root/.q4_cpuquant ]; then
  touch /root/.q4_cpuquant
  pkill -f "evalkee[p].sh"; pkill -f "pool_eval.p[y]"; sleep 10
  pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 3
  echo "shards stopped; they come back on the CPU-side quantizer ($(cat /root/work/q4_out_*.jsonl 2>/dev/null | wc -l) rollouts kept)"
fi

# ctl only re-runs this file when its content changes, so a shard that dies would otherwise sit dead
# until the next edit. This watchdog restarts one when its process is gone and its output is short;
# pool_eval resumes by skipping the questions already in its own output.
cat > /root/evalkeep.sh <<'KG'
#!/bin/bash
while :; do
  for i in 0 1 2; do
    want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
    have=$(wc -l < /root/work/q4_out_$i.jsonl 2>/dev/null || echo 0)
    [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
    pgrep -f "pool_eval.py .* /root/work/ev_$i.jsonl" >/dev/null && continue
    echo "[keep $(date -u +%H:%M)] shard $i is dead at $have/$want - restarting"
    grep -viE "^\s*$" /root/q4_$i.log 2>/dev/null | tail -4 | cut -c1-200
    cd /root/work && SP_BASE=/root/eval_hf200 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py \
      /root/pooler200.safetensors /root/work/ev_$i.jsonl /root/work/q4_out_$i.jsonl \
      --n 999 --rw 768 --maxd 384 --samepage 1 --decode plain --q4 1 --tag "[q$i]" \
      >> /root/q4_$i.log 2>&1 < /dev/null &
    sleep 60          # stagger the model loads; three landing together is what killed a shard last time
  done
  sleep 120
done
KG
chmod +x /root/evalkeep.sh
pkill -f "evalkee[p].sh"; sleep 1
setsid nohup bash /root/evalkeep.sh >> /proc/1/fd/1 2>&1 < /dev/null &

cat > /usr/local/bin/t <<'TT'
#!/bin/bash
python3 - <<'PY'
import json,glob,os,time,math,collections

def read(pat):
    rows=[]
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows

def summary(rows):
    n=len(rows)
    if not n: return None
    by=collections.defaultdict(list)
    for r in rows: by[r.get("q","")].append(bool(r.get("correct")))
    qm={k:sum(v)/len(v) for k,v in by.items()}
    mu=sum(qm.values())/len(qm)
    sd=(sum((x-mu)**2 for x in qm.values())/len(qm))**0.5
    def per_q(key):
        d=collections.defaultdict(list)
        for r in rows: d[r.get("q","")].append(float(r.get(key) or 0))
        return {k:sum(v)/len(v) for k,v in d.items()}
    return dict(n=n, q=len(qm), qm=qm, gq=per_q("grounded"), sq=per_q("ns"),
                c=100*sum(1 for r in rows if r.get("correct"))/n,
                g=100*sum(1 for r in rows if r.get("grounded"))/n,
                l=100*sum(1 for r in rows if r.get("landed"))/n,
                s=sum(r.get("ns",0) for r in rows)/n,
                se=100*sd/math.sqrt(len(qm)))

alive=os.popen("pgrep -fc 'pool_eval.p[y]'").read().strip() or "0"
qh=os.popen("grep '^\\[q4\\]' /root/q4_0.log 2>/dev/null | head -2").read().rstrip()
t0=0
try: t0=int(open("/root/.q4_t0").read().strip())
except Exception: pass
Q=summary(read("/root/work/q4_out_*.jsonl")); F=summary(read("/root/work/ev_out_*.jsonl"))
nq=Q["n"] if Q else 0
el=max(time.time()-t0,1) if t0 else 0
eta=f"  残り~{(300-nq)/(nq/el)/60:.0f}分" if nq and el and nq<300 else ""
print(f"{time.strftime('%H:%M',time.gmtime())}Z q4eval {alive}本  {nq}/300{eta}")
if qh: print("  "+qh.replace("\n","\n  "))
for tag,S in (("bf16",F),("4bit",Q)):
    if S: print(f"  {tag}  correct {S['c']:5.1f}% ±{S['se']:.1f}  gnd {S['g']:3.0f}%  land {S['l']:3.0f}%  srch {S['s']:.1f}  ({S['n']} roll / {S['q']} q)")
if Q and F:
    common=sorted(set(Q["qm"]) & set(F["qm"]))
    if common:
        def pair(key, scale, unit):
            d=[Q[key][k]-F[key][k] for k in common]
            m=sum(d)/len(d); sd=(sum((x-m)**2 for x in d)/len(d))**0.5
            return f"{scale*m:+.1f}{unit} ±{scale*sd/math.sqrt(len(d)):.1f}"
        print(f"  paired 4bit-bf16 over {len(common)} shared questions: "
              f"correct {pair('qm',100,'pt')}  gnd {pair('gq',100,'pt')}  srch {pair('sq',1,'')}")
PY
TT
chmod +x /usr/local/bin/t
cat > /root/status.sh <<'ST'
#!/bin/bash
t 2>/dev/null
ST
chmod +x /root/status.sh
pkill -f "status_pu[b]"; sleep 1
cat > /root/status_pub.sh <<'SP2'
#!/bin/bash
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
while true; do
  { echo "=== $(date -u) === box E (4-bit held-out)"; t 2>/dev/null; } > /root/work/status.txt 2>&1
  echo "--- STATUS $(date -u +%H:%M) ---"; cat /root/work/status.txt
  for i in 0 1 2; do hf upload baya1116/hypernet-sp-distill /root/work/q4_out_$i.jsonl pooler_distill/grpo_pool3_step200/q4_shard$i.jsonl >/dev/null 2>&1; done
  sleep 300
done
SP2
chmod +x /root/status_pub.sh
setsid nohup bash /root/status_pub.sh >> /proc/1/fd/1 2>&1 < /dev/null &
sleep 30; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader; t
echo "Q4EVAL_LAUNCH_DONE $(date -u)"
