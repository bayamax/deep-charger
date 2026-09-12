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
# The token arrives as an environment variable on the instance; keep a copy so anything this
# script starts later still has it, and so the onstart can stay as short as possible.
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
# ---- one-time bootstrap -------------------------------------------------------------------------
# This used to live in the instance's onstart. Four hosts in a row refused to create a container with
# it there, and the only thing that had changed was its content, so it moved here: the onstart is now
# a handful of lines that fetch this file, and everything the box needs is in git where it can be
# fixed without renting a new machine.
if [ ! -f /root/.bootstrapped ]; then
  # one bootstrap at a time: a re-run while this is still downloading must not start a second copy
  mkdir /root/.bootstrap_lock 2>/dev/null || { echo "bootstrap already running - not starting another"; exit 0; }
  echo "=== bootstrap $(date -u) ==="
  mkdir -p /root/work/fft_out /root/work/runtime /root/work/dwq_calib /root/hfdl
  touch /root/work/runtime/__init__.py
  pip install -q "transformers==4.44.2" "peft==0.12.0" "safetensors==0.8.0" \
    "huggingface_hub>=0.34,<1.0" accelerate certifi datasets scipy 2>&1 | tail -1
  R=baya1116/hypernet-sp-distill; S=pooler_distill/grpo_pool3_step200; Q=pooler_distill/grpo_pool3_step200_q4
  for inc in "box_recover/scripts/*" "fft_out/pooler.pt" \
             "grpo_assets/mus_run/eval_step200/eval_heldout_300q_g2.jsonl" \
             "pooler_distill/pool_eval_cache.jsonl" "pooler_distill/dwq_calib/*" \
             "$S/*" "$Q/rollouts/*"; do
    for try in 1 2 3 4 5 6; do hf download $R --include "$inc" --local-dir /root/hfdl 2>&1 | tail -1 && break; sleep 20; done
    echo "dl $inc $(date -u +%H:%M)"
  done
  SC=/root/hfdl/box_recover/scripts; cp $SC/*.py $SC/*.sh /root/work/ 2>/dev/null
  ln -sfn /root/hfdl/fft_out/pooler.pt /root/work/fft_out/pooler.pt
  cp /root/hfdl/grpo_assets/mus_run/eval_step200/eval_heldout_300q_g2.jsonl /root/work/eval300.jsonl
  cp /root/hfdl/pooler_distill/pool_eval_cache.jsonl /root/work/pool_eval_cache.jsonl
  cp /root/hfdl/pooler_distill/dwq_calib/train.jsonl /root/work/dwq_calib/train.jsonl
  ln -sfn /root/hfdl/$S/model /root/eval_hf200
  ln -sfn /root/hfdl/$S/pooler.safetensors /root/pooler200.safetensors
  for i in 0 1 2; do cp /root/hfdl/$S/eval_shard$i.jsonl /root/work/ev_out_$i.jsonl; done
  # every arm measured on the previous box, so comparisons stay paired and a cut-short arm resumes
  for a in q4 qa d3 g32 j1 p_embfloat p_t06q4 p_t06bf16 s1; do
    for i in 0 1 2; do
      f=/root/hfdl/$Q/rollouts/${a}_$i.jsonl
      [ -s $f ] && cp $f /root/work/${a}_out_$i.jsonl
    done
  done
  python3 - <<'PYB'
import json
qs=[l for l in open("/root/work/eval300.jsonl") if l.strip()][:300]
for i in range(3): open(f"/root/work/ev_{i}.jsonl","w").writelines(qs[i::3])
print(f"[shard] first 300 lines, {len({json.loads(l).get('q','') for l in qs})} distinct questions")
PYB
  echo "model $(ls -lL /root/eval_hf200/model.safetensors 2>/dev/null | awk '{print $5}') bytes | " \
       "bf16 $(cat /root/work/ev_out_*.jsonl 2>/dev/null | wc -l) rollouts | " \
       "plain-4bit $(cat /root/work/q4_out_*.jsonl 2>/dev/null | wc -l) restored | " \
       "calib $(wc -l < /root/work/dwq_calib/train.jsonl 2>/dev/null)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /root | tail -1
  [ -s /root/eval_hf200/model.safetensors ] && touch /root/.bootstrapped && echo "BOOTSTRAP_DONE $(date -u)"
  rmdir /root/.bootstrap_lock 2>/dev/null
fi
[ -f /root/.bootstrapped ] || { echo "bootstrap incomplete - stopping here"; exit 0; }

DRUN=d4
DPOL=1
DFOCUS=1
PRUN=p_embfloat
PSKIP=embed_tokens
JRUN=j2
JLRQ=0
JCLIP=0
PRUNS="p_t06q4:1:0.6 p_t06bf16:0:0.6"
PG=64
PSKIP=
MODE=publish2
SHARDS=3
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
for f in pool_eval.py q4.py qat.py dwq.py poolerfit.py jointfit.py checkmlx.py build_merged.py web_search.py; do
  for try in 1 2 3; do curl -sS -o /root/work/$f "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/work/$f && break; sleep 5; done
done
cp /root/work/web_search.py /root/work/runtime/web_search.py 2>/dev/null
echo "fetched: pool_eval $(wc -l < /root/work/pool_eval.py) lines, q4 $(wc -l < /root/work/q4.py) lines"

# The status command is the same whatever mode this file is in, so it is installed once,
# before the modes - a probe that forgets to reinstall it would otherwise report a stale table.
cat > /usr/local/bin/t <<'TTD'
#!/bin/bash
echo "$(date -u +%H:%M)Z dwq $(pgrep -fc 'dwq.p[y]')本  eval $(pgrep -fc 'pool_eval.p[y]')本  $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
# a shard that is running but has written nothing is indistinguishable from a shard that is stuck,
# unless someone looks inside it
for f in $(ls -t /root/*_[0-9].log 2>/dev/null | head -3); do
  echo "  $(basename $f): $(grep -viE '^\s*$' $f | tail -1 | cut -c1-140)"
done
for f in /root/dwq_run_*.log /root/joint_run_*.log; do grep -hE "^\[dwq\]|^\[joint\]|taking step" "$f" 2>/dev/null | tail -3; done
python3 - <<'PYD' 2>/dev/null
import re,glob,os
for f in sorted(glob.glob("/root/dwq_d*.log")) + sorted(glob.glob("/root/joint_j*.log")) + sorted(glob.glob("/root/sft_s*.log")):
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
for p in sorted(glob.glob("/root/work/j[0-9]_out_0.jsonl")) + sorted(glob.glob("/root/work/s[0-9]_out_0.jsonl")) + sorted(glob.glob("/root/work/p_*_out_0.jsonl")):
  t=os.path.basename(p)[:-len("_out_0.jsonl")]
  runs.append((f"4bit {t}", f"/root/work/{t}_out_*.jsonl"))
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
# an arm whose tag ends in bf16 is a reference of its own: every arm sharing its prefix is paired
# against it too, so a probe run at another temperature is read in its own frame as well
for rn,_ in runs[1:]:
  if not rn.endswith("bf16") or not L[rn]: continue
  pre=rn[:-len("bf16")]
  for n,_ in runs[1:]:
    if n==rn or not n.startswith(pre) or not L[n]: continue
    R=L[rn]; S=L[n]
    c=sorted(set(R[2]["correct"])&set(S[2]["correct"]))
    if not c: continue
    out=[]
    for k,sc,u in (("correct",100,"pt"),("grounded",100,"pt"),("landed",100,"pt"),("ns",1,"")):
        d=[S[2][k][q]-R[2][k][q] for q in c]; m=sum(d)/len(d)
        sd=(sum((x-m)**2 for x in d)/len(d))**0.5
        out.append(f"{k[:4]} {sc*m:+.1f}{u}±{sc*sd/math.sqrt(len(d)):.1f}")
    print(f"  vs {rn[5:]:12s} {n:16s} " + "  ".join(out) + f"  ({len(c)} q)")
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

if [ "$MODE" = "sft" ]; then
  # Reproduce, under quantization and in the compressed context, the traces that scored. This is
  # not matching bf16 - it is the quantized system learning the behaviour that worked, which is how
  # this lineage was trained when it lived in 4-bit. Scales and biases move; codes and pooler stay.
  SRUN=${SRUN:-s1}
  pkill -f "afterkee[p].sh"; pkill -f "evalkee[p].sh"; pkill -f "dwqkee[p].sh"; pkill -f "probekee[p].sh"; pkill -f "poolkee[p].sh"; pkill -f "jointkee[p].sh"; sleep 5
  if [ ! -s /root/work/dwq_calib/correct.jsonl ]; then
    for try in 1 2 3; do hf download baya1116/hypernet-sp-distill --include "pooler_distill/grpo_pool3/rollouts.jsonl" --local-dir /root/hfdl 2>&1 | tail -1 && break; sleep 10; done
    python3 - <<'PYS'
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/root/eval_hf200")
held = set()
for line in open("/root/work/eval300.jsonl"):
    try: held.add(json.loads(line).get("q", ""))
    except Exception: pass
n = 0
with open("/root/work/dwq_calib/correct.jsonl", "w") as out:
    for line in open("/root/hfdl/pooler_distill/grpo_pool3/rollouts.jsonl"):
        try: d = json.loads(line)
        except Exception: continue
        if not (d.get("correct") and d.get("grounded") and d.get("landed")): continue
        q, t = d.get("q", ""), d.get("text", "")
        if not q or not t or q in held: continue
        head = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False) + "<think>\n"
        out.write(json.dumps({"text": head + t}, ensure_ascii=False) + "\n"); n += 1
print(f"[sft] {n} correct, grounded, landed traces; held-out questions excluded")
PYS
  fi
  R=baya1116/hypernet-sp-distill; D=pooler_distill/grpo_pool3_step200_q4
  for p in p_t06q4 p_t06bf16; do
    for i in 0 1 2; do
      [ -s /root/work/${p}_out_$i.jsonl ] && hf upload $R /root/work/${p}_out_$i.jsonl $D/rollouts/${p}_$i.jsonl >/dev/null 2>&1
    done
  done
  echo "probe rollouts uploaded: $(cat /root/work/p_t06*_out_*.jsonl 2>/dev/null | wc -l)"
  echo "traces that scored: $(wc -l < /root/work/dwq_calib/correct.jsonl)"
  HF=/root/sft_hf_$SRUN; POOL=/root/pooler_sft_$SRUN.safetensors; LOG=/root/sft_$SRUN.log
  RUNNING=0
  pgrep -f "jointfit.p[y]" >/dev/null && { RUNNING=1; echo "already running: $(tail -1 $LOG)"; }
  [ -s $HF/model.safetensors ] && { RUNNING=1; echo "sft run $SRUN already finished"; }
  if [ "$RUNNING" = "0" ]; then
    pkill -f "pool_eval.p[y]"; sleep 8; pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 2
    rm -f $LOG
    cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/jointfit.py \
      --ckpt /root/pooler200.safetensors --data /root/work/dwq_calib/correct.jsonl --objective ce \
      --out-hf $HF --out-mlx /root/sft_mlx4_$SRUN --out-pooler $POOL --state /root/sft/$SRUN.pt --log $LOG \
      --clip-search 0 --lr-q ${SLRQ:-2e-6} --lr-p 0 --val 8 --val-every 2 --selftest 3 2>&1 | tail -14
    grep -q "^step 3 " $LOG 2>/dev/null || { echo "SFT SELFTEST FAILED - not launching"; exit 0; }
    rm -f $LOG
    cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/jointfit.py \
      --ckpt /root/pooler200.safetensors --data /root/work/dwq_calib/correct.jsonl --objective ce \
      --out-hf $HF --out-mlx /root/sft_mlx4_$SRUN --out-pooler $POOL --state /root/sft/$SRUN.pt --log $LOG \
      --clip-search 0 --lr-q ${SLRQ:-2e-6} --lr-p 0 --steps ${SSTEPS:-1500} --val 24 --val-every ${SVAL:-50} \
      >> /root/sft_run_$SRUN.log 2>&1 < /dev/null &
    sleep 20
  fi
  cat > /root/sftkeep.sh <<SKP
#!/bin/bash
SRUN=$SRUN; HF=$HF; POOL=$POOL
SKP
  cat >> /root/sftkeep.sh <<'SKP2'
until [ -s $HF/model.safetensors ] && grep -q JOINTFIT_DONE /root/sft_run_$SRUN.log 2>/dev/null; do sleep 60; done
pkill -f "jointfit.p[y]"; sleep 10
while :; do
  done=1
  for i in 0 1 2; do
    want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
    have=$(wc -l < /root/work/${SRUN}_out_$i.jsonl 2>/dev/null || echo 0)
    [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
    done=0
    pgrep -f "pool_eval.py .* /root/work/ev_$i.jsonl" >/dev/null && continue
    echo "[$SRUN-eval $(date -u +%H:%M)] shard $i at $have/$want - starting (quantized, trained on its own scored traces)"
    cd /root/work && SP_BASE=$HF SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py \
      /root/pooler200.safetensors /root/work/ev_$i.jsonl /root/work/${SRUN}_out_$i.jsonl \
      --n 999 --rw 768 --maxd 384 --samepage 1 --decode plain --temp ${STEMP:-0.9} --tag "[$SRUN$i]" \
      >> /root/${SRUN}_$i.log 2>&1 < /dev/null &
    sleep 60
  done
  [ "$done" = "1" ] && break
  sleep 120
done
echo "[$SRUN-eval $(date -u +%H:%M)] complete"
SKP2
  chmod +x /root/sftkeep.sh
  pkill -f "sftkee[p].sh"; sleep 1
  setsid nohup bash /root/sftkeep.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  sleep 30; t; echo "SFT_LAUNCH_DONE $SRUN $(date -u)"
  exit 0
fi

if [ "$MODE" = "idle" ]; then
  # bootstrapped and waiting for instructions; nothing below must run
  echo "idle $(date -u +%H:%M): $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader 2>/dev/null)"
  exit 0
fi
if [ "$MODE" = "publish2" ]; then
  # Everything worth keeping leaves the box before it is stopped.
  pkill -f "afterkee[p].sh"; pkill -f "evalkee[p].sh"; pkill -f "dwqkee[p].sh"; pkill -f "probekee[p].sh"
  pkill -f "poolkee[p].sh"; pkill -f "jointkee[p].sh"; pkill -f "sftkee[p].sh"; sleep 5
  pkill -f "pool_eval.p[y]"; sleep 8; pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 2
  export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
  R=baya1116/hypernet-sp-distill; D=pooler_distill/grpo_pool3_step200_q4
  # the directory the app would load must unpack to the values the evaluator measured
  [ -s /root/sft_mlx4_s1/model.safetensors ] && python3 /root/work/checkmlx.py /root/sft_mlx4_s1 /root/sft_hf_s1 2>&1 | tail -3
  python3 - <<'PYP' > /root/work/q4_metrics.json
import json, glob, collections, math, os

def load(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            try: rows.append(json.loads(line))
            except Exception: pass
    if not rows: return None
    per = {}
    for k in ("correct", "grounded", "landed", "ns"):
        d = collections.defaultdict(list)
        for r in rows: d[r.get("q", "")].append(float(r.get(k) or 0))
        per[k] = {q: sum(v)/len(v) for q, v in d.items()}
    return {"rollouts": len(rows), "questions": len(per["correct"]),
            "correct": round(100*sum(float(r.get("correct") or 0) for r in rows)/len(rows), 1),
            "grounded": round(100*sum(float(r.get("grounded") or 0) for r in rows)/len(rows), 1),
            "landed": round(100*sum(float(r.get("landed") or 0) for r in rows)/len(rows), 1),
            "searches": round(sum(float(r.get("ns") or 0) for r in rows)/len(rows), 2), "_per": per}

RUNS = [("bf16", "/root/work/ev_out_*.jsonl"), ("4bit_plain", "/root/work/q4_out_*.jsonl"),
        ("4bit_ste_lora", "/root/work/qa_out_*.jsonl"), ("4bit_dwq_temp2", "/root/work/dw_out_*.jsonl"),
        ("4bit_dwq_temp1", "/root/work/d3_out_*.jsonl"), ("4bit_group32", "/root/work/g32_out_*.jsonl"),
        ("4bit_joint_pooler", "/root/work/j1_out_*.jsonl"),
        ("4bit_self_trace_s1", "/root/work/s1_out_*.jsonl"),
        ("bf16_temp06", "/root/work/p_t06bf16_out_*.jsonl"), ("4bit_plain_temp06", "/root/work/p_t06q4_out_*.jsonl"),
        ("4bit_pooler_only", "/root/work/j2_out_*.jsonl"),
        ("4bit_float_embed", "/root/work/p_embfloat_out_*.jsonl")]
L = {n: load(p) for n, p in RUNS}
F = L["bf16"]

def paired(S):
    if not (F and S): return None
    c = sorted(set(F["_per"]["correct"]) & set(S["_per"]["correct"]))
    if not c: return None
    out = {"questions": len(c)}
    for k in ("correct", "grounded", "landed", "ns"):
        d = [S["_per"][k][q] - F["_per"][k][q] for q in c]
        m = sum(d)/len(d); sd = (sum((x-m)**2 for x in d)/len(d))**0.5
        sc = 1 if k == "ns" else 100
        out[k] = {"delta": round(sc*m, 1), "se": round(sc*sd/math.sqrt(len(c)), 1)}
    return out

print(json.dumps({
  "what": "What the phone's 4-bit conversion costs this model, and six attempts to recover it",
  "format": {"scheme": "MLX affine, group 64, 4 bits, scales and biases fp16",
             "tensors": "7 projections x 28 blocks + embed_tokens + lm_head = 198, 1776.9M weights",
             "relative_weight_error": {"overall": 9.51, "embed_tokens": 10.19, "lm_head": 10.35}},
  "heldout": {"file": "eval300.jsonl first 300 lines, 150 distinct questions x 2",
              "settings": {"rw": 768, "maxd": 384, "samepage": 1, "temp": 0.9, "gen": 1500, "maxs": 5}},
  "runs": {n: {k: v for k, v in (L[n] or {}).items() if k != "_per"} for n, _ in RUNS if L[n]},
  "paired_against_bf16": {n: paired(L[n]) for n, _ in RUNS[1:] if L[n]},
  "conclusion": "The loss is about seven points and none of the six attempts separates from the "
                "others or from doing nothing: they span -5.7 to -11.6 with standard errors of 3.2 "
                "to 5.5. Every run closed most of the KL gap to bf16 and recovered grounding; none "
                "recovered the score. See docs/quantization_4bit.md.",
}, ensure_ascii=False, indent=2))
PYP
  cat /root/work/q4_metrics.json
  hf upload $R /root/work/q4_metrics.json $D/q4_metrics.json 2>&1 | tail -1
  for f in /root/dwq_d3.log /root/dwq_d4.log /root/joint_j1.log /root/joint_j2.log /root/qat.log /root/sft_s1.log; do
    [ -s $f ] && hf upload $R $f $D/logs/$(basename $f) >/dev/null 2>&1
  done
  for p in q4 qa d3 d4 g32 j1 j2 p_embfloat p_t06q4 p_t06bf16 s1; do
    for i in 0 1 2; do
      [ -s /root/work/${p}_out_$i.jsonl ] && hf upload $R /root/work/${p}_out_$i.jsonl $D/rollouts/${p}_$i.jsonl >/dev/null 2>&1
    done
  done
  echo "--- deployable directories ---"
  for n in d3 d4; do
    [ -s /root/dwq_mlx4_$n/model.safetensors ] && { echo "dwq_$n $(du -shL /root/dwq_mlx4_$n | cut -f1)"; hf upload $R /root/dwq_mlx4_$n $D/dwq_${n}_mlx4 2>&1 | tail -1; }
  done
  [ -s /root/sft_mlx4_s1/model.safetensors ] && { echo "sft s1 $(du -shL /root/sft_mlx4_s1 | cut -f1)"; hf upload $R /root/sft_mlx4_s1 $D/sft_s1_mlx4 2>&1 | tail -1; }
  [ -s /root/sft/s1.pt ] && hf upload $R /root/sft/s1.pt $D/sft_s1_params.pt >/dev/null 2>&1
  [ -s /root/joint_hf_j1/model.safetensors ] && { echo "joint j1 $(du -shL /root/joint_hf_j1 | cut -f1)"; hf upload $R /root/joint_hf_j1 $D/joint_j1_hf 2>&1 | tail -1; }
  [ -s /root/pooler_joint_j1.safetensors ] && hf upload $R /root/pooler_joint_j1.safetensors $D/pooler_joint_j1.safetensors 2>&1 | tail -1
  # the pooler-only run: its model directory is the untouched 4-bit grid, so only the pooler is new
  [ -s /root/pooler_joint_j2.safetensors ] && hf upload $R /root/pooler_joint_j2.safetensors $D/pooler_only_j2.safetensors 2>&1 | tail -1
  [ -s /root/joint/j2.pt ] && hf upload $R /root/joint/j2.pt $D/pooler_only_j2_params.pt >/dev/null 2>&1
  [ -s /root/dwq/j1.pt ] && hf upload $R /root/dwq/j1.pt $D/joint_j1_params.pt >/dev/null 2>&1
  [ -s /root/joint/j1.pt ] && hf upload $R /root/joint/j1.pt $D/joint_j1_params.pt 2>&1 | tail -1
  [ -s /root/dwq/d4.pt ] && hf upload $R /root/dwq/d4.pt $D/dwq_d4_params.pt 2>&1 | tail -1
  echo "PUBLISH2_DONE $(date -u)"
  exit 0
fi

if [ "$MODE" = "joint" ]; then
  # The quantization parameters and the pooler, trained together against the 16-bit system, through
  # the compressed forward the evaluator actually runs. Everything before this trained on plain
  # contexts, so the pooler - where the compression happens - never saw a gradient at all.
  JRUN=${JRUN:-j1}
  pkill -f "afterkee[p].sh"; pkill -f "evalkee[p].sh"; pkill -f "dwqkee[p].sh"; pkill -f "probekee[p].sh"; pkill -f "poolkee[p].sh"; sleep 5
  HF=/root/joint_hf_$JRUN; POOL=/root/pooler_joint_$JRUN.safetensors; LOG=/root/joint_$JRUN.log
  RUNNING=0
  pgrep -f "jointfit.p[y]" >/dev/null && { RUNNING=1; echo "already running: $(tail -1 $LOG)"; }
  [ -s $HF/model.safetensors ] && { RUNNING=1; echo "joint run $JRUN already finished"; }
  if [ "$RUNNING" = "0" ]; then
    pkill -f "dwq.p[y]"; pkill -f "poolerfit.p[y]"; pkill -f "pool_eval.p[y]"; sleep 8
    pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 2
    python3 /root/work/q4.py || { echo "Q4 SELFTEST FAILED - not launching"; exit 0; }
    rm -f $LOG
    cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/jointfit.py \
      --ckpt /root/pooler200.safetensors --data /root/work/dwq_calib/train.jsonl \
      --out-hf $HF --out-mlx /root/joint_mlx4_$JRUN --out-pooler $POOL \
      --state /root/joint/$JRUN.pt --log $LOG --clip-search ${JCLIP:-1} --lr-q ${JLRQ:-2e-6} \
      --val 8 --val-every 2 --selftest 3 2>&1 | tail -16
    grep -q "^step 3 " $LOG 2>/dev/null || { echo "JOINTFIT SELFTEST FAILED - not launching"; exit 0; }
    rm -f $LOG
    cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/jointfit.py \
      --ckpt /root/pooler200.safetensors --data /root/work/dwq_calib/train.jsonl \
      --out-hf $HF --out-mlx /root/joint_mlx4_$JRUN --out-pooler $POOL \
      --state /root/joint/$JRUN.pt --log $LOG \
      --steps ${JSTEPS:-1200} --lr-q ${JLRQ:-2e-6} --lr-p ${JLRP:-1e-5} --clip-search ${JCLIP:-1} \
      --val 24 --val-every ${JVAL:-50} \
      >> /root/joint_run_$JRUN.log 2>&1 < /dev/null &
    sleep 20
  fi
  cat > /root/jointkeep.sh <<JKP
#!/bin/bash
JRUN=$JRUN; HF=$HF; POOL=$POOL
JKP
  cat >> /root/jointkeep.sh <<'JKP2'
until [ -s $HF/model.safetensors ] && grep -q JOINTFIT_DONE /root/joint_run_$JRUN.log 2>/dev/null; do sleep 60; done
pkill -f "jointfit.p[y]"; sleep 10
while :; do
  for i in 0 1 2; do
    want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
    have=$(wc -l < /root/work/${JRUN}_out_$i.jsonl 2>/dev/null || echo 0)
    [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
    pgrep -f "pool_eval.py .* /root/work/ev_$i.jsonl" >/dev/null && continue
    echo "[$JRUN-eval $(date -u +%H:%M)] shard $i at $have/$want - starting (quantized body + fitted pooler)"
    grep -viE "^\s*$" /root/${JRUN}_$i.log 2>/dev/null | tail -4 | cut -c1-200
    cd /root/work && SP_BASE=$HF SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py \
      $POOL /root/work/ev_$i.jsonl /root/work/${JRUN}_out_$i.jsonl \
      --n 999 --rw 768 --maxd 384 --samepage 1 --decode plain --tag "[$JRUN$i]" \
      >> /root/${JRUN}_$i.log 2>&1 < /dev/null &
    sleep 60
  done
  done=1
  for i in 0 1 2; do
    want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
    have=$(wc -l < /root/work/${JRUN}_out_$i.jsonl 2>/dev/null || echo 0)
    [ "$have" -ge "$want" ] || done=0
  done
  if [ "$done" = "1" ]; then
    # The untouched 4-bit arm is the reference every other number leans on and it stopped at 174
    # rollouts. pool_eval resumes from its own output, so this only runs what is missing.
    for i in 0 1 2; do
      want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
      have=$(wc -l < /root/work/q4_out_$i.jsonl 2>/dev/null || echo 0)
      [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
      pgrep -f "pool_eval.py .* /root/work/ev_$i.jsonl" >/dev/null && continue
      echo "[baseline $(date -u +%H:%M)] plain 4-bit shard $i at $have/$want - filling in"
      cd /root/work && SP_BASE=/root/eval_hf200 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py \
        /root/pooler200.safetensors /root/work/ev_$i.jsonl /root/work/q4_out_$i.jsonl \
        --n 999 --rw 768 --maxd 384 --samepage 1 --decode plain --q4 1 --tag "[q4$i]" \
        >> /root/q4_$i.log 2>&1 < /dev/null &
      sleep 60
    done
  fi
  sleep 120
done
JKP2
  chmod +x /root/jointkeep.sh
  pkill -f "jointkee[p].sh"; sleep 1
  setsid nohup bash /root/jointkeep.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  sleep 30; t; echo "JOINT_LAUNCH_DONE $JRUN $(date -u)"
  exit 0
fi

if [ "$MODE" = "pool" ]; then
  # Fit the pooler to the embedding table the phone actually feeds it. Ships as a pooler file:
  # no model directory, no size change, no app code.
  pkill -f "afterkee[p].sh"; pkill -f "evalkee[p].sh"; pkill -f "dwqkee[p].sh"; pkill -f "probekee[p].sh"; sleep 5
  PRUN=${PRUN:-p_poolfit}
  RUNNING=0
  pgrep -f "poolerfit.p[y]" >/dev/null && { RUNNING=1; echo "already running: $(tail -1 /root/poolerfit.log)"; }
  [ -s /root/pooler_q4.safetensors ] && { RUNNING=1; echo "pooler already fitted"; }
  if [ "$RUNNING" = "0" ]; then
    pkill -f "dwq.p[y]"; pkill -f "pool_eval.p[y]"; sleep 8; pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 2
    cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/poolerfit.py \
      --ckpt /root/pooler200.safetensors --out /root/pooler_q4.safetensors \
      --data /root/work/dwq_calib/train.jsonl --selftest 3 2>&1 | tail -14
    grep -q "^step 3 " /root/poolerfit.log 2>/dev/null || { echo "POOLERFIT SELFTEST FAILED - not launching"; exit 0; }
    cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/poolerfit.py \
      --ckpt /root/pooler200.safetensors --out /root/pooler_q4.safetensors \
      --data /root/work/dwq_calib/train.jsonl --steps ${PSTEPS:-3000} --lr ${PLR:-1e-5} \
      >> /root/poolerfit_run.log 2>&1 < /dev/null &
    sleep 20
  fi
  cat > /root/poolkeep.sh <<PKP
#!/bin/bash
PRUN=$PRUN
PKP
  cat >> /root/poolkeep.sh <<'PKP2'
until [ -s /root/pooler_q4.safetensors ] && grep -q POOLERFIT_DONE /root/poolerfit_run.log 2>/dev/null; do sleep 30; done
pkill -f "poolerfit.p[y]"; sleep 5
while :; do
  for i in 0 1 2; do
    want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
    have=$(wc -l < /root/work/${PRUN}_out_$i.jsonl 2>/dev/null || echo 0)
    [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
    pgrep -f "pool_eval.py .* /root/work/ev_$i.jsonl" >/dev/null && continue
    echo "[$PRUN $(date -u +%H:%M)] shard $i at $have/$want - starting (pooler fitted to the 4-bit table)"
    grep -viE "^\s*$" /root/${PRUN}_$i.log 2>/dev/null | tail -4 | cut -c1-200
    cd /root/work && SP_BASE=/root/eval_hf200 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py \
      /root/pooler_q4.safetensors /root/work/ev_$i.jsonl /root/work/${PRUN}_out_$i.jsonl \
      --n 999 --rw 768 --maxd 384 --samepage 1 --decode plain --q4 1 --tag "[$PRUN$i]" \
      >> /root/${PRUN}_$i.log 2>&1 < /dev/null &
    sleep 60
  done
  sleep 120
done
PKP2
  chmod +x /root/poolkeep.sh
  pkill -f "poolkee[p].sh"; sleep 1
  setsid nohup bash /root/poolkeep.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  sleep 30; t; echo "POOLFIT_LAUNCH_DONE $(date -u)"
  exit 0
fi

if [ "$MODE" = "probe" ]; then
  # Three training attempts have each closed most of the KL gap to bf16 and left the score where it
  # was. Before a fourth, ask a different question: is the format itself the binding constraint?
  # Group 32 halves how many weights share a scale. It needs one line changed in the app and a
  # reconversion, so it is a recommendation rather than a drop-in - but if it recovers the gap, that
  # is the answer, and if it does not, no amount of training the group-64 grid will help either.
  PRUN=${PRUN:-g32}; PG=${PG:-64}; PSKIP=${PSKIP:-}; PTEMP=${PTEMP:-0.9}; PQ4=${PQ4:-1}
  # PRUNS overrides the single run: a space-separated list of tag:q4:temp, worked through in order
  PRUNS=${PRUNS:-"$PRUN:$PQ4:$PTEMP"}
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
PRUNS="$PRUNS"; PG=$PG; PSKIP="$PSKIP"
PKQ
  cat >> /root/probekeep.sh <<'PKQ2'
# a training run in flight owns the card; let it finish and write its directory first
while pgrep -f "dwq.p[y]" >/dev/null; do sleep 60; done
for spec in $PRUNS; do
  PRUN=${spec%%:*}; rest=${spec#*:}; PQ4=${rest%%:*}; PTEMP=${rest#*:}
  Q4FLAGS=""; [ "$PQ4" = "1" ] && Q4FLAGS="--q4 1 --q4group $PG"
  [ "$PQ4" = "1" ] && [ -n "$PSKIP" ] && Q4FLAGS="$Q4FLAGS --q4skip $PSKIP"
  while :; do
    done=1
    for i in 0 1 2; do
      want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
      have=$(wc -l < /root/work/${PRUN}_out_$i.jsonl 2>/dev/null || echo 0)
      [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
      done=0
      pgrep -f "pool_eval.py .* /root/work/ev_$i.jsonl" >/dev/null && continue
      echo "[$PRUN-probe $(date -u +%H:%M)] shard $i at $have/$want - starting (q4=$PQ4 temp=$PTEMP group $PG)"
      grep -viE "^\s*$" /root/${PRUN}_$i.log 2>/dev/null | tail -4 | cut -c1-200
      cd /root/work && SP_BASE=/root/eval_hf200 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py \
        /root/pooler200.safetensors /root/work/ev_$i.jsonl /root/work/${PRUN}_out_$i.jsonl \
        --n 999 --rw 768 --maxd 384 --samepage 1 --decode plain --temp $PTEMP $Q4FLAGS --tag "[$PRUN$i]" \
        >> /root/${PRUN}_$i.log 2>&1 < /dev/null &
      sleep 60
    done
    [ "$done" = "1" ] && break
    sleep 120
  done
  echo "[$PRUN-probe $(date -u +%H:%M)] complete"
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
RUN=$RUN; HF=$HF; ET=${DEVALTEMP:-0.9}
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
      --n 999 --rw 768 --maxd 384 --samepage 1 --decode plain --temp $ET --tag "[$RUN$i]" \
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
# CTL-END
