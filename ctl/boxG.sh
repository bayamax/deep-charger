# PEEK (one-shot): the thinking in tokens, not words, both sides
python3 - <<'PYS'
import json, re, statistics, glob
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/root/eval_hf200")
INFO = re.compile(r"<information>.*?</information>", re.S)
rows = [json.loads(l) for l in open("/root/online_g5/rollouts.jsonl") if l.strip()]
mx = max(r["step"] for r in rows)
print(f"TOK through step {mx}")
for kind in ("reason", "search"):
    x = [r for r in rows if r.get("kind") == kind and r["step"] > mx - 60]
    if not x: continue
    th = sorted(len(tok.encode(INFO.sub("", r["text"].split("</think>")[0]), add_special_tokens=False)) for r in x)
    rp = sorted(len(tok.encode(r["text"].split("</think>")[-1], add_special_tokens=False)) for r in x if "</think>" in r["text"])
    n = len(th)
    print(f"TOK {kind}: last 60 steps, n={n} | thinking tokens med {th[n//2]} p75 {th[int(n*.75)]} p90 {th[int(n*.9)]} max {th[-1]} | reply tokens med {rp[len(rp)//2] if rp else 0}")
    ref = [json.loads(l) for l in open("/root/hfdl/pooler_distill/chatsft/dolphin_v2.jsonl") if l.strip()]
    if kind == "reason":
        rt = sorted(len(tok.encode(r.get("thinking",""), add_special_tokens=False)) for r in ref)
        print(f"TOK   the reference answers it is scored against: med {rt[len(rt)//2]} p90 {rt[int(len(rt)*.9)]} max {rt[-1]}")
PYS
exit 0
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
MODE=reeval
RRUN=g5dolph
RQSRC=dolphin
RQN=12
RMODEL=g5
RKIND=ckpt
RSHARDS=1
RTEMP=0.6
RCAP=600
RGEN=7000
RN=12
RRUN=g3dolph
RQSRC=dolphin
RQN=12
RMODEL=g3
RKIND=ckpt
RSHARDS=1
RTEMP=0.6
RCAP=600
RGEN=4000
RN=100
ORUN=g5
OMODEL=s4_hf
OB=8
ODRATIO=1
ODMIN=1
ODOLPHIN=dolphin_v2.jsonl
OLR=1e-4
OACCUM=2
OLAYERS=all
OREPLAY=replay_v1.jsonl
OREPLAYFILTER=1
OTRAINALL=1
OQUEUE=1
OCOMPLETE=1
OGEN=7000
OBUDGET=1500
OGUARD=0
# the reasoning problems. dolphin_v2 is the short-thinking subset (reference thinking 245 words at the
# median, 300 at the most); dolphin_v1 is the whole set, 6776 problems whose references think 718 words
# at the median and 1384 at the ninth decile. Move to v1 when v2 stops teaching anything, which reads as:
# the constrained-format problems stop scoring zero across all twelve, or the mean reward stops moving
# over a hundred steps. Changing this line and adding a restart marker is the whole switch.
OREASON=dolphin_v2.jsonl
OREASONG=12
OSEARCHEVERY=2
OWHEELS=1
OWTALK=0.5
OSEARCHDEMO=
OSEARCHWHEELS=0
OREASONSTUB=0
OJUDGEAPI=openai
OJUDGEMODEL=gpt-5-nano
OMAXSRCH=7
OJUDGE=0
OREPLAYN=0
OROLLEVERY=1
OQFILE=pooler_distill/measureq.jsonl
ORESUME=0
OSTEPS=200
OTEMP=0.6
SHARDS=3
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
for f in pool_eval.py q4.py qat.py dwq.py poolerfit.py jointfit.py checkmlx.py packmlx.py sft_lora.py selfgen_gpu.py build_merged.py web_search.py grpo_pool.py online_loop.py; do
  for try in 1 2 3; do curl -sS -o /root/work/$f "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/work/$f && break; sleep 5; done
done
cp /root/work/web_search.py /root/work/runtime/web_search.py 2>/dev/null
echo "fetched: pool_eval $(wc -l < /root/work/pool_eval.py) lines, q4 $(wc -l < /root/work/q4.py) lines"

# The status command is the same whatever mode this file is in, so it is installed once,
# before the modes - a probe that forgets to reinstall it would otherwise report a stale table.
cat > /usr/local/bin/s <<'SSD'
#!/bin/bash
# the score over the run: mean reward per block, both sides, beside what produced it
d=$(ls -td /root/online_*/ 2>/dev/null | head -1)
python3 - "$d" "${1:-20}" <<'PYS'
import json, re, statistics, sys
d, W = sys.argv[1], int(sys.argv[2])
INFO = re.compile(r"<information>.*?</information>", re.S)


def think_words(t):
    """the model's own thinking: the served pages sit in the same span and are not it"""
    return len(INFO.sub("", t.split("</think>")[0]).split())


def reply_words(t):
    return len(t.split("</think>")[-1].split()) if "</think>" in t else 0
rows = [json.loads(l) for l in open(d + "/rollouts.jsonl") if l.strip()]
mx = max(r["step"] for r in rows)
print(f"{d.rstrip('/').split('/')[-1]}  through step {mx}   ({W}-step blocks)")
print("  steps    | reason  mean think reply unfin | search  mean srch think reply unfin")
for b in range(0, mx, W):
    rea = [r for r in rows if b < r["step"] <= b + W and r.get("kind") == "reason"]
    sea = [r for r in rows if b < r["step"] <= b + W and r.get("kind") == "search"]
    if not rea and not sea: continue
    def col(x, s=False):
        if not x: return "    -     -    -    -    -    -" if s else "    -     -    -    -    -"
        th = statistics.median(think_words(r["text"]) for r in x)
        rp = statistics.median(reply_words(r["text"]) for r in x)
        un = 100 * sum(1 for r in x if "</think>" not in r["text"]) / len(x)
        p = 100 * sum(1 for r in x if r["reward"] >= 1.0) / len(x)
        m = sum(r["reward"] for r in x) / len(x)
        ns = f" {statistics.median([r['ns'] for r in x]):>3.0f}" if s else ""
        return f"{p:>5.0f}% {m:>+6.2f}{ns} {th:>5.0f} {rp:>5.0f} {un:>4.0f}%"
    print(f"  {b+1:>4}-{b+W:<4} | {col(rea)} | {col(sea, True)}")
for k in ("reason", "search"):
    v = [r["reward"] for r in rows if r.get("kind") == k]
    if v: print(f"  {k}: {len(v)} samples, mean {sum(v)/len(v):+.2f}, pass {100*sum(1 for x in v if x>=1)/len(v):.0f}%")
PYS
SSD
chmod +x /usr/local/bin/s
cat > /usr/local/bin/t <<'TTD'
#!/bin/bash
echo "$(date -u +%H:%M)Z loop $(pgrep -fc 'online_loop.p[y]')  eval $(pgrep -fc 'pool_eval.p[y]')  $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)  $(df -h /root | tail -1 | awk '{print $4" free"}')"
# the run itself: the last steps, then the distribution that the pass rates hide
for f in $(ls -t /root/online_*.log 2>/dev/null | head -1); do
  grep -E "^\[step|^ONLINE_ROLLBACK|^ONLINE_COLLAPSE" "$f" | tail -5 | cut -c1-200
done
for d in $(ls -td /root/online_*/ 2>/dev/null | head -1); do
python3 - "$d" <<'PYT' 2>/dev/null
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1] + "/rollouts.jsonl") if l.strip()]
mx = max(r["step"] for r in rows)
for kind in ("reason", "search"):
    x = [r for r in rows if r.get("kind") == kind and r["step"] > mx - 20]
    if not x: continue
    ns = [r["ns"] for r in x]
    th = [len(r["text"].split("</think>")[0].split()) for r in x]
    print(f"  last 20 steps {kind}: pass {100*sum(r['reward'] for r in x)/len(x):.0f}% | searches med {statistics.median(ns):.0f} max {max(ns)} | think med {statistics.median(th):.0f} | unfinished {sum(1 for r in x if '</think>' not in r['text'])}/{len(x)}")
PYT
done
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
  m["zs"]=sum(1 for r in rows if not r.get("ns"))/len(rows)
  m["tt"]=sum(1 for r in rows if r.get("tail_tags") or ("</think>" in (r.get("text") or "") and "<search>" in (r.get("text") or "").split("</think>")[-1]))/len(rows)
  return len(rows), m, per
runs=[("bf16","/root/work/ev_out_*.jsonl"),("4bit plain","/root/work/q4_out_*.jsonl"),
    ("4bit +STE-lora","/root/work/qa_out_*.jsonl"),("4bit +dwq d1","/root/work/dw_out_*.jsonl")]
for p in sorted(glob.glob("/root/work/d[0-9]_out_0.jsonl")):
  t=os.path.basename(p).split("_")[0]
  runs.append((f"4bit +dwq {t}", f"/root/work/{t}_out_*.jsonl"))
for p in sorted(glob.glob("/root/work/g[0-9]*_out_0.jsonl")):
  t=os.path.basename(p).split("_")[0]
  runs.append((f"4bit {t} plain", f"/root/work/{t}_out_*.jsonl"))
for p in sorted(glob.glob("/root/work/j[0-9]_out_0.jsonl")) + sorted(glob.glob("/root/work/s[0-9]_out_0.jsonl")) + sorted(glob.glob("/root/work/p_*_out_0.jsonl")) + sorted(glob.glob("/root/work/g[0-9]_out_0.jsonl")):
  t=os.path.basename(p)[:-len("_out_0.jsonl")]
  runs.append((f"4bit {t}", f"/root/work/{t}_out_*.jsonl"))
L={n:load(p) for n,p in runs}
for n,_ in runs:
  S=L[n]
  if S: print(f"  {n:16s} {100*S[1]['correct']:5.1f}%  gnd {100*S[1]['grounded']:3.0f}%  "
              f"land {100*S[1]['landed']:3.0f}%  srch {S[1]['ns']:.1f}  zero-srch {100*S[1]['zs']:.0f}%  tail-tags {100*S[1]['tt']:.0f}%  ({S[0]} roll)")
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
if [ "$MODE" = "pack" ]; then
  # jointfit measured a dequantized directory and saved its trained scales and biases, but never
  # wrote the packed directory the app loads. Build it, prove it unpacks to what was measured, ship it.
  export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
  R=baya1116/hypernet-sp-distill; D=pooler_distill/grpo_pool3_step200_q4
  SRUN=${SRUN:-s1}
  ls -la /root/sft/$SRUN.pt /root/sft_hf_$SRUN/model.safetensors 2>&1 | tail -2
  cd /root/work && python3 /root/work/packmlx.py --base /root/eval_hf200 --hf /root/sft_hf_$SRUN \
    --state /root/sft/$SRUN.pt --out /root/sft_mlx4_$SRUN 2>&1 | tail -4
  python3 /root/work/checkmlx.py /root/sft_mlx4_$SRUN /root/sft_hf_$SRUN 2>&1 | tail -4 | tee /root/checkmlx_$SRUN.txt
  grep -q MLX_CHECK_OK /root/checkmlx_$SRUN.txt || { echo "PACK_FAILED $SRUN - not uploading"; exit 0; }
  echo "--- uploading ($(du -shL /root/sft_mlx4_$SRUN | cut -f1)) ---"
  hf upload $R /root/sft_mlx4_$SRUN $D/sft_${SRUN}_mlx4 2>&1 | tail -1
  hf upload $R /root/sft/$SRUN.pt $D/sft_${SRUN}_params.pt 2>&1 | tail -1
  hf upload $R /root/sft_$SRUN.log $D/logs/sft_$SRUN.log >/dev/null 2>&1
  hf upload $R /root/sft_run_$SRUN.log $D/logs/sft_run_$SRUN.log >/dev/null 2>&1
  for i in 0 1 2; do hf upload $R /root/work/${SRUN}_out_$i.jsonl $D/rollouts/${SRUN}_$i.jsonl >/dev/null 2>&1; done
  echo "PACK_DONE $SRUN $(date -u)"
  exit 0
fi
if [ "$MODE" = "publish3" ]; then
  # credit is nearly gone: ship what the s2 measurement has produced so far, without stopping it
  export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
  R=baya1116/hypernet-sp-distill
  for i in 0 1 2; do [ -s /root/work/s2_out_$i.jsonl ] && hf upload $R /root/work/s2_out_$i.jsonl pooler_distill/chatsft/rollouts/s2_$i.jsonl 2>&1 | tail -1; done
  for i in 0 1 2; do [ -s /root/s2_$i.log ] && hf upload $R /root/s2_$i.log pooler_distill/chatsft/logs/s2_$i.log >/dev/null 2>&1; done
  [ -s /root/sft2_run_s2.log ] && hf upload $R /root/sft2_run_s2.log pooler_distill/chatsft/logs/sft2_run_s2.log >/dev/null 2>&1
  echo "PUBLISH3_DONE $(cat /root/work/s2_out_*.jsonl 2>/dev/null | wc -l) rollouts $(date -u)"
  exit 0
fi
if [ "$MODE" = "online" ]; then
  # the on-the-fly alternating loop: rollouts -> accepted ones trained at once -> a Dolphin step -> repeat
  pkill -f "afterkee[p].sh"; pkill -f "evalkee[p].sh"; pkill -f "genkee[p].sh"; pkill -f "sft2kee[p].sh"; pkill -f "reevalkee[p].sh"; pkill -f "onlinekee[p].sh"; sleep 2
  export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
  [ -s /root/.dsk ] || { [ -n "$DSK_KEY" ] && printf '%s' "$DSK_KEY" > /root/.dsk && chmod 600 /root/.dsk; }
  [ -s /root/.oai ] || { [ -n "$OAI_KEY" ] && printf '%s' "$OAI_KEY" > /root/.oai && chmod 600 /root/.oai; }
  R=baya1116/hypernet-sp-distill
  OHF=/root/hfdl/pooler_distill/chatsft/$OMODEL
  for try in 1 2 3 4 5 6; do hf download $R --include "pooler_distill/chatsft/${OMODEL}/*" --local-dir /root/hfdl >/dev/null 2>&1; [ -s $OHF/model.safetensors ] && break; sleep 30; done
  [ -s $OHF/model.safetensors ] || { echo "ONLINE_ABORT: $OMODEL not on the hub"; exit 0; }
  for f in pooler_distill/chatsft/${ODOLPHIN:-dolphin_v1.jsonl} pooler_distill/chatsft/${OREASON:-dolphin_v2.jsonl} pooler_distill/chatsft/${OREPLAY:-replay_v1.jsonl} pooler_distill/selfq_0.jsonl pooler_distill/selfq_1.jsonl pooler_distill/selfq_2.jsonl; do
    [ -s /root/hfdl/$f ] || for try in 1 2 3; do hf download $R --include "$f" --local-dir /root/hfdl >/dev/null 2>&1; [ -s /root/hfdl/$f ] && break; sleep 10; done
  done
  if [ -n "${OSEARCHDEMO:-}" ]; then
    [ -s /root/hfdl/$OSEARCHDEMO ] || for try in 1 2 3; do hf download $R --include "$OSEARCHDEMO" --local-dir /root/hfdl >/dev/null 2>&1; [ -s /root/hfdl/$OSEARCHDEMO ] && break; sleep 10; done
  fi
  if [ -n "${OQFILE:-}" ]; then
    [ -s /root/hfdl/$OQFILE ] || for try in 1 2 3; do hf download $R --include "$OQFILE" --local-dir /root/hfdl >/dev/null 2>&1; [ -s /root/hfdl/$OQFILE ] && break; sleep 10; done
    cp /root/hfdl/$OQFILE /root/work/selfq_all.jsonl
  else cat /root/hfdl/pooler_distill/selfq_?.jsonl > /root/work/selfq_all.jsonl; fi; cp /root/hfdl/pooler_distill/chatsft/${ODOLPHIN:-dolphin_v1.jsonl} /root/work/dolphin_v1.jsonl
  # r5: drop replay traces that carry the stray thought markers or a repetition loop (41 of 672 did; replaying them raised the marker rate from 20% to 35% of rollouts in r4)
  REPLAYF=/root/hfdl/pooler_distill/chatsft/${OREPLAY:-replay_v1.jsonl}
  if [ "${OREPLAYFILTER:-0}" = "1" ]; then
    python3 - "$REPLAYF" /root/work/replay_clean.jsonl <<'PYF'
import json, re, sys
degen = re.compile(r"begin_of_thought|end_of_thought|\b(\w+(?:\W+\w+){0,3})\b(?:\W+\1\b){4,}")
keep = [l for l in open(sys.argv[1]) if l.strip() and not degen.search(l)]
open(sys.argv[2], "w").writelines(keep); print(f"[replay filter] kept {len(keep)}")
PYF
    REPLAYF=/root/work/replay_clean.jsonl
  fi
  OUT=/root/online_$ORUN; mkdir -p $OUT
  if [ ! -f /root/.frozen_${ORUN}_150 ] && [ -s $OUT/latest.safetensors ]; then
    for f in latest.safetensors state.json loop.log rollouts.jsonl; do
      [ -s $OUT/$f ] && hf upload $R $OUT/$f pooler_distill/chatsft/online/${ORUN}_step150/$f >/dev/null 2>&1
    done
    touch /root/.frozen_${ORUN}_150; echo "ONLINE_FROZEN ${ORUN}_step150 $(date -u)"
  fi
  # every checkpoint is ~3.5 GB and every run keeps two (latest and the guard's healthy copy): six finished
  # runs filled the disk and killed r10 at step 5 with "No space left on device" while it wrote one.
  for d in /root/online_*/; do [ "$d" = "$OUT/" ] || rm -f $d/latest.safetensors $d/good.safetensors $d/*.tmp; done
  rm -f /root/reeval_*.safetensors /root/online_*/latest.safetensors.tmp
  echo "[disk] $(df -h /root | tail -1 | awk '{print $3" used, "$4" free"}')"
  # resume from the hub copy of this run (uploaded every 30 min) when this box did not start it
  if [ "${ORESUME:-0}" = "1" ] && [ ! -s $OUT/state.json ]; then
    for f in latest.safetensors state.json loop.log accepted.jsonl rollouts.jsonl; do
      for try in 1 2 3 4 5 6; do hf download $R --include "pooler_distill/chatsft/online/$ORUN/$f" --local-dir /root/hfdl >/dev/null 2>&1; [ -s /root/hfdl/pooler_distill/chatsft/online/$ORUN/$f ] && break; sleep 20; done
      [ -s /root/hfdl/pooler_distill/chatsft/online/$ORUN/$f ] && cp /root/hfdl/pooler_distill/chatsft/online/$ORUN/$f $OUT/$f
    done
    echo "ONLINE_RESUME $ORUN from step $(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null) $(date -u)"
  fi
  # one-time: the first run used 8 rollouts per step and 6 GB of a 24 GB card; restart with 16 (the loop resumes from its state file)
  # 16 rollouts per step took 4.5 min against 1.4 min for 8 (per rollout slower, not faster): back to 8, once
  # r4: retention by replaying fixed verified search traces (not by training on the newest own samples, which sharpened into repetition twice); rollouts every 10 steps only to measure the skip rates and top up the replay set
  # once: the measurement questions move from nq_open to the GRPO pool (the loop resumes from its local state)
  if [ ! -f /root/.restart_${ORUN}_q ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_q; echo "ONLINE_RESTART $ORUN questions=$OQFILE $(date -u)"; fi
  # once: the searches trip needs an absolute floor as well (the capped mean sits near 3, and a
  # window holding a couple of capped rollouts would otherwise read as a runaway)
  # once: the guard also watches the share of rollouts cut for searching without end, which is the
  # signal that moves first (5%% at the start of r11, 11%% by step 360 while the mean stayed near 3)
  # once: the judge decides nothing now that every finished rollout is trained on, so stop calling it
  # thinking has grown to about 850 tokens on the search side, which is what chain of thought is for;
  # what it collided with was the budget, so the budget moves rather than the thinking being penalised
  # one row that keeps reading holds the whole batch: the wall-clock ceiling comes down so a step
  # costs about twenty-five minutes at worst, which still leaves a long rollout room to finish
  # past the fifth search the environment returns a notice and no content, so a rollout that issues
  # 181 of them is reading nothing: it is cut at seven, two past the last one that can serve a page
  if [ ! -f /root/.restart_${ORUN}_srch7 ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_srch7; echo "ONLINE_RESTART $ORUN maxsrch 7 $(date -u)"; fi
  if [ ! -f /root/.restart_${ORUN}_budget ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_budget; echo "ONLINE_RESTART $ORUN budget 1500 $(date -u)"; fi
  if [ ! -f /root/.restart_${ORUN}_gen7k ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_gen7k; echo "ONLINE_RESTART $ORUN gen 7000 $(date -u)"; fi
  if [ ! -f /root/.restart_${ORUN}_nojudge ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_nojudge; echo "ONLINE_RESTART $ORUN judge off $(date -u)"; fi
  if [ ! -f /root/.restart_${ORUN}_cut ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_cut; echo "ONLINE_RESTART $ORUN guard cut share $(date -u)"; fi
  if [ ! -f /root/.restart_${ORUN}_nsfloor ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_nsfloor; echo "ONLINE_RESTART $ORUN guard floor $(date -u)"; fi
  # once: a search group that scores zero now learns from that question's own teacher trajectory
  if [ ! -f /root/.restart_${ORUN}_demo ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_demo; echo "ONLINE_RESTART $ORUN search demos $(date -u)"; fi
  if [ ! -f /root/.restart_$ORUN ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_$ORUN; echo "ONLINE_RESTART $ORUN dolphin ratio $ODRATIO min $ODMIN $(date -u)"; fi
  if ! pgrep -f "online_loop.p[y]" >/dev/null; then
    cd /root/work && DSK_KEY=$(cat /root/.dsk 2>/dev/null) OAI_KEY=$(cat /root/.oai 2>/dev/null) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/online_loop.py $OHF $OUT \
      --questions /root/work/selfq_all.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl \
      --b $OB --steps $OSTEPS --temp $OTEMP --lr $OLR --dolphin-ratio ${ODRATIO:-2} --dolphin-min ${ODMIN:-2} --accum ${OACCUM:-1} --replay $REPLAYF --replay-per-step ${OREPLAYN:-2} --rollout-every ${OROLLEVERY:-1} --train-all ${OTRAINALL:-0} --queue ${OQUEUE:-0} --complete-only ${OCOMPLETE:-0} --gen ${OGEN:-1500} --budget ${OBUDGET:-900} --guard ${OGUARD:-0} --maxsrch ${OMAXSRCH:-0} --judge ${OJUDGE:-1} ${OREASON:+--reason /root/hfdl/pooler_distill/chatsft/$OREASON --reason-g ${OREASONG:-8} --reason-stub ${OREASONSTUB:-0} --judge-api ${OJUDGEAPI:-deepseek} --judge-model ${OJUDGEMODEL:-} --search-every ${OSEARCHEVERY:-0} --w-talk ${OWTALK:-0.5} --wheels ${OWHEELS:-0} --search-wheels ${OSEARCHWHEELS:-0} ${OSEARCHDEMO:+--search-demo /root/hfdl/$OSEARCHDEMO}} --pooler none --lora-rank 16 --lora-layers ${OLAYERS:-all} --gradckpt 1 --save-every 5 --stop eos \
      >> /root/online_$ORUN.log 2>&1 < /dev/null &
  fi
  # once: in reasoning mode the empty search loop printed the end marker at launch, and the keeper
  # read it as the run being over and stopped uploading. Strip it so the keeper runs to the real end.
  if [ ! -f /root/.fixed_marker_$ORUN ]; then sed -i '/^ONLINE_LOOP_DONE$/d' /root/online_$ORUN.log 2>/dev/null; touch /root/.fixed_marker_$ORUN; echo "ONLINE_MARKER_FIX $ORUN $(date -u)"; fi
  # only what this launch wrote counts as its end: a previous run's marker sits in the same log and
  # twice now has made the keeper quit at the first upload, leaving the run saving nothing
  MARK=$(( $(pgrep -f "online_loop.p[y]" >/dev/null && cat /root/.logmark_$ORUN 2>/dev/null || wc -c < /root/online_$ORUN.log 2>/dev/null || echo 0) + 1 ))
  [ -f /root/.logmark_$ORUN ] || echo $((MARK - 1)) > /root/.logmark_$ORUN
  cat > /root/onlinekeep.sh <<OK
#!/bin/bash
ORUN=$ORUN; OUT=$OUT; R=$R; MARK=$MARK
OK
  cat >> /root/onlinekeep.sh <<'OKB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
last=0
while :; do
  now=$(date +%s)
  if [ $((now - last)) -ge 1800 ] || tail -c +$MARK /root/online_$ORUN.log 2>/dev/null | grep -q "ONLINE_LOOP_DONE"; then
    [ -s $OUT/latest.safetensors ] && hf upload $R $OUT/latest.safetensors pooler_distill/chatsft/online/$ORUN/latest.safetensors >/dev/null 2>&1
    for f in loop.log accepted.jsonl rollouts.jsonl state.json; do [ -s $OUT/$f ] && hf upload $R $OUT/$f pooler_distill/chatsft/online/$ORUN/$f >/dev/null 2>&1; done
    hf upload $R /root/online_$ORUN.log pooler_distill/chatsft/online/$ORUN/run.log >/dev/null 2>&1
    last=$now; echo "[online $ORUN $(date -u +%H:%M)] uploaded | $(tail -1 $OUT/loop.log 2>/dev/null | cut -c1-200)"
    /usr/local/bin/s 20 2>/dev/null | sed 's/^/SCORE /' 
    grep -hE "^ONLINE_ROLLBACK|^ONLINE_COLLAPSE|^\[guard\]" /root/online_$ORUN.log 2>/dev/null | tail -3
    tail -c +$MARK /root/online_$ORUN.log 2>/dev/null | grep -q "ONLINE_LOOP_DONE" && { echo "ONLINE_DONE $ORUN $(date -u)"; break; }
    pgrep -f "online_loop.p[y]" >/dev/null || { echo "ONLINE_DIED $ORUN: $(grep -E 'Error|error|Traceback' /root/online_$ORUN.log | tail -2 | cut -c1-160)"; break; }
  fi
  sleep 60
done
OKB
  chmod +x /root/onlinekeep.sh; setsid nohup bash /root/onlinekeep.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  sleep 90; tail -3 /root/online_$ORUN.log | cut -c1-200; echo "ONLINE_LAUNCH_DONE $ORUN $(date -u)"; exit 0
fi

if [ "$MODE" = "reeval" ]; then
  pkill -f "onlinekee[p].sh"; pkill -f "online_loop.p[y]"; sleep 5; pkill -9 -f "online_loop.p[y]" 2>/dev/null
  # Re-measure an earlier checkpoint under the fixed conversational stop rule (EOS terminal), so the
  # table compares like with like: RMODEL is "base" (the step-200 student) or an adapter name on the hub.
  RRUN=${RRUN:-basefix}; RMODEL=${RMODEL:-base}   # re-run for the no-search set
  pkill -f "afterkee[p].sh"; pkill -f "evalkee[p].sh"; pkill -f "genkee[p].sh"; pkill -f "sft2kee[p].sh"; pkill -f "reevalkee[p].sh"; sleep 3
  pkill -f "pool_eval.p[y]"; sleep 8; pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 2   # a re-run must not leave the old evaluator holding the card
  export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
  R=baya1116/hypernet-sp-distill
  RCKPT=${RCKPT:-/root/pooler200.safetensors}
  if [ "${RKIND:-adapter}" = "ckpt" ]; then
    # RMODEL names a run under chatsft/online: its latest.safetensors is a full state dict, so the
    # structure comes from the step-200 directory and every weight is overwritten at load time.
    RHF=/root/eval_hf200; RCKPT=/root/reeval_$RRUN.safetensors
    if [ -s /root/online_$RMODEL/latest.safetensors ]; then cp /root/online_$RMODEL/latest.safetensors $RCKPT
    else
      for try in 1 2 3 4 5 6; do hf download $R --include "pooler_distill/chatsft/online/$RMODEL/latest.safetensors" --local-dir /root/hfdl >/dev/null 2>&1; [ -s /root/hfdl/pooler_distill/chatsft/online/$RMODEL/latest.safetensors ] && break; sleep 20; done
      cp /root/hfdl/pooler_distill/chatsft/online/$RMODEL/latest.safetensors $RCKPT 2>/dev/null
    fi
    [ -s $RCKPT ] || { echo "REEVAL_ABORT $RRUN: online/$RMODEL checkpoint not found"; exit 0; }
  elif [ "$RMODEL" = "base" ]; then RHF=/root/eval_hf200; elif [ "${RKIND:-adapter}" = "dir" ]; then
    RHF=/root/hfdl/pooler_distill/chatsft/$RMODEL   # a merged model directory published by the training box
    for try in $(seq 1 60); do   # it may still be uploading: poll for up to an hour
      hf download $R --include "pooler_distill/chatsft/${RMODEL}/*" --local-dir /root/hfdl >/dev/null 2>&1
      [ -s $RHF/model.safetensors ] && [ -s $RHF/config.json ] && break; sleep 60
    done
    [ -s $RHF/model.safetensors ] || { echo "REEVAL_ABORT $RRUN: $RMODEL not on the hub"; exit 0; }
  else
    RHF=/root/reeval_hf_$RMODEL
    if [ ! -s $RHF/model.safetensors ]; then
      for try in 1 2 3; do hf download $R --include "pooler_distill/chatsft/${RMODEL}/*" --local-dir /root/hfdl 2>&1 | tail -1; [ "${PIPESTATUS[0]}" -eq 0 ] && break; sleep 10; done
      cd /root/work && python3 - <<PYM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("/root/eval_hf200", torch_dtype=torch.bfloat16)
m = PeftModel.from_pretrained(base, "/root/hfdl/pooler_distill/chatsft/$RMODEL").merge_and_unload()
m.save_pretrained("$RHF", safe_serialization=True); AutoTokenizer.from_pretrained("/root/eval_hf200").save_pretrained("$RHF"); print("MERGED $RMODEL")
PYM
    fi
    [ -s $RHF/model.safetensors ] || { echo "REEVAL_ABORT $RRUN: merge failed"; exit 0; }
  fi
  if [ "${RQSRC:-}" = "dolphin" ]; then
    [ -s /root/hfdl/pooler_distill/chatsft/dolphin_v2.jsonl ] || hf download $R --include "pooler_distill/chatsft/dolphin_v2.jsonl" --local-dir /root/hfdl >/dev/null 2>&1
    python3 - <<PYD
import json
rows = [json.loads(l) for l in open("/root/hfdl/pooler_distill/chatsft/dolphin_v2.jsonl") if l.strip()]
out = open("/root/work/dolphinq.jsonl", "w")
for r in rows[:${RQN:-12}]:
    out.write(json.dumps({"q": r["q"]}, ensure_ascii=False) + "\n")
print("[dolphin questions]", min(len(rows), ${RQN:-12}))
PYD
  fi
  [ -s /root/hfdl/pooler_distill/chat_eval60.jsonl ] || hf download $R --include "pooler_distill/chat_eval60.jsonl" --local-dir /root/hfdl 2>&1 | tail -1
  cat > /root/reevalkeep.sh <<RK
#!/bin/bash
RRUN=$RRUN; RHF=$RHF; R=$R; RCKPT=$RCKPT; RQSRC=${RQSRC:-}; RSHARDS=${RSHARDS:-3}; RTEMP=${RTEMP:-0.9}; RCAP=${RCAP:-600}; RGEN=${RGEN:-1500}; RN=${RN:-999}
RK
  cat >> /root/reevalkeep.sh <<'RKB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
run_one() {  # $1 questions file, $2 out file, $3 tag
  want=$(wc -l < "$1"); [ "${RN:-999}" -lt "$want" ] && want=${RN:-999}
  [ -s "$2" ] && [ "$(wc -l < "$2")" -ge "$want" ] && return 0
  cd /root/work && SP_BASE=$RHF SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/pool_eval.py     $RCKPT "$1" "$2" --n $RN --rw 768 --maxd 384 --samepage 1 --decode plain --temp $RTEMP --gen $RGEN --stop eos --replycap $RCAP --tag "[$3]" >> /root/${RRUN}_$3.log 2>&1
  [ -s "$2" ] && [ "$(wc -l < "$2")" -ge "$want" ] || { echo "REEVAL_ABORT $RRUN at $3: $(tail -1 /root/${RRUN}_$3.log | cut -c1-100)"; exit 1; }
}
if [ -n "$RQSRC" ]; then
  run_one /root/work/dolphinq.jsonl /root/work/${RRUN}_out_0.jsonl ${RRUN}0
  python3 - <<'PYT'
import json
import glob
for r in [json.loads(l) for l in open(sorted(glob.glob("/root/work/*dolph_out_0.jsonl"))[-1]) if l.strip()]:
    think, _, reply = r["text"].partition("</think>")
    print(f"DTEXT ===== {r['q'][:220]}")
    print(f"DTEXT searches {r['ns']} | think {len(think.split())} words | reply {len(reply.split())} words")
    print("DTEXT reply:", reply.strip()[:900].replace("\n", " "))
PYT
  echo "REEVAL_DONE $RRUN $(date -u)"; exit 0
fi
for i in $(seq 0 $((${RSHARDS:-3} - 1))); do run_one /root/work/ev_$i.jsonl /root/work/${RRUN}_out_$i.jsonl ${RRUN}$i; hf upload $R /root/work/${RRUN}_out_$i.jsonl pooler_distill/chatsft/rollouts/${RRUN}_$i.jsonl >/dev/null 2>&1; echo "[$RRUN] shard $i uploaded $(tail -1 /root/${RRUN}_${RRUN}$i.log | cut -c1-110)"; done
run_one /root/hfdl/pooler_distill/chat_eval60.jsonl /root/work/${RRUN}_chat_out.jsonl ${RRUN}chat
hf upload $R /root/work/${RRUN}_chat_out.jsonl pooler_distill/chatsft/rollouts/${RRUN}_chat.jsonl >/dev/null 2>&1
python3 - <<'PYT'
import json, glob
for f in sorted(glob.glob("/root/work/" + "s4prod" + "_out_*.jsonl")):
    for r in [json.loads(l) for l in open(f) if l.strip()][:6]:
        think, _, reply = r["text"].partition("</think>")
        print(f"TEXT ===== {r['q'][:100]} | gold {r['gold']} | correct {r['correct']} grounded {r['grounded']} searches {r['ns']}")
        print("TEXT queries:", "; ".join(r.get("queries", [])[:8]))
        print("TEXT think tail:", think[-700:].replace("\n", " "))
        print("TEXT reply:", reply.strip()[:900].replace("\n", " "))
PYT
echo "REEVAL_DONE $RRUN $(date -u)"
RKB
  chmod +x /root/reevalkeep.sh; setsid nohup bash /root/reevalkeep.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  sleep 5; echo "REEVAL_LAUNCH_DONE $RRUN model=$RHF $(date -u)"; exit 0
fi

if [ "$MODE" = "sft2" ]; then
  # Supervised fine-tuning of the step-200 student on its own search prefixes continued by a
  # teacher: the thinking after the last result and a conversational reply. Then the held-out
  # measurement, so the reply style is checked on the same 150 questions as everything else.
  SRUN2=${SRUN2:-s2}; SDATA=${SDATA:-pooler_distill/chatsft/search_sft_v2.jsonl}
  pkill -f "afterkee[p].sh"; pkill -f "evalkee[p].sh"; pkill -f "dwqkee[p].sh"; pkill -f "probekee[p].sh"; pkill -f "poolkee[p].sh"; pkill -f "jointkee[p].sh"; pkill -f "sftkee[p].sh"; pkill -f "genkee[p].sh"; sleep 5
  export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
  R=baya1116/hypernet-sp-distill
  for try in 1 2 3; do hf download $R --include "$SDATA" --local-dir /root/hfdl 2>&1 | tail -1 && break; sleep 10; done
  cp /root/hfdl/$SDATA /root/work/sft2_data.jsonl
  # replay: strict single-sentence QA traces of the same student, so the search reflex is not traded away
  if [ "${S2REPLAY:-0}" -gt 0 ]; then
    [ -s /root/hfdl/pooler_distill/grpo_pool3/rollouts.jsonl ] || for try in 1 2 3; do hf download $R --include "pooler_distill/grpo_pool3/rollouts.jsonl" --local-dir /root/hfdl 2>&1 | tail -1 && break; sleep 10; done
    python3 - <<PYR
import json, random
held=set()
for line in open("/root/work/eval300.jsonl"):
    try: held.add(json.loads(line).get("q",""))
    except Exception: pass
rows=[]; seen=set()
for line in open("/root/hfdl/pooler_distill/grpo_pool3/rollouts.jsonl"):
    try: d=json.loads(line)
    except Exception: continue
    if not (d.get("correct") and d.get("grounded") and d.get("landed")): continue
    q,t=d.get("q",""),d.get("text","")
    if not q or not t or q in held or (q,t[:200]) in seen: continue
    seen.add((q,t[:200])); rows.append({"q":q,"text":t})
random.seed(0); random.shuffle(rows); rows=rows[:$S2REPLAY]
with open("/root/work/sft2_data.jsonl","a") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")
print(f"[sft2] replay {len(rows)} strict traces appended")
PYR
  fi
  echo "traces: $(wc -l < /root/work/sft2_data.jsonl)"
  HF2=/root/sft2_hf_$SRUN2; LOG2=/root/sft2_$SRUN2.log
  RUNNING=0
  pgrep -f "sft_lora.p[y]" >/dev/null && { RUNNING=1; echo "already training: $(tail -1 $LOG2)"; }
  [ -s $HF2/model.safetensors ] && { RUNNING=1; echo "sft2 $SRUN2 already finished"; }
  if [ "$RUNNING" = "0" ]; then
    pkill -f "pool_eval.p[y]"; sleep 8; pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 2
    rm -f $LOG2
    cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/sft_lora.py \
      --base /root/eval_hf200 --data /root/work/sft2_data.jsonl --out $HF2 --log $LOG2 \
      --rank ${S2RANK:-16} --lr ${S2LR:-3e-5} --epochs ${S2EPOCHS:-2} --accum ${S2ACCUM:-8} \
      >> /root/sft2_run_$SRUN2.log 2>&1 < /dev/null &
    sleep 20
  fi
  cat > /root/sft2keep.sh <<SK2
#!/bin/bash
SRUN2=$SRUN2; HF2=$HF2; R=$R
SK2
  cat >> /root/sft2keep.sh <<'SK2B'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
until [ -s $HF2/model.safetensors ] && grep -q SFT2_DONE /root/sft2_run_$SRUN2.log 2>/dev/null; do sleep 60; done
pkill -f "sft_lora.p[y]"; sleep 10
hf upload $R ${HF2}_adapter pooler_distill/chatsft/${SRUN2}_adapter >/dev/null 2>&1
hf upload $R /root/sft2_$SRUN2.log pooler_distill/chatsft/logs/sft2_$SRUN2.log >/dev/null 2>&1
echo "[$SRUN2] adapter and log uploaded"
while :; do
  done=1
  for i in 0 1 2; do
    want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
    have=$(wc -l < /root/work/${SRUN2}_out_$i.jsonl 2>/dev/null || echo 0)
    [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
    done=0
    pgrep -f "pool_eval.py .* /root/work/ev_$i.jsonl" >/dev/null && continue
    echo "[$SRUN2-eval $(date -u +%H:%M)] shard $i at $have/$want - starting"
    cd /root/work && SP_BASE=$HF2 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py       /root/pooler200.safetensors /root/work/ev_$i.jsonl /root/work/${SRUN2}_out_$i.jsonl       --n 999 --rw 768 --maxd 384 --samepage 1 --decode plain --temp ${S2TEMP:-0.9} --stop eos --replycap 200 --tag "[$SRUN2$i]"       >> /root/${SRUN2}_$i.log 2>&1 < /dev/null &
    sleep 60
  done
  if [ "$done" = "1" ]; then
    for i in 0 1 2; do hf upload $R /root/work/${SRUN2}_out_$i.jsonl pooler_distill/chatsft/rollouts/${SRUN2}_$i.jsonl >/dev/null 2>&1; done
    echo "SFT2_EVAL_DONE $SRUN2 $(date -u)"; break
  fi
  sleep 120
done
SK2B
  chmod +x /root/sft2keep.sh
  pkill -f "sft2kee[p].sh"; sleep 1
  setsid nohup bash /root/sft2keep.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  sleep 40; tail -3 /root/sft2_run_$SRUN2.log 2>/dev/null | cut -c1-160; echo "SFT2_LAUNCH_DONE $SRUN2 $(date -u)"
  exit 0
fi
if [ "$MODE" = "selfgen" ]; then
  # The no-search side, on-policy: the lineage's base writes candidate replies to real prompts; a judge
  # picks later, off the box. Then hand the card to the nq_open generation.
  export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
  R=baya1116/hypernet-sp-distill
  pkill -f "genkee[p].sh"; sleep 2
  for try in 1 2 3; do hf download $R --include "pooler_distill/chatsft/chat_prompts.jsonl" --local-dir /root/hfdl 2>&1 | tail -1 && break; sleep 10; done
  [ -s /root/base_distill/model.safetensors ] || for try in 1 2 3; do hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir /root/base_distill 2>&1 | tail -1 && break; sleep 10; done
  SGOUT=${SGOUT:-self_cands.jsonl}
  if [ ! -s /root/work/$SGOUT ] || ! grep -q SELFGEN_DONE /root/selfgen.log 2>/dev/null; then
    cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/selfgen_gpu.py \
      --base /root/base_distill --prompts /root/hfdl/pooler_distill/chatsft/chat_prompts.jsonl --out /root/work/$SGOUT \
      --k ${SGK:-4} --batch ${SGBATCH:-8} --maxnew ${SGMAXNEW:-360} > /root/selfgen.log 2>&1
    tail -2 /root/selfgen.log
  fi
  hf upload $R /root/work/$SGOUT pooler_distill/chatsft/$SGOUT 2>&1 | tail -1
  echo "SELFGEN_UPLOADED $(wc -l < /root/work/$SGOUT) candidates $(date -u)"
  [ "${SGCHAIN:-1}" = "1" ] || exit 0
  MODE=gen   # fall through into the nq_open generation with the same card
fi
if [ "$MODE" = "gen" ]; then
  # Data generation for the conversational lineage: the step-200 student answers real questions
  # (nq_open, people's own search queries) through the same environment the evaluator uses, one
  # rollout each. Correct, grounded traces become the search prefixes a teacher continues.
  GRUN=${GRUN:-g1}; GN=${GN:-1500}; GTEMP=${GTEMP:-0.8}; GPAR=${GPAR:-3}   # GPAR: evaluators at once (1 on a 12 GB card)
  pkill -f "afterkee[p].sh"; pkill -f "evalkee[p].sh"; pkill -f "dwqkee[p].sh"; pkill -f "probekee[p].sh"; pkill -f "poolkee[p].sh"; pkill -f "jointkee[p].sh"; pkill -f "sftkee[p].sh"; sleep 5
  export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
  R=baya1116/hypernet-sp-distill
  GQFILE=${GQFILE:-pooler_distill/nq_pool.jsonl}   # any {"q","gold"} jsonl on the HF repo (e.g. a chat-register rewrite pool)
  [ -s /root/hfdl/$GQFILE ] || for try in 1 2 3; do hf download $R --include "$GQFILE" --local-dir /root/hfdl 2>&1 | tail -1 && break; sleep 10; done
  python3 - <<PYG
import json
qs=[l for l in open("/root/hfdl/$GQFILE") if l.strip()][:$GN]
for i in range(3): open(f"/root/work/gq_{i}.jsonl","w").writelines(qs[i::3])
print(f"[gen] {len(qs)} questions -> {[len(qs[i::3]) for i in range(3)]}")
PYG
  for i in 0 1 2; do for try in 1 2 3; do hf download $R --include "pooler_distill/nq_gen/${GRUN}_$i.jsonl" --local-dir /root/hfdl 2>/dev/null | tail -1 && break; sleep 5; done
    [ -s /root/hfdl/pooler_distill/nq_gen/${GRUN}_$i.jsonl ] && cp /root/hfdl/pooler_distill/nq_gen/${GRUN}_$i.jsonl /root/work/${GRUN}_out_$i.jsonl; done
  echo "restored $(cat /root/work/${GRUN}_out_*.jsonl 2>/dev/null | wc -l) rollouts"
  cat > /root/genkeep.sh <<GKQ
#!/bin/bash
GRUN=$GRUN; GTEMP=$GTEMP; R=$R; GPAR=$GPAR
GKQ
  cat >> /root/genkeep.sh <<'GKQ2'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
last_up=0
while :; do
  done=1
  for i in 0 1 2; do
    want=$(wc -l < /root/work/gq_$i.jsonl 2>/dev/null || echo 0)
    have=$(wc -l < /root/work/${GRUN}_out_$i.jsonl 2>/dev/null || echo 0)
    [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
    done=0
    pgrep -f "pool_eval.py .* /root/work/gq_$i.jsonl" >/dev/null && continue
    [ "$(pgrep -fc "pool_eval.p[y]")" -ge "$GPAR" ] && continue
    echo "[$GRUN-gen $(date -u +%H:%M)] shard $i at $have/$want - starting (temp $GTEMP)"
    cd /root/work && SP_BASE=/root/eval_hf200 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py       /root/pooler200.safetensors /root/work/gq_$i.jsonl /root/work/${GRUN}_out_$i.jsonl       --n 9999 --rw 768 --maxd 384 --samepage 1 --decode plain --temp $GTEMP --tag "[$GRUN$i]"       >> /root/${GRUN}_$i.log 2>&1 < /dev/null &
    sleep 60
  done
  now=$(date +%s)
  if [ "$done" = "1" ] || [ $((now - last_up)) -ge 1800 ]; then
    for i in 0 1 2; do [ -s /root/work/${GRUN}_out_$i.jsonl ] && hf upload $R /root/work/${GRUN}_out_$i.jsonl pooler_distill/nq_gen/${GRUN}_$i.jsonl >/dev/null 2>&1; done
    last_up=$now; echo "[$GRUN-gen $(date -u +%H:%M)] uploaded $(cat /root/work/${GRUN}_out_*.jsonl 2>/dev/null | wc -l) rollouts"
  fi
  [ "$done" = "1" ] && { echo "GEN_DONE $GRUN $(date -u)"; break; }
  sleep 120
done
GKQ2
  chmod +x /root/genkeep.sh
  pkill -f "genkee[p].sh"; sleep 1
  setsid nohup bash /root/genkeep.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  sleep 30; echo "GEN_LAUNCH_DONE $GRUN n=$GN $(date -u)"
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
