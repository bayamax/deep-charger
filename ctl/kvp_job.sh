#!/bin/bash
# Pooler retraining with the model's own attention as the prior (the user: the pooler sits in the attention-sink valley -
# half its mass on ~20 sink tokens whatever the distant length, so the memory's content hardly changes with the past; put
# the model's actual attention proportions where the pooler takes its attention mass). --att-prior A adds A x log(share) of
# the attention each token received from the following tokens while it sat in the raw window (decoder layers 8-19) to the
# pooler's attention logits; the first two generated tokens (the model's own sinks) are zeroed. The LLM (v38's tuned base +
# LoRA) stays frozen, the pooler starts from v38 and is distilled again as ctl/kvm_job.sh did (KL to the full-KV teacher on
# chunks after a full window, 16 chunks per trace, 600 traces per segment, lr 1e-4), with the prior on. After each segment:
# the held-out KL by domain (with the prior); every 2nd segment the massprof (importance, effective mass) and the 60
# problems at seed 1. Baselines first: v38's held-out KL without and with the (untrained) prior, and the massprof.
# Stops at a plateau of the held-out KL (3 segments without a 1% gain) or STOP_KP; then the 60 problems at seed 2 too.
# FROM=base (the user: retrain the pooler from the base model): a new pooler on the original model, no checkpoint - the
# kvm run's recipe exactly plus the prior, so its held-out KL per segment is read against kvm's curve (segment 1 0.0918,
# 2 ~0.08, 5 0.0675, 8 0.0648; the raw window alone 0.1073). The baselines are skipped (nothing to measure yet).
# CAP=c (--att-cap): segments 1-2 with the prior alone matched kvm's KL and scored 39/60 (kvm 35) with the mass spread over
# hundreds of tokens, but by segment 4 the pooler's own logits had grown past the prior (50% of the mass on 46 tokens again,
# 31/60, KL unchanged at kvm's 0.0691) - the KL does not see the collapse, the 60 problems do. With the cap the pooler's logits
# are bounded to +-c before the prior is added, so it can reweight a token by at most e^(2c) and cannot collapse. EVERY=n:
# the 60 problems every n segments (1: every segment, since they are the measure the KL misses).
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
get() { for _i in 1 2 3; do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o "$2" "$H/$1" && return 0; sleep 10; done; return 1; }
for _t in $(seq 60); do curl -sS -o sp_new.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q "att-cap" sp_new.py && { mv sp_new.py sp_qwen3.py; break; }; sleep 120; done
curl -sS -o sp_mathcheck.py "$RAW/tools/sp_mathcheck.py?nocache=$(date +%s)"
LG=/root/w/sp_kvp.log; LR=${LR:-1e-4}; FROM=${FROM:-v38}
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
AP="--att-prior ${AP:-1.0} --att-layers ${AL:-8-19}"; CAP=${CAP:-0}; [ "$CAP" != 0 ] && AP="$AP --att-cap $CAP"   # CAP: the pooler's own logits bounded to +-CAP (0 = off)
EVERY=${EVERY:-2}   # massprof + the 60 problems every EVERY segments
if [ ! -s rm_pool.jsonl ]; then   # the KV-memory distillation data (ctl/kvm_job.sh): the gm_* pool, 1500 at the end held out
  for n in $(python3 -c "
from huggingface_hub import HfApi
print(' '.join(sorted(f.rsplit('/', 1)[-1][3:5] for f in HfApi().list_repo_files('$R') if f.startswith('archive/sp3/more/gm_'))))"); do get archive/sp3/more/gm_$n.jsonl gm_$n.jsonl; done
  python3 -c "
import glob, json, random
rs = [l for f in sorted(glob.glob('gm_*.jsonl')) for l in open(f) if l.strip()]
random.Random(0).shuffle(rs); tail = rs[-1500:]; by = {}
for l in tail: by.setdefault(json.loads(l).get('src', '?').split('-')[0], []).append(l)
out = []
for k in sorted(by): out += by[k][:13]
open('rm_held.jsonl', 'w').writelines(out); open('rm_pool.jsonl', 'w').writelines(rs[:-1500])
print('[kp] pool', len(rs) - 1500, '| held-out', len(out))"; rm -f gm_*.jsonl
fi
[ -s kvg_best.pt ] || get archive/sp3/final/spKVF_best_v38.pt kvg_best.pt
[ -s kvp.pt ] || [ $FROM = base ] || cp kvg_best.pt kvp.pt
[ -s kvg_v38_s1.jsonl ] || get archive/sp3/kvg/kvg_v38_s1.jsonl kvg_v38_s1.jsonl
echo "[kp] ready $(date -u) | att-cap in the script: $(grep -c 'def capped_cross' sp_qwen3.py) | $AP | lr $LR | start $FROM | from $(cat kvp_seg 2>/dev/null || echo 0) segments | eval every $EVERY"
score() { python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open(sys.argv[1])]
print(f'{sum(correct(r) for r in rs)}/{len(rs)} correct, closed {sum(\"</think>\" in r[\"text\"] for r in rs)}')" "$1"; }
evkl() { out=""; for d in chat code comp math mc; do python3 -c "
import json, sys; [sys.stdout.write(l) for l in open('rm_held.jsonl') if json.loads(l).get('src', '?').split('-')[0] == sys.argv[1]]" $d > held_$d.jsonl
  k=$(python3 sp_qwen3.py eval --traces held_$d.jsonl --max-len 6144 $M "$@" --log $LG 2>&1 | grep "\[sp\] eval" | tail -1 | grep -oP "KL \K[0-9.]+(?=  top1)"); out="$out $d $k"; done
  echo "$out | mean $(echo "$out" | python3 -c "import sys; v = sys.stdin.read().split()[1::2]; print(round(sum(map(float, v)) / len(v), 4))")"; }
mprof() {   # mprof <ckpt> <label>: the pooler's mass, the importance and the effective mass with the prior over the 60 problems' v38 texts
  python3 sp_qwen3.py massprof --traces kvg_v38_s1.jsonl $M $AP --ckpt $1 --max-len 12288 --log $LG > run_kp_mp.log 2>&1
  grep -E "massprof (importance|effective) .*distant (2048|4096)|massprof distant (2048|4096)|massprof importance top|Error|Traceback" run_kp_mp.log | cut -c1-400 | sed "s/^\[sp\]/[kp] $2/"; }
ev60() {   # ev60 <ckpt> <seed> <label>  (the label may hold spaces: they leave the file name)
  o=kvp_$(echo "$3" | tr -d " ")_s$2.jsonl; rm -f $o
  python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $M $AP --ckpt $1 --max-len 12288 --roll-batch 48 --grpo-tok-budget 60000 --seed $2 --roll-out $o --log $LG > run_kpgen.log 2>&1
  grep -E "Error|Traceback" run_kpgen.log | tail -1 | sed "s/^/[kp] $3 gen: /"; put $o kvp/$o > /dev/null
  echo "[kp] $3 seed $2: $(score $o) (v38 plain: 44 / 41, raw window 42 / 44) $(date -u)"; }
[ $FROM = base ] && touch kvp_base_done
if [ ! -e kvp_base_done ]; then
  echo "[kp] baseline v38, held-out KL by domain: no prior:$(evkl --ckpt kvp.pt)"
  echo "[kp] baseline v38, held-out KL by domain: prior on, pooler untrained:$(evkl --ckpt kvp.pt $AP)"
  mprof kvp.pt "baseline v38"; touch kvp_base_done
fi
seg=$(cat kvp_seg 2>/dev/null || echo 0)
# PRE_EV=<ckpt>:<label>: one 60-problem run at seed 1 before the loop (the segment-2 run of 07:55 fell to the file-name bug above)
[ -n "$PRE_EV" ] && ev60 "${PRE_EV%%:*}" 1 "${PRE_EV##*:}"
[ -n "$PRE_MP" ] && mprof "${PRE_MP%%:*}" "${PRE_MP##*:}"   # PRE_MP=<ckpt>:<label>: one massprof before the loop
while [ ! -e STOP_KP ]; do seg=$((seg + 1)); t0=$(date +%s); x="--const-lr"; [ $seg = 1 ] && x=""
  CK=kvp.pt; [ -s kvp.pt ] || CK=""   # FROM=base: segment 1 starts a new pooler on the original model
  python3 sp_qwen3.py train --traces rm_pool.jsonl --skip $(( ((seg - 1) * 600) % 9600 )) --n 600 --max-len 8192 $M $AP --chunks-per-trace 16 --accum 8 --lr $LR $x --lora-lr 0 ${CK:+--ckpt $CK} --seed $((100 + seg)) --out kvp_next.pt --log $LG > run_kp.log 2>&1
  if [ -s kvp_next.pt ] && grep -q "^\[sp\] trained" run_kp.log; then mv kvp_next.pt kvp.pt; echo $seg > kvp_seg
  else echo "[kp] segment $seg FAILED: $(grep -E 'Error' run_kp.log | tail -1 | cut -c1-200)"; break; fi
  [ $seg = 1 ] && grep "att-prior" run_kp.log | head -1 | cut -c1-300 | sed 's/^\[sp\]/[kp]/'
  tk=$(grep -oP "KL [0-9.]+ top1 [0-9.]+%" run_kp.log | awk 'NR==1{f=$0} {l=$0} END{print "first 25 " f " / last 25 " l}')
  ev=$(evkl --ckpt kvp.pt $AP)
  echo "[kp] segment $seg ($(( $(date +%s) - t0 )) s, OOM skips $(grep -c 'out of memory' run_kp.log)): train $tk | held-out KL by domain, prior on:$ev"
  echo "$ev" | grep -oP "mean \K[0-9.]+" >> kvp_hist; put kvp.pt spKVP_latest.pt; put $LG kvp/kvp.log > /dev/null
  if [ $((seg % EVERY)) = 0 ]; then mprof kvp.pt "segment $seg"; ev60 kvp.pt 1 "segment $seg"; fi
  if python3 -c "
import sys; h = [float(x) for x in open('kvp_hist')]
sys.exit(0 if len(h) > 3 and min(h[-3:]) > min(h[:-3]) * 0.99 else 1)"; then echo "[kp] plateau ($(tr '\n' ' ' < kvp_hist))"; break; fi
done
[ $((seg % EVERY)) = 0 ] || ev60 kvp.pt 1 "segment $seg"
ev60 kvp.pt 2 "segment $seg"
echo "[kp] JOB_DONE $(date -u)"
