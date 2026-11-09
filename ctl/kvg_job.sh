#!/bin/bash
# GRPO with raw-window guides (my call, the user left the timing to me): the second unfrozen DPO round did not generalise -
# chosen-ahead on its first, unseen pairs 48% / 56% (round 1: 80%); its pairs are mostly closed-but-wrong, a content difference
# DPO did not spread to new problems. Here the collectors are rollout workers (ctl/kvc_job.sh MODE=grpo): per problem the
# memory G=4 times with the newest weights + the raw window once (a right-and-closed raw draw joins the group as a guide with
# reward 1), uploaded to archive/sp3/dpo_col_g/. The trainer (this box): once NEWG new problems are in, GRPO on their groups
# (--grpo-groups: reward 1 = closed and right, advantage = reward - group mean, uniform groups dropped, 4 groups per update,
# the model unfrozen at lr 1e-6, hypernet 2e-5), publish (spKVF_latest.pt + kvf/VERSION, the same names the collectors read),
# the 60 problems at seed 1; while waiting it works its own shard as a rollout worker.
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
for _t in $(seq 60); do curl -sS -o sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q '"--grad-sketch"' sp_qwen3.py && break; sleep 120; done
LG=/root/w/sp_kvg.log; NEWG=${NEWG:-24}; NSH=${NSH:-4}; EVERY=${EVERY:-3}; GMIN=${GMIN:-12}
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
gt() {   # the GRPO flags, re-read every round: kvg_accum / kvg_ftlr / kvg_ftlayers in /root/w override the defaults without a restart
  echo "--grpo-groups --ft-layers $(cat kvg_ftlayers 2>/dev/null || echo 28) --opt8bit --ft-lr $(cat kvg_ftlr 2>/dev/null || echo ${FTLR:-1e-6}) --lr $(cat kvg_plr 2>/dev/null || echo 2e-5) --lora $(cat kvg_lora 2>/dev/null || echo 0) --lora-lr $(cat kvg_loralr 2>/dev/null || echo 1e-5) --epochs 1 --accum $(cat kvg_accum 2>/dev/null || echo ${ACCUM:-4}) --const-lr --grpo-wrong $(cat kvg_wrong 2>/dev/null || echo 0) --opt-state kvg_opt.pt"; }
GT=$(gt)
rd=$(cat kvg_round 2>/dev/null || echo 0); v0=$(cat VERSION 2>/dev/null || echo 0)
if [ ! -s kvg.pt ]; then   # the starting weights go out first, so the rollout workers sample from the policy being trained
  cp ${START:-kvf.pt} kvg.pt; cp kvg.pt kvf.pt; v0=$((v0 + 1)); echo $v0 > VERSION; echo $v0 > kvg_v0
  put kvg.pt spKVF_latest.pt && put VERSION kvf/VERSION; fi
v0=$(cat kvg_v0 2>/dev/null || echo $v0)
echo "[kg] ready $(date -u) | grpo-groups in the script: $(grep -c '"--grpo-groups"' sp_qwen3.py) | start ${START:-kvf.pt} (version $v0) | $GT"
ev60() {   # ev60 <round> <version>: the 60 problems at seed 1 with kvg.pt
  o=kvg_v$2_s1.jsonl; rm -f $o   # (an older run's file of the same name was appended to once)
  python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $M --ckpt kvg.pt --max-len 12288 --roll-batch 48 --grpo-tok-budget 60000 --seed 1 --roll-out $o --log $LG > run_kggen.log 2>&1
  put $o kvg/$o; echo "[kg] round $1 (weights v$2) seed 1: $(ok60 $o | awk '{print $1 "/60 correct, closed " $2}')"; put $LG kvg/kvg.log; }
auto_lr() {   # auto_lr <round> <version> (the user: tune the lr to the accuracy as we go). After the round's eval, from E = the 60
  # problems, W = closed-but-wrong and C = closed on the round's own groups (first 4 draws; the policy that drew them is the
  # one evaluated): a new best E is kept as kvg_best.pt; E <= best-4 or W > 28% -> back to the best, lr halved, optimizer
  # state dropped, the versions since skipped; W > 22% or C > 82% -> lr halved; W < 18%, C < 78%, E >= best-1 -> lr x1.1 (never
  # above 1e-5: steady steps, the user). E = the two-seed mean when the workers' seed-2 parts are on the hub (waited up to 40 min). Decisions in kvg_auto.txt (hub kvg/auto.txt).
  local act; act=$(python3 - $1 $2 <<'PY3'
import json, sys, os; sys.path.insert(0, '.')
from sp_mathcheck import correct
rd, v = int(sys.argv[1]), int(sys.argv[2])
rs = [json.loads(l) for l in open(f'kvg_v{v}_s1.jsonl')]; E1 = sum(correct(r) for r in rs); E = E1; E2 = None
try:   # seed 2 from the workers (4 parts of 15), waited for up to 40 min: the best is chosen on the two-seed mean
    import time
    from huggingface_hub import HfApi, hf_hub_download as d
    ps = [f for f in HfApi().list_repo_files('baya1116/hypernet-sp-distill') if f'/evalx_v{v}_s2_p' in f]
    if len(ps) >= 4:
        E2 = sum(correct(json.loads(l)) for f in ps for l in open(d('baya1116/hypernet-sp-distill', f))); E = (E1 + E2) / 2
except Exception as e_: pass
n = cl = cw = 0
for l in open(f'kvg_groups_r{rd}.jsonl'):
    g = json.loads(l)
    for t in g['samples'][:4]:
        c = '</think>' in t; ok = c and correct({'text': t, 'gold': g['gold']}); n += 1; cl += c; cw += c and not ok
W, C = cw / max(n, 1), cl / max(n, 1)
lr = float(open('kvg_loralr').read()) if os.path.exists('kvg_loralr') else 1e-5
tgt, lo_, hi_ = 'kvg_loralr', 2.5e-6, 1e-5
if lr == 0:   # the LLM frozen: the controller steers the hypernet's lr instead
    tgt, lo_, hi_ = 'kvg_plr', 5e-7, 2e-5; lr = float(open('kvg_plr').read()) if os.path.exists('kvg_plr') else 1e-5
best = float(open('kvg_best_score').read()) if os.path.exists('kvg_best_score') else -1
act = 'keep'; new = lr
if E > best or not os.path.exists('kvg_best.pt'):
    best = E; open('kvg_best_score', 'w').write(str(E)); open('kvg_best_v', 'w').write(str(v)); act = 'new best'
lastE = float(open('kvg_last_E').read()) if os.path.exists('kvg_last_E') else 99; open('kvg_last_E', 'w').write(str(E))
low = E <= best - 4
if W > 0.28 or (low and (W > 0.22 or C > 0.82 or lastE <= best - 3)): act = 'revert'; new = max(lr / 2, lo_)
elif W > 0.22 or C > 0.82: act = 'halve'; new = max(lr / 2, lo_)
elif low: act = 'watch'   # one low eval alone: noise until a second one
elif W < 0.18 and C < 0.78 and E >= best - 1: act = 'raise' if act == 'keep' else act + '+raise'; new = min(lr * 1.1, hi_)   # steady: a small step, never above the cap
open(tgt, 'w').write(f'{new:g}')
line = f"round {rd} v{v}: E {E:g}/60 (seeds {E1}/{E2}) best {best:g} (v{open('kvg_best_v').read().strip()}) | W {100*W:.0f}% C {100*C:.0f}% ({n//4} problems) | {tgt[4:]} {lr:g} -> {new:g} | {act}"
open('kvg_auto.txt', 'a').write(line + '\n'); print(act + '|' + line)
PY3
)
  echo "[kg] auto: ${act#*|}"; put kvg_auto.txt kvg/auto.txt > /dev/null
  local rv=0
  case "${act%%|*}" in
    *"new best"*) cp kvg.pt kvg_best.pt ;;
    revert)
      bv=$(cat kvg_best_v); cp kvg_best.pt kvg.pt; cp kvg.pt kvf.pt; rm -f kvg_opt.pt; cp kvg.pt kvg_prev.pt
      seq $((bv + 1)) $v | tr '\n' ' ' >> kvg_skip_versions; echo >> kvg_skip_versions
      rd=$((rd + 1)); echo $rd > kvg_round; v=$((v0 + rd)); echo $v > VERSION
      put kvg.pt spKVF_latest.pt && put VERSION kvf/VERSION; sk kvg.pt $v
      echo "[kg] auto: back to the best (v$bv) as v$v, optimizer state dropped, versions $((bv + 1))-$((v - 1)) skipped"; rv=1 ;;
  esac; return $rv; }
ok60() { python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open(sys.argv[1])]; print(sum(correct(r) for r in rs), sum('</think>' in r['text'] for r in rs))" "$1"; }
curl -sS -o param_sketch.py "$RAW/ctl/param_sketch.py?nocache=$(date +%s)"
sk() {   # sk <ckpt> <version>: the weights' sketch (parameter-movement tracking), kept locally and on the hub
  [ -s sk_v$2.pt ] || { python3 param_sketch.py sketch $1 sk_v$2.pt && put sk_v$2.pt kvg/sk_v$2.pt > /dev/null; }; }
[ -s ${START:-kvf_round1.pt} ] && sk ${START:-kvf_round1.pt} $v0
[ -s kvg_prev.pt ] && [ -s kvg.pt ] && [ "$(cat VERSION 2>/dev/null)" -gt $((v0 + 1)) ] 2>/dev/null && sk kvg_prev.pt $(( $(cat VERSION) - 1 ))
[ -s kvg.pt ] && sk kvg.pt $(cat VERSION)
moves() { python3 param_sketch.py report $(ls sk_v*.pt | sort -t v -k2 -n) 2>&1 | sed 's/^/[kg] params /'; }
moves
touch used_groups.txt
if [ -e kvg_eval_now ] && [ -s kvg.pt ]; then rm -f kvg_eval_now; ev60 $(cat kvg_round 2>/dev/null || echo 0) $(cat VERSION); fi
while [ ! -e STOP_KG ]; do
  if [ -s kvg_extra ]; then mv kvg_extra kvg_extra.running; bash kvg_extra.running; rm -f kvg_extra.running; fi   # a one-off job between rounds (an eval with other flags), GPU free
  V0=$v0 GMIN=$GMIN CUR=$(cat VERSION) python3 - <<'PY' > newg_list.txt
import os, re
from huggingface_hub import HfApi
used = set(open('used_groups.txt').read().split()); v0 = int(os.environ['V0'])
fs = [f for f in HfApi().list_repo_files('baya1116/hypernet-sp-distill') if f.startswith('archive/sp3/dpo_col_g/') and f not in used]
gmin = int(os.environ.get('GMIN', '0'))
def g_(f):   # group size from the file name (g12_b..); the first G=4 files carry none
    m = re.search(r'/g(\d+)_b', f); return int(m.group(1)) if m else 4
lo = max(v0, int(os.environ.get('CUR', v0)) - 1)   # this run's weights, at most one version stale (asynchronous workers)
skip = set(int(x) for x in open('kvg_skip_versions').read().split()) if os.path.exists('kvg_skip_versions') else set()   # versions reverted from
print('\n'.join(sorted(f for f in fs if int(re.search(r'_v(\d+)\.jsonl$', f).group(1)) >= lo and g_(f) >= gmin
                       and int(re.search(r'_v(\d+)\.jsonl$', f).group(1)) not in skip)))
PY
  : > newg.jsonl
  for f in $(cat newg_list.txt); do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" "$H/$f" >> newg.jsonl 2>/dev/null; done
  n=$(wc -l < newg.jsonl)
  NEWG=$(cat kvg_newg 2>/dev/null || echo $NEWG); NEWINF=$(cat kvg_newinf 2>/dev/null || echo 16)
  ninf=$(python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
k = 0
for l in open('newg.jsonl'):
    g = json.loads(l); r = [('</think>' in t and correct({'text': t, 'gold': g['gold']})) for t in g['samples']]
    gd = g.get('guide'); r += [bool(gd) and '</think>' in gd and correct({'text': gd, 'gold': g['gold']})] if gd else []
    k += 0 < sum(r) < len(r)
print(k)")
  if [ $n -lt $NEWG ] || [ $ninf -lt $NEWINF ]; then   # a round needs NEWINF informative groups (one full update), not just NEWG problems   # meanwhile one rollout block of its own shard
    curl -sS -o kvc_job.sh "$RAW/ctl/kvc_job.sh?nocache=$(date +%s)"
    OWN="SHARD=$((NSH - 1)) NSHARD=$NSH RB=48 TB=60000"; [ -s kvg_own ] && OWN=$(cat kvg_own)   # kvg_own: e.g. the weak-area pool, a shard of its own
    env $OWN MODE=grpo ONE=1 bash kvc_job.sh | grep "^\[kc\] grpo" | sed 's/^\[kc\]/[kg] own/' || sleep 600
    continue; fi
  if [ -e kvg_auto ] && [ -s kvg_pending_auto ]; then
    pa=$(cat kvg_pending_auto); rm -f kvg_pending_auto
    auto_lr $pa || continue   # a revert: the groups just gathered were drawn with a version now skipped - gate again
  fi
  rd=$((rd + 1)); cat newg_list.txt >> used_groups.txt; cp newg.jsonl kvg_groups_r$rd.jsonl
  cp kvg.pt kvg_prev.pt
  GT=$(gt); echo "[kg] round $rd flags: $GT"
  python3 sp_qwen3.py train --traces kvg_groups_r$rd.jsonl $M --max-len 12288 --ckpt kvg_prev.pt $GT --seed $((6000 + rd)) --grad-sketch kvg_gsk_r$rd.pt --out kvg_next.pt --log $LG > run_kg_tr.log 2>&1
  echo "[kg] round $rd: $n problems ($ninf informative at the gate) | $(grep '\[sp\] grpo groups' run_kg_tr.log | tail -1 | cut -d: -f2-) | $(grep -E '\[sp\] train .*grpo group' run_kg_tr.log | tail -1 | grep -oP 'mean reward.*') | $(grep -oP '\[sp\] LoRA.*' run_kg_tr.log | head -1)"
  grep -E "Error|Traceback|out of memory" run_kg_tr.log | tail -1 | sed "s/^/[kg] round $rd: /"
  [ -s kvg_next.pt ] && grep -q "^\[sp\] trained" run_kg_tr.log || { echo "[kg] round $rd: training FAILED"; break; }
  gq=$(python3 - $rd <<'PY2' 2>/dev/null
import sys, os, torch   # the gradient itself, not the optimizer's steps: even vs odd groups of this round, and this round vs the last
r = int(sys.argv[1]); a = torch.load(f'kvg_gsk_r{r}.pt')
cos = lambda x, y: (x @ y / (x.norm() * y.norm() + 1e-30)).item()
b = torch.load(f'kvg_gsk_r{r - 1}.pt') if os.path.exists(f'kvg_gsk_r{r - 1}.pt') else None
out = []
for nm, e, o in (("LoRA", "even", "odd"), ("hypernet", "even_h", "odd_h")):
    if e not in a: continue
    x = f"{nm} gradient halves cos {cos(a[e], a[o]):+.3f}"
    if b is not None and e in b: x += f", vs the last round {cos(a[e] + a[o], b[e] + b[o]):+.3f}"
    out.append(x)
print(" | ".join(out) + f" ({a['n'][0]}+{a['n'][1]} groups)")
PY2
)
  echo "[kg] round $rd: $gq"; [ -s kvg_gsk_r$rd.pt ] && put kvg_gsk_r$rd.pt kvg/gsk_r$rd.pt > /dev/null
  mv kvg_next.pt kvg.pt; echo $rd > kvg_round; v=$((v0 + rd)); echo $v > VERSION; cp kvg.pt kvf.pt
  put kvg.pt spKVF_latest.pt && put VERSION kvf/VERSION
  sk kvg.pt $v; moves
  { echo "round $rd -> v$v $(date -u +%H:%MZ) | $(grep '\[sp\] grpo groups' run_kg_tr.log | tail -1 | cut -d: -f2-) | $(grep -E '\[sp\] train .*grpo group' run_kg_tr.log | tail -1 | grep -oP 'mean reward [0-9.]+') | $gq"
    python3 param_sketch.py report $(ls sk_v*.pt | sort -t v -k2 -n) 2>&1 | grep "^\[lora\]" | sed 's/^/  /'; } >> kvg_rounds.txt
  put kvg_rounds.txt kvg/rounds.txt > /dev/null   # the container log drops lines; the round summaries live here too
  EVERY=$(cat kvg_every 2>/dev/null || echo $EVERY)
  if [ $((rd % EVERY)) != 0 ]; then echo "[kg] round $rd (weights v$v) published"; continue; fi
  ev60 $rd $v
  [ -e kvg_auto ] && echo "$rd $v" > kvg_pending_auto   # judged when the next round is about to start (the workers' seed 2 has had time to arrive)
done
echo "[kg] JOB_DONE $(date -u)"
