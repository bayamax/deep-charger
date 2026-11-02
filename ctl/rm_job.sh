#!/bin/bash
# The RMT run's job, resumable on any box (box RM serial 1 ran it first): data as box RM (gm_* pool shuffled with seed 0,
# 65 held-out traces from its tail), carry on from the hub's spRM_latest.pt when there is one (segments done = "[sp] trained"
# lines in the hub's rm/rm.log; the per-segment held-out means rebuilt from its eval lines), train 600 traces per segment until
# the held-out mean has not improved by 1% in 3 segments, then the 60 problems (seeds 1-2) with the memory and without it.
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); mkdir -p /root/w; cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
get() { for _i in 1 2 3; do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o "$2" "$H/$1" && return 0; sleep 10; done; return 1; }
for f in tools/sp_qwen3.py tools/sp_mathcheck.py ctl/rl_eval120.jsonl; do curl -sS -o $(basename $f) "$RAW/$f?nocache=$(date +%s)"; done
python3 -c "
rs = open('rl_eval120.jsonl').readlines(); open('rl_eval60.jsonl', 'w').writelines(rs[i] for i in range(len(rs)) if i % 24 < 12)"
python3 -c "import transformers" 2>/dev/null || python3 -m pip install -q "huggingface_hub[hf_xet]" transformers "math-verify[antlr4_13_2]" 2>&1 | grep -v -i warn | tail -1
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B')" > dl.log 2>&1
if [ ! -s rm_pool.jsonl ]; then
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
print('[rm] pool', len(rs) - 1500, '| held-out', len(out))"
fi
LG=/root/w/sp.log
if [ ! -s rm.pt ] && get archive/sp3/spRM_latest.pt rm.pt && get archive/sp3/rm/rm.log $LG; then
  python3 - <<'PY'
import re
L = open('/root/w/sp.log').read().splitlines()
seg = sum(1 for l in L if l.startswith('[sp] trained') and 'rmt512' in l)
ev = [float(m.group(1)) for l in L if l.startswith('[sp] eval') and 'rmt512 rm.pt' in l for m in [re.search(r'KL ([0-9.]+)  top1', l)] if m]
means = [sum(ev[i:i + 5]) / 5 for i in range(0, len(ev) - len(ev) % 5, 5)]
open('/root/w/rm_seg', 'w').write(str(seg)); open('/root/w/rm_hist', 'w').write(''.join(f'{m:.4f}\n' for m in means))
open('/root/w/rm_base_done', 'w').write('from the hub log\n')
print(f'[rm] resumed from the hub: {seg} segments done, held-out means {[round(m, 4) for m in means]}')
PY
fi
echo "[rm] ready $(date -u)"
M="--S 32 --rw 1024 --C 128 --maxd 0 --rmt --rmt-block 512 --rmt-rank 16"
score() { python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open(sys.argv[1])]
print('[rm] ' + sys.argv[2] + f': {sum(correct(r) for r in rs)}/{len(rs)} correct, closed {sum(\"</think>\" in r[\"text\"] for r in rs)}')" "$1" "$2"; }
evkl() { out=""; : > rm_pos_last.txt; for d in chat code comp math mc; do python3 -c "
import json, sys; [sys.stdout.write(l) for l in open('rm_held.jsonl') if json.loads(l).get('src', '?').split('-')[0] == sys.argv[1]]" $d > held_$d.jsonl
  r_=$(python3 sp_qwen3.py eval --traces held_$d.jsonl --max-len 6144 $M "$@" --log $LG 2>&1 | grep "\[sp\] eval" | tail -1)
  k=$(echo "$r_" | grep -oP "KL \K[0-9.]+(?=  top1)"); out="$out $d $k"; echo "$d $(echo "$r_" | grep -oP 'by position \(0, 1-7, 8-63, 64\+\): \K.*')" >> rm_pos_last.txt; done; echo "$out"; }
if [ ! -e rm_base_done ]; then
  e=$(evkl --no-sp); echo "[rm] held-out KL to full KV by domain, raw part only (no memory):$e | by position: $(tr '\n' ';' < rm_pos_last.txt)"
  touch rm_base_done; fi
seg=$(cat rm_seg 2>/dev/null || echo 0); cur=""; [ -s rm.pt ] && cur=rm.pt
while [ ! -e STOP_RM ]; do seg=$((seg + 1)); t0=$(date +%s); x="--const-lr"; [ $seg = 1 ] && x=""
  python3 sp_qwen3.py train --traces rm_pool.jsonl --skip $(( (seg - 1) * 600 )) --n 600 --max-len 8192 $M --chunks-per-trace 4 --rmt-bptt 2 --accum 8 --lr 1e-4 $x ${cur:+--ckpt $cur} --seed $seg --out rm_next.pt --log $LG > run_rm.log 2>&1
  if [ -s rm_next.pt ] && grep -q "^\[sp\] trained" run_rm.log; then mv rm_next.pt rm.pt; cur=rm.pt; echo $seg > rm_seg
  else echo "[rm] segment $seg FAILED: $(grep -E 'Error' run_rm.log | tail -1 | cut -c1-200)"; break; fi
  tk=$(grep -oP "KL [0-9.]+ top1 [0-9.]+%" run_rm.log | awk 'NR==1{f=$0} {l=$0} END{print "first 25 " f " / last 25 " l}')
  ev=$(evkl --ckpt rm.pt); m=$(echo "$ev" | python3 -c "import sys; v = sys.stdin.read().split()[1::2]; print(round(sum(map(float, v)) / len(v), 4))")
  echo "[rm] segment $seg ($(( $(date +%s) - t0 )) s, OOM skips $(grep -c 'out of memory' run_rm.log)): train $tk | held-out KL by domain with memory:$ev | mean $m | by position: $(tr '\n' ';' < rm_pos_last.txt)"
  echo "$m" >> rm_hist; put rm.pt spRM_latest.pt; put $LG rm/rm.log
  if python3 -c "
import sys; h = [float(x) for x in open('rm_hist')]
sys.exit(0 if len(h) > 3 and min(h[-3:]) > min(h[:-3]) * 0.99 else 1)"; then echo "[rm] plateau ($(tr '\n' ' ' < rm_hist))"; break; fi
done
for cfg in "mem:" "rawpart:--no-sp"; do n=${cfg%%:*}; x=${cfg#*:}
  for s in 1 2; do
    python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $M --max-len 12288 --ckpt rm.pt $x --roll-cb --roll-batch 48 --grpo-tok-budget 60000 --seed $s --roll-out rm_${n}_s$s.jsonl --log $LG > run_rmgen.log 2>&1
    grep -E "Error|Traceback" run_rmgen.log | tail -1 | sed "s/^/[rm] gen $n s$s: /"
    score rm_${n}_s$s.jsonl "segment $(cat rm_seg) $n seed $s"; put rm_${n}_s$s.jsonl rm/rm_${n}_s$s.jsonl
  done
done
date -u > DONE_RM1; put DONE_RM1 rm/DONE_RM1; echo "[rm] JOB_DONE $(date -u)"
