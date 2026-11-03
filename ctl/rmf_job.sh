#!/bin/bash
# The RMT run, paper-faithful phase (the user): reading and writing share the weights (the LoRA on in every student pass),
# the write block's inputs are a copy of the memory, and the whole model is fine-tuned (all 28 layers, lr 1e-5, 8-bit AdamW,
# gradient checkpointing) - starting from the frozen-reader run's segment 13 (spRM_latest.pt); resumable from spRMF_latest.pt.
# Same data and held-out split as rm_job.sh (the pool is read again from its start); after each segment the held-out KL per
# domain; at a plateau the 60 problems with the memory (seeds 1-2) and without it (seed 1: what the tuning did to the model).
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); mkdir -p /root/w; cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
get() { for _i in 1 2 3; do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o "$2" "$H/$1" && return 0; sleep 10; done; return 1; }
for f in tools/sp_qwen3.py tools/sp_mathcheck.py ctl/rl_eval120.jsonl; do curl -sS -o $(basename $f) "$RAW/$f?nocache=$(date +%s)"; done
python3 -c "
rs = open('rl_eval120.jsonl').readlines(); open('rl_eval60.jsonl', 'w').writelines(rs[i] for i in range(len(rs)) if i % 24 < 12)"
python3 -c "import transformers, bitsandbytes" 2>/dev/null || python3 -m pip install -q "huggingface_hub[hf_xet]" transformers bitsandbytes "math-verify[antlr4_13_2]" 2>&1 | grep -v -i warn | tail -1
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
LG=/root/w/sp_f.log
if [ ! -s rmf.pt ]; then
  if get archive/sp3/spRMF_latest.pt rmf.pt && get archive/sp3/rmf/rmf.log rmf_sp.log; then
    python3 - <<'PY'
import re
L = open('/root/w/rmf_sp.log').read().splitlines()
seg = sum(1 for l in L if l.startswith('[sp] trained') and 'FT28' in l)
ev = [float(m.group(1)) for l in L if l.startswith('[sp] eval') and 'rmf.pt' in l for m in [re.search(r'KL ([0-9.]+)  top1', l)] if m]
means = [sum(ev[i:i + 5]) / 5 for i in range(0, len(ev) - len(ev) % 5, 5)]
open('/root/w/rmf_seg', 'w').write(str(seg)); open('/root/w/rmf_hist', 'w').write(''.join(f'{m:.4f}\n' for m in means[1:]))
print(f'[rm] F resumed from the hub: {seg} segments done, held-out means {[round(m, 4) for m in means]}')
PY
    cp rmf_sp.log sp_f.log
  else rm -f rmf.pt; get archive/sp3/spRM_latest.pt rmf.pt && echo "[rm] F starts from the frozen-reader run's last weights (spRM_latest.pt)"; fi
fi
echo "[rm] ready $(date -u)"
M="--S 32 --rw 1024 --C 128 --maxd 0 --rmt --rmt-block 512 --rmt-rank 16 --rmt-shared --rmt-copy"
score() { python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open(sys.argv[1])]
print('[rm] ' + sys.argv[2] + f': {sum(correct(r) for r in rs)}/{len(rs)} correct, closed {sum(\"</think>\" in r[\"text\"] for r in rs)}')" "$1" "$2"; }
evkl() { out=""; : > rm_pos_last.txt; for d in chat code comp math mc; do python3 -c "
import json, sys; [sys.stdout.write(l) for l in open('rm_held.jsonl') if json.loads(l).get('src', '?').split('-')[0] == sys.argv[1]]" $d > held_$d.jsonl
  r_=$(python3 sp_qwen3.py eval --traces held_$d.jsonl --max-len 6144 $M "$@" --log $LG 2>&1 | grep "\[sp\] eval" | tail -1)
  k=$(echo "$r_" | grep -oP "KL \K[0-9.]+(?=  top1)"); out="$out $d $k"; echo "$d $(echo "$r_" | grep -oP 'by position \(0, 1-7, 8-63, 64\+\): \K.*')" >> rm_pos_last.txt; done; echo "$out"; }
if [ ! -e rmf_start_done ]; then
  e=$(evkl --ckpt rmf.pt); echo "[rm] F start (frozen-reader weights read with the shared LoRA and copy writes):$e"; touch rmf_start_done; fi
seg=$(cat rmf_seg 2>/dev/null || echo 0); cur=rmf.pt
while [ ! -e STOP_RMF ]; do seg=$((seg + 1)); t0=$(date +%s)
  python3 sp_qwen3.py train --traces rm_pool.jsonl --skip $(( ((seg - 1) * 600) % 9600 )) --n 600 --max-len 8192 $M --ft-layers 28 --opt8bit --ft-lr ${FTLR:-1e-5} --grad-ckpt --chunks-per-trace 4 --rmt-bptt 2 --accum 8 --lr 1e-4 --const-lr --ckpt $cur --seed $((100 + seg)) --out rmf_next.pt --log $LG > run_rmf.log 2>&1
  if [ -s rmf_next.pt ] && grep -q "^\[sp\] trained" run_rmf.log; then mv rmf_next.pt rmf.pt; echo $seg > rmf_seg
  else echo "[rm] F segment $seg FAILED: $(grep -E 'Error' run_rmf.log | tail -1 | cut -c1-200)"; break; fi
  tk=$(grep -oP "KL [0-9.]+ top1 [0-9.]+%" run_rmf.log | awk 'NR==1{f=$0} {l=$0} END{print "first 25 " f " / last 25 " l}')
  ev=$(evkl --ckpt rmf.pt); m=$(echo "$ev" | python3 -c "import sys; v = sys.stdin.read().split()[1::2]; print(round(sum(map(float, v)) / len(v), 4))")
  echo "[rm] F segment $seg ($(( $(date +%s) - t0 )) s, OOM skips $(grep -c 'out of memory' run_rmf.log)): train $tk | held-out KL by domain with memory:$ev | mean $m | by position: $(tr '\n' ';' < rm_pos_last.txt)"
  echo "$m" >> rmf_hist; put rmf.pt spRMF_latest.pt; put $LG rmf/rmf.log
  if python3 -c "
import sys; h = [float(x) for x in open('rmf_hist')]
sys.exit(0 if len(h) > 3 and min(h[-3:]) > min(h[:-3]) * 0.99 else 1)"; then echo "[rm] F plateau ($(tr '\n' ' ' < rmf_hist))"; break; fi
done
for cfg in "mem:1" "mem:2" "nomem:1"; do n=${cfg%%:*}; sd=${cfg#*:}; x=""; [ $n = nomem ] && x="--no-sp"
  python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $M --max-len 12288 --ckpt rmf.pt $x --roll-cb --roll-batch 48 --grpo-tok-budget 60000 --seed $sd --roll-out rmf_${n}_s$sd.jsonl --log $LG > run_rmfgen.log 2>&1
  grep -E "Error|Traceback" run_rmfgen.log | tail -1 | sed "s/^/[rm] F gen $n s$sd: /"
  score rmf_${n}_s$sd.jsonl "F segment $(cat rmf_seg) $n seed $sd"; put rmf_${n}_s$sd.jsonl rmf/rmf_${n}_s$sd.jsonl
done
date -u > DONE_RMF; put DONE_RMF rmf/DONE_RMF; echo "[rm] F JOB_DONE $(date -u)"
