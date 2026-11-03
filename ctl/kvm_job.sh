#!/bin/bash
# KV memory from the hypernet (the user: KV-level memory, but with the light separate pooler): the pooler reads the distant
# past (token embeddings, all of it - no cap, the user) and its 32 outputs become per-layer KEYS/VALUES (low-rank heads,
# keys through each layer's k_norm and RoPE at the slots after the prompt), put straight into the frozen model's cache:
# [prompt | memory KV (32) | raw window 1024 | chunk]. Training: KL to the full-KV teacher on chunks after a full window,
# 16 chunks per trace, 600 traces per segment; data and held-out split as the RMT runs (gm_* pool, 65 held out). After each
# segment the held-out KL per domain (the no-memory floor once); at a plateau the 60 problems with the memory, seeds 1-2.
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
print('[kv] pool', len(rs) - 1500, '| held-out', len(out))"
fi
LG=/root/w/sp_kv.log
[ -s kvm.pt ] || { get archive/sp3/spKVM_latest.pt kvm.pt && get archive/sp3/kvm/kvm.log $LG && python3 - <<'PY'
import re
L = open('/root/w/sp_kv.log').read().splitlines()
seg = sum(1 for l in L if l.startswith('[sp] trained') and 'kvmem' in l)
ev = [float(m.group(1)) for l in L if l.startswith('[sp] eval') and 'kvm.pt' in l for m in [re.search(r'KL ([0-9.]+)  top1', l)] if m]
means = [sum(ev[i:i + 5]) / 5 for i in range(0, len(ev) - len(ev) % 5, 5)]
open('/root/w/kvm_seg', 'w').write(str(seg)); open('/root/w/kvm_hist', 'w').write(''.join(f'{m:.4f}\n' for m in means))
open('/root/w/kvm_base_done', 'w').write('hub\n'); print(f'[kv] resumed from the hub: {seg} segments, held-out means {means}')
PY
} || rm -f kvm.pt
echo "[kv] ready $(date -u)"
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
echo "[kv] config: $M"
score() { python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open(sys.argv[1])]
print('[kv] ' + sys.argv[2] + f': {sum(correct(r) for r in rs)}/{len(rs)} correct, closed {sum(\"</think>\" in r[\"text\"] for r in rs)}')" "$1" "$2"; }
evkl() { out=""; for d in chat code comp math mc; do python3 -c "
import json, sys; [sys.stdout.write(l) for l in open('rm_held.jsonl') if json.loads(l).get('src', '?').split('-')[0] == sys.argv[1]]" $d > held_$d.jsonl
  k=$(python3 sp_qwen3.py eval --traces held_$d.jsonl --max-len 6144 $M "$@" --log $LG 2>&1 | grep "\[sp\] eval" | tail -1 | grep -oP "KL \K[0-9.]+(?=  top1)"); out="$out $d $k"; done; echo "$out"; }
if [ ! -e kvm_base_done ]; then
  echo "[kv] held-out KL to full KV by domain, raw window 1024 only (no memory):$(evkl --no-sp)"; touch kvm_base_done; fi
seg=$(cat kvm_seg 2>/dev/null || echo 0); cur=""; [ -s kvm.pt ] && cur=kvm.pt
while [ ! -e STOP_KV ]; do seg=$((seg + 1)); t0=$(date +%s); x="--const-lr"; [ $seg = 1 ] && x=""
  python3 sp_qwen3.py train --traces rm_pool.jsonl --skip $(( ((seg - 1) * 600) % 9600 )) --n 600 --max-len 8192 $M --chunks-per-trace 16 --accum 8 --lr 1e-4 $x ${cur:+--ckpt $cur} --seed $seg --out kvm_next.pt --log $LG > run_kv.log 2>&1
  if [ -s kvm_next.pt ] && grep -q "^\[sp\] trained" run_kv.log; then mv kvm_next.pt kvm.pt; cur=kvm.pt; echo $seg > kvm_seg
  else echo "[kv] segment $seg FAILED: $(grep -E 'Error' run_kv.log | tail -1 | cut -c1-200)"; break; fi
  tk=$(grep -oP "KL [0-9.]+ top1 [0-9.]+%" run_kv.log | awk 'NR==1{f=$0} {l=$0} END{print "first 25 " f " / last 25 " l}')
  ev=$(evkl --ckpt kvm.pt); m=$(echo "$ev" | python3 -c "import sys; v = sys.stdin.read().split()[1::2]; print(round(sum(map(float, v)) / len(v), 4))")
  echo "[kv] segment $seg ($(( $(date +%s) - t0 )) s, OOM skips $(grep -c 'out of memory' run_kv.log)): train $tk | held-out KL by domain with KV memory:$ev | mean $m"
  echo "$m" >> kvm_hist; put kvm.pt spKVM_latest.pt; put $LG kvm/kvm.log
  if python3 -c "
import sys; h = [float(x) for x in open('kvm_hist')]
sys.exit(0 if len(h) > 3 and min(h[-3:]) > min(h[:-3]) * 0.99 else 1)"; then echo "[kv] plateau ($(tr '\n' ' ' < kvm_hist))"; break; fi
done
for sd in 1 2; do
  python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $M --max-len 12288 --ckpt kvm.pt --roll-batch 48 --grpo-tok-budget 60000 --seed $sd --roll-out kvm_mem_s$sd.jsonl --log $LG > run_kvgen.log 2>&1
  grep -E "Error|Traceback" run_kvgen.log | tail -1 | sed "s/^/[kv] gen s$sd: /"
  score kvm_mem_s$sd.jsonl "segment $(cat kvm_seg) KV memory seed $sd"; put kvm_mem_s$sd.jsonl kvm/kvm_mem_s$sd.jsonl
done
date -u > DONE_KV; put DONE_KV kvm/DONE_KV; echo "[kv] JOB_DONE $(date -u)"
