#!/bin/bash
# DPO on the KV memory (the user: the raw-window trajectory as the ideal). Pairs from the DPO data (ctl/kvd_job.sh): the same
# problem, chosen = the raw window 1024's generation that closed its thought and is right, rejected = the memory model's that
# did not close or is wrong; and the other way round where only the memory got it (its path chosen over the failed raw one). Sequence log-probs under the KV-memory deployment loop (seq_logp_kv; checked equal to the
# generator's own log-probs), only the memory hypernet trains (the model stays frozen); reference = the starting weights
# (--ctl-window segment 1). Round 2 of the data (ctl/kvd2_job.sh) adds raw-window redraws. Then the 60 problems, seeds 1-2.
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
for _t in $(seq 60); do curl -sS -o sp_qwen3.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q "def seq_logp_kv" sp_qwen3.py && break; sleep 120; done
LG=/root/w/sp_kvo.log
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
BETA=${BETA:-0.1}; LR=${LR:-2e-5}; EP=${EP:-2}
echo "[ko] ready $(date -u) | seq_logp_kv in the script: $(grep -c 'def seq_logp_kv' sp_qwen3.py) | beta $BETA lr $LR epochs $EP"
python3 - <<'PY'
import json, glob, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
def load(side): return {r['id']: r for f in sorted(glob.glob(f'dpo_{side}_b*.jsonl')) for l in open(f) for r in [json.loads(l)]}
raw, mem = load('raw'), load('mem')
ok = lambda r: '</think>' in r['text'] and correct(r)
for f in sorted(glob.glob('dpo_raw2_*.jsonl')):   # round 2: a raw-window redraw that succeeds replaces a failed first draw
    for l in open(f):
        r = json.loads(l)
        if r['id'] in raw and not ok(raw[r['id']]) and ok(r): raw[r['id']] = r
pairs = [{'id': k, 'prompt': raw[k]['prompt'], 'chosen': raw[k]['text'], 'rejected': mem[k]['text'],
          'why': 'unclosed' if '</think>' not in mem[k]['text'] else 'wrong'} for k in raw if k in mem and ok(raw[k]) and not ok(mem[k])]
# and the other way round: the memory right and closed where the raw window failed - its own successful path chosen over the
# raw window's failed one (keeps what the memory is good for: the distant past it carries)
pairs += [{'id': k, 'prompt': mem[k]['prompt'], 'chosen': mem[k]['text'], 'rejected': raw[k]['text'], 'why': 'memory-only'}
          for k in raw if k in mem and ok(mem[k]) and not ok(raw[k])]
both = sum(1 for k in raw if k in mem)
print(f"[ko] problems with both sides {both}: raw right {sum(ok(raw[k]) for k in raw if k in mem)}, memory right {sum(ok(mem[k]) for k in raw if k in mem)}"
      f" | pairs {len(pairs)} (raw chosen, memory unclosed {sum(p['why'] == 'unclosed' for p in pairs)} / wrong {sum(p['why'] == 'wrong' for p in pairs)}; memory chosen over a failed raw {sum(p['why'] == 'memory-only' for p in pairs)})"
      f" | memory right, raw not: {sum(1 for k in raw if k in mem and ok(mem[k]) and not ok(raw[k]))}")
open('dpo_pairs.jsonl', 'w').writelines(json.dumps(p) + '\n' for p in pairs)
PY
put dpo_pairs.jsonl dpo/dpo_pairs.jsonl
score() { python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open(sys.argv[1])]
print('[ko] ' + sys.argv[2] + f': {sum(correct(r) for r in rs)}/{len(rs)} correct, closed {sum(\"</think>\" in r[\"text\"] for r in rs)}')" "$1" "$2"; }
if [ ! -s kvo.pt ]; then
  python3 sp_qwen3.py train --spin --traces dpo_pairs.jsonl $M --max-len 12288 --ckpt dpo_ref.pt --accum 4 --lr $LR --const-lr --epochs $EP --spin-beta $BETA --seed 300 --out kvo.pt --log $LG > run_ko_tr.log 2>&1
  grep -E "\[sp\] (spin|train .* pair)|Error|Traceback|out of memory" run_ko_tr.log | tail -n 8 | cut -c1-250 | sed 's/^/[ko] /'
  [ -s kvo.pt ] && put kvo.pt spKVO_latest.pt
fi
# does the memory still carry the past, or did DPO make it inert? held-out KL to full KV by domain (segment 8 memory 0.0648
# mean, the raw window alone 0.107)
if [ -s kvo.pt ] && [ ! -e kvo_kl_done ]; then out=""
  for d in chat code comp math mc; do python3 -c "
import json, sys; [sys.stdout.write(l) for l in open('rm_held.jsonl') if json.loads(l).get('src', '?').split('-')[0] == sys.argv[1]]" $d > held_$d.jsonl
    for ck in dpo_ref.pt kvo.pt; do k=$(timeout 2700 python3 sp_qwen3.py eval --traces held_$d.jsonl --max-len 6144 $M --ckpt $ck --log $LG 2>&1 | grep "\[sp\] eval" | tail -1 | grep -oP "KL \K[0-9.]+(?=  top1)"); out="$out $d/${ck%.pt} $k"; done; done
  echo "[ko] held-out KL to full KV, before / after DPO:$out"; touch kvo_kl_done; fi
for sd in 1 2; do
  [ -s kvo_s$sd.jsonl ] && continue
  python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $M --max-len 12288 --ckpt kvo.pt --roll-batch 48 --grpo-tok-budget 60000 --seed $sd --roll-out kvo_s$sd.jsonl --log $LG > run_kogen.log 2>&1
  grep -E "Error|Traceback" run_kogen.log | tail -1 | sed "s/^/[ko] gen s$sd: /"
  score kvo_s$sd.jsonl "DPO (beta $BETA, lr $LR, $EP epochs) KV memory seed $sd"; put kvo_s$sd.jsonl kvo/kvo_s$sd.jsonl
done
put $LG kvo/kvo.log; date -u > DONE_KO; put DONE_KO kvo/DONE_KO; echo "[ko] JOB_DONE $(date -u)"
