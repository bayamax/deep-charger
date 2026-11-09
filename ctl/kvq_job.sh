#!/bin/bash
# On-policy distillation of the pooler (the user: score the whole text the student writes in one pass - and make that the
# loss). The held-out KL of the distillation so far is teacher-forced on the teacher's own texts: summed over a text it is
# the text-level KL(teacher || student), which asks whether the student can follow what the teacher writes, never whether
# what the student writes looks right to the teacher. The pooler's attention collapsed (segment 4: 50% of its mass on 46
# tokens, 60 problems 31) while that KL kept falling. Here each round the current pooler WRITES answers to fresh problems
# (deployment sampling, its memory built from its own text), then trains on those texts: the full-context original model
# reads each one in one pass and the loss at every position of the student's own path is (1-W) KL(teacher || student) +
# W KL(student || teacher) (sp_qwen3.py train --rkl-weight W on the rollout file; exact token ids via --roll-lp). The LLM stays
# frozen; the prior and the cap (--att-prior 1.0 --att-cap $CAP) as in ctl/kvp_job.sh. Every EVERY rounds the 60 problems at
# seed 1. Problems: math / competition / MMLU-Pro prompts of the gm_* pool taken from its end (the distillation used its
# start), each used once, the 60 problems' own prompts left out. STOP_KQ stops after the round.
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main; RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for _i in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
for _t in $(seq 60); do curl -sS -o sp_new.py "$RAW/tools/sp_qwen3.py?nocache=$(date +%s)"; grep -q "def gen_chunk" sp_new.py && { mv sp_new.py sp_qwen3.py; break; }; sleep 120; done
curl -sS -o sp_mathcheck.py "$RAW/tools/sp_mathcheck.py?nocache=$(date +%s)"
LG=/root/w/sp_kvq.log; LR=${LR:-1e-4}; W=${W:-0.5}; NP=${NP:-48}; CAP=${CAP:-1.0}; EVERY=${EVERY:-2}
M="--S 32 --heads 8 --layers 3 --ffn 2048 --rw 1024 --C 128 --maxd 0 --kv-mem --kv-rank 512"
AP="--att-prior 1.0 --att-layers 8-19"; [ "$CAP" != 0 ] && AP="$AP --att-cap $CAP"
[ -s kvq.pt ] || cp "${START:-kvp.pt}" kvq.pt
if [ ! -s kvq_prompts.jsonl ] || ! head -n 1 kvq_prompts.jsonl | grep -q '"gold": "[^"]'; then python3 -c "
import json
def boxed(t):   # the pool rows carry no gold: the teacher's own final answer stands in (so 'correct' = agrees with the teacher)
    i = t.rfind('\\\\boxed{')
    if i < 0: return ''
    j = i + 7; d = 1; out = []
    while j < len(t) and d:
        c = t[j]; d += (c == '{') - (c == '}')
        if d: out.append(c)
        j += 1
    return ''.join(out) if not d else ''
ban = {json.loads(l)['prompt'] for l in open('rl_eval60.jsonl')}
rs = [json.loads(l) for l in open('rm_pool.jsonl')]
rs = [r for r in rs if r.get('src', '?').split('-')[0] in ('math', 'comp', 'mc') and r['prompt'] not in ban][::-1]   # from the end
with open('kvq_prompts.jsonl', 'w') as f:
    for r in rs: f.write(json.dumps({'src': r.get('src', ''), 'id': r.get('id', ''), 'gold': r.get('gold') or boxed(r.get('text', '')), 'prompt': r['prompt']}) + '\n')
print('[kq] prompts', len(rs), 'with a teacher answer', sum(1 for r in rs if r.get('gold') or boxed(r.get('text', ''))))"; fi
score() { python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open(sys.argv[1])]
print(f'{sum(correct(r) for r in rs)}/{len(rs)} correct, closed {sum(\"</think>\" in r[\"text\"] for r in rs)}')" "$1"; }
quiet() { while pgrep -f "sp_gk.py" > /dev/null; do sleep 30; done; }   # a side evaluation (boxRMc serials 78-79) holds memory
rd=$(cat kvq_round 2>/dev/null || echo 0)
echo "[kq] ready $(date -u) | start $(cat kvq_start 2>/dev/null || echo "${START:-kvp.pt}") | $AP | W $W (reverse KL share) | lr $LR | $NP problems per round | from round $rd"
echo "${START:-kvp.pt}" > kvq_start
while [ ! -e STOP_KQ ]; do rd=$((rd + 1)); t0=$(date +%s)
  python3 -c "
import sys; L = open('kvq_prompts.jsonl').readlines(); i = ($rd - 1) * $NP
open('kvq_in.jsonl', 'w').writelines(L[i:i + $NP]); print(len(L[i:i + $NP]))" > /dev/null
  quiet; rm -f kvq_gen.jsonl
  python3 sp_qwen3.py rollout --traces kvq_in.jsonl $M $AP --ckpt kvq.pt --max-len 12288 --roll-batch 48 --grpo-tok-budget 60000 --roll-lp --seed $((500 + rd)) --roll-out kvq_gen.jsonl --log $LG > run_kq_gen.log 2>&1
  [ -s kvq_gen.jsonl ] || { echo "[kq] round $rd: generation FAILED: $(grep -E 'Error' run_kq_gen.log | tail -1 | cut -c1-200)"; break; }
  t1=$(date +%s); g="$(score kvq_gen.jsonl)"; quiet   # quiet before training too: the side checks start whenever the GPU is idle
  python3 sp_qwen3.py train --traces kvq_gen.jsonl --max-len 12288 $M $AP --rkl-weight $W --chunks-per-trace 32 --accum 8 --lr $LR --const-lr --lora-lr 0 --ckpt kvq.pt --seed $((600 + rd)) --out kvq_next.pt --log $LG > run_kq.log 2>&1
  if [ -s kvq_next.pt ] && grep -q "^\[sp\] trained" run_kq.log; then mv kvq_next.pt kvq.pt; echo $rd > kvq_round
  else echo "[kq] round $rd: training FAILED: $(grep -E 'Error' run_kq.log | tail -1 | cut -c1-200)"; break; fi
  tk=$(grep -oP "KL [0-9.]+ top1 [0-9.]+% \| reverse KL [0-9.]+" run_kq.log | awk 'NR==1{f=$0} {l=$0} END{print "first " f " / last " l}')
  echo "[kq] round $rd (generation $((t1 - t0)) s, training $(( $(date +%s) - t1 )) s, OOM skips $(grep -c 'out of memory' run_kq.log)): its own texts agree with the teacher's answer $g | on them $tk"
  put kvq.pt spKVQ_latest.pt > /dev/null; cp kvq_gen.jsonl kvq_gen_r$rd.jsonl; put kvq_gen.jsonl kvq/gen_r$rd.jsonl > /dev/null; put $LG kvq/kvq.log > /dev/null
  if [ $((rd % EVERY)) = 0 ]; then
    quiet; o=kvq_r${rd}_s1.jsonl; rm -f $o; cp kvq.pt kvq_r$rd.pt
    python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $M $AP --ckpt kvq.pt --max-len 12288 --roll-batch 48 --grpo-tok-budget 60000 --seed 1 --roll-out $o --log $LG > run_kq_ev.log 2>&1
    put $o kvq/$o > /dev/null
    echo "[kq] round $rd seed 1: $(score $o) (segment 2: 39, v38 plain: 44 / 41, raw window 42 / 44) $(date -u)"
  fi
done
echo "[kq] stopped after round $rd $(date -u)"
