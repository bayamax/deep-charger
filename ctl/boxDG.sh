#!/bin/bash
# Box DG (a 24 GB card): why does SP make generation WORSE than the raw window alone (60 GRPO problems: raw 43-44, SP 35-38
# for every SP so far) while the teacher-forced KL says SP helps (on the model's own text too)? With the latest window-0
# all-layer state (spW0_window0.pt):
#  1. the 60 problems again (seed 2), SP at window 1024 and the raw window alone (1056), each storing the generated token ids
#     and every sampled token's log p under the generator's own logits (--roll-lp);
#  2. eval --dump on both runs' text, scored with SP and with the raw window: per token log p (student / full-KV teacher),
#     forward and reverse KL. The SP run scored with SP is the generation path vs the teacher-forced path, token by token
#     (a bug in either shows as sampled tokens the teacher-forced student finds nearly impossible); the teacher's log p of
#     the tokens each run actually sampled shows how often a run wanders where the full-KV model would not go.
# Results to archive/sp3/dg/, marker DONE_DG1.
SER=1
if [ -f /root/.boxdg_serial ] && [ "$(cat /root/.boxdg_serial)" -gt "$SER" ] 2>/dev/null; then echo "STALE $SER"; exit 0; fi
echo $SER > /root/.boxdg_serial
echo "[dg] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
mkdir -p /root/w
cat > /root/status.sh <<'ST'
grep -E "^\[(dg|hub)\]|Error|Traceback|out of memory" /root/w/w.log 2>/dev/null | tail -n 4 | cut -c1-300
tail -n 1 /root/w/sp.log 2>/dev/null | cut -c1-200
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.job1 ]; then touch /root/.job1
  cat > /root/job.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/w
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i
put() { for t in 1 2 3 4 5; do python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/sp3/'+sys.argv[2], repo_id='$R')" "$1" "$2" > put.err 2>&1 && { echo "[hub] $2"; return 0; }; sleep 60; done; echo "[hub] $2 FAILED"; }
get() { for t in 1 2 3; do curl -sSfL -H "Authorization: Bearer $HF_TOKEN" -o "$2" "$H/$1" && return 0; sleep 10; done; return 1; }
for f in tools/sp_qwen3.py tools/sp_mathcheck.py ctl/rl_eval120.jsonl; do curl -sS -o $(basename $f) "$RAW/$f?nocache=$(date +%s)"; done
python3 -c "
rs = open('rl_eval120.jsonl').readlines(); open('rl_eval60.jsonl', 'w').writelines(rs[i] for i in range(len(rs)) if i % 24 < 12)"
python3 -m pip install -q "huggingface_hub[hf_xet]" transformers "math-verify[antlr4_13_2]" 2>&1 | grep -v -i warn | tail -1
for t in 1 2 3; do x=""; [ $t -gt 1 ] && x="HF_HUB_DISABLE_XET=1"
  env $x timeout 900 python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-1.7B'))" > dl.log 2>&1 && break; echo "[dg] model download attempt $t failed"; done
get archive/sp3/spW0_window0.pt spW0.pt || echo "[dg] checkpoint download FAILED"
echo "[dg] ready: ckpt $(du -h spW0.pt | cut -f1), $(wc -l < rl_eval60.jsonl) problems $(date -u)"
LG=/root/w/sp.log; M="--S 32 --heads 8 --layers 3 --ffn 2048 --C 128 --maxd 0 --max-len 12288 --ckpt spW0.pt"
SPC="--rw 1024"; RWC="--no-sp --rw 1056"
score() { python3 -c "
import json, sys; sys.path.insert(0, '.')
from sp_mathcheck import correct
rs = [json.loads(l) for l in open(sys.argv[1])]
print('[dg] ' + sys.argv[2] + f': {sum(correct(r) for r in rs)}/{len(rs)} correct, closed {sum(\"</think>\" in r[\"text\"] for r in rs)}')" "$1" "$2"; }
for c in sp raw; do cfg=$SPC; [ $c = raw ] && cfg=$RWC
  python3 sp_qwen3.py rollout --traces rl_eval60.jsonl $M $cfg --roll-cb --roll-batch 48 --grpo-tok-budget 60000 --roll-lp --seed 2 --roll-out gen_$c.jsonl --log $LG > run_gen_$c.log 2>&1
  grep -E "Error|Traceback" run_gen_$c.log | tail -1 | sed "s/^/[dg] gen $c: /"
  score gen_$c.jsonl "gen $c seed 2"; put gen_$c.jsonl dg/gen_$c.jsonl
done
for t in sp raw; do for c in sp raw; do cfg=$SPC; [ $c = raw ] && cfg=$RWC
  python3 sp_qwen3.py eval --traces gen_$t.jsonl $M $cfg --dump dump_${t}text_${c}.pt --log $LG > run_ev_${t}_$c.log 2>&1
  echo "[dg] text of the $t run scored with $c: $(grep -oP 'KL [0-9.]+ .*top1 agree [0-9.]+%' run_ev_${t}_$c.log | tail -1) $(grep -E 'Error|Traceback' run_ev_${t}_$c.log | tail -1)"
  put dump_${t}text_${c}.pt dg/dump_${t}text_${c}.pt
done; done
put $LG dg/dg.log; date -u > DONE_DG1; put DONE_DG1 dg/DONE_DG1; echo "[dg] JOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/job.sh >> /root/w/w.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[dg] JOB_LAUNCHED $(date -u)"
fi
# CTL-END
