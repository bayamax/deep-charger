#!/bin/bash
# GRPO rounds with the reward model's score, one after another from a queue (2026-10-14, the user: the judgment-model GRPO
# is new, so try things; and use the whole card). Each line of /root/rm_queue.txt:
#   <tag> <items.jsonl> [extra online_loop flags, which override the defaults below]
# Every round starts from 2.1.0 (mtg7_s40), trains 60 steps, copies at 20/40/60; the screens run whenever the card has room
# (a lane needs ~7.5 GB free): single-turn shard 0 for each copy, then for the best copy shards 1-2, br3 and chain3, and the
# reward model's reading of the replies. One summary line per round goes to /root/rm_rounds.txt (and the hub).
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
read M C S D < /root/rm_choice.txt
cnt() { python3 -c "import json,sys;print(sum(bool(json.loads(l).get('correct')) for l in open(sys.argv[1]) if l.strip()))" "$1" 2>/dev/null || echo 0; }
turns() { python3 -c "
import json,sys; r=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(' '.join(str(sum(bool(x['correct']) for x in r if x['turn']==t)) for t in range(int(sys.argv[2]))))" "$1" "$2" 2>/dev/null; }
lines() { wc -l < "$1" 2>/dev/null || echo 0; }
freemb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -dc 0-9; }
lane() {   # wait for room, then start one screen in the background: lane <out> <pool_eval args...>
  local out=$1; shift
  until [ "$(freemb)" -ge 7500 ] && [ "$(pgrep -fc 'pool_eval.py')" -lt 3 ]; do sleep 30; done
  ( for t in 1 2; do env $ENV python3 /root/work/pool_eval.py "$@" > /root/$(basename ${out%.jsonl}).log 2>&1
      grep -qE "out of memory|OutOfMemoryError" /root/$(basename ${out%.jsonl}).log || break; echo "[lane] $(basename $out) ran out of memory - again"; sleep 120; done ) &
  sleep 90   # its memory to show before the next lane looks
}
while :; do
  line=$(while read -r t rest; do [ -n "$t" ] && [ "${t:0:1}" != "#" ] && [ ! -e /root/rmr/$t.done ] && { echo "$t $rest"; break; }; done < /root/rm_queue.txt)
  [ -z "$line" ] && { sleep 300; continue; }
  set -- $line; T=$1; IT=$2; shift 2; EXTRA="$*"; mkdir -p /root/rmr; OUT=/root/online_$T; RUN=/root/${T}_run.log
  echo "[$T] start 2.1.0, items $IT, extra: ${EXTRA:-none} $(date -u +%H:%M)"
  t0=$(date +%s); rm -rf $OUT
  ( last=0; while sleep 60; do s=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null || echo 0)
      if [ "$s" != "$last" ] && [ $((s % 20)) -eq 0 ] && [ "$s" -gt 0 ] && [ ! -s /root/${T}_s$s.safetensors ]; then sleep 20; cp $OUT/latest.safetensors /root/${T}_s$s.safetensors
        hf upload $R /root/${T}_s$s.safetensors pooler_distill/chatsft/rmgrpo/${T}_s$s.safetensors >/dev/null 2>&1; echo "[$T] copy at step $s"
        [ "$s" -lt 60 ] && ( lane /root/work/${T}_s${s}st_0.jsonl /root/${T}_s$s.safetensors /root/work/ev_0.jsonl /root/work/${T}_s${s}st_0.jsonl --n 100 $EVARGS --tag "[${T}_s${s}st0]" ) &
      fi; last=$s; done ) & CP=$!
  env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 /root/work/online_loop.py /root/mtg7_s40.safetensors $OUT --pooler none --lora-layers all --lora-rank 16 --save-lora-only 1 \
    --mt-items $IT --heldout /root/work/eval300.jsonl --search-g 12 --demo-on-fail 0 --followup 0 --r1-on-fail 0 --cot-on-fail 0 \
    --steps 60 --save-every 20 --lr 5e-6 --search-lr 5e-6 --search-temp 0.9 --search-gen 2000 --temp 0.6 --gen 7000 --budget 3600 --maxsrch 7 --stop eos \
    --w-talk 0.5 --dolphin-min 0 --adv-std 1 --pg-norm mean --search-pg-norm const --search-pg-norm-len 256 --kl 0 --guard 1 --guard-steps 20 \
    --rm-model $M --w-rm 0.5 --rm-center $C --rm-scale $S --rm-device $D --rm-path /root/rmenv $EXTRA > $RUN 2>&1
  sleep 90; kill $CP 2>/dev/null
  ST=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null); tmin=$(( ($(date +%s) - t0) / 60 ))
  echo "[$T] stopped at step ${ST:-0} after $tmin min $(date -u +%H:%M)"; grep -E "Error|Traceback" $RUN | tail -2 | cut -c1-250
  [ "${ST:-0}" -gt 0 ] && [ ! -s /root/${T}_s$ST.safetensors ] && cp $OUT/latest.safetensors /root/${T}_s$ST.safetensors
  [ -s $OUT/rollouts.jsonl ] && hf upload $R $OUT/rollouts.jsonl pooler_distill/chatsft/rmgrpo/${T}_rollouts.jsonl >/dev/null 2>&1; hf upload $R $RUN pooler_distill/chatsft/rmgrpo/${T}_run.log >/dev/null 2>&1
  if [ "${ST:-0}" -lt 20 ]; then echo "$T FAILED at step ${ST:-0} ($EXTRA)" | tee -a /root/rm_rounds.txt; touch /root/rmr/$T.done; continue; fi
  blocks=$(python3 - $OUT/rollouts.jsonl <<'PB' 2>/dev/null
import json, sys, collections
b = collections.defaultdict(lambda: [0, 0, 0])
for l in open(sys.argv[1]):
    if not l.strip(): continue
    x = json.loads(l); v = b[(x["step"] - 1) // 20]; v[0] += 1; v[1] += x.get("reward", 0) >= 1.0; v[2] += bool(x.get("why", {}).get("unfinished"))
print(" ".join(f"{100*p//n}/{100*u//n}" for n, p, u in (b[k] for k in sorted(b))))
PB
)
  for k in 20 40 60; do f=/root/work/${T}_s${k}st_0.jsonl; [ -s /root/${T}_s$k.safetensors ] || continue
    pgrep -f "\[${T}_s${k}st0\]" >/dev/null && continue
    [ "$(lines $f)" -ge 100 ] || { rm -f $f; lane $f /root/${T}_s$k.safetensors /root/work/ev_0.jsonl $f --n 100 $EVARGS --tag "[${T}_s${k}st0]"; }
  done
  while pgrep -f "\[${T}_s[0-9]*st0\]" >/dev/null; do sleep 30; done
  best=""; bc=-1; sh=""
  for k in 20 40 60; do f=/root/work/${T}_s${k}st_0.jsonl; [ -s $f ] || continue; c=$(cnt $f); sh="$sh s$k $c"; [ "$c" -gt "$bc" ] && { bc=$c; best=$k; }; done
  B=${T}_s$best; CK=/root/$B.safetensors; echo "[$T] shard 0:$sh (start 56; nano-bonus rounds 46-53) - full screens for $B $(date -u +%H:%M)"
  [ "$(lines /root/work/${B}st_1.jsonl)" -ge 100 ] || lane /root/work/${B}st_1.jsonl $CK /root/work/ev_1.jsonl /root/work/${B}st_1.jsonl --n 100 $EVARGS --tag "[${B}st1]"
  [ "$(lines /root/work/${B}st_2.jsonl)" -ge 100 ] || lane /root/work/${B}st_2.jsonl $CK /root/work/ev_2.jsonl /root/work/${B}st_2.jsonl --n 100 $EVARGS --tag "[${B}st2]"
  [ "$(lines /root/work/br3_$B.jsonl)" -ge 150 ] || lane /root/work/br3_$B.jsonl $CK /root/work/eval300.jsonl /root/work/br3_$B.jsonl --multiturn /root/work/mt_eval_bridge3.jsonl --mt-mode win --n 1000 $EVARGS --tag "[br3-$B]"
  [ "$(lines /root/work/ch3_$B.jsonl)" -ge 180 ] || lane /root/work/ch3_$B.jsonl $CK /root/work/eval300.jsonl /root/work/ch3_$B.jsonl --multiturn /root/work/mt_eval_chain3.jsonl --mt-mode win --n 1000 $EVARGS --tag "[ch3-$B]"
  while pgrep -f "${B}st[12]\]|br3-$B\]|ch3-$B\]" >/dev/null; do sleep 30; done
  s300=$(( $(cnt /root/work/${B}st_0.jsonl) + $(cnt /root/work/${B}st_1.jsonl) + $(cnt /root/work/${B}st_2.jsonl) ))
  q() { PYTHONPATH=/root/rmenv python3 /root/work/rm_quality.py $M $1 x 2>/dev/null | sed -E 's/.*rm_mean_correct ([-0-9.na]+).*len_mean_correct ([0-9na]+).*/\1@\2/'; }
  sum="$T [${EXTRA:-base}] train ${tmin}min pass/unfin% by 20 steps: $blocks | shard0:$sh | best $B: single $s300/300 (2.1.0 162), br3 $(turns /root/work/br3_$B.jsonl 2) (35 31), ch3 $(turns /root/work/ch3_$B.jsonl 3) (28 17 4) | rm@len st0 $(q /root/work/${B}st_0.jsonl) br3 $(q /root/work/br3_$B.jsonl) ch3 $(q /root/work/ch3_$B.jsonl) (2.1.0: $(q /root/work/mtg7_s40st_0.jsonl) / $(q /root/work/br3_mtg7_s40.jsonl) / $(q /root/work/ch3_mtg7_s40.jsonl))"
  echo "$sum" | tee -a /root/rm_rounds.txt
  for f in /root/work/${T}_s*st_*.jsonl /root/work/br3_${T}_*.jsonl /root/work/ch3_${T}_*.jsonl; do [ -s $f ] && hf upload $R $f pooler_distill/chatsft/rmgrpo/screens/$(basename $f) >/dev/null 2>&1; done
  hf upload $R /root/rm_rounds.txt pooler_distill/chatsft/rmgrpo/rm_rounds.txt >/dev/null 2>&1
  touch /root/rmr/$T.done
done
