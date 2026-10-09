#!/bin/bash
# GRPO in short rounds that only move forward (2026-10-08, after mtg7: 56 at step 40, then 47 and 44 - the copies past the
# peak read the page and answered from memory). Each round: 60 steps from the current start on the next block of fresh
# questions (40 TriviaQA + 20 MuSiQue two-hop from the pools), copies every 20 steps (on the hub), the shard-0 screen for
# each; the best copy above the start's shard 0 gets the full screens and becomes the next start only if it holds the
# single-turn 300, the two-turn br3 (within 2) and the Dolphin hundred (within 3). Otherwise the start is kept and the
# next round takes the next questions. The Dolphin (reasoning) problems get no search at all (--reason-no-search: a sample
# that writes a search tag is cut there, unfinished) - the user, after mtg7's searches on maths problems.
# From mtg9 (2026-10-09): the search side's policy gradient is divided by a fixed 256 tokens, not by each rollout's own length
# (--search-pg-norm const). mtg8 under the per-rollout mean: unfinished search rollouts 11% -> 35%, searches 2.75 -> 4.06 per
# rollout, the shard-0 screens 53 -> 51 -> 46 (start 56) - a long rollout that never answers was punished half as hard per token.
#   state: /root/round_n.txt (the round's number), /root/round_start.txt ("<ckpt> <tag of its lines>"), /root/lines/<tag>.txt
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
cnt() { python3 -c "import json,sys;print(sum(bool(json.loads(l).get('correct')) for l in open(sys.argv[1]) if l.strip()))" "$1" 2>/dev/null || echo 0; }
while :; do
  [ -e /root/.rounds_stop ] && { echo "ROUNDS_STOPPED $(date -u)"; exit 0; }
  N=$(cat /root/round_n.txt); read START ST < /root/round_start.txt; read _ L0 L300 LB1 LB2 LDL < /root/lines/$ST.txt
  K=$((N - 8)); OUT=/root/online_mtg$N; RUN=/root/mtg${N}_run.log
  python3 - "$N" "$K" <<'PI'
import json, random, sys
N, K = int(sys.argv[1]), int(sys.argv[2])
t = [dict(json.loads(l), src="triviaqa") for l in open("/root/work/tqa_pool.jsonl") if l.strip()][600 + 40 * K: 640 + 40 * K]
m = [dict(json.loads(l), src="musique") for l in open("/root/work/musique_pool.jsonl") if l.strip()][200 + 20 * K: 220 + 20 * K]
it = [{k: x[k] for k in ("q", "gold", "golds", "hist", "src") if k in x} for x in t + m]; random.Random(N).shuffle(it)
open(f"/root/work/mtg{N}_items.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in it))
print(f"[mtg{N}] items: {len(t)} TriviaQA + {len(m)} MuSiQue two-hop", flush=True)
PI
  echo "[mtg$N] start $START ($ST: shard 0 $L0, single $L300/300, br3 $LB1/$LB2, Dolphin $LDL) $(date -u +%H:%M); $(df -h /root | tail -1 | awk '{print $4}') free"
  rm -rf $OUT
  ( last=0; while sleep 60; do s=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null || echo 0)
      if [ "$s" != "$last" ] && [ $((s % 20)) -eq 0 ] && [ "$s" -gt 0 ] && [ ! -s /root/mtg${N}_s$s.safetensors ]; then sleep 20; cp $OUT/latest.safetensors /root/mtg${N}_s$s.safetensors
        hf upload $R /root/mtg${N}_s$s.safetensors pooler_distill/chatsft/multiturn/mtg${N}_s$s.safetensors >/dev/null 2>&1; echo "[mtg$N] copy at step $s (on the hub)"; fi; last=$s; done ) & CP=$!
  env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OAI_KEY=$(cat /root/.oai 2>/dev/null) DSK_KEY=$(cat /root/.dsk 2>/dev/null) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 /root/work/online_loop.py $START $OUT --pooler none --lora-layers all --lora-rank 16 --save-lora-only 1 \
    --mt-items /root/work/mtg${N}_items.jsonl --heldout /root/work/eval300.jsonl --reason /root/work/dolphin_rft.jsonl --reason-every 4 --reason-g 12 --search-g 12 \
    --demo-on-fail 0.5 --followup 0.5 --r1-on-fail 1 --cot-on-fail 0.5 --reason-no-search 1 \
    --steps 60 --save-every 20 --lr 5e-6 --search-lr 5e-6 --search-temp 0.9 --search-gen 2000 --temp 0.6 --gen 7000 --budget 3600 --maxsrch 7 --stop eos \
    --judge-api openai --judge-model gpt-5-nano --w-talk 0.5 --dolphin-min 0 --adv-std 1 --pg-norm mean --search-pg-norm const --search-pg-norm-len 256 --kl 0 \
    --guard 1 --guard-steps 20 > $RUN 2>&1
  sleep 90; kill $CP 2>/dev/null
  grep -E "Error|Traceback" $RUN | tail -2 | cut -c1-250
  S=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null); echo "[mtg$N] stopped at step ${S:-0} $(date -u +%H:%M)"
  [ "${S:-0}" -gt 0 ] && [ ! -s /root/mtg${N}_s$S.safetensors ] && cp $OUT/latest.safetensors /root/mtg${N}_s$S.safetensors
  for f in rollouts.jsonl followups.jsonl r1_onfly.jsonl; do [ -s $OUT/$f ] && hf upload $R $OUT/$f pooler_distill/chatsft/multiturn/mtg${N}_$f >/dev/null 2>&1; done
  python3 - "$N" <<'PYL'
import json, sys
N = sys.argv[1]
qs = {json.loads(l)["q"] for l in open(f"/root/online_mtg{N}/rollouts.jsonl") if json.loads(l).get("kind") == "search"}
with open("/root/work/trained_items.jsonl", "a") as f:
    for q in qs: f.write(json.dumps({"q": q, "run": f"mtg{N}"}, ensure_ascii=False) + "\n")
print(f"[mtg{N}] {len(qs)} search questions used -> the ledger", flush=True)
PYL
  hf upload $R /root/work/trained_items.jsonl pooler_distill/chatsft/multiturn/trained_items.jsonl >/dev/null 2>&1
  rm -f "${OUT:?}"/*.pt "${OUT:?}"/good.safetensors
  if [ "${S:-0}" -lt 20 ]; then echo "[mtg$N] too short to screen - the round is run again"; rm -f /root/mtg${N}_s*.safetensors; sleep 600; continue; fi
  BEST=; BC=$L0
  for CK in $(ls /root/mtg${N}_s*.safetensors 2>/dev/null | sort -V); do
    T=$(basename $CK .safetensors)
    env $ENV python3 /root/work/pool_eval.py $CK /root/work/ev_0.jsonl /root/work/${T}st_0.jsonl --n 100 $EVARGS --tag "[${T}st0]" > /root/${T}st_0.log 2>&1
    c=$(cnt /root/work/${T}st_0.jsonl); echo "[mtg$N] $T single shard 0: $c/100 (start $L0)"; [ "$c" -gt "$BC" ] && { BC=$c; BEST=$CK; }
  done
  ACC=0
  if [ -n "$BEST" ]; then
    T=$(basename $BEST .safetensors); echo "[mtg$N] full screens for $T"
    read _ c0 c300 b1 b2 dl < <(bash /root/work/grpo_screen.sh $BEST $T)
    echo "[mtg$N] $T: single $c300/300 (start $L300), br3 $b1/$b2 (start $LB1/$LB2), Dolphin $dl (start $LDL)"
    if [ "$c300" -ge "$L300" ] && [ $((b1 + b2)) -ge $((LB1 + LB2 - 2)) ] && [ "$dl" -ge $((LDL - 3)) ]; then
      ACC=1; cp $BEST /root/start_mtg${N}.safetensors; echo "/root/start_mtg${N}.safetensors $T" > /root/round_start.txt
      hf upload $R /root/start_mtg${N}.safetensors pooler_distill/chatsft/multiturn/accepted_mtg${N}_$T.safetensors >/dev/null 2>&1
      echo "[mtg$N] ACCEPTED $T as the next start"
    fi
  fi
  [ "$ACC" = 1 ] || echo "[mtg$N] start kept ($ST)"
  rm -f /root/mtg${N}_s*.safetensors; rm -rf /root/evalrun_*
  echo $((N + 1)) > /root/round_n.txt; echo "ROUND_DONE mtg$N $(date -u)"
done
