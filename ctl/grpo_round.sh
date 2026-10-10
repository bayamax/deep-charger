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
# From 2026-10-10 the gate is the conversation's (see before the screens): single turn only has to hold; the later turns
# (br3 follow-up, chain3 turns 2-3) and the nano-judged replies have to gain.
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
  if [ "$(cat /root/round_screen_only.txt 2>/dev/null)" = "$N" ] && [ -s /root/mtg${N}_s20.safetensors ]; then
    # trained already (box G2's restart onto the new gate, 2026-10-10): its copies are screened, nothing is trained again
    rm -f /root/round_screen_only.txt; S=$(ls /root/mtg${N}_s*.safetensors | sed -E 's/.*_s([0-9]+)\.safetensors/\1/' | sort -n | tail -1)
    echo "[mtg$N] trained before (copies to step $S): screens only"
  else
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
  fi
  if [ "${S:-0}" -lt 20 ]; then echo "[mtg$N] too short to screen - the round is run again"; rm -f /root/mtg${N}_s*.safetensors; sleep 600; continue; fi
  # the gate (2026-10-10, the user: better second and third turns and better-fitting replies are worth taking even when
  # single-turn accuracy stays where it is). Single turn only has to hold: shard 0 at least the start's 300 rate less 6
  # (the start's own shard 0 was picked as the best of its round's copies, 56 against 54 per hundred over its 300), and the
  # full 300 within 6 of the start. Of the copies that hold shard 0, the two best get the conversation screens
  # (grpo_mtscreen.sh); the one that gains most on the later turns (br3 follow-up + chain3 turns 2 and 3) plus the judged
  # good replies (shard 0 + br3 + chain3) plus the conversations judged whole (br3 + chain3, conv_judge.py, from mtg10),
  # none of the three down by more than 3, by at least 4 in all, gets the full screens.
  [ -s /root/lines/${ST}_mt.txt ] || { echo "[mtg$N] conversation screens for the start $ST $(date -u +%H:%M)"; bash /root/work/grpo_mtscreen.sh $START $ST > /dev/null; }
  [ -s /root/lines/${ST}_conv.txt ] || bash /root/work/grpo_mtscreen.sh $START $ST > /dev/null   # the whole-conversation judge (2026-10-10), reusing the rest
  read _ M1 M2 MC1 MC2 MC3 MQ0 MQ1 MQ2 < /root/lines/${ST}_mt.txt; read _ CB0 CC0 < /root/lines/${ST}_conv.txt
  M0=$((M2 + MC2 + MC3)); Q0=$((MQ0 + MQ1 + MQ2)); C0=$((CB0 + CC0)); FLOOR=$((L300 / 3 - 6))
  echo "[mtg$N] start $ST conversation: br3 $M1/$M2, chain3 $MC1/$MC2/$MC3, good replies $MQ0+$MQ1+$MQ2 = $Q0, good conversations $CB0+$CC0 = $C0"
  CANDS=""
  for CK in $(ls /root/mtg${N}_s*.safetensors 2>/dev/null | sort -V); do
    T=$(basename $CK .safetensors)
    env $ENV python3 /root/work/pool_eval.py $CK /root/work/ev_0.jsonl /root/work/${T}st_0.jsonl --n 100 $EVARGS --tag "[${T}st0]" > /root/${T}st_0.log 2>&1
    c=$(cnt /root/work/${T}st_0.jsonl); echo "[mtg$N] $T single shard 0: $c/100 (floor $FLOOR)"; [ "$c" -ge "$FLOOR" ] && CANDS="$CANDS$c $CK"$'\n'
  done
  BEST=; BS=3
  for CK in $(printf '%s' "$CANDS" | grep . | sort -k1,1nr -k2,2Vr | head -2 | awk '{print $2}'); do
    T=$(basename $CK .safetensors); echo "[mtg$N] conversation screens for $T $(date -u +%H:%M)"
    read _ a1 a2 k1 k2 k3 p0 p1 p2 < <(bash /root/work/grpo_mtscreen.sh $CK $T); read _ e1 e2 < /root/lines/${T}_conv.txt
    M=$((a2 + k2 + k3)); Q=$((p0 + p1 + p2)); C=$((e1 + e2)); S=$((M - M0 + Q - Q0 + C - C0))
    echo "[mtg$N] $T conversation: br3 $a1/$a2, chain3 $k1/$k2/$k3, good replies $p0+$p1+$p2 = $Q, good conversations $e1+$e2 = $C; later turns $M vs $M0, good replies $Q vs $Q0, good conversations $C vs $C0, gain $S"
    [ $((M - M0)) -ge -3 ] && [ $((Q - Q0)) -ge -3 ] && [ $((C - C0)) -ge -3 ] && [ "$S" -gt "$BS" ] && { BS=$S; BEST=$CK; }
  done
  ACC=0
  if [ -n "$BEST" ]; then
    T=$(basename $BEST .safetensors); echo "[mtg$N] full screens for $T"
    read _ c0 c300 b1 b2 dl < <(bash /root/work/grpo_screen.sh $BEST $T)
    echo "[mtg$N] $T: single $c300/300 (start $L300), br3 $b1/$b2 (start $LB1/$LB2), Dolphin $dl (start $LDL)"
    if [ "$c300" -ge $((L300 - 6)) ] && [ "$b1" -ge $((LB1 - 3)) ] && [ "$dl" -ge $((LDL - 3)) ]; then
      ACC=1; cp $BEST /root/start_mtg${N}.safetensors; echo "/root/start_mtg${N}.safetensors $T" > /root/round_start.txt
      hf upload $R /root/start_mtg${N}.safetensors pooler_distill/chatsft/multiturn/accepted_mtg${N}_$T.safetensors >/dev/null 2>&1
      echo "[mtg$N] ACCEPTED $T as the next start"
    fi
  fi
  [ "$ACC" = 1 ] || echo "[mtg$N] start kept ($ST)"
  rm -f /root/mtg${N}_s*.safetensors; rm -rf /root/evalrun_*
  echo $((N + 1)) > /root/round_n.txt; echo "ROUND_DONE mtg$N $(date -u)"
done
