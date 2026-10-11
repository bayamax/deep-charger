#!/bin/bash
# Run pool_eval.py jobs side by side on the one card (2026-10-11, the user: nine hours of screens is not sane - parallelise).
# One pool_eval holds ~6.5 GB of the 12 GB card at ~25% use, decoding one question at a time; several at once fill the
# card. Up to LANES jobs run together; a job that runs out of GPU memory is queued again and run once fewer are running
# (pool_eval resumes from what its output already holds). A multi-turn set is split into LANES parts by dialogue and the
# parts are joined after, so every screen is the same questions, the same settings, the same yardstick as before.
#   ENV="SP_BASE=... SP_RANK=16 ..." EVARGS="..." bash par_eval.sh <lanes> <ckpt> <jobfile>
#   jobfile lines:  single <questions.jsonl> <out.jsonl> [n]      |      multi <dialogues.jsonl> <out.jsonl>
LANES=$1; CK=$2; JOBS=$3; cd /root/work; : > /root/par_eval.status
Q=(); MERGE=()
while read -r kind src out n; do
  [ -z "$kind" ] && continue
  if [ "$kind" = single ]; then Q+=("$out|$src --n ${n:-100}")
  else
    for k in $(seq 0 $((LANES - 1))); do
      python3 -c "import sys;ls=[l for l in open(sys.argv[1]) if l.strip()];open(sys.argv[2],'w').write(''.join(l for i,l in enumerate(ls) if i%int(sys.argv[3])==int(sys.argv[4])))" $src ${out%.jsonl}.d$k.jsonl $LANES $k
      Q+=("${out%.jsonl}.p$k.jsonl|/root/work/eval300.jsonl --multiturn ${out%.jsonl}.d$k.jsonl --mt-mode win --n 1000")
    done
    MERGE+=("$out")
  fi
done < $JOBS
run() {   # $1 = out|args
  local out=${1%%|*} args=${1#*|}
  env $ENV python3 /root/work/pool_eval.py $CK ${args%% *} $out ${args#* } $EVARGS --tag "[par $(basename $out .jsonl)]" > ${out%.jsonl}.log 2>&1
  if grep -qE "out of memory|OutOfMemoryError" ${out%.jsonl}.log; then echo "OOM $1"; else echo "DONE $(basename $out)"; fi
}
retry=(); live=0; i=0
while [ $i -lt ${#Q[@]} ] || [ $live -gt 0 ]; do
  while [ $i -lt ${#Q[@]} ] && [ $live -lt $LANES ]; do
    ( r=$(run "${Q[$i]}"); echo "$r" >> /root/par_eval.status ) & live=$((live + 1)); i=$((i + 1)); sleep 20
  done
  wait -n; live=$((live - 1))
done
grep "^OOM " /root/par_eval.status 2>/dev/null | sed 's/^OOM //' > /root/par_eval.retry; : > /root/par_eval.status
while IFS= read -r j; do [ -n "$j" ] && echo "[par] again alone after running out of memory: $(basename ${j%%|*})" && run "$j" >> /root/par_eval.status; done < /root/par_eval.retry
for out in "${MERGE[@]}"; do cat ${out%.jsonl}.p*.jsonl > $out; rm -f ${out%.jsonl}.d*.jsonl; done
echo "[par] done: $(grep -c . $JOBS) screens, $LANES lanes"
