#!/bin/bash
# box RW2 (2026-10-14, the 3090 that replaces box RW's 3060 for the GRPO round: ~2.6x the speed for 1.5x the price) - box RW, the user: rent a second box, as cheap as possible, and try GRPO with a reward model's 0-1 score as
# the reward). First step: can a dedicated reward model tell Deep Charger's better replies from its worse ones? The
# Skywork-Reward-V2 (Qwen3) family scores the archived GRPO rollouts of mtg7-mtg10 (rm_scan.py). Pull-based as the others.
BOXRW2_SERIAL=17
if [ -f /root/.boxrw2_serial ] && [ "$(cat /root/.boxrw2_serial)" -gt "$BOXRW2_SERIAL" ] 2>/dev/null; then echo "BOXRW2_STALE $BOXRW2_SERIAL"; exit 0; fi
echo $BOXRW2_SERIAL > /root/.boxrw2_serial
mkdir -p /root/work /root/hfdl; cd /root/work
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
for f in rm_scan.py rm_server.py rm_quality.py online_loop.py pool_eval.py web_search.py; do
  :; done
mkdir -p /root/sb; for f in sentbart/prep_dolphin.py sentbart/train.py sentbart/train_e2e.py sentbart/sb_eval.py; do
  for try in 1 2 3; do curl -sS -o /root/sb/$(basename $f) "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/sb/$(basename $f) && break; sleep 5; done
done
for f in rm_rounds.sh; do curl -sS -o /root/work/$f.new "$RAW/$f?nocache=$(date +%s)" && bash -n /root/work/$f.new && mv /root/work/$f.new /root/work/$f; done
for f in rm_scan.py rm_server.py rm_quality.py rm_pair.py online_loop.py pool_eval.py web_search.py; do
  for try in 1 2 3; do curl -sS -o /root/work/$f "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/work/$f && break; sleep 5; done
done

# ---- one-time bootstrap ----
if [ ! -f /root/.bootstrapped ]; then
  mkdir /root/.bootstrap_lock 2>/dev/null || { echo "bootstrap already running"; exit 0; }
  echo "=== bootstrap $(date -u) ==="
  pip install -q "transformers>=4.51,<4.57" accelerate "huggingface_hub>=0.34,<1.0" safetensors 2>&1 | tail -1
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /root | tail -1; free -g | head -2
  touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"
fi

# ---- the mirror: the box log to the hub every 10 minutes ----
if [ ! -e /root/.rwmirror_r2 ]; then touch /root/.rwmirror_r2; pkill -f "rwmirro[r].sh"; sleep 1; fi
if [ ! -e /root/.rwmirror_r5 ]; then touch /root/.rwmirror_r5; pkill -f "rwmirro[r].sh"; sleep 1; fi
if [ ! -e /root/.rwmirror_r6 ]; then touch /root/.rwmirror_r6; pkill -f "rwmirro[r].sh"; sleep 1; fi
if [ ! -e /root/.rwmirror_r7 ]; then touch /root/.rwmirror_r7; pkill -f "rwmirro[r].sh"; sleep 1; fi
if [ ! -e /root/.rwmirror_r8 ]; then touch /root/.rwmirror_r8; pkill -f "rwmirro[r].sh"; sleep 1; fi
if ! pgrep -f "rwmirro[r].sh" >/dev/null; then
  cat > /root/rwmirror.sh <<'MR'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
while :; do
  { echo "=== boxRW2 $(date -u) ==="; echo "--- ctl.log ---"; tail -n 30 /root/ctl.log 2>/dev/null
    for f in /root/fromi.log /root/rm_scan.log /root/grpoboot.log /root/rmround.log /root/earlyscreen.log /root/parscreen.log /root/bartprep.log /root/bart7.log /root/rm_rounds.log /root/rm_rounds.txt /root/rm_pairs.txt; do [ -s $f ] && { echo "--- $f ---"; grep -vE "Warning|warn|Loading weights" $f | tail -n 30 | cut -c1-600; }; done
    [ -s /root/rm1_run.log ] && { echo "--- rm1 steps ---"; grep -E "^\[step|^\[rm\]|rollback|Traceback|out of memory|Error" /root/rm1_run.log | sed -E 's/\| ce=.*cumulative/| cum/' | tail -n 10 | cut -c1-240; }
    [ -s /root/online_rm1/rollouts.jsonl ] && { echo "--- rm1 by 20-step block ---"; python3 - /root/online_rm1/rollouts.jsonl <<'PB' 2>/dev/null
import json, sys, collections
b = collections.defaultdict(lambda: [0, 0, 0, 0, 0.0, 0])   # samples, passing, unfinished, searches, rm_p sum, rm n
for l in open(sys.argv[1]):
    if not l.strip(): continue
    x = json.loads(l); w = x.get("why", {}); v = b[(x["step"] - 1) // 20]
    v[0] += 1; v[1] += x.get("reward", 0) >= 1.0; v[2] += bool(w.get("unfinished")); v[3] += x.get("ns", 0)
    if "rm_p" in w: v[4] += w["rm_p"]; v[5] += 1
for k in sorted(b):
    n, p, u, s_, rp, rn = b[k]; print(f"  steps {20*k+1:3d}-{20*k+20:3d}: {n:3d} rollouts, pass {100*p/n:3.0f}%, unfinished {100*u/n:3.0f}%, searches {s_/n:.2f}, rm_p {rp/max(rn,1):.3f} ({rn})")
PB
    }
    f=$(ls -t /root/rm[2-9]*_run.log /root/rm[1-9][0-9]*_run.log 2>/dev/null | head -1); [ -n "$f" ] && { echo "--- $(basename $f) steps ---"; grep -E "^\[step|^\[qbatch|^\[rm\]|rollback|Traceback|out of memory|Error" $f | sed -E "s/\| ce=.*cumulative/| cum/" | tail -n 8 | cut -c1-230; }
    [ -s /root/sb/dolphin_once7/train.log ] && { echo "--- once7 on this box (train.log) ---"; grep -E "^\[eval|^\[best|^\[val\]|Error|Traceback|memory" /root/sb/dolphin_once7/train.log | tail -n 8 | cut -c1-260; grep "^\[step" /root/sb/dolphin_once7/train.log | tail -1 | cut -c1-200; }
    echo "--- gpu ---"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
    echo "--- disk ---"; df -h /root | tail -1
    echo "--- processes ---"; pgrep -fa "python3|kee[p].sh" | cut -c1-200; } > /root/boxlog.txt 2>&1
  hf upload $R /root/boxlog.txt pooler_distill/chatsft/audit/boxlog_RW2.txt > /root/mirror_up.txt 2>&1 || echo "MIRROR_UP_FAIL $(date -u)" > /proc/1/fd/1
  sleep 600
done
MR
  setsid nohup bash /root/rwmirror.sh > /dev/null 2>&1 < /dev/null &
  echo "RWMIRROR_LAUNCHED $(date -u)"
fi

# ---- step 2: the GRPO environment (box G2's, minus the teachers: this box has no judge keys), after the scan.
# The loop keeps its pinned transformers; the reward model gets its own library path (/root/rmenv). ----
if [ ! -e /root/.grpoboot ]; then touch /root/.grpoboot
  cat > /root/grpobootkeep.sh <<'GB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
until [ -f /root/.bootstrapped ]; do sleep 30; done
echo "[grpoboot] start $(date -u +%H:%M)"
pip install -q --target /root/rmenv "transformers>=4.51,<4.57" 2>&1 | grep -v WARNING | tail -1
pip install -q --target /root/rmenv --no-deps bitsandbytes 2>&1 | grep -v WARNING | tail -1   # the 4-bit reward model; torch comes from the system
pip install -q "transformers==4.44.2" "peft==0.12.0" "safetensors==0.8.0" "huggingface_hub>=0.34,<1.0" accelerate certifi datasets pyarrow scipy 2>&1 | grep -v WARNING | tail -1
python3 -c "import transformers; print('[grpoboot] loop transformers', transformers.__version__)"; PYTHONPATH=/root/rmenv python3 -c "import transformers; print('[grpoboot] reward model transformers', transformers.__version__)"
dl() { for try in 1 2 3 4 5 6; do hf download $R --include "$1" --local-dir /root/hfdl >/dev/null 2>&1 && return 0; sleep 20; done; echo "[grpoboot] dl failed: $1"; return 1; }
A=pooler_distill/chatsft/archive_G2/work
dl "box_recover/scripts/*"; dl "fft_out/pooler.pt"; dl "grpo_assets/mus_run/eval_step200/eval_heldout_300q_g2.jsonl"; dl "pooler_distill/pool_eval_cache.jsonl"
dl "pooler_distill/chatsft/multiturn/mt_eval_bridge3.jsonl"; dl "pooler_distill/chatsft/multiturn/mt_eval_chain3.jsonl"; dl "pooler_distill/chatsft/multiturn/mtg7_s40.safetensors"
dl "pooler_distill/chatsft/gq14_hf/*"; dl "$A/mtg11_items.jsonl"; dl "$A/mtg7_s40st_0.jsonl"; dl "$A/br3_mtg7_s40.jsonl"; dl "$A/ch3_mtg7_s40.jsonl"
SC=/root/hfdl/box_recover/scripts; for f in $SC/*.py $SC/*.sh; do b=$(basename $f); [ -s /root/work/$b ] || cp $f /root/work/; done
mkdir -p /root/work/fft_out /root/work/runtime; ln -sfn /root/hfdl/fft_out/pooler.pt /root/work/fft_out/pooler.pt; cp /root/work/web_search.py /root/work/runtime/web_search.py 2>/dev/null; touch /root/work/runtime/__init__.py
cp /root/hfdl/grpo_assets/mus_run/eval_step200/eval_heldout_300q_g2.jsonl /root/work/eval300.jsonl; cp /root/hfdl/pooler_distill/pool_eval_cache.jsonl /root/work/pool_eval_cache.jsonl
cp /root/hfdl/pooler_distill/chatsft/multiturn/mt_eval_bridge3.jsonl /root/hfdl/pooler_distill/chatsft/multiturn/mt_eval_chain3.jsonl /root/work/
cp /root/hfdl/pooler_distill/chatsft/multiturn/mtg7_s40.safetensors /root/mtg7_s40.safetensors; rm -rf /root/gptq_hf_gq14; cp -r /root/hfdl/pooler_distill/chatsft/gq14_hf /root/gptq_hf_gq14
cp /root/hfdl/$A/mtg11_items.jsonl /root/hfdl/$A/mtg7_s40st_0.jsonl /root/hfdl/$A/br3_mtg7_s40.jsonl /root/hfdl/$A/ch3_mtg7_s40.jsonl /root/work/
python3 - <<'PYB'
import json
qs = [l for l in open("/root/work/eval300.jsonl") if l.strip()][:300]
for i in range(3): open(f"/root/work/ev_{i}.jsonl", "w").writelines(qs[i::3])
print("[grpoboot] eval shards 3 x", len(qs) // 3)
PYB
[ -s /root/gptq_hf_gq14/model.safetensors ] && [ -s /root/mtg7_s40.safetensors ] && [ -s /root/work/mtg11_items.jsonl ] && echo "GRPOBOOT_DONE $(date -u)" || echo "GRPOBOOT_FAILED $(date -u)"
df -h /root | tail -1
GB
  setsid nohup bash -c 'bash /root/grpobootkeep.sh 2>&1 | tee -a /root/grpoboot.log' > /dev/null 2>&1 < /dev/null &
  echo "GRPOBOOT_LAUNCHED $(date -u)"
fi

# the loop's default inputs (box G2 had them from its bootstrap): the self-generated questions and Dolphin; the first try,
# stopped at step 0 for want of them, is cleared so the round starts again
if [ ! -e /root/.rw2_fix1 ] && grep -q "GRPOBOOT_DONE" /root/grpoboot.log 2>/dev/null; then touch /root/.rw2_fix1
  export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
  for f in "pooler_distill/selfq_*.jsonl" "pooler_distill/chatsft/dolphin_v1.jsonl" "pooler_distill/chatsft/replay_v1.jsonl"; do
    for try in 1 2 3; do hf download baya1116/hypernet-sp-distill --include "$f" --local-dir /root/hfdl >/dev/null 2>&1 && break; sleep 20; done; done
  cat /root/hfdl/pooler_distill/selfq_*.jsonl > /root/work/selfq_all.jsonl; cp /root/hfdl/pooler_distill/chatsft/dolphin_v1.jsonl /root/hfdl/pooler_distill/chatsft/replay_v1.jsonl /root/work/
  echo "RW2_FIX1 selfq $(wc -l < /root/work/selfq_all.jsonl) dolphin $(wc -l < /root/work/dolphin_v1.jsonl) $(date -u)"
  if grep -q "RMROUND_JOB_DONE" /root/rmround.log 2>/dev/null && [ ! -s /root/rm1_s20.safetensors ]; then
    pkill -f "rmroundkee[p].sh"; mv /root/rmround.log /root/rmround_try1.log; mv /root/rm1_run.log /root/rm1_run_try1.log 2>/dev/null; rm -f /root/.rmround /root/work/rm1_s*st_*.jsonl; echo "RW2_RMROUND_RESET $(date -u)"
  fi
fi
# ---- step 3: one round of GRPO with the reward model's score (2.1.0 = mtg7_s40 as the start, mtg11's fresh 60 questions,
# search turns only, the previous rounds' settings otherwise), then gold-scored screens: single-turn shard 0 for each copy,
# the best copy on the full 300, br3 and chain3, and the reward model's own reading of the replies beside the start's.
# Waits for /root/rm_choice.txt ("<model> <center> <scale> <device>"), written once the scan has been read. ----
if [ ! -e /root/.rmround ]; then touch /root/.rmround
  cat > /root/rmroundkeep.sh <<'RR'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
until grep -q "GRPOBOOT_DONE" /root/grpoboot.log 2>/dev/null && [ -s /root/rm_choice.txt ]; do sleep 60; done
read M C S D < /root/rm_choice.txt
OUT=/root/online_rm1; RUN=/root/rm1_run.log
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
cnt() { python3 -c "import json,sys;print(sum(bool(json.loads(l).get('correct')) for l in open(sys.argv[1]) if l.strip()))" "$1" 2>/dev/null || echo 0; }
turns() { python3 -c "
import json,sys; r=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(' '.join(str(sum(bool(x['correct']) for x in r if x['turn']==t)) for t in range(int(sys.argv[2]))))" "$1" "$2" 2>/dev/null; }
if [ ! -s /root/rm1_s60.safetensors ]; then
  echo "[rm1] start mtg7_s40 (2.1.0: shard 0 56, single 162/300, br3 35/31, chain3 28/17/4); reward model $M (center $C, scale $S, $D) $(date -u +%H:%M)"
  rm -rf $OUT
  ( last=0; while sleep 60; do s=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null || echo 0)
      if [ "$s" != "$last" ] && [ $((s % 20)) -eq 0 ] && [ "$s" -gt 0 ] && [ ! -s /root/rm1_s$s.safetensors ]; then sleep 20; cp $OUT/latest.safetensors /root/rm1_s$s.safetensors
        hf upload $R /root/rm1_s$s.safetensors pooler_distill/chatsft/rmgrpo/rm1_s$s.safetensors >/dev/null 2>&1; echo "[rm1] copy at step $s (on the hub)"; fi; last=$s; done ) & CP=$!
  env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 /root/work/online_loop.py /root/mtg7_s40.safetensors $OUT --pooler none --lora-layers all --lora-rank 16 --save-lora-only 1 \
    --mt-items /root/work/mtg11_items.jsonl --heldout /root/work/eval300.jsonl --search-g 12 \
    --demo-on-fail 0 --followup 0 --r1-on-fail 0 --cot-on-fail 0 \
    --steps 60 --save-every 20 --lr 5e-6 --search-lr 5e-6 --search-temp 0.9 --search-gen 2000 --temp 0.6 --gen 7000 --budget 3600 --maxsrch 7 --stop eos \
    --w-talk 0.5 --dolphin-min 0 --adv-std 1 --pg-norm mean --search-pg-norm const --search-pg-norm-len 256 --kl 0 --guard 1 --guard-steps 20 \
    --rm-model $M --w-rm 0.5 --rm-center $C --rm-scale $S --rm-device $D --rm-path /root/rmenv > $RUN 2>&1
  sleep 90; kill $CP 2>/dev/null
  grep -E "Error|Traceback" $RUN | tail -2 | cut -c1-250
  ST=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null); echo "[rm1] stopped at step ${ST:-0} $(date -u +%H:%M)"
  [ "${ST:-0}" -gt 0 ] && [ ! -s /root/rm1_s$ST.safetensors ] && cp $OUT/latest.safetensors /root/rm1_s$ST.safetensors
  [ "${ST:-0}" -gt 0 ] || { echo "[rm1] the training did not start: $(grep -E "Error|Traceback" $RUN | tail -1 | cut -c1-200)"; echo "RMROUND_JOB_DONE failed $(date -u)"; exit 1; }
  [ -s $OUT/rollouts.jsonl ] && hf upload $R $OUT/rollouts.jsonl pooler_distill/chatsft/rmgrpo/rm1_rollouts.jsonl >/dev/null 2>&1
  hf upload $R $RUN pooler_distill/chatsft/rmgrpo/rm1_run.log >/dev/null 2>&1
fi
best=""; bc=-1
for k in 20 40 60; do
  CK=/root/rm1_s$k.safetensors; [ -s $CK ] || continue; f=/root/work/rm1_s${k}st_0.jsonl
  [ "$(wc -l < $f 2>/dev/null || echo 0)" -ge 100 ] || env $ENV python3 /root/work/pool_eval.py $CK /root/work/ev_0.jsonl $f --n 100 $EVARGS --tag "[rm1_s${k}st0]" > /root/rm1_s${k}st_0.log 2>&1
  c=$(cnt $f); echo "[rm1] rm1_s$k single shard 0: $c/100 (start 56; the previous rounds' copies 47-49)"
  PYTHONPATH=/root/rmenv python3 /root/work/rm_quality.py $M $f rm1_s${k}_st0 2>/dev/null | sed 's/^/[rm1] /'
  [ "$c" -gt "$bc" ] && { bc=$c; best=$k; }
done
PYTHONPATH=/root/rmenv python3 /root/work/rm_quality.py $M /root/work/mtg7_s40st_0.jsonl start_st0 2>/dev/null | sed 's/^/[rm1] /'
CK=/root/rm1_s$best.safetensors; T=rm1_s$best; echo "[rm1] best copy by shard 0: $T ($bc) - the full screens $(date -u +%H:%M)"
for i in 1 2; do f=/root/work/${T}st_$i.jsonl; [ "$(wc -l < $f 2>/dev/null || echo 0)" -ge 100 ] || env $ENV python3 /root/work/pool_eval.py $CK /root/work/ev_$i.jsonl $f --n 100 $EVARGS --tag "[${T}st$i]" > /root/${T}st_$i.log 2>&1; done
s300=$(( $(cnt /root/work/${T}st_0.jsonl) + $(cnt /root/work/${T}st_1.jsonl) + $(cnt /root/work/${T}st_2.jsonl) )); echo "[rm1] $T single: $s300/300 (start 162)"
f=/root/work/br3_$T.jsonl; [ "$(wc -l < $f 2>/dev/null || echo 0)" -ge 150 ] || env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl $f --multiturn /root/work/mt_eval_bridge3.jsonl --mt-mode win --n 1000 $EVARGS --tag "[br3-$T]" > /root/br3_$T.log 2>&1
echo "[rm1] $T br3: $(turns $f 2) (start 35 31)"; PYTHONPATH=/root/rmenv python3 /root/work/rm_quality.py $M $f ${T}_br3 2>/dev/null | sed 's/^/[rm1] /'; PYTHONPATH=/root/rmenv python3 /root/work/rm_quality.py $M /root/work/br3_mtg7_s40.jsonl start_br3 2>/dev/null | sed 's/^/[rm1] /'
g=/root/work/ch3_$T.jsonl; [ "$(wc -l < $g 2>/dev/null || echo 0)" -ge 180 ] || env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl $g --multiturn /root/work/mt_eval_chain3.jsonl --mt-mode win --n 1000 $EVARGS --tag "[ch3-$T]" > /root/ch3_$T.log 2>&1
echo "[rm1] $T chain3: $(turns $g 3) (start 28 17 4)"; PYTHONPATH=/root/rmenv python3 /root/work/rm_quality.py $M $g ${T}_ch3 2>/dev/null | sed 's/^/[rm1] /'; PYTHONPATH=/root/rmenv python3 /root/work/rm_quality.py $M /root/work/ch3_mtg7_s40.jsonl start_ch3 2>/dev/null | sed 's/^/[rm1] /'
for f in /root/work/rm1_s*st_*.jsonl /root/work/br3_rm1_*.jsonl /root/work/ch3_rm1_*.jsonl; do [ -s $f ] && hf upload $R $f pooler_distill/chatsft/rmgrpo/screens/$(basename $f) >/dev/null 2>&1; done
echo "RMROUND_JOB_DONE $(date -u)"
RR
  setsid nohup bash -c 'bash /root/rmroundkeep.sh 2>&1 | tee -a /root/rmround.log' > /dev/null 2>&1 < /dev/null &
  echo "RMROUND_LAUNCHED $(date -u)"
fi
# (the relaunch guard for the first round's keepers is gone: it brought the serial screens back beside the parallel ones)
# the scan (box RW, 3,600 rollouts of mtg7-mtg10, within-group pairwise): 0.6B 0.69 / 0.75 / 0.65, 1.7B 0.79 / 0.79 / 0.71,
# 4B 0.81 / 0.82 / 0.74 (right over wrong / solved over not / nano-good over not among the right), no length bias (-0.08).
# The 4B on the card (24 GB), centred on its mean score (5.75) with its within-group spread (3.0) as the scale.
# 2026-10-14, the 8B scan (box RW): Qwen3-8B in 4 bits 0.83 / 0.80 / 0.77, in 8 bits 0.83 / 0.81 / 0.76 - above the 4B on
# right-over-wrong and on the nano verdicts, ~5.5 GB on the card. Centred on its mean (0.13), its within-group spread (3.2).
[ -s /root/rm_choice.txt ] || { echo "Skywork/Skywork-Reward-V2-Qwen3-8B@4bit 0.13 3.2 cuda" > /root/rm_choice.txt; echo "RM_CHOICE $(cat /root/rm_choice.txt)"; }
# ---- 2026-10-14: box I (the context BART) stopped for the credit and its GPU went to another renter. Its sixth run's end
# and seventh run's state were on its disk only; Vast copies them here (/root/fromI), and this box puts them on the hub. ----
if ! pgrep -f "fromikee[p].sh" >/dev/null && [ ! -e /root/.fromi_done ]; then
  cat > /root/fromikeep.sh <<'FI'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
up() { [ -s "$1" ] || return 1; a=$(stat -c %s "$1"); sleep 90; [ "$(stat -c %s "$1")" = "$a" ] || return 1; hf upload $R "$1" "$2" >/dev/null 2>&1 && echo "[fromI] $1 -> $2 ($(du -h "$1" | cut -f1)) $(date -u +%H:%M)"; }
for i in $(seq 1 60); do
  [ -e /root/.fi6 ] || { up /root/fromI/once6/state_e2e.pt sentbart/dolphin/once6/state_end.pt && touch /root/.fi6; }
  [ -e /root/.fi7 ] || { up /root/fromI/once7/state_e2e.pt sentbart/dolphin/once7/state_latest.pt && touch /root/.fi7; }
  [ -e /root/.fi7b ] || { up /root/fromI/once7/model_best.pt sentbart/dolphin/once7/model_best_partial.pt && touch /root/.fi7b; }
  [ -e /root/.fi7l ] || { up /root/fromI/once7/train.log sentbart/dolphin/once7/train_partial.log && touch /root/.fi7l; }
  [ -e /root/.fi6 ] && [ -e /root/.fi7 ] && [ -e /root/.fi7b ] && [ -e /root/.fi7l ] && break
  sleep 60
done
for f in /root/fromI/logs/*.log; do [ -s $f ] && hf upload $R $f sentbart/dolphin/logs/$(basename $f) >/dev/null 2>&1; done
python3 -c "import torch; s=torch.load('/root/fromI/once7/state_e2e.pt', map_location='cpu', weights_only=False); print('[fromI] once7 state step', s.get('step'), 'best', s.get('best'), 'layers', s.get('layers'))" 2>/dev/null
python3 -c "import torch; s=torch.load('/root/fromI/once6/state_e2e.pt', map_location='cpu', weights_only=False); print('[fromI] once6 state step', s.get('step'), 'layers', s.get('layers'))" 2>/dev/null
ls -la /root/fromI/once6 /root/fromI/once7 2>&1 | head -12
touch /root/.fromi_done; echo "FROMI_DONE $(date -u)"
FI
  setsid nohup bash -c 'bash /root/fromikeep.sh 2>&1 | tee -a /root/fromi.log' > /dev/null 2>&1 < /dev/null &
  echo "FROMI_LAUNCHED $(date -u)"
fi

# ---- 2026-10-14, the user: one box at a time (box I closes once its work is on the hub), and this one used to the full:
# GPU and VRAM. During the round the card ran at ~23% with half its 24 GB free. ----
# (a) each copy's single-turn shard 0 is screened as soon as the copy is written, beside the training (one lane, ~6.5 GB)
if [ ! -e /root/.earlyscreen ]; then touch /root/.earlyscreen
  cat > /root/earlyscreenkeep.sh <<'ES'
cd /root/work
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
while ! grep -q "\[rm1\] stopped at step" /root/rmround.log 2>/dev/null; do
  for k in 20 40; do
    CK=/root/rm1_s$k.safetensors; f=/root/work/rm1_s${k}st_0.jsonl; e=/root/work/rm1_s${k}st_0e.jsonl
    [ -s $CK ] || continue; [ "$(wc -l < $f 2>/dev/null || echo 0)" -ge 100 ] && continue; [ -e /root/.es_$k ] && continue; touch /root/.es_$k
    echo "[early] rm1_s$k shard 0 beside the training $(date -u +%H:%M)"
    env $ENV python3 /root/work/pool_eval.py $CK /root/work/ev_0.jsonl $e --n 100 $EVARGS --tag "[rm1_s${k}st0e]" > /root/rm1_s${k}st_0e.log 2>&1
    [ "$(wc -l < $e 2>/dev/null || echo 0)" -ge 100 ] && [ "$(wc -l < $f 2>/dev/null || echo 0)" -lt 100 ] && cp $e $f
    echo "[early] rm1_s$k shard 0: $(python3 -c "import json,sys;print(sum(bool(json.loads(l).get('correct')) for l in open(sys.argv[1]) if l.strip()))" $e)/100 $(date -u +%H:%M)"
  done
  sleep 60
done
echo "EARLYSCREEN_DONE $(date -u)"
ES
  setsid nohup bash -c 'bash /root/earlyscreenkeep.sh 2>&1 | tee -a /root/earlyscreen.log' > /dev/null 2>&1 < /dev/null &
fi

# (b) after the training the screens run side by side (up to three lanes of ~6.5 GB on the 24 GB card) instead of one after
# another: the round keeper's serial screens are stopped and this keeper does them, with the same outputs and lines.
if [ ! -e /root/.parscreen ]; then touch /root/.parscreen
  cat > /root/parscreenkeep.sh <<'PS'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
read M C S D < /root/rm_choice.txt
cnt() { python3 -c "import json,sys;print(sum(bool(json.loads(l).get('correct')) for l in open(sys.argv[1]) if l.strip()))" "$1" 2>/dev/null || echo 0; }
turns() { python3 -c "
import json,sys; r=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(' '.join(str(sum(bool(x['correct']) for x in r if x['turn']==t)) for t in range(int(sys.argv[2]))))" "$1" "$2" 2>/dev/null; }
lines() { wc -l < "$1" 2>/dev/null || echo 0; }
until grep -q "\[rm1\] stopped at step" /root/rmround.log 2>/dev/null; do sleep 60; done
grep -q "RMROUND_JOB_DONE failed" /root/rmround.log && { echo "PARSCREEN_SKIPPED (no training)"; exit 0; }
until pgrep -f "pool_eval.py.*rm1_s[0-9]*st0\]" >/dev/null || grep -q "RMROUND_JOB_DONE" /root/rmround.log; do sleep 20; done
pkill -f "rmroundkee[p].sh"; sleep 1; pkill -f "pool_eval.py.*\[rm1_s[0-9]*st0\]"; pkill -f "rm_quality.py"; sleep 5
echo "[par] the round keeper's serial screens stopped; side by side from here $(date -u +%H:%M)" | tee -a /root/rmround.log
pe() { env $ENV python3 /root/work/pool_eval.py "$@" > /root/$(basename ${3%.jsonl}).log 2>&1; }
# shard 0 of each copy (the early keeper's run is waited for, not repeated)
for k in 20 40 60; do f=/root/work/rm1_s${k}st_0.jsonl; [ -s /root/rm1_s$k.safetensors ] || continue
  if pgrep -f "rm1_s${k}st_0e" >/dev/null; then ( while pgrep -f "rm1_s${k}st_0e" >/dev/null; do sleep 30; done; [ "$(lines /root/work/rm1_s${k}st_0e.jsonl)" -ge 100 ] && cp /root/work/rm1_s${k}st_0e.jsonl $f ) &
  elif [ "$(lines $f)" -lt 100 ]; then rm -f $f; pe /root/rm1_s$k.safetensors /root/work/ev_0.jsonl $f --n 100 $EVARGS --tag "[rm1_s${k}st0]" & sleep 20; fi
done; wait
best=""; bc=-1
for k in 20 40 60; do f=/root/work/rm1_s${k}st_0.jsonl; [ -s $f ] || continue; c=$(cnt $f)
  echo "[rm1] rm1_s$k single shard 0: $c/100 (start 56; the previous rounds' copies 47-49)" | tee -a /root/rmround.log
  [ "$c" -gt "$bc" ] && { bc=$c; best=$k; }
done
CK=/root/rm1_s$best.safetensors; T=rm1_s$best; echo "[rm1] best copy by shard 0: $T ($bc) - the full screens, side by side $(date -u +%H:%M)" | tee -a /root/rmround.log
[ "$(lines /root/work/${T}st_1.jsonl)" -ge 100 ] || { rm -f /root/work/${T}st_1.jsonl; pe $CK /root/work/ev_1.jsonl /root/work/${T}st_1.jsonl --n 100 $EVARGS --tag "[${T}st1]" & sleep 20; }
[ "$(lines /root/work/${T}st_2.jsonl)" -ge 100 ] || { rm -f /root/work/${T}st_2.jsonl; pe $CK /root/work/ev_2.jsonl /root/work/${T}st_2.jsonl --n 100 $EVARGS --tag "[${T}st2]" & sleep 20; }
[ "$(lines /root/work/br3_$T.jsonl)" -ge 150 ] || { rm -f /root/work/br3_$T.jsonl; pe $CK /root/work/eval300.jsonl /root/work/br3_$T.jsonl --multiturn /root/work/mt_eval_bridge3.jsonl --mt-mode win --n 1000 $EVARGS --tag "[br3-$T]" & sleep 20; }
wait
[ "$(lines /root/work/ch3_$T.jsonl)" -ge 180 ] || pe $CK /root/work/eval300.jsonl /root/work/ch3_$T.jsonl --multiturn /root/work/mt_eval_chain3.jsonl --mt-mode win --n 1000 $EVARGS --tag "[ch3-$T]"
s300=$(( $(cnt /root/work/${T}st_0.jsonl) + $(cnt /root/work/${T}st_1.jsonl) + $(cnt /root/work/${T}st_2.jsonl) ))
{ echo "[rm1] $T single: $s300/300 (start 162)"; echo "[rm1] $T br3: $(turns /root/work/br3_$T.jsonl 2) (start 35 31)"; echo "[rm1] $T chain3: $(turns /root/work/ch3_$T.jsonl 3) (start 28 17 4)"; } | tee -a /root/rmround.log
for x in "/root/work/${T}st_0.jsonl ${T}_st0" "/root/work/mtg7_s40st_0.jsonl start_st0" "/root/work/br3_$T.jsonl ${T}_br3" "/root/work/br3_mtg7_s40.jsonl start_br3" "/root/work/ch3_$T.jsonl ${T}_ch3" "/root/work/ch3_mtg7_s40.jsonl start_ch3"; do
  set -- $x; PYTHONPATH=/root/rmenv python3 /root/work/rm_quality.py $M $1 $2 2>/dev/null | sed 's/^/[rm1] /' | tee -a /root/rmround.log; done
for f in /root/work/rm1_s*st_*.jsonl /root/work/br3_rm1_*.jsonl /root/work/ch3_rm1_*.jsonl; do [ -s $f ] && hf upload $R $f pooler_distill/chatsft/rmgrpo/screens/$(basename $f) >/dev/null 2>&1; done
echo "RMROUND_JOB_DONE $(date -u)" | tee -a /root/rmround.log
PS
  setsid nohup bash -c 'bash /root/parscreenkeep.sh 2>&1 | tee -a /root/parscreen.log' > /dev/null 2>&1 < /dev/null &
fi

# (c) the context BART moves here from box I: its data is prepared now on this box's idle CPU (the same corpora, the same
# commands, so the same plan), and the seventh run resumes from box I's state once that is on the hub and the round's
# screens are done (BART ~8 GB beside them would not fit).
if [ ! -e /root/.bartprep ]; then touch /root/.bartprep
  cat > /root/bartprepkeep.sh <<'BP'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/sb
pip install -q blingfire zstandard 2>&1 | grep -v WARNING | tail -1
D=/root/sb/data/dolphin; AH=/root/sb/data/am_held; DB=/root/sb/data/r1b_768; D7=/root/sb/data/dolphin768; DF7=/root/sb/data/dolphin768full
J=/root/sb/data/dr1_deepseek.jsonl; Z=/root/sb/data/am_0.9M.jsonl.zst; mkdir -p $D $AH $DB $D7 $DF7
dl() { for try in 1 2 3 4; do [ -s $1 ] && [ "$(stat -c %s $1)" -ge $3 ] && return 0; curl -sSfL -C - -o $1 $2 || sleep 60; done; [ "$(stat -c %s $1 2>/dev/null || echo 0)" -ge $3 ]; }
hf download $R sentbart/dolphin/data/docs_001.jsonl --local-dir /root/hfdl >/dev/null 2>&1; cp /root/hfdl/sentbart/dolphin/data/docs_001.jsonl $D/docs_001.jsonl
python3 -c "import json, numpy as np; L=[len(json.loads(l)['sents']) for l in open('$D/docs_001.jsonl') if l.strip()]; np.save('$D/off_001.npy', np.cumsum([0]+L).astype(np.int64)); print('[bartprep] held-out', len(L), 'conversations')"
dl $J https://huggingface.co/datasets/cognitivecomputations/dolphin-r1/resolve/main/dolphin-r1-reasoning-deepseek.jsonl 3848000000 || { echo "BARTPREP_FAILED deepseek"; exit 1; }
dl $Z https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M/resolve/main/am_0.9M.jsonl.zst 3200000000 || { echo "BARTPREP_FAILED am"; exit 1; }
EX=/root/work/dolphin_v1.jsonl
python3 /root/sb/prep_dolphin.py --out $D7 --all 1 --shard 000 --max-sents 768 --max-q 24 --keep-eval $D/docs_001.jsonl --exclude $EX --jsonl $J 2>&1 | grep -E "PREP_DOLPHIN|Error|Traceback" | sed 's/^/[bartprep deepseek] /' | cut -c1-300 &
python3 /root/sb/prep_dolphin.py --out $DF7 --rewrite-eval $D/docs_001.jsonl --max-sents 768 --jsonl $J 2>&1 | grep -E "PREP_DOLPHIN|Error|Traceback" | sed 's/^/[bartprep held-out 768] /' | cut -c1-300 &
python3 /root/sb/prep_dolphin.py --out $AH --schema am --all 1 --shard 005 --id-prefix am1 --max-sents 768 --max-q 24 --skip-rows 342045 --stride 40 --max-docs 4000 --keep-eval $D/docs_001.jsonl --exclude-questions $D/docs_001.jsonl --exclude $EX --jsonl $Z 2>&1 | grep -E "PREP_DOLPHIN|Error|Traceback" | sed 's/^/[bartprep am held-out] /' | cut -c1-300
python3 /root/sb/prep_dolphin.py --out $DB --schema am --all 1 --shard 006 --id-prefix am1 --max-sents 768 --max-q 24 --skip-rows 342045 --keep-eval $D/docs_001.jsonl --exclude-questions $D/docs_001.jsonl $AH/docs_005.jsonl --exclude $EX --jsonl $Z 2>&1 | grep -E "PREP_DOLPHIN|Error|Traceback" | sed 's/^/[bartprep am] /' | cut -c1-300
wait
cp $D/docs_001.jsonl $D/off_001.npy $DB/; ln -sf $D7/docs_000.jsonl $DB/docs_000.jsonl; ln -sf $D7/off_000.npy $DB/off_000.npy
n0=$(wc -l < $DB/docs_000.jsonl); n6=$(wc -l < $DB/docs_006.jsonl); nh=$(wc -l < $AH/docs_005.jsonl)
echo "[bartprep] dolphin $n0 (box I: 245072), AM $n6 (box I: 505681), AM held-out $nh (4000) - $([ "$n0" = 245072 ] && [ "$n6" = 505681 ] && echo "the same set as box I's" || echo "NOT the same as box I's: the plan will differ"); $(df -h /root | tail -1 | awk '{print $4}') free"
rm -f $J $Z
echo ok > $DB/ready.ok; echo "BARTPREP_DONE $(date -u)"
BP
  setsid nohup bash -c 'bash /root/bartprepkeep.sh 2>&1 | tee -a /root/bartprep.log' > /dev/null 2>&1 < /dev/null &
fi
if [ ! -e /root/.bart7 ]; then touch /root/.bart7
  cat > /root/bart7keep.sh <<'B7'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/sb
DB=/root/sb/data/r1b_768; AH=/root/sb/data/am_held; DF7=/root/sb/data/dolphin768full
until [ -e $DB/ready.ok ] && grep -q "RMROUND_JOB_DONE" /root/rmround.log 2>/dev/null; do sleep 120; done
mkdir -p /root/sb/dolphin_once6 /root/sb/dolphin_once7
until hf download $R sentbart/dolphin/once6/state_end.pt sentbart/dolphin/once7/state_latest.pt --local-dir /root/hfdl >/dev/null 2>&1 && [ -s /root/hfdl/sentbart/dolphin/once7/state_latest.pt ] && [ -s /root/hfdl/sentbart/dolphin/once6/state_end.pt ]; do sleep 600; done
mv /root/hfdl/sentbart/dolphin/once6/state_end.pt /root/sb/dolphin_once6/state_e2e.pt; mv /root/hfdl/sentbart/dolphin/once7/state_latest.pt /root/sb/dolphin_once7/state_e2e.pt
hf download $R sentbart/dolphin/once7/model_best_partial.pt sentbart/dolphin/once7/train_partial.log --local-dir /root/hfdl >/dev/null 2>&1
[ -s /root/hfdl/sentbart/dolphin/once7/model_best_partial.pt ] && mv /root/hfdl/sentbart/dolphin/once7/model_best_partial.pt /root/sb/dolphin_once7/model_best.pt
[ -s /root/hfdl/sentbart/dolphin/once7/train_partial.log ] && cp /root/hfdl/sentbart/dolphin/once7/train_partial.log /root/sb/dolphin_once7/train.log
echo "[bart7] the seventh run resumes here from step $(python3 -c "import torch; print(torch.load('/root/sb/dolphin_once7/state_e2e.pt', map_location='cpu', weights_only=False).get('step'))") $(date -u +%H:%M)"
( while sleep 7200; do f=/root/sb/dolphin_once7/state_e2e.pt; [ -s $f ] && cp $f /root/once7_up.pt && hf upload $R /root/once7_up.pt sentbart/dolphin/once7/state_latest.pt >/dev/null 2>&1 && echo "[bart7] state on the hub $(date -u +%H:%M)"; rm -f /root/once7_up.pt; done ) &
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -u /root/sb/train_e2e.py --vec $DB --text $DB --eval-text $DB/docs_001.jsonl --out /root/sb/dolphin_once7 --init /root/sb/dolphin_once6/state_e2e.pt --layers 16 --enc-layers 0 \
  --buckets 64,128,192,256,384,512,768 --bucket-order random --sent-budget 1900 --plan-accum 2 --max-bs 64 --eval-seq 128 --page-queue 16 --pool 0 --anchor-q 0 --select dec_top1 --patience 0 \
  --eval-every 1000 --save-every 1000 2>&1 | grep --line-buffered -E "^\[e2e\]|^\[eval|^\[best|^\[val\]|^\[init|^\[model|TRAIN_DONE|Error|Traceback|out of memory" | sed -u 's/^\(.\{400\}\).*/\1/'
[ -s /root/sb/dolphin_once7/model_best.pt ] && hf upload $R /root/sb/dolphin_once7/model_best.pt sentbart/dolphin/once7/model_best.pt >/dev/null 2>&1 && echo "[bart7] best on the hub"
hf upload $R /root/sb/dolphin_once7/train.log sentbart/dolphin/once7/train.log >/dev/null 2>&1; hf upload $R /root/sb/dolphin_once7/state_e2e.pt sentbart/dolphin/once7/state_end.pt >/dev/null 2>&1
hf download $R sentbart/dolphin/once4/model_best.pt --local-dir /root/hfdl >/dev/null 2>&1
CK=""; for spec in run4_best=/root/hfdl/sentbart/dolphin/once4/model_best.pt run6_end=/root/sb/dolphin_once6/state_e2e.pt run7_best=/root/sb/dolphin_once7/model_best.pt run7_end=/root/sb/dolphin_once7/state_e2e.pt; do [ -s "${spec#*=}" ] && CK="$CK $spec"; done
python3 -u /root/sb/sb_eval.py --docs $DF7/docs_001.jsonl --ckpt $CK --bands 128,256,512,768 --out /root/sb_eval768b.json 2>&1 | grep --line-buffered -vE "Warning|warn|documents \(" | sed 's/^/[bart7 dolphin] /' | cut -c1-300
python3 -u /root/sb/sb_eval.py --docs $AH/docs_005.jsonl --ckpt $CK --bands 128,256,512,768 --out /root/sb_eval_amb.json 2>&1 | grep --line-buffered -vE "Warning|warn|documents \(" | sed 's/^/[bart7 am] /' | cut -c1-300
python3 - <<'PC'
import json
a = json.load(open("/root/sb_eval768b.json"))["runs"]; b = json.load(open("/root/sb_eval_amb.json"))["runs"]
print("[bart7 composite] top-1 at 3223+1 candidates, all hidden positions: dolphin | AM | mean")
for k in a:
    if k in b: print(f"  {k:10s} {a[k]['all']['top1']:.3f} | {b[k]['all']['top1']:.3f} | {(a[k]['all']['top1'] + b[k]['all']['top1']) / 2:.3f}")
PC
hf upload $R /root/sb_eval768b.json sentbart/dolphin/sb_eval/sb_eval768b.json >/dev/null 2>&1; hf upload $R /root/sb_eval_amb.json sentbart/dolphin/sb_eval/sb_eval_amb.json >/dev/null 2>&1
echo "BART7_JOB_DONE $(date -u)"
B7
  setsid nohup bash -c 'bash /root/bart7keep.sh 2>&1 | tee -a /root/bart7.log' > /dev/null 2>&1 < /dev/null &
fi
for k in earlyscreen parscreen bartprep; do
  if [ -s /root/${k}keep.sh ] && ! pgrep -f "${k}kee[p].sh" >/dev/null && ! grep -qE "EARLYSCREEN_DONE|PARSCREEN_SKIPPED|RMROUND_JOB_DONE|BARTPREP_DONE|BARTPREP_FAILED|BART7_JOB_DONE" /root/$k.log 2>/dev/null; then
    [ "$k" = parscreen ] && grep -q "RMROUND_JOB_DONE" /root/rmround.log 2>/dev/null && continue
    setsid nohup bash -c "bash /root/${k}keep.sh 2>&1 | tee -a /root/$k.log" > /dev/null 2>&1 < /dev/null &
  fi
done

# ---- 2026-10-14, the user (asleep after this): after the first round, more rounds with the reward model - it is new, so
# try things - each on the same questions as an earlier nano-bonus round where possible (mtg10's, mtg9's), so only the
# reward differs; the rollouts of three steps decoded together (--qbatch 3) to fill the card. The queue can be edited. ----
if [ ! -e /root/.rmrounds ]; then touch /root/.rmrounds
  export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
  for n in 8 9 10; do f=pooler_distill/chatsft/archive_G2/work/mtg${n}_items.jsonl; for t in 1 2 3; do hf download baya1116/hypernet-sp-distill $f --local-dir /root/hfdl >/dev/null 2>&1 && break; sleep 20; done; cp /root/hfdl/$f /root/work/; done
  [ -s /root/rm_queue.txt ] || cat > /root/rm_queue.txt <<'QQ'
# tag items [extra flags]  - each round from 2.1.0, 60 steps
rm2 /root/work/mtg10_items.jsonl --qbatch 3
rm3 /root/work/mtg10_items.jsonl --qbatch 3 --w-rm 1.0
rm4 /root/work/mtg9_items.jsonl --qbatch 3 --w-rm 1.0 --adv-std 0
QQ
  cat > /root/rmroundskeep.sh <<'RK'
until grep -q "RMROUND_JOB_DONE" /root/rmround.log 2>/dev/null; do sleep 60; done
exec bash /root/work/rm_rounds.sh
RK
  setsid nohup bash -c 'bash /root/rmroundskeep.sh 2>&1 | tee -a /root/rm_rounds.log' > /dev/null 2>&1 < /dev/null &
  echo "RMROUNDS_LAUNCHED $(date -u)"
fi
if [ -e /root/.rmrounds ] && ! pgrep -f "rmroundskee[p].sh|rm_rounds.sh" >/dev/null; then
  setsid nohup bash -c 'bash /root/rmroundskeep.sh 2>&1 | tee -a /root/rm_rounds.log' > /dev/null 2>&1 < /dev/null &
fi
# the next round starts once round 1's shard-0 screens are done (its last full screens run alone on a quarter of the card)
if [ ! -e /root/.rmrounds_early ] && [ -e /root/.rmrounds ] && [ ! -e /root/rmr/rm2.trained ] && ! pgrep -f "rm_rounds.sh" >/dev/null; then touch /root/.rmrounds_early
  cat > /root/rmroundskeep.sh <<'RK'
until grep -qE "best copy by shard 0|RMROUND_JOB_DONE" /root/rmround.log 2>/dev/null; do sleep 60; done
exec bash /root/work/rm_rounds.sh
RK
  pkill -f "rmroundskee[p].sh"; sleep 1
  setsid nohup bash -c 'bash /root/rmroundskeep.sh 2>&1 | tee -a /root/rm_rounds.log' > /dev/null 2>&1 < /dev/null &
  echo "RMROUNDS_EARLY $(date -u)"
fi
# round 2 ran without its reward model (it could not load beside round 1's screens and the batched rollouts, and was
# retried every step): it is abandoned and run again as rm2b with the loop that loads the reward model first and stops
# when it cannot; the runner restarts with the 9 GB lane threshold
if [ ! -e /root/.rm2_redo ] && grep -q "def rm_start" /root/work/online_loop.py && grep -q "ge 9000" /root/work/rm_rounds.sh; then touch /root/.rm2_redo
  cat > /root/rm_queue.txt <<'QQ'
# tag items [extra flags]  - each round from 2.1.0, 60 steps
rm2 /root/work/mtg10_items.jsonl --qbatch 3
rm2b /root/work/mtg10_items.jsonl --qbatch 3
rm3 /root/work/mtg10_items.jsonl --qbatch 3 --w-rm 1.0
rm4 /root/work/mtg9_items.jsonl --qbatch 3 --w-rm 1.0 --adv-std 0
QQ
  pkill -f "rmroundskee[p].sh"; pkill -f "rm_rounds.sh"; sleep 1; pkill -f "online_loop.py.*online_rm2 "; pkill -f "rm_server.py"; sleep 5
  mkdir -p /root/rmr; touch /root/rmr/rm2.trained /root/rmr/rm2.done
  echo "rm2 ABANDONED at step $(python3 -c "import json;print(json.load(open('/root/online_rm2/state.json'))['step'])" 2>/dev/null || echo '?'): the reward model never loaded (memory) - run again as rm2b" | tee -a /root/rm_rounds.txt
  setsid nohup bash -c 'bash /root/rmroundskeep.sh 2>&1 | tee -a /root/rm_rounds.log' > /dev/null 2>&1 < /dev/null &
  echo "RM2_REDO $(date -u)"
fi
# rm2b, rm3 and rm4 stopped at step 0: the first round's serial screens had come back (the relaunch guard) beside the
# parallel ones and left no room for the reward model. The serial keeper goes; the runner waits for 14 GB before a round;
# the queue again, with a lower learning rate added (round 1: shard 0 52 at step 60, but single 153/300 and br3 31/29
# against 2.1.0's 162 and 35/31 - the drift the nano-bonus rounds showed too)
if [ ! -e /root/.rmq2 ] && grep -q "ge 14000" /root/work/rm_rounds.sh; then touch /root/.rmq2
  pkill -f "rmroundkee[p].sh"; pkill -f "rm_quality.py"; pkill -f "rmroundskee[p].sh"; pkill -f "rm_rounds.sh"; sleep 3
  cat > /root/rm_queue.txt <<'QQ'
# tag items [extra flags]  - each round from 2.1.0, 60 steps
rm2c /root/work/mtg10_items.jsonl --qbatch 3
rm5 /root/work/mtg10_items.jsonl --qbatch 3 --lr 2.5e-6 --search-lr 2.5e-6
rm3b /root/work/mtg10_items.jsonl --qbatch 3 --w-rm 1.0
rm6 /root/work/mtg9_items.jsonl --qbatch 3 --w-rm 1.0 --lr 2.5e-6 --search-lr 2.5e-6
QQ
  setsid nohup bash -c 'bash /root/rmroundskeep.sh 2>&1 | tee -a /root/rm_rounds.log' > /dev/null 2>&1 < /dev/null &
  echo "RMQ2 $(date -u)"
fi
# round 1 lifted the reward model's reading of single-turn replies (-0.41 -> -0.15) but lowered it on br3 (1.77 -> 1.21) and
# chain3 (-0.82 -> -1.85): trained on single-turn search turns only, the later turns of a conversation suffered. rm7 mixes
# in later turns: mtg10's 60 questions with 30 conversation turns from mtg2's items (an earlier exchange in the history).
if [ ! -e /root/.rm7 ]; then touch /root/.rm7
  export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
  f=pooler_distill/chatsft/archive_G2/work/mtg2_items.jsonl; for t in 1 2 3; do hf download baya1116/hypernet-sp-distill $f --local-dir /root/hfdl >/dev/null 2>&1 && break; sleep 20; done
  python3 - <<'PM'
import json, random
a = [json.loads(l) for l in open("/root/work/mtg10_items.jsonl") if l.strip()]
b = [json.loads(l) for l in open("/root/hfdl/pooler_distill/chatsft/archive_G2/work/mtg2_items.jsonl") if l.strip()]
b = [x for x in b if x.get("hist") and len(x["hist"]) in (2, 4)]; random.Random(7).shuffle(b); b = b[:30]
it = a + [{"q": x["q"], "gold": x["gold"], "golds": [x["gold"]], "hist": x["hist"], "src": "mtg2_turn"} for x in b]; random.Random(11).shuffle(it)
open("/root/work/mix10_items.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in it))
print(f"[rm7] items: {len(a)} mtg10 + {len(b)} later turns", flush=True)
PM
  grep -q "^rm7 " /root/rm_queue.txt || echo "rm7 /root/work/mix10_items.jsonl --qbatch 3" >> /root/rm_queue.txt
  echo "RM7_QUEUED $(date -u)"
fi
# rm7 stays at 60 steps (the runner screens the copies at 20/40/60): 60 of the 90 mixed items
[ -e /root/.rm7b ] || { touch /root/.rm7b; sed -i "s#^rm7 /root/work/mix10_items.jsonl --qbatch 3 --steps 90#rm7 /root/work/mix10_items.jsonl --qbatch 3#" /root/rm_queue.txt; grep "^rm7" /root/rm_queue.txt; }
# ---- the seventh run on this box fell over at its start beside the round (no room) and its keeper then put the partial
# state up as the end. It goes again: wait for 10 GB free, resume from the saved state after a crash, and only a run that
# reached TRAIN_DONE is uploaded as the end. ----
if [ ! -e /root/.bart7b ]; then touch /root/.bart7b
  pkill -f "bart7kee[p].sh"; sleep 1
  cat > /root/bart7bkeep.sh <<'B7'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/sb
DB=/root/sb/data/r1b_768; AH=/root/sb/data/am_held; DF7=/root/sb/data/dolphin768full; O=/root/sb/dolphin_once7
freemb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -dc 0-9; }
mkdir -p /root/sb/dolphin_once6 $O
[ -s /root/sb/dolphin_once6/state_e2e.pt ] || { hf download $R sentbart/dolphin/once6/state_end.pt --local-dir /root/hfdl >/dev/null 2>&1; mv /root/hfdl/sentbart/dolphin/once6/state_end.pt /root/sb/dolphin_once6/state_e2e.pt; }
if [ ! -s $O/state_e2e.pt ] || [ ! -e /root/.bart7b_seeded ]; then
  hf download $R sentbart/dolphin/once7/state_latest.pt sentbart/dolphin/once7/model_best_partial.pt sentbart/dolphin/once7/train_partial.log --local-dir /root/hfdl >/dev/null 2>&1
  cp /root/hfdl/sentbart/dolphin/once7/state_latest.pt $O/state_e2e.pt; cp /root/hfdl/sentbart/dolphin/once7/model_best_partial.pt $O/model_best.pt; cp /root/hfdl/sentbart/dolphin/once7/train_partial.log $O/train.log; touch /root/.bart7b_seeded
fi
( while sleep 7200; do f=$O/state_e2e.pt; [ -s $f ] && cp $f /root/once7_up.pt && hf upload $R /root/once7_up.pt sentbart/dolphin/once7/state_latest.pt >/dev/null 2>&1 && echo "[bart7] state on the hub $(date -u +%H:%M)"; rm -f /root/once7_up.pt; done ) &
for try in $(seq 1 40); do
  until [ "$(freemb)" -ge 10000 ]; do sleep 120; done
  echo "[bart7] the seventh run resumes from step $(python3 -c "import torch; print(torch.load('$O/state_e2e.pt', map_location='cpu', weights_only=False).get('step'))") (try $try) $(date -u +%H:%M)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -u /root/sb/train_e2e.py --vec $DB --text $DB --eval-text $DB/docs_001.jsonl --out $O --init /root/sb/dolphin_once6/state_e2e.pt --layers 16 --enc-layers 0 \
    --buckets 64,128,192,256,384,512,768 --bucket-order random --sent-budget 1900 --plan-accum 2 --max-bs 64 --eval-seq 128 --page-queue 16 --pool 0 --anchor-q 0 --select dec_top1 --patience 0 \
    --eval-every 1000 --save-every 1000 > /root/bart7_try.log 2>&1
  grep -E "^\[e2e\] --buckets|^\[eval|^\[best|^\[val\]|TRAIN_DONE|Error|Traceback|out of memory" /root/bart7_try.log | tail -n 6 | cut -c1-300
  grep -q "TRAIN_DONE" /root/bart7_try.log && break
  echo "[bart7] stopped without finishing - again in 10 min $(date -u +%H:%M)"; sleep 600
done
grep -q "TRAIN_DONE" /root/bart7_try.log || { echo "BART7_JOB_DONE gave up $(date -u)"; exit 1; }
hf upload $R $O/model_best.pt sentbart/dolphin/once7/model_best.pt >/dev/null 2>&1; hf upload $R $O/train.log sentbart/dolphin/once7/train.log >/dev/null 2>&1; hf upload $R $O/state_e2e.pt sentbart/dolphin/once7/state_end.pt >/dev/null 2>&1; echo "[bart7] the end on the hub"
hf download $R sentbart/dolphin/once4/model_best.pt --local-dir /root/hfdl >/dev/null 2>&1
CK=""; for spec in run4_best=/root/hfdl/sentbart/dolphin/once4/model_best.pt run6_end=/root/sb/dolphin_once6/state_e2e.pt run7_best=$O/model_best.pt run7_end=$O/state_e2e.pt; do [ -s "${spec#*=}" ] && CK="$CK $spec"; done
python3 -u /root/sb/sb_eval.py --docs $DF7/docs_001.jsonl --ckpt $CK --bands 128,256,512,768 --out /root/sb_eval768b.json 2>&1 | grep --line-buffered -vE "Warning|warn|documents \(" | sed 's/^/[bart7 dolphin] /' | cut -c1-300
python3 -u /root/sb/sb_eval.py --docs $AH/docs_005.jsonl --ckpt $CK --bands 128,256,512,768 --out /root/sb_eval_amb.json 2>&1 | grep --line-buffered -vE "Warning|warn|documents \(" | sed 's/^/[bart7 am] /' | cut -c1-300
python3 - <<'PC'
import json
a = json.load(open("/root/sb_eval768b.json"))["runs"]; b = json.load(open("/root/sb_eval_amb.json"))["runs"]
print("[bart7 composite] top-1 at 3223+1 candidates, all hidden positions: dolphin | AM | mean")
for k in a:
    if k in b: print(f"  {k:10s} {a[k]['all']['top1']:.3f} | {b[k]['all']['top1']:.3f} | {(a[k]['all']['top1'] + b[k]['all']['top1']) / 2:.3f}")
PC
hf upload $R /root/sb_eval768b.json sentbart/dolphin/sb_eval/sb_eval768b.json >/dev/null 2>&1; hf upload $R /root/sb_eval_amb.json sentbart/dolphin/sb_eval/sb_eval_amb.json >/dev/null 2>&1
echo "BART7_JOB_DONE $(date -u)"
B7
  setsid nohup bash -c 'bash /root/bart7bkeep.sh 2>&1 | tee -a /root/bart7.log' > /dev/null 2>&1 < /dev/null &
  echo "BART7B_LAUNCHED $(date -u)"
fi
if [ -e /root/.bart7b ] && [ -s /root/bart7bkeep.sh ] && ! pgrep -f "bart7bkee[p].sh" >/dev/null && ! grep -q "BART7_JOB_DONE" /root/bart7.log 2>/dev/null; then
  setsid nohup bash -c 'bash /root/bart7bkeep.sh 2>&1 | tee -a /root/bart7.log' > /dev/null 2>&1 < /dev/null &
fi
# 2026-10-15, the user: is the concept itself in - the judgment model's output as the reward, its continuous value making the
# gradient? Not yet: the rounds so far add it (x0.5) to correct replies only, on top of the right/wrong reward. rm8 is the
# concept as stated (--rm-pure: every finished reply's reward is the reward model's 0-1 score, right or wrong), on mtg10's
# questions like rm2c, and goes next in the queue.
if [ ! -e /root/.rm8 ] && grep -q "rm_pure" /root/work/online_loop.py; then touch /root/.rm8
  cat > /root/rm_queue.txt <<'QQ'
# tag items [extra flags]  - each round from 2.1.0, 60 steps
rm2c /root/work/mtg10_items.jsonl --qbatch 3
rm8 /root/work/mtg10_items.jsonl --qbatch 3 --rm-pure 1 --w-rm 1.0
rm5 /root/work/mtg10_items.jsonl --qbatch 3 --lr 2.5e-6 --search-lr 2.5e-6
rm3b /root/work/mtg10_items.jsonl --qbatch 3 --w-rm 1.0
rm7 /root/work/mix10_items.jsonl --qbatch 3
rm6 /root/work/mtg9_items.jsonl --qbatch 3 --w-rm 1.0 --lr 2.5e-6 --search-lr 2.5e-6
QQ
  echo "RM8_QUEUED $(date -u)"
fi
# the user's form is score x correct (1/0), not the score alone: rm9 (--rm-mult) goes first, rm8 (the score alone) later
if [ ! -e /root/.rm9 ] && grep -q "rm_mult" /root/work/online_loop.py; then touch /root/.rm9
  cat > /root/rm_queue.txt <<'QQ'
# tag items [extra flags]  - each round from 2.1.0, 60 steps
rm2c /root/work/mtg10_items.jsonl --qbatch 3
rm9 /root/work/mtg10_items.jsonl --qbatch 3 --rm-mult 1 --w-rm 1.0
rm5 /root/work/mtg10_items.jsonl --qbatch 3 --lr 2.5e-6 --search-lr 2.5e-6
rm7 /root/work/mix10_items.jsonl --qbatch 3
rm8 /root/work/mtg10_items.jsonl --qbatch 3 --rm-pure 1 --w-rm 1.0
rm3b /root/work/mtg10_items.jsonl --qbatch 3 --w-rm 1.0
rm6 /root/work/mtg9_items.jsonl --qbatch 3 --w-rm 1.0 --lr 2.5e-6 --search-lr 2.5e-6
QQ
  echo "RM9_QUEUED $(date -u)"
fi
# ---- 2026-10-15, the user: the point is how good the replies read, so at each milestone look at that. The training reward
# model's own reading is no yardstick; each round's best copy is set against 2.1.0 on the same questions by a reward model
# not used in training (Skywork-Reward-V2 on Llama-3.1-8B, 4 bits): win rate where both are right, lengths; the pairs go
# to the hub for a reader. ----
if ! pgrep -f "pairkee[p].sh" >/dev/null; then
  cat > /root/pairkeep.sh <<'PK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
J=Skywork/Skywork-Reward-V2-Llama-3.1-8B@4bit
freemb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -dc 0-9; }
lines() { wc -l < "$1" 2>/dev/null || echo 0; }
while :; do
  bests=$( { grep -ho "best copy by shard 0: rm1_s[0-9]*" /root/rmround.log 2>/dev/null | awk '{print $NF}'; grep -ho "full screens for rm[0-9a-z]*_s[0-9]*" /root/rm_rounds.log 2>/dev/null | awk '{print $NF}'; } | sort -u )
  for B in $bests; do
    for kind in st0 br3 ch3; do
      case $kind in st0) fc=/root/work/${B}st_0.jsonl; fs=/root/work/mtg7_s40st_0.jsonl; need=100;; br3) fc=/root/work/br3_$B.jsonl; fs=/root/work/br3_mtg7_s40.jsonl; need=150;; ch3) fc=/root/work/ch3_$B.jsonl; fs=/root/work/ch3_mtg7_s40.jsonl; need=170;; esac
      [ -e /root/.pair_${B}_$kind ] && continue; [ "$(lines $fc)" -ge $need ] || continue
      pgrep -f "\[(br3|ch3)-$B\]|\[${B}st" >/dev/null && continue
      until [ "$(freemb)" -ge 7500 ]; do sleep 60; done
      PYTHONPATH=/root/rmenv python3 /root/work/rm_pair.py $J $fc $fs ${B}_$kind /root/work/pairs_${B}_$kind.jsonl 2>/dev/null | grep "^PAIR" | tee -a /root/rm_pairs.txt
      [ -s /root/work/pairs_${B}_$kind.jsonl ] && { touch /root/.pair_${B}_$kind; hf upload $R /root/work/pairs_${B}_$kind.jsonl pooler_distill/chatsft/rmgrpo/pairs/${B}_$kind.jsonl >/dev/null 2>&1; }
      hf upload $R /root/rm_pairs.txt pooler_distill/chatsft/rmgrpo/rm_pairs.txt >/dev/null 2>&1
    done
  done
  sleep 300
done
PK
  setsid nohup bash -c 'bash /root/pairkeep.sh 2>&1 | tee -a /root/pairkeep.log' > /dev/null 2>&1 < /dev/null &
  echo "PAIRKEEP_LAUNCHED $(date -u)"
fi
echo "BOXRW2_OK serial $BOXRW2_SERIAL $(date -u)"
# CTL-END
