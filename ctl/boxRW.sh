#!/bin/bash
# box RW (2026-10-14, the user: rent a second box, as cheap as possible, and try GRPO with a reward model's 0-1 score as
# the reward). First step: can a dedicated reward model tell Deep Charger's better replies from its worse ones? The
# Skywork-Reward-V2 (Qwen3) family scores the archived GRPO rollouts of mtg7-mtg10 (rm_scan.py). Pull-based as the others.
BOXRW_SERIAL=2
if [ -f /root/.boxrw_serial ] && [ "$(cat /root/.boxrw_serial)" -gt "$BOXRW_SERIAL" ] 2>/dev/null; then echo "BOXRW_STALE $BOXRW_SERIAL"; exit 0; fi
echo $BOXRW_SERIAL > /root/.boxrw_serial
mkdir -p /root/work /root/hfdl; cd /root/work
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
for f in rm_scan.py rm_server.py rm_quality.py online_loop.py pool_eval.py web_search.py; do
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
if ! pgrep -f "rwmirro[r].sh" >/dev/null; then
  cat > /root/rwmirror.sh <<'MR'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
while :; do
  { echo "=== boxRW $(date -u) ==="; echo "--- ctl.log ---"; tail -n 30 /root/ctl.log 2>/dev/null
    for f in /root/rm_scan.log /root/grpoboot.log /root/rmround.log; do [ -s $f ] && { echo "--- $f ---"; grep -vE "Warning|warn|Loading weights" $f | tail -n 30 | cut -c1-600; }; done
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
    echo "--- gpu ---"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
    echo "--- disk ---"; df -h /root | tail -1
    echo "--- processes ---"; pgrep -fa "python3|kee[p].sh" | cut -c1-200; } > /root/boxlog.txt 2>&1
  hf upload $R /root/boxlog.txt pooler_distill/chatsft/audit/boxlog_RW.txt > /root/mirror_up.txt 2>&1 || echo "MIRROR_UP_FAIL $(date -u)" > /proc/1/fd/1
  sleep 600
done
MR
  setsid nohup bash /root/rwmirror.sh > /dev/null 2>&1 < /dev/null &
  echo "RWMIRROR_LAUNCHED $(date -u)"
fi

# ---- step 1: the reward models on the archived rollouts ----
if [ ! -e /root/.rmscan ]; then touch /root/.rmscan
  cat > /root/rmscankeep.sh <<'RS'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
until [ -f /root/.bootstrapped ]; do sleep 30; done
A=pooler_distill/chatsft/archive_G2/online
for r in mtg7 mtg8 mtg9 mtg10; do
  for try in 1 2 3 4; do hf download $R $A/online_$r/rollouts.jsonl --local-dir /root/hfdl >/dev/null 2>&1 && break; sleep 30; done
done
FS=$(ls /root/hfdl/$A/online_*/rollouts.jsonl); echo "[rm_scan] rollouts: $(cat $FS | wc -l) lines in $(echo $FS | wc -w) files"
python3 -u /root/work/rm_scan.py --rollouts $FS --models Skywork/Skywork-Reward-V2-Qwen3-0.6B Skywork/Skywork-Reward-V2-Qwen3-1.7B Skywork/Skywork-Reward-V2-Qwen3-4B --out /root/rm_scan.json --batch 8
hf upload $R /root/rm_scan.json pooler_distill/chatsft/rm_scan/rm_scan.json >/dev/null 2>&1 && echo "[rm_scan] on the hub: pooler_distill/chatsft/rm_scan/rm_scan.json"
echo "RMSCAN_JOB_DONE $(date -u)"
RS
  setsid nohup bash -c 'bash /root/rmscankeep.sh 2>&1 | tee -a /root/rm_scan.log' > /dev/null 2>&1 < /dev/null &
  echo "RMSCAN_LAUNCHED $(date -u)"
fi
if [ -e /root/.rmscan ] && [ -s /root/rmscankeep.sh ] && ! pgrep -f "rmscankee[p].sh" >/dev/null && ! grep -q "RMSCAN_JOB_DONE" /root/rm_scan.log 2>/dev/null; then
  setsid nohup bash -c 'bash /root/rmscankeep.sh 2>&1 | tee -a /root/rm_scan.log' > /dev/null 2>&1 < /dev/null &
fi

# ---- step 2: the GRPO environment (box G2's, minus the teachers: this box has no judge keys), after the scan.
# The loop keeps its pinned transformers; the reward model gets its own library path (/root/rmenv). ----
if [ ! -e /root/.grpoboot ]; then touch /root/.grpoboot
  cat > /root/grpobootkeep.sh <<'GB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
until grep -q "RMSCAN_JOB_DONE" /root/rm_scan.log 2>/dev/null; do sleep 60; done
echo "[grpoboot] start $(date -u +%H:%M)"
pip install -q --target /root/rmenv "transformers>=4.51,<4.57" 2>&1 | grep -v WARNING | tail -1
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
for k in grpoboot rmround; do
  if [ -e /root/.$k ] && [ -s /root/${k}keep.sh ] && ! pgrep -f "${k}kee[p].sh" >/dev/null && ! grep -qE "GRPOBOOT_DONE|GRPOBOOT_FAILED|RMROUND_JOB_DONE" /root/$k.log 2>/dev/null; then
    setsid nohup bash -c "bash /root/${k}keep.sh 2>&1 | tee -a /root/$k.log" > /dev/null 2>&1 < /dev/null &
  fi
done
echo "BOXRW_OK serial $BOXRW_SERIAL $(date -u)"
# CTL-END
