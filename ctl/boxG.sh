# box E (24GB, replaces the A4000 whose host had no free GPU left): measure the held-out set through
# the 4-bit grid the phone actually runs.
#
# The app loads an MLX affine 4-bit conversion, group 64. Its weight index says what that touches:
# the seven projections of every block plus embed_tokens and lm_head, scales and biases in fp16,
# and nothing else - the norms, the attention biases and the pooler stay in float. q4.py reproduces
# that grid in torch and pool_eval.py --q4 1 applies it after loading.
#
# The comparison is against the 39.7% the same evaluator produced on the same 150 questions, whose
# per-rollout output the onstart restores as ev_out_*.jsonl, so the difference can be paired over
# questions instead of being read off two independent means.
cd /root/work
# The token arrives as an environment variable on the instance; keep a copy so anything this
# script starts later still has it, and so the onstart can stay as short as possible.
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
# ---- one-time bootstrap -------------------------------------------------------------------------
# This used to live in the instance's onstart. Four hosts in a row refused to create a container with
# it there, and the only thing that had changed was its content, so it moved here: the onstart is now
# a handful of lines that fetch this file, and everything the box needs is in git where it can be
# fixed without renting a new machine.
if [ ! -f /root/.bootstrapped ]; then
  # one bootstrap at a time: a re-run while this is still downloading must not start a second copy
  mkdir /root/.bootstrap_lock 2>/dev/null || { echo "bootstrap already running - not starting another"; exit 0; }
  echo "=== bootstrap $(date -u) ==="
  mkdir -p /root/work/fft_out /root/work/runtime /root/work/dwq_calib /root/hfdl
  touch /root/work/runtime/__init__.py
  pip install -q "transformers==4.44.2" "peft==0.12.0" "safetensors==0.8.0" \
    "huggingface_hub>=0.34,<1.0" accelerate certifi datasets scipy 2>&1 | tail -1
  R=baya1116/hypernet-sp-distill; S=pooler_distill/grpo_pool3_step200; Q=pooler_distill/grpo_pool3_step200_q4
  for inc in "box_recover/scripts/*" "fft_out/pooler.pt" \
             "grpo_assets/mus_run/eval_step200/eval_heldout_300q_g2.jsonl" \
             "pooler_distill/pool_eval_cache.jsonl" "pooler_distill/dwq_calib/*" \
             "$S/*" "$Q/rollouts/*"; do
    for try in 1 2 3 4 5 6; do hf download $R --include "$inc" --local-dir /root/hfdl 2>&1 | tail -1 && break; sleep 20; done
    echo "dl $inc $(date -u +%H:%M)"
  done
  SC=/root/hfdl/box_recover/scripts; cp $SC/*.py $SC/*.sh /root/work/ 2>/dev/null
  ln -sfn /root/hfdl/fft_out/pooler.pt /root/work/fft_out/pooler.pt
  cp /root/hfdl/grpo_assets/mus_run/eval_step200/eval_heldout_300q_g2.jsonl /root/work/eval300.jsonl
  cp /root/hfdl/pooler_distill/pool_eval_cache.jsonl /root/work/pool_eval_cache.jsonl
  cp /root/hfdl/pooler_distill/dwq_calib/train.jsonl /root/work/dwq_calib/train.jsonl
  ln -sfn /root/hfdl/$S/model /root/eval_hf200
  ln -sfn /root/hfdl/$S/pooler.safetensors /root/pooler200.safetensors
  for i in 0 1 2; do cp /root/hfdl/$S/eval_shard$i.jsonl /root/work/ev_out_$i.jsonl; done
  # every arm measured on the previous box, so comparisons stay paired and a cut-short arm resumes
  for a in q4 qa d3 g32 j1 p_embfloat p_t06q4 p_t06bf16 s1; do
    for i in 0 1 2; do
      f=/root/hfdl/$Q/rollouts/${a}_$i.jsonl
      [ -s $f ] && cp $f /root/work/${a}_out_$i.jsonl
    done
  done
  python3 - <<'PYB'
import json
qs=[l for l in open("/root/work/eval300.jsonl") if l.strip()][:300]
for i in range(3): open(f"/root/work/ev_{i}.jsonl","w").writelines(qs[i::3])
print(f"[shard] first 300 lines, {len({json.loads(l).get('q','') for l in qs})} distinct questions")
PYB
  echo "model $(ls -lL /root/eval_hf200/model.safetensors 2>/dev/null | awk '{print $5}') bytes | " \
       "bf16 $(cat /root/work/ev_out_*.jsonl 2>/dev/null | wc -l) rollouts | " \
       "plain-4bit $(cat /root/work/q4_out_*.jsonl 2>/dev/null | wc -l) restored | " \
       "calib $(wc -l < /root/work/dwq_calib/train.jsonl 2>/dev/null)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /root | tail -1
  [ -s /root/eval_hf200/model.safetensors ] && touch /root/.bootstrapped && echo "BOOTSTRAP_DONE $(date -u)"
  rmdir /root/.bootstrap_lock 2>/dev/null
fi
[ -f /root/.bootstrapped ] || { echo "bootstrap incomplete - stopping here"; exit 0; }

DRUN=d4
DPOL=1
DFOCUS=1
PRUN=p_embfloat
PSKIP=embed_tokens
JRUN=j2
JLRQ=0
JCLIP=0
PRUNS="p_t06q4:1:0.6 p_t06bf16:0:0.6"
PG=64
PSKIP=
# The raw GitHub copy this box fetches can lag and hand a control run an OLDER version of this file (04:48 on 09-22 it
# relaunched the online loop under the previous mode while the newer run was merging a checkpoint on the same card).
# Every edit bumps BOXG_SERIAL; a run that sees a lower serial than one already executed stops here.
BOXG_SERIAL=2026100620
if [ -f /root/.boxg_serial ] && [ "$(cat /root/.boxg_serial)" -gt "$BOXG_SERIAL" ] 2>/dev/null; then echo "BOXG_STALE $BOXG_SERIAL < $(cat /root/.boxg_serial)"; exit 0; fi
echo $BOXG_SERIAL > /root/.boxg_serial
MODE=idle         # 2026-09-29: the side jobs hold the card (multi-turn measurement and training); a finished reeval re-launched on every control run and killed their evaluators
ORUN=g14          # 2026-09-26: distillation. The R1 thinking and answer of dolphin_v1 (minus the held-out hundred) as plain SFT, 8 records a step,
                  # one verified search trace at half weight beside them; no rollouts, no judge. Length is allowed to grow (up to ~1000 tokens is
                  # fine by the user); the yardsticks are the Dolphin held-out (85%) and the search held-out (40%) at 400 and at the epoch's end.
OSEED=            # 2026-09-20: g7 starts clean from s4_hf. g6 never carried g5's weights (its seed download left no checkpoint) and after the
                  # layer change ran with layers 0-19 of the stock model; both are now caught at launch (ONLINE_SEED_ABORT, ONLINE_ABORT)
OMODEL=g10m_hf    # 2026-09-25: the search-RL result (g10 step 400, held-out 43.1%) merged into s4_hf: the base the reasoning line starts from
OMERGEFROM=g10_step400
OMERGEBASE=s4_hf
OMERGELAYERS=20-27
OB=8
ODRATIO=1
ODMIN=0           # 2026-09-24: the Dolphin SFT record off. Its references think ~245 words against the model's ~110, a length pressure of its own; the KL anchor now does the retention
ODOLPHIN=dolphin_v2.jsonl
OLR=5e-5          # 2026-09-26: distillation rate for an all-layer r16 LoRA
OACCUM=1          # 2026-09-20: two optimizers now, one step each
OLAYERS=all       # 2026-09-25: full capacity for the reasoning side
OREPLAY=replay_v1.jsonl
OREPLAYFILTER=1
OTRAINALL=1
OQUEUE=1
OCOMPLETE=1
OGEN=3000         # 2026-09-25: 7000 made a step 7-8 min (the batch waits for its longest row); a sample past 3000 tokens rarely passes and RFT only trains on passes
OBUDGET=600
OGUARD=1
OGUARDSTEPS=20    # 2026-09-21: 20 search steps per window (240 rollouts); 10 tripped twice on hard stretches with nothing drifting
OGUARDHALVE=0     # a rollback restores the weights only; the rates stay as configured
# the reasoning problems. dolphin_v2 is the short-thinking subset (reference thinking 245 words at the
# median, 300 at the most); dolphin_v1 is the whole set, 6776 problems whose references think 718 words
# at the median and 1384 at the ninth decile. Move to v1 when v2 stops teaching anything, which reads as:
# the constrained-format problems stop scoring zero across all twelve, or the mean reward stops moving
# over a hundred steps. Changing this line and adding a restart marker is the whole switch.
OREASON=dolphin_v2.jsonl
OREASONG=12
OSEARCHEVERY=1    # 2026-09-24: every step is a search step
OKL=0             # 2026-09-24: off, as in pool3
OREASONEVERY=1    # 2026-09-25: every step is a reasoning step
OREASONSRC=dolphin_rft
OREASONVERIFY=judge
ORFT=1
ORFTREPLAY=0.5
OWHEELMAX=0
OSFTONLY=8
ODOLPHINON=reason # the Dolphin SFT record only on reasoning steps
OWHEELS=1         # a problem none of the twelve pass: the R1 reference is trained on instead
OWTALK=0.5
OSEARCHDEMO=
OSEARCHWHEELS=0   # 2026-09-20: off again, as in the search GRPO; the layer limit is the fix, not a crutch
OPGNORM=mean      # 2026-09-24: pool3's normalisation
OADVSTD=1
OREASONSTUB=0
OJUDGEAPI=openai    # 2026-09-25: back to nano once the OpenAI credit was topped up (it had run out at g12 step 7; DeepSeek bridged it)
OJUDGEMODEL=gpt-5-nano
OMAXSRCH=7
OJUDGE=0
OREPLAYN=0
OROLLEVERY=1
OQFILE=           # 2026-09-20: empty; the search questions come from the corpus pool the search GRPO trained on (OPOOL3Q)
OPOOL3Q=1         # search questions = corpus q/gold pairs with a gold of at most 6 words, held-out removed, exactly as grpo_pool did
ORESUME=0
OSTEPS=835        # one epoch of the 6676 training records at 8 a step
OSEARCHLR=1e-5    # 2026-09-20: the search side steps its own Adam at the search GRPO's rate; --lr is the reasoning side's
OSEARCHTEMP=0.9   # 2026-09-24: pool3's temperature; the guard watches the zero-search rate
OPOOLORDER=pool3  # 2026-09-21: the search questions continue pool3's own sequence from its 201st question (step 162 of g7 = pool index 200)
OPOOLOFFSET=200   # a fresh run: its first search step is pool3's 201st question
OPOOLER=lora      # 2026-09-22: the pooler's rank-8 adapter trains again, as in the search GRPO (its params ride with the search optimizer)
OPOOLERRANK=8
OPOOLERLR=1e-5
OSEARCHGEN=1500   # and capped as it capped them; the reasoning side keeps OGEN
OTEMP=0.6
SHARDS=3
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
for f in pool_eval.py q4.py qat.py dwq.py poolerfit.py jointfit.py checkmlx.py packmlx.py dequant_state.py gptq.py sft_lora.py selfgen_gpu.py build_merged.py web_search.py grpo_pool.py online_loop.py mt_gen.py memfit.py mt_sim.py r1_traj.py nq_items.py; do
  for try in 1 2 3; do curl -sS -o /root/work/$f "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/work/$f && break; sleep 5; done
done
cp /root/work/web_search.py /root/work/runtime/web_search.py 2>/dev/null
mkdir -p /root/work/localsearch
for f in build_store.py store.py embed.py search.py pq.py ivf.py memcheck.py train_retriever.py export_onnx.py test_retriever.py train_ranker.py reward_prep.py terms.py build_terms.py; do
  for try in 1 2 3; do curl -sS -o /root/work/localsearch/$f "$RAW/localsearch/$f?nocache=$(date +%s)" && python3 -m py_compile /root/work/localsearch/$f && break; sleep 5; done
done
mkdir -p /root/work/localsearch/data   # the retriever's pairs; a missing file must not leave a 404 body behind
for f in train_pairs.json train_negs.json test_queries.json reward_force.json reward_questions.jsonl; do
  for try in 1 2 3; do curl -sSf -o /root/work/localsearch/data/$f "$RAW/localsearch/data/$f?nocache=$(date +%s)" && python3 -c "import json,sys; f='/root/work/localsearch/data/$f'; json.load(open(f)) if f.endswith('.json') else [json.loads(l) for l in open(f) if l.strip()]" && break; rm -f /root/work/localsearch/data/$f; sleep 5; done
done
echo "fetched: pool_eval $(wc -l < /root/work/pool_eval.py) lines, q4 $(wc -l < /root/work/q4.py) lines"

# The local search's corpus is a side job: it needs the CPU and the disk for its first hours and the card only
# briefly at the end, so it runs beside whatever mode is active. WIKI=1 starts it once; it resumes shard by shard.
WIKI=${WIKI:-1}
if [ "$WIKI" = 1 ] && ! pgrep -f "wikikee[p].sh" >/dev/null && ! grep -q WIKI_DONE /root/wiki.log 2>/dev/null; then
  cat > /root/wikikeep.sh <<'WK'
#!/bin/bash
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
ST=/root/wiki_store; mkdir -p $ST /root/wikidl
pip list 2>/dev/null | grep -qi zstandard || pip install -q zstandard pyarrow onnxruntime >/dev/null 2>&1
for i in $(seq -f %05g 0 40); do
  f=20231101.en/train-$i-of-00041.parquet
  grep -q "\"$f\"" $ST/meta.json 2>/dev/null && continue
  for try in 1 2 3 4 5; do hf download wikimedia/wikipedia --repo-type dataset --include "$f" --local-dir /root/wikidl >/dev/null 2>&1; [ -s /root/wikidl/$f ] && break; sleep 30; done
  [ -s /root/wikidl/$f ] || { echo "WIKI_ABORT shard $i download failed $(date -u)"; exit 0; }
  (cd /root/wikidl && python3 /root/work/localsearch/build_store.py --out $ST "$f" 2>&1 | grep "^\[store\]")
  rm -f /root/wikidl/$f
done
[ -s /root/bge-small/model.safetensors ] || for try in 1 2 3; do hf download BAAI/bge-small-en-v1.5 --local-dir /root/bge-small >/dev/null 2>&1 && break; sleep 30; done
n=$(python3 -c "import json;print(json.load(open('$ST/meta.json'))['n_docs'])")
until [ "$(( $(stat -c %s $ST/emb.bin 2>/dev/null || echo 0) / 48 ))" -ge "$n" ]; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/localsearch/embed.py --store $ST --model /root/bge-small --backend torch --batch 256 2>&1 | grep -E "EMBED_DONE|Error|error" | tail -2
  sleep 60
done
python3 -c "import sys; sys.path.insert(0,'/root/work/localsearch'); from search import build_title_index; print(build_title_index('$ST'))"
echo "[wiki] store $(du -sh $ST | cut -f1): $(ls -la $ST | awk 'NR>1{print $9"="$5}' | tr '\n' ' ')"
for try in 1 2 3; do hf upload $R $ST localsearch/wiki_en_20231101 >/dev/null 2>&1 && break; sleep 60; done
echo "WIKI_DONE $n articles $(date -u)"
WK
  chmod +x /root/wikikeep.sh
  setsid nohup bash -c 'bash /root/wikikeep.sh 2>&1 | tee -a /root/wiki.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "WIKI_LAUNCHED $(date -u)"
fi

# one-shot: the lexical index the evaluation built on the box goes to the hub beside the store; the title-only index it replaces comes down
if [ -s /root/wiki_store/lex.sqlite ] && [ ! -f /root/.lex_uploaded ]; then
  export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
  if hf upload baya1116/hypernet-sp-distill /root/wiki_store/lex.sqlite localsearch/wiki_en_20231101/lex.sqlite >/dev/null 2>&1; then
    touch /root/.lex_uploaded; echo "LEX_UPLOADED $(du -h /root/wiki_store/lex.sqlite | cut -f1) $(date -u)"
    python3 - <<'PYD' 2>/dev/null
from huggingface_hub import HfApi
HfApi().delete_file("localsearch/wiki_en_20231101/titles.sqlite", "baya1116/hypernet-sp-distill"); print("titles.sqlite removed from the hub")
PYD
  fi
fi

# The store with 6,000 characters an article (zstd 19): the same articles in the same order, so the index files of
# the 1,500-character store serve it (symlinked). WIKI6=1 builds it once beside whatever runs; it resumes shard by shard.
WIKI6=${WIKI6:-1}
if [ "$WIKI6" = 1 ] && ! pgrep -f "wiki6kee[p].sh" >/dev/null && ! grep -q WIKI6_DONE /root/wiki6.log 2>/dev/null && [ -s /root/wiki_store/emb_ivf.bin ]; then
  cat > /root/wiki6keep.sh <<'W6'
#!/bin/bash
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
ST=/root/wiki_store6; mkdir -p $ST /root/wikidl
for i in $(seq -f %05g 0 40); do
  f=20231101.en/train-$i-of-00041.parquet
  grep -q "\"$f\"" $ST/meta.json 2>/dev/null && continue
  [ "$(df -BG /root | awk 'NR==2{print $4+0}')" -ge 3 ] || { echo "WIKI6_ABORT disk $(df -h /root | awk 'NR==2{print $4}') free $(date -u)"; exit 0; }
  for try in 1 2 3 4 5; do hf download wikimedia/wikipedia --repo-type dataset --include "$f" --local-dir /root/wikidl >/dev/null 2>&1; [ -s /root/wikidl/$f ] && break; sleep 30; done
  [ -s /root/wikidl/$f ] || { echo "WIKI6_ABORT shard $i download failed $(date -u)"; exit 0; }
  (cd /root/wikidl && python3 /root/work/localsearch/build_store.py --out $ST --chars 6000 --level 19 "$f" 2>&1 | grep "^\[store\]")
  rm -f /root/wikidl/$f
done
n=$(python3 -c "import json;print(json.load(open('$ST/meta.json'))['n_docs'])"); n0=$(python3 -c "import json;print(json.load(open('/root/wiki_store/meta.json'))['n_docs'])")
[ "$n" = "$n0" ] && cmp -s $ST/titles.txt /root/wiki_store/titles.txt || { echo "WIKI6_ABORT $n articles against $n0, or the titles differ"; exit 0; }
for f in emb.bin emb_ivf.bin ivf_centroids.npy ivf_order.npy ivf_offsets.npy lex.sqlite idf.pkl; do [ -e $ST/$f ] || ln -s /root/wiki_store/$f $ST/$f; done
echo "[wiki6] store $(du -shL $ST/docs.bin | cut -f1) docs.bin"
for f in docs.bin blocks.idx meta.json; do for try in 1 2 3; do hf upload $R $ST/$f localsearch/wiki_en_20231101_6000/$f >/dev/null 2>&1 && break; sleep 60; done; done
echo "WIKI6_DONE $n articles $(date -u)"
W6
  chmod +x /root/wiki6keep.sh
  setsid nohup bash -c 'bash /root/wiki6keep.sh 2>&1 | tee -a /root/wiki6.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "WIKI6_LAUNCHED $(date -u)"
fi

# The whole-text term index (build_terms.py): two passes over the dump on the CPU, packed to varints, uploaded.
# TERMS=1 builds it once beside whatever runs (the 6,000-character store's job first: the dump is streamed twice more).
TERMS=${TERMS:-1}
# the first packer was a per-posting Python loop (hours) and wrote postings_vb.bin in place: stop it once and drop
# its partial output; the vectorised packer writes a .part file and renames it when complete
if [ ! -e /root/.terms_packfix ] && pgrep -f "localsearch/terms.py /root/wiki_store/terms" >/dev/null; then
  pkill -f "termskee[p].sh"; pkill -f "localsearch/terms.py /root/wiki_store/terms"; sleep 2
  rm -f /root/wiki_store/terms/postings_vb.bin; touch /root/.terms_packfix; echo "TERMS_PACKFIX restarted the packer $(date -u)"
fi
if [ "$TERMS" = 1 ] && ! pgrep -f "termskee[p].sh" >/dev/null && ! grep -q TERMS_JOB_DONE /root/terms.log 2>/dev/null && [ -s /root/wiki_store/titles.txt ] && grep -q WIKI6_DONE /root/wiki6.log 2>/dev/null; then
  cat > /root/termskeep.sh <<'TK'
#!/bin/bash
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; L=/root/work/localsearch; OUT=/root/wiki_store/terms
[ "$(df -BG /root | awk 'NR==2{print $4+0}')" -ge 6 ] || { echo "TERMS_ABORT disk $(df -h /root | awk 'NR==2{print $4}') free"; exit 0; }
mkdir -p $OUT
[ -s $OUT/postings.bin ] && [ -s $OUT/terms_meta.json ] || python3 $L/build_terms.py --out $OUT --store /root/wiki_store 2>&1 | grep -E "^\[terms\] (pass 1 file 40|pass 2 file 40|vocabulary|terms whose)|TERMS_DONE|Error|Traceback"
[ -s $OUT/terms_meta.json ] || { echo "TERMS_ABORT build"; exit 1; }
[ -s $OUT/postings_vb.bin ] || python3 $L/terms.py $OUT 2>&1 | grep -E "PACK_DONE|Error|Traceback"
[ -s $OUT/postings_vb.bin ] && rm -f $OUT/postings.bin
for f in terms.txt term_df.npy term_voffsets.npy postings_vb.bin terms_meta.json; do for try in 1 2 3; do hf upload $R $OUT/$f localsearch/wiki_en_20231101/terms/$f >/dev/null 2>&1 && break; sleep 60; done; done
echo "[terms] $(du -sh $OUT | cut -f1) on disk"
echo "TERMS_JOB_DONE $(date -u)"
TK
  chmod +x /root/termskeep.sh
  setsid nohup bash -c 'bash /root/termskeep.sh 2>&1 | tee -a /root/terms.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "TERMS_LAUNCHED $(date -u)"
fi

# One-shot cleanup before the multi-turn jobs relaunch (mt2 / mem2): the re-launched reeval's chat evaluator, the mt1
# cutter, and the mem1 training keeper (which saw mt1's premature MT_JOB_DONE and would start its traces).
if [ ! -e /root/.mt2_clean ]; then
  touch /root/.mt2_clean
  pkill -f "reevalkee[p].sh"; pkill -f "pool_eval.py .*chat_eval60"; pkill -f "none 25"
  pkill -f "mttkee[p].sh"; pkill -f "mt_gen.p[y]"; pkill -f "pool_eval.py .*--mt-save-tokens"; sleep 5
  echo "MT2_CLEAN $(date -u)"
fi

# Multi-turn: the dialogues (the teacher splits held-out questions into turns), then the 4-bit model on them under
# each history protocol (none / full / stream), Wikipedia's search, one mode at a time when the card is free.
# MT_TAG bumps to redo; MT_N dialogues per mode (the screen), MT_MODES which protocols.
MT=${MT:-1}; MT_TAG=${MT_TAG:-mt2}; MT_N=${MT_N:-30}; MT_MODES=${MT_MODES:-"full stream none"}
if [ "$MT" = 1 ] && ! pgrep -f "mtkee[p].sh" >/dev/null && ! grep -q "MT_JOB_DONE $MT_TAG" /root/mt.log 2>/dev/null && [ -s /root/gptq_hf_gq14/model.safetensors ]; then
  cat > /root/mtkeep.sh <<MK1
MTAG=$MT_TAG; MN=$MT_N; MODES="$MT_MODES"
MK1
  cat >> /root/mtkeep.sh <<'MK2'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
cd /root/work
[ -s /root/work/mt_eval.jsonl ] || python3 /root/work/mt_gen.py --seeds /root/work/eval300.jsonl --out /root/work/mt_eval.jsonl --n-bridge 40 --n-memory 30 --n-switch 20 2>&1 | grep -E "^\[mtgen\]|MTGEN_DONE|Error|Traceback"
[ -s /root/work/mt_eval.jsonl ] || { echo "MT_ABORT no dialogues"; exit 1; }
hf upload $R /root/work/mt_eval.jsonl pooler_distill/chatsft/multiturn/mt_eval.jsonl >/dev/null 2>&1
for mode in $MODES; do
  while pgrep -f "pool_eval.py .*--multiturn" >/dev/null || pgrep -f "reevalkee[p].sh" >/dev/null; do sleep 120; done
  OUT=/root/work/${MTAG}_${mode}.jsonl
  SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 /root/work/pool_eval.py /root/reeval_g14m_pooler.safetensors /root/work/eval300.jsonl $OUT --multiturn /root/work/mt_eval.jsonl --mt-mode $mode \
    --n $MN --rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600 --tag "[$MTAG-$mode]" > /root/${MTAG}_${mode}.log 2>&1
  grep -E "EVAL_DONE|Error|Traceback" /root/${MTAG}_${mode}.log | tail -2 | cut -c1-300
  hf upload $R $OUT pooler_distill/chatsft/multiturn/${MTAG}_${mode}.jsonl >/dev/null 2>&1
done
echo "MT_JOB_DONE $MTAG $(date -u)"
MK2
  chmod +x /root/mtkeep.sh
  setsid nohup bash -c 'bash /root/mtkeep.sh 2>&1 | tee -a /root/mt.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MT_LAUNCHED $MT_TAG $(date -u)"
fi

# The no-history mode (last) is cut at ~10 dialogues: without the history its answer is plain.
if [ ! -e /root/.mt2_cut ] && pgrep -f "mtkee[p].sh" >/dev/null; then
  touch /root/.mt2_cut
  setsid nohup bash -c 'until [ "$(wc -l < /root/work/mt2_none.jsonl 2>/dev/null || echo 0)" -ge 25 ] && pgrep -f "pool_eval.py .*--mt-mode none" >/dev/null; do
      pgrep -f "mtkee[p].sh" >/dev/null || exit 0; sleep 60; done
    pkill -f "pool_eval.py .*--mt-mode none"; echo "[mt2] none cut at $(wc -l < /root/work/mt2_none.jsonl) turns $(date -u +%H:%M)" >> /root/mt.log' > /dev/null 2>&1 < /dev/null &
  echo "MT2_CUT armed $(date -u)"
fi

if [ ! -e /root/.mem3_swap ]; then touch /root/.mem3_swap; pkill -f "mttkee[p].sh"; sleep 2; echo "MEM3_SWAP stopped the mem2 keeper $(date -u)"; fi
if [ ! -e /root/.mem4_swap ]; then touch /root/.mem4_swap; pkill -f "mttkee[p].sh"; sleep 2; echo "MEM4_SWAP stopped the mem3 keeper (its traces run carries on and is resumed) $(date -u)"; fi
# The teacher traces run at ~3.3 minutes a dialogue (full thinking per turn): 120 would end near 22:30. Stop them at
# ~80 dialogues (185 turns); the keeper then goes on to memfit with what is there (switches are also assembled from it).
if [ ! -e /root/.mem5_tracecut ] && pgrep -f "mttkee[p].sh" >/dev/null; then
  touch /root/.mem5_tracecut
  setsid nohup bash -c 'until [ "$(wc -l < /root/work/mtt_solo.jsonl 2>/dev/null || echo 0)" -ge 185 ]; do pgrep -f "mttkee[p].sh" >/dev/null || exit 0; sleep 120; done
    pkill -f "pool_eval.py .*--mt-save-tokens"; echo "[mtt] traces cut at $(wc -l < /root/work/mtt_solo.jsonl) turns $(date -u +%H:%M)" >> /root/mtt.log' > /dev/null 2>&1 < /dev/null &
  echo "MEM5_TRACECUT armed $(date -u)"
fi

# Multi-turn, switch first (2026-09-29, the user: no long-range recall needed; a natural rally, the topic switch is
# the bottleneck; keep existing ability). mem5: a switch-heavy held-out set (all 20 switch dialogues of mt_eval + 40
# more pairs of eval300 questions mt_eval did not use) measured before and after; training dialogues from TRAINING
# seeds; the model's solo runs of each turn's self-contained version (teacher trajectories); memfit --student full
# (history verbatim in the prompt, pooler + LoRA, KL to the model's own single-turn behaviour, half the examples
# assembled topic switches); after training: the switch set, the first 30 mt_eval dialogues (vs mt2 full), and the
# single-turn search held-out (102, vs gq14 48.0) as the regression check. MTT=0 cancels; MTT_TAG bumps to redo.
if [ ! -e /root/.mem5_swap ]; then touch /root/.mem5_swap; pkill -f "mttkee[p].sh"; pkill -f "mtmkee[p].sh"; sleep 2; echo "MEM5_SWAP $(date -u)"; fi
MTT=${MTT:-1}; MTT_TAG=${MTT_TAG:-mem5}; MTT_N=${MTT_N:-120}
if [ "$MTT" = 1 ] && ! pgrep -f "mttkee[p].sh" >/dev/null && ! grep -q "MTT_JOB_DONE $MTT_TAG" /root/mtt.log 2>/dev/null && [ -s /root/gptq_hf_gq14/model.safetensors ]; then
  cat > /root/mttkeep.sh <<TK1
TTAG=$MTT_TAG; TN=$MTT_N
TK1
  cat >> /root/mttkeep.sh <<'TK2'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
until grep -q "MT_JOB_DONE mt2" /root/mt.log 2>/dev/null; do sleep 300; done
[ -s /root/work/mt_eval_sw.jsonl ] || python3 - <<'PY'
import json, random
mt = [json.loads(l) for l in open("/root/work/mt_eval.jsonl")]
used = {t["q"] for d in mt for t in d["turns"]} | {d.get("seed", "") for d in mt}
ev = [json.loads(l) for l in open("/root/work/eval300.jsonl")]
free = [(r["q"], r["gold"]) for r in ev if r.get("q") and r.get("gold") and r["q"] not in used]
random.Random(11).shuffle(free)
out = [d for d in mt if d["kind"] == "switch"]
for k in range(min(40, len(free) // 2)):
    a, b = free[2 * k], free[2 * k + 1]
    out.append({"id": f"sw{k:03d}", "kind": "switch", "seed": a[0], "turns": [{"q": a[0], "gold": a[1], "standalone": a[0]}, {"q": b[0], "gold": b[1], "standalone": b[0]}]})
random.Random(12).shuffle(out)
with open("/root/work/mt_eval_sw.jsonl", "w") as fh:
    for d in out: fh.write(json.dumps(d, ensure_ascii=False) + "\n")
print(f"[mtt] switch held-out set: {len(out)} dialogues")
PY
hf upload $R /root/work/mt_eval_sw.jsonl pooler_distill/chatsft/multiturn/mt_eval_sw.jsonl >/dev/null 2>&1
grep -q standalone /root/work/mt_train.jsonl 2>/dev/null || python3 /root/work/mt_gen.py --seeds /root/work/selfq_all.jsonl --exclude /root/work/eval300.jsonl --out /root/work/mt_train.jsonl --n-bridge 80 --n-memory 40 --n-switch 30 --seed 7 2>&1 | grep -E "^\[mtgen\]|MTGEN_DONE|Error|Traceback"
grep -q standalone /root/work/mt_train.jsonl 2>/dev/null || { echo "MTT_ABORT no training dialogues"; exit 1; }
hf upload $R /root/work/mt_train.jsonl pooler_distill/chatsft/multiturn/mt_train.jsonl >/dev/null 2>&1
ev() {  # $1 ckpt $2 dialogues $3 out $4 mode $5 n $6 tag
  while pgrep -f "pool_eval.py" >/dev/null; do sleep 60; done
  env $ENV python3 /root/work/pool_eval.py $1 /root/work/eval300.jsonl $3 --multiturn $2 --mt-mode $4 --n $5 $EVARGS --tag "[$6]" > /root/$6.log 2>&1
  grep -E "EVAL_DONE|Error|Traceback" /root/$6.log | tail -2 | cut -c1-300
  hf upload $R $3 pooler_distill/chatsft/multiturn/$(basename $3) >/dev/null 2>&1
}
echo "[mtt] switch baseline start $(date -u +%H:%M)"
ev /root/reeval_g14m_pooler.safetensors /root/work/mt_eval_sw.jsonl /root/work/sw0_full.jsonl full 60 sw0-full
echo "[mtt] traces start $(date -u +%H:%M)"
while pgrep -f "pool_eval.py" >/dev/null; do sleep 60; done
env $ENV python3 /root/work/pool_eval.py /root/reeval_g14m_pooler.safetensors /root/work/eval300.jsonl /root/work/mtt_solo.jsonl --multiturn /root/work/mt_train.jsonl --mt-mode none --mt-standalone 1 --mt-save-tokens 1 \
  --n $TN $EVARGS --tag "[mtt-solo]" > /root/mtt_traces.log 2>&1
grep -E "EVAL_DONE|Error|Traceback" /root/mtt_traces.log | tail -2 | cut -c1-300
hf upload $R /root/work/mtt_solo.jsonl pooler_distill/chatsft/multiturn/mtt_solo.jsonl >/dev/null 2>&1
echo "[mtt] memfit start $(date -u +%H:%M)"
env SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /root/work/memfit.py --student full --ckpt /root/reeval_g14m_pooler.safetensors --data /root/work/mtt_solo.jsonl --out /root/pooler_$TTAG.safetensors --log /root/memfit_$TTAG.log --steps 600 --lr 1e-5 --lr-lora 1e-5 > /root/memfit_${TTAG}_run.log 2>&1
grep -E "^\[memfit\]|^\[data\]|^val |MEMFIT_DONE|Error|Traceback" /root/memfit_${TTAG}_run.log | tail -16 | cut -c1-200
[ -s /root/pooler_$TTAG.safetensors ] || { echo "MTT_ABORT memfit: $(tail -3 /root/memfit_${TTAG}_run.log | tr '\n' ' ' | cut -c1-300)"; exit 1; }
hf upload $R /root/pooler_$TTAG.safetensors pooler_distill/chatsft/multiturn/pooler_$TTAG.safetensors >/dev/null 2>&1
ev /root/pooler_$TTAG.safetensors /root/work/mt_eval_sw.jsonl /root/work/${TTAG}_sw_full.jsonl full 60 ${TTAG}-sw-full
ev /root/pooler_$TTAG.safetensors /root/work/mt_eval.jsonl /root/work/${TTAG}_full.jsonl full 30 ${TTAG}-full
# the regression check: single-turn search held-out, the same 102 questions as every number in the lineage
for i in 0 1 2; do
  while pgrep -f "pool_eval.py" >/dev/null; do sleep 60; done
  env $ENV python3 /root/work/pool_eval.py /root/pooler_$TTAG.safetensors /root/work/ev_$i.jsonl /root/work/${TTAG}st_out_$i.jsonl --n 34 $EVARGS --tag "[${TTAG}st$i]" > /root/${TTAG}st_$i.log 2>&1
  grep -E "EVAL_DONE|Error|Traceback" /root/${TTAG}st_$i.log | tail -1 | cut -c1-300
  hf upload $R /root/work/${TTAG}st_out_$i.jsonl pooler_distill/chatsft/rollouts/${TTAG}st_$i.jsonl >/dev/null 2>&1
done
echo "MTT_JOB_DONE $TTAG $(date -u)"
TK2
  chmod +x /root/mttkeep.sh
  setsid nohup bash -c 'bash /root/mttkeep.sh 2>&1 | tee -a /root/mtt.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MTT_LAUNCHED $MTT_TAG $(date -u)"
fi

# The mixed student, queued behind mem3 on the same traces: the last exchange verbatim in the prompt (names travel
# as text), older turns through the pooler; then the mix protocol on the held-out dialogues. MTM=0 cancels.
MTM=${MTM:-0}; MTM_TAG=${MTM_TAG:-mem3m}
if [ "$MTM" = 1 ] && ! pgrep -f "mtmkee[p].sh" >/dev/null && ! grep -q "MTM_JOB_DONE $MTM_TAG" /root/mtm.log 2>/dev/null && [ -s /root/gptq_hf_gq14/model.safetensors ]; then
  cat > /root/mtmkeep.sh <<MM1
MTAG=$MTM_TAG
MM1
  cat >> /root/mtmkeep.sh <<'MM2'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
until grep -qE "MTT_JOB_DONE|MTT_ABORT memfit" /root/mtt.log 2>/dev/null && [ -s /root/work/mtt_solo.jsonl ]; do sleep 300; done
while pgrep -f "pool_eval.py|memfit.py" >/dev/null; do sleep 120; done
echo "[mtm] memfit (mix) start $(date -u +%H:%M)"
SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /root/work/memfit.py --student mix --ckpt /root/reeval_g14m_pooler.safetensors --data /root/work/mtt_solo.jsonl --out /root/pooler_$MTAG.safetensors --log /root/memfit_$MTAG.log --steps 600 --lr 1e-5 --lr-lora 1e-5 > /root/memfit_${MTAG}_run.log 2>&1
grep -E "^\[memfit\]|^\[data\]|^val |MEMFIT_DONE|Error|Traceback" /root/memfit_${MTAG}_run.log | tail -16 | cut -c1-200
[ -s /root/pooler_$MTAG.safetensors ] || { echo "MTM_ABORT memfit: $(tail -3 /root/memfit_${MTAG}_run.log | tr '\n' ' ' | cut -c1-300)"; exit 1; }
hf upload $R /root/pooler_$MTAG.safetensors pooler_distill/chatsft/multiturn/pooler_$MTAG.safetensors >/dev/null 2>&1
SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /root/work/pool_eval.py /root/pooler_$MTAG.safetensors /root/work/eval300.jsonl /root/work/${MTAG}_mix.jsonl --multiturn /root/work/mt_eval.jsonl --mt-mode mix \
  --n 30 --rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600 --tag "[$MTAG-mix]" > /root/${MTAG}_mix.log 2>&1
grep -E "EVAL_DONE|Error|Traceback" /root/${MTAG}_mix.log | tail -2 | cut -c1-300
hf upload $R /root/work/${MTAG}_mix.jsonl pooler_distill/chatsft/multiturn/${MTAG}_mix.jsonl >/dev/null 2>&1
echo "MTM_JOB_DONE $MTAG $(date -u)"
MM2
  chmod +x /root/mtmkeep.sh
  setsid nohup bash -c 'bash /root/mtmkeep.sh 2>&1 | tee -a /root/mtm.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MTM_LAUNCHED $MTM_TAG $(date -u)"
fi

# The fluency stage's data, smoke first (API only, no card): 6 simulated conversations, nano as the user, R1 as the
# reference assistant, uploaded for reading. MTSIM_SERIAL bumps to redo.
MTSIM_SERIAL=1
if [ "$(cat /root/.mtsim_serial 2>/dev/null)" != "$MTSIM_SERIAL" ] && [ -s /root/.oai ] && [ -s /root/.dsk ]; then
  echo "$MTSIM_SERIAL" > /root/.mtsim_serial
  ( cd /root/work && python3 /root/work/mt_sim.py --seeds /root/work/selfq_all.jsonl --exclude /root/work/eval300.jsonl --out /root/work/mt_sim_smoke.jsonl --n 6 --workers 6 2>&1 | grep -E "MTSIM_DONE|Error|Traceback" | tail -3 | sed 's/^/[mtsim] /'
    export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); hf upload baya1116/hypernet-sp-distill /root/work/mt_sim_smoke.jsonl pooler_distill/chatsft/multiturn/mt_sim_smoke.jsonl >/dev/null 2>&1
    echo "[mtsim] smoke uploaded $(date -u +%H:%M)" ) >> /root/mtsim.log 2>&1 &
  echo "MTSIM_SMOKE_LAUNCHED $(date -u)"
fi

# mem6, queued behind mem5 on the same teacher traces: mem5 moved the switch turn 17 -> 26 of 60 but lost the first
# turns 41 -> 32 (searches per turn 3.1 -> 5.6): the LoRA drifted where it was never held. mem6 adds single-turn anchors
# (no history, the student's prompt = the teacher's, as many as the real examples), trains fewer steps at half the
# LoRA rate, and is measured the same way. MEM6=0 cancels.
MEM6=${MEM6:-1}
if [ "$MEM6" = 1 ] && ! pgrep -f "mem6kee[p].sh" >/dev/null && ! grep -q "MEM6_JOB_DONE" /root/mem6.log 2>/dev/null; then
  cat > /root/mem6keep.sh <<'M6'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; T=mem6
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
until grep -qE "MTT_JOB_DONE mem5|MTT_ABORT" /root/mtt.log 2>/dev/null; do sleep 300; done
while pgrep -f "pool_eval.py|memfit.py" >/dev/null; do sleep 60; done
echo "[mem6] memfit start $(date -u +%H:%M)"
env SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /root/work/memfit.py --student full --anchor -1 --ckpt /root/reeval_g14m_pooler.safetensors --data /root/work/mtt_solo.jsonl --out /root/pooler_$T.safetensors --log /root/memfit_$T.log --steps 300 --val-every 50 --lr 5e-6 --lr-lora 5e-6 > /root/memfit_${T}_run.log 2>&1
grep -E "^\[memfit\]|^\[data\]|^val |MEMFIT_DONE|Error|Traceback" /root/memfit_${T}_run.log | tail -14 | cut -c1-200
[ -s /root/pooler_$T.safetensors ] || { echo "MEM6_ABORT memfit: $(tail -3 /root/memfit_${T}_run.log | tr '\n' ' ' | cut -c1-300)"; exit 1; }
hf upload $R /root/pooler_$T.safetensors pooler_distill/chatsft/multiturn/pooler_$T.safetensors >/dev/null 2>&1
ev() { while pgrep -f "pool_eval.py" >/dev/null; do sleep 60; done
  env $ENV python3 /root/work/pool_eval.py $1 /root/work/eval300.jsonl $3 --multiturn $2 --mt-mode $4 --n $5 $EVARGS --tag "[$6]" > /root/$6.log 2>&1
  grep -E "EVAL_DONE|Error|Traceback" /root/$6.log | tail -2 | cut -c1-300; hf upload $R $3 pooler_distill/chatsft/multiturn/$(basename $3) >/dev/null 2>&1; }
ev /root/pooler_$T.safetensors /root/work/mt_eval_sw.jsonl /root/work/${T}_sw_full.jsonl full 60 ${T}-sw-full
for i in 0 1 2; do while pgrep -f "pool_eval.py" >/dev/null; do sleep 60; done
  env $ENV python3 /root/work/pool_eval.py /root/pooler_$T.safetensors /root/work/ev_$i.jsonl /root/work/${T}st_out_$i.jsonl --n 34 $EVARGS --tag "[${T}st$i]" > /root/${T}st_$i.log 2>&1
  grep -E "EVAL_DONE|Error|Traceback" /root/${T}st_$i.log | tail -1 | cut -c1-300; hf upload $R /root/work/${T}st_out_$i.jsonl pooler_distill/chatsft/rollouts/${T}st_$i.jsonl >/dev/null 2>&1; done
ev /root/pooler_$T.safetensors /root/work/mt_eval.jsonl /root/work/${T}_full.jsonl full 30 ${T}-full
echo "MEM6_JOB_DONE $(date -u)"
M6
  chmod +x /root/mem6keep.sh
  setsid nohup bash -c 'bash /root/mem6keep.sh 2>&1 | tee -a /root/mem6.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MEM6_LAUNCHED $(date -u)"
fi

if [ ! -e /root/.mem7b_swap ]; then touch /root/.mem7b_swap
  pkill -f "mem6kee[p].sh"; pkill -f "pool_eval.py /root/pooler_mem6"; pkill -f "mem7kee[p].sh"; sleep 3
  echo "MEM6_JOB_DONE (stopped after the switch set: first turns 41 -> 27) $(date -u)" >> /root/mem6.log; echo "MEM7B_SWAP $(date -u)"; fi
# mem7, rejection-sampled self-training from the BASE model, pooler frozen (the user's suggestion: train only on the model's own successes,
# so it finds the way to keep both): mem6 runs 100 training switch pairs + 40 training dialogues with the history in the
# prompt (tokens saved); every turn it got right - first turns (no history) and later turns alike - is trained on as
# its own target (memfit --objective ce), from mem6's LoRA and pooler; measured the same way. MEM7=0 cancels.
MEM7=${MEM7:-1}
if [ "$MEM7" = 1 ] && ! pgrep -f "mem7kee[p].sh" >/dev/null && ! grep -q "MEM7_JOB_DONE" /root/mem7.log 2>/dev/null && [ -e /root/.mem7b_swap ]; then
  cat > /root/mem7keep.sh <<'M7'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; T=mem7; B=/root/reeval_g14m_pooler.safetensors
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
until grep -qE "MEM6_JOB_DONE|MEM6_ABORT" /root/mem6.log 2>/dev/null; do sleep 300; done
[ -s /root/work/mt_train_sw.jsonl ] || python3 - <<'PY'
import json, random
used = set()
for f in ("/root/work/eval300.jsonl",):
    for l in open(f): used.add(json.loads(l).get("q", "").strip())
for l in open("/root/work/mt_train.jsonl"):
    d = json.loads(l); used.add(d.get("seed", "")); used |= {t["q"] for t in d["turns"]}
seeds = []
for l in open("/root/work/selfq_all.jsonl"):
    try: r = json.loads(l)
    except Exception: continue
    if r.get("q") and r.get("gold") and r["q"].strip() not in used: seeds.append((r["q"].strip(), r["gold"].strip()))
seeds = list(dict.fromkeys(seeds)); random.Random(21).shuffle(seeds)
with open("/root/work/mt_train_sw.jsonl", "w") as fh:
    for k in range(min(120, len(seeds) // 2)):
        a, b = seeds[2 * k], seeds[2 * k + 1]
        fh.write(json.dumps({"id": f"tsw{k:03d}", "kind": "switch", "seed": a[0], "turns": [{"q": a[0], "gold": a[1], "standalone": a[0]}, {"q": b[0], "gold": b[1], "standalone": b[0]}]}, ensure_ascii=False) + "\n")
print("[mem7] training switch pairs written")
PY
while pgrep -f "pool_eval.py|memfit.py" >/dev/null; do sleep 60; done
echo "[mem7] collection start $(date -u +%H:%M)"
env $ENV python3 /root/work/pool_eval.py $B /root/work/eval300.jsonl /root/work/rft_sw.jsonl --multiturn /root/work/mt_train_sw.jsonl --mt-mode full --mt-save-tokens 1 --n 120 $EVARGS --tag "[rft-sw]" > /root/rft_sw.log 2>&1
grep -E "EVAL_DONE|Error|Traceback" /root/rft_sw.log | tail -1 | cut -c1-300
env $ENV python3 /root/work/pool_eval.py $B /root/work/eval300.jsonl /root/work/rft_mt.jsonl --multiturn /root/work/mt_train.jsonl --mt-mode full --mt-save-tokens 1 --n 40 $EVARGS --tag "[rft-mt]" > /root/rft_mt.log 2>&1
grep -E "EVAL_DONE|Error|Traceback" /root/rft_mt.log | tail -1 | cut -c1-300
cat /root/work/rft_sw.jsonl /root/work/rft_mt.jsonl > /root/work/rft_all.jsonl
hf upload $R /root/work/rft_all.jsonl pooler_distill/chatsft/multiturn/rft_all.jsonl >/dev/null 2>&1
echo "[mem7] memfit (ce) start $(date -u +%H:%M)"
env SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /root/work/memfit.py --objective ce --student full --ckpt $B --data /root/work/rft_all.jsonl --out /root/pooler_$T.safetensors --log /root/memfit_$T.log --steps 200 --val-every 50 --lr 0 --lr-lora 5e-6 > /root/memfit_${T}_run.log 2>&1
grep -E "^\[memfit\]|^\[data\]|^\[lora\]|^val |MEMFIT_DONE|Error|Traceback" /root/memfit_${T}_run.log | tail -14 | cut -c1-200
[ -s /root/pooler_$T.safetensors ] || { echo "MEM7_ABORT memfit: $(tail -3 /root/memfit_${T}_run.log | tr '\n' ' ' | cut -c1-300)"; exit 1; }
hf upload $R /root/pooler_$T.safetensors pooler_distill/chatsft/multiturn/pooler_$T.safetensors >/dev/null 2>&1
ev() { while pgrep -f "pool_eval.py" >/dev/null; do sleep 60; done
  env $ENV python3 /root/work/pool_eval.py $1 /root/work/eval300.jsonl $3 --multiturn $2 --mt-mode $4 --n $5 $EVARGS --tag "[$6]" > /root/$6.log 2>&1
  grep -E "EVAL_DONE|Error|Traceback" /root/$6.log | tail -2 | cut -c1-300; hf upload $R $3 pooler_distill/chatsft/multiturn/$(basename $3) >/dev/null 2>&1; }
ev /root/pooler_$T.safetensors /root/work/mt_eval_sw.jsonl /root/work/${T}_sw_full.jsonl full 60 ${T}-sw-full
for i in 0 1 2; do while pgrep -f "pool_eval.py" >/dev/null; do sleep 60; done
  env $ENV python3 /root/work/pool_eval.py /root/pooler_$T.safetensors /root/work/ev_$i.jsonl /root/work/${T}st_out_$i.jsonl --n 34 $EVARGS --tag "[${T}st$i]" > /root/${T}st_$i.log 2>&1
  grep -E "EVAL_DONE|Error|Traceback" /root/${T}st_$i.log | tail -1 | cut -c1-300; hf upload $R /root/work/${T}st_out_$i.jsonl pooler_distill/chatsft/rollouts/${T}st_$i.jsonl >/dev/null 2>&1; done
echo "MEM7_JOB_DONE $(date -u)"
M7
  chmod +x /root/mem7keep.sh
  setsid nohup bash -c 'bash /root/mem7keep.sh 2>&1 | tee -a /root/mem7.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MEM7_LAUNCHED $(date -u)"
fi

# No-training protocols on the switch set, queued behind mem7 (base model): quote - the earlier exchanges quoted as
# context inside the one user message; and the base full protocol again (how far a 60-dialogue reading moves by chance).
QT=${QT:-1}
if [ "$QT" = 1 ] && ! pgrep -f "qtkee[p].sh" >/dev/null && ! grep -q "QT_JOB_DONE" /root/qt.log 2>/dev/null; then
  cat > /root/qtkeep.sh <<'QK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
until grep -qE "MEM7_JOB_DONE|MEM7_ABORT" /root/mem7.log 2>/dev/null; do sleep 300; done
for m in "quote sw0q" "full sw0r"; do set -- $m
  while pgrep -f "pool_eval.py|memfit.py" >/dev/null; do sleep 60; done
  env $ENV python3 /root/work/pool_eval.py /root/reeval_g14m_pooler.safetensors /root/work/eval300.jsonl /root/work/$2_full.jsonl --multiturn /root/work/mt_eval_sw.jsonl --mt-mode $1 --n 60 $EVARGS --tag "[$2]" > /root/$2.log 2>&1
  grep -E "EVAL_DONE|Error|Traceback" /root/$2.log | tail -1 | cut -c1-300; hf upload $R /root/work/$2_full.jsonl pooler_distill/chatsft/multiturn/$2_full.jsonl >/dev/null 2>&1
done
echo "QT_JOB_DONE $(date -u)"
QK
  chmod +x /root/qtkeep.sh
  setsid nohup bash -c 'bash /root/qtkeep.sh 2>&1 | tee -a /root/qt.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "QT_LAUNCHED $(date -u)"
fi

# Quote, second round (after QT): the base model with quote on the first 30 mt_eval dialogues (does it keep the
# references and recall that the full protocol had, bridge 41.7 / memory 40.0?), and mem6 with quote on the switch set
# (does training add to the protocol fix?). QT2=0 cancels.
# 2026-09-30: cancelled for good - its keeper was re-launched on every control run after the MEM8 swap (no DONE line
# was ever written) and took the card after mem8; mem9 measures quote on the chains instead.
QT2=${QT2:-0}
if [ ! -e /root/.mem9_swap ]; then touch /root/.mem9_swap; pkill -f "qt2kee[p].sh"; pkill -f "pool_eval.py .*/(mtq|mem6q)_full"; sleep 3; echo "MEM9_SWAP stopped the re-launched quote follow-ups $(date -u)"; fi
if [ "$QT2" = 1 ] && ! pgrep -f "qt2kee[p].sh" >/dev/null && ! grep -q "QT2_JOB_DONE" /root/qt2.log 2>/dev/null; then
  cat > /root/qt2keep.sh <<'QK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
until grep -q "QT_JOB_DONE" /root/qt.log 2>/dev/null; do sleep 300; done
for m in "/root/reeval_g14m_pooler.safetensors /root/work/mt_eval.jsonl 30 mtq" "/root/pooler_mem6.safetensors /root/work/mt_eval_sw.jsonl 60 mem6q"; do set -- $m
  while pgrep -f "pool_eval.py|memfit.py" >/dev/null; do sleep 60; done
  env $ENV python3 /root/work/pool_eval.py $1 /root/work/eval300.jsonl /root/work/$4_full.jsonl --multiturn $2 --mt-mode quote --n $3 $EVARGS --tag "[$4]" > /root/$4.log 2>&1
  grep -E "EVAL_DONE|Error|Traceback" /root/$4.log | tail -1 | cut -c1-300; hf upload $R /root/work/$4_full.jsonl pooler_distill/chatsft/multiturn/$4_full.jsonl >/dev/null 2>&1
done
echo "QT2_JOB_DONE $(date -u)"
QK
  chmod +x /root/qt2keep.sh
  setsid nohup bash -c 'bash /root/qt2keep.sh 2>&1 | tee -a /root/qt2.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "QT2_LAUNCHED $(date -u)"
fi

# mem8 (2026-09-30): the user does not want the quote protocol in the app - the model itself must take a multi-turn
# chat. Quote is used only to COLLECT: under it the base model gets the switch turn right about twice as often, so the
# same box time yields twice its own correct trajectories; those are trained into the NATIVE multi-turn prompt
# (memfit --objective ce --ce-native 1, LoRA only). Four-turn chains, so the history is one to three exchanges deep.
# Training: 70 chains of 4 unrelated training questions. Measured on 30 held-out chains of 4 (the switch set's 60
# dialogues joined two by two), native protocol, base and mem8, turn by turn, and the single-turn search held-out.
if [ ! -e /root/.mem8_swap ]; then touch /root/.mem8_swap; pkill -f "qt2kee[p].sh"; pkill -f "pool_eval.py .*--mt-mode quote"; sleep 3; echo "MEM8_SWAP stopped the quote follow-ups $(date -u)"; fi
MEM8=${MEM8:-1}
if [ "$MEM8" = 1 ] && ! pgrep -f "mem8kee[p].sh" >/dev/null && ! grep -q "MEM8_JOB_DONE" /root/mem8.log 2>/dev/null; then
  cat > /root/mem8keep.sh <<'M8'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; T=mem8; B=/root/reeval_g14m_pooler.safetensors
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
python3 - <<'PY'
import json, random
used = set()
for f in ("/root/work/eval300.jsonl",):
    for l in open(f): used.add(json.loads(l).get("q", "").strip())
for f in ("/root/work/mt_train.jsonl", "/root/work/mt_train_sw.jsonl"):
    for l in open(f):
        d = json.loads(l); used.add(d.get("seed", "")); used |= {t["q"] for t in d["turns"]}
seeds = []
for l in open("/root/work/selfq_all.jsonl"):
    try: r = json.loads(l)
    except Exception: continue
    if r.get("q") and r.get("gold") and r["q"].strip() not in used: seeds.append((r["q"].strip(), r["gold"].strip()))
seeds = list(dict.fromkeys(seeds)); random.Random(31).shuffle(seeds)
with open("/root/work/mt_train_chain.jsonl", "w") as fh:
    for k in range(min(70, len(seeds) // 4)):
        qs = seeds[4 * k: 4 * k + 4]
        fh.write(json.dumps({"id": f"tch{k:03d}", "kind": "switch", "turns": [{"q": q, "gold": g, "standalone": q} for q, g in qs]}, ensure_ascii=False) + "\n")
sw = [json.loads(l) for l in open("/root/work/mt_eval_sw.jsonl")]
with open("/root/work/mt_eval_chain.jsonl", "w") as fh:
    for k in range(len(sw) // 2):
        a, b = sw[2 * k], sw[2 * k + 1]
        fh.write(json.dumps({"id": f"ech{k:03d}", "kind": "switch", "turns": a["turns"] + b["turns"]}, ensure_ascii=False) + "\n")
print(f"[mem8] {min(70, len(seeds)//4)} training chains of 4 ({len(seeds)} free seeds), {len(sw)//2} held-out chains of 4")
PY
while pgrep -f "pool_eval.py|memfit.py" >/dev/null; do sleep 60; done
echo "[mem8] collection (quote) start $(date -u +%H:%M)"
env $ENV python3 /root/work/pool_eval.py $B /root/work/eval300.jsonl /root/work/rft8.jsonl --multiturn /root/work/mt_train_chain.jsonl --mt-mode quote --mt-save-tokens 1 --n 70 $EVARGS --tag "[rft8]" > /root/rft8.log 2>&1
grep -E "EVAL_DONE|Error|Traceback" /root/rft8.log | tail -1 | cut -c1-300
hf upload $R /root/work/rft8.jsonl pooler_distill/chatsft/multiturn/rft8.jsonl >/dev/null 2>&1
echo "[mem8] memfit (ce, native) start $(date -u +%H:%M)"
env SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /root/work/memfit.py --objective ce --ce-native 1 --student full --ckpt $B --data /root/work/rft8.jsonl --out /root/pooler_$T.safetensors --log /root/memfit_$T.log --steps 250 --val-every 50 --lr 0 --lr-lora 5e-6 > /root/memfit_${T}_run.log 2>&1
grep -E "^\[memfit\]|^\[data\]|^val |MEMFIT_DONE|Error|Traceback" /root/memfit_${T}_run.log | tail -12 | cut -c1-200
[ -s /root/pooler_$T.safetensors ] || { echo "MEM8_ABORT memfit: $(tail -3 /root/memfit_${T}_run.log | tr '\n' ' ' | cut -c1-300)"; exit 1; }
hf upload $R /root/pooler_$T.safetensors pooler_distill/chatsft/multiturn/pooler_$T.safetensors >/dev/null 2>&1
ev() { while pgrep -f "pool_eval.py" >/dev/null; do sleep 60; done
  env $ENV python3 /root/work/pool_eval.py $1 /root/work/eval300.jsonl $3 --multiturn $2 --mt-mode $4 --n $5 $EVARGS --tag "[$6]" > /root/$6.log 2>&1
  grep -E "EVAL_DONE|Error|Traceback" /root/$6.log | tail -2 | cut -c1-300; hf upload $R $3 pooler_distill/chatsft/multiturn/$(basename $3) >/dev/null 2>&1; }
ev /root/pooler_$T.safetensors /root/work/mt_eval_chain.jsonl /root/work/${T}_chain.jsonl full 30 ${T}-chain
for i in 0 1 2; do while pgrep -f "pool_eval.py" >/dev/null; do sleep 60; done
  env $ENV python3 /root/work/pool_eval.py /root/pooler_$T.safetensors /root/work/ev_$i.jsonl /root/work/${T}st_out_$i.jsonl --n 34 $EVARGS --tag "[${T}st$i]" > /root/${T}st_$i.log 2>&1
  grep -E "EVAL_DONE|Error|Traceback" /root/${T}st_$i.log | tail -1 | cut -c1-300; hf upload $R /root/work/${T}st_out_$i.jsonl pooler_distill/chatsft/rollouts/${T}st_$i.jsonl >/dev/null 2>&1; done
ev $B /root/work/mt_eval_chain.jsonl /root/work/base_chain.jsonl full 30 base-chain
echo "MEM8_JOB_DONE $(date -u)"
M8
  chmod +x /root/mem8keep.sh
  setsid nohup bash -c 'bash /root/mem8keep.sh 2>&1 | tee -a /root/mem8.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MEM8_LAUNCHED $(date -u)"
fi

# mem9 (2026-09-30): mem8 did not move the native chains (turns 2-4: base 34/90, mem8 35/90, +1.1 +- 5.6), and on
# them the base's turn 3 (a first question after two unrelated exchanges) reads as high as turn 1. Before more
# collection: how much of the later turns' drop is the history at all. Each chain question alone (no history), twice
# (the reading's spread), and quote on the same chains - the measured ceiling a history protocol can reach.
MEM9=${MEM9:-1}
if [ "$MEM9" = 1 ] && grep -q "MEM8_JOB_DONE" /root/mem8.log 2>/dev/null && ! pgrep -f "mem9kee[p].sh" >/dev/null && ! grep -q "MEM9_JOB_DONE" /root/mem9.log 2>/dev/null; then
  cat > /root/mem9keep.sh <<'M9'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; B=/root/reeval_g14m_pooler.safetensors
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
ev() { while pgrep -f "pool_eval.py|memfit.py" >/dev/null; do sleep 60; done
  env $ENV python3 /root/work/pool_eval.py $B /root/work/eval300.jsonl /root/work/$1.jsonl --multiturn /root/work/mt_eval_chain.jsonl --mt-mode $2 --n 30 $EVARGS --tag "[$1]" > /root/$1.log 2>&1
  grep -E "EVAL_DONE|Error|Traceback" /root/$1.log | tail -2 | cut -c1-300; hf upload $R /root/work/$1.jsonl pooler_distill/chatsft/multiturn/$1.jsonl >/dev/null 2>&1; }
echo "[mem9] start $(date -u +%H:%M)"
ev base_chain_alone1 none
ev base_chain_quote quote
ev base_chain_alone2 none
echo "MEM9_JOB_DONE $(date -u)"
M9
  chmod +x /root/mem9keep.sh
  setsid nohup bash -c 'bash /root/mem9keep.sh 2>&1 | tee -a /root/mem9.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MEM9_LAUNCHED $(date -u)"
fi

# mem10 (2026-10-01): mem9's first reading says the history costs the switch turns (turn 2: alone 19, history 10 of 30;
# turns 2-4: 21 vs 6 discordant), but 30 chains are too few to size it. A held-out of 100 fresh chains of 4 (400
# questions never trained or evaluated), base with history and alone; and a second training round 4x mem8's size,
# collected ALONE (each question run by itself - the most successes per box hour; their tokens trained into the native
# history prompt, ce-native, LoRA only, one success per distinct question, nothing solved twice). Two lanes on the card:
# the held-out measurements and the collection run side by side.
MEM10=${MEM10:-0}   # 2026-10-01 02:10: replaced by mem10b (only two processes fit the card: 6.5 GB each)
# mem10b: mem9's two alone runs agree (turn 2 alone 19 / 18 vs history 10 of 30; turns 2-4 49 / 48 vs 34), so the
# history's cost is settled enough to act on and the 100-chain base measurement is dropped. Two fixed lanes (all the
# card holds): the training collection (202 chains, each question alone) in four shards, two at a time; ce-native
# training (LoRA only); then mem10 on the same 30 chains as base / mem8 and the single-turn 102, split across the lanes.
if [ ! -e /root/.mem10b_swap ]; then touch /root/.mem10b_swap; pkill -f "mem10kee[p].sh"; pkill -f "pool_eval.py .*base_c100_"; pkill -f "pool_eval.py .*rft10_s"; sleep 3; echo "MEM10B_SWAP stopped mem10 $(date -u)"; fi
MEM10B=${MEM10B:-0}   # 2026-10-01 02:30: replaced by mem10c (its second lane hit OOM at load, both shards)
# mem10c: the full-sequence logits (152k vocabulary, fp32, over prompts of thousands of tokens) were most of the 6.3 GB
# a process held; SP_LASTLOGIT=1 computes the head on the last position only (the only one read). Lane A waits for
# mem10b's running shard s0 and then runs s2; lane B runs s1 and s3 now; then training and the evaluation as in mem10b.
if [ ! -e /root/.mem10c_swap ]; then touch /root/.mem10c_swap; pkill -f "mem10bkee[p].sh"; sleep 2; echo "MEM10C_SWAP stopped the mem10b keeper (its running shard s0 continues) $(date -u)"; fi
MEM10C=${MEM10C:-0}   # 2026-10-01 02:50: replaced by mem10d - the second process still hit OOM, at LOAD (5.3 GB before
# generating anything: the trainer's prefix that builds the model), so the last-position head did not make room.
# mem10d: one process at a time. The collection keeps shards s0 (running) and s2 - 101 chains, 404 questions alone,
# about 3x mem8's successes - and drops s1 / s3; then training, then mem10 on the 30 chains and the single-turn 102,
# serially (the chains first).
if [ ! -e /root/.mem10d_swap ]; then touch /root/.mem10d_swap; pkill -f "mem10ckee[p].sh"; sleep 2; echo "MEM10D_SWAP stopped the mem10c keeper (shard s0 continues) $(date -u)"; fi
MEM10D=${MEM10D:-1}
if [ "$MEM10D" = 1 ] && ! pgrep -f "mem10dkee[p].sh" >/dev/null && ! grep -q "MEM10D_JOB_DONE" /root/mem10d.log 2>/dev/null; then
  cat > /root/mem10dkeep.sh <<'M10'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; T=mem10; B=/root/reeval_g14m_pooler.safetensors
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
idle() { while pgrep -f "pool_eval.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done; }
mt() { idle; env $ENV python3 /root/work/pool_eval.py $1 /root/work/eval300.jsonl /root/work/$2.jsonl --multiturn $3 --mt-mode $4 --n 1000 $5 $EVARGS --tag "[$2]" > /root/$2.log 2>&1
  echo "[mem10d] $(grep -E "EVAL_DONE|Error|Traceback" /root/$2.log | tail -1 | cut -c1-250) $(date -u +%H:%M)"; }
st() { idle; env $ENV python3 /root/work/pool_eval.py /root/pooler_$T.safetensors /root/work/ev_$1.jsonl /root/work/${T}st_out_$1.jsonl --n 34 $EVARGS --tag "[${T}st$1]" > /root/${T}st_$1.log 2>&1
  echo "[mem10d] $(grep -E "EVAL_DONE|Error|Traceback" /root/${T}st_$1.log | tail -1 | cut -c1-250)"; hf upload $R /root/work/${T}st_out_$1.jsonl pooler_distill/chatsft/rollouts/${T}st_$1.jsonl >/dev/null 2>&1; }
echo "[mem10d] start $(date -u +%H:%M), waiting for shard s0"
mt $B rft10_s2 /root/work/mt_train_c10_s2.jsonl none "--mt-save-tokens 1"; idle
cat /root/work/rft10_s0.jsonl /root/work/rft10_s2.jsonl > /root/work/rft10.jsonl; echo "[mem10d] rft10: $(wc -l < /root/work/rft10.jsonl) rows"
hf upload $R /root/work/rft10.jsonl pooler_distill/chatsft/multiturn/rft10.jsonl >/dev/null 2>&1
echo "[mem10d] memfit (ce, native) start $(date -u +%H:%M)"
env SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /root/work/memfit.py --objective ce --ce-native 1 --student full --ckpt $B --data /root/work/rft10.jsonl --out /root/pooler_$T.safetensors --log /root/memfit_$T.log --steps 500 --val-every 100 --lr 0 --lr-lora 5e-6 > /root/memfit_${T}_run.log 2>&1
grep -E "^\[memfit\]|^\[data\]|^val |MEMFIT_DONE|Error|Traceback" /root/memfit_${T}_run.log | tail -12 | cut -c1-200
[ -s /root/pooler_$T.safetensors ] || { echo "MEM10D_ABORT memfit: $(tail -3 /root/memfit_${T}_run.log | tr '\n' ' ' | cut -c1-300)"; exit 1; }
hf upload $R /root/pooler_$T.safetensors pooler_distill/chatsft/multiturn/pooler_$T.safetensors >/dev/null 2>&1
mt /root/pooler_$T.safetensors ${T}_chain /root/work/mt_eval_chain.jsonl full ""
hf upload $R /root/work/${T}_chain.jsonl pooler_distill/chatsft/multiturn/${T}_chain.jsonl >/dev/null 2>&1
st 0; st 1; st 2
echo "MEM10D_JOB_DONE $(date -u)"
M10
  chmod +x /root/mem10dkeep.sh
  setsid nohup bash -c 'bash /root/mem10dkeep.sh 2>&1 | tee -a /root/mem10d.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MEM10D_LAUNCHED $(date -u)"
fi

# mtg1 (2026-10-01): multi-turn GRPO with the teacher's naturalness check - the user's next step ("seeds exist, go
# straight to nano-judged GRPO"). online_loop.py --mt-items: every step is one search turn of a training conversation,
# rolled out 8 times with the earlier exchanges in the prompt exactly as the app sends them (the model's own replies
# from the rft8 / rft10 collections), scored as in the search GRPO (grounded-correct 1.5, nano's natural/sound/clean
# +0.5 - the judge now also sees the history and fails a reply that answers or drags in the earlier topic),
# group-normalised advantage, LoRA all layers r16 at 1e-5, pooler frozen, collapse guard on. Starts from the base when
# mem10d is done; 100 steps, then the 30 held-out chains and the single-turn screen (shard 0) on the checkpoint.
# pqjudge (2026-10-01): box J measures the 4-bit pooler on the Dolphin held-out but holds no judge key; this box judges
# its replies (nano, REASON_SYS, the same judge and held-out as every Dolphin number so far) as each arm's file lands
# on the hub. CPU and API only - the card stays with the multi-turn work.
# 2026-10-01: mtg1 died at step 50 on "No space left on device" while saving; what fills the 45 GB disk (once per serial)
if [ ! -e /root/.du_2026092705 ]; then touch /root/.du_2026092705; { echo "DISK_REPORT $(date -u)"; df -h /root | tail -1; du -xsh /root/* /root/.cache 2>/dev/null | sort -h | tail -25; du -xsh /root/work/* 2>/dev/null | sort -h | tail -8; ls -la /root/online_mtg1 2>/dev/null; } > /root/disk_report.txt 2>&1; hf upload baya1116/hypernet-sp-distill /root/disk_report.txt pooler_distill/chatsft/audit/disk_report_G.txt >/dev/null 2>&1; echo "DISK_REPORT uploaded"; fi
# mtg1b: mtg1 died at step 50 on a full disk (its checkpoints carried the 3.5 GB base each); it resumes from its step-25
# save with LoRA-and-pooler-only checkpoints, after the mem5-7 poolers (on the hub) are cleared, and is measured again
if [ ! -e /root/.free_2026092706 ]; then touch /root/.free_2026092706; for n in 5 6 7; do rm -f /root/pooler_mem$n.safetensors; done; echo "FREED mem5-7 poolers (on the hub): $(df -h /root | tail -1)"; fi
if ! pgrep -f "mtg1bkee[p].sh" >/dev/null && ! grep -q "MTG1B_JOB_DONE" /root/mtg1b.log 2>/dev/null; then
  cat > /root/mtg1bkeep.sh <<'MB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; OUT=/root/online_mtg1
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
until grep -qE "MTG1_JOB_DONE|MTG1_ABORT" /root/mtg1.log 2>/dev/null; do sleep 60; done
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
cp /root/work/mtg1_chain.jsonl /root/work/mtg1s25_chain.jsonl 2>/dev/null; cp /root/work/mtg1st_out_0.jsonl /root/work/mtg1s25st_out_0.jsonl 2>/dev/null
rm -f $OUT/*.tmp; echo "[mtg1b] resume from step $(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])") $(date -u +%H:%M); $(df -h /root | tail -1)"
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OAI_KEY=$(cat /root/.oai 2>/dev/null) DSK_KEY=$(cat /root/.dsk 2>/dev/null) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /root/work/online_loop.py /root/pooler_mem10.safetensors $OUT --pooler none --lora-layers all --lora-rank 16 --save-lora-only 1 \
  --mt-items /root/work/mtg_items.jsonl --heldout /root/work/eval300.jsonl --reason-g 8 --steps 100 --save-every 25 \
  --search-lr 1e-5 --search-temp 0.9 --search-gen 2000 --temp 0.6 --gen 2000 --budget 900 --maxsrch 7 --stop eos \
  --judge-api openai --judge-model gpt-5-nano --w-talk 0.5 --dolphin-min 0 --adv-std 1 --pg-norm mean --kl 0 \
  --guard 1 --guard-steps 20 >> /root/mtg1_run.log 2>&1
grep -E "ONLINE_|Error|Traceback" /root/mtg1_run.log | tail -3 | cut -c1-250; grep -E "^\[step" /root/mtg1_run.log | tail -2 | cut -c1-200
S=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])")
[ "$S" -ge 100 ] || { echo "MTG1B_ABORT: stopped at step $S"; exit 1; }
hf upload $R $OUT/latest.safetensors pooler_distill/chatsft/multiturn/mtg1_s100.safetensors >/dev/null 2>&1
env $ENV python3 /root/work/pool_eval.py $OUT/latest.safetensors /root/work/eval300.jsonl /root/work/mtg1s100_chain.jsonl --multiturn /root/work/mt_eval_chain.jsonl --mt-mode full --n 1000 $EVARGS --tag "[mtg1s100-chain]" > /root/mtg1s100_chain.log 2>&1
echo "[mtg1b] $(grep -E "EVAL_DONE|Error|Traceback" /root/mtg1s100_chain.log | tail -1 | cut -c1-250)"; hf upload $R /root/work/mtg1s100_chain.jsonl pooler_distill/chatsft/multiturn/mtg1s100_chain.jsonl >/dev/null 2>&1
for i in 0 1 2; do
  env $ENV python3 /root/work/pool_eval.py $OUT/latest.safetensors /root/work/ev_$i.jsonl /root/work/mtg1s100st_out_$i.jsonl --n 34 $EVARGS --tag "[mtg1s100st$i]" > /root/mtg1s100st_$i.log 2>&1
  echo "[mtg1b] $(grep -E "EVAL_DONE|Error|Traceback" /root/mtg1s100st_$i.log | tail -1 | cut -c1-250)"
done
echo "MTG1B_JOB_DONE $(date -u)"
MB
  setsid nohup bash -c 'bash /root/mtg1bkeep.sh 2>&1 | tee -a /root/mtg1b.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MTG1B_LAUNCHED $(date -u)"
fi
# mtg1c (2026-10-02 00:50 JST, the user: "GRPO continues"): mtg1 resumes from its step-100 save (LoRA + pooler) with the
# same settings, to 200 - measured there (30 chains + the single-turn screen, shards 0-2) - then on to 300, measured again.
if ! pgrep -f "mtg1ckee[p].sh" >/dev/null && ! grep -q "MTG1C_JOB_DONE\|MTG1C_ABORT" /root/mtg1c.log 2>/dev/null; then
  cat > /root/mtg1ckeep.sh <<'MC'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; OUT=/root/online_mtg1
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
until grep -qE "MTG1B_JOB_DONE|MTG1B_ABORT" /root/mtg1b.log 2>/dev/null; do sleep 60; done
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
for T in 200 300; do
  rm -f $OUT/*.tmp; echo "[mtg1c] resume from step $(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])") to $T $(date -u +%H:%M); $(df -h /root | tail -1)"
  env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OAI_KEY=$(cat /root/.oai 2>/dev/null) DSK_KEY=$(cat /root/.dsk 2>/dev/null) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 /root/work/online_loop.py /root/pooler_mem10.safetensors $OUT --pooler none --lora-layers all --lora-rank 16 --save-lora-only 1 \
    --mt-items /root/work/mtg_items.jsonl --heldout /root/work/eval300.jsonl --reason-g 8 --steps $T --save-every 25 \
    --search-lr 1e-5 --search-temp 0.9 --search-gen 2000 --temp 0.6 --gen 2000 --budget 900 --maxsrch 7 --stop eos \
    --judge-api openai --judge-model gpt-5-nano --w-talk 0.5 --dolphin-min 0 --adv-std 1 --pg-norm mean --kl 0 \
    --guard 1 --guard-steps 20 >> /root/mtg1_run.log 2>&1
  grep -E "^\[init\]" /root/mtg1_run.log | tail -1 | cut -c1-200; grep -E "ONLINE_|Error|Traceback" /root/mtg1_run.log | tail -2 | cut -c1-250
  S=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])")
  [ "$S" -ge $T ] || { echo "MTG1C_ABORT: stopped at step $S"; exit 1; }
  cp $OUT/latest.safetensors /root/mtg1_s$T.safetensors
  hf upload $R /root/mtg1_s$T.safetensors pooler_distill/chatsft/multiturn/mtg1_s$T.safetensors >/dev/null 2>&1
  env $ENV python3 /root/work/pool_eval.py /root/mtg1_s$T.safetensors /root/work/eval300.jsonl /root/work/mtg1s${T}_chain.jsonl --multiturn /root/work/mt_eval_chain.jsonl --mt-mode full --n 1000 $EVARGS --tag "[mtg1s$T-chain]" > /root/mtg1s${T}_chain.log 2>&1
  echo "[mtg1c] $(grep -E "EVAL_DONE|Error|Traceback" /root/mtg1s${T}_chain.log | tail -1 | cut -c1-250)"; hf upload $R /root/work/mtg1s${T}_chain.jsonl pooler_distill/chatsft/multiturn/mtg1s${T}_chain.jsonl >/dev/null 2>&1
  for i in 0 1 2; do
    env $ENV python3 /root/work/pool_eval.py /root/mtg1_s$T.safetensors /root/work/ev_$i.jsonl /root/work/mtg1s${T}st_out_$i.jsonl --n 34 $EVARGS --tag "[mtg1s${T}st$i]" > /root/mtg1s${T}st_$i.log 2>&1
    echo "[mtg1c] $(grep -E "EVAL_DONE|Error|Traceback" /root/mtg1s${T}st_$i.log | tail -1 | cut -c1-250)"
  done
done
echo "MTG1C_JOB_DONE $(date -u)"
MC
  setsid nohup bash -c 'bash /root/mtg1ckeep.sh 2>&1 | tee -a /root/mtg1c.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MTG1C_LAUNCHED $(date -u)"
fi
# mtg1d (2026-10-02 12:30 JST): step 300 is the best on the chains so far (turns 2-4 52/90, turn 4 17/30) and the user
# asked GRPO to continue: same settings on to 400 and 500, each measured the same way, after mtg1c.
if ! pgrep -f "mtg1dkee[p].sh" >/dev/null && ! grep -q "MTG1D_JOB_DONE\|MTG1D_ABORT" /root/mtg1d.log 2>/dev/null; then
  cat > /root/mtg1dkeep.sh <<'MD'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; OUT=/root/online_mtg1
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
until grep -qE "MTG1C_JOB_DONE|MTG1C_ABORT" /root/mtg1c.log 2>/dev/null; do sleep 60; done
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
for T in 400 500; do
  rm -f $OUT/*.tmp; echo "[mtg1d] resume from step $(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])") to $T $(date -u +%H:%M); $(df -h /root | tail -1)"
  env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OAI_KEY=$(cat /root/.oai 2>/dev/null) DSK_KEY=$(cat /root/.dsk 2>/dev/null) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 /root/work/online_loop.py /root/pooler_mem10.safetensors $OUT --pooler none --lora-layers all --lora-rank 16 --save-lora-only 1 \
    --mt-items /root/work/mtg_items.jsonl --heldout /root/work/eval300.jsonl --reason-g 8 --steps $T --save-every 25 \
    --search-lr 1e-5 --search-temp 0.9 --search-gen 2000 --temp 0.6 --gen 2000 --budget 900 --maxsrch 7 --stop eos \
    --judge-api openai --judge-model gpt-5-nano --w-talk 0.5 --dolphin-min 0 --adv-std 1 --pg-norm mean --kl 0 \
    --guard 1 --guard-steps 20 >> /root/mtg1_run.log 2>&1
  grep -E "^\[init\]" /root/mtg1_run.log | tail -1 | cut -c1-200; grep -E "ONLINE_|Error|Traceback" /root/mtg1_run.log | tail -2 | cut -c1-250
  S=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])")
  [ "$S" -ge $T ] || { echo "MTG1D_ABORT: stopped at step $S"; exit 1; }
  cp $OUT/latest.safetensors /root/mtg1_s$T.safetensors
  hf upload $R /root/mtg1_s$T.safetensors pooler_distill/chatsft/multiturn/mtg1_s$T.safetensors >/dev/null 2>&1
  env $ENV python3 /root/work/pool_eval.py /root/mtg1_s$T.safetensors /root/work/eval300.jsonl /root/work/mtg1s${T}_chain.jsonl --multiturn /root/work/mt_eval_chain.jsonl --mt-mode full --n 1000 $EVARGS --tag "[mtg1s$T-chain]" > /root/mtg1s${T}_chain.log 2>&1
  echo "[mtg1d] $(grep -E "EVAL_DONE|Error|Traceback" /root/mtg1s${T}_chain.log | tail -1 | cut -c1-250)"; hf upload $R /root/work/mtg1s${T}_chain.jsonl pooler_distill/chatsft/multiturn/mtg1s${T}_chain.jsonl >/dev/null 2>&1
  for i in 0 1 2; do
    env $ENV python3 /root/work/pool_eval.py /root/mtg1_s$T.safetensors /root/work/ev_$i.jsonl /root/work/mtg1s${T}st_out_$i.jsonl --n 34 $EVARGS --tag "[mtg1s${T}st$i]" > /root/mtg1s${T}st_$i.log 2>&1
    echo "[mtg1d] $(grep -E "EVAL_DONE|Error|Traceback" /root/mtg1s${T}st_$i.log | tail -1 | cut -c1-250)"
  done
done
echo "MTG1D_JOB_DONE $(date -u)"
MD
  setsid nohup bash -c 'bash /root/mtg1dkeep.sh 2>&1 | tee -a /root/mtg1d.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MTG1D_LAUNCHED $(date -u)"
fi
# one-shot (2026-10-02 09:40 JST, the user: is the thinking length being capped? ~1500 tokens incl. served pages should
# be allowed): why mtg1's unfinished rollouts did not finish - the token cap (--search-gen 2000 model tokens), the
# 900 s batch budget, the 7-search guard, a loop - and their lengths in tokens with and without the served pages
# ---- mix0 (2026-10-02 13:00 JST, the user: switch is about enough - add follow-up questions and reasoning, one at a
# time). The switch-only continuation (mtg1d) is cancelled. Baseline first, on the original model and on GRPO step 300:
# follow-up (bridge) dialogues, the 40 of mt_eval, and the Dolphin reasoning held-out (100, nano, REASON_SYS). In
# parallel (API only) more bridge training dialogues are written; then step 300 answers their first turns, which become
# the history of the follow-up training items (mtg2 trains on switch + ~30% bridge from step 300).
if [ ! -e /root/.mtg1d_cancel ]; then touch /root/.mtg1d_cancel; touch /root/mtg1d.log; echo "MTG1D_ABORT cancelled for mix0 $(date -u)" >> /root/mtg1d.log
  pkill -f "mtg1dkee[p].sh"; pkill -f "online_loop.py /root/pooler_mem10.safetensors /root/online_mtg1 .*--steps (400|500)"; echo "MTG1D_CANCELLED $(date -u)"; fi
if ! pgrep -f "mix0kee[p].sh" >/dev/null && ! grep -q "MIX0_JOB_DONE\|MIX0_ABORT" /root/mix0.log 2>/dev/null; then
  cat > /root/mix0keep.sh <<'MX'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; B=/root/reeval_g14m_pooler.safetensors; S3=/root/mtg1_s300.safetensors
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
# bridge training dialogues (API, in the background while the card measures)
( python3 /root/work/mt_gen.py --seeds /root/work/selfq_all.jsonl --exclude /root/work/eval300.jsonl --out /root/work/mt_train_br2.jsonl --n-bridge 320 --n-memory 0 --n-switch 0 --seed 11 2>&1 | grep -E "^\[mtgen\]|MTGEN_DONE|Error" ) &
GEN=$!
until grep -qE "MTG1C_JOB_DONE|MTG1C_ABORT" /root/mtg1c.log 2>/dev/null; do sleep 60; done
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
python3 -c "
import json
r=[json.loads(l) for l in open('/root/work/mt_eval.jsonl')]
b=[d for d in r if d['kind']=='bridge']
open('/root/work/mt_eval_bridge.jsonl','w').write(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in b)); print('[mix0] bridge held-out', len(b), 'dialogues')"
for arm in "base:$B" "s300:$S3"; do T=${arm%%:*}; CK=${arm#*:}
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl /root/work/br_$T.jsonl --multiturn /root/work/mt_eval_bridge.jsonl --mt-mode full --n 1000 $EVARGS --tag "[br-$T]" > /root/br_$T.log 2>&1
  echo "[mix0] $(grep -E "EVAL_DONE|Error|Traceback" /root/br_$T.log | tail -1 | cut -c1-200)"; hf upload $R /root/work/br_$T.jsonl pooler_distill/chatsft/multiturn/br_$T.jsonl >/dev/null 2>&1
done
python3 -c "
import json
rows=[json.loads(l) for l in open('/root/work/dolphin_heldout100.jsonl') if l.strip()]
open('/root/work/dolphinq.jsonl','w').write(''.join(json.dumps({'q':r['q']},ensure_ascii=False)+'\n' for r in rows))"
for arm in "base:/root/gptq_hf_gq14:--pooler-init $B" "s300:$S3:"; do T=${arm%%:*}; r=${arm#*:}; CK=${r%%:*}; PI=${r#*:}
  env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $CK /root/evalrun_dl_$T \
    --questions /root/work/eval300.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl $PI \
    --b 12 --gen 7000 --budget 2400 --temp 0.6 --maxsrch 7 --pooler none --lora-rank 16 --lora-layers all --stop eos --loop-break answer \
    --eval-file /root/work/dolphinq.jsonl --eval-out /root/work/dl_$T.jsonl > /root/dl_$T.log 2>&1
  echo "[mix0] dolphin $T: $(wc -l < /root/work/dl_$T.jsonl 2>/dev/null) replies $(grep -E 'EVAL_DONE|Error|Traceback' /root/dl_$T.log | tail -1 | cut -c1-120)"
  OAI_KEY=$(cat /root/.oai 2>/dev/null) python3 - /root/work/dl_$T.jsonl $T <<'PYJ'
import json, os, sys, urllib.request, time
from concurrent.futures import ThreadPoolExecutor
src = open("/root/work/online_loop.py").read(); i = src.index("REASON_SYS = "); j = src.index('"""', src.index('"""', i) + 3) + 3
ns = {}; exec(src[i:j], ns); SYS = ns["REASON_SYS"]
ref = {json.loads(l)["q"].strip(): json.loads(l)["ref"] for l in open("/root/work/dolphin_heldout100.jsonl") if l.strip()}
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]; key = os.environ.get("OAI_KEY", "")
def judge(r):
    t = r["text"]; reply = t.split("</think>")[-1].strip() if "</think>" in t else ""
    if not reply: return 0, "unfinished"
    body = {"model": "gpt-5-nano", "max_completion_tokens": 2000, "messages": [{"role": "system", "content": SYS},
            {"role": "user", "content": f"QUESTION:\n{r['q'][:2000]}\n\nREFERENCE ANSWER:\n{ref.get(r['q'].strip(), '')[:3000]}\n\nASSISTANT ANSWER:\n{reply[:3000]}"}]}
    err = "?"
    for _ in range(3):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}), timeout=120))
            c = d["choices"][0]["message"].get("content") or ""; v = json.loads(c[c.find("{"): c.rfind("}") + 1])
            return int(all(bool(v.get(k)) for k in ("solves_it", "follows_the_request", "language_english", "clean"))), v
        except Exception as e: time.sleep(3); err = type(e).__name__
    return 0, {"error": err}
with ThreadPoolExecutor(max_workers=8) as ex: res = list(ex.map(judge, rows))
with open(sys.argv[1].replace(".jsonl", "_judged.jsonl"), "w") as o:
    for r, (a, v) in zip(rows, res): o.write(json.dumps({"q": r["q"], "pass": a, "why": str(v)[:300], "text": r["text"]}, ensure_ascii=False) + "\n")
ok = sum(a for a, _ in res); unf = sum(1 for _, v in res if v == "unfinished"); errs = sum(1 for _, v in res if isinstance(v, dict) and "error" in v)
print(f"[mix0] DOLPHIN {sys.argv[2]} {100*ok/max(len(rows),1):.1f}% ({ok}/{len(rows)}) unfinished {unf} judge errors {errs}", flush=True)
PYJ
  hf upload $R /root/work/dl_${T}_judged.jsonl pooler_distill/chatsft/multiturn/dl_${T}_judged.jsonl >/dev/null 2>&1
done
wait $GEN
python3 - <<'PY2'
import json
held = set()
for f in ("/root/work/eval300.jsonl", "/root/work/mt_eval.jsonl", "/root/work/mt_eval_chain.jsonl"):
    for l in open(f):
        d = json.loads(l); held.add(d.get("q", "").strip()); held |= {t["q"].strip() for t in d.get("turns", [])}; held.add(d.get("seed", ""))
out = []
for f in ("/root/work/mt_train.jsonl", "/root/work/mt_train_br2.jsonl"):
    try:
        for l in open(f):
            d = json.loads(l)
            if d["kind"] == "bridge" and d.get("seed", "") not in held and not any(t["q"].strip() in held for t in d["turns"]): out.append(d)
    except FileNotFoundError: pass
seen = set(); out = [d for d in out if not (d["turns"][0]["q"] in seen or seen.add(d["turns"][0]["q"]))]
open("/root/work/mt_train_bridge.jsonl", "w").write("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in out))
print(f"[mix0] bridge training dialogues: {len(out)}")
PY2
# step 300 answers the first turns (the batched rollout, 12 at a time); a dialogue whose first answer names the bridge
# entity gives two items: turn 1 alone, and the follow-up with [turn 1, step 300's reply] as its history
python3 -c "
import json
d=[json.loads(l) for l in open('/root/work/mt_train_bridge.jsonl')]
open('/root/work/br_t1q.jsonl','w').write(''.join(json.dumps({'q':x['turns'][0]['q']},ensure_ascii=False)+'\n' for x in d))"
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $S3 /root/evalrun_brt1 \
  --questions /root/work/eval300.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl \
  --b 12 --gen 2000 --budget 900 --temp 0.6 --maxsrch 7 --pooler none --lora-rank 16 --lora-layers all --stop eos \
  --eval-file /root/work/br_t1q.jsonl --eval-out /root/work/br_t1_s300.jsonl > /root/br_t1.log 2>&1
echo "[mix0] first turns answered: $(wc -l < /root/work/br_t1_s300.jsonl 2>/dev/null) $(grep -E 'EVAL_DONE|Error|Traceback' /root/br_t1.log | tail -1 | cut -c1-120)"
python3 - <<'PY3'
import json, random
d = [json.loads(l) for l in open("/root/work/mt_train_bridge.jsonl")]
a = {}
for l in open("/root/work/br_t1_s300.jsonl"):
    r = json.loads(l); t = r.get("text", ""); a[r["q"].strip()] = t.split("</think>")[-1].strip() if "</think>" in t else ""
br = []; ok = 0
for x in d:
    t1, t2 = x["turns"][0], x["turns"][1]; rep = a.get(t1["q"].strip(), "")
    br.append({"q": t1["q"], "gold": t1["gold"], "hist": []})
    if rep and t1["gold"].lower() in rep.lower():
        ok += 1; br.append({"q": t2["q"], "gold": t2["gold"], "hist": [{"role": "user", "content": t1["q"]}, {"role": "assistant", "content": rep[:1500]}]})
sw = [json.loads(l) for l in open("/root/work/mtg_items.jsonl")]
n_sw = int(len(br) * 7 / 3)                 # bridge items ~30% of the mix
random.Random(5).shuffle(sw); mix = br + sw[:n_sw]; random.Random(6).shuffle(mix)
open("/root/work/mtg2_items.jsonl", "w").write("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in mix))
print(f"[mix0] bridge: {len(d)} dialogues, first turn right in {ok}; mtg2 items {len(mix)} = {len(br)} bridge ({sum(1 for b in br if b['hist'])} follow-ups) + {min(n_sw, len(sw))} switch")
PY3
hf upload $R /root/work/mtg2_items.jsonl pooler_distill/chatsft/multiturn/mtg2_items.jsonl >/dev/null 2>&1
echo "MIX0_JOB_DONE $(date -u)"
MX
  setsid nohup bash -c 'bash /root/mix0keep.sh 2>&1 | tee -a /root/mix0.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MIX0_LAUNCHED $(date -u)"
fi
[ -s /root/dl_judge.py ] || { awk "/<<'PYJ'/{f=1;next} /^PYJ/{f=0} f" /root/mix0keep.sh > /root/dl_judge.py; echo "DL_JUDGE extracted $(wc -l < /root/dl_judge.py) lines"; }
# ---- mtg2 (queued 2026-10-02 16:35 JST): GRPO from step 300 on switch + ~30% bridge items (mix0's mtg2_items), same
# settings as mtg1, 100 steps; then the four screens: the 30 switch chains, the single-turn 102, the 40 bridge
# dialogues, the Dolphin reasoning 100 (nano, REASON_SYS) - against mix0's baselines for the original model and step 300.
if ! pgrep -f "mtg2kee[p].sh" >/dev/null && ! grep -q "MTG2_JOB_DONE\|MTG2_ABORT" /root/mtg2.log 2>/dev/null; then
  cat > /root/mtg2keep.sh <<'M2'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; OUT=/root/online_mtg2
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
until grep -qE "MIX0_JOB_DONE|MIX0_ABORT" /root/mix0.log 2>/dev/null; do sleep 120; done
[ -s /root/work/mtg2_items.jsonl ] || { echo "MTG2_ABORT no items"; exit 1; }
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
echo "[mtg2] $(wc -l < /root/work/mtg2_items.jsonl) items; start $(date -u +%H:%M); $(df -h /root | tail -1)"
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OAI_KEY=$(cat /root/.oai 2>/dev/null) DSK_KEY=$(cat /root/.dsk 2>/dev/null) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /root/work/online_loop.py /root/mtg1_s300.safetensors $OUT --pooler none --lora-layers all --lora-rank 16 --save-lora-only 1 \
  --mt-items /root/work/mtg2_items.jsonl --heldout /root/work/eval300.jsonl --reason-g 8 --steps 100 --save-every 25 \
  --search-lr 1e-5 --search-temp 0.9 --search-gen 2000 --temp 0.6 --gen 2000 --budget 900 --maxsrch 7 --stop eos \
  --judge-api openai --judge-model gpt-5-nano --w-talk 0.5 --dolphin-min 0 --adv-std 1 --pg-norm mean --kl 0 \
  --guard 1 --guard-steps 20 > /root/mtg2_run.log 2>&1
grep -E "ONLINE_|Error|Traceback" /root/mtg2_run.log | tail -2 | cut -c1-250
S=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])"); [ "$S" -ge 100 ] || { echo "MTG2_ABORT: stopped at step $S"; exit 1; }
cp $OUT/latest.safetensors /root/mtg2_s100.safetensors; hf upload $R /root/mtg2_s100.safetensors pooler_distill/chatsft/multiturn/mtg2_s100.safetensors >/dev/null 2>&1
CK=/root/mtg2_s100.safetensors
env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl /root/work/mtg2_chain.jsonl --multiturn /root/work/mt_eval_chain.jsonl --mt-mode full --n 1000 $EVARGS --tag "[mtg2-chain]" > /root/mtg2_chain.log 2>&1
echo "[mtg2] $(grep -E "EVAL_DONE|Error|Traceback" /root/mtg2_chain.log | tail -1 | cut -c1-200)"; hf upload $R /root/work/mtg2_chain.jsonl pooler_distill/chatsft/multiturn/mtg2_chain.jsonl >/dev/null 2>&1
env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl /root/work/br_mtg2.jsonl --multiturn /root/work/mt_eval_bridge.jsonl --mt-mode full --n 1000 $EVARGS --tag "[br-mtg2]" > /root/br_mtg2.log 2>&1
echo "[mtg2] $(grep -E "EVAL_DONE|Error|Traceback" /root/br_mtg2.log | tail -1 | cut -c1-200)"; hf upload $R /root/work/br_mtg2.jsonl pooler_distill/chatsft/multiturn/br_mtg2.jsonl >/dev/null 2>&1
for i in 0 1 2; do
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/ev_$i.jsonl /root/work/mtg2st_out_$i.jsonl --n 34 $EVARGS --tag "[mtg2st$i]" > /root/mtg2st_$i.log 2>&1
  echo "[mtg2] $(grep -E "EVAL_DONE|Error|Traceback" /root/mtg2st_$i.log | tail -1 | cut -c1-200)"
done
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $CK /root/evalrun_dl_mtg2 \
  --questions /root/work/eval300.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl \
  --b 12 --gen 7000 --budget 2400 --temp 0.6 --maxsrch 7 --pooler none --lora-rank 16 --lora-layers all --stop eos --loop-break answer \
  --eval-file /root/work/dolphinq.jsonl --eval-out /root/work/dl_mtg2.jsonl > /root/dl_mtg2.log 2>&1
[ -s /root/dl_judge.py ] && OAI_KEY=$(cat /root/.oai 2>/dev/null) python3 /root/dl_judge.py /root/work/dl_mtg2.jsonl mtg2 | sed 's/\[mix0\]/[mtg2]/'
hf upload $R /root/work/dl_mtg2_judged.jsonl pooler_distill/chatsft/multiturn/dl_mtg2_judged.jsonl >/dev/null 2>&1
echo "MTG2_JOB_DONE $(date -u)"
M2
  setsid nohup bash -c 'bash /root/mtg2keep.sh 2>&1 | tee -a /root/mtg2.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MTG2_LAUNCHED $(date -u)"
fi
# ---- searchq (2026-10-02 17:00 JST): questions for box I's article-search test - two per held-out passage, written by
# DeepSeek (natural; and one avoiding the article's name), API only
if ! pgrep -f "searchqkee[p].sh" >/dev/null && ! grep -q "SEARCHQ_DONE" /root/searchq.log 2>/dev/null; then
  cat > /root/searchqkeep.sh <<'SQ'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
until hf download $R sentbart/searcheval/docs500.jsonl --local-dir /root/sq >/dev/null 2>&1 && [ -s /root/sq/sentbart/searcheval/docs500.jsonl ]; do sleep 120; done
python3 - <<'PY2'
import json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
KEY = open("/root/.dsk").read().strip()
SYS = """You write search questions for testing an article search engine. Given an encyclopedia article's title and a short passage from it, write two questions a curious person might type, each answerable from the passage:
- "q_nat": a natural question; it may name the subject.
- "q_hard": a question about the same fact that does NOT use the article's title or any distinctive word of it (describe the subject instead).
Output JSON: {"q_nat": "...", "q_hard": "..."}"""
def ask(d):
    body = {"model": "deepseek-flash", "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": f"TITLE: {d['title']}\nPASSAGE: {d['passage']}"}],
            "max_tokens": 300, "temperature": 0.7, "response_format": {"type": "json_object"}}
    for t in range(4):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request("https://api.deepseek.com/chat/completions", data=json.dumps(body).encode(),
                headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}), timeout=120))
            v = json.loads(r["choices"][0]["message"]["content"]); return {"idx": d["idx"], "title": d["title"], "q_nat": v.get("q_nat", ""), "q_hard": v.get("q_hard", "")}
        except Exception: time.sleep(3 * (t + 1))
    return None
docs = [json.loads(l) for l in open("/root/sq/sentbart/searcheval/docs500.jsonl")]
with ThreadPoolExecutor(8) as ex: out = [o for o in ex.map(ask, docs) if o and o["q_nat"]]
open("/root/sq/queries.jsonl", "w").write("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out)); print(f"[searchq] {len(out)} of {len(docs)} passages got questions")
PY2
hf upload $R /root/sq/queries.jsonl sentbart/searcheval/queries.jsonl >/dev/null 2>&1; head -3 /root/sq/queries.jsonl | cut -c1-300
echo "SEARCHQ_DONE $(date -u)"
SQ
  setsid nohup bash -c 'bash /root/searchqkeep.sh 2>&1 | tee -a /root/searchq.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "SEARCHQ_LAUNCHED $(date -u)"
fi
# ---- dcq (2026-10-02 17:50 JST): (query the app model wrote -> page it was served) from every rollout on this box, kept
# when the page is one of box I's held-out documents. q_good: the rollout answered right and the page held the gold
# answer (the page it needed); q_api: any served page (agreement with Wikipedia's own search).
if ! pgrep -f "dcqkee[p].sh" >/dev/null && ! grep -q "DCQ_DONE" /root/dcq.log 2>/dev/null; then
  cat > /root/dcqkeep.sh <<'DQ'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
until hf download $R sentbart/searcheval/titles001.txt --local-dir /root/dcq >/dev/null 2>&1 && [ -s /root/dcq/sentbart/searcheval/titles001.txt ]; do sleep 120; done
python3 - <<'PY2'
import json, glob, re
T = {}
for i, t in enumerate(open("/root/dcq/sentbart/searcheval/titles001.txt")):
    T.setdefault(t.rstrip("\n").strip(), i)
SR = re.compile(r"<search>(.*?)</search", re.S)
files = glob.glob("/root/work/*.jsonl") + glob.glob("/root/online_*/rollouts.jsonl") + glob.glob("/root/work/**/*.jsonl", recursive=True)
good, api, nrows = {}, {}, 0
for f in sorted(set(files)):
    try:
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            t = r.get("text") if isinstance(r, dict) else None
            if not isinstance(t, str) or "<information>" not in t: continue
            nrows += 1
            gold = (r.get("gold") or "").strip().lower()
            ok = bool(r.get("correct")) and bool(r.get("grounded", True))
            for m in SR.finditer(t):
                q = m.group(1).split("||")[0].strip(); rest = t[m.end():m.end() + 3000]
                j = rest.find("<information>\n")
                if not q or j < 0 or j > 20: continue
                body = rest[j + 14:]; title = body.split(": ", 1)[0].strip()
                if title not in T: continue
                api[(q, title)] = T[title]
                chunk = body.split("</information>", 1)[0].lower()
                if ok and gold and gold in chunk: good[(q, title)] = T[title]
    except Exception: pass
rows = [{"idx": i, "q_good": q, "title": t} for (q, t), i in good.items()] + [{"idx": i, "q_api": q, "title": t} for (q, t), i in api.items()]
open("/root/dcq/dcq.jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
print(f"[dcq] {nrows} rollouts with served pages; pairs on held-out pages: {len(good)} needed (right answer, gold on the page), {len(api)} served")
PY2
hf upload $R /root/dcq/dcq.jsonl sentbart/searcheval/dcq.jsonl >/dev/null 2>&1; head -3 /root/dcq/dcq.jsonl | cut -c1-200
echo "DCQ_DONE $(date -u)"
DQ
  setsid nohup bash -c 'bash /root/dcqkeep.sh 2>&1 | tee -a /root/dcq.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "DCQ_LAUNCHED $(date -u)"
fi
# ---- memwin (2026-10-03 11:45 JST, the user: measure the plain spec - the newest raw-window tokens stay exposed,
# everything older goes into the pooler, turn boundaries or not; pinning the last exchange is not the spec). --mt-mode
# win: one stream across turns exactly as within a reply (only the new question pinned), memory bounded by
# question + 32 soft tokens + 768 raw. Replaces memcap/memcap2 (cancelled); mixcol held until the spec is chosen.
if [ ! -e /root/.memwin_swap ]; then
  touch /root/.memwin_swap
  pkill -f "memcapkee[p].sh"; pkill -f "memcap2kee[p].sh"; pkill -f "mixcolkee[p].sh"; pkill -f "pool_eval.py .*mc_"; sleep 5
  echo "MEMCAP_JOB_DONE cancelled for memwin $(date -u)" >> /root/memcap.log
  echo "MEMCAP2_JOB_DONE cancelled for memwin $(date -u)" >> /root/memcap2.log
  echo "MEMWIN_SWAP memcap/memcap2 cancelled, mixcol held $(date -u)"
fi
if ! pgrep -f "memwinkee[p].sh" >/dev/null && ! grep -q "MEMWIN_JOB_DONE" /root/memwin.log 2>/dev/null; then
  cat > /root/memwinkeep.sh <<'MW'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; CK=/root/mtg1_s300.safetensors
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
T=win
env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl /root/work/mc_$T.jsonl --multiturn /root/work/mt_eval_chain.jsonl --mt-mode win --n 1000 $EVARGS --tag "[mc-$T]" > /root/mc_$T.log 2>&1
python3 - /root/work/mc_$T.jsonl $T <<'PY2'
import json, sys
c = [0] * 4
for l in open(sys.argv[1]):
    r = json.loads(l); c[r["turn"]] += bool(r["correct"])
print(f"[memwin] {sys.argv[2]}: per turn {c}, turns 2-4 {sum(c[1:])}/90 (full history: 21 18 17 17 -> 52)", flush=True)
PY2
grep -E "EVAL_DONE|Error|Traceback" /root/mc_$T.log | tail -2 | cut -c1-200
hf upload $R /root/work/mc_$T.jsonl pooler_distill/chatsft/multiturn/mc_$T.jsonl >/dev/null 2>&1
echo "MEMWIN_JOB_DONE $(date -u)"
MW
  setsid nohup bash -c 'bash /root/memwinkeep.sh 2>&1 | tee -a /root/memwin.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MEMWIN_LAUNCHED $(date -u)"
fi
# ---- 2026-10-03 12:00 JST, the user: the topic-switch chains are settled; only the follow-ups matter. Stop the win
# chains where they are (partial file kept) so the bridge runs now.
if [ ! -e /root/.memwin_cut ]; then touch /root/.memwin_cut
  pkill -f "memwinkee[p].sh"; pkill -f "pool_eval.py .*mc_win"; sleep 5
  echo "[memwin] chains cut at $(wc -l < /root/work/mc_win.jsonl 2>/dev/null) turns: $(grep 'dialog' /root/mc_win.log | tail -1 | cut -c1-160)"  >> /root/memwin.log
  echo "MEMWIN_JOB_DONE cut for the bridge $(date -u)" >> /root/memwin.log
  HF_TOKEN=$(tr -d "[:space:]" < /root/.hf_token 2>/dev/null) hf upload baya1116/hypernet-sp-distill /root/work/mc_win.jsonl pooler_distill/chatsft/multiturn/mc_win_partial.jsonl >/dev/null 2>&1
  echo "MEMWIN_CUT $(date -u)"
fi
# ---- 2026-10-03 16:50 JST: disk at 94%; list the biggest items once (no deletion here)
if [ ! -e /root/.du_list1 ]; then touch /root/.du_list1
  { echo "DU_LIST $(date -u)"; du -xsh /root/* /root/work/* 2>/dev/null | sort -rh | head -40; } > /root/du_list.txt 2>&1
fi
# ---- 2026-10-03 20:20 JST (the user: is mtg3 failing? find out why): mtg3's rollouts and run log so far, plus mtg1's
# and mtg2's rollouts, on the hub for the analysis (once)
if [ ! -e /root/.mtg3_snap1 ]; then touch /root/.mtg3_snap1
  ( HF_TOKEN=$(tr -d "[:space:]" < /root/.hf_token 2>/dev/null); export HF_TOKEN; RR=baya1116/hypernet-sp-distill
    cp /root/online_mtg3/rollouts.jsonl /root/mtg3_roll_snap.jsonl; cp /root/mtg3_run.log /root/mtg3_run_snap.log
    hf upload $RR /root/mtg3_roll_snap.jsonl pooler_distill/chatsft/multiturn/analysis/mtg3_rollouts_snap.jsonl >/dev/null 2>&1
    hf upload $RR /root/mtg3_run_snap.log pooler_distill/chatsft/multiturn/analysis/mtg3_run_snap.log >/dev/null 2>&1
    [ -s /root/online_mtg2/rollouts.jsonl ] && hf upload $RR /root/online_mtg2/rollouts.jsonl pooler_distill/chatsft/multiturn/analysis/mtg2_rollouts.jsonl >/dev/null 2>&1
    [ -s /root/online_mtg1/rollouts.jsonl ] && hf upload $RR /root/online_mtg1/rollouts.jsonl pooler_distill/chatsft/multiturn/analysis/mtg1_rollouts.jsonl >/dev/null 2>&1
    hf upload $RR /root/work/mtg2_items.jsonl pooler_distill/chatsft/multiturn/analysis/mtg2_items.jsonl >/dev/null 2>&1
    hf upload $RR /root/work/mt_train_bridge.jsonl pooler_distill/chatsft/multiturn/analysis/mt_train_bridge.jsonl >/dev/null 2>&1
    hf upload $RR /root/mtg2_run.log pooler_distill/chatsft/multiturn/analysis/mtg2_run.log >/dev/null 2>&1
    hf upload $RR /root/mtg1_run.log pooler_distill/chatsft/multiturn/analysis/mtg1_run.log >/dev/null 2>&1
    echo "MTG3_SNAP_UP $(date -u)" >> /root/mtg3.log ) > /dev/null 2>&1 &
fi
# ---- 2026-10-05 08:05 JST: the disk filled during teach (its SFT/screens died "No space left on device"). Upload the
# teach logs for the post-mortem, then free space: finished GRPO run dirs (their checkpoints were copied out and are on
# the hub), the old SFT states, superseded local search stores and old pooler checkpoints. Nothing a running job reads.
if [ ! -e /root/.diskfree_g1 ]; then touch /root/.diskfree_g1
  echo "GDISK_BEFORE $(df -h /root | tail -1)"
  for f in /root/teach.log /root/probe_s100.log /root/teach1_run.log; do [ -s $f ] && HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token) hf upload baya1116/hypernet-sp-distill $f pooler_distill/chatsft/teach/logs/$(basename $f) >/dev/null 2>&1; done
  for f in /root/work/probe_s100.jsonl /root/work/r1_traj.jsonl /root/work/own_traces_s100.jsonl; do [ -s $f ] && HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token) hf upload baya1116/hypernet-sp-distill $f pooler_distill/chatsft/teach/$(basename $f) >/dev/null 2>&1; done
  rm -rf /root/online_mtg1 /root/online_mtg2 /root/sft /root/wiki_store6 /root/wiki_store_ft1 /root/reeval_hf_g14m /root/teach1/state.pt /root/probe_s100/latest.safetensors
  rm -f /root/gptq_state_gq14.pt /root/pooler_sft_q14.safetensors /root/pooler_sft_q14b.safetensors /root/pooler_sft_q14c.safetensors /root/pooler_sft_q14d.safetensors /root/pooler_sft_q14g.safetensors /root/pooler_eval_q14c.safetensors /root/pooler_mem8.safetensors
  echo "GDISK_AFTER $(df -h /root | tail -1)"
  du -xsh /root/* 2>/dev/null | sort -rh | head -12 | tr '\n' ';'; echo
fi
# ---- r1smoke (2026-10-04 19:25 JST; again 19:40 after R1 searched for answers it already knew - such trajectories are dropped now): r1_traj.py on 6 training items now (API only, CPU) so the format and the
# verification are checked before the probe hands it the real set.
if [ ! -e /root/.r1smoke2 ]; then touch /root/.r1smoke2
  ( cd /root/work; python3 -c "
import json, random
it=[json.loads(l) for l in open('/root/work/mtg2_items.jsonl')]; random.Random(1).shuffle(it)
open('/root/work/r1smoke_in.jsonl','w').write(''.join(json.dumps({**x,'pass':0},ensure_ascii=False)+'\n' for x in it[:6]))"
    rm -f /root/work/r1smoke_out.jsonl
    timeout 1800 python3 /root/work/r1_traj.py --probe /root/work/r1smoke_in.jsonl --out /root/work/r1smoke_out.jsonl --workers 6 2>&1 | grep -E "^\[r1\]|R1_TRAJ|Error|Traceback" | tail -3
    python3 -c "
import json
for l in open('/root/work/r1smoke_out.jsonl'):
    x=json.loads(l); print('[r1smoke]', x['q'][:60], '| gold', x['gold'][:30], '| ns', x['ns'], '|', x['traj'][:500].replace(chr(10),' / '))" 2>&1 | head -4
    echo "R1SMOKE_DONE $(date -u)" ) > /root/r1smoke.log 2>&1 &
fi
# ---- 2026-10-05 10:45 JST: teach1's post-mortem - it searched 13.8 times a question against s100's 3.9 (thinking and
# reply length unchanged): it took on R1's habit of searching on and re-reading pages. mtg4 restarts with the R1
# trajectories of at most 3 searches only (81 of 99) - the habit is in the long ones.
if [ ! -e /root/.mtg4_short ]; then touch /root/.mtg4_short
  pkill -f "mtg4kee[p].sh"; pkill -f "online_loop.py /root/mtg3_s100.safetensors /root/online_mtg4t"; sleep 10
  rm -rf /root/online_mtg4t; rm -f /root/mtg4.log /root/mtg4_run.log
  python3 -c "
import json
r=[json.loads(l) for l in open('/root/work/r1_traj.jsonl')]
k=[x for x in r if x.get('ns',9)<=3]
open('/root/work/r1_traj_short.jsonl','w').write(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in k))
print('MTG4_SHORT', len(k), 'of', len(r), 'R1 trajectories kept (<= 3 searches)')"
fi
# ---- 2026-10-05 12:30 JST (the user: are the evaluation sets really untrained?): every evaluation set against every
# training file on the box (SFT, replay, teacher trajectories, GRPO rollouts and items, Dolphin)
if [ ! -e /root/.leak1 ]; then touch /root/.leak1
  python3 - <<'PYL' > /root/leak.txt 2>&1
import json, glob, os, re
def norm(q): return re.sub(r"\s+", " ", q.strip().lower())
def qs(f):
    out = set()
    try:
        for l in open(f):
            try: d = json.loads(l)
            except Exception: continue
            for k in ("q", "question", "seed"):
                if isinstance(d.get(k), str) and d[k].strip(): out.add(norm(d[k]))
            for t in d.get("turns", []) or []:
                if t.get("q"): out.add(norm(t["q"]))
            for m in (d.get("messages") or d.get("hist") or []):
                if isinstance(m, dict) and m.get("role") == "user" and m.get("content"): out.add(norm(m["content"]))
    except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError): pass
    return out
evals = {"eval300": "/root/work/eval300.jsonl", "dolphin_heldout100": "/root/work/dolphin_heldout100.jsonl", "mt_eval (bridge40+chains)": "/root/work/mt_eval.jsonl",
         "mt_eval_chain": "/root/work/mt_eval_chain.jsonl", "bridge3 (97)": "/root/work/mt_eval_bridge3.jsonl"}
E = {k: qs(v) for k, v in evals.items()}
train_files = [f for f in glob.glob("/root/work/*.jsonl") + glob.glob("/root/hfdl/**/*.jsonl", recursive=True) + glob.glob("/root/online_*/rollouts.jsonl") + glob.glob("/root/online_*/*.jsonl")
               if not any(x in f for x in ("eval300", "heldout", "mt_eval", "dolphinq", "_out_", "st_", "br_", "br2_", "br3_", "mc_", "dl_", "chain", "cache", "probe_s100", "pool_eval"))]
hits = {}
for f in sorted(set(train_files)):
    T = qs(f)
    if not T: continue
    for k, S in E.items():
        n = len(S & T)
        if n: hits.setdefault(k, []).append(f"{f.replace('/root/', '')}:{n}")
for k, S in E.items():
    print(f"LEAK {k}: {len(S)} questions; found in training files: " + ("; ".join(hits.get(k, [])[:8]) or "none"))
PYL
fi
# ---- 2026-10-05 12:20 JST (the user: are unseen questions left?): count the question pools and what training has used
if [ ! -e /root/.qcount1 ]; then touch /root/.qcount1
  python3 - <<'PYQ' > /root/qcount.txt 2>&1
import json, glob, os
def qs(f, key="q"):
    out = set()
    try:
        for l in open(f):
            try:
                d = json.loads(l)
                if d.get(key): out.add(d[key].strip())
                for t in d.get("turns", []): out.add(t["q"].strip())
                if d.get("seed"): out.add(d["seed"].strip())
            except Exception: pass
    except FileNotFoundError: pass
    return out
pools = {"selfq_all": qs("/root/work/selfq_all.jsonl"), "corpus_box_final": qs("/root/work/corpus_box_final.jsonl")}
held = qs("/root/work/eval300.jsonl") | qs("/root/work/mt_eval.jsonl") | qs("/root/work/mt_eval_chain.jsonl") | qs("/root/work/mt_eval_bridge3.jsonl")
used = set(); src = {}
for f in glob.glob("/root/online_*/rollouts.jsonl") + glob.glob("/root/hfdl/**/rollouts*.jsonl", recursive=True) + ["/root/work/mtg2_items.jsonl", "/root/work/mtg_items.jsonl", "/root/work/mt_train.jsonl", "/root/work/mt_train_bridge.jsonl", "/root/work/probe_items.jsonl"]:
    u = qs(f); src[f] = len(u); used |= u
for k, P in pools.items():
    print(f"QCOUNT {k}: {len(P)} questions, held-out {len(P & held)}, seen in training files {len(P & used)}, unseen {len(P - held - used)}")
print("QCOUNT sources: " + "; ".join(f"{os.path.basename(os.path.dirname(f)) or f}={n}" for f, n in sorted(src.items(), key=lambda x: -x[1])[:12]))
PYQ
fi
# ---- mtg4 (2026-10-05 10:50 JST, the user: the R1 trajectories go INTO the GRPO, not a separate supervised pass -
# teach1, two epochs of plain SFT on the 99, fell to 39/100 vs s100's 54). From mtg3 step 100 on the items that carry
# signal: those s100 solves sometimes (1-3 of 4 in the probe, 160) and the never-solved ones R1 has a verified
# trajectory for (99); a search group none of whose samples passes also takes one supervised step on that question's
# R1 trajectory at half weight (--demo-on-fail 0.5). One step in four Dolphin reasoning, rates 5e-6, 80 steps.
# Gate: single-turn shard 0 (s100 54/100); the full screens only when it holds (>= 52).
if [ -s /root/work/r1_traj_short.jsonl ] && ! pgrep -f "mtg4kee[p].sh" >/dev/null && ! grep -q "MTG4_JOB_DONE" /root/mtg4.log 2>/dev/null; then
  cat > /root/mtg4keep.sh <<'M4'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; OUT=/root/online_mtg4t; S0=/root/mtg3_s100.safetensors
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
python3 - <<'PYI'
import json
demo = {json.loads(l)["q"].strip() for l in open("/root/work/r1_traj_short.jsonl")}
out = []
for l in open("/root/work/probe_s100.jsonl"):
    r = json.loads(l)
    if 1 <= r["pass"] <= 3 or (r["pass"] == 0 and r["q"].strip() in demo): out.append({"q": r["q"], "gold": r["gold"], "hist": r.get("hist") or []})
open("/root/work/mtg4_items.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
print(f"[mtg4] {len(out)} items ({sum(1 for x in out if x['q'].strip() in demo)} with an R1 trajectory)", flush=True)
PYI
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
echo "[mtg4] start $(date -u +%H:%M); $(df -h /root | tail -1 | awk '{print $4}') free"
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OAI_KEY=$(cat /root/.oai 2>/dev/null) DSK_KEY=$(cat /root/.dsk 2>/dev/null) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /root/work/online_loop.py $S0 $OUT --pooler none --lora-layers all --lora-rank 16 --save-lora-only 1 \
  --mt-items /root/work/mtg4_items.jsonl --heldout /root/work/eval300.jsonl --reason /root/work/dolphin_rft.jsonl --reason-every 4 --reason-g 8 \
  --search-demo /root/work/r1_traj_short.jsonl --demo-on-fail 0.5 \
  --steps 80 --save-every 40 --lr 5e-6 --search-lr 5e-6 --search-temp 0.9 --search-gen 2000 --temp 0.6 --gen 3000 --budget 900 --maxsrch 7 --stop eos \
  --judge-api openai --judge-model gpt-5-nano --w-talk 0.5 --dolphin-min 0 --adv-std 1 --pg-norm mean --kl 0 \
  --guard 1 --guard-steps 20 > /root/mtg4_run.log 2>&1
grep -E "^\[data\]|ONLINE_|Error|Traceback" /root/mtg4_run.log | tail -4 | cut -c1-250
S=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null); echo "[mtg4] stopped at step ${S:-0} $(date -u +%H:%M)"
[ -s $OUT/latest.safetensors ] || { echo "MTG4_JOB_DONE no checkpoint"; exit 1; }
CK=/root/mtg4_s${S}.safetensors; cp $OUT/latest.safetensors $CK; hf upload $R $CK pooler_distill/chatsft/multiturn/$(basename $CK) >/dev/null 2>&1
hf upload $R $OUT/rollouts.jsonl pooler_distill/chatsft/multiturn/mtg4_rollouts.jsonl >/dev/null 2>&1
rm -f $OUT/*.pt $OUT/good.safetensors
env $ENV python3 /root/work/pool_eval.py $CK /root/work/ev_0.jsonl /root/work/mtg4st_0.jsonl --n 100 $EVARGS --tag "[mtg4st0]" > /root/mtg4st_0.log 2>&1
c0=$(python3 -c "import json;print(sum(bool(json.loads(l).get('correct')) for l in open('/root/work/mtg4st_0.jsonl')))" 2>/dev/null || echo 0)
echo "[mtg4] single shard 0: $c0/100 (s100 54, teach1 39)"
[ "$c0" -ge 52 ] || { echo "MTG4_JOB_DONE below the gate"; exit 0; }
SN=$c0; for i in 1 2; do env $ENV python3 /root/work/pool_eval.py $CK /root/work/ev_$i.jsonl /root/work/mtg4st_$i.jsonl --n 100 $EVARGS --tag "[mtg4st$i]" > /root/mtg4st_$i.log 2>&1
  c=$(python3 -c "import json;print(sum(bool(json.loads(l).get('correct')) for l in open('/root/work/mtg4st_$i.jsonl')))" 2>/dev/null || echo 0); SN=$((SN + c)); echo "[mtg4] single shard $i: $c/100"; done
echo "[mtg4] single-turn: $SN/300 (s100 160)"
for b in "br3:/root/work/mt_eval_bridge3.jsonl:97 (s100 43/31)" "br:/root/work/mt_eval_bridge.jsonl:40 (s100 23/19)"; do T=${b%%:*}; r=${b#*:}; F=${r%%:*}; NOTE=${r#*:}
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl /root/work/${T}_mtg4.jsonl --multiturn $F --mt-mode win --n 1000 $EVARGS --tag "[$T-mtg4]" > /root/${T}_mtg4.log 2>&1
  echo "[mtg4] $T (win): $(python3 -c "
import json; r=[json.loads(l) for l in open('/root/work/${T}_mtg4.jsonl')]
print('turn 1', sum(x['correct'] for x in r if x['turn']==0), '/', sum(1 for x in r if x['turn']==0), ', follow-up', sum(x['correct'] for x in r if x['turn']==1), '/', sum(1 for x in r if x['turn']==1))" 2>&1 | tail -1) (s100 on $NOTE)"
done
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $CK /root/evalrun_dl_mtg4 \
  --questions /root/work/eval300.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl \
  --b 12 --gen 7000 --budget 2400 --temp 0.6 --maxsrch 7 --pooler none --lora-rank 16 --lora-layers all --stop eos --loop-break answer \
  --eval-file /root/work/dolphinq.jsonl --eval-out /root/work/dl_mtg4.jsonl > /root/dl_mtg4.log 2>&1
[ -s /root/dl_judge.py ] && OAI_KEY=$(cat /root/.oai 2>/dev/null) python3 /root/dl_judge.py /root/work/dl_mtg4.jsonl mtg4 | sed 's/\[mix0\]/[mtg4]/'
echo "[mtg4] (s100: Dolphin 56)"
echo "MTG4_JOB_DONE $(date -u)"
M4
  setsid nohup bash -c 'bash /root/mtg4keep.sh 2>&1 | tee -a /root/mtg4.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MTG4_LAUNCHED $(date -u)"
fi
# ---- s100q (2026-10-05 17:00 JST, the user: s100 to 4 bits for the app, measured, then GRPO again). mtg3 step 100 is a
# LoRA (rank 16, all layers) on the GPTQ model's dequantized weights (gptq_hf_gq14): folded in (build_merged), GPTQ onto
# the app's grid again (gptq.py, the g14 calibration traces plus s100's own verified traces), packed and checked
# (checkmlx), published as chatsft/s100_mlx4g; then the s100 screens on the 4-bit: single-turn 300, bridge3 / bridge40
# follow-ups (win), Dolphin held-out 100. s100 in float: 160/300, br3 follow-up 31/97, br40 follow-up 19/40, Dolphin 56.
if [ -s /root/gptq_hf_gq14/model.safetensors ] && grep -q "MTG4_JOB_DONE" /root/mtg4.log 2>/dev/null && ! pgrep -f "s100qkee[p].sh" >/dev/null && ! grep -q "S100Q_JOB_DONE" /root/s100q.log 2>/dev/null; then
  cat > /root/s100qkeep.sh <<'SQ'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
CK=/root/mtg3_s100.safetensors; HFM=/root/s100m_hf; PCK=/root/s100m_pooler.safetensors; HF=/root/gptq_hf_s100; MLX=/root/gptq_mlx4_s100
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
ENV="SP_BASE=$HF SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
rm -rf /root/evalrun_* /root/online_mtg4t/*.pt
echo "[s100q] start $(date -u +%H:%M); $(df -h /root | tail -1 | awk '{print $4}') free"
[ -s $CK ] || hf download $R pooler_distill/chatsft/multiturn/mtg3_s100.safetensors --local-dir /root/hfdl >/dev/null 2>&1 && [ -s $CK ] || cp /root/hfdl/pooler_distill/chatsft/multiturn/mtg3_s100.safetensors $CK
if [ ! -s $HF/model.safetensors ]; then
  if [ ! -s $HFM/model.safetensors ]; then
    SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 python3 /root/work/build_merged.py $CK $HFM $PCK 16 all 2>&1 | grep -E "^\[merge\]|MERGE_DONE|Error|assert|unexpected" | tail -4
    [ -s $HFM/model.safetensors ] && [ -s $PCK ] || { echo "S100Q_JOB_DONE merge failed"; exit 1; }
  fi
  # calibration: the g14 traces gptq.py was tuned on, plus s100's own verified traces (the chat-era, multi-turn format)
  python3 - <<'PYC'
import json, random
rows = [l for l in open("/root/work/qcal_q14.jsonl") if l.strip()]
own = 0
try:
    for l in open("/root/work/own_traces_s100.jsonl"):
        d = json.loads(l); t = d.get("text") or d.get("traj")
        if t: rows.append(json.dumps({"text": t}, ensure_ascii=False) + "\n"); own += 1
except FileNotFoundError: pass
random.Random(0).shuffle(rows); open("/root/work/qcal_s100.jsonl", "w").write("".join(rows))
print(f"[s100q] calibration: {len(rows) - own} g14 traces + {own} s100 traces", flush=True)
PYC
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/gptq.py --base $HFM --data /root/work/qcal_s100.jsonl --out-hf $HF --out-mlx $MLX --state /root/gptq_state_s100.pt > /root/gptq_s100.log 2>&1
  grep -q GPTQ_DONE /root/gptq_s100.log || { echo "S100Q_JOB_DONE gptq failed: $(grep -E 'Error|error' /root/gptq_s100.log | tail -1 | cut -c1-200)"; exit 1; }
  echo "[s100q] $(grep -E '^\[gptq\] (layer 27|lm_head)' /root/gptq_s100.log | tail -1 | cut -c1-200)"
fi
python3 /root/work/checkmlx.py $MLX $HF 2>&1 | tail -3 | tee /root/checkmlx_s100.txt
grep -q MLX_CHECK_OK /root/checkmlx_s100.txt || { echo "S100Q_JOB_DONE pack check failed"; exit 1; }
# the pooler: the same tensors as g14's (whose 4-bit GPTQ file the app already ships) or not
hf download $R pooler_distill/chatsft/g14_mlx4g/pooler.safetensors --local-dir /root/hfdl >/dev/null 2>&1
python3 - $PCK /root/hfdl/pooler_distill/chatsft/g14_mlx4g/pooler.safetensors <<'PYP'
import sys, torch
from safetensors.torch import load_file
a, b = load_file(sys.argv[1]), load_file(sys.argv[2])
strip = lambda d: {k[len("pooler."):] if k.startswith("pooler.") else k: v for k, v in d.items()}
a, b = strip(a), strip(b); same = set(a) == set(b)
d = max(float((a[k].float() - b[k].float()).abs().max()) for k in a if k in b) if same else -1
print(f"[s100q] pooler vs g14's: {len(a)} / {len(b)} tensors, max |diff| {d:.3g} -> {'identical: the shipped pooler_4bit applies' if same and d == 0 else 'DIFFERENT: needs its own 4-bit pooler'}", flush=True)
PYP
cp $PCK $MLX/pooler.safetensors
for try in 1 2 3; do hf upload $R $MLX pooler_distill/chatsft/s100_mlx4g >/dev/null 2>&1 && break; sleep 30; done
hf upload $R /root/gptq_s100.log pooler_distill/chatsft/logs/gptq_s100.log >/dev/null 2>&1
echo "[s100q] uploaded chatsft/s100_mlx4g $(date -u +%H:%M)"
rm -rf $HFM; rm -f /root/gptq_state_s100.pt
SN=0; for i in 0 1 2; do env $ENV python3 /root/work/pool_eval.py $PCK /root/work/ev_$i.jsonl /root/work/s100q_$i.jsonl --n 100 $EVARGS --tag "[s100q$i]" > /root/s100q_$i.log 2>&1
  c=$(python3 -c "import json;print(sum(bool(json.loads(l).get('correct')) for l in open('/root/work/s100q_$i.jsonl')))" 2>/dev/null || echo 0); SN=$((SN + c))
  echo "[s100q] single shard $i: $c/100 $(grep -h -m1 'pooler restored\|WARNING: no pooler' /root/s100q_$i.log)"; done
echo "[s100q] single-turn: $SN/300 (s100 float 160)"
for b in "br3:/root/work/mt_eval_bridge3.jsonl:97 (s100 float 43/31)" "br:/root/work/mt_eval_bridge.jsonl:40 (s100 float 23/19)"; do T=${b%%:*}; r=${b#*:}; F=${r%%:*}; NOTE=${r#*:}
  env $ENV python3 /root/work/pool_eval.py $PCK /root/work/eval300.jsonl /root/work/${T}_s100q.jsonl --multiturn $F --mt-mode win --n 1000 $EVARGS --tag "[$T-s100q]" > /root/${T}_s100q.log 2>&1
  echo "[s100q] $T (win): $(python3 -c "
import json; r=[json.loads(l) for l in open('/root/work/${T}_s100q.jsonl')]
print('turn 1', sum(x['correct'] for x in r if x['turn']==0), '/', sum(1 for x in r if x['turn']==0), ', follow-up', sum(x['correct'] for x in r if x['turn']==1), '/', sum(1 for x in r if x['turn']==1))" 2>&1 | tail -1) (on $NOTE)"
done
env SP_BASE=$HF SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $HF /root/evalrun_dl_s100q \
  --questions /root/work/eval300.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl --pooler-init $PCK \
  --b 12 --gen 7000 --budget 2400 --temp 0.6 --maxsrch 7 --pooler none --lora-rank 16 --lora-layers all --stop eos --loop-break answer \
  --eval-file /root/work/dolphinq.jsonl --eval-out /root/work/dl_s100q.jsonl > /root/dl_s100q.log 2>&1
grep -m1 "pooler <-" /root/dl_s100q.log
[ -s /root/dl_judge.py ] && OAI_KEY=$(cat /root/.oai 2>/dev/null) python3 /root/dl_judge.py /root/work/dl_s100q.jsonl s100q | sed 's/\[mix0\]/[s100q]/'
echo "[s100q] (s100 float: Dolphin 56)"
for f in /root/work/s100q_*.jsonl /root/work/br3_s100q.jsonl /root/work/br_s100q.jsonl /root/work/dl_s100q.jsonl; do [ -s $f ] && hf upload $R $f pooler_distill/chatsft/s100q/$(basename $f) >/dev/null 2>&1; done
echo "S100Q_JOB_DONE $(date -u)"
SQ
  setsid nohup bash -c 'bash /root/s100qkeep.sh 2>&1 | tee -a /root/s100q.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "S100Q_LAUNCHED $(date -u)"
fi
# (2026-10-06 03:45 JST: the box runs this file only when it changes; the 4-bit screens ended at 01:25 with no change
# after them, so mtg5 waited - this edit is what starts it.)
# ---- mtg5 (2026-10-05 17:30 JST, the user: GRPO again after the 4-bit, "there is something there"). What mtg4 showed: in
# 80 steps only 60 search groups ran, one question each, so 182 of its 242 items were never seen; 45% of the search
# rollouts went to never-solved items with an R1 demo, which passed 8-11% (little signal for the compute); the items
# solved 1-3 of 4 by s100 ran 61/62/39/59% against 54/46/53/59% expected. Changes: (1) the policy trains on the app's
# own weights - the 4-bit s100 (gptq_hf_s100) with a fresh LoRA, so what is trained is what ships; (2) items = mtg4's
# unseen ones, the never-solved-with-demo share capped at a quarter; (3) 160 steps, a copy every 40, each copy screened
# on single-turn shard 0 against the 4-bit s100's own shard 0; the best one, if above it, gets the full screens.
# Starts after the 4-bit screens; if the 4-bit s100 lost more than 10 of 300 single-turn, it trains on gq14 + s100's
# LoRA as mtg4 did.
if [ -s /root/work/mtg4_items.jsonl ] && grep -q "S100Q_JOB_DONE" /root/s100q.log 2>/dev/null && ! pgrep -f "mtg5kee[p].sh" >/dev/null && ! grep -q "MTG5_JOB_DONE" /root/mtg5.log 2>/dev/null; then
  cat > /root/mtg5keep.sh <<'M5'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; OUT=/root/online_mtg5
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
SQ=$(grep -oE "single-turn: [0-9]+/300" /root/s100q.log | tail -1 | grep -oE "[0-9]+/" | tr -d /); SQ0=$(grep -oE "single shard 0: [0-9]+" /root/s100q.log | tail -1 | grep -oE "[0-9]+$")
if [ -s /root/gptq_hf_s100/model.safetensors ] && [ "${SQ:-0}" -ge 150 ]; then
  BASE=/root/gptq_hf_s100; INIT="$BASE --pooler-init /root/s100m_pooler.safetensors"; REF0=${SQ0:-0}; echo "[mtg5] on the 4-bit s100 (single-turn $SQ/300, shard 0 $SQ0)"
else
  BASE=/root/gptq_hf_gq14; INIT=/root/mtg3_s100.safetensors; REF0=54; echo "[mtg5] the 4-bit s100 read ${SQ:-?}/300: training on gq14 + s100's LoRA as mtg4 did"
fi
ENV="SP_BASE=$BASE SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
python3 - <<'PYI'
import json, random
seen = set()
try:
    seen = {json.loads(l)["q"].strip() for l in open("/root/online_mtg4t/rollouts.jsonl") if json.loads(l).get("kind") == "search"}
except FileNotFoundError: pass
demo = {json.loads(l)["q"].strip() for l in open("/root/work/r1_traj_short.jsonl")}
it = [json.loads(l) for l in open("/root/work/mtg4_items.jsonl")]
new = [x for x in it if x["q"].strip() not in seen]
mixed = [x for x in new if x["q"].strip() not in demo]; dm = [x for x in new if x["q"].strip() in demo]
random.Random(5).shuffle(dm); dm = dm[:len(mixed) // 3]
out = mixed + dm
open("/root/work/mtg5_items.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
print(f"[mtg5] {len(out)} items: {len(mixed)} solved 1-3 of 4, {len(dm)} never solved with an R1 demo ({len(seen)} seen by mtg4 left out)", flush=True)
PYI
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
rm -rf /root/evalrun_*
echo "[mtg5] start $(date -u +%H:%M); $(df -h /root | tail -1 | awk '{print $4}') free"
( last=0; while sleep 60; do s=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null || echo 0)
    if [ "$s" != "$last" ] && [ $((s % 40)) -eq 0 ] && [ "$s" -gt 0 ] && [ ! -s /root/mtg5_s$s.safetensors ]; then sleep 20; cp $OUT/latest.safetensors /root/mtg5_s$s.safetensors; echo "[mtg5] copy at step $s"; fi; last=$s; done ) & CP=$!
env SP_BASE=$BASE SP_NOSYS=1 SP_EPISODIC=1 OAI_KEY=$(cat /root/.oai 2>/dev/null) DSK_KEY=$(cat /root/.dsk 2>/dev/null) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /root/work/online_loop.py $INIT $OUT --pooler none --lora-layers all --lora-rank 16 --save-lora-only 1 \
  --mt-items /root/work/mtg5_items.jsonl --heldout /root/work/eval300.jsonl --reason /root/work/dolphin_rft.jsonl --reason-every 4 --reason-g 8 \
  --search-demo /root/work/r1_traj_short.jsonl --demo-on-fail 0.5 \
  --steps 160 --save-every 40 --lr 5e-6 --search-lr 5e-6 --search-temp 0.9 --search-gen 2000 --temp 0.6 --gen 3000 --budget 900 --maxsrch 7 --stop eos \
  --judge-api openai --judge-model gpt-5-nano --w-talk 0.5 --dolphin-min 0 --adv-std 1 --pg-norm mean --kl 0 \
  --guard 1 --guard-steps 20 > /root/mtg5_run.log 2>&1
sleep 90; kill $CP 2>/dev/null
grep -E "^\[data\]|^\[init\]|ONLINE_|Error|Traceback" /root/mtg5_run.log | tail -5 | cut -c1-250
S=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null); echo "[mtg5] stopped at step ${S:-0} $(date -u +%H:%M)"
[ -s $OUT/latest.safetensors ] && [ ! -s /root/mtg5_s$S.safetensors ] && cp $OUT/latest.safetensors /root/mtg5_s$S.safetensors
hf upload $R $OUT/rollouts.jsonl pooler_distill/chatsft/multiturn/mtg5_rollouts.jsonl >/dev/null 2>&1
rm -f $OUT/*.pt $OUT/good.safetensors
BEST=; BC=$REF0
for CK in $(ls /root/mtg5_s*.safetensors 2>/dev/null | sort -V); do
  T=$(basename $CK .safetensors); hf upload $R $CK pooler_distill/chatsft/multiturn/$T.safetensors >/dev/null 2>&1
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/ev_0.jsonl /root/work/${T}st_0.jsonl --n 100 $EVARGS --tag "[${T}st0]" > /root/${T}st_0.log 2>&1
  c=$(python3 -c "import json;print(sum(bool(json.loads(l).get('correct')) for l in open('/root/work/${T}st_0.jsonl')))" 2>/dev/null || echo 0)
  echo "[mtg5] $T single shard 0: $c/100 (reference $REF0)"; [ "$c" -gt "$BC" ] && { BC=$c; BEST=$CK; }
done
[ -n "$BEST" ] || { echo "MTG5_JOB_DONE no copy above the reference shard 0 ($REF0)"; exit 0; }
T=$(basename $BEST .safetensors); echo "[mtg5] full screens for $T"
SN=$BC; for i in 1 2; do env $ENV python3 /root/work/pool_eval.py $BEST /root/work/ev_$i.jsonl /root/work/${T}st_$i.jsonl --n 100 $EVARGS --tag "[${T}st$i]" > /root/${T}st_$i.log 2>&1
  c=$(python3 -c "import json;print(sum(bool(json.loads(l).get('correct')) for l in open('/root/work/${T}st_$i.jsonl')))" 2>/dev/null || echo 0); SN=$((SN + c)); echo "[mtg5] single shard $i: $c/100"; done
echo "[mtg5] $T single-turn: $SN/300 (4-bit s100 ${SQ:-?})"
for b in "br3:/root/work/mt_eval_bridge3.jsonl" "br:/root/work/mt_eval_bridge.jsonl"; do X=${b%%:*}; F=${b#*:}
  env $ENV python3 /root/work/pool_eval.py $BEST /root/work/eval300.jsonl /root/work/${X}_$T.jsonl --multiturn $F --mt-mode win --n 1000 $EVARGS --tag "[$X-$T]" > /root/${X}_$T.log 2>&1
  echo "[mtg5] $X (win): $(python3 -c "
import json; r=[json.loads(l) for l in open('/root/work/${X}_$T.jsonl')]
print('turn 1', sum(x['correct'] for x in r if x['turn']==0), '/', sum(1 for x in r if x['turn']==0), ', follow-up', sum(x['correct'] for x in r if x['turn']==1), '/', sum(1 for x in r if x['turn']==1))" 2>&1 | tail -1) (4-bit s100: $(grep -E "\[s100q\] $X \(win\)" /root/s100q.log | tail -1 | cut -d: -f2- | cut -c1-60))"
done
env SP_BASE=$BASE SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $BEST /root/evalrun_dl_mtg5 \
  --questions /root/work/eval300.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl \
  --b 12 --gen 7000 --budget 2400 --temp 0.6 --maxsrch 7 --pooler none --lora-rank 16 --lora-layers all --stop eos --loop-break answer \
  --eval-file /root/work/dolphinq.jsonl --eval-out /root/work/dl_$T.jsonl > /root/dl_$T.log 2>&1
[ -s /root/dl_judge.py ] && OAI_KEY=$(cat /root/.oai 2>/dev/null) python3 /root/dl_judge.py /root/work/dl_$T.jsonl $T | sed "s/\[mix0\]/[mtg5]/"
echo "[mtg5] (4-bit s100 Dolphin: $(grep -iE "^\[s100q\].*(dolphin|judge)" /root/s100q.log | grep -v float | tail -1 | cut -c1-100))"
echo "MTG5_JOB_DONE $(date -u)"
M5
  setsid nohup bash -c 'bash /root/mtg5keep.sh 2>&1 | tee -a /root/mtg5.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MTG5_LAUNCHED $(date -u)"
fi
# ---- 2026-10-05 10:35 JST: teach1 (R1 demos, 2 epochs at 2e-5) fell to 39/100 on single-turn shard 0 (s100 54):
# clearly worse, stop its remaining screens; print what changed (searches, landed, grounded, reply length) for the
# post-mortem, against s100 on the same 100 questions.
if [ ! -e /root/.teach2_stop ]; then touch /root/.teach2_stop
  pkill -f "teach2kee[p].sh"; pkill -f "pool_eval.py /root/teach1_s50"; pkill -f "online_loop.py /root/teach1_s50"; sleep 5
  echo "TEACH2_JOB_DONE stopped after shard 0 (39/100) $(date -u)" >> /root/teach2.log
  echo "TEACH1_DIAG $(grep -h EVAL_DONE /root/teach1st_0.log | tail -1 | cut -c1-200)"
  echo "S100_DIAG $(grep -h EVAL_DONE /root/mtg3s100stfull0.log | tail -1 | cut -c1-200)"
  python3 - <<'PYD'
import json, statistics as st
def load(f):
    try: return {json.loads(l)["q"]: json.loads(l) for l in open(f)}
    except Exception as e: return {}
a = load("/root/work/mtg3s100st_0.jsonl"); b = load("/root/work/teach1st_0.jsonl")
ks = [k for k in b if k in a]
def stats(d):
    rs = [d[k] for k in ks]
    rl = [len((r.get("reply") or r.get("text", "").split("</think>")[-1]).split()) for r in rs]
    tl = [len(r.get("text", "").split("</think>")[0].split()) for r in rs]
    return f"correct {sum(bool(r.get('correct')) for r in rs)}, searches {st.mean(r.get('ns', 0) for r in rs):.2f}, thinking words {st.median(tl)}, reply words {st.median(rl)}"
print(f"TEACH1_CMP on {len(ks)} questions | s100: {stats(a)} | teach1: {stats(b)}")
lost = [k for k in ks if a[k].get("correct") and not b[k].get("correct")]
for k in lost[:3]:
    print("TEACH1_LOST", k[:70], "|", (b[k].get("text", "")[:300]).replace("\n", " / "))
PYD
fi
# ---- teach2 (2026-10-05 08:20 JST): teach's probe and R1 stages finished (600 probed: 316 never solved, 59/44/57
# sometimes, 124 always; R1 verified 99 trajectories, 81 with history; 284 own traces) but the supervised step died at
# its first save (disk full) and the screens with it. Rerun stage 3 and 4 from the saved files once the disk is freed:
# 50 steps (two epochs over the 99), one save at the end.
if [ -e /root/.diskfree_g1 ] && ! pgrep -f "teach2kee[p].sh" >/dev/null && ! grep -q "TEACH2_JOB_DONE" /root/teach2.log 2>/dev/null; then
  cat > /root/teach2keep.sh <<'TK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; S0=/root/mtg3_s100.safetensors
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
OL="--pooler none --lora-layers all --lora-rank 16 --save-lora-only 1 --heldout /root/work/eval300.jsonl --maxsrch 7 --stop eos --judge 0 --reason-stub 1"
FREE=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9); echo "[teach2] $FREE GB free $(date -u +%H:%M)"
[ "${FREE:-0}" -ge 6 ] || { echo "TEACH2_JOB_DONE disk still full"; exit 1; }
[ -s /root/work/r1_traj.jsonl ] && [ -s /root/work/own_traces_s100.jsonl ] || { echo "TEACH2_JOB_DONE data missing"; exit 1; }
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
rm -rf /root/teach1
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $S0 /root/teach1 $OL \
  --mt-items /root/work/probe_items.jsonl --demo-sft /root/work/r1_traj.jsonl --demo-per-step 4 --replay /root/work/own_traces_s100.jsonl --replay-per-step 2 \
  --steps 50 --save-every 50 --lr 2e-5 > /root/teach1_run.log 2>&1
grep -E "^\[demo-sft\]|ONLINE_|Error|Traceback" /root/teach1_run.log | tail -3 | cut -c1-200; grep -E "^\[step (1|25|50)\]" /root/teach1_run.log | cut -c1-200
[ -s /root/teach1/latest.safetensors ] || { echo "TEACH2_JOB_DONE no checkpoint"; exit 1; }
CK=/root/teach1_s50.safetensors; cp /root/teach1/latest.safetensors $CK; rm -f /root/teach1/*.pt /root/teach1/step*.safetensors
hf upload $R $CK pooler_distill/chatsft/teach/teach1_s50.safetensors >/dev/null 2>&1
SN=0; for i in 0 1 2; do env $ENV python3 /root/work/pool_eval.py $CK /root/work/ev_$i.jsonl /root/work/teach1st_$i.jsonl --n 100 $EVARGS --tag "[teach1st$i]" > /root/teach1st_$i.log 2>&1
  c=$(python3 -c "import json;print(sum(bool(json.loads(l).get('correct')) for l in open('/root/work/teach1st_$i.jsonl')))" 2>/dev/null || echo 0); SN=$((SN + c)); echo "[teach2] single shard $i: $c/100 (s100 54/54/52)"; done
echo "[teach2] single-turn: $SN/300 (s100 160, s300 144)"
for b in "br3:/root/work/mt_eval_bridge3.jsonl:97 (s100 43/31)" "br:/root/work/mt_eval_bridge.jsonl:40 (s100 23/19)"; do T=${b%%:*}; r=${b#*:}; F=${r%%:*}; NOTE=${r#*:}
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl /root/work/${T}_teach1.jsonl --multiturn $F --mt-mode win --n 1000 $EVARGS --tag "[$T-teach1]" > /root/${T}_teach1.log 2>&1
  echo "[teach2] $T (win): $(python3 -c "
import json; r=[json.loads(l) for l in open('/root/work/${T}_teach1.jsonl')]
print('turn 1', sum(x['correct'] for x in r if x['turn']==0), '/', sum(1 for x in r if x['turn']==0), ', follow-up', sum(x['correct'] for x in r if x['turn']==1), '/', sum(1 for x in r if x['turn']==1))" 2>&1 | tail -1) (s100 on $NOTE)"
done
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $CK /root/evalrun_dl_teach1 \
  --questions /root/work/eval300.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl \
  --b 12 --gen 7000 --budget 2400 --temp 0.6 --maxsrch 7 --pooler none --lora-rank 16 --lora-layers all --stop eos --loop-break answer \
  --eval-file /root/work/dolphinq.jsonl --eval-out /root/work/dl_teach1.jsonl > /root/dl_teach1.log 2>&1
[ -s /root/dl_judge.py ] && OAI_KEY=$(cat /root/.oai 2>/dev/null) python3 /root/dl_judge.py /root/work/dl_teach1.jsonl teach1 | sed 's/\[mix0\]/[teach2]/'
echo "[teach2] (s100: Dolphin 56)"
echo "TEACH2_JOB_DONE $(date -u)"
TK
  setsid nohup bash -c 'bash /root/teach2keep.sh 2>&1 | tee -a /root/teach2.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "TEACH2_LAUNCHED $(date -u)"
fi
# ---- teach (2026-10-04 19:40 JST, the user: supervised from R1 for what GRPO cannot reach; DeepSeek topped up). From
# mtg3 step 100 (s100): (1) probe 600 training items (every bridge follow-up + the rest at random), 4 samples each at
# the training temperature, gold string only; (2) R1 solves the items s100 never solved, in the student's own search
# environment, kept when verified (gold in the reply and on a served chunk); (3) supervised steps on those teacher
# trajectories (searching trained, pages and reply masked) beside s100's own verified traces from the probe, two
# epochs over the teacher set at 2e-5; (4) screens against s100: single-turn 300, bridge 40 + bridge3 97 under win,
# Dolphin 100.
if ! pgrep -f "teachkee[p].sh" >/dev/null && ! grep -q "TEACH_JOB_DONE" /root/teach.log 2>/dev/null; then
  cat > /root/teachkeep.sh <<'TK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; S0=/root/mtg3_s100.safetensors
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
OL="--pooler none --lora-layers all --lora-rank 16 --save-lora-only 1 --heldout /root/work/eval300.jsonl --maxsrch 7 --stop eos --judge 0 --reason-stub 1"
until grep -q "MOREQ4_JOB_DONE" /root/moreq4.log 2>/dev/null; do sleep 60; done
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
# (1) probe
python3 - <<'PYP'
import json, random
it = [json.loads(l) for l in open("/root/work/mtg2_items.jsonl")]
br = set()
for l in open("/root/work/mt_train_bridge.jsonl"):
    d = json.loads(l); br.add(d["turns"][1]["q"].strip())
fol = [x for x in it if x["q"].strip() in br and x.get("hist")]; rest = [x for x in it if x not in fol]
random.Random(9).shuffle(rest); sel = fol + rest[:600 - len(fol)]; random.Random(10).shuffle(sel)
open("/root/work/probe_items.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in sel))
print(f"[teach] probe items: {len(sel)} ({len(fol)} bridge follow-ups)", flush=True)
PYP
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $S0 /root/probe_s100 $OL \
  --mt-items /root/work/probe_items.jsonl --probe-out /root/work/probe_s100.jsonl --probe-g 4 --b 12 --temp 0.9 --gen 2000 --budget 900 > /root/probe_s100.log 2>&1
grep -E "PROBE_DONE|Error|Traceback" /root/probe_s100.log | tail -1 | cut -c1-200
python3 - <<'PYS'
import json, collections
r = [json.loads(l) for l in open("/root/work/probe_s100.jsonl")]
c = collections.Counter(x["pass"] for x in r)
print(f"[teach] probe: {len(r)} items, passes of 4: " + " ".join(f"{k}:{c[k]}" for k in range(5)), flush=True)
own = [{"q": x["q"], "hist": x.get("hist") or [], "text": t} for x in r if 1 <= x["pass"] for t in x["texts"][:1]]
open("/root/work/own_traces_s100.jsonl", "w").write("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in own))
print(f"[teach] own verified traces: {len(own)}", flush=True)
PYS
hf upload $R /root/work/probe_s100.jsonl pooler_distill/chatsft/teach/probe_s100.jsonl >/dev/null 2>&1
# (2) R1 on the never-solved items
python3 /root/work/r1_traj.py --probe /root/work/probe_s100.jsonl --out /root/work/r1_traj.jsonl --max-pass 0 --workers 6 2>&1 | grep -E "^\[r1\]|R1_TRAJ_DONE|Error|Traceback" | tail -4
hf upload $R /root/work/r1_traj.jsonl pooler_distill/chatsft/teach/r1_traj.jsonl >/dev/null 2>&1
ND=$(wc -l < /root/work/r1_traj.jsonl 2>/dev/null || echo 0); [ "$ND" -ge 20 ] || { echo "TEACH_JOB_DONE only $ND teacher trajectories"; exit 1; }
python3 -c "
import json; r=[json.loads(l) for l in open('/root/work/r1_traj.jsonl')]
x=r[0]; print('[teach] example:', x['q'][:80], '| gold', x['gold'], '|', x['traj'][:400].replace(chr(10),' / '))"
# (3) supervised steps
ST=$(( (ND * 2 + 3) / 4 ))
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $S0 /root/teach1 $OL \
  --mt-items /root/work/probe_items.jsonl --demo-sft /root/work/r1_traj.jsonl --demo-per-step 4 --replay /root/work/own_traces_s100.jsonl --replay-per-step 2 \
  --steps $ST --save-every 25 --lr 2e-5 > /root/teach1_run.log 2>&1
grep -E "^\[demo-sft\]|ONLINE_|Error|Traceback" /root/teach1_run.log | tail -3 | cut -c1-200; grep -E "^\[step" /root/teach1_run.log | tail -2 | cut -c1-200
[ -s /root/teach1/latest.safetensors ] || { echo "TEACH_JOB_DONE no checkpoint"; exit 1; }
CK=/root/teach1_s$ST.safetensors; cp /root/teach1/latest.safetensors $CK; hf upload $R $CK pooler_distill/chatsft/teach/$(basename $CK) >/dev/null 2>&1
# (4) screens
SN=0; for i in 0 1 2; do env $ENV python3 /root/work/pool_eval.py $CK /root/work/ev_$i.jsonl /root/work/teach1st_$i.jsonl --n 100 $EVARGS --tag "[teach1st$i]" > /root/teach1st_$i.log 2>&1
  c=$(python3 -c "import json;print(sum(bool(json.loads(l).get('correct')) for l in open('/root/work/teach1st_$i.jsonl')))" 2>/dev/null || echo 0); SN=$((SN + c)); echo "[teach] single shard $i: $c/100"; done
echo "[teach] single-turn: $SN/300 (s100 160, s300 144)"
for b in "br:/root/work/mt_eval_bridge.jsonl:40 (s100 23/19)" "br3:/root/work/mt_eval_bridge3.jsonl:97"; do T=${b%%:*}; r=${b#*:}; F=${r%%:*}; NOTE=${r#*:}
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl /root/work/${T}_teach1.jsonl --multiturn $F --mt-mode win --n 1000 $EVARGS --tag "[$T-teach1]" > /root/${T}_teach1.log 2>&1
  echo "[teach] $T (win): $(python3 -c "
import json; r=[json.loads(l) for l in open('/root/work/${T}_teach1.jsonl')]
print('turn 1', sum(x['correct'] for x in r if x['turn']==0), '/', sum(1 for x in r if x['turn']==0), ', follow-up', sum(x['correct'] for x in r if x['turn']==1), '/', sum(1 for x in r if x['turn']==1))") (s100 on $NOTE)"
done
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $CK /root/evalrun_dl_teach1 \
  --questions /root/work/eval300.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl \
  --b 12 --gen 7000 --budget 2400 --temp 0.6 --maxsrch 7 --pooler none --lora-rank 16 --lora-layers all --stop eos --loop-break answer \
  --eval-file /root/work/dolphinq.jsonl --eval-out /root/work/dl_teach1.jsonl > /root/dl_teach1.log 2>&1
[ -s /root/dl_judge.py ] && OAI_KEY=$(cat /root/.oai 2>/dev/null) python3 /root/dl_judge.py /root/work/dl_teach1.jsonl teach1 | sed 's/\[mix0\]/[teach]/'
echo "[teach] (s100: Dolphin 56)"
echo "TEACH_JOB_DONE $(date -u)"
TK
  setsid nohup bash -c 'bash /root/teachkeep.sh 2>&1 | tee -a /root/teach.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "TEACH_LAUNCHED $(date -u)"
fi
# ---- moreq4 (2026-10-04 19:10 JST): moreq3's two bridge3 runs died of CUDA OOM at 14:18 - they started beside the
# single-turn pass (the card is 11.6 GB, two evaluators do not fit). Run them now, one at a time, card otherwise idle.
if ! pgrep -f "moreq4kee[p].sh" >/dev/null && ! grep -q "MOREQ4_JOB_DONE" /root/moreq4.log 2>/dev/null; then
  cat > /root/moreq4keep.sh <<'MQ'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
[ -s /root/work/mt_eval_bridge3.jsonl ] || { echo "MOREQ4_JOB_DONE no dialogues"; exit 1; }
for arm in "s100:/root/mtg3_s100.safetensors" "s300:/root/mtg1_s300.safetensors"; do T=${arm%%:*}; CK=${arm#*:}
  while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
  rm -f /root/work/br3_$T.jsonl
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl /root/work/br3_$T.jsonl --multiturn /root/work/mt_eval_bridge3.jsonl --mt-mode win --n 1000 $EVARGS --tag "[br3-$T]" > /root/br3_$T.log 2>&1
  grep -E "Traceback|Error" /root/br3_$T.log | tail -1 | cut -c1-160
  python3 - /root/work/br3_$T.jsonl $T <<'PY2'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])]
t1 = [x for x in r if x["turn"] == 0]; t2 = [x for x in r if x["turn"] == 1]
print(f"[moreq4] bridge3 (win) {sys.argv[2]}: turn 1 {sum(x['correct'] for x in t1)}/{len(t1)}, follow-up {sum(x['correct'] for x in t2)}/{len(t2)}", flush=True)
PY2
  hf upload $R /root/work/br3_$T.jsonl pooler_distill/chatsft/multiturn/br3_$T.jsonl >/dev/null 2>&1
done
echo "MOREQ4_JOB_DONE $(date -u)"
MQ
  setsid nohup bash -c 'bash /root/moreq4keep.sh 2>&1 | tee -a /root/moreq4.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MOREQ4_LAUNCHED $(date -u)"
fi
# ---- 2026-10-04 17:45 JST: why do DeepSeek chat calls fail (balance answers 200)? One tiny call per model name; status
# code and the error text only (never the key).
if [ ! -e /root/.dsk_chat_check ]; then touch /root/.dsk_chat_check
  for m in deepseek-chat deepseek-reasoner deepseek-flash; do
    out=$(curl -s -m 60 -w ' HTTP%{http_code}' -H "Authorization: Bearer $(tr -d '[:space:]' < /root/.dsk 2>/dev/null)" -H "Content-Type: application/json" \
      -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":20}" https://api.deepseek.com/chat/completions)
    echo "DSK_CHAT $m: $(echo "$out" | tr '\n' ' ' | sed -E 's/"content":"[^"]*"/"content":"..."/g' | cut -c1-220)"
  done
fi
# ---- moreq3 (2026-10-04 13:15 JST): the DeepSeek key answers (balance 200) but its chat calls fail; nano wrote only 25
# dialogues from the 123 eval300 seeds left over, and they are poor ("So, what's that company?") - those leftovers are
# the seeds DeepSeek had already rejected as single-hop when mt_eval was written. Seeds now from selfq_all minus every
# question in any training or evaluation file (the single-turn GRPO saw selfq_all questions, so these follow-ups are
# held out for the dialogue but not for the final fact - the same holds for the training follow-ups, so s100 vs s300
# stays a fair comparison), written by gpt-5-mini, 100 of them. moreq2 is stopped before it evaluates its 25.
if [ ! -e /root/.moreq2_stop ]; then touch /root/.moreq2_stop; pkill -f "moreq2kee[p].sh"; echo "MOREQ2_JOB_DONE replaced by moreq3 $(date -u)" >> /root/moreq2.log; fi
if ! pgrep -f "moreq3kee[p].sh" >/dev/null && ! grep -q "MOREQ3_JOB_DONE" /root/moreq3.log 2>/dev/null; then
  cat > /root/moreq3keep.sh <<'MQ'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
curl -sSf -o /root/work/mt_gen.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/mt_gen.py?$(date +%s)"
grep -q "6000" /root/work/mt_gen.py || { echo "MOREQ3_JOB_DONE stale mt_gen"; exit 1; }
python3 - <<'PYX'
import json
ex = set()
for f in ("/root/work/eval300.jsonl", "/root/work/mt_eval.jsonl", "/root/work/mt_eval_chain.jsonl", "/root/work/mt_eval_bridge2.jsonl", "/root/work/mt_train.jsonl", "/root/work/mt_train_br2.jsonl", "/root/work/mt_train_bridge.jsonl", "/root/work/mtg2_items.jsonl", "/root/work/mtg_items.jsonl"):
    try:
        for l in open(f):
            d = json.loads(l); ex.add((d.get("q") or "").strip()); ex.add((d.get("seed") or "").strip()); ex |= {t["q"].strip() for t in d.get("turns", [])}
            for m in d.get("hist") or []: ex.add((m.get("content") or "").strip())
    except FileNotFoundError: pass
open("/root/work/mt_exclude3.jsonl", "w").write("".join(json.dumps({"q": q}) + "\n" for q in ex if q))
print(f"[moreq3] {len(ex)} questions excluded as seeds")
PYX
python3 /root/work/mt_gen.py --api openai --model gpt-5-mini --seeds /root/work/selfq_all.jsonl --exclude /root/work/mt_exclude3.jsonl --out /root/work/mt_eval_bridge3_raw.jsonl --n-bridge 100 --n-memory 0 --n-switch 0 --seed 33 2>&1 | grep -E "^\[mtgen\]|MTGEN_DONE|Error|Traceback" | cut -c1-200
python3 - <<'PYY'
import json
ex = {json.loads(l)["q"] for l in open("/root/work/mt_exclude3.jsonl")}
out = []
try:
    for l in open("/root/work/mt_eval_bridge3_raw.jsonl"):
        d = json.loads(l)
        if d.get("kind") == "bridge" and len(d.get("turns", [])) == 2 and not any(t["q"].strip() in ex for t in d["turns"]): out.append(d)
except FileNotFoundError: pass
open("/root/work/mt_eval_bridge3.jsonl", "w").write("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in out))
print(f"[moreq3] bridge3: {len(out)} new follow-up dialogues", flush=True)
for d in out[:6]: print("[moreq3]   " + " -> ".join(t["q"][:60] for t in d["turns"]) + " | " + " / ".join(t["gold"][:20] for t in d["turns"]), flush=True)
PYY
hf upload $R /root/work/mt_eval_bridge3.jsonl pooler_distill/chatsft/multiturn/mt_eval_bridge3.jsonl >/dev/null 2>&1
[ -s /root/work/mt_eval_bridge3.jsonl ] || { echo "MOREQ3_JOB_DONE no dialogues"; exit 1; }
until grep -q "chains (win) s300" /root/moreq.log 2>/dev/null || grep -q "MOREQ_JOB_DONE" /root/moreq.log 2>/dev/null; do sleep 60; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 8500 ]; do sleep 60; done
tally() { python3 - "$1" "$2" <<'PY2'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])]
t1 = [x for x in r if x["turn"] == 0]; t2 = [x for x in r if x["turn"] == 1]
print(f"[moreq3] {sys.argv[2]}: turn 1 {sum(x['correct'] for x in t1)}/{len(t1)}, follow-up {sum(x['correct'] for x in t2)}/{len(t2)}", flush=True)
PY2
}
for arm in "s100:/root/mtg3_s100.safetensors" "s300:/root/mtg1_s300.safetensors"; do T=${arm%%:*}; CK=${arm#*:}
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl /root/work/br3_$T.jsonl --multiturn /root/work/mt_eval_bridge3.jsonl --mt-mode win --n 1000 $EVARGS --tag "[br3-$T]" > /root/br3_$T.log 2>&1
  grep -E "Traceback|Error" /root/br3_$T.log | tail -1 | cut -c1-160; tally /root/work/br3_$T.jsonl "bridge3 (win) $T"
  hf upload $R /root/work/br3_$T.jsonl pooler_distill/chatsft/multiturn/br3_$T.jsonl >/dev/null 2>&1
done
echo "MOREQ3_JOB_DONE $(date -u)"
MQ
  setsid nohup bash -c 'bash /root/moreq3keep.sh 2>&1 | tee -a /root/moreq3.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MOREQ3_LAUNCHED $(date -u)"
fi
# ---- moreq2 (2026-10-04 12:50 JST): mt_gen wrote 0 of the 100 new follow-up dialogues (every DeepSeek call failed;
# searchq got 1/500 the same way on 10-02). Check the DeepSeek key once (status only), and write the dialogues with
# the OpenAI model instead (mt_gen --api openai). Their evaluation (s100, s300 under win) runs once moreq's chains are
# done, beside its single-turn pass when the card has room.
if [ ! -e /root/.dsk_check ]; then touch /root/.dsk_check
  echo "DSK_CHECK http $(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $(tr -d '[:space:]' < /root/.dsk 2>/dev/null)" https://api.deepseek.com/user/balance) $(date -u)"
fi
if ! pgrep -f "moreq2kee[p].sh" >/dev/null && ! grep -q "MOREQ2_JOB_DONE" /root/moreq2.log 2>/dev/null; then
  cat > /root/moreq2keep.sh <<'MQ'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
curl -sSf -o /root/work/mt_gen.py "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl/mt_gen.py?$(date +%s)"
grep -q -- "--api" /root/work/mt_gen.py || { echo "MOREQ2_JOB_DONE mt_gen has no --api"; exit 1; }
[ -s /root/work/mt_exclude2.jsonl ] || { until [ -s /root/work/mt_exclude2.jsonl ]; do sleep 30; done; }
python3 /root/work/mt_gen.py --api openai --model gpt-5-nano --seeds /root/work/eval300.jsonl --exclude /root/work/mt_exclude2.jsonl --out /root/work/mt_eval_bridge2_raw.jsonl --n-bridge 100 --n-memory 0 --n-switch 0 --seed 21 2>&1 | grep -E "^\[mtgen\]|MTGEN_DONE|Error|Traceback" | cut -c1-200
python3 - <<'PYY'
import json
old = set()
for l in open("/root/work/mt_eval.jsonl"):
    d = json.loads(l); old |= {t["q"].strip() for t in d["turns"]}; old.add((d.get("seed") or "").strip())
out = []
try:
    for l in open("/root/work/mt_eval_bridge2_raw.jsonl"):
        d = json.loads(l)
        if d.get("kind") == "bridge" and len(d.get("turns", [])) == 2 and not any(t["q"].strip() in old for t in d["turns"]): out.append(d)
except FileNotFoundError: pass
open("/root/work/mt_eval_bridge2.jsonl", "w").write("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in out))
print(f"[moreq2] bridge2: {len(out)} new held-out follow-up dialogues", flush=True)
for d in out[:3]: print("[moreq2]   " + " -> ".join(t["q"][:70] for t in d["turns"]), flush=True)
PYY
hf upload $R /root/work/mt_eval_bridge2.jsonl pooler_distill/chatsft/multiturn/mt_eval_bridge2.jsonl >/dev/null 2>&1
[ -s /root/work/mt_eval_bridge2.jsonl ] || { echo "MOREQ2_JOB_DONE no dialogues"; exit 1; }
until grep -q "chains (win) s300" /root/moreq.log 2>/dev/null || grep -q "MOREQ_JOB_DONE" /root/moreq.log 2>/dev/null; do sleep 60; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 8500 ]; do sleep 60; done
tally() { python3 - "$1" "$2" <<'PY2'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])]
t1 = [x for x in r if x["turn"] == 0]; t2 = [x for x in r if x["turn"] == 1]
print(f"[moreq2] {sys.argv[2]}: turn 1 {sum(x['correct'] for x in t1)}/{len(t1)}, follow-up {sum(x['correct'] for x in t2)}/{len(t2)}", flush=True)
PY2
}
for arm in "s100:/root/mtg3_s100.safetensors" "s300:/root/mtg1_s300.safetensors"; do T=${arm%%:*}; CK=${arm#*:}
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl /root/work/br2_$T.jsonl --multiturn /root/work/mt_eval_bridge2.jsonl --mt-mode win --n 1000 $EVARGS --tag "[br2-$T]" > /root/br2_$T.log 2>&1
  grep -E "Traceback|Error" /root/br2_$T.log | tail -1 | cut -c1-160; tally /root/work/br2_$T.jsonl "bridge2 (win) $T"
  hf upload $R /root/work/br2_$T.jsonl pooler_distill/chatsft/multiturn/br2_$T.jsonl >/dev/null 2>&1
done
echo "MOREQ2_JOB_DONE $(date -u)"
MQ
  setsid nohup bash -c 'bash /root/moreq2keep.sh 2>&1 | tee -a /root/moreq2.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MOREQ2_LAUNCHED $(date -u)"
fi
# ---- moreq (2026-10-04 09:30 JST, the user: screen the GRPO candidate on more questions). mtg3 step 100 against
# step 300 (the current app candidate), both under the app's memory spec (win): (1) 100 NEW held-out follow-up
# dialogues written from eval300 seeds not used by mt_eval (API); (2) all 30 switch chains (s300's win file resumes
# from its 12); (3) the full single-turn 300 (the 102-question files resume, so only the other 198 run). After dl100.
if ! pgrep -f "moreqkee[p].sh" >/dev/null && ! grep -q "MOREQ_JOB_DONE" /root/moreq.log 2>/dev/null; then
  cat > /root/moreqkeep.sh <<'MQ'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
# (1) new follow-up dialogues, API only, started now
( python3 - <<'PYX'
import json
ex = set()
for f in ("/root/work/mt_eval.jsonl", "/root/work/mt_eval_chain.jsonl", "/root/work/mt_train.jsonl", "/root/work/mt_train_br2.jsonl", "/root/work/mt_train_bridge.jsonl"):
    try:
        for l in open(f):
            d = json.loads(l); ex.add((d.get("seed") or "").strip()); ex |= {t["q"].strip() for t in d.get("turns", [])}
    except FileNotFoundError: pass
open("/root/work/mt_exclude2.jsonl", "w").write("".join(json.dumps({"q": q}) + "\n" for q in ex if q))
print(f"[moreq] {len(ex)} questions excluded as seeds")
PYX
  DSK_KEY=$(cat /root/.dsk 2>/dev/null) OAI_KEY=$(cat /root/.oai 2>/dev/null) python3 /root/work/mt_gen.py --seeds /root/work/eval300.jsonl --exclude /root/work/mt_exclude2.jsonl --out /root/work/mt_eval_bridge2_raw.jsonl --n-bridge 100 --n-memory 0 --n-switch 0 --seed 21 2>&1 | grep -E "^\[mtgen\]|MTGEN_DONE|Error|Traceback"
  python3 - <<'PYY'
import json
old = set()
for l in open("/root/work/mt_eval.jsonl"):
    d = json.loads(l); old |= {t["q"].strip() for t in d["turns"]}; old.add((d.get("seed") or "").strip())
out = []
try:
    for l in open("/root/work/mt_eval_bridge2_raw.jsonl"):
        d = json.loads(l)
        if d.get("kind") == "bridge" and len(d.get("turns", [])) == 2 and not any(t["q"].strip() in old for t in d["turns"]): out.append(d)
except FileNotFoundError: pass
open("/root/work/mt_eval_bridge2.jsonl", "w").write("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in out))
print(f"[moreq] bridge2: {len(out)} new held-out follow-up dialogues")
PYY
  hf upload $R /root/work/mt_eval_bridge2.jsonl pooler_distill/chatsft/multiturn/mt_eval_bridge2.jsonl >/dev/null 2>&1 ) &
GEN=$!
until grep -q "DL100_JOB_DONE" /root/dl100.log 2>/dev/null; do sleep 60; done
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
wait $GEN
tally() { python3 - "$1" "$2" <<'PY2'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])]
t1 = [x for x in r if x["turn"] == 0]; t2 = [x for x in r if x["turn"] == 1]
print(f"[moreq] {sys.argv[2]}: turn 1 {sum(x['correct'] for x in t1)}/{len(t1)}, follow-up {sum(x['correct'] for x in t2)}/{len(t2)}", flush=True)
PY2
}
for arm in "s100:/root/mtg3_s100.safetensors" "s300:/root/mtg1_s300.safetensors"; do T=${arm%%:*}; CK=${arm#*:}
  [ -s /root/work/mt_eval_bridge2.jsonl ] && {
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl /root/work/br2_$T.jsonl --multiturn /root/work/mt_eval_bridge2.jsonl --mt-mode win --n 1000 $EVARGS --tag "[br2-$T]" > /root/br2_$T.log 2>&1
  grep -E "Traceback|Error" /root/br2_$T.log | tail -1 | cut -c1-160; tally /root/work/br2_$T.jsonl "bridge2 (win) $T"
  hf upload $R /root/work/br2_$T.jsonl pooler_distill/chatsft/multiturn/br2_$T.jsonl >/dev/null 2>&1; }
done
for arm in "s100:/root/mtg3_s100.safetensors:/root/work/mc_win_s100.jsonl" "s300:/root/mtg1_s300.safetensors:/root/work/mc_win.jsonl"; do T=${arm%%:*}; r=${arm#*:}; CK=${r%%:*}; F=${r#*:}
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl $F --multiturn /root/work/mt_eval_chain.jsonl --mt-mode win --n 1000 $EVARGS --tag "[chain-win-$T]" > /root/chainwin_$T.log 2>&1
  echo "[moreq] chains (win) $T: $(grep -E 'EVAL_DONE|Traceback' /root/chainwin_$T.log | tail -1 | cut -c1-160); $(python3 -c "
import json; c=[0]*4
for l in open('$F'):
    r=json.loads(l); c[r['turn']]+=bool(r['correct'])
print('per turn', c, 'turns 2-4', sum(c[1:]), '/90 (full-history s300: 21 18 17 17 -> 52)')")"
  hf upload $R $F pooler_distill/chatsft/multiturn/$(basename $F) >/dev/null 2>&1
done
for arm in "s100:/root/mtg3_s100.safetensors:mtg3s100st" "s300:/root/mtg1_s300.safetensors:mtg1s300st"; do T=${arm%%:*}; r=${arm#*:}; CK=${r%%:*}; P=${r#*:}
  SN=0; NN=0
  for i in 0 1 2; do
    F=$(ls /root/work/${P}_$i.jsonl /root/work/${P}_out_$i.jsonl 2>/dev/null | head -1); [ -n "$F" ] || F=/root/work/${P}_full_$i.jsonl
    env $ENV python3 /root/work/pool_eval.py $CK /root/work/ev_$i.jsonl $F --n 100 $EVARGS --tag "[${P}full$i]" > /root/${P}full_$i.log 2>&1
    read c n <<< $(python3 -c "
import json; r=[json.loads(l) for l in open('$F')]; print(sum(bool(x.get('correct')) for x in r), len(r))"); SN=$((SN + c)); NN=$((NN + n))
    echo "[moreq] single $T shard $i: $c/$n ($(grep -E 'EVAL_DONE|Traceback' /root/${P}full_$i.log | tail -1 | cut -c1-120))"
    hf upload $R $F pooler_distill/chatsft/multiturn/${P}_full_$i.jsonl >/dev/null 2>&1
  done
  echo "[moreq] single-turn $T: $SN/$NN"
done
echo "MOREQ_JOB_DONE $(date -u)"
MQ
  setsid nohup bash -c 'bash /root/moreqkeep.sh 2>&1 | tee -a /root/moreq.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MOREQ_LAUNCHED $(date -u)"
fi
# ---- dl100 (2026-10-04 08:40 JST): step 100 is mtg3's best screen (64/102, 19/40) and the candidate; it lacks the
# Dolphin reasoning screen (only 200 got it). Run it after mtg3chk.
if ! pgrep -f "dl100kee[p].sh" >/dev/null && ! grep -q "DL100_JOB_DONE" /root/dl100.log 2>/dev/null; then
  cat > /root/dl100keep.sh <<'DL'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; CK=/root/mtg3_s100.safetensors
until grep -q "MTG3CHK_JOB_DONE" /root/mtg3chk.log 2>/dev/null; do sleep 60; done
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
[ -s $CK ] || { echo "DL100_JOB_DONE no checkpoint"; exit 1; }
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $CK /root/evalrun_dl_mtg3s100 \
  --questions /root/work/eval300.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl \
  --b 12 --gen 7000 --budget 2400 --temp 0.6 --maxsrch 7 --pooler none --lora-rank 16 --lora-layers all --stop eos --loop-break answer \
  --eval-file /root/work/dolphinq.jsonl --eval-out /root/work/dl_mtg3s100.jsonl > /root/dl_mtg3s100.log 2>&1
[ -s /root/dl_judge.py ] && OAI_KEY=$(cat /root/.oai 2>/dev/null) python3 /root/dl_judge.py /root/work/dl_mtg3s100.jsonl mtg3s100 | sed 's/\[mix0\]/[dl100]/'
hf upload $R /root/work/dl_mtg3s100_judged.jsonl pooler_distill/chatsft/multiturn/dl_mtg3s100_judged.jsonl >/dev/null 2>&1
echo "DL100_JOB_DONE $(date -u)"
DL
  setsid nohup bash -c 'bash /root/dl100keep.sh 2>&1 | tee -a /root/dl100.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "DL100_LAUNCHED $(date -u)"
fi
# ---- mtg3b (2026-10-04 01:05 JST, the user asleep: keep the card busy). After mtg3chk screens step 200, continue
# to 300 when 200 still holds step 300's level (single >= 59/102 and follow-ups >= 17/40, i.e. no worse than s300),
# then screen 300 the same way (bridge under win, single-turn, Dolphin).
if ! pgrep -f "mtg3bkee[p].sh" >/dev/null && ! grep -q "MTG3B_JOB_DONE" /root/mtg3b.log 2>/dev/null; then
  cat > /root/mtg3bkeep.sh <<'MB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; OUT=/root/online_mtg3
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
until grep -q "MTG3CHK_JOB_DONE" /root/mtg3chk.log 2>/dev/null; do sleep 120; done
grep -q "MTG3_CONTINUE" /root/mtg3chk.log || { echo "MTG3B_JOB_DONE (mtg3 was switched, nothing to continue)"; exit 0; }
SN=$(grep -oE "mtg3 s200 single-turn: [0-9]+" /root/mtg3chk.log | grep -oE "[0-9]+$" | tail -1); F=$(grep -oE "mtg3 s200 bridge \(win\): turn 1 [0-9]+ /40, follow-up [0-9]+" /root/mtg3chk.log | grep -oE "[0-9]+$" | tail -1)
echo "[mtg3b] step 200: single ${SN:-?}/102, follow-ups ${F:-?}/40"
[ -n "$SN" ] && [ -n "$F" ] && [ "$SN" -ge 59 ] && [ "$F" -ge 17 ] || { echo "MTG3B_JOB_DONE (200 below s300: not continued)"; exit 0; }
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
echo "[mtg3b] continuing to 300 $(date -u +%H:%M)"
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OAI_KEY=$(cat /root/.oai 2>/dev/null) DSK_KEY=$(cat /root/.dsk 2>/dev/null) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /root/work/online_loop.py /root/mtg1_s300.safetensors $OUT --pooler none --lora-layers all --lora-rank 16 --save-lora-only 1 \
  --mt-items /root/work/mtg2_items.jsonl --heldout /root/work/eval300.jsonl --reason /root/work/dolphin_rft.jsonl --reason-every 4 --reason-g 8 \
  --steps 300 --save-every 25 --lr 5e-6 --search-lr 5e-6 --search-temp 0.9 --search-gen 2000 --temp 0.6 --gen 3000 --budget 900 --maxsrch 7 --stop eos \
  --judge-api openai --judge-model gpt-5-nano --w-talk 0.5 --dolphin-min 0 --adv-std 1 --pg-norm mean --kl 0 \
  --guard 1 --guard-steps 20 >> /root/mtg3_run.log 2>&1
grep -E "ONLINE_|Error|Traceback" /root/mtg3_run.log | tail -2 | cut -c1-250
S=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null); echo "[mtg3b] stopped at step ${S:-0} $(date -u +%H:%M)"
[ "${S:-0}" -ge 300 ] || { echo "MTG3B_JOB_DONE short"; exit 1; }
CK=/root/mtg3_s300.safetensors; cp $OUT/latest.safetensors $CK; hf upload $R $CK pooler_distill/chatsft/multiturn/mtg3_s300.safetensors >/dev/null 2>&1
hf upload $R $OUT/rollouts.jsonl pooler_distill/chatsft/multiturn/mtg3_rollouts.jsonl >/dev/null 2>&1
env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl /root/work/br_mtg3s300.jsonl --multiturn /root/work/mt_eval_bridge.jsonl --mt-mode win --n 1000 $EVARGS --tag "[br-mtg3s300]" > /root/br_mtg3s300.log 2>&1
echo "[mtg3b] s300 bridge (win): $(python3 -c "
import json; r=[json.loads(l) for l in open('/root/work/br_mtg3s300.jsonl')]
print('turn 1', sum(x['correct'] for x in r if x['turn']==0), '/40, follow-up', sum(x['correct'] for x in r if x['turn']==1), '/40')") (s300: 25, 17)"
SN=0; for i in 0 1 2; do env $ENV python3 /root/work/pool_eval.py $CK /root/work/ev_$i.jsonl /root/work/mtg3s300st_$i.jsonl --n 34 $EVARGS --tag "[mtg3s300st$i]" > /root/mtg3s300st_$i.log 2>&1
  c=$(python3 -c "import json;print(sum(bool(json.loads(l).get('correct')) for l in open('/root/work/mtg3s300st_$i.jsonl')))" 2>/dev/null || echo 0); SN=$((SN + c)); done
echo "[mtg3b] s300 single-turn: $SN/102 (s300: 59)"
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $CK /root/evalrun_dl_mtg3s300 \
  --questions /root/work/eval300.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl \
  --b 12 --gen 7000 --budget 2400 --temp 0.6 --maxsrch 7 --pooler none --lora-rank 16 --lora-layers all --stop eos --loop-break answer \
  --eval-file /root/work/dolphinq.jsonl --eval-out /root/work/dl_mtg3s300.jsonl > /root/dl_mtg3s300.log 2>&1
[ -s /root/dl_judge.py ] && OAI_KEY=$(cat /root/.oai 2>/dev/null) python3 /root/dl_judge.py /root/work/dl_mtg3s300.jsonl mtg3s300 | sed 's/\[mix0\]/[mtg3b]/'
echo "MTG3B_JOB_DONE $(date -u)"
MB
  setsid nohup bash -c 'bash /root/mtg3bkeep.sh 2>&1 | tee -a /root/mtg3b.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MTG3B_LAUNCHED $(date -u)"
fi
# ---- mtg3chk (2026-10-03 21:40 JST, the user: if mtg3 has bottomed out keep going, if not switch now). The training
# pass rate cannot say (mtg2 and mtg3 draw the same questions at the same steps and match there: the dips are hard
# stretches of the list), so mtg3 pauses at step 100 and step 100 is screened: the 40 bridge dialogues under win and
# the single-turn 102. Keep going (resume to 200) when it holds step 300's level within noise (single >= 56/102 and
# follow-ups >= 15/40; step 300: 59/102, 17/40), else switch to mtg4: the same recipe with the teacher also checking
# correctness (--judge-correct 1: Claire for Claire Casey counts; the evaluation stays strict). Before either, the
# teacher's correctness check is tried on mtg3's wrong replies so its verdicts can be read.
if ! pgrep -f "mtg3chkkee[p].sh" >/dev/null && ! grep -q "MTG3CHK_JOB_DONE" /root/mtg3chk.log 2>/dev/null; then
  cat > /root/mtg3chkkeep.sh <<'MC'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; OUT=/root/online_mtg3
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
step() { python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null || echo 0; }
until [ "$(step)" -ge 100 ] || grep -q "MTG3_JOB_DONE\|MTG3_ABORT" /root/mtg3.log 2>/dev/null; do sleep 30; done
pkill -f "mtg3kee[p].sh"; pkill -f "online_loop.py /root/mtg1_s300.safetensors /root/online_mtg3"; sleep 15
S=$(step); cp $OUT/latest.safetensors /root/mtg3_s$S.safetensors; echo "[mtg3chk] mtg3 paused at step $S $(date -u +%H:%M)"
echo "MTG3_JOB_DONE paused by mtg3chk at $S $(date -u)" >> /root/mtg3.log
hf upload $R /root/mtg3_s$S.safetensors pooler_distill/chatsft/multiturn/mtg3_s$S.safetensors >/dev/null 2>&1
# the teacher's correctness check on mtg3's wrong replies (API only, beside the screens)
OAI_KEY=$(cat /root/.oai 2>/dev/null) python3 - <<'PYJ' &
import json, os, re, urllib.request, random
from concurrent.futures import ThreadPoolExecutor
src = open("/root/work/online_loop.py").read(); i = src.index("CORRECT_SYS = "); j = src.index('"""', src.index('"""', i) + 3) + 3
ns = {}; exec(src[i:j], ns); SYS = ns["CORRECT_SYS"]
gold = {json.loads(l)["q"].strip(): json.loads(l)["gold"] for l in open("/root/work/mtg2_items.jsonl")}
rows = []
for l in open("/root/online_mtg3/rollouts.jsonl"):
    r = json.loads(l)
    if r.get("kind") == "reason" or r["reward"] >= 1.0 or "</think>" not in r["text"]: continue
    rep = r["text"].split("</think>")[-1].strip()
    if rep and r["q"].strip() in gold: rows.append((r["q"].strip(), rep[:2000], r["text"]))
random.Random(0).shuffle(rows); rows = rows[:150]; key = os.environ.get("OAI_KEY", "")
INFO = re.compile(r"<information>(.*?)</information>", re.S)
def ask(x):
    q, rep, t = x; served = "\n\n".join(INFO.findall(t))[-4000:]
    body = {"model": "gpt-5-nano", "max_completion_tokens": 2000, "messages": [{"role": "system", "content": SYS},
            {"role": "user", "content": f"QUESTION:\n{q}\n\nREFERENCE ANSWER:\n{gold[q]}\n\nWHAT THE SEARCH RETURNED:\n{served}\n\nREPLY:\n{rep}"}]}
    for _ in range(3):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}), timeout=120))
            c = d["choices"][0]["message"].get("content") or ""; return json.loads(c[c.find("{"): c.rfind("}") + 1])
        except Exception: pass
    return {"error": 1}
with ThreadPoolExecutor(max_workers=8) as ex: V = list(ex.map(ask, rows))
ok = [(x, v) for x, v in zip(rows, V) if v.get("same_answer")]
print(f"[mtg3chk] teacher check on {len(rows)} wrong mtg3 replies: {len(ok)} judged the same answer, {sum(1 for v in V if 'error' in v)} errors", flush=True)
for (q, rep, _), v in ok[:15]: print(f"[mtg3chk]   gold {gold[q][:40]!r} | reply {rep[:110]!r}", flush=True)
PYJ
JP=$!
env $ENV python3 /root/work/pool_eval.py /root/mtg3_s$S.safetensors /root/work/eval300.jsonl /root/work/br_mtg3s$S.jsonl --multiturn /root/work/mt_eval_bridge.jsonl --mt-mode win --n 1000 $EVARGS --tag "[br-mtg3s$S]" > /root/br_mtg3s$S.log 2>&1
read T1 F <<< $(python3 -c "
import json; r=[json.loads(l) for l in open('/root/work/br_mtg3s$S.jsonl')]
print(sum(x['correct'] for x in r if x['turn']==0), sum(x['correct'] for x in r if x['turn']==1))")
echo "[mtg3chk] s$S bridge (win): turn 1 $T1/40, follow-up $F/40 (step 300: 25, 17)"
SN=0; for i in 0 1 2; do env $ENV python3 /root/work/pool_eval.py /root/mtg3_s$S.safetensors /root/work/ev_$i.jsonl /root/work/mtg3s${S}st_$i.jsonl --n 34 $EVARGS --tag "[mtg3s${S}st$i]" > /root/mtg3s${S}st_$i.log 2>&1
  c=$(python3 -c "import json;print(sum(bool(json.loads(l).get('correct')) for l in open('/root/work/mtg3s${S}st_$i.jsonl')))" 2>/dev/null || echo 0); SN=$((SN + c)); done
echo "[mtg3chk] s$S single-turn: $SN/102 (step 300: 59)"
wait $JP
hf upload $R /root/work/br_mtg3s$S.jsonl pooler_distill/chatsft/multiturn/br_mtg3s$S.jsonl >/dev/null 2>&1
if [ "$SN" -ge 56 ] && [ "$F" -ge 15 ]; then
  echo "[mtg3chk] holding step 300's level: mtg3 resumes to 200 $(date -u +%H:%M)"; echo "MTG3_CONTINUE"
  while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
  env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OAI_KEY=$(cat /root/.oai 2>/dev/null) DSK_KEY=$(cat /root/.dsk 2>/dev/null) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 /root/work/online_loop.py /root/mtg1_s300.safetensors $OUT --pooler none --lora-layers all --lora-rank 16 --save-lora-only 1 \
    --mt-items /root/work/mtg2_items.jsonl --heldout /root/work/eval300.jsonl --reason /root/work/dolphin_rft.jsonl --reason-every 4 --reason-g 8 \
    --steps 200 --save-every 25 --lr 5e-6 --search-lr 5e-6 --search-temp 0.9 --search-gen 2000 --temp 0.6 --gen 3000 --budget 900 --maxsrch 7 --stop eos \
    --judge-api openai --judge-model gpt-5-nano --w-talk 0.5 --dolphin-min 0 --adv-std 1 --pg-norm mean --kl 0 \
    --guard 1 --guard-steps 20 >> /root/mtg3_run.log 2>&1
  TAG=mtg3; CK=$OUT/latest.safetensors; S2=$(step)
else
  echo "[mtg3chk] below step 300's level: switch to mtg4 (teacher-checked correctness) $(date -u +%H:%M)"; echo "MTG3_SWITCH"
  while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
  env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OAI_KEY=$(cat /root/.oai 2>/dev/null) DSK_KEY=$(cat /root/.dsk 2>/dev/null) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 /root/work/online_loop.py /root/mtg1_s300.safetensors /root/online_mtg4 --pooler none --lora-layers all --lora-rank 16 --save-lora-only 1 \
    --mt-items /root/work/mtg2_items.jsonl --heldout /root/work/eval300.jsonl --reason /root/work/dolphin_rft.jsonl --reason-every 4 --reason-g 8 \
    --steps 200 --save-every 25 --lr 5e-6 --search-lr 5e-6 --search-temp 0.9 --search-gen 2000 --temp 0.6 --gen 3000 --budget 900 --maxsrch 7 --stop eos \
    --judge-api openai --judge-model gpt-5-nano --judge-correct 1 --w-talk 0.5 --dolphin-min 0 --adv-std 1 --pg-norm mean --kl 0 \
    --guard 1 --guard-steps 20 > /root/mtg4_run.log 2>&1
  TAG=mtg4; OUT=/root/online_mtg4; CK=$OUT/latest.safetensors; S2=$(step)
fi
grep -E "ONLINE_|Error|Traceback" /root/${TAG}_run.log | tail -2 | cut -c1-250
echo "[mtg3chk] $TAG stopped at step $S2 $(date -u +%H:%M)"
[ -s $CK ] || { echo "MTG3CHK_JOB_DONE no checkpoint"; exit 1; }
cp $CK /root/${TAG}_s$S2.safetensors; CK=/root/${TAG}_s$S2.safetensors; hf upload $R $CK pooler_distill/chatsft/multiturn/${TAG}_s$S2.safetensors >/dev/null 2>&1
hf upload $R $OUT/rollouts.jsonl pooler_distill/chatsft/multiturn/${TAG}_rollouts.jsonl >/dev/null 2>&1
env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl /root/work/br_${TAG}s$S2.jsonl --multiturn /root/work/mt_eval_bridge.jsonl --mt-mode win --n 1000 $EVARGS --tag "[br-${TAG}s$S2]" > /root/br_${TAG}s$S2.log 2>&1
echo "[mtg3chk] $TAG s$S2 bridge (win): $(python3 -c "
import json; r=[json.loads(l) for l in open('/root/work/br_${TAG}s$S2.jsonl')]
print('turn 1', sum(x['correct'] for x in r if x['turn']==0), '/40, follow-up', sum(x['correct'] for x in r if x['turn']==1), '/40')") (step 300: 25, 17)"
SN=0; for i in 0 1 2; do env $ENV python3 /root/work/pool_eval.py $CK /root/work/ev_$i.jsonl /root/work/${TAG}s${S2}st_$i.jsonl --n 34 $EVARGS --tag "[${TAG}s${S2}st$i]" > /root/${TAG}s${S2}st_$i.log 2>&1
  c=$(python3 -c "import json;print(sum(bool(json.loads(l).get('correct')) for l in open('/root/work/${TAG}s${S2}st_$i.jsonl')))" 2>/dev/null || echo 0); SN=$((SN + c)); done
echo "[mtg3chk] $TAG s$S2 single-turn: $SN/102 (step 300: 59)"
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $CK /root/evalrun_dl_${TAG}s$S2 \
  --questions /root/work/eval300.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl \
  --b 12 --gen 7000 --budget 2400 --temp 0.6 --maxsrch 7 --pooler none --lora-rank 16 --lora-layers all --stop eos --loop-break answer \
  --eval-file /root/work/dolphinq.jsonl --eval-out /root/work/dl_${TAG}s$S2.jsonl > /root/dl_${TAG}s$S2.log 2>&1
[ -s /root/dl_judge.py ] && OAI_KEY=$(cat /root/.oai 2>/dev/null) python3 /root/dl_judge.py /root/work/dl_${TAG}s$S2.jsonl ${TAG}s$S2 | sed 's/\[mix0\]/[mtg3chk]/'
echo "MTG3CHK_JOB_DONE $(date -u)"
MC
  setsid nohup bash -c 'bash /root/mtg3chkkeep.sh 2>&1 | tee -a /root/mtg3chk.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MTG3CHK_LAUNCHED $(date -u)"
fi
# ---- mtg3 (2026-10-03 15:00 JST, the user: GRPO with reasoning mixed in, mtg2's failure fixed). mtg2 (step 300 on
# switch + bridge, every step a search turn, lr 1e-5) rolled back at step 56 (unfinished 22% vs 7%, searches 3.2 vs
# 2.1, pass 15% vs 39%) and lost Dolphin 54 -> 49. mtg3, from step 300: one step in four is a Dolphin reasoning
# problem (dolphin_rft, the held-out hundred excluded; 8 samples, nano against the R1 reference, up to 3000 tokens),
# the other three mtg2's search turns; both rates 5e-6 (where mtg2 ran stably after its rollback); 200 steps.
# Screens at 100 and 200 under the app's memory spec (win): the 40 bridge dialogues (step 300: follow-ups 17/40),
# the single-turn 102 (58/102), Dolphin 100 (54).
if ! pgrep -f "mtg3kee[p].sh" >/dev/null && ! grep -q "MTG3_JOB_DONE\|MTG3_ABORT" /root/mtg3.log 2>/dev/null; then
  cat > /root/mtg3keep.sh <<'M3'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; OUT=/root/online_mtg3
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
[ -s /root/work/mtg2_items.jsonl ] && [ -s /root/work/dolphin_rft.jsonl ] || { echo "MTG3_ABORT data missing"; exit 1; }
# what mtg2's first 60 steps looked like, by item kind (bridge first turn / bridge follow-up / switch)
python3 - <<'PYD'
import json, collections
items = {}
for l in open("/root/work/mtg2_items.jsonl"):
    r = json.loads(l); items[r["q"].strip()] = r
br = set()
for l in open("/root/work/mt_train_bridge.jsonl"):
    d = json.loads(l); br |= {t["q"].strip() for t in d["turns"]}
st = collections.defaultdict(lambda: [0, 0, 0, 0])
try:
    for l in open("/root/online_mtg2/rollouts.jsonl"):
        r = json.loads(l)
        if r["step"] > 60: continue
        q = r["q"].strip(); it = items.get(q, {})
        k = ("bridge-follow" if it.get("hist") else "bridge-first") if q in br else "switch"
        k += " 1-40" if r["step"] <= 40 else " 41-60"
        w = r.get("why") or {}; s = st[k]; s[0] += 1; s[1] += r["reward"] >= 1.0; s[2] += bool(w.get("unfinished")); s[3] += r.get("ns", 0)
    for k in sorted(st): n, p, u, ns = st[k]; print(f"[mtg3] mtg2 {k}: {n} rollouts, pass {100*p/n:.0f}%, unfinished {100*u/n:.0f}%, searches {ns/n:.1f}")
except FileNotFoundError: print("[mtg3] no mtg2 rollouts on disk")
PYD
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
echo "[mtg3] start $(date -u +%H:%M); $(df -h /root | tail -1)"
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OAI_KEY=$(cat /root/.oai 2>/dev/null) DSK_KEY=$(cat /root/.dsk 2>/dev/null) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /root/work/online_loop.py /root/mtg1_s300.safetensors $OUT --pooler none --lora-layers all --lora-rank 16 --save-lora-only 1 \
  --mt-items /root/work/mtg2_items.jsonl --heldout /root/work/eval300.jsonl --reason /root/work/dolphin_rft.jsonl --reason-every 4 --reason-g 8 \
  --steps 200 --save-every 25 --lr 5e-6 --search-lr 5e-6 --search-temp 0.9 --search-gen 2000 --temp 0.6 --gen 3000 --budget 900 --maxsrch 7 --stop eos \
  --judge-api openai --judge-model gpt-5-nano --w-talk 0.5 --dolphin-min 0 --adv-std 1 --pg-norm mean --kl 0 \
  --guard 1 --guard-steps 20 > /root/mtg3_run.log 2>&1 &
LP=$!
( while kill -0 $LP 2>/dev/null; do sleep 60; [ -s $OUT/step100.safetensors ] || { S=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null); [ "${S:-0}" -ge 100 ] && cp $OUT/latest.safetensors $OUT/step100.safetensors; }; done ) &
wait $LP
grep -E "^\[data\]|ONLINE_|Error|Traceback" /root/mtg3_run.log | tail -4 | cut -c1-250
S=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null); echo "[mtg3] stopped at step ${S:-0} $(date -u +%H:%M)"
hf upload $R $OUT/rollouts.jsonl pooler_distill/chatsft/multiturn/mtg3_rollouts.jsonl >/dev/null 2>&1
[ "${S:-0}" -ge 100 ] || { echo "MTG3_ABORT: stopped at step ${S:-0}"; exit 1; }
for T in 100 200; do
  CK=$OUT/step$T.safetensors; [ $T = 200 ] && CK=$OUT/latest.safetensors; [ $T = 200 ] && [ "${S:-0}" -lt 200 ] && continue
  [ -s $CK ] || continue
  cp $CK /root/mtg3_s$T.safetensors; hf upload $R /root/mtg3_s$T.safetensors pooler_distill/chatsft/multiturn/mtg3_s$T.safetensors >/dev/null 2>&1; CK=/root/mtg3_s$T.safetensors
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl /root/work/br_mtg3s$T.jsonl --multiturn /root/work/mt_eval_bridge.jsonl --mt-mode win --n 1000 $EVARGS --tag "[br-mtg3s$T]" > /root/br_mtg3s$T.log 2>&1
  python3 - /root/work/br_mtg3s$T.jsonl $T <<'PY2'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])]
t1 = [x for x in r if x["turn"] == 0]; t2 = [x for x in r if x["turn"] == 1]
print(f"[mtg3] s{sys.argv[2]} bridge (win): turn 1 {sum(x['correct'] for x in t1)}/{len(t1)}, follow-up {sum(x['correct'] for x in t2)}/{len(t2)} (step 300: 25/40, 17/40)", flush=True)
PY2
  hf upload $R /root/work/br_mtg3s$T.jsonl pooler_distill/chatsft/multiturn/br_mtg3s$T.jsonl >/dev/null 2>&1
  for i in 0 1 2; do env $ENV python3 /root/work/pool_eval.py $CK /root/work/ev_$i.jsonl /root/work/mtg3s${T}st_$i.jsonl --n 34 $EVARGS --tag "[mtg3s${T}st$i]" > /root/mtg3s${T}st_$i.log 2>&1; done
  echo "[mtg3] s$T single-turn: $(grep -hoE 'correct=[0-9.]+%' /root/mtg3s${T}st_*.log | tr '\n' ' ') (step 300: 52.9 61.8 58.8)"
  env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $CK /root/evalrun_dl_mtg3s$T \
    --questions /root/work/eval300.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl \
    --b 12 --gen 7000 --budget 2400 --temp 0.6 --maxsrch 7 --pooler none --lora-rank 16 --lora-layers all --stop eos --loop-break answer \
    --eval-file /root/work/dolphinq.jsonl --eval-out /root/work/dl_mtg3s$T.jsonl > /root/dl_mtg3s$T.log 2>&1
  [ -s /root/dl_judge.py ] && OAI_KEY=$(cat /root/.oai 2>/dev/null) python3 /root/dl_judge.py /root/work/dl_mtg3s$T.jsonl mtg3s$T | sed 's/\[mix0\]/[mtg3]/'
  hf upload $R /root/work/dl_mtg3s${T}_judged.jsonl pooler_distill/chatsft/multiturn/dl_mtg3s${T}_judged.jsonl >/dev/null 2>&1
done
echo "MTG3_JOB_DONE $(date -u)"
M3
  setsid nohup bash -c 'bash /root/mtg3keep.sh 2>&1 | tee -a /root/mtg3.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MTG3_LAUNCHED $(date -u)"
fi
# ---- 2026-10-03 13:35 JST, the user: the no-history floor is not needed. Stop it; the GPU is free.
if [ ! -e /root/.brnone_stop ]; then touch /root/.brnone_stop
  pkill -f "memwin6kee[p].sh"; sleep 2; pkill -f "pool_eval.py .*br_none"
  echo "MEMWIN6_JOB_DONE none stopped $(date -u)" >> /root/memwin.log; echo "BRNONE_STOPPED $(date -u)"
fi
# ---- memwin6 (2026-10-03 12:40 JST, the user: win IS the spec - the recent exchange raw in the window, older history in
# the pooler; pooler-only was never the goal). Stop stream; bridge in win (fixed: first question in the stream), all 40,
# then none as the floor.
if [ ! -e /root/.memwin6_swap ]; then touch /root/.memwin6_swap
  pkill -f "memwin5kee[p].sh"; sleep 2; pkill -f "pool_eval.py .*br_"; sleep 5
  echo "MEMWIN5_JOB_DONE replaced by memwin6 $(date -u)" >> /root/memwin.log; echo "MEMWIN6_SWAP $(date -u)"
fi
if ! pgrep -f "memwin6kee[p].sh" >/dev/null && ! grep -q "MEMWIN6_JOB_DONE" /root/memwin.log 2>/dev/null; then
  cat > /root/memwin6keep.sh <<'MW'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; CK=/root/mtg1_s300.safetensors
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
tally() { python3 - /root/work/br_$1.jsonl $1 <<'PY2'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])]
t1 = [x for x in r if x["turn"] == 0]; t2 = [x for x in r if x["turn"] == 1]
print(f"[memwin6] bridge {sys.argv[2]}: turn 1 {sum(x['correct'] for x in t1)}/{len(t1)}, follow-up {sum(x['correct'] for x in t2)}/{len(t2)} (full history: follow-up 18/40 = 45.0%)", flush=True)
PY2
hf upload $R /root/work/br_$1.jsonl pooler_distill/chatsft/multiturn/br_$1.jsonl >/dev/null 2>&1; }
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
tally stream
for v in win2:win none:none; do T=${v%%:*}; M=${v#*:}
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl /root/work/br_$T.jsonl --multiturn /root/work/mt_eval_bridge.jsonl --mt-mode $M --n 1000 $EVARGS --tag "[br-$T]" > /root/br_$T.log 2>&1
  grep -E "Error|Traceback" /root/br_$T.log | tail -2 | cut -c1-200
  tally $T
done
echo "MEMWIN6_JOB_DONE $(date -u)"
MW
  setsid nohup bash -c 'bash /root/memwin6keep.sh 2>&1 | tee -a /root/memwin.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MEMWIN6_LAUNCHED $(date -u)"
fi
# ---- memwin5 (2026-10-03 12:40 JST, the user: no more win runs; the pooler-only test now). Stop the bridge win run
# and memwin4; run stream (pooler only) then none (floor).
if [ ! -e /root/.memwin5_swap ]; then touch /root/.memwin5_swap
  pkill -f "memwin2kee[p].sh"; pkill -f "memwin3kee[p].sh"; pkill -f "memwin4kee[p].sh"; sleep 2; pkill -f "pool_eval.py .*br_"; sleep 5
  echo "MEMWIN4_JOB_DONE replaced by memwin5 $(date -u)" >> /root/memwin.log; echo "MEMWIN5_SWAP $(date -u)"
fi
if ! pgrep -f "memwin5kee[p].sh" >/dev/null && ! grep -q "MEMWIN5_JOB_DONE" /root/memwin.log 2>/dev/null; then
  cat > /root/memwin5keep.sh <<'MW'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; CK=/root/mtg1_s300.safetensors
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
tally() { python3 - /root/work/br_$1.jsonl $1 <<'PY2'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])]
t1 = [x for x in r if x["turn"] == 0]; t2 = [x for x in r if x["turn"] == 1]
print(f"[memwin5] bridge {sys.argv[2]}: turn 1 {sum(x['correct'] for x in t1)}/{len(t1)}, follow-up {sum(x['correct'] for x in t2)}/{len(t2)} (full history: follow-up 18/40 = 45.0%)", flush=True)
PY2
hf upload $R /root/work/br_$1.jsonl pooler_distill/chatsft/multiturn/br_$1.jsonl >/dev/null 2>&1; }
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
tally win
for M in stream none; do
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl /root/work/br_$M.jsonl --multiturn /root/work/mt_eval_bridge.jsonl --mt-mode $M --n 1000 $EVARGS --tag "[br-$M]" > /root/br_$M.log 2>&1
  grep -E "Error|Traceback" /root/br_$M.log | tail -2 | cut -c1-200
  tally $M
done
echo "MEMWIN5_JOB_DONE $(date -u)"
MW
  setsid nohup bash -c 'bash /root/memwin5keep.sh 2>&1 | tee -a /root/memwin.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MEMWIN5_LAUNCHED $(date -u)"
fi
# ---- memwin4 (2026-10-03 12:40 JST, the user: success means the past comes back THROUGH THE POOLER alone - stream:
# only the new question pinned, the raw window starts empty, every earlier token (questions, thinking, pages, replies)
# is in the pooler). Bridge order now: (win, already running) -> stream -> none -> win2. Totals from the files.
if [ ! -e /root/.memwin4_swap ]; then touch /root/.memwin4_swap
  pkill -f "memwin2kee[p].sh"; pkill -f "memwin3kee[p].sh"; sleep 2
  echo "MEMWIN2_JOB_DONE MEMWIN3_JOB_DONE replaced by memwin4 $(date -u)" >> /root/memwin.log; echo "MEMWIN4_SWAP $(date -u)"
fi
if ! pgrep -f "memwin4kee[p].sh" >/dev/null && ! grep -q "MEMWIN4_JOB_DONE" /root/memwin.log 2>/dev/null; then
  cat > /root/memwin4keep.sh <<'MW'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; CK=/root/mtg1_s300.safetensors
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
tally() { python3 - /root/work/br_$1.jsonl $1 <<'PY2'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])]
t1 = [x for x in r if x["turn"] == 0]; t2 = [x for x in r if x["turn"] == 1]
print(f"[memwin4] bridge {sys.argv[2]}: turn 1 {sum(x['correct'] for x in t1)}/{len(t1)}, follow-up {sum(x['correct'] for x in t2)}/{len(t2)} (full history: follow-up 18/40 = 45.0%)", flush=True)
PY2
hf upload $R /root/work/br_$1.jsonl pooler_distill/chatsft/multiturn/br_$1.jsonl >/dev/null 2>&1; }
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
tally win
for v in stream:stream none:none win2:win; do T=${v%%:*}; M=${v#*:}
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl /root/work/br_$T.jsonl --multiturn /root/work/mt_eval_bridge.jsonl --mt-mode $M --n 1000 $EVARGS --tag "[br-$T]" > /root/br_$T.log 2>&1
  grep -E "Error|Traceback" /root/br_$T.log | tail -2 | cut -c1-200
  tally $T
done
echo "MEMWIN4_JOB_DONE $(date -u)"
MW
  setsid nohup bash -c 'bash /root/memwin4keep.sh 2>&1 | tee -a /root/memwin.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MEMWIN4_LAUNCHED $(date -u)"
fi
# ---- memwin2 (2026-10-03 11:55 JST): the switch chains' later questions stand on their own (no history at all scored
# 36.7% last / 64.4% mid), so a right answer there does not show the memory works. The 40 held-out bridge dialogues
# (turn 2 is a follow-up that needs turn 1): win, and none (no history) as the floor. Full history: last 45.0%.
if ! pgrep -f "memwin2kee[p].sh" >/dev/null && ! grep -q "MEMWIN2_JOB_DONE" /root/memwin.log 2>/dev/null; then
  cat > /root/memwin2keep.sh <<'MW'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; CK=/root/mtg1_s300.safetensors
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
until grep -q "MEMWIN_JOB_DONE" /root/memwin.log 2>/dev/null; do sleep 60; done
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
for M in win none; do
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl /root/work/br_$M.jsonl --multiturn /root/work/mt_eval_bridge.jsonl --mt-mode $M --n 1000 $EVARGS --tag "[br-$M]" > /root/br_$M.log 2>&1
  echo "[memwin2] bridge $M: $(grep -E 'EVAL_DONE|Error|Traceback' /root/br_$M.log | tail -1 | cut -c1-200) (full history: last 45.0% mid 57.5%)"
  hf upload $R /root/work/br_$M.jsonl pooler_distill/chatsft/multiturn/br_$M.jsonl >/dev/null 2>&1
done
echo "MEMWIN2_JOB_DONE $(date -u)"
MW
  setsid nohup bash -c 'bash /root/memwin2keep.sh 2>&1 | tee -a /root/memwin.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MEMWIN2_LAUNCHED $(date -u)"
fi
# ---- memwin3 (2026-10-03 12:35 JST): in win the first turn's question sat only in that turn's pinned prompt and never
# entered the stream (later questions did). Fixed in pool_eval; the bridge runs again in win after memwin2 (br_win2).
if ! pgrep -f "memwin3kee[p].sh" >/dev/null && ! grep -q "MEMWIN3_JOB_DONE" /root/memwin.log 2>/dev/null; then
  cat > /root/memwin3keep.sh <<'MW'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; CK=/root/mtg1_s300.safetensors
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
until grep -q "MEMWIN2_JOB_DONE" /root/memwin.log 2>/dev/null; do sleep 60; done
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
M=win2
env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl /root/work/br_$M.jsonl --multiturn /root/work/mt_eval_bridge.jsonl --mt-mode win --n 1000 $EVARGS --tag "[br-$M]" > /root/br_$M.log 2>&1
echo "[memwin3] bridge $M: $(grep -E 'EVAL_DONE|Error|Traceback' /root/br_$M.log | tail -1 | cut -c1-200) (full history: last 45.0% mid 57.5%)"
hf upload $R /root/work/br_$M.jsonl pooler_distill/chatsft/multiturn/br_$M.jsonl >/dev/null 2>&1
echo "MEMWIN3_JOB_DONE $(date -u)"
MW
  setsid nohup bash -c 'bash /root/memwin3keep.sh 2>&1 | tee -a /root/memwin.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MEMWIN3_LAUNCHED $(date -u)"
fi
# ---- memcap (2026-10-03 10:30 JST, the user: priority over GRPO - the app's memory must stay bounded; the pooler bounds
# one reply, but the multi-turn history sits in the prompt and grows every turn). GRPO step 300 (the app candidate) on
# the 30 held-out switch chains with the history bounded two ways: only the last K exchanges pinned (--mt-keep 1, 2),
# and mix (the last exchange pinned, everything older through the pooler). Against full history: turns 2-4 52/90.
if ! pgrep -f "memcapkee[p].sh" >/dev/null && ! grep -q "MEMCAP_JOB_DONE" /root/memcap.log 2>/dev/null; then
  cat > /root/memcapkeep.sh <<'MC'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; CK=/root/mtg1_s300.safetensors
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
for v in "k1:full:--mt-keep 1" "mix:mix:" "k2:full:--mt-keep 2"; do T=${v%%:*}; r=${v#*:}; M=${r%%:*}; X=${r#*:}
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl /root/work/mc_$T.jsonl --multiturn /root/work/mt_eval_chain.jsonl --mt-mode $M $X --n 1000 $EVARGS --tag "[mc-$T]" > /root/mc_$T.log 2>&1
  python3 - /root/work/mc_$T.jsonl $T <<'PY2'
import json, sys
c = [0] * 4
for l in open(sys.argv[1]):
    r = json.loads(l); c[r["turn"]] += bool(r["correct"])
print(f"[memcap] {sys.argv[2]}: per turn {c}, turns 2-4 {sum(c[1:])}/90 (full history: 21 18 17 17 -> 52)", flush=True)
PY2
  hf upload $R /root/work/mc_$T.jsonl pooler_distill/chatsft/multiturn/mc_$T.jsonl >/dev/null 2>&1
done
echo "MEMCAP_JOB_DONE $(date -u)"
MC
  setsid nohup bash -c 'bash /root/memcapkeep.sh 2>&1 | tee -a /root/memcap.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MEMCAP_LAUNCHED $(date -u)"
fi
# ---- memcap2 (2026-10-03 11:00 JST, the user: the v1.0 model without search held a conversation through the pooler).
# With search, the pooler's 384 kept tokens fill with served pages, crowding out the conversation. phist hands the
# pooler only the conversation (earlier questions + replies, no thinking, no pages): K=0 (all of it through the pooler,
# only the new question pinned) and K=1 (the last exchange pinned too). Bounded memory either way. Before mixcol.
if [ ! -e /root/.mixcol_regate ]; then touch /root/.mixcol_regate; pkill -f "mixcolkee[p].sh"; echo "MIXCOL_REGATED behind memcap2 $(date -u)"; fi
if ! pgrep -f "memcap2kee[p].sh" >/dev/null && ! grep -q "MEMCAP2_JOB_DONE" /root/memcap2.log 2>/dev/null; then
  cat > /root/memcap2keep.sh <<'MC3'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; CK=/root/mtg1_s300.safetensors
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
until grep -q "MEMCAP_JOB_DONE" /root/memcap.log 2>/dev/null; do sleep 60; done
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
for v in "ph1:--mt-keep 1" "ph0:--mt-keep 0"; do T=${v%%:*}; X=${v#*:}
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl /root/work/mc_$T.jsonl --multiturn /root/work/mt_eval_chain.jsonl --mt-mode phist $X --n 1000 $EVARGS --tag "[mc-$T]" > /root/mc_$T.log 2>&1
  python3 - /root/work/mc_$T.jsonl $T <<'PY2'
import json, sys
c = [0] * 4
for l in open(sys.argv[1]):
    r = json.loads(l); c[r["turn"]] += bool(r["correct"])
print(f"[memcap2] {sys.argv[2]}: per turn {c}, turns 2-4 {sum(c[1:])}/90 (full history: 21 18 17 17 -> 52)", flush=True)
PY2
  grep -E "Error|Traceback" /root/mc_$T.log | tail -2 | cut -c1-200
  hf upload $R /root/work/mc_$T.jsonl pooler_distill/chatsft/multiturn/mc_$T.jsonl >/dev/null 2>&1
done
echo "MEMCAP2_JOB_DONE $(date -u)"
MC3
  setsid nohup bash -c 'bash /root/memcap2keep.sh 2>&1 | tee -a /root/memcap2.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MEMCAP2_LAUNCHED $(date -u)"
fi
# ---- mixcol (2026-10-03 10:45 JST, the user: train on the follow-ups the model gets right under the bounded-memory
# spec, everything unfrozen): after memcap, step 300 runs the training dialogues in mix (the bounded protocol: last
# exchange pinned, older history through the pooler) with its token streams saved - the 70 training switch chains and
# 200 of the bridge (follow-up) dialogues. The right answers are the rejection-sampled data for memfit --objective ce
# --student mix with the pooler AND the LoRA training (queued once memcap shows which bound is the spec).
if ! pgrep -f "mixcolkee[p].sh" >/dev/null && ! grep -q "MIXCOL_JOB_DONE" /root/mixcol.log 2>/dev/null; then
  cat > /root/mixcolkeep.sh <<'MC2'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; CK=/root/mtg1_s300.safetensors
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
until [ -e /root/.mixcol_go ]; do sleep 60; done   # held: started by hand once memwin shows the spec
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
head -200 /root/work/mt_train_bridge.jsonl > /root/work/mt_train_bridge200.jsonl
for v in "ch:/root/work/mt_train_chain.jsonl:70" "br:/root/work/mt_train_bridge200.jsonl:200"; do T=${v%%:*}; r=${v#*:}; F=${r%%:*}; N=${r#*:}
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/eval300.jsonl /root/work/mixcol_$T.jsonl --multiturn $F --mt-mode mix --mt-save-tokens 1 --n $N $EVARGS --tag "[mixcol-$T]" > /root/mixcol_$T.log 2>&1
  echo "[mixcol] $T: $(grep -E 'EVAL_DONE|Error|Traceback' /root/mixcol_$T.log | tail -1 | cut -c1-200); $(python3 -c "
import json; r=[json.loads(l) for l in open('/root/work/mixcol_$T.jsonl')]; print(sum(x['correct'] for x in r), 'right of', len(r), 'turns;', sum(x['correct'] for x in r if x['turn']>0), 'right after turn 1')")"
  hf upload $R /root/work/mixcol_$T.jsonl pooler_distill/chatsft/multiturn/mixcol_$T.jsonl >/dev/null 2>&1
done
echo "MIXCOL_JOB_DONE $(date -u)"
MC2
  setsid nohup bash -c 'bash /root/mixcolkeep.sh 2>&1 | tee -a /root/mixcol.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MIXCOL_LAUNCHED $(date -u)"
fi
if [ ! -e /root/.unfin_v1 ]; then touch /root/.unfin_v1
  python3 - > /root/unfin_report.txt 2>&1 <<'PYU'
import json, re, statistics, collections
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/root/gptq_hf_gq14")
INFO = re.compile(r"<information>.*?</information>", re.S)
rows = [json.loads(l) for l in open("/root/online_mtg1/rollouts.jsonl") if l.strip()]
print("fields:", sorted(rows[-1].keys()))
for lo, hi in ((1, 100), (101, 200), (201, 300)):
    x = [r for r in rows if lo <= r["step"] <= hi]
    if not x: continue
    un = [r for r in x if "</think>" not in r["text"]]
    why = collections.Counter()
    L_all, L_own = [], []
    for r in un:
        t = r["text"]; own = len(tok(INFO.sub("", t)).input_ids); full = len(tok(t).input_ids)
        L_all.append(full); L_own.append(own)
        nq = len(r.get("queries") or [])
        if r.get("dead"): why["loop (dead)"] += 1
        elif nq >= 7: why["7-search guard"] += 1
        elif own >= 1950: why["token cap 2000"] += 1
        else: why["time budget / other"] += 1
    fin = [r for r in x if "</think>" in r["text"]]
    fl = [len(tok(r["text"].split("</think>")[0]).input_ids) for r in fin]
    print(f"steps {lo}-{hi}: {len(x)} rollouts, unfinished {len(un)} ({100*len(un)/len(x):.0f}%) {dict(why)}")
    if un: print(f"   unfinished length tokens: own median {statistics.median(L_own):.0f} max {max(L_own)} | incl. pages median {statistics.median(L_all):.0f} max {max(L_all)}")
    if fl: print(f"   finished thinking tokens incl. pages: median {statistics.median(fl):.0f}, 90th pct {sorted(fl)[int(.9*len(fl))]}, max {max(fl)}")
PYU
  grep -c "hit the .*batch budget" /root/mtg1_run.log | sed 's/^/budget warnings in the run log: /' >> /root/unfin_report.txt
  grep "hit the .*batch budget" /root/mtg1_run.log | tail -5 >> /root/unfin_report.txt
  hf upload baya1116/hypernet-sp-distill /root/unfin_report.txt pooler_distill/chatsft/audit/unfin_report.txt >/dev/null 2>&1; echo "UNFIN_REPORT uploaded"
fi
if [ ! -e /root/.unfin_v2 ]; then touch /root/.unfin_v2
  python3 - > /root/unfin_report2.txt 2>&1 <<'PYU'
import json, collections
rows = [json.loads(l) for l in open("/root/online_mtg1/rollouts.jsonl") if l.strip()]
for lo, hi in ((1, 100), (101, 200), (201, 300)):
    un = [r for r in rows if lo <= r["step"] <= hi and "</think>" not in r["text"]]
    print(f"steps {lo}-{hi}: unfinished {len(un)}; why {dict(collections.Counter(str(r.get('why'))[:60] for r in un).most_common(6))}; searches {dict(sorted(collections.Counter(r.get('ns') for r in un).items()))}")
    fin = [r for r in rows if lo <= r["step"] <= hi and "</think>" in r["text"]]
    print(f"   finished: searches {dict(sorted(collections.Counter(r.get('ns') for r in fin).items()))}")
r = next(r for r in reversed(rows) if "</think>" not in r["text"]); print("--- one unfinished tail ---"); print(r["text"][-700:])
PYU
  hf upload baya1116/hypernet-sp-distill /root/unfin_report2.txt pooler_distill/chatsft/audit/unfin_report2.txt >/dev/null 2>&1; echo "UNFIN_REPORT2 uploaded"
fi
if [ ! -e /root/.pqjudge_v5 ]; then touch /root/.pqjudge_v5; pkill -f "pqjudgekee[p].sh"; sed -i "/PQJUDGE_DONE/d" /root/pqjudge.log 2>/dev/null; echo "PQJUDGE_RESTART (second draws) $(date -u)"; fi
if ! pgrep -f "pqjudgekee[p].sh" >/dev/null && ! grep -q "PQJUDGE_DONE" /root/pqjudge.log 2>/dev/null; then
  cat > /root/pqjudgekeep.sh <<'PJ'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work
for f in dolphin_v1.jsonl dolphin_v2.jsonl; do [ -s /root/hfdl/pooler_distill/chatsft/$f ] || hf download $R --include "pooler_distill/chatsft/$f" --local-dir /root/hfdl >/dev/null 2>&1; done
python3 - <<'PD'
import json, random
v1 = [json.loads(l) for l in open("/root/hfdl/pooler_distill/chatsft/dolphin_v1.jsonl") if l.strip()]
v1 = [r for r in v1 if r.get("q") and r.get("reply")]
v2q = set((json.loads(l).get("q") or "").strip() for l in open("/root/hfdl/pooler_distill/chatsft/dolphin_v2.jsonl") if l.strip())
cand = [r for r in v1 if r["q"].strip() not in v2q]; random.Random(0).shuffle(cand)
with open("/root/work/dolphin_heldout100_pq.jsonl", "w") as o:
    for r in cand[:100]: o.write(json.dumps({"q": r["q"], "ref": r["reply"]}, ensure_ascii=False) + "\n")
PD
left="pqf pqm pqf2 pqm2 pqe pqt"
while [ -n "$left" ]; do
  nl=""
  for T in $left; do
    [ -s /root/work/${T}_dolphin_judged.jsonl ] && continue   # judged already
    hf download $R --include "pooler_distill/pooler4bit/${T}_dolphin.jsonl" --local-dir /root/pqj >/dev/null 2>&1
    f=/root/pqj/pooler_distill/pooler4bit/${T}_dolphin.jsonl
    if [ -s $f ] && [ "$(wc -l < $f)" -ge 100 ]; then
      OAI_KEY=$(cat /root/.oai 2>/dev/null) python3 - "$f" "$T" <<'PYJ'
import json, os, sys, urllib.request, time
from concurrent.futures import ThreadPoolExecutor
src = open("/root/work/online_loop.py").read(); i = src.index("REASON_SYS = "); j = src.index('"""', src.index('"""', i) + 3) + 3
ns = {}; exec(src[i:j], ns); SYS = ns["REASON_SYS"]
ref = {json.loads(l)["q"].strip(): json.loads(l)["ref"] for l in open("/root/work/dolphin_heldout100_pq.jsonl") if l.strip()}
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
key = os.environ.get("OAI_KEY", "")
def judge(r):
    t = r["text"]; reply = t.split("</think>")[-1].strip() if "</think>" in t else ""
    if not reply: return 0, "unfinished"
    body = {"model": "gpt-5-nano", "max_completion_tokens": 2000, "messages": [{"role": "system", "content": SYS},
            {"role": "user", "content": f"QUESTION:\n{r['q'][:2000]}\n\nREFERENCE ANSWER:\n{ref.get(r['q'].strip(), '')[:3000]}\n\nASSISTANT ANSWER:\n{reply[:3000]}"}]}
    err = "?"
    for _ in range(3):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}), timeout=180))
            c = d["choices"][0]["message"].get("content") or ""; v = json.loads(c[c.find("{"): c.rfind("}") + 1])
            return int(all(bool(v.get(k)) for k in ("solves_it", "follows_the_request", "language_english", "clean"))), v
        except Exception as e: time.sleep(3); err = type(e).__name__
    return 0, {"error": err}
with ThreadPoolExecutor(max_workers=8) as ex: res = list(ex.map(judge, rows))
ok = sum(a for a, _ in res); unf = sum(1 for _, v in res if v == "unfinished"); errs = sum(1 for _, v in res if isinstance(v, dict) and "error" in v)
print(f"PQJ {sys.argv[2]} DOLPHIN_ACC {100*ok/max(len(rows),1):.1f}% ({ok}/{len(rows)}) unfinished {unf} judge errors {errs}", flush=True)
with open(f"/root/work/{sys.argv[2]}_dolphin_judged.jsonl", "w") as o:
    for r, (a, v) in zip(rows, res): o.write(json.dumps({"q": r["q"], "pass": a, "why": v, "text": r["text"]}, ensure_ascii=False) + "\n")
PYJ
      hf upload $R /root/work/${T}_dolphin_judged.jsonl pooler_distill/pooler4bit/${T}_dolphin_judged.jsonl >/dev/null 2>&1
    else nl="$nl $T"; fi
  done
  left=$(echo $nl); [ -n "$left" ] && sleep 600
done
echo "PQJUDGE_DONE $(date -u)"
PJ
  setsid nohup bash -c 'bash /root/pqjudgekeep.sh 2>&1 | tee -a /root/pqjudge.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "PQJUDGE_LAUNCHED $(date -u)"
fi
if [ ! -e /root/.mtg1_swap ]; then touch /root/.mtg1_swap; pkill -f "mtg1kee[p].sh"; sleep 2; echo "MTG1_SWAP restarts the waiting GRPO keeper (init now chosen from mem10's single-turn result) $(date -u)"; fi
MTG1=${MTG1:-1}
if [ "$MTG1" = 1 ] && ! pgrep -f "mtg1kee[p].sh" >/dev/null && ! grep -q "MTG1_JOB_DONE" /root/mtg1.log 2>/dev/null; then
  cat > /root/mtg1keep.sh <<'MG'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; B=/root/reeval_g14m_pooler.safetensors; OUT=/root/online_mtg1
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
until grep -qE "MEM10D_JOB_DONE|MEM10D_ABORT" /root/mem10d.log 2>/dev/null; do sleep 120; done
while pgrep -f "pool_eval.p[y]|memfit.p[y]|online_loop.p[y]" >/dev/null; do sleep 30; done
python3 - <<'PY2'
import json, collections
held = set()
for f in ("/root/work/eval300.jsonl", "/root/work/mt_eval_chain.jsonl", "/root/work/mt_eval_sw.jsonl"):
    for l in open(f):
        d = json.loads(l); held.add(d.get("q", "").strip()); held |= {t["q"].strip() for t in d.get("turns", [])}
D = collections.defaultdict(list)
for f in ("/root/work/rft10.jsonl", "/root/work/rft8.jsonl"):
    for l in open(f):
        r = json.loads(l); D[(f, r["dialog"])].append(r)
n = nh = 0
with open("/root/work/mtg_items.jsonl", "w") as fh:
    for rs in D.values():
        hist = []
        for r in sorted(rs, key=lambda r: r["turn"]):
            q = r["q"].strip()
            if q not in held and r.get("gold"):
                fh.write(json.dumps({"q": q, "gold": r["gold"], "hist": hist}, ensure_ascii=False) + "\n"); n += 1; nh += bool(hist)
            hist = hist + [{"role": "user", "content": q}, {"role": "assistant", "content": (r.get("reply") or "").strip() or "(no reply)"}]
print(f"[mtg1] {n} training turns ({nh} with history) from {len(D)} conversations")
PY2
# start from mem10 when it kept the single-turn search (its three shards average >= 45, base 48.0): mem10 closed most
# of the switch gap on the 30 chains (turn 2: 10 -> 17 of 30, alone 18-19); otherwise from the base
ST=$(grep -hoE "EVAL_DONE\[mem10st[0-2]\] n=34 correct=[0-9.]+" /root/mem10st_*.log 2>/dev/null | grep -oE "[0-9.]+$" | awk '{s+=$1;n++} END{if(n==3) printf "%.1f", s/n}')
INIT=/root/gptq_hf_gq14; PI="--pooler-init $B"
if [ -n "$ST" ] && awk "BEGIN{exit !($ST >= 45.0)}" && [ -s /root/pooler_mem10.safetensors ]; then INIT=/root/pooler_mem10.safetensors; PI=""; fi
grpo() { env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OAI_KEY=$(cat /root/.oai 2>/dev/null) DSK_KEY=$(cat /root/.dsk 2>/dev/null) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /root/work/online_loop.py $1 $OUT $2 --pooler none --lora-layers all --lora-rank 16 \
  --mt-items /root/work/mtg_items.jsonl --heldout /root/work/eval300.jsonl --reason-g 8 --steps 100 --save-every 25 \
  --search-lr 1e-5 --search-temp 0.9 --search-gen 2000 --temp 0.6 --gen 2000 --budget 900 --maxsrch 7 --stop eos \
  --judge-api openai --judge-model gpt-5-nano --w-talk 0.5 --dolphin-min 0 --adv-std 1 --pg-norm mean --kl 0 \
  --guard 1 --guard-steps 20 >> /root/mtg1_run.log 2>&1; }
echo "[mtg1] grpo start $(date -u +%H:%M), init $INIT (mem10 single-turn mean ${ST:-n/a})"
grpo $INIT "$PI"
if [ "$INIT" != /root/gptq_hf_gq14 ] && grep -q "ONLINE_ABORT" /root/mtg1_run.log; then
  echo "[mtg1] mem10 checkpoint refused ($(grep -m1 ONLINE_ABORT /root/mtg1_run.log | cut -c1-160)); from the base instead"; rm -rf $OUT; grpo /root/gptq_hf_gq14 "--pooler-init $B"
fi
grep -E "^\[data\]|^\[init\]|^\[cfg\]|ONLINE_|Error|Traceback" /root/mtg1_run.log | tail -6 | cut -c1-250
grep -E "^\[step" /root/mtg1_run.log | tail -3 | cut -c1-250
[ -s $OUT/latest.safetensors ] || { echo "MTG1_ABORT: $(tail -3 /root/mtg1_run.log | tr '\n' ' ' | cut -c1-300)"; exit 1; }
hf upload $R $OUT/latest.safetensors pooler_distill/chatsft/multiturn/mtg1_latest.safetensors >/dev/null 2>&1
hf upload $R $OUT/rollouts.jsonl pooler_distill/chatsft/multiturn/mtg1_rollouts.jsonl >/dev/null 2>&1
env $ENV python3 /root/work/pool_eval.py $OUT/latest.safetensors /root/work/eval300.jsonl /root/work/mtg1_chain.jsonl --multiturn /root/work/mt_eval_chain.jsonl --mt-mode full --n 1000 $EVARGS --tag "[mtg1-chain]" > /root/mtg1_chain.log 2>&1
echo "[mtg1] $(grep -E "EVAL_DONE|Error|Traceback" /root/mtg1_chain.log | tail -1 | cut -c1-250)"; hf upload $R /root/work/mtg1_chain.jsonl pooler_distill/chatsft/multiturn/mtg1_chain.jsonl >/dev/null 2>&1
env $ENV python3 /root/work/pool_eval.py $OUT/latest.safetensors /root/work/ev_0.jsonl /root/work/mtg1st_out_0.jsonl --n 34 $EVARGS --tag "[mtg1st0]" > /root/mtg1st_0.log 2>&1
echo "[mtg1] $(grep -E "EVAL_DONE|Error|Traceback" /root/mtg1st_0.log | tail -1 | cut -c1-250)"
echo "MTG1_JOB_DONE $(date -u)"
MG
  chmod +x /root/mtg1keep.sh
  setsid nohup bash -c 'bash /root/mtg1keep.sh 2>&1 | tee -a /root/mtg1.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MTG1_LAUNCHED $(date -u)"
fi
if [ "$MEM10C" = 1 ] && ! pgrep -f "mem10ckee[p].sh" >/dev/null && ! grep -q "MEM10C_JOB_DONE" /root/mem10c.log 2>/dev/null; then
  cat > /root/mem10ckeep.sh <<'M10'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; T=mem10; B=/root/reeval_g14m_pooler.safetensors
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 SP_LASTLOGIT=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
mt() { env $ENV python3 /root/work/pool_eval.py $1 /root/work/eval300.jsonl /root/work/$2.jsonl --multiturn $3 --mt-mode $4 --n 1000 $5 $EVARGS --tag "[$2]" > /root/$2.log 2>&1
  echo "[mem10c] $(grep -E "EVAL_DONE|Error|Traceback" /root/$2.log | tail -1 | cut -c1-250) $(date -u +%H:%M)"; }
st() { env $ENV python3 /root/work/pool_eval.py /root/pooler_$T.safetensors /root/work/ev_$1.jsonl /root/work/${T}st_out_$1.jsonl --n 34 $EVARGS --tag "[${T}st$1]" > /root/${T}st_$1.log 2>&1
  echo "[mem10c] $(grep -E "EVAL_DONE|Error|Traceback" /root/${T}st_$1.log | tail -1 | cut -c1-250)"; hf upload $R /root/work/${T}st_out_$1.jsonl pooler_distill/chatsft/rollouts/${T}st_$1.jsonl >/dev/null 2>&1; }
cat_up() { o=/root/work/$1.jsonl; cat /root/work/$1_$2*.jsonl > $o; echo "[mem10c] $1: $(wc -l < $o) rows"; hf upload $R $o pooler_distill/chatsft/multiturn/$1.jsonl >/dev/null 2>&1; }
gpu() { echo "[mem10c] gpu $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader) $(date -u +%H:%M)"; }
echo "[mem10c] start $(date -u +%H:%M)"
( while pgrep -f "pool_eval.py .*rft10_s0.jsonl" >/dev/null; do sleep 30; done; mt $B rft10_s2 /root/work/mt_train_c10_s2.jsonl none "--mt-save-tokens 1" ) &
( mt $B rft10_s1 /root/work/mt_train_c10_s1.jsonl none "--mt-save-tokens 1"; mt $B rft10_s3 /root/work/mt_train_c10_s3.jsonl none "--mt-save-tokens 1" ) &
sleep 300; gpu
wait; while pgrep -f "pool_eval.p[y]" >/dev/null; do sleep 30; done; cat_up rft10 s
echo "[mem10c] memfit (ce, native) start $(date -u +%H:%M)"
env SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /root/work/memfit.py --objective ce --ce-native 1 --student full --ckpt $B --data /root/work/rft10.jsonl --out /root/pooler_$T.safetensors --log /root/memfit_$T.log --steps 600 --val-every 100 --lr 0 --lr-lora 5e-6 > /root/memfit_${T}_run.log 2>&1
grep -E "^\[memfit\]|^\[data\]|^val |MEMFIT_DONE|Error|Traceback" /root/memfit_${T}_run.log | tail -12 | cut -c1-200
[ -s /root/pooler_$T.safetensors ] || { echo "MEM10C_ABORT memfit: $(tail -3 /root/memfit_${T}_run.log | tr '\n' ' ' | cut -c1-300)"; exit 1; }
hf upload $R /root/pooler_$T.safetensors pooler_distill/chatsft/multiturn/pooler_$T.safetensors >/dev/null 2>&1
( mt /root/pooler_$T.safetensors ${T}_chain_h0 /root/work/mt_eval_chain_h0.jsonl full ""; st 0; st 2 ) &
sleep 120
( mt /root/pooler_$T.safetensors ${T}_chain_h1 /root/work/mt_eval_chain_h1.jsonl full ""; st 1 ) &
wait; cat_up ${T}_chain h
echo "MEM10C_JOB_DONE $(date -u)"
M10
  chmod +x /root/mem10ckeep.sh
  setsid nohup bash -c 'bash /root/mem10ckeep.sh 2>&1 | tee -a /root/mem10c.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MEM10C_LAUNCHED $(date -u)"
fi
if [ "$MEM10B" = 1 ] && ! pgrep -f "mem10bkee[p].sh" >/dev/null && ! grep -q "MEM10B_JOB_DONE" /root/mem10b.log 2>/dev/null; then
  cat > /root/mem10bkeep.sh <<'M10'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; T=mem10; B=/root/reeval_g14m_pooler.safetensors
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
python3 - <<'PY3'
L = open("/root/work/mt_train_c10.jsonl").readlines()
for i in range(4): open(f"/root/work/mt_train_c10_s{i}.jsonl", "w").writelines(L[i::4])
L = open("/root/work/mt_eval_chain.jsonl").readlines()
for i in range(2): open(f"/root/work/mt_eval_chain_h{i}.jsonl", "w").writelines(L[i::2])
PY3
mt() { env $ENV python3 /root/work/pool_eval.py $1 /root/work/eval300.jsonl /root/work/$2.jsonl --multiturn $3 --mt-mode $4 --n 1000 $5 $EVARGS --tag "[$2]" > /root/$2.log 2>&1
  echo "[mem10b] $(grep -E "EVAL_DONE|Error|Traceback" /root/$2.log | tail -1 | cut -c1-250) $(date -u +%H:%M)"; }
st() { env $ENV python3 /root/work/pool_eval.py /root/pooler_$T.safetensors /root/work/ev_$1.jsonl /root/work/${T}st_out_$1.jsonl --n 34 $EVARGS --tag "[${T}st$1]" > /root/${T}st_$1.log 2>&1
  echo "[mem10b] $(grep -E "EVAL_DONE|Error|Traceback" /root/${T}st_$1.log | tail -1 | cut -c1-250)"; hf upload $R /root/work/${T}st_out_$1.jsonl pooler_distill/chatsft/rollouts/${T}st_$1.jsonl >/dev/null 2>&1; }
cat_up() { o=/root/work/$1.jsonl; cat /root/work/$1_$2*.jsonl > $o; echo "[mem10b] $1: $(wc -l < $o) rows"; hf upload $R $o pooler_distill/chatsft/multiturn/$1.jsonl >/dev/null 2>&1; }
while pgrep -f "pool_eval.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
echo "[mem10b] collection start $(date -u +%H:%M) (two lanes)"
( mt $B rft10_s0 /root/work/mt_train_c10_s0.jsonl none "--mt-save-tokens 1"; mt $B rft10_s2 /root/work/mt_train_c10_s2.jsonl none "--mt-save-tokens 1" ) &
sleep 120
( mt $B rft10_s1 /root/work/mt_train_c10_s1.jsonl none "--mt-save-tokens 1"; mt $B rft10_s3 /root/work/mt_train_c10_s3.jsonl none "--mt-save-tokens 1" ) &
wait; cat_up rft10 s
echo "[mem10b] memfit (ce, native) start $(date -u +%H:%M)"
env SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /root/work/memfit.py --objective ce --ce-native 1 --student full --ckpt $B --data /root/work/rft10.jsonl --out /root/pooler_$T.safetensors --log /root/memfit_$T.log --steps 600 --val-every 100 --lr 0 --lr-lora 5e-6 > /root/memfit_${T}_run.log 2>&1
grep -E "^\[memfit\]|^\[data\]|^val |MEMFIT_DONE|Error|Traceback" /root/memfit_${T}_run.log | tail -12 | cut -c1-200
[ -s /root/pooler_$T.safetensors ] || { echo "MEM10B_ABORT memfit: $(tail -3 /root/memfit_${T}_run.log | tr '\n' ' ' | cut -c1-300)"; exit 1; }
hf upload $R /root/pooler_$T.safetensors pooler_distill/chatsft/multiturn/pooler_$T.safetensors >/dev/null 2>&1
( mt /root/pooler_$T.safetensors ${T}_chain_h0 /root/work/mt_eval_chain_h0.jsonl full ""; st 0; st 2 ) &
sleep 120
( mt /root/pooler_$T.safetensors ${T}_chain_h1 /root/work/mt_eval_chain_h1.jsonl full ""; st 1 ) &
wait; cat_up ${T}_chain h
echo "MEM10B_JOB_DONE $(date -u)"
M10
  chmod +x /root/mem10bkeep.sh
  setsid nohup bash -c 'bash /root/mem10bkeep.sh 2>&1 | tee -a /root/mem10b.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MEM10B_LAUNCHED $(date -u)"
fi
if [ "$MEM10" = 1 ] && ! pgrep -f "mem10kee[p].sh" >/dev/null && ! grep -q "MEM10_JOB_DONE" /root/mem10.log 2>/dev/null; then
  cat > /root/mem10keep.sh <<'M10'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; T=mem10; B=/root/reeval_g14m_pooler.safetensors
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
[ -s /root/work/mt_eval_c100.jsonl ] || python3 - <<'PY2'
import json, random
used = set()
for l in open("/root/work/eval300.jsonl"): used.add(json.loads(l).get("q", "").strip())
for f in ("/root/work/mt_train.jsonl", "/root/work/mt_train_sw.jsonl", "/root/work/mt_train_chain.jsonl", "/root/work/mt_eval.jsonl", "/root/work/mt_eval_sw.jsonl"):
    for l in open(f):
        d = json.loads(l); used.add(d.get("seed", "")); used |= {t["q"].strip() for t in d["turns"]}
seeds = []
for l in open("/root/work/selfq_all.jsonl"):
    try: r = json.loads(l)
    except Exception: continue
    if r.get("q") and r.get("gold") and r["q"].strip() not in used: seeds.append((r["q"].strip(), r["gold"].strip()))
seeds = list(dict.fromkeys(seeds)); random.Random(41).shuffle(seeds)
ne = min(100, len(seeds) // 12); ch = [seeds[4 * k: 4 * k + 4] for k in range(len(seeds) // 4)]
def dump(path, chains, pre):
    with open(path, "w") as fh:
        for k, qs in enumerate(chains):
            fh.write(json.dumps({"id": f"{pre}{k:03d}", "kind": "switch", "turns": [{"q": q, "gold": g, "standalone": q} for q, g in qs]}, ensure_ascii=False) + "\n")
dump("/root/work/mt_eval_c100.jsonl", ch[:ne], "e10_"); dump("/root/work/mt_train_c10.jsonl", ch[ne:], "t10_")
print(f"[mem10] {len(seeds)} free seeds: {ne} held-out chains of 4, {len(ch) - ne} training chains of 4")
PY2
# Parallel: one batch-1 decoder leaves the card mostly idle, so the work runs as shards, each its own process, launched
# one at a time while the card has room (>= 3.5 GB free VRAM, >= 6 GB free RAM, at most MAXP at once); the held-out
# shards go first. 15 h of serial evaluation becomes a few hours.
MAXP=8; NS=4
python3 - <<'PY3'
import json
for src, pre in (("/root/work/mt_eval_c100.jsonl", "/root/work/mt_eval_c100_s"), ("/root/work/mt_train_c10.jsonl", "/root/work/mt_train_c10_s")):
    L = open(src).readlines()
    for i in range(4): open(f"{pre}{i}.jsonl", "w").writelines(L[i::4])
PY3
room() { while :; do n=$(pgrep -fc "pool_eval.p[y]"); fv=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); fr=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  [ "$n" -lt $MAXP ] && [ "${fv:-0}" -ge 3500 ] && [ "${fr:-0}" -ge 6000 ] && return 0; sleep 30; done; }
run() { room; env $ENV nohup python3 /root/work/pool_eval.py $1 /root/work/eval300.jsonl /root/work/$2.jsonl --multiturn $3 --mt-mode $4 --n 1000 $5 $EVARGS --tag "[$2]" > /root/$2.log 2>&1 &
  echo "[mem10] launched $2 $(date -u +%H:%M) (running $(pgrep -fc "pool_eval.p[y]"), free VRAM $(nvidia-smi --query-gpu=memory.free --format=csv,noheader | head -1))"; sleep 150; }
merge() { o=/root/work/$1.jsonl; : > $o; for i in $(seq 0 $((NS-1))); do cat /root/work/$1_s$i.jsonl >> $o 2>/dev/null; done
  grep -hE "Error|Traceback" /root/$1_s*.log | tail -2 | cut -c1-200; echo "[mem10] $1 merged: $(wc -l < $o) rows"; hf upload $R $o pooler_distill/chatsft/multiturn/$1.jsonl >/dev/null 2>&1; }
waitall() { while pgrep -f "pool_eval.p[y]" >/dev/null; do sleep 60; done; }
waitall
echo "[mem10] start $(date -u +%H:%M)"
for i in $(seq 0 $((NS-1))); do run $B base_c100_full_s$i /root/work/mt_eval_c100_s$i.jsonl full ""; run $B base_c100_alone_s$i /root/work/mt_eval_c100_s$i.jsonl none ""; done
for i in $(seq 0 $((NS-1))); do run $B rft10_s$i /root/work/mt_train_c10_s$i.jsonl none "--mt-save-tokens 1"; done
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/[mem10] gpu /'
while [ $(ls /root/base_c100_*_s*.log 2>/dev/null | xargs grep -l "EVAL_DONE" 2>/dev/null | wc -l) -lt $((2*NS)) ] && pgrep -f "base_c100_.*_s[0-9]" >/dev/null; do sleep 120; done
merge base_c100_full; merge base_c100_alone; echo "[mem10] held-out base done $(date -u +%H:%M)"
waitall; merge rft10
echo "[mem10] memfit (ce, native) start $(date -u +%H:%M)"
env SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /root/work/memfit.py --objective ce --ce-native 1 --student full --ckpt $B --data /root/work/rft10.jsonl --out /root/pooler_$T.safetensors --log /root/memfit_$T.log --steps 600 --val-every 100 --lr 0 --lr-lora 5e-6 > /root/memfit_${T}_run.log 2>&1
grep -E "^\[memfit\]|^\[data\]|^val |MEMFIT_DONE|Error|Traceback" /root/memfit_${T}_run.log | tail -12 | cut -c1-200
[ -s /root/pooler_$T.safetensors ] || { echo "MEM10_ABORT memfit: $(tail -3 /root/memfit_${T}_run.log | tr '\n' ' ' | cut -c1-300)"; exit 1; }
hf upload $R /root/pooler_$T.safetensors pooler_distill/chatsft/multiturn/pooler_$T.safetensors >/dev/null 2>&1
for i in $(seq 0 $((NS-1))); do run /root/pooler_$T.safetensors ${T}_c100_full_s$i /root/work/mt_eval_c100_s$i.jsonl full ""; done
for i in 0 1 2; do room; env $ENV nohup python3 /root/work/pool_eval.py /root/pooler_$T.safetensors /root/work/ev_$i.jsonl /root/work/${T}st_out_$i.jsonl --n 34 $EVARGS --tag "[${T}st$i]" > /root/${T}st_$i.log 2>&1 & sleep 150; done
waitall; merge ${T}_c100_full
for i in 0 1 2; do grep -E "EVAL_DONE|Error|Traceback" /root/${T}st_$i.log | tail -1 | cut -c1-300; hf upload $R /root/work/${T}st_out_$i.jsonl pooler_distill/chatsft/rollouts/${T}st_$i.jsonl >/dev/null 2>&1; done
echo "MEM10_JOB_DONE $(date -u)"
M10
  chmod +x /root/mem10keep.sh
  setsid nohup bash -c 'bash /root/mem10keep.sh 2>&1 | tee -a /root/mem10.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MEM10_LAUNCHED $(date -u)"
fi

# The device layout of the local search, as a side job on the CPU: the int8 embedder, the IVF layout of the sign
# index, the memory and time of the search as its own process on the whole store, all uploaded beside the store.
IVF=${IVF:-1}
if [ "$IVF" = 1 ] && ! pgrep -f "ivfkee[p].sh" >/dev/null && ! grep -q IVF_JOB_DONE /root/ivf.log 2>/dev/null && [ -s /root/wiki_store/emb.bin ]; then
  cat > /root/ivfkeep.sh <<'IK'
#!/bin/bash
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; ST=/root/wiki_store; M=/root/bge-small
pip install -q onnxruntime onnx 2>&1 | grep -v WARNING | tail -1
[ -s $M/model.onnx ] || cp $M/onnx/model.onnx $M/model.onnx 2>/dev/null
[ -s $M/model_int8.onnx ] || python3 -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('$M/model.onnx', '$M/model_int8.onnx', weight_type=QuantType.QInt8); print('[ivf] int8 embedder written')"
[ -s $ST/emb_ivf.bin ] && [ -s $ST/ivf_offsets.npy ] || python3 /root/work/localsearch/ivf.py --store $ST --k 2048 2>&1 | grep -E "^\[ivf\]|IVF_DONE|Error"
[ -s $ST/lex.sqlite ] || python3 -c "import sys; sys.path.insert(0,'/root/work/localsearch'); from search import build_lex_index; build_lex_index('$ST')"
OMP_NUM_THREADS=4 python3 /root/work/localsearch/memcheck.py --store $ST --model $M --queries /root/work/ev_0.jsonl --n 40 2>&1 | grep -E "MEMCHECK|Error"
SP_LOCAL_NPROBE=16 OMP_NUM_THREADS=4 python3 /root/work/localsearch/memcheck.py --store $ST --model $M --queries /root/work/ev_0.jsonl --n 40 2>&1 | grep -E "MEMCHECK|Error"
for f in ivf_centroids.npy ivf_order.npy ivf_offsets.npy emb_ivf.bin; do hf upload $R $ST/$f localsearch/wiki_en_20231101/$f >/dev/null 2>&1; done
hf upload $R $M/model_int8.onnx localsearch/bge-small-en-v1.5/model_int8.onnx >/dev/null 2>&1
for f in tokenizer.json tokenizer_config.json config.json special_tokens_map.json; do hf upload $R $M/$f localsearch/bge-small-en-v1.5/$f >/dev/null 2>&1; done
echo "IVF_JOB_DONE $(date -u)"
IK
  chmod +x /root/ivfkeep.sh
  setsid nohup bash -c 'bash /root/ivfkeep.sh 2>&1 | tee -a /root/ivf.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "IVF_LAUNCHED $(date -u)"
fi

# The dedicated retriever: bge-small fine-tuned on the lineage's own (query, page) pairs, then the dump re-embedded
# with it into a second store beside the first (the live one keeps serving), IVF rebuilt, the 400 test queries
# scored, everything uploaded. Waits for the GPU (no evaluation running). RETR_TAG bumps to redo.
RETRIEVER=${RETRIEVER:-1}; RETR_TAG=${RETR_TAG:-ft1}
if [ "$RETRIEVER" = 1 ] && ! pgrep -f "retrkee[p].sh" >/dev/null && ! grep -q "RETR_JOB_DONE $RETR_TAG" /root/retr.log 2>/dev/null && [ -s /root/work/localsearch/data/train_negs.json ] && [ -s /root/wiki_store/emb_ivf.bin ]; then
  cat > /root/retrkeep.sh <<RK1
RTAG=$RETR_TAG
RK1
  cat >> /root/retrkeep.sh <<'RK2'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; ST=/root/wiki_store; M=/root/bge-small
L=/root/work/localsearch; FT=/root/bge_$RTAG; ST2=/root/wiki_store_$RTAG
while pgrep -f "pool_eval.py" >/dev/null; do sleep 120; done
echo "[retr] $RTAG start $(date -u +%H:%M)"
if ! [ -s $FT/model_int8.onnx ]; then
  python3 $L/train_retriever.py --base $M --store $ST --pairs $L/data/train_pairs.json --negs $L/data/train_negs.json --out $FT --epochs 2 --bs 32 2>&1 | grep -E "^\[retriever\]|step [0-9]*0/|TRAIN_DONE|Error|Traceback" | tail -40
  grep -q . $FT/config.json 2>/dev/null || { echo "RETR_ABORT $RTAG: training left no model"; exit 1; }
  python3 $L/export_onnx.py $FT $FT 2>&1 | grep -E "EXPORT_DONE|Error|Traceback"
  for f in model.safetensors model_int8.onnx tokenizer.json tokenizer_config.json config.json special_tokens_map.json vocab.txt; do hf upload $R $FT/$f localsearch/bge-small-$RTAG/$f >/dev/null 2>&1; done
  echo "[retr] model uploaded $(date -u +%H:%M)"
fi
# the ranking alone with the trained model, on the live index, before the re-embedding
SP_LOCAL_RERANK_MODEL=$FT OMP_NUM_THREADS=4 python3 $L/test_retriever.py --store $ST --model $M --tag "$RTAG-rerank-only" 2>&1 | grep -E "RETR_TEST|Error|Traceback"
OMP_NUM_THREADS=4 python3 $L/test_retriever.py --store $ST --model $M --tag "base" 2>&1 | grep -E "RETR_TEST|Error|Traceback"
mkdir -p $ST2; for f in docs.bin blocks.idx titles.txt meta.json lex.sqlite idf.pkl; do [ -e $ST2/$f ] || ln -s $ST/$f $ST2/$f; done
while pgrep -f "pool_eval.py" >/dev/null; do sleep 120; done
[ -s $ST2/emb_ivf.bin ] || python3 $L/embed.py --store $ST2 --model $FT --backend torch --batch 256 --float16-out 0 2>&1 | grep -E "^\[embed\].*(000000|done|articles in)|EMBED_DONE|Error|Traceback" | tail -20
[ -s $ST2/emb_ivf.bin ] || python3 $L/ivf.py --store $ST2 --k 2048 2>&1 | grep -E "^\[ivf\] [0-9]+ articles|IVF_DONE|Error|Traceback"
OMP_NUM_THREADS=4 python3 $L/test_retriever.py --store $ST2 --model $FT --tag "$RTAG-full" 2>&1 | grep -E "RETR_TEST|Error|Traceback"
for f in emb.bin ivf_centroids.npy ivf_order.npy ivf_offsets.npy emb_ivf.bin; do hf upload $R $ST2/$f localsearch/wiki_en_20231101_$RTAG/$f >/dev/null 2>&1; done
echo "RETR_JOB_DONE $RTAG $(date -u)"
RK2
  chmod +x /root/retrkeep.sh
  setsid nohup bash -c 'bash /root/retrkeep.sh 2>&1 | tee -a /root/retr.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "RETR_LAUNCHED $RETR_TAG $(date -u)"
fi

# The retriever trained on the frozen model's verdicts (the reward table rew1), once the table is complete and
# the card is free: train, export, the ranking-only test on the 400 test queries, upload. RANKER_TAG bumps to redo.
RANKER=${RANKER:-1}; RANKER_TAG=${RANKER_TAG:-rl2}; RANKER_ROWS=${RANKER_ROWS:-/root/work/rew1_out_0.jsonl}
if [ "$RANKER" = 1 ] && ! pgrep -f "rankkee[p].sh" >/dev/null && ! grep -q "RANKER_JOB_DONE $RANKER_TAG" /root/ranker.log 2>/dev/null && [ -s "$RANKER_ROWS" ] && [ "$(wc -l < $RANKER_ROWS)" -ge "${RANKER_MIN:-1000}" ]; then
  cat > /root/rankkeep.sh <<RK1
RTAG=$RANKER_TAG; ROWS=$RANKER_ROWS
RK1
  cat >> /root/rankkeep.sh <<'RK2'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; ST=/root/wiki_store; M=/root/bge-small
L=/root/work/localsearch; FT=/root/bge_$RTAG
while pgrep -f "pool_eval.py" >/dev/null; do sleep 120; done
echo "[ranker] $RTAG start $(date -u +%H:%M) on $(wc -l < $ROWS) rows"
python3 $L/train_ranker.py --rows $ROWS --base $M --store $ST --out $FT --epochs 4 > /root/ranker_${RTAG}_bi.log 2>&1; grep -E "^\[ranker\]|RANKER_DONE|Error|Traceback" /root/ranker_${RTAG}_bi.log | tail -20
grep -q . $FT/config.json 2>/dev/null || { echo "RANKER_ABORT $RTAG: no model written"; tail -5 /root/ranker_${RTAG}_bi.log | cut -c1-200; exit 1; }
python3 $L/export_onnx.py $FT $FT 2>&1 | grep -E "EXPORT_DONE|Error|Traceback"
for f in model.safetensors model_int8.onnx tokenizer.json tokenizer_config.json config.json special_tokens_map.json vocab.txt; do hf upload $R $FT/$f localsearch/bge-small-$RTAG/$f >/dev/null 2>&1; done
SP_LOCAL_RERANK_MODEL=$FT OMP_NUM_THREADS=4 python3 $L/test_retriever.py --store $ST --model $M --tag "$RTAG-rerank-only" 2>&1 | grep -E "RETR_TEST|Error|Traceback"
# the cross-encoder arm on the same verdicts: MiniLM-L6 (ms-marco) trained listwise, reranking the fused top 16
CE=/root/ce_$RTAG; CEB=/root/ce_base
[ -s $CEB/config.json ] || hf download cross-encoder/ms-marco-MiniLM-L-6-v2 --local-dir $CEB >/dev/null 2>&1
python3 $L/train_ranker.py --arch ce --rows $ROWS --base $CEB --store $ST --out $CE --epochs 4 --lr 2e-5 > /root/ranker_${RTAG}_ce.log 2>&1; grep -E "^\[ranker\]|RANKER_DONE|Error|Traceback" /root/ranker_${RTAG}_ce.log | tail -20
grep -q . $CE/config.json 2>/dev/null || tail -5 /root/ranker_${RTAG}_ce.log | cut -c1-200
if grep -q . $CE/config.json 2>/dev/null; then
  python3 $L/export_onnx.py $CE $CE --cls 2>&1 | grep -E "EXPORT_DONE|Error|Traceback"
  for f in model.safetensors model_int8.onnx tokenizer.json tokenizer_config.json config.json special_tokens_map.json vocab.txt; do hf upload $R $CE/$f localsearch/ce-$RTAG/$f >/dev/null 2>&1; done
  SP_LOCAL_CE=$CEB OMP_NUM_THREADS=4 python3 $L/test_retriever.py --store $ST --model $M --tag "ce-base" 2>&1 | grep -E "RETR_TEST|Error|Traceback"
  SP_LOCAL_CE=$CE OMP_NUM_THREADS=4 python3 $L/test_retriever.py --store $ST --model $M --tag "ce-$RTAG" 2>&1 | grep -E "RETR_TEST|Error|Traceback"
fi
echo "RANKER_JOB_DONE $RTAG $(date -u)"
RK2
  chmod +x /root/rankkeep.sh
  setsid nohup bash -c 'bash /root/rankkeep.sh 2>&1 | tee -a /root/ranker.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "RANKER_LAUNCHED $RANKER_TAG $(date -u)"
fi

# one-shot: the local search released under release/local-search/ - the store's files copied on the hub server side
# (LFS copies, nothing re-uploaded), the embedder, the code, a README (LSREL_SERIAL bumps to redo)
LSREL_SERIAL=1
if [ "$(cat /root/.lsrel_serial 2>/dev/null)" != "$LSREL_SERIAL" ] && ! pgrep -f "lsrelkee[p].sh" >/dev/null; then
  curl -sSf -o /root/lsrel_README.md "$RAW/release/local-search/README.md?nocache=$(date +%s)" && cat > /root/lsrelkeep.sh <<'LK'
#!/bin/bash
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
python3 - <<'PY'
import os
from huggingface_hub import HfApi, CommitOperationCopy, CommitOperationAdd
R = "baya1116/hypernet-sp-distill"; api = HfApi(); have = set(api.list_repo_files(R))
SRC = "localsearch/wiki_en_20231101"; DST = "release/local-search"
ops = []
for f in ("docs.bin", "blocks.idx", "titles.txt", "emb.bin", "lex.sqlite", "ivf_centroids.npy", "ivf_order.npy", "ivf_offsets.npy", "emb_ivf.bin"):
    if f"{SRC}/{f}" in have and f"{DST}/{f}" not in have: ops.append(CommitOperationCopy(f"{SRC}/{f}", f"{DST}/{f}"))
for f in ("model_int8.onnx", "tokenizer.json", "tokenizer_config.json", "config.json", "special_tokens_map.json", "vocab.txt"):
    src = f"localsearch/bge-small-en-v1.5/{f}"
    if src in have and f"{DST}/bge-small-en-v1.5/{f}" not in have:
        if f == "model_int8.onnx": ops.append(CommitOperationCopy(src, f"{DST}/bge-small-en-v1.5/{f}"))
        elif os.path.exists(f"/root/bge-small/{f}"): ops.append(CommitOperationAdd(f"{DST}/bge-small-en-v1.5/{f}", f"/root/bge-small/{f}"))
for f in ("meta.json",):
    if os.path.exists(f"/root/wiki_store/{f}"): ops.append(CommitOperationAdd(f"{DST}/{f}", f"/root/wiki_store/{f}"))
for f in ("store.py", "search.py", "ivf.py", "embed.py", "build_store.py", "memcheck.py", "pq.py"):
    if os.path.exists(f"/root/work/localsearch/{f}"): ops.append(CommitOperationAdd(f"{DST}/code/{f}", f"/root/work/localsearch/{f}"))
ops.append(CommitOperationAdd(f"{DST}/README.md", "/root/lsrel_README.md"))
print(f"[lsrel] {len(ops)} operations ({sum(isinstance(o, CommitOperationCopy) for o in ops)} server-side copies)", flush=True)
for attempt in range(3):
    try:
        api.create_commit(repo_id=R, operations=ops, commit_message="release: the local search"); break
    except Exception as e:
        print(f"[lsrel] attempt {attempt+1} failed: {str(e)[:200]}", flush=True)
        import time; time.sleep(60)
else:
    print("LSREL_ABORT"); raise SystemExit(1)
have = set(api.list_repo_files(R)); print("[lsrel] on the hub:", sorted(f[len(DST)+1:] for f in have if f.startswith(DST + "/")))
print("LSREL_DONE")
PY
LK
  chmod +x /root/lsrelkeep.sh; echo "$LSREL_SERIAL" > /root/.lsrel_serial
  setsid nohup bash -c 'bash /root/lsrelkeep.sh 2>&1 | tee -a /root/release.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "LSREL_LAUNCHED $(date -u)"
fi

# one-shot: the released directories, copied on the hub under release/ with a README (RELEASE_SERIAL bumps to redo)
RELEASE_SERIAL=7
if [ "$(cat /root/.release_serial 2>/dev/null)" != "$RELEASE_SERIAL" ] && ! pgrep -f "releasekee[p].sh" >/dev/null; then
  curl -sS -o /root/release_README.md "$RAW/release/README.md?nocache=$(date +%s)"
  curl -sSf -o /root/release_USAGE.md "$RAW/release/USAGE.md?nocache=$(date +%s)"; curl -sSf -o /root/release_pooler_config.json "$RAW/release/pooler_config.json?nocache=$(date +%s)"
  curl -sSf -o /root/release_spec.md "https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/docs/iphone_agent_v2_spec.md?nocache=$(date +%s)"
  cat > /root/releasekeep.sh <<'RK'
#!/bin/bash
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; W=/root/release_stage; mkdir -p $W
# room: the merged 16-bit model exists twice on the box, and g14's training directory is on the hub
if [ -d /root/hfdl/pooler_distill/chatsft/g14m_hf ] && [ ! -L /root/hfdl/pooler_distill/chatsft/g14m_hf ] && [ -s /root/reeval_hf_g14m/model.safetensors ]; then
  rm -rf /root/hfdl/pooler_distill/chatsft/g14m_hf && ln -s /root/reeval_hf_g14m /root/hfdl/pooler_distill/chatsft/g14m_hf; fi
rm -rf /root/online_g14 /root/release_stage/dl; echo "[release] $(df -h /root | awk 'NR==2{print $4" free"}')"
copy() {  # $1 hub source dir, $2 release name, $3 a local copy of it when there is one
  rm -rf $W/$2; mkdir -p $W/$2
  if [ -n "$3" ] && [ -s $3/model.safetensors ]; then cp -r $3/. $W/$2/
  else for try in 1 2 3; do hf download $R --include "$1/*" --local-dir $W/dl >/dev/null 2>&1 && break; sleep 30; done; cp -r $W/dl/$1/. $W/$2/ 2>/dev/null; fi
  [ -s $W/$2/model.safetensors ] || { echo "RELEASE_ABORT $2: $1 did not download"; return 1; }
  [ -s $W/$2/pooler.safetensors ] || cp /root/reeval_g14m_pooler.safetensors $W/$2/pooler.safetensors
  for try in 1 2 3; do hf upload $R $W/$2 release/$2 >/dev/null 2>&1 && break; sleep 30; done
  echo "[release] $2 <- $1 ($(du -sh $W/$2 | cut -f1))"; rm -rf $W/$2 $W/dl/$1
}
onhub() { python3 -c "import sys; from huggingface_hub import HfApi; sys.exit(0 if '$1' in HfApi().list_repo_files('baya1116/hypernet-sp-distill') else 1)" 2>/dev/null; }
onhub release/g14-bf16/model.safetensors || copy pooler_distill/chatsft/g14m_hf g14-bf16 /root/reeval_hf_g14m
onhub release/g14-4bit-gptq/model.safetensors || copy pooler_distill/chatsft/g14_mlx4g g14-4bit-gptq /root/gptq_mlx4_gq14
onhub release/g14-4bit-gptq-trained/model.safetensors || copy pooler_distill/chatsft/g14_mlx4gt g14-4bit-gptq-trained /root/sft_mlx4_q14g
hf upload $R /root/release_README.md release/README.md >/dev/null 2>&1 && echo "[release] README"
[ -s /root/release_USAGE.md ] && hf upload $R /root/release_USAGE.md release/USAGE.md >/dev/null 2>&1 && echo "[release] USAGE"
[ -s /root/release_spec.md ] && hf upload $R /root/release_spec.md release/docs/iphone_agent_v2_spec.md >/dev/null 2>&1 && echo "[release] spec"
for d in g14-bf16 g14-4bit-gptq g14-4bit-gptq-trained; do [ -s /root/release_pooler_config.json ] && hf upload $R /root/release_pooler_config.json release/$d/pooler_config.json >/dev/null 2>&1; done && echo "[release] pooler_config"
rm -rf $W; echo "RELEASE_DONE $(date -u)"
RK
  chmod +x /root/releasekeep.sh; echo $RELEASE_SERIAL > /root/.release_serial
  setsid nohup bash -c 'bash /root/releasekeep.sh 2>&1 | tee -a /root/release.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "RELEASE_LAUNCHED $(date -u)"
fi

# one-shot: the search's memory and time on the whole store with the current code (MEMCHECK_SERIAL bumps to redo)
MEMCHECK_SERIAL=1
if [ "$(cat /root/.memcheck_serial 2>/dev/null)" != "$MEMCHECK_SERIAL" ] && [ -s /root/wiki_store/emb_ivf.bin ] && ! pgrep -f "memcheck.p[y]" >/dev/null; then
  echo $MEMCHECK_SERIAL > /root/.memcheck_serial
  setsid nohup bash -c 'for np in 48 16; do SP_LOCAL_NPROBE=$np OMP_NUM_THREADS=4 python3 /root/work/localsearch/memcheck.py --store /root/wiki_store --model /root/bge-small --queries /root/work/ev_0.jsonl --n 40 2>&1 | grep -E "MEMCHECK|Error"; done' >> /root/ivf.log 2>&1 < /dev/null &
fi

# The status command is the same whatever mode this file is in, so it is installed once,
# before the modes - a probe that forgets to reinstall it would otherwise report a stale table.
cat > /usr/local/bin/s <<'SSD'
#!/bin/bash
# the score over the run: mean reward per block, both sides, beside what produced it
d=$(ls -td /root/online_*/ 2>/dev/null | head -1)
python3 - "$d" "${1:-20}" <<'PYS'
import json, re, statistics, sys
d, W = sys.argv[1], int(sys.argv[2])
INFO = re.compile(r"<information>.*?</information>", re.S)


def think_words(t):
    """the model's own thinking: the served pages sit in the same span and are not it"""
    return len(INFO.sub("", t.split("</think>")[0]).split())


def reply_words(t):
    return len(t.split("</think>")[-1].split()) if "</think>" in t else 0
rows = [json.loads(l) for l in open(d + "/rollouts.jsonl") if l.strip()]
mx = max(r["step"] for r in rows)
blocks = []
print(f"{d.rstrip('/').split('/')[-1]}  through step {mx}   ({W}-step blocks)")
print("  steps    | reason  mean think reply unfin | search  mean srch think reply unfin")
for b in range(0, mx, W):
    rea = [r for r in rows if b < r["step"] <= b + W and r.get("kind") == "reason"]
    sea = [r for r in rows if b < r["step"] <= b + W and r.get("kind") == "search"]
    if not rea and not sea: continue
    def col(x, s=False):
        if not x: return "    -     -    -    -    -    -" if s else "    -     -    -    -    -"
        th = statistics.median(think_words(r["text"]) for r in x)
        rp = statistics.median(reply_words(r["text"]) for r in x)
        un = 100 * sum(1 for r in x if "</think>" not in r["text"]) / len(x)
        p = 100 * sum(1 for r in x if r["reward"] >= 1.0) / len(x)
        m = sum(r["reward"] for r in x) / len(x)
        ns = f" {statistics.median([r['ns'] for r in x]):>3.0f}" if s else ""
        return f"{p:>5.0f}% {m:>+6.2f}{ns} {th:>5.0f} {rp:>5.0f} {un:>4.0f}%"
    print(f"  {b+1:>4}-{b+W:<4} | {col(rea)} | {col(sea, True)}")
    blocks.append((b, [100 * sum(1 for r in x if r["reward"] >= 1.0) / len(x) if x else None for x in (rea, sea)],
                      [sum(r["reward"] for r in x) / len(x) if x else None for x in (rea, sea)]))
# the same, smoothed: each block averaged with the one before and the one after (3 blocks = 60 steps at W=20),
# because one hard question swings a 20-step block by 20 points and the trend is invisible in the raw rows
if len(blocks) >= 3:
    print(f"  trend: each block averaged with its neighbours ({3*W} steps)")
    print("  steps    | reason pass  mean | search pass  mean")
    done = [x for x in blocks if x[0] + W <= mx]          # a block still filling is shown on its own row but never averaged into its neighbours
    for i, (b, _, _) in enumerate(blocks):
        partial = b + W > mx
        nb = [blocks[i]] if partial else [x for x in blocks[max(0, i - 1):i + 2] if x in done]
        def avg(j, k):
            v = [x[j][k] for x in nb if x[j][k] is not None]
            return sum(v) / len(v) if v else None
        f = lambda v, pct: ("    -" if v is None else (f"{v:>4.0f}%" if pct else f"{v:>+5.2f}"))
        print(f"  {b+1:>4}-{b+W:<4} |    {f(avg(1,0),1)}   {f(avg(2,0),0)} |    {f(avg(1,1),1)}   {f(avg(2,1),0)}" + (f"   (through {mx}, not averaged)" if partial else ""))
for k in ("reason", "search"):
    v = [r["reward"] for r in rows if r.get("kind") == k]
    if v: print(f"  {k}: {len(v)} samples, mean {sum(v)/len(v):+.2f}, pass {100*sum(1 for x in v if x>=1)/len(v):.0f}%")
h = mx // 2   # first half of the run against the second half: the one comparison the blocks cannot show
for k in ("reason", "search"):
    a = [r["reward"] for r in rows if r.get("kind") == k and r["step"] <= h]
    b = [r["reward"] for r in rows if r.get("kind") == k and r["step"] > h]
    if a and b:
        f = lambda v: f"pass {100*sum(1 for x in v if x>=1)/len(v):>3.0f}% mean {sum(v)/len(v):+.2f}"
        print(f"  {k:6s} halves: steps 1-{h} {f(a)} | steps {h+1}-{mx} {f(b)}")
PYS
SSD
chmod +x /usr/local/bin/s
cat > /usr/local/bin/t <<'TTD'
#!/bin/bash
echo "$(date -u +%H:%M)Z loop $(pgrep -fc 'online_loop.p[y]')  eval $(pgrep -fc 'pool_eval.p[y]')  $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)  $(df -h /root | tail -1 | awk '{print $4" free"}')"
# the run itself: the last steps, then the distribution that the pass rates hide
for f in $(ls -t /root/online_*.log 2>/dev/null | head -1); do
  grep -E "^\[step|^ONLINE_ROLLBACK|^ONLINE_COLLAPSE" "$f" | tail -5 | cut -c1-200
done
for d in $(ls -td /root/online_*/ 2>/dev/null | head -1); do
python3 - "$d" <<'PYT' 2>/dev/null
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1] + "/rollouts.jsonl") if l.strip()]
mx = max(r["step"] for r in rows)
for kind in ("reason", "search"):
    x = [r for r in rows if r.get("kind") == kind and r["step"] > mx - 20]
    if not x: continue
    ns = [r["ns"] for r in x]
    th = [len(r["text"].split("</think>")[0].split()) for r in x]
    print(f"  last 20 steps {kind}: pass {100*sum(r['reward'] for r in x)/len(x):.0f}% | searches med {statistics.median(ns):.0f} max {max(ns)} | think med {statistics.median(th):.0f} | unfinished {sum(1 for r in x if '</think>' not in r['text'])}/{len(x)}")
PYT
done
# a shard that is running but has written nothing is indistinguishable from a shard that is stuck,
# unless someone looks inside it
for f in $(ls -t /root/*_[0-9].log 2>/dev/null | head -3); do
  echo "  $(basename $f): $(grep -viE '^\s*$' $f | tail -1 | cut -c1-140)"
done
for f in /root/dwq_run_*.log /root/joint_run_*.log; do grep -hE "^\[dwq\]|^\[joint\]|taking step" "$f" 2>/dev/null | tail -3; done
python3 - <<'PYD' 2>/dev/null
import re,glob,os
for f in sorted(glob.glob("/root/dwq_d*.log")) + sorted(glob.glob("/root/joint_j*.log")) + sorted(glob.glob("/root/sft_s*.log")):
  tr=[];va=[]
  for l in open(f):
      m=re.match(r"step (\d+) kl=([\d.]+)",l)
      if m: tr.append((int(m.group(1)),float(m.group(2))))
      m=re.match(r"val (\d+) kl=([\d.]+)",l)
      if m: va.append((int(m.group(1)),float(m.group(2))))
  if not tr and not va: continue
  tag=os.path.basename(f)[4:-4]
  mean=lambda xs: sum(xs)/len(xs)
  s=f"  {tag}: "
  if tr: s+=f"step {tr[-1][0]} train kl {mean([r[1] for r in tr[:20]]):.4f} -> {mean([r[1] for r in tr[-20:]]):.4f}"
  if va: s+=f"   val {va[0][1]:.4f} -> {va[-1][1]:.4f} (best {min(v for _,v in va):.4f} @ {min(va,key=lambda x:x[1])[0]})"
  print(s)
PYD
python3 - <<'PYE' 2>/dev/null
import json,glob,collections,math,os
KEYS=("correct","grounded","landed","ns")
def load(pat):
  rows=[]
  for f in sorted(glob.glob(pat)):
      for line in open(f):
          try: rows.append(json.loads(line))
          except Exception: pass
  if not rows: return None
  per={}
  for k in KEYS:
      d=collections.defaultdict(list)
      for r in rows: d[r.get("q","")].append(float(r.get(k) or 0))
      per[k]={q:sum(v)/len(v) for q,v in d.items()}
  m={k:sum(float(r.get(k) or 0) for r in rows)/len(rows) for k in KEYS}
  m["zs"]=sum(1 for r in rows if not r.get("ns"))/len(rows)
  m["tt"]=sum(1 for r in rows if r.get("tail_tags") or ("</think>" in (r.get("text") or "") and "<search>" in (r.get("text") or "").split("</think>")[-1]))/len(rows)
  return len(rows), m, per
runs=[("bf16","/root/work/ev_out_*.jsonl"),("4bit plain","/root/work/q4_out_*.jsonl"),
    ("4bit +STE-lora","/root/work/qa_out_*.jsonl"),("4bit +dwq d1","/root/work/dw_out_*.jsonl")]
for p in sorted(glob.glob("/root/work/d[0-9]_out_0.jsonl")):
  t=os.path.basename(p).split("_")[0]
  runs.append((f"4bit +dwq {t}", f"/root/work/{t}_out_*.jsonl"))
for p in sorted(glob.glob("/root/work/g[0-9]*_out_0.jsonl")):
  t=os.path.basename(p).split("_")[0]
  runs.append((f"4bit {t} plain", f"/root/work/{t}_out_*.jsonl"))
for p in sorted(glob.glob("/root/work/j[0-9]_out_0.jsonl")) + sorted(glob.glob("/root/work/s[0-9]_out_0.jsonl")) + sorted(glob.glob("/root/work/p_*_out_0.jsonl")) + sorted(glob.glob("/root/work/g[0-9]_out_0.jsonl")):
  t=os.path.basename(p)[:-len("_out_0.jsonl")]
  runs.append((f"4bit {t}", f"/root/work/{t}_out_*.jsonl"))
L={n:load(p) for n,p in runs}
for n,_ in runs:
  S=L[n]
  if S: print(f"  {n:16s} {100*S[1]['correct']:5.1f}%  gnd {100*S[1]['grounded']:3.0f}%  "
              f"land {100*S[1]['landed']:3.0f}%  srch {S[1]['ns']:.1f}  zero-srch {100*S[1]['zs']:.0f}%  tail-tags {100*S[1]['tt']:.0f}%  ({S[0]} roll)")
F=L["bf16"]
for n,_ in runs[1:]:
  S=L[n]
  if not (F and S): continue
  c=sorted(set(F[2]["correct"])&set(S[2]["correct"]))
  if not c: continue
  out=[]
  for k,sc,u in (("correct",100,"pt"),("grounded",100,"pt"),("landed",100,"pt"),("ns",1,"")):
      d=[S[2][k][q]-F[2][k][q] for q in c]; m=sum(d)/len(d)
      sd=(sum((x-m)**2 for x in d)/len(d))**0.5
      out.append(f"{k[:4]} {sc*m:+.1f}{u}±{sc*sd/math.sqrt(len(d)):.1f}")
  print(f"  vs bf16 {n:16s} " + "  ".join(out) + f"  ({len(c)} q)")
# an arm whose tag ends in bf16 is a reference of its own: every arm sharing its prefix is paired
# against it too, so a probe run at another temperature is read in its own frame as well
for rn,_ in runs[1:]:
  if not rn.endswith("bf16") or not L[rn]: continue
  pre=rn[:-len("bf16")]
  for n,_ in runs[1:]:
    if n==rn or not n.startswith(pre) or not L[n]: continue
    R=L[rn]; S=L[n]
    c=sorted(set(R[2]["correct"])&set(S[2]["correct"]))
    if not c: continue
    out=[]
    for k,sc,u in (("correct",100,"pt"),("grounded",100,"pt"),("landed",100,"pt"),("ns",1,"")):
        d=[S[2][k][q]-R[2][k][q] for q in c]; m=sum(d)/len(d)
        sd=(sum((x-m)**2 for x in d)/len(d))**0.5
        out.append(f"{k[:4]} {sc*m:+.1f}{u}±{sc*sd/math.sqrt(len(d)):.1f}")
    print(f"  vs {rn[5:]:12s} {n:16s} " + "  ".join(out) + f"  ({len(c)} q)")
PYE
TTD
chmod +x /usr/local/bin/t
cat > /root/status.sh <<'STD'
#!/bin/bash
t 2>/dev/null
STD
chmod +x /root/status.sh
pkill -f "status_pu[b]"; sleep 1
cat > /root/status_pub.sh <<'SPD'
#!/bin/bash
while true; do
{ echo "=== $(date -u) === box E (dwq)"; t 2>/dev/null; } > /root/work/status.txt 2>&1
echo "--- STATUS $(date -u +%H:%M) ---"; cat /root/work/status.txt
sleep 300
done
SPD
chmod +x /root/status_pub.sh
setsid nohup bash /root/status_pub.sh >> /proc/1/fd/1 2>&1 < /dev/null &

# ---- hyde (2026-10-05 20:10 JST, the user: if the searching model wrote HyDE-style queries, the BART's page vector -
# trained to be found from a sentence of its article - may be enough). API only (nano), no GPU: the app model's 614
# search queries of the article-search test rewritten as one Wikipedia-style sentence each; box I measures them.
if [ ! -e /root/.hyde3 ] && [ -s /root/.oai ]; then touch /root/.hyde3; pkill -f "hyde_gen.p[y]"; rm -f /root/hyde/dcq_hyde.jsonl
  ( export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; mkdir -p /root/hyde; cd /root/hyde
    curl -sS -L -o hyde_gen.py "$RAW/sentbart/hyde_gen.py?$(date +%s)"
    hf download $R sentbart/searcheval/dcq.jsonl --local-dir /root/hyde/dl >/dev/null 2>&1
    python3 hyde_gen.py --queries /root/hyde/dl/sentbart/searcheval/dcq.jsonl --out /root/hyde/dcq_hyde.jsonl --n 100 2>&1 | grep -E "HYDE_DONE|api error|Error|Traceback" | cut -c1-300
    [ -s /root/hyde/dcq_hyde.jsonl ] && hf upload $R /root/hyde/dcq_hyde.jsonl sentbart/searcheval/dcq_hyde.jsonl >/dev/null 2>&1 && echo "HYDE_UPLOADED $(date -u)"
  ) > /root/hyde.log 2>&1 &
  echo "HYDE_LAUNCHED $(date -u)"
fi
# ---- 2026-10-06 08:45 JST: mtg5's rollouts so far on the hub (the reasoning blocks read 42% without </think> yet 71% passed).
if [ ! -e /root/.mtg5_partial1 ] && [ -s /root/online_mtg5/rollouts.jsonl ]; then touch /root/.mtg5_partial1
  HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token) hf upload baya1116/hypernet-sp-distill /root/online_mtg5/rollouts.jsonl pooler_distill/chatsft/multiturn/mtg5_rollouts_partial.jsonl >/dev/null 2>&1 && echo "MTG5_PARTIAL_UP $(date -u)"
fi
# ---- mtg5fix (2026-10-06 09:05 JST, the user: the reasoning reward credited unfinished thinking - likely the push toward
# longer thinking - fix the scoring and see whether it comes back). The run is resumed under the fixed online_loop.py
# right after its step-80 save (same command, so it continues from latest.safetensors and its optimizer state).
if [ ! -e /root/.mtg5fix ] && grep -q "if \"</think>\" in text else \"\"   # no </think>: unfinished" /root/work/online_loop.py 2>/dev/null; then touch /root/.mtg5fix
  cat > /root/mtg5fixkeep.sh <<'FX'
until [ -s /root/mtg5_s80.safetensors ]; do sleep 30; done; sleep 30
pkill -f "mtg5kee[p].sh"; pkill -f "online_loop.py /root/mtg3_s100.safetensors /root/online_mtg5"; sleep 15
pkill -9 -f "online_loop.py /root/mtg3_s100.safetensors /root/online_mtg5" 2>/dev/null; sleep 5
echo "[mtg5] restarted at step $(python3 -c "import json;print(json.load(open('/root/online_mtg5/state.json'))['step'])") under the fixed reasoning reward $(date -u +%H:%M)" >> /root/mtg5.log
setsid nohup bash -c 'bash /root/mtg5keep.sh 2>&1 | tee -a /root/mtg5.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
FX
  setsid nohup bash /root/mtg5fixkeep.sh > /root/mtg5fix.log 2>&1 < /dev/null &
  echo "MTG5FIX_ARMED $(date -u)"
fi
# ---- mtg5b (2026-10-06 12:25 JST, as agreed with the user: if reasoning does not come back under the fixed reward, start
# again from step 40). After the fix (step 80) reasoning read 68% / 8% unfinished for 81-100, then 42% / 42% for 101-120
# and 41% / 41% for 121-140, thinking up to a median 1177 words: it did not hold. mtg5 stops at ~137; mtg5b trains from
# mtg5_s40 (before the drift) under the fixed reward, 120 steps on mtg5's items minus the ones steps 1-40 already used,
# a copy every 40; then shard-0 screens of mtg5_s40 and mtg5b's copies against s100's 54, the best above it fully.
if [ ! -e /root/.mtg5b ] && [ -s /root/mtg5_s40.safetensors ] && grep -q 'no </think>: unfinished' /root/work/online_loop.py 2>/dev/null; then touch /root/.mtg5b
  pkill -f "mtg5kee[p].sh"; pkill -f "mtg5fixkee[p].sh"; pkill -f "online_loop.py /root/mtg3_s100.safetensors /root/online_mtg5"; sleep 15
  pkill -9 -f "online_loop.py /root/mtg3_s100.safetensors /root/online_mtg5" 2>/dev/null
  echo "MTG5_JOB_DONE stopped at step $(python3 -c "import json;print(json.load(open('/root/online_mtg5/state.json'))['step'])") for mtg5b $(date -u)" >> /root/mtg5.log
  cp /root/online_mtg5/rollouts.jsonl /root/work/mtg5_rollouts.jsonl; HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token) hf upload baya1116/hypernet-sp-distill /root/work/mtg5_rollouts.jsonl pooler_distill/chatsft/multiturn/mtg5_rollouts.jsonl >/dev/null 2>&1
  rm -f /root/online_mtg5/*.pt /root/mtg5_s80.safetensors /root/mtg5_s120.safetensors
  cat > /root/mtg5bkeep.sh <<'M5'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; OUT=/root/online_mtg5b
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
python3 - <<'PYI'
import json
seen = {json.loads(l)["q"].strip() for l in open("/root/work/mtg5_rollouts.jsonl") if json.loads(l).get("kind") == "search" and json.loads(l)["step"] <= 40}
it = [json.loads(l) for l in open("/root/work/mtg5_items.jsonl")]
out = [x for x in it if x["q"].strip() not in seen]
open("/root/work/mtg5b_items.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
print(f"[mtg5b] {len(out)} items ({len(it) - len(out)} used by steps 1-40 left out)", flush=True)
PYI
while pgrep -f "pool_eval.p[y]|online_loop.p[y]|memfit.p[y]" >/dev/null; do sleep 30; done
rm -rf /root/evalrun_*
echo "[mtg5b] start from mtg5_s40 $(date -u +%H:%M); $(df -h /root | tail -1 | awk '{print $4}') free"
( last=0; while sleep 60; do s=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null || echo 0)
    if [ "$s" != "$last" ] && [ $((s % 40)) -eq 0 ] && [ "$s" -gt 0 ] && [ ! -s /root/mtg5b_s$s.safetensors ]; then sleep 20; cp $OUT/latest.safetensors /root/mtg5b_s$s.safetensors; echo "[mtg5b] copy at step $s"; fi; last=$s; done ) & CP=$!
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OAI_KEY=$(cat /root/.oai 2>/dev/null) DSK_KEY=$(cat /root/.dsk 2>/dev/null) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /root/work/online_loop.py /root/mtg5_s40.safetensors $OUT --pooler none --lora-layers all --lora-rank 16 --save-lora-only 1 \
  --mt-items /root/work/mtg5b_items.jsonl --heldout /root/work/eval300.jsonl --reason /root/work/dolphin_rft.jsonl --reason-every 4 --reason-g 8 \
  --search-demo /root/work/r1_traj_short.jsonl --demo-on-fail 0.5 \
  --steps 120 --save-every 40 --lr 5e-6 --search-lr 5e-6 --search-temp 0.9 --search-gen 2000 --temp 0.6 --gen 3000 --budget 900 --maxsrch 7 --stop eos \
  --judge-api openai --judge-model gpt-5-nano --w-talk 0.5 --dolphin-min 0 --adv-std 1 --pg-norm mean --kl 0 \
  --guard 1 --guard-steps 20 > /root/mtg5b_run.log 2>&1
sleep 90; kill $CP 2>/dev/null
grep -E "^\[data\]|^\[init\]|ONLINE_|Error|Traceback" /root/mtg5b_run.log | tail -4 | cut -c1-250
S=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null); echo "[mtg5b] stopped at step ${S:-0} $(date -u +%H:%M)"
[ -s $OUT/latest.safetensors ] && [ ! -s /root/mtg5b_s$S.safetensors ] && cp $OUT/latest.safetensors /root/mtg5b_s$S.safetensors
hf upload $R $OUT/rollouts.jsonl pooler_distill/chatsft/multiturn/mtg5b_rollouts.jsonl >/dev/null 2>&1
rm -f $OUT/*.pt $OUT/good.safetensors
BEST=; BC=54
for CK in /root/mtg5_s40.safetensors $(ls /root/mtg5b_s*.safetensors 2>/dev/null | sort -V); do
  T=$(basename $CK .safetensors); hf upload $R $CK pooler_distill/chatsft/multiturn/$T.safetensors >/dev/null 2>&1
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/ev_0.jsonl /root/work/${T}st_0.jsonl --n 100 $EVARGS --tag "[${T}st0]" > /root/${T}st_0.log 2>&1
  c=$(python3 -c "import json;print(sum(bool(json.loads(l).get('correct')) for l in open('/root/work/${T}st_0.jsonl')))" 2>/dev/null || echo 0)
  echo "[mtg5b] $T single shard 0: $c/100 (s100 54)"; [ "$c" -gt "$BC" ] && { BC=$c; BEST=$CK; }
done
[ -n "$BEST" ] || { echo "MTG5B_JOB_DONE no copy above s100's shard 0 (54)"; exit 0; }
T=$(basename $BEST .safetensors); echo "[mtg5b] full screens for $T"
SN=$BC; for i in 1 2; do env $ENV python3 /root/work/pool_eval.py $BEST /root/work/ev_$i.jsonl /root/work/${T}st_$i.jsonl --n 100 $EVARGS --tag "[${T}st$i]" > /root/${T}st_$i.log 2>&1
  c=$(python3 -c "import json;print(sum(bool(json.loads(l).get('correct')) for l in open('/root/work/${T}st_$i.jsonl')))" 2>/dev/null || echo 0); SN=$((SN + c)); echo "[mtg5b] single shard $i: $c/100"; done
echo "[mtg5b] $T single-turn: $SN/300 (s100 160)"
for b in "br3:/root/work/mt_eval_bridge3.jsonl:(s100 43/31)" "br:/root/work/mt_eval_bridge.jsonl:(s100 23/19)"; do X=${b%%:*}; r=${b#*:}; F=${r%%:*}; NOTE=${r#*:}
  env $ENV python3 /root/work/pool_eval.py $BEST /root/work/eval300.jsonl /root/work/${X}_$T.jsonl --multiturn $F --mt-mode win --n 1000 $EVARGS --tag "[$X-$T]" > /root/${X}_$T.log 2>&1
  echo "[mtg5b] $X (win): $(python3 -c "
import json; r=[json.loads(l) for l in open('/root/work/${X}_$T.jsonl')]
print('turn 1', sum(x['correct'] for x in r if x['turn']==0), '/', sum(1 for x in r if x['turn']==0), ', follow-up', sum(x['correct'] for x in r if x['turn']==1), '/', sum(1 for x in r if x['turn']==1))" 2>&1 | tail -1) $NOTE"
done
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $BEST /root/evalrun_dl_mtg5b \
  --questions /root/work/eval300.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl \
  --b 12 --gen 7000 --budget 2400 --temp 0.6 --maxsrch 7 --pooler none --lora-rank 16 --lora-layers all --stop eos --loop-break answer \
  --eval-file /root/work/dolphinq.jsonl --eval-out /root/work/dl_$T.jsonl > /root/dl_$T.log 2>&1
[ -s /root/dl_judge.py ] && OAI_KEY=$(cat /root/.oai 2>/dev/null) python3 /root/dl_judge.py /root/work/dl_$T.jsonl $T | sed "s/\[mix0\]/[mtg5b]/"
echo "[mtg5b] (s100: Dolphin 56)"
echo "MTG5B_JOB_DONE $(date -u)"
M5
  setsid nohup bash -c 'bash /root/mtg5bkeep.sh 2>&1 | tee -a /root/mtg5b.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MTG5B_LAUNCHED $(date -u)"
fi
# ---- mtg5b at the evaluation's length (2026-10-06 12:55 JST, the user): the reasoning samples were cut at 3000 tokens
# while the Dolphin evaluation allows 7000 - the heavy problems were punished for thinking they would have finished.
# mtg5b (a few steps in, no save yet) restarts from mtg5_s40 with --gen 7000 --budget 2400 (the evaluation's settings).
if [ ! -e /root/.mtg5b_long ] && [ -s /root/mtg5bkeep.sh ]; then touch /root/.mtg5b_long
  pkill -f "mtg5bkee[p].sh"; pkill -f "online_loop.py /root/mtg5_s40.safetensors /root/online_mtg5b"; sleep 15
  pkill -9 -f "online_loop.py /root/mtg5_s40.safetensors /root/online_mtg5b" 2>/dev/null; sleep 3
  sed -i 's/--temp 0.6 --gen 3000 --budget 900/--temp 0.6 --gen 7000 --budget 2400/' /root/mtg5bkeep.sh
  grep -q -- "--gen 7000 --budget 2400" /root/mtg5bkeep.sh && echo "[mtg5b] restart with the reasoning cap at 7000 tokens (was 3000) $(date -u +%H:%M)" >> /root/mtg5b.log
  rm -rf /root/online_mtg5b /root/mtg5b_s*.safetensors /root/mtg5b_run.log
  setsid nohup bash -c 'bash /root/mtg5bkeep.sh 2>&1 | tee -a /root/mtg5b.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MTG5B_LONG_LAUNCHED $(date -u)"
fi
# ---- mtg5b back at 3000 (2026-10-06 13:00 JST, the user: if the longer cap does not change anything it is wasted - back).
if [ ! -e /root/.mtg5b_short ] && [ -s /root/mtg5bkeep.sh ]; then touch /root/.mtg5b_short
  pkill -f "mtg5bkee[p].sh"; pkill -f "online_loop.py /root/mtg5_s40.safetensors /root/online_mtg5b"; sleep 15
  pkill -9 -f "online_loop.py /root/mtg5_s40.safetensors /root/online_mtg5b" 2>/dev/null; sleep 3
  sed -i 's/--temp 0.6 --gen 7000 --budget 2400/--temp 0.6 --gen 3000 --budget 900/' /root/mtg5bkeep.sh
  grep -q -- "--gen 3000 --budget 900" /root/mtg5bkeep.sh && echo "[mtg5b] restart with the reasoning cap back at 3000 tokens $(date -u +%H:%M)" >> /root/mtg5b.log
  rm -rf /root/online_mtg5b /root/mtg5b_s*.safetensors /root/mtg5b_run.log
  setsid nohup bash -c 'bash /root/mtg5bkeep.sh 2>&1 | tee -a /root/mtg5b.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MTG5B_SHORT_LAUNCHED $(date -u)"
fi
# ---- mtg5b at 7000 again, as a trial (2026-10-06 13:10 JST, the user: try 7000; if unfinished improves but accuracy does
# not, or the thinking just grows into the new cap, go back to 3000). Restart from mtg5_s40 at --gen 7000 --budget 2400;
# its rollouts go to the hub at step 40 for the comparison with mtg5 (3000) on the same reasoning problems (same steps).
if [ ! -e /root/.mtg5b_trial ] && [ -s /root/mtg5bkeep.sh ]; then touch /root/.mtg5b_trial
  pkill -f "mtg5bkee[p].sh"; pkill -f "online_loop.py /root/mtg5_s40.safetensors /root/online_mtg5b"; sleep 15
  pkill -9 -f "online_loop.py /root/mtg5_s40.safetensors /root/online_mtg5b" 2>/dev/null; sleep 3
  sed -i 's/--temp 0.6 --gen 3000 --budget 900/--temp 0.6 --gen 7000 --budget 2400/' /root/mtg5bkeep.sh
  grep -q -- "--gen 7000 --budget 2400" /root/mtg5bkeep.sh && echo "[mtg5b] trial with the reasoning cap at 7000 tokens $(date -u +%H:%M)" >> /root/mtg5b.log
  rm -rf /root/online_mtg5b /root/mtg5b_s*.safetensors /root/mtg5b_run.log
  setsid nohup bash -c 'bash /root/mtg5bkeep.sh 2>&1 | tee -a /root/mtg5b.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  ( until [ -s /root/mtg5b_s40.safetensors ]; do sleep 60; done; sleep 30
    HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token) hf upload baya1116/hypernet-sp-distill /root/online_mtg5b/rollouts.jsonl pooler_distill/chatsft/multiturn/mtg5b_rollouts_s40.jsonl >/dev/null 2>&1 \
      && echo "[mtg5b] rollouts to step 40 on the hub $(date -u +%H:%M)" >> /root/mtg5b.log ) > /dev/null 2>&1 &
  echo "MTG5B_TRIAL_LAUNCHED $(date -u)"
fi
# ---- qgen (2026-10-06 23:30 JST, the user: the HyDE-style and search-query losses go into the sentence BART's e2e). The
# query writer, API only (nano, ~22,000 documents, 8 at a time): box I's docs_for_qgen.jsonl from the hub, a search query
# and a HyDE sentence per document, back to the hub as sentbart/e2e2/queries.jsonl.
if [ ! -e /root/.qgen1 ] && [ -s /root/.oai ]; then touch /root/.qgen1
  ( export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; mkdir -p /root/qgen; cd /root/qgen
    curl -sS -L -o qgen.py "$RAW/sentbart/qgen.py?$(date +%s)"
    until hf download $R sentbart/e2e2/docs_for_qgen.jsonl --local-dir /root/qgen/dl >/dev/null 2>&1 && [ -s /root/qgen/dl/sentbart/e2e2/docs_for_qgen.jsonl ]; do sleep 120; done
    echo "[qgen] $(wc -l < /root/qgen/dl/sentbart/e2e2/docs_for_qgen.jsonl) documents $(date -u +%H:%M)"
    ( while sleep 900; do [ -s /root/qgen/queries.jsonl ] && hf upload $R /root/qgen/queries.jsonl sentbart/e2e2/queries_partial.jsonl >/dev/null 2>&1; done ) & UP=$!
    python3 -u qgen.py --docs /root/qgen/dl/sentbart/e2e2/docs_for_qgen.jsonl --out /root/qgen/queries.jsonl --workers 8 2>&1 | grep --line-buffered -E "^\[qgen\]|QGEN_" | cut -c1-300
    kill $UP 2>/dev/null
    [ -s /root/qgen/queries.jsonl ] && hf upload $R /root/qgen/queries.jsonl sentbart/e2e2/queries.jsonl >/dev/null 2>&1 && echo "QGEN_UPLOADED $(wc -l < /root/qgen/queries.jsonl) rows $(date -u)"
  ) > /root/qgen.log 2>&1 &
  echo "QGEN_LAUNCHED $(date -u)"
fi
# ---- 2026-10-07 01:45 JST (the user: follow-up questions on the fly): mtg6's waiting keeper is replaced by one that passes
# --followup 0.5 (online_loop writes the user's follow-up to a passing exchange with the teacher, checks its answer on
# the named page, and asks it next with the exchange as history).
if [ ! -e /root/.mtg6_v2 ]; then touch /root/.mtg6_v2; pkill -f "mtg6kee[p].sh"; sleep 2; fi
# 2026-10-07 02:50 JST (the user: a never-solved reasoning group takes Dolphin's own CoT, a never-solved search or follow-up
# group gets R1 on the fly - what it can solve it solves better, what it cannot it is shown): --cot-on-fail 0.5 --r1-on-fail 1.
if [ ! -e /root/.mtg6_v3 ]; then touch /root/.mtg6_v3; pkill -f "mtg6kee[p].sh"; sleep 2; fi
if [ ! -e /root/.mtg6_v4 ]; then touch /root/.mtg6_v4; pkill -f "mtg6kee[p].sh"; sleep 2; fi   # the teacher gate
# ---- mtg6 without the probe (2026-10-07 07:20 JST, the user: why not learn as you go?). The probe (600 x 4 rollouts, ~10 h,
# no learning) is dropped: GRPO's own eight samples grade each fresh question as it comes - 1-7 pass: the gradient; 8 pass:
# nothing much (the advantage is ~0); 0 pass: R1 on the fly, then the demonstration (--r1-on-fail). Training starts now
# from s100 on the 600 fresh questions in order (each at most once), follow-ups on the fly, Dolphin's CoT on an all-miss
# reasoning group. The questions used go to the ledger afterwards. Copies every 40, screens as before.
# 2026-10-07 07:45 JST (the user: the search GRPO used 12 samples before; the multi-turn runs had 8): search groups of 12
# again (--search-g 12), reasoning stays at 8 under the 7000 cap. The run is restarted from s100 (a few steps lost).
if [ ! -e /root/.mtg6_g12 ] && [ -e /root/.mtg6_direct ]; then touch /root/.mtg6_g12; rm -f /root/.mtg6_direct
  pkill -f "mtg6kee[p].sh"; pkill -f "online_loop.py /root/mtg3_s100.safetensors /root/online_mtg6"; sleep 10; pkill -9 -f "online_loop.py /root/mtg3_s100.safetensors /root/online_mtg6" 2>/dev/null
  echo "[mtg6] restarted with 12 samples a search question $(date -u +%H:%M)" >> /root/mtg6.log
fi
# 2026-10-07 08:10 JST (the user: the memory is bounded, so 12 must fit - and if it is not bounded, that is the problem).
# It was not, with history: the training rollouts pinned the whole conversation in the prompt, where pool_eval's win
# mode (the app's scheme) pins only the current question and runs the rest as one stream through the raw window and
# the pooler. rollout_batch now takes the stream as a seed; a follow-up continues the stream of the exchange it was
# written from. mtg6 restarted once more so every follow-up trains the way the app runs.
if [ ! -e /root/.mtg6_win ] && [ -e /root/.mtg6_direct ] && grep -q "def seed_from_roll" /root/work/online_loop.py; then touch /root/.mtg6_win; rm -f /root/.mtg6_direct
  pkill -f "mtg6kee[p].sh"; pkill -f "online_loop.py /root/mtg3_s100.safetensors /root/online_mtg6"; sleep 10; pkill -9 -f "online_loop.py /root/mtg3_s100.safetensors /root/online_mtg6" 2>/dev/null
  echo "[mtg6] restarted with the history as a stream (win), 12 samples a search question $(date -u +%H:%M)" >> /root/mtg6.log
fi
if [ ! -e /root/.mtg6_direct ] && [ -s /root/work/nq_items_1.jsonl ]; then touch /root/.mtg6_direct
  pkill -f "mtg6kee[p].sh"; pkill -f "online_loop.py /root/mtg3_s100.safetensors /root/probe_nq1"; sleep 10
  pkill -9 -f "online_loop.py /root/mtg3_s100.safetensors /root/probe_nq1" 2>/dev/null
  echo "[mtg6] probe dropped at $(tail -n 1 /root/probe_nq1.log 2>/dev/null | cut -c1-40) - training directly $(date -u +%H:%M)" >> /root/mtg6.log
  cat > /root/mtg6keep.sh <<'M6'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; OUT=/root/online_mtg6
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
S0=/root/mtg3_s100.safetensors; B0=54
while pgrep -f "pool_eval.p[y]|online_loop.p[y]" >/dev/null; do sleep 30; done
rm -rf /root/evalrun_* /root/probe_nq1 /root/online_mtg6
echo "[mtg6] start from s100 on 600 fresh questions, no probe $(date -u +%H:%M); $(df -h /root | tail -1 | awk '{print $4}') free"
( last=0; while sleep 60; do s=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null || echo 0)
    if [ "$s" != "$last" ] && [ $((s % 40)) -eq 0 ] && [ "$s" -gt 0 ] && [ ! -s /root/mtg6_s$s.safetensors ]; then sleep 20; cp $OUT/latest.safetensors /root/mtg6_s$s.safetensors; echo "[mtg6] copy at step $s"; fi; last=$s; done ) & CP=$!
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OAI_KEY=$(cat /root/.oai 2>/dev/null) DSK_KEY=$(cat /root/.dsk 2>/dev/null) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /root/work/online_loop.py $S0 $OUT --pooler none --lora-layers all --lora-rank 16 --save-lora-only 1 \
  --mt-items /root/work/nq_items_1.jsonl --heldout /root/work/eval300.jsonl --reason /root/work/dolphin_rft.jsonl --reason-every 4 --reason-g 8 --search-g 12 \
  --demo-on-fail 0.5 --followup 0.5 --r1-on-fail 1 --cot-on-fail 0.5 \
  --steps 120 --save-every 40 --lr 5e-6 --search-lr 5e-6 --search-temp 0.9 --search-gen 2000 --temp 0.6 --gen 7000 --budget 2400 --maxsrch 7 --stop eos \
  --judge-api openai --judge-model gpt-5-nano --w-talk 0.5 --dolphin-min 0 --adv-std 1 --pg-norm mean --kl 0 \
  --guard 1 --guard-steps 20 > /root/mtg6_run.log 2>&1
sleep 90; kill $CP 2>/dev/null
grep -E "^\[data\]|^\[init\]|ONLINE_|Error|Traceback" /root/mtg6_run.log | tail -4 | cut -c1-250
S=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null); echo "[mtg6] stopped at step ${S:-0} $(date -u +%H:%M)"
[ -s $OUT/latest.safetensors ] && [ ! -s /root/mtg6_s$S.safetensors ] && cp $OUT/latest.safetensors /root/mtg6_s$S.safetensors
hf upload $R $OUT/rollouts.jsonl pooler_distill/chatsft/multiturn/mtg6_rollouts.jsonl >/dev/null 2>&1
for f in followups.jsonl r1_onfly.jsonl; do [ -s $OUT/$f ] && hf upload $R $OUT/$f pooler_distill/chatsft/multiturn/mtg6_$f >/dev/null 2>&1; done
python3 - <<'PYL'
import json
qs = {json.loads(l)["q"] for l in open("/root/online_mtg6/rollouts.jsonl") if json.loads(l).get("kind") == "search"}
with open("/root/work/trained_items.jsonl", "a") as f:
    for q in qs: f.write(json.dumps({"q": q, "run": "mtg6"}, ensure_ascii=False) + "\n")
print(f"[mtg6] {len(qs)} search questions used (follow-ups included) -> the ledger", flush=True)
PYL
rm -f "${OUT:?}"/*.pt "${OUT:?}"/good.safetensors
BEST=; BC=$B0
for CK in $(ls /root/mtg6_s*.safetensors 2>/dev/null | sort -V); do
  T=$(basename $CK .safetensors); hf upload $R $CK pooler_distill/chatsft/multiturn/$T.safetensors >/dev/null 2>&1
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/ev_0.jsonl /root/work/${T}st_0.jsonl --n 100 $EVARGS --tag "[${T}st0]" > /root/${T}st_0.log 2>&1
  c=$(python3 -c "import json;print(sum(bool(json.loads(l).get('correct')) for l in open('/root/work/${T}st_0.jsonl')))" 2>/dev/null || echo 0)
  echo "[mtg6] $T single shard 0: $c/100 (s100 54)"; [ "$c" -gt "$BC" ] && { BC=$c; BEST=$CK; }
done
[ -n "$BEST" ] || { echo "MTG6_JOB_DONE no copy above s100's shard 0 (54)"; exit 0; }
T=$(basename $BEST .safetensors); echo "[mtg6] full screens for $T"
SN=$BC; for i in 1 2; do env $ENV python3 /root/work/pool_eval.py $BEST /root/work/ev_$i.jsonl /root/work/${T}st_$i.jsonl --n 100 $EVARGS --tag "[${T}st$i]" > /root/${T}st_$i.log 2>&1
  c=$(python3 -c "import json;print(sum(bool(json.loads(l).get('correct')) for l in open('/root/work/${T}st_$i.jsonl')))" 2>/dev/null || echo 0); SN=$((SN + c)); echo "[mtg6] single shard $i: $c/100"; done
echo "[mtg6] $T single-turn: $SN/300 (s100 160)"
for b in "br3:/root/work/mt_eval_bridge3.jsonl:(s100 43/31)" "br:/root/work/mt_eval_bridge.jsonl:(s100 23/19)"; do X=${b%%:*}; r=${b#*:}; F=${r%%:*}; NOTE=${r#*:}
  env $ENV python3 /root/work/pool_eval.py $BEST /root/work/eval300.jsonl /root/work/${X}_$T.jsonl --multiturn $F --mt-mode win --n 1000 $EVARGS --tag "[$X-$T]" > /root/${X}_$T.log 2>&1
  echo "[mtg6] $X (win): $(python3 -c "
import json; r=[json.loads(l) for l in open('/root/work/${X}_$T.jsonl')]
print('turn 1', sum(x['correct'] for x in r if x['turn']==0), '/', sum(1 for x in r if x['turn']==0), ', follow-up', sum(x['correct'] for x in r if x['turn']==1), '/', sum(1 for x in r if x['turn']==1))" 2>&1 | tail -1) $NOTE"
done
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $BEST /root/evalrun_dl_mtg6 \
  --questions /root/work/eval300.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl \
  --b 12 --gen 7000 --budget 2400 --temp 0.6 --maxsrch 7 --pooler none --lora-rank 16 --lora-layers all --stop eos --loop-break answer \
  --eval-file /root/work/dolphinq.jsonl --eval-out /root/work/dl_$T.jsonl > /root/dl_$T.log 2>&1
[ -s /root/dl_judge.py ] && OAI_KEY=$(cat /root/.oai 2>/dev/null) python3 /root/dl_judge.py /root/work/dl_$T.jsonl $T | sed "s/\[mix0\]/[mtg6]/"
echo "[mtg6] (s100: Dolphin 56)"
echo "MTG6_JOB_DONE $(date -u)"
M6
  setsid nohup bash -c 'bash /root/mtg6keep.sh 2>&1 | tee -a /root/mtg6.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MTG6_DIRECT_LAUNCHED $(date -u)"
fi
# ---- mtg6 (2026-10-07 01:15 JST, the user: from now on train on questions never seen, barely learned, or unsolved; the
# home-made pool is nearly used up and nq_open has ~88k). After mtg5b's screens: the best of {mtg5_s40, mtg5b_s40/80/120,
# s100} on shard 0 is the start; 600 fresh Natural Questions (none in any evaluation or earlier training file) probed
# 4x by it; items = passed 1-3 of 4 (shallow) + never-solved ones with an R1 trajectory (<= 3 searches), at most a
# quarter; a ledger of every question used. Then 120 steps as mtg5b (fixed reward, 7000 cap), copies every 40, screens.
if ! pgrep -f "mtg6kee[p].sh" >/dev/null && ! grep -q "MTG6_JOB_DONE" /root/mtg6.log 2>/dev/null && [ -s /root/work/nq_items.py ]; then
  cat > /root/mtg6keep.sh <<'M6'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/work; OUT=/root/online_mtg6
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
OL="--pooler none --lora-layers all --lora-rank 16 --save-lora-only 1 --heldout /root/work/eval300.jsonl --maxsrch 7 --stop eos --judge 0 --reason-stub 1"
until grep -q "MTG5B_JOB_DONE" /root/mtg5b.log 2>/dev/null; do sleep 120; done
while pgrep -f "pool_eval.p[y]|online_loop.p[y]" >/dev/null; do sleep 30; done
# the teacher must be able to answer (03:15 JST the OpenAI account ran out of credit: every judge call would fail and
# the run would train on nothing): one tiny call every ten minutes until it does
oai_ok() { python3 - <<'PYO'
import json, urllib.request, urllib.error, sys
key = open("/root/.oai").read().strip()
body = {"model": "gpt-5-nano", "messages": [{"role": "user", "content": "Reply with the word ok."}], "max_completion_tokens": 200}
try:
    d = json.load(urllib.request.urlopen(urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}), timeout=60)); sys.exit(0 if "choices" in d else 1)
except urllib.error.HTTPError as e:
    print("[mtg6] teacher: http", e.code, e.read().decode()[:120].replace("\n", " "), flush=True); sys.exit(1)
except Exception as e:
    print("[mtg6] teacher:", type(e).__name__, flush=True); sys.exit(1)
PYO
}
until oai_ok; do sleep 600; done
echo "[mtg6] teacher reachable $(date -u +%H:%M)"
# the start: the best shard-0 screen of mtg5b's job, s100 (54) if none beat it
S0=/root/mtg3_s100.safetensors; B0=54
while read -r c ck; do [ "$c" -gt "$B0" ] && [ -s "$ck" ] && { B0=$c; S0=$ck; }; done < <(grep -oE "\[mtg5b\] mtg5b?_s[0-9]+ single shard 0: [0-9]+" /root/mtg5b.log | sed -E 's/\[mtg5b\] (mtg5b?_s[0-9]+) single shard 0: ([0-9]+)/\2 \/root\/\1.safetensors/')
echo "[mtg6] start model $S0 (shard 0: $B0) $(date -u +%H:%M); $(df -h /root | tail -1 | awk '{print $4}') free"
# fresh questions, none in any evaluation or earlier training file; the ledger holds every question any run has trained on
python3 /root/work/nq_items.py --n 600 --out /root/work/nq_items_1.jsonl --pool /root/work/nq_pool.jsonl \
  --exclude /root/work/eval300.jsonl /root/work/mt_eval.jsonl /root/work/mt_eval_bridge.jsonl /root/work/mt_eval_bridge2.jsonl /root/work/mt_eval_bridge3.jsonl /root/work/mt_eval_chain.jsonl \
            /root/work/dolphin_heldout100.jsonl /root/work/dolphin_v1.jsonl /root/work/selfq_all.jsonl /root/work/mtg2_items.jsonl /root/work/mtg_items.jsonl /root/work/mtg4_items.jsonl /root/work/mtg5_items.jsonl \
            /root/work/replay_v1.jsonl /root/work/trained_items.jsonl 2>&1 | grep -E "NQ_ITEMS_DONE|Error|Traceback" | cut -c1-200
[ -s /root/work/nq_items_1.jsonl ] || { echo "MTG6_JOB_DONE no items"; exit 1; }
rm -rf /root/evalrun_* /root/probe_nq1
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $S0 /root/probe_nq1 $OL \
  --mt-items /root/work/nq_items_1.jsonl --probe-out /root/work/probe_nq1.jsonl --probe-g 4 --b 12 --temp 0.9 --gen 2000 --budget 900 > /root/probe_nq1.log 2>&1
grep -E "PROBE_DONE|Error|Traceback" /root/probe_nq1.log | tail -1 | cut -c1-200
python3 - <<'PYS'
import json, collections
r = [json.loads(l) for l in open("/root/work/probe_nq1.jsonl")]
c = collections.Counter(x["pass"] for x in r)
print(f"[mtg6] probe: {len(r)} fresh questions, passes of 4: " + " ".join(f"{k}:{c[k]}" for k in range(5)), flush=True)
PYS
hf upload $R /root/work/probe_nq1.jsonl pooler_distill/chatsft/teach/probe_nq1.jsonl >/dev/null 2>&1
# R1 on the never-solved ones (verified trajectories, <= 3 searches)
python3 /root/work/r1_traj.py --probe /root/work/probe_nq1.jsonl --out /root/work/r1_nq1.jsonl --max-pass 0 --workers 6 2>&1 | grep -E "^\[r1\]|R1_TRAJ_DONE|Error|Traceback" | tail -3
python3 - <<'PYI'
import json, random, os
demo = {}
if os.path.exists("/root/work/r1_nq1.jsonl"):
    for l in open("/root/work/r1_nq1.jsonl"):
        d = json.loads(l)
        if d.get("ns", 9) <= 3: demo[d["q"].strip()] = d
open("/root/work/r1_nq1_short.jsonl", "w").write("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in demo.values()))
pr = [json.loads(l) for l in open("/root/work/probe_nq1.jsonl")]
mixed = [{"q": x["q"], "gold": x["gold"], "hist": []} for x in pr if 1 <= x["pass"] <= 3]
hard = [{"q": x["q"], "gold": x["gold"], "hist": []} for x in pr if x["pass"] == 0 and x["q"].strip() in demo]
random.Random(6).shuffle(hard); hard = hard[:len(mixed) // 3]
items = mixed + hard; random.Random(7).shuffle(items)
open("/root/work/mtg6_items.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))
with open("/root/work/trained_items.jsonl", "a") as f:
    for x in items: f.write(json.dumps({"q": x["q"], "run": "mtg6"}, ensure_ascii=False) + "\n")
print(f"[mtg6] {len(items)} items: {len(mixed)} solved 1-3 of 4, {len(hard)} never solved with an R1 trajectory ({len(demo)} trajectories)", flush=True)
PYI
hf upload $R /root/work/r1_nq1.jsonl pooler_distill/chatsft/teach/r1_nq1.jsonl >/dev/null 2>&1
[ "$(wc -l < /root/work/mtg6_items.jsonl)" -ge 40 ] || { echo "MTG6_JOB_DONE too few items"; exit 1; }
rm -rf /root/probe_nq1
echo "[mtg6] start $(date -u +%H:%M); $(df -h /root | tail -1 | awk '{print $4}') free"
( last=0; while sleep 60; do s=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null || echo 0)
    if [ "$s" != "$last" ] && [ $((s % 40)) -eq 0 ] && [ "$s" -gt 0 ] && [ ! -s /root/mtg6_s$s.safetensors ]; then sleep 20; cp $OUT/latest.safetensors /root/mtg6_s$s.safetensors; echo "[mtg6] copy at step $s"; fi; last=$s; done ) & CP=$!
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OAI_KEY=$(cat /root/.oai 2>/dev/null) DSK_KEY=$(cat /root/.dsk 2>/dev/null) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /root/work/online_loop.py $S0 $OUT --pooler none --lora-layers all --lora-rank 16 --save-lora-only 1 \
  --mt-items /root/work/mtg6_items.jsonl --heldout /root/work/eval300.jsonl --reason /root/work/dolphin_rft.jsonl --reason-every 4 --reason-g 8 \
  --search-demo /root/work/r1_nq1_short.jsonl --demo-on-fail 0.5 --followup 0.5 --r1-on-fail 1 --cot-on-fail 0.5 \
  --steps 120 --save-every 40 --lr 5e-6 --search-lr 5e-6 --search-temp 0.9 --search-gen 2000 --temp 0.6 --gen 7000 --budget 2400 --maxsrch 7 --stop eos \
  --judge-api openai --judge-model gpt-5-nano --w-talk 0.5 --dolphin-min 0 --adv-std 1 --pg-norm mean --kl 0 \
  --guard 1 --guard-steps 20 > /root/mtg6_run.log 2>&1
sleep 90; kill $CP 2>/dev/null
grep -E "^\[data\]|^\[init\]|ONLINE_|Error|Traceback" /root/mtg6_run.log | tail -4 | cut -c1-250
S=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null); echo "[mtg6] stopped at step ${S:-0} $(date -u +%H:%M)"
[ -s $OUT/latest.safetensors ] && [ ! -s /root/mtg6_s$S.safetensors ] && cp $OUT/latest.safetensors /root/mtg6_s$S.safetensors
hf upload $R $OUT/rollouts.jsonl pooler_distill/chatsft/multiturn/mtg6_rollouts.jsonl >/dev/null 2>&1
rm -f "${OUT:?}"/*.pt "${OUT:?}"/good.safetensors
BEST=; BC=$B0
for CK in $(ls /root/mtg6_s*.safetensors 2>/dev/null | sort -V); do
  T=$(basename $CK .safetensors); hf upload $R $CK pooler_distill/chatsft/multiturn/$T.safetensors >/dev/null 2>&1
  env $ENV python3 /root/work/pool_eval.py $CK /root/work/ev_0.jsonl /root/work/${T}st_0.jsonl --n 100 $EVARGS --tag "[${T}st0]" > /root/${T}st_0.log 2>&1
  c=$(python3 -c "import json;print(sum(bool(json.loads(l).get('correct')) for l in open('/root/work/${T}st_0.jsonl')))" 2>/dev/null || echo 0)
  echo "[mtg6] $T single shard 0: $c/100 (start $B0)"; [ "$c" -gt "$BC" ] && { BC=$c; BEST=$CK; }
done
[ -n "$BEST" ] || { echo "MTG6_JOB_DONE no copy above the start's shard 0 ($B0)"; exit 0; }
T=$(basename $BEST .safetensors); echo "[mtg6] full screens for $T"
SN=$BC; for i in 1 2; do env $ENV python3 /root/work/pool_eval.py $BEST /root/work/ev_$i.jsonl /root/work/${T}st_$i.jsonl --n 100 $EVARGS --tag "[${T}st$i]" > /root/${T}st_$i.log 2>&1
  c=$(python3 -c "import json;print(sum(bool(json.loads(l).get('correct')) for l in open('/root/work/${T}st_$i.jsonl')))" 2>/dev/null || echo 0); SN=$((SN + c)); echo "[mtg6] single shard $i: $c/100"; done
echo "[mtg6] $T single-turn: $SN/300 (s100 160)"
for b in "br3:/root/work/mt_eval_bridge3.jsonl:(s100 43/31)" "br:/root/work/mt_eval_bridge.jsonl:(s100 23/19)"; do X=${b%%:*}; r=${b#*:}; F=${r%%:*}; NOTE=${r#*:}
  env $ENV python3 /root/work/pool_eval.py $BEST /root/work/eval300.jsonl /root/work/${X}_$T.jsonl --multiturn $F --mt-mode win --n 1000 $EVARGS --tag "[$X-$T]" > /root/${X}_$T.log 2>&1
  echo "[mtg6] $X (win): $(python3 -c "
import json; r=[json.loads(l) for l in open('/root/work/${X}_$T.jsonl')]
print('turn 1', sum(x['correct'] for x in r if x['turn']==0), '/', sum(1 for x in r if x['turn']==0), ', follow-up', sum(x['correct'] for x in r if x['turn']==1), '/', sum(1 for x in r if x['turn']==1))" 2>&1 | tail -1) $NOTE"
done
env SP_BASE=/root/gptq_hf_gq14 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $BEST /root/evalrun_dl_mtg6 \
  --questions /root/work/eval300.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl \
  --b 12 --gen 7000 --budget 2400 --temp 0.6 --maxsrch 7 --pooler none --lora-rank 16 --lora-layers all --stop eos --loop-break answer \
  --eval-file /root/work/dolphinq.jsonl --eval-out /root/work/dl_$T.jsonl > /root/dl_$T.log 2>&1
[ -s /root/dl_judge.py ] && OAI_KEY=$(cat /root/.oai 2>/dev/null) python3 /root/dl_judge.py /root/work/dl_$T.jsonl $T | sed "s/\[mix0\]/[mtg6]/"
echo "[mtg6] (s100: Dolphin 56)"
echo "MTG6_JOB_DONE $(date -u)"
M6
  setsid nohup bash -c 'bash /root/mtg6keep.sh 2>&1 | tee -a /root/mtg6.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  echo "MTG6_LAUNCHED $(date -u)"
fi
# ---- qgen, more workers (2026-10-07 02:20 JST): 2982 queries in 2 h at 8 workers (~20 s a call, nano reasons) - 22k would
# take 15 h. Resumable: the writer restarts at 32 workers; the partial file keeps going to the hub every 15 min.
if [ ! -e /root/.qgen2 ] && [ -s /root/.oai ]; then touch /root/.qgen2
  pkill -f "qgen.py --docs" 2>/dev/null; sleep 3
  ( export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/qgen
    curl -sS -L -o qgen.py "$RAW/sentbart/qgen.py?$(date +%s)"
    echo "[qgen] restart at 32 workers, $(wc -l < /root/qgen/queries.jsonl 2>/dev/null || echo 0) done $(date -u +%H:%M)"
    ( while sleep 900; do [ -s /root/qgen/queries.jsonl ] && hf upload $R /root/qgen/queries.jsonl sentbart/e2e2/queries_partial.jsonl >/dev/null 2>&1; done ) & UP=$!
    python3 -u qgen.py --docs /root/qgen/dl/sentbart/e2e2/docs_for_qgen.jsonl --out /root/qgen/queries.jsonl --workers 32 2>&1 | grep --line-buffered -E "^\[qgen\]|QGEN_"
    kill $UP 2>/dev/null
    [ -s /root/qgen/queries.jsonl ] && hf upload $R /root/qgen/queries.jsonl sentbart/e2e2/queries.jsonl >/dev/null 2>&1 && echo "QGEN_UPLOADED $(wc -l < /root/qgen/queries.jsonl) rows $(date -u)"
  ) >> /root/qgen.log 2>&1 &
  echo "QGEN2_LAUNCHED $(date -u)"
fi
# ---- qgen, third start (2026-10-07 02:45 JST): at 32 workers the writer processed 500 documents and wrote none - the
# failures were not shown (five errors were printed, then silence). qgen.py now says what fails (HTTP code and body,
# empty content with its finish reason), waits 20 s on 429, retries an empty content at 6000 tokens; 16 workers.
if [ ! -e /root/.qgen3 ] && [ -s /root/.oai ]; then touch /root/.qgen3
  pkill -f "qgen.py --docs" 2>/dev/null; sleep 3
  ( export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/qgen
    curl -sS -L -o qgen.py "$RAW/sentbart/qgen.py?$(date +%s)"
    echo "[qgen] restart at 16 workers with error reporting, $(wc -l < /root/qgen/queries.jsonl 2>/dev/null || echo 0) done $(date -u +%H:%M)"
    ( while sleep 900; do [ -s /root/qgen/queries.jsonl ] && hf upload $R /root/qgen/queries.jsonl sentbart/e2e2/queries_partial.jsonl >/dev/null 2>&1; done ) & UP=$!
    python3 -u qgen.py --docs /root/qgen/dl/sentbart/e2e2/docs_for_qgen.jsonl --out /root/qgen/queries.jsonl --workers 16 2>&1 | grep --line-buffered -E "^\[qgen\]|QGEN_"
    kill $UP 2>/dev/null
    [ -s /root/qgen/queries.jsonl ] && hf upload $R /root/qgen/queries.jsonl sentbart/e2e2/queries.jsonl >/dev/null 2>&1 && echo "QGEN_UPLOADED $(wc -l < /root/qgen/queries.jsonl) rows $(date -u)"
  ) >> /root/qgen.log 2>&1 &
  echo "QGEN3_LAUNCHED $(date -u)"
fi
# ---- qgen, fourth start (2026-10-07 02:55 JST): the rate limit (429) hit at 16 workers too, and each 429 spent one of
# the four tries. A 429 now only waits (up to 10 min a document); 12 workers. Resumable as before.
if [ ! -e /root/.qgen4 ] && [ -s /root/.oai ]; then touch /root/.qgen4
  pkill -f "qgen.py --docs" 2>/dev/null; sleep 3
  ( export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/qgen
    curl -sS -L -o qgen.py "$RAW/sentbart/qgen.py?$(date +%s)"
    echo "[qgen] restart at 12 workers, 429 waits, $(wc -l < /root/qgen/queries.jsonl 2>/dev/null || echo 0) done $(date -u +%H:%M)"
    ( while sleep 900; do [ -s /root/qgen/queries.jsonl ] && hf upload $R /root/qgen/queries.jsonl sentbart/e2e2/queries_partial.jsonl >/dev/null 2>&1; done ) & UP=$!
    python3 -u qgen.py --docs /root/qgen/dl/sentbart/e2e2/docs_for_qgen.jsonl --out /root/qgen/queries.jsonl --workers 12 2>&1 | grep --line-buffered -E "^\[qgen\]|QGEN_"
    kill $UP 2>/dev/null
    [ -s /root/qgen/queries.jsonl ] && hf upload $R /root/qgen/queries.jsonl sentbart/e2e2/queries.jsonl >/dev/null 2>&1 && echo "QGEN_UPLOADED $(wc -l < /root/qgen/queries.jsonl) rows $(date -u)"
  ) >> /root/qgen.log 2>&1 &
  echo "QGEN4_LAUNCHED $(date -u)"
fi
# ---- qgen, fifth start (2026-10-07 03:10 JST): 400 rate-limit hits in ten minutes at 12 workers - the limit is tokens a
# minute and nano reasons at length on a short task. reasoning_effort low, 2000 tokens: each call quick and light; the
# 429 body is now printed whole (which limit). 12 workers, resumable.
if [ ! -e /root/.qgen5 ] && [ -s /root/.oai ]; then touch /root/.qgen5
  pkill -f "qgen.py --docs" 2>/dev/null; sleep 3
  ( export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/qgen
    curl -sS -L -o qgen.py "$RAW/sentbart/qgen.py?$(date +%s)"
    echo "[qgen] restart with low reasoning, 12 workers, $(wc -l < /root/qgen/queries.jsonl 2>/dev/null || echo 0) done $(date -u +%H:%M)"
    ( while sleep 900; do [ -s /root/qgen/queries.jsonl ] && hf upload $R /root/qgen/queries.jsonl sentbart/e2e2/queries_partial.jsonl >/dev/null 2>&1; done ) & UP=$!
    python3 -u qgen.py --docs /root/qgen/dl/sentbart/e2e2/docs_for_qgen.jsonl --out /root/qgen/queries.jsonl --workers 12 2>&1 | grep --line-buffered -E "^\[qgen\]|QGEN_"
    kill $UP 2>/dev/null
    [ -s /root/qgen/queries.jsonl ] && hf upload $R /root/qgen/queries.jsonl sentbart/e2e2/queries.jsonl >/dev/null 2>&1 && echo "QGEN_UPLOADED $(wc -l < /root/qgen/queries.jsonl) rows $(date -u)"
  ) >> /root/qgen.log 2>&1 &
  echo "QGEN5_LAUNCHED $(date -u)"
fi
# ---- qgen, waiting for credit (2026-10-07 03:30 JST): the writer aborted on "no credits remaining". This keeper tries the
# teacher every ten minutes and, once it answers, runs the writer again (resumable, 12 workers, low reasoning).
if [ ! -e /root/.qgen6 ] && [ -s /root/.oai ] && ! pgrep -f "qgen6kee[p].sh" >/dev/null; then touch /root/.qgen6
  cat > /root/qgen6keep.sh <<'QK'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill; cd /root/qgen
oai_ok() { python3 - <<'PYO'
import json, urllib.request, sys
key = open("/root/.oai").read().strip()
body = {"model": "gpt-5-nano", "messages": [{"role": "user", "content": "Reply with the word ok."}], "max_completion_tokens": 200}
try:
    d = json.load(urllib.request.urlopen(urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}), timeout=60)); sys.exit(0 if "choices" in d else 1)
except Exception: sys.exit(1)
PYO
}
until oai_ok; do sleep 600; done
pkill -f "qgen.py --docs" 2>/dev/null; sleep 2
curl -sS -L -o qgen.py "$RAW/sentbart/qgen.py?$(date +%s)"
echo "[qgen] credit is back; resuming at $(wc -l < /root/qgen/queries.jsonl 2>/dev/null || echo 0) $(date -u +%H:%M)"
( while sleep 900; do [ -s /root/qgen/queries.jsonl ] && hf upload $R /root/qgen/queries.jsonl sentbart/e2e2/queries_partial.jsonl >/dev/null 2>&1; done ) & UP=$!
python3 -u qgen.py --docs /root/qgen/dl/sentbart/e2e2/docs_for_qgen.jsonl --out /root/qgen/queries.jsonl --workers 12 2>&1 | grep --line-buffered -E "^\[qgen\]|QGEN_"
kill $UP 2>/dev/null
[ -s /root/qgen/queries.jsonl ] && hf upload $R /root/qgen/queries.jsonl sentbart/e2e2/queries.jsonl >/dev/null 2>&1 && echo "QGEN_UPLOADED $(wc -l < /root/qgen/queries.jsonl) rows $(date -u)"
QK
  sed -i "s#\$RAW#$RAW#g" /root/qgen6keep.sh
  setsid nohup bash -c 'bash /root/qgen6keep.sh >> /root/qgen.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "QGEN6_ARMED $(date -u)"
fi
# ---- the box's own logs, mirrored to the hub every ten minutes: readable without the Vast API ----
pkill -f "logmirro[r].sh" 2>/dev/null; pkill -f "logmirror[2].sh" 2>/dev/null   # replaced by logmirror3 (adds the score table)
cat > /root/logmirror3.sh <<'LM'
#!/bin/bash
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
while :; do
  { echo "=== boxlog $(date -u) ==="; echo "--- ctl.log (tail) ---"; tail -n 300 /root/ctl.log 2>/dev/null | cut -c1-300
    echo "--- reeval.log (tail) ---"; tail -n 120 /root/reeval.log 2>/dev/null | cut -c1-300
    echo "--- quant.log (tail) ---"; tail -n 30 /root/quant.log 2>/dev/null | cut -c1-300
    echo "--- wiki.log (tail) ---"; tail -n 8 /root/wiki.log 2>/dev/null | cut -c1-300
    echo "--- ivf.log (tail) ---"; tail -n 8 /root/ivf.log 2>/dev/null | cut -c1-300
    echo "--- retr.log (tail) ---"; tail -n 8 /root/retr.log 2>/dev/null | cut -c1-300
    echo "--- wiki6.log (tail) ---"; tail -n 4 /root/wiki6.log 2>/dev/null | cut -c1-300
    echo "--- terms.log (tail) ---"; tail -n 4 /root/terms.log 2>/dev/null | cut -c1-300
    echo "--- mtsim.log (tail) ---"; tail -n 4 /root/mtsim.log 2>/dev/null | cut -c1-300
    echo "--- mtt.log (tail) ---"; tail -n 12 /root/mtt.log 2>/dev/null | cut -c1-300; tail -n 2 /root/mtt_traces.log 2>/dev/null | cut -c1-300; grep -E "^val " /root/memfit_mem5_run.log 2>/dev/null | tail -3
    echo "--- mtm.log (tail) ---"; tail -n 6 /root/mtm.log 2>/dev/null | cut -c1-300; grep -E "^val " /root/memfit_mem3m_run.log 2>/dev/null | tail -3
    echo "--- mem6.log (tail) ---"; tail -n 10 /root/mem6.log 2>/dev/null | cut -c1-300; grep -E "^val " /root/memfit_mem6_run.log 2>/dev/null | tail -2
    echo "--- mem7.log (tail) ---"; tail -n 10 /root/mem7.log 2>/dev/null | cut -c1-300; grep -E "^val " /root/memfit_mem7_run.log 2>/dev/null | tail -2
    echo "--- qt.log (tail) ---"; tail -n 4 /root/qt.log 2>/dev/null | cut -c1-300
    echo "--- pqjudge.log (tail) ---"; tail -n 5 /root/pqjudge.log 2>/dev/null | cut -c1-200
    echo "--- du_list ---"; cat /root/du_list.txt 2>/dev/null | head -42
    echo "--- r1smoke.log ---"; cat /root/r1smoke.log 2>/dev/null | cut -c1-700
    echo "--- leak ---"; cat /root/leak.txt 2>/dev/null | cut -c1-700
    echo "--- qcount ---"; cat /root/qcount.txt 2>/dev/null | cut -c1-600
    echo "--- hyde.log ---"; tail -n 4 /root/hyde.log 2>/dev/null | cut -c1-300
    echo "--- mtg6.log (tail) ---"; tail -n 12 /root/mtg6.log 2>/dev/null | cut -c1-250; grep -E "^\[step|ONLINE_|rollback|guard" /root/mtg6_run.log 2>/dev/null | tail -n 3 | cut -c1-220; tail -n 1 /root/probe_nq1.log 2>/dev/null | cut -c1-120
    echo "--- qgen.log ---"; tail -n 4 /root/qgen.log 2>/dev/null | cut -c1-300
    echo "--- mtg5b.log (tail) ---"; tail -n 14 /root/mtg5b.log 2>/dev/null | cut -c1-250; grep -E "^\[step|ONLINE_|rollback|guard" /root/mtg5b_run.log 2>/dev/null | tail -n 3 | cut -c1-220
    echo "--- mtg5.log (tail) ---"; tail -n 14 /root/mtg5.log 2>/dev/null | cut -c1-250; grep -E "^\[step|ONLINE_|rollback|guard" /root/mtg5_run.log 2>/dev/null | tail -n 3 | cut -c1-220
    echo "--- s100q.log (tail) ---"; tail -n 14 /root/s100q.log 2>/dev/null | cut -c1-250
    echo "--- mtg4.log (tail) ---"; tail -n 12 /root/mtg4.log 2>/dev/null | cut -c1-250; grep -E "^\[step|ONLINE_" /root/mtg4_run.log 2>/dev/null | tail -n 3 | cut -c1-220
    echo "--- teach2.log (tail) ---"; tail -n 12 /root/teach2.log 2>/dev/null | cut -c1-250; grep -E "^\[step" /root/teach1_run.log 2>/dev/null | tail -n 1 | cut -c1-200
    echo "--- teach.log (tail) ---"; tail -n 16 /root/teach.log 2>/dev/null | cut -c1-250; tail -n 1 /root/probe_s100.log 2>/dev/null | cut -c1-200
    echo "--- moreq4.log (tail) ---"; tail -n 4 /root/moreq4.log 2>/dev/null | cut -c1-250
    echo "--- moreq3.log (tail) ---"; tail -n 10 /root/moreq3.log 2>/dev/null | cut -c1-250
    echo "--- moreq2.log (tail) ---"; tail -n 8 /root/moreq2.log 2>/dev/null | cut -c1-250
    echo "--- moreq.log (tail) ---"; tail -n 14 /root/moreq.log 2>/dev/null | cut -c1-250
    echo "--- dl100.log (tail) ---"; tail -n 3 /root/dl100.log 2>/dev/null | cut -c1-250
    echo "--- mtg3b.log (tail) ---"; tail -n 6 /root/mtg3b.log 2>/dev/null | cut -c1-250
    echo "--- mtg3chk.log (tail) ---"; tail -n 24 /root/mtg3chk.log 2>/dev/null | cut -c1-250; grep -E "^\[step|ONLINE_" /root/mtg4_run.log 2>/dev/null | tail -n 3 | cut -c1-250
    echo "--- mtg3.log (tail) ---"; tail -n 8 /root/mtg3.log 2>/dev/null | cut -c1-250; grep -E "^\[step|^\[guard|ONLINE_" /root/mtg3_run.log 2>/dev/null | tail -n 4 | cut -c1-250
    echo "--- memwin.log (tail) ---"; tail -n 4 /root/memwin.log 2>/dev/null | cut -c1-250; for f in /root/br_win2.log /root/br_none.log; do grep "dialog" $f 2>/dev/null | tail -n 1 | cut -c1-250; done
    echo "--- memcap2.log (tail) ---"; tail -n 4 /root/memcap2.log 2>/dev/null | cut -c1-250
    echo "--- mixcol.log (tail) ---"; tail -n 4 /root/mixcol.log 2>/dev/null | cut -c1-250
    echo "--- memcap.log (tail) ---"; tail -n 6 /root/memcap.log 2>/dev/null | cut -c1-250
    echo "--- dcq.log (tail) ---"; tail -n 5 /root/dcq.log 2>/dev/null | cut -c1-300
    echo "--- searchq.log (tail) ---"; tail -n 5 /root/searchq.log 2>/dev/null | cut -c1-300
    echo "--- mtg2.log (tail) ---"; tail -n 12 /root/mtg2.log 2>/dev/null | cut -c1-250
    echo "--- mix0.log (tail) ---"; tail -n 14 /root/mix0.log 2>/dev/null | cut -c1-250
    echo "--- mtg1d.log (tail) ---"; tail -n 12 /root/mtg1d.log 2>/dev/null | cut -c1-250
    echo "--- mtg1c.log (tail) ---"; tail -n 12 /root/mtg1c.log 2>/dev/null | cut -c1-250
    echo "--- mtg1b.log (tail) ---"; tail -n 8 /root/mtg1b.log 2>/dev/null | cut -c1-250
    echo "--- mtg1.log (tail) ---"; tail -n 10 /root/mtg1.log 2>/dev/null | cut -c1-300; grep -E "^\[step|ONLINE_|\[guard\]|\[warn\]" /root/mtg1_run.log 2>/dev/null | tail -4 | cut -c1-300
    echo "--- mem10d.log (tail) ---"; tail -n 14 /root/mem10d.log 2>/dev/null | cut -c1-300; for f in /root/work/rft10_s0.jsonl /root/work/rft10_s2.jsonl /root/work/mem10_chain.jsonl; do [ -e $f ] && echo "$(basename $f) $(wc -l < $f)"; done | tr "\n" " "; echo
    echo "--- mem10c.log (tail) ---"; tail -n 14 /root/mem10c.log 2>/dev/null | cut -c1-300
    echo "--- mem10b.log (tail) ---"; tail -n 14 /root/mem10b.log 2>/dev/null | cut -c1-300; for f in /root/work/rft10_s*.jsonl /root/work/mem10_chain_h*.jsonl; do [ -e $f ] && echo "$(basename $f) $(wc -l < $f)"; done | tr "\n" " "; echo
    echo "--- mem10.log (tail) ---"; tail -n 12 /root/mem10.log 2>/dev/null | cut -c1-300; for f in /root/work/base_c100_*_s*.jsonl /root/work/rft10_s*.jsonl /root/work/mem10_c100_full_s*.jsonl; do [ -e $f ] && echo "$(basename $f) $(wc -l < $f)"; done | tr "\n" " "; echo
    echo "--- mem9.log (tail) ---"; tail -n 6 /root/mem9.log 2>/dev/null | cut -c1-300
    echo "--- mem8.log (tail) ---"; tail -n 10 /root/mem8.log 2>/dev/null | cut -c1-300; grep -E "^val " /root/memfit_mem8_run.log 2>/dev/null | tail -2
    echo "--- mt.log (tail) ---"; tail -n 8 /root/mt.log 2>/dev/null | cut -c1-300; for f in $(ls -t /root/mt*_*.log 2>/dev/null | head -1); do echo "--- $f (tail) ---"; grep -E "^\[mt |EVAL_DONE|Error|Traceback" $f | tail -3 | cut -c1-300; done
    echo "--- ranker.log (tail) ---"; tail -n 8 /root/ranker.log 2>/dev/null | cut -c1-300
    echo "--- release.log (tail) ---"; tail -n 6 /root/release.log 2>/dev/null | cut -c1-200
    for f in /root/gptq_*.log; do [ -s "$f" ] && { echo "--- $f (tail) ---"; grep -E "^\[gptq\]|^\[out\]|GPTQ_DONE|Error" "$f" | tail -n 4 | cut -c1-200; }; done
    f=$(ls -t /root/q14*_q14*[0-9].log /root/g14*_g14*[0-9].log /root/gq14*_gq14*[0-9].log 2>/dev/null | head -1); [ -s "$f" ] && { echo "--- $f (tail) ---"; tail -n 8 "$f" | cut -c1-220; }
    for f in /root/sft_q*.log; do [ -s "$f" ] && { echo "--- $f (tail) ---"; grep -E "^step [0-9]+ |val" "$f" | tail -n 6 | cut -c1-200; }; done
    for f in $(ls -t /root/online_*.log 2>/dev/null | head -1); do echo "--- $f (tail) ---"; tail -n 60 "$f" | cut -c1-300; done
    echo "--- gpu ---"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null; df -h /root | tail -1
    [ -x /usr/local/bin/s ] && /usr/local/bin/s 20 2>/dev/null | sed 's/^/SCORE /'
    echo "--- eval progress ---"; for f in $(ls -t /root/work/*_out_*.jsonl 2>/dev/null | head -3); do echo "$f $(wc -l < $f) lines, last write $(date -u -r $f +%H:%M)"; done
    echo "--- processes ---"; pgrep -fa "online_loop.p[y]|pool_eval.p[y]|reevalkee[p].sh|build_merged.p[y]|jointfit.p[y]|quantkee[p].sh|gptqkee[p].sh|gptq.p[y]|ivfkee[p].sh|ivf.p[y]|memcheck.p[y]|wikikee[p].sh|build_store.p[y]|embed.p[y]|hf downloa[d]|hf uploa[d]" | cut -c1-120; } > /root/boxlog.txt 2>&1
  hf upload $R /root/boxlog.txt pooler_distill/chatsft/audit/boxlog.txt >/dev/null 2>&1
  sleep 600
done
LM
chmod +x /root/logmirror3.sh
# restart the mirror when its script changed (a running copy keeps the old sections and process list)
if ! pgrep -f "logmirror[3].sh" >/dev/null || [ "$(md5sum < /root/logmirror3.sh)" != "$(cat /root/.logmirror3.md5 2>/dev/null)" ]; then
  pkill -f "logmirror[3].sh" 2>/dev/null; sleep 1; md5sum < /root/logmirror3.sh > /root/.logmirror3.md5
  setsid nohup bash /root/logmirror3.sh >> /proc/1/fd/1 2>&1 < /dev/null &
fi
# ---- the teacher judge's reachability, every control run (one tiny call; the status and the error text, never the key) ----
if [ -s /root/.dsk ]; then
  python3 - <<'DD'
import json, urllib.request, urllib.error
key = open("/root/.dsk").read().strip()
body = {"model": "deepseek-flash", "max_tokens": 20, "temperature": 0, "messages": [{"role": "user", "content": "Reply with the word ok."}]}
try:
    d = json.load(urllib.request.urlopen(urllib.request.Request("https://api.deepseek.com/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}), timeout=60))
    print("DSK_DIAG ok", d.get("model"))
except urllib.error.HTTPError as e:
    print("DSK_DIAG http", e.code, e.read().decode()[:300].replace("\n", " "))
except Exception as e:
    print("DSK_DIAG fail", type(e).__name__, str(e)[:200])
DD
else echo "DSK_DIAG no /root/.dsk on this box"; fi
if [ ! -f /root/.rejudge_s4dol ] && [ -s /root/work/s4dol_out_0.jsonl ] && [ -s /root/.dsk ]; then
  touch /root/.rejudge_s4dol
  ( cd /root/work && mkdir -p /root/rj && cp /root/work/s4dol_out_0.jsonl /root/rj/s4dsk_out_0.jsonl
    sed -n "/^  if \[ \"\$RQSRC\" = \"dolphinh\" \]; then/,/^PYJ$/p" /root/ctl_cmd.sh | sed '1d' | sed 's#/root/work/\*_out_0.jsonl#/root/rj/*_out_0.jsonl#g; s#"/root/work/" + os.path.basename#"/root/rj/" + os.path.basename#' > /root/rj/judge.sh
    bash /root/rj/judge.sh 2>&1 | sed 's/^DOLPHIN_ACC/DOLPHIN_ACC_REJUDGE s4 deepseek:/'
    export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); hf upload baya1116/hypernet-sp-distill /root/rj/s4dsk_judged.jsonl pooler_distill/chatsft/rollouts/s4dsk_judged.jsonl >/dev/null 2>&1 ) &
fi
if [ -s /root/.oai ]; then
  python3 - <<'OD'
import json, urllib.request, urllib.error
key = open("/root/.oai").read().strip()
body = {"model": "gpt-5-nano", "max_completion_tokens": 50, "messages": [{"role": "user", "content": "Reply with the word ok."}]}
try:
    d = json.load(urllib.request.urlopen(urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}), timeout=60))
    print("OAI_DIAG ok", d.get("model"))
except urllib.error.HTTPError as e:
    print("OAI_DIAG http", e.code, e.read().decode()[:300].replace("\n", " "))
except Exception as e:
    print("OAI_DIAG fail", type(e).__name__, str(e)[:200])
OD
fi
# ---- one-off audit of the teacher judge (no GPU): does nano's pass inflate over training against a strict judge? ----
if [ "${AUDIT:-0}" = 1 ] && [ ! -f /root/.audit_judge ]; then
  touch /root/.audit_judge
  ( export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
    for run in g7 g8 g9; do hf download $R --include "pooler_distill/chatsft/online/$run/rollouts.jsonl" --local-dir /root/hfdl >/dev/null 2>&1; done
    OAI_KEY=$(cat /root/.oai 2>/dev/null) python3 - <<'PYAUD'
import json, os, re, random, urllib.request, time
from concurrent.futures import ThreadPoolExecutor
INFO = re.compile(r"<information>.*?</information>\n?", re.S)
def L(t): return len(INFO.sub("", t.split("</think>")[0]).split())
key = os.environ.get("OAI_KEY", "")
STRICT = ("You are a strict grader. Judge ONLY whether the assistant's FINAL answer is correct and complete relative to the reference answer. "
          "Ignore length, explanations, tone and formatting entirely. If the assistant never commits to a final answer, or the final answer differs from the reference "
          "in substance, it is incorrect. Reply with JSON only: {\"final_answer_correct\": true/false, \"commits_to_an_answer\": true/false}")
def strict(q, ref, text):
    reply = text.split("</think>")[-1].strip() if "</think>" in text else ""
    if not reply: return {"final_answer_correct": False, "commits_to_an_answer": False, "unfinished": True}
    body = {"model": "gpt-5-mini", "max_completion_tokens": 1500, "messages": [{"role": "system", "content": STRICT},
            {"role": "user", "content": f"QUESTION:\n{q[:2000]}\n\nREFERENCE ANSWER:\n{ref[:3000]}\n\nASSISTANT ANSWER:\n{reply[:3000]}"}]}
    for _ in range(3):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}), timeout=180))
            c = d["choices"][0]["message"].get("content") or ""; return json.loads(c[c.find("{"): c.rfind("}") + 1])
        except Exception as e: time.sleep(3); err = type(e).__name__
    return {"error": err}
# the reference answers: dolphin_v2 (the training problems of g7-g9)
ref = {}
for l in open("/root/hfdl/pooler_distill/chatsft/dolphin_v2.jsonl"):
    try: r = json.loads(l); ref[r["q"].strip()] = r["reply"]
    except Exception: pass
rows = []
for run, early, late in (("g7", 80, 500), ("g8", 60, 140), ("g9", 60, 140)):
    for l in open(f"/root/hfdl/pooler_distill/chatsft/online/{run}/rollouts.jsonl"):
        try: r = json.loads(l)
        except Exception: continue
        if r.get("kind") != "reason" or r["q"].strip() not in ref: continue
        ph = "early" if r["step"] <= early else ("late" if r["step"] >= late else None)
        if ph: rows.append({"run": run, "phase": ph, "step": r["step"], "q": r["q"], "text": r["text"], "nano": int(r["reward"] >= 1.0), "think": L(r["text"])})
random.Random(0).shuffle(rows)
sample = []
for ph in ("early", "late"):
    for nano in (1, 0):
        sample += [r for r in rows if r["phase"] == ph and r["nano"] == nano][:70]
with ThreadPoolExecutor(max_workers=8) as ex: res = list(ex.map(lambda r: strict(r["q"], ref[r["q"].strip()], r["text"]), sample))
for r, v in zip(sample, res): r["strict"] = v; r.pop("text", None)
def summ(ph, nano):
    g = [r for r in sample if r["phase"] == ph and r["nano"] == nano and "error" not in r["strict"]]
    if not g: return "n/a"
    sc = sum(1 for r in g if r["strict"].get("final_answer_correct")); th = sorted(r["think"] for r in g)
    return f"strict-correct {100*sc/len(g):.0f}% of {len(g)}, think median {th[len(th)//2]}"
print("JUDGE_AUDIT")
for ph in ("early", "late"):
    print(f"  {ph}: nano-pass -> {summ(ph, 1)} | nano-fail -> {summ(ph, 0)}")
# nano pass rate by thinking length within the late phase (all late rows, no API)
late = [r for r in rows if r["phase"] == "late"]; late.sort(key=lambda r: r["think"]); q = len(late) // 4
for k in range(4):
    g = late[k*q:(k+1)*q]
    if g: print(f"  late, thinking quartile {k+1} (median {sorted(r['think'] for r in g)[len(g)//2]} words): nano pass {100*sum(r['nano'] for r in g)/len(g):.0f}%")
json.dump({"sample": sample}, open("/root/work/judge_audit.json", "w"))
PYAUD
    hf upload $R /root/work/judge_audit.json pooler_distill/chatsft/audit/judge_audit.json >/dev/null 2>&1; echo "AUDIT_DONE $(date -u)"
  ) >> /proc/1/fd/1 2>&1 &
fi
if [ "$MODE" = "sft" ]; then
  # Reproduce, under quantization and in the compressed context, the traces that scored. This is
  # not matching bf16 - it is the quantized system learning the behaviour that worked, which is how
  # this lineage was trained when it lived in 4-bit. Scales and biases move; codes and pooler stay.
  SRUN=${SRUN:-s1}
  pkill -f "afterkee[p].sh"; pkill -f "evalkee[p].sh"; pkill -f "dwqkee[p].sh"; pkill -f "probekee[p].sh"; pkill -f "poolkee[p].sh"; pkill -f "jointkee[p].sh"; sleep 5
  if [ ! -s /root/work/dwq_calib/correct.jsonl ]; then
    for try in 1 2 3; do hf download baya1116/hypernet-sp-distill --include "pooler_distill/grpo_pool3/rollouts.jsonl" --local-dir /root/hfdl 2>&1 | tail -1 && break; sleep 10; done
    python3 - <<'PYS'
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/root/eval_hf200")
held = set()
for line in open("/root/work/eval300.jsonl"):
    try: held.add(json.loads(line).get("q", ""))
    except Exception: pass
n = 0
with open("/root/work/dwq_calib/correct.jsonl", "w") as out:
    for line in open("/root/hfdl/pooler_distill/grpo_pool3/rollouts.jsonl"):
        try: d = json.loads(line)
        except Exception: continue
        if not (d.get("correct") and d.get("grounded") and d.get("landed")): continue
        q, t = d.get("q", ""), d.get("text", "")
        if not q or not t or q in held: continue
        head = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False) + "<think>\n"
        out.write(json.dumps({"text": head + t}, ensure_ascii=False) + "\n"); n += 1
print(f"[sft] {n} correct, grounded, landed traces; held-out questions excluded")
PYS
  fi
  R=baya1116/hypernet-sp-distill; D=pooler_distill/grpo_pool3_step200_q4
  for p in p_t06q4 p_t06bf16; do
    for i in 0 1 2; do
      [ -s /root/work/${p}_out_$i.jsonl ] && hf upload $R /root/work/${p}_out_$i.jsonl $D/rollouts/${p}_$i.jsonl >/dev/null 2>&1
    done
  done
  echo "probe rollouts uploaded: $(cat /root/work/p_t06*_out_*.jsonl 2>/dev/null | wc -l)"
  echo "traces that scored: $(wc -l < /root/work/dwq_calib/correct.jsonl)"
  HF=/root/sft_hf_$SRUN; POOL=/root/pooler_sft_$SRUN.safetensors; LOG=/root/sft_$SRUN.log
  RUNNING=0
  pgrep -f "jointfit.p[y]" >/dev/null && { RUNNING=1; echo "already running: $(tail -1 $LOG)"; }
  [ -s $HF/model.safetensors ] && { RUNNING=1; echo "sft run $SRUN already finished"; }
  if [ "$RUNNING" = "0" ]; then
    pkill -f "pool_eval.p[y]"; sleep 8; pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 2
    rm -f $LOG
    cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/jointfit.py \
      --ckpt /root/pooler200.safetensors --data /root/work/dwq_calib/correct.jsonl --objective ce \
      --out-hf $HF --out-mlx /root/sft_mlx4_$SRUN --out-pooler $POOL --state /root/sft/$SRUN.pt --log $LOG \
      --clip-search 0 --lr-q ${SLRQ:-2e-6} --lr-p 0 --val 8 --val-every 2 --selftest 3 2>&1 | tail -14
    grep -q "^step 3 " $LOG 2>/dev/null || { echo "SFT SELFTEST FAILED - not launching"; exit 0; }
    rm -f $LOG
    cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/jointfit.py \
      --ckpt /root/pooler200.safetensors --data /root/work/dwq_calib/correct.jsonl --objective ce \
      --out-hf $HF --out-mlx /root/sft_mlx4_$SRUN --out-pooler $POOL --state /root/sft/$SRUN.pt --log $LOG \
      --clip-search 0 --lr-q ${SLRQ:-2e-6} --lr-p 0 --steps ${SSTEPS:-1500} --val 24 --val-every ${SVAL:-50} \
      >> /root/sft_run_$SRUN.log 2>&1 < /dev/null &
    sleep 20
  fi
  cat > /root/sftkeep.sh <<SKP
#!/bin/bash
SRUN=$SRUN; HF=$HF; POOL=$POOL
SKP
  cat >> /root/sftkeep.sh <<'SKP2'
until [ -s $HF/model.safetensors ] && grep -q JOINTFIT_DONE /root/sft_run_$SRUN.log 2>/dev/null; do sleep 60; done
pkill -f "jointfit.p[y]"; sleep 10
while :; do
  done=1
  for i in 0 1 2; do
    want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
    have=$(wc -l < /root/work/${SRUN}_out_$i.jsonl 2>/dev/null || echo 0)
    [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
    done=0
    pgrep -f "pool_eval.py .* /root/work/ev_$i.jsonl" >/dev/null && continue
    echo "[$SRUN-eval $(date -u +%H:%M)] shard $i at $have/$want - starting (quantized, trained on its own scored traces)"
    cd /root/work && SP_BASE=$HF SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py \
      /root/pooler200.safetensors /root/work/ev_$i.jsonl /root/work/${SRUN}_out_$i.jsonl \
      --n 999 --rw 768 --maxd 384 --samepage 1 --decode plain --temp ${STEMP:-0.9} --tag "[$SRUN$i]" \
      >> /root/${SRUN}_$i.log 2>&1 < /dev/null &
    sleep 60
  done
  [ "$done" = "1" ] && break
  sleep 120
done
echo "[$SRUN-eval $(date -u +%H:%M)] complete"
SKP2
  chmod +x /root/sftkeep.sh
  pkill -f "sftkee[p].sh"; sleep 1
  setsid nohup bash /root/sftkeep.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  sleep 30; t; echo "SFT_LAUNCH_DONE $SRUN $(date -u)"
  exit 0
fi

if [ "$MODE" = "idle" ]; then
  # bootstrapped and waiting for instructions; nothing below must run
  echo "idle $(date -u +%H:%M): $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader 2>/dev/null)"
  exit 0
fi
if [ "$MODE" = "pack" ]; then
  # jointfit measured a dequantized directory and saved its trained scales and biases, but never
  # wrote the packed directory the app loads. Build it, prove it unpacks to what was measured, ship it.
  export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
  R=baya1116/hypernet-sp-distill; D=pooler_distill/grpo_pool3_step200_q4
  SRUN=${SRUN:-s1}
  ls -la /root/sft/$SRUN.pt /root/sft_hf_$SRUN/model.safetensors 2>&1 | tail -2
  cd /root/work && python3 /root/work/packmlx.py --base /root/eval_hf200 --hf /root/sft_hf_$SRUN \
    --state /root/sft/$SRUN.pt --out /root/sft_mlx4_$SRUN 2>&1 | tail -4
  python3 /root/work/checkmlx.py /root/sft_mlx4_$SRUN /root/sft_hf_$SRUN 2>&1 | tail -4 | tee /root/checkmlx_$SRUN.txt
  grep -q MLX_CHECK_OK /root/checkmlx_$SRUN.txt || { echo "PACK_FAILED $SRUN - not uploading"; exit 0; }
  echo "--- uploading ($(du -shL /root/sft_mlx4_$SRUN | cut -f1)) ---"
  hf upload $R /root/sft_mlx4_$SRUN $D/sft_${SRUN}_mlx4 2>&1 | tail -1
  hf upload $R /root/sft/$SRUN.pt $D/sft_${SRUN}_params.pt 2>&1 | tail -1
  hf upload $R /root/sft_$SRUN.log $D/logs/sft_$SRUN.log >/dev/null 2>&1
  hf upload $R /root/sft_run_$SRUN.log $D/logs/sft_run_$SRUN.log >/dev/null 2>&1
  for i in 0 1 2; do hf upload $R /root/work/${SRUN}_out_$i.jsonl $D/rollouts/${SRUN}_$i.jsonl >/dev/null 2>&1; done
  echo "PACK_DONE $SRUN $(date -u)"
  exit 0
fi
if [ "$MODE" = "quant" ]; then
  # The chat-era model packed for the app, by the one quantization that worked (docs/quantization_4bit.md): the
  # 4-bit grid's scales and biases learn the lineage's own verified traces in the compressed context; codes and
  # pooler stay. QSRC is an online run folded into QBASE first; the traces are the search GRPO's scored rollouts
  # plus the replay pool, both in the chat-era format, held-out questions removed.
  QRUN=${QRUN:-q14}; QSRC=${QSRC:-g14}; QBASE=${QBASE:-g10m_hf}; QLAYERS=${QLAYERS:-all}; QTAG=${QTAG:-}   # QTAG names a second arm's hub directories
  export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
  HF=/root/sft_hf_$QRUN; LOG=/root/sft_$QRUN.log; MLX=/root/sft_mlx4_$QRUN
  if [ "${QREDO:-0}" = 1 ] && [ ! -f /root/.qredo_$QRUN ]; then   # once: measure again with the current evaluator (and its pooler loading)
    pkill -f "quantkee[p].sh"; pkill -f "pool_eval.p[y]"; sleep 5; rm -f /root/work/${QRUN}_out_*.jsonl /root/checkmlx_$QRUN.txt; touch /root/.qredo_$QRUN; echo "[quant] $QRUN: evaluation reset"
  fi
  # a keeper for this run is left alone while its training runs or its model directory exists; a keeper waiting on a
  # training that died (the disk filled up as q14b wrote its directory) is replaced
  if pgrep -f "quantkee[p].sh" >/dev/null && grep -q "^QRUN=$QRUN;" /root/quantkeep.sh 2>/dev/null && grep -q "QSHARDS=" /root/quantkeep.sh && { pgrep -f "jointfit.p[y]" >/dev/null || [ -s $HF/model.safetensors ]; }; then echo "QUANT_SKIP: $QRUN is already running"; exit 0; fi
  pkill -f "onlinekee[p].sh"; pkill -f "reevalkee[p].sh"; pkill -f "quantkee[p].sh"; pkill -f "online_loop.p[y]"; pkill -f "pool_eval.p[y]"; pkill -f "jointfit.p[y]"; sleep 8
  pkill -9 -f "online_loop.p[y]" 2>/dev/null; pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 2
  # room for the run's outputs (a dequantized directory is 3.5 GB): earlier arms' directories are on the hub packed
  rm -rf /root/evalrun_* /root/wikidl /root/hfdl/pooler_distill/chatsft/online/*/latest.safetensors /root/.cache/pip /root/.cache/huggingface/hub/datasets--wikimedia--wikipedia /root/hfdl/pooler_distill/chatsft/s4_hf /root/reeval_hf_g10m /root/base_distill /root/online_g8 /root/online_g9 /root/online_g7 /root/gptq_mlx4_* 
  for d in /root/sft_hf_q* /root/sft_mlx4_q*; do [ -d "$d" ] && [ "$d" != "$HF" ] && grep -q "QUANT_UPLOADED $(basename $d | sed 's/sft_hf_//; s/sft_mlx4_//') " /root/quant.log 2>/dev/null && rm -rf "$d"; done
  du -sh /root/* /root/hfdl/pooler_distill/chatsft/* 2>/dev/null | sort -h | tail -12 | tr '\n' ' '; echo
  echo "[disk] $(df -h /root | awk 'NR==2{print $4" free"}')"
  MB=/root/hfdl/pooler_distill/chatsft/$QBASE; HFM=/root/reeval_hf_${QSRC}m; PCK=/root/reeval_${QSRC}m_pooler.safetensors
  if [ ! -s $HFM/model.safetensors ] || [ ! -s $PCK ]; then
    [ -s $MB/model.safetensors ] || hf download $R --include "pooler_distill/chatsft/$QBASE/*" --local-dir /root/hfdl >/dev/null 2>&1
    SRC=/root/online_$QSRC/latest.safetensors
    if [ ! -s $SRC ]; then
      for try in 1 2 3 4 5 6; do hf download $R --include "pooler_distill/chatsft/online/$QSRC/latest.safetensors" --local-dir /root/hfdl >/dev/null 2>&1; [ -s /root/hfdl/pooler_distill/chatsft/online/$QSRC/latest.safetensors ] && break; sleep 20; done
      SRC=/root/hfdl/pooler_distill/chatsft/online/$QSRC/latest.safetensors
    fi
    [ -s $SRC ] || { echo "QUANT_ABORT $QRUN: online/$QSRC checkpoint not found"; exit 0; }
    rm -rf $HFM
    cd /root/work && SP_BASE=$MB SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 python3 /root/work/build_merged.py $SRC $HFM $PCK 16 $QLAYERS 2>&1 | grep -E "^\[merge\]|MERGE_DONE|Error|assert|unexpected" | tail -4
    [ -s $HFM/model.safetensors ] || { echo "QUANT_ABORT $QRUN: merge of online/$QSRC failed"; exit 0; }
  fi
  QCODES=${QCODES:-}   # a gptq.py state: the grid's training starts from GPTQ's codes instead of round-to-nearest
  if [ -n "$QCODES" ] && [ -s "$QCODES" ] && ! python3 -c "import torch,sys; torch.load(sys.argv[1], map_location='cpu')" $QCODES 2>/dev/null; then
    echo "[quant] $QCODES is not readable (cut short by the disk) - made again"; rm -f $QCODES
  fi
  if [ -n "$QCODES" ] && [ ! -s "$QCODES" ]; then
    [ -s /root/work/qcal_q14.jsonl ] || { echo "QUANT_ABORT $QRUN: no calibration traces for the GPTQ state"; exit 0; }
    rm -f $QCODES
    cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/gptq.py --base $HFM --data /root/work/qcal_q14.jsonl --out-hf /root/gptq_hf_tmp --out-mlx /root/gptq_mlx4_tmp --state $QCODES --no-dirs 1 > /root/gptq_state.log 2>&1
    python3 -c "import torch,sys; torch.load(sys.argv[1], map_location='cpu')" $QCODES 2>/dev/null || rm -f $QCODES   # a state cut short by the disk is not a state
    [ -s "$QCODES" ] || { echo "QUANT_ABORT $QRUN: the GPTQ state was not made: $(grep -E 'Error' /root/gptq_state.log | tail -1 | cut -c1-160)"; exit 0; }
    echo "[quant] GPTQ state made: $QCODES"
  fi
  mkdir -p /root/sft; [ -n "$QCODES" ] && echo "$QCODES" > /root/sft/$QRUN.codes   # a rebuild of the directory must use the same codes
  QDATA=/root/work/qcal_$QRUN.jsonl
  [ -s /root/online_g10/rollouts.jsonl ] || hf download $R --include "pooler_distill/chatsft/online/g10/rollouts.jsonl" --local-dir /root/hfdl >/dev/null 2>&1
  python3 - "$QDATA" "$HFM" <<'PYQ'
import json, re, random, sys
from transformers import AutoTokenizer
out, base = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(base)
held = set()
for line in open("/root/work/eval300.jsonl"):
    try: held.add((json.loads(line).get("q") or "").strip())
    except Exception: pass
degen = re.compile(r"begin_of_thought|end_of_thought|\b(\w+(?:\W+\w+){0,3})\b(?:\W+\1\b){4,}")
def head(q): return tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False) + "<think>\n"
rows, src = [], "/root/online_g10/rollouts.jsonl"
import os
if not os.path.exists(src): src = "/root/hfdl/pooler_distill/chatsft/online/g10/rollouts.jsonl"
n_g = n_r = 0
for line in open(src):
    try: d = json.loads(line)
    except Exception: continue
    w = d.get("why") if isinstance(d.get("why"), dict) else {}
    if not (w.get("correct") and w.get("grounded") and w.get("clean") and not w.get("unfinished")): continue
    q, t = (d.get("q") or "").strip(), d.get("text") or ""
    if not q or not t or q in held or degen.search(t) or "</think>" not in t: continue
    rows.append({"text": head(q) + t}); n_g += 1
for f in ("/root/work/replay_clean.jsonl", "/root/hfdl/pooler_distill/chatsft/replay_v1.jsonl"):
    if os.path.exists(f):
        for line in open(f):
            try: d = json.loads(line)
            except Exception: continue
            q, t = (d.get("q") or "").strip(), d.get("text") or ""
            if not q or not t or q in held or degen.search(t) or "</think>" not in t: continue
            rows.append({"text": head(q) + t}); n_r += 1
        break
random.Random(0).shuffle(rows)
with open(out, "w") as o:
    for r in rows: o.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[quant data] {len(rows)} traces: {n_g} scored search-GRPO rollouts + {n_r} replay traces; held-out questions excluded")
PYQ
  [ -s $QDATA ] || { echo "QUANT_ABORT $QRUN: no calibration traces"; exit 0; }
  if [ ! -s $HF/model.safetensors ] && [ -s /root/sft/$QRUN.pt ] && grep -q "^step ${QSTEPS:-1500} " $LOG 2>/dev/null; then
    # the training finished but its directory was never written: rebuild it from the state
    cd /root/work && python3 /root/work/dequant_state.py --base $HFM --state /root/sft/$QRUN.pt --out $HF 2>&1 | tail -3 | tee -a /root/sft_run_$QRUN.log
  fi
  if [ ! -s $HF/model.safetensors ]; then
    rm -f $LOG
    cd /root/work && SP_BASE=$HFM PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/jointfit.py \
      --ckpt $PCK --data $QDATA --objective ce --out-hf $HF --out-mlx $MLX --out-pooler /root/pooler_sft_$QRUN.safetensors \
      --state /root/sft/$QRUN.pt --log $LOG --clip-search 0 --lr-q ${QLRQ:-2e-6} --lr-p ${QLRP:-0} ${QCODES:+--codes-from $QCODES} --val 8 --val-every 2 --selftest 3 2>&1 | tail -8
    grep -q "^step 3 " $LOG 2>/dev/null || { echo "QUANT_ABORT $QRUN: selftest failed"; exit 0; }
    rm -f $LOG
    cd /root/work && SP_BASE=$HFM PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/jointfit.py \
      --ckpt $PCK --data $QDATA --objective ce --out-hf $HF --out-mlx $MLX --out-pooler /root/pooler_sft_$QRUN.safetensors \
      --state /root/sft/$QRUN.pt --log $LOG --clip-search 0 --lr-q ${QLRQ:-2e-6} --lr-p ${QLRP:-0} ${QCODES:+--codes-from $QCODES} --steps ${QSTEPS:-1500} --val 24 --val-every ${QVAL:-50} \
      >> /root/sft_run_$QRUN.log 2>&1 < /dev/null &
    sleep 20
  fi
  cat > /root/quantkeep.sh <<QKP
#!/bin/bash
QRUN=$QRUN; HF=$HF; HFM=$HFM; PCK=$PCK; MLX=$MLX; R=$R; QSRC=$QSRC; QTAG=$QTAG; QLRP=${QLRP:-0}; QCODES=$QCODES; QTEMP=${QTEMP:-0.6}; QGEN=${QGEN:-4000}; QN=${QN:-34}; QSHARDS=${QSHARDS:-1}
export HF_TOKEN=$HF_TOKEN
QKP
  cat >> /root/quantkeep.sh <<'QKP2'
until [ -s $HF/model.safetensors ] && grep -q JOINTFIT_DONE /root/sft_run_$QRUN.log 2>/dev/null; do sleep 60; done
pkill -f "jointfit.p[y]"; sleep 10
echo "[$QRUN] jointfit done: $(grep -E '^step [0-9]+ ' /root/sft_$QRUN.log | tail -1 | cut -c1-120)"
if [ "$QLRP" != "0" ]; then
  # the pooler trained too: the evaluation and the app directory take the trained tensors over the run's input pooler
  python3 - "$PCK" /root/pooler_sft_$QRUN.safetensors /root/pooler_eval_$QRUN.safetensors <<'PYP'
import sys
from safetensors.torch import load_file, save_file
base, trained, out = sys.argv[1:4]
d = load_file(base); t = load_file(trained); n = 0
for k, v in t.items():
    k = k[len("pooler."):] if k.startswith("pooler.") else k   # jointfit writes prefixed keys, the merged pooler is bare
    if k in d and d[k].shape == v.shape: d[k] = v.to(d[k].dtype); n += 1
save_file({k: v.contiguous() for k, v in d.items()}, out); print(f"[pooler] {n} of {len(d)} tensors taken from the trained pooler")
PYP
  PCK=/root/pooler_eval_$QRUN.safetensors
fi
if [ ! -s $MLX/model.safetensors ] || ! grep -q MLX_CHECK_OK /root/checkmlx_$QRUN.txt 2>/dev/null; then
  cd /root/work && python3 /root/work/packmlx.py --base $HFM --hf $HF --state /root/sft/$QRUN.pt --out $MLX ${QCODES:+--codes-from $QCODES} 2>&1 | tail -3
  python3 /root/work/checkmlx.py $MLX $HF 2>&1 | tail -4 | tee /root/checkmlx_$QRUN.txt
fi
if grep -q MLX_CHECK_OK /root/checkmlx_$QRUN.txt; then
  cp $PCK $MLX/pooler.safetensors
  echo "[$QRUN] packed ($(du -shL $MLX | cut -f1)), uploading the app directory and the merged bf16 model"
  for try in 1 2 3; do hf upload $R $MLX pooler_distill/chatsft/${QSRC}_mlx4$QTAG >/dev/null 2>&1 && break; sleep 30; done
  if [ -z "$QTAG" ]; then cp $PCK $HFM/pooler.safetensors
    for try in 1 2 3; do hf upload $R $HFM pooler_distill/chatsft/${QSRC}m_hf >/dev/null 2>&1 && break; sleep 30; done; fi
  hf upload $R /root/sft/$QRUN.pt pooler_distill/chatsft/${QSRC}_mlx4${QTAG}_params.pt >/dev/null 2>&1
  hf upload $R /root/sft_$QRUN.log pooler_distill/chatsft/logs/sft_$QRUN.log >/dev/null 2>&1
  echo "QUANT_UPLOADED $QRUN -> ${QSRC}_mlx4$QTAG $(date -u)"
else
  echo "QUANT_PACK_FAILED $QRUN - not uploading"
fi
# the packed model on the search held-out, same protocol as the bf16 measurement (temperature 0.6, 4000 tokens); the first
# shard (34 rollouts, the same 17 questions for every arm) is the screen, and a promising arm gets the other shards
# through the reeval mode (RKIND=sftdir), then RN=100 for the one that ships
for i in $(seq 0 $((QSHARDS - 1))); do
  [ -s /root/work/${QRUN}_out_$i.jsonl ] && [ "$(wc -l < /root/work/${QRUN}_out_$i.jsonl)" -ge "$QN" ] && continue
  cd /root/work && SP_BASE=$HF SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/pool_eval.py \
    $PCK /root/work/ev_$i.jsonl /root/work/${QRUN}_out_$i.jsonl --n $QN --rw 768 --maxd 384 --samepage 1 --decode plain --temp $QTEMP --gen $QGEN --stop eos --replycap 600 --tag "[$QRUN$i]" >> /root/${QRUN}_$i.log 2>&1
  hf upload $R /root/work/${QRUN}_out_$i.jsonl pooler_distill/chatsft/rollouts/${QRUN}_$i.jsonl >/dev/null 2>&1
  echo "[$QRUN] shard $i $(tail -1 /root/${QRUN}_$i.log | cut -c1-120) | $(grep -h -m1 "pooler restored\|WARNING: no pooler" /root/${QRUN}_$i.log)"
done
python3 - "$QRUN" <<'PYS'
import json, sys, glob
rows = [json.loads(l) for f in sorted(glob.glob(f"/root/work/{sys.argv[1]}_out_*.jsonl")) for l in open(f) if l.strip()]
n = len(rows); c = sum(1 for r in rows if r.get("correct")); g = sum(1 for r in rows if r.get("grounded")); s = sum(float(r.get("ns", 0) or 0) for r in rows)
print(f"QUANT_EVAL_DONE {sys.argv[1]}: correct {100*c/max(n,1):.1f}%  grounded {100*g/max(n,1):.0f}%  searches {s/max(n,1):.1f}  ({n} rollouts, shard 0 of the held-out = the screen)")
PYS
QKP2
  chmod +x /root/quantkeep.sh
  setsid nohup bash -c 'bash /root/quantkeep.sh 2>&1 | tee -a /root/quant.log' >> /proc/1/fd/1 2>&1 < /dev/null &
  sleep 30; tail -2 /root/sft_run_$QRUN.log 2>/dev/null | cut -c1-160; echo "QUANT_LAUNCH_DONE $QRUN $(date -u)"
  exit 0
fi
if [ "$MODE" = "gptq" ]; then
  # GPTQ onto the app's grid (gptq.py): codes chosen to preserve each layer's outputs on the lineage's own traces,
  # instead of round-to-nearest. Packed, checked, published as ${GSRC}_mlx4${GTAG}, measured with the model's own pooler.
  GRUN=${GRUN:-gq14}; GSRC=${GSRC:-g14}; GTAG=${GTAG:-g}; GSKIP=${GSKIP:-}
  export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); R=baya1116/hypernet-sp-distill
  HF=/root/gptq_hf_$GRUN; MLX=/root/gptq_mlx4_$GRUN; HFM=/root/reeval_hf_${GSRC}m; PCK=/root/reeval_${GSRC}m_pooler.safetensors
  if pgrep -f "gptqkee[p].sh" >/dev/null && grep -q "^GRUN=$GRUN;" /root/gptqkeep.sh 2>/dev/null; then echo "GPTQ_SKIP: $GRUN is already running"; exit 0; fi
  pkill -f "onlinekee[p].sh"; pkill -f "reevalkee[p].sh"; pkill -f "quantkee[p].sh"; pkill -f "gptqkee[p].sh"; pkill -f "online_loop.p[y]"; pkill -f "pool_eval.p[y]"; pkill -f "jointfit.p[y]"; sleep 8
  pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 2
  rm -rf /root/evalrun_* /root/wikidl /root/.cache/pip
  for d in /root/sft_hf_q* /root/sft_mlx4_q*; do [ -d "$d" ] && grep -q "QUANT_UPLOADED $(basename $d | sed 's/sft_hf_//; s/sft_mlx4_//') " /root/quant.log 2>/dev/null && rm -rf "$d"; done
  echo "[disk] $(df -h /root | awk 'NR==2{print $4" free"}')"
  [ -s $HFM/model.safetensors ] && [ -s $PCK ] || { echo "GPTQ_ABORT $GRUN: $HFM or its pooler is missing (run the quant mode once for $GSRC)"; exit 0; }
  QDATA=/root/work/qcal_q14.jsonl; [ -s $QDATA ] || { echo "GPTQ_ABORT $GRUN: no calibration traces at $QDATA"; exit 0; }
  cat > /root/gptqkeep.sh <<GK
#!/bin/bash
GRUN=$GRUN; GSRC=$GSRC; GTAG=$GTAG; GSKIP=$GSKIP; HF=$HF; MLX=$MLX; HFM=$HFM; PCK=$PCK; R=$R; QDATA=$QDATA; GN=${GN:-34}; GSHARDS=${GSHARDS:-3}
export HF_TOKEN=$HF_TOKEN
GK
  cat >> /root/gptqkeep.sh <<'GK2'
if [ ! -s $HF/model.safetensors ]; then
  rm -rf $HF $MLX
  cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/gptq.py --base $HFM --data $QDATA --out-hf $HF --out-mlx $MLX --state /root/gptq_state_$GRUN.pt ${GSKIP:+--skip $GSKIP} > /root/gptq_$GRUN.log 2>&1
  grep -q GPTQ_DONE /root/gptq_$GRUN.log || { echo "GPTQ_ABORT $GRUN: $(grep -E 'Error|error' /root/gptq_$GRUN.log | tail -1 | cut -c1-200)"; exit 0; }
  echo "[$GRUN] $(grep -E '^\[gptq\] (layer 27|lm_head)' /root/gptq_$GRUN.log | tail -1)"
fi
python3 /root/work/checkmlx.py $MLX $HF 2>&1 | tail -3 | tee /root/checkmlx_$GRUN.txt
if grep -q MLX_CHECK_OK /root/checkmlx_$GRUN.txt; then
  cp $PCK $MLX/pooler.safetensors
  for try in 1 2 3; do hf upload $R $MLX pooler_distill/chatsft/${GSRC}_mlx4$GTAG >/dev/null 2>&1 && break; sleep 30; done
  hf upload $R /root/gptq_$GRUN.log pooler_distill/chatsft/logs/gptq_$GRUN.log >/dev/null 2>&1
  echo "GPTQ_UPLOADED $GRUN -> ${GSRC}_mlx4$GTAG $(date -u)"
else
  echo "GPTQ_PACK_FAILED $GRUN - not uploading"
fi
for i in $(seq 0 $((GSHARDS - 1))); do
  [ -s /root/work/${GRUN}_out_$i.jsonl ] && [ "$(wc -l < /root/work/${GRUN}_out_$i.jsonl)" -ge "$GN" ] && continue
  cd /root/work && SP_BASE=$HF SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/pool_eval.py \
    $PCK /root/work/ev_$i.jsonl /root/work/${GRUN}_out_$i.jsonl --n $GN --rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600 --tag "[$GRUN$i]" >> /root/${GRUN}_$i.log 2>&1
  hf upload $R /root/work/${GRUN}_out_$i.jsonl pooler_distill/chatsft/rollouts/${GRUN}_$i.jsonl >/dev/null 2>&1
  echo "[$GRUN] shard $i $(tail -1 /root/${GRUN}_$i.log | cut -c1-120) | $(grep -h -m1 "pooler restored\|WARNING: no pooler" /root/${GRUN}_$i.log)"
done
python3 - "$GRUN" <<'PYS'
import json, sys, glob
rows = [json.loads(l) for f in sorted(glob.glob(f"/root/work/{sys.argv[1]}_out_*.jsonl")) for l in open(f) if l.strip()]
n = len(rows); c = sum(1 for r in rows if r.get("correct")); g = sum(1 for r in rows if r.get("grounded")); s = sum(float(r.get("ns", 0) or 0) for r in rows)
print(f"GPTQ_EVAL_DONE {sys.argv[1]}: correct {100*c/max(n,1):.1f}%  grounded {100*g/max(n,1):.0f}%  searches {s/max(n,1):.1f}  ({n} rollouts)")
PYS
GK2
  chmod +x /root/gptqkeep.sh
  setsid nohup bash -c 'bash /root/gptqkeep.sh 2>&1 | tee -a /root/quant.log' >> /proc/1/fd/1 2>&1 < /dev/null &
  sleep 20; echo "GPTQ_LAUNCH_DONE $GRUN $(date -u)"
  exit 0
fi
if [ "$MODE" = "publish3" ]; then
  # credit is nearly gone: ship what the s2 measurement has produced so far, without stopping it
  export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
  R=baya1116/hypernet-sp-distill
  for i in 0 1 2; do [ -s /root/work/s2_out_$i.jsonl ] && hf upload $R /root/work/s2_out_$i.jsonl pooler_distill/chatsft/rollouts/s2_$i.jsonl 2>&1 | tail -1; done
  for i in 0 1 2; do [ -s /root/s2_$i.log ] && hf upload $R /root/s2_$i.log pooler_distill/chatsft/logs/s2_$i.log >/dev/null 2>&1; done
  [ -s /root/sft2_run_s2.log ] && hf upload $R /root/sft2_run_s2.log pooler_distill/chatsft/logs/sft2_run_s2.log >/dev/null 2>&1
  echo "PUBLISH3_DONE $(cat /root/work/s2_out_*.jsonl 2>/dev/null | wc -l) rollouts $(date -u)"
  exit 0
fi
if [ "$MODE" = "online" ]; then
  # the on-the-fly alternating loop: rollouts -> accepted ones trained at once -> a Dolphin step -> repeat
  pkill -f "afterkee[p].sh"; pkill -f "evalkee[p].sh"; pkill -f "genkee[p].sh"; pkill -f "sft2kee[p].sh"; pkill -f "reevalkee[p].sh"; pkill -f "onlinekee[p].sh"; sleep 2
  export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
  [ -s /root/.dsk ] || { [ -n "$DSK_KEY" ] && printf '%s' "$DSK_KEY" > /root/.dsk && chmod 600 /root/.dsk; }
  [ -s /root/.oai ] || { [ -n "$OAI_KEY" ] && printf '%s' "$OAI_KEY" > /root/.oai && chmod 600 /root/.oai; }
  R=baya1116/hypernet-sp-distill
  OHF=/root/hfdl/pooler_distill/chatsft/$OMODEL
  for try in 1 2 3 4 5 6; do hf download $R --include "pooler_distill/chatsft/${OMODEL}/*" --local-dir /root/hfdl >/dev/null 2>&1; [ -s $OHF/model.safetensors ] && break; sleep 30; done
  if [ ! -s $OHF/model.safetensors ] && [ -n "${OMERGEFROM:-}" ]; then
    # OMERGEFROM=<run_stepN>: fold that frozen LoRA run into its base and publish the result as the new base $OMODEL
    MB=/root/hfdl/pooler_distill/chatsft/${OMERGEBASE:-s4_hf}
    [ -s $MB/model.safetensors ] || hf download $R --include "pooler_distill/chatsft/${OMERGEBASE:-s4_hf}/*" --local-dir /root/hfdl >/dev/null 2>&1
    for try in 1 2 3 4 5 6; do hf download $R --include "pooler_distill/chatsft/online/$OMERGEFROM/latest.safetensors" --local-dir /root/hfdl >/dev/null 2>&1; [ -s /root/hfdl/pooler_distill/chatsft/online/$OMERGEFROM/latest.safetensors ] && break; sleep 20; done
    cd /root/work && SP_BASE=$MB SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 python3 /root/work/build_merged.py /root/hfdl/pooler_distill/chatsft/online/$OMERGEFROM/latest.safetensors $OHF $OHF/pooler.safetensors 16 ${OMERGELAYERS:-20-27} 2>&1 | grep -E "^\[merge\]|MERGE_DONE|Error|assert" | tail -3
    if [ -s $OHF/model.safetensors ]; then
      ( export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); hf upload $R $OHF pooler_distill/chatsft/$OMODEL >/dev/null 2>&1 && echo "ONLINE_BASE_PUBLISHED $OMODEL (from $OMERGEFROM on ${OMERGEBASE:-s4_hf}) $(date -u)" ) &
    else echo "ONLINE_ABORT: merge of $OMERGEFROM into $OMODEL failed"; exit 0; fi
    rm -f /root/hfdl/pooler_distill/chatsft/online/$OMERGEFROM/latest.safetensors
  fi
  [ -s $OHF/model.safetensors ] || { echo "ONLINE_ABORT: $OMODEL not on the hub"; exit 0; }
  for f in pooler_distill/chatsft/${ODOLPHIN:-dolphin_v1.jsonl} pooler_distill/chatsft/${OREASON:-dolphin_v2.jsonl} pooler_distill/chatsft/${OREPLAY:-replay_v1.jsonl} pooler_distill/selfq_0.jsonl pooler_distill/selfq_1.jsonl pooler_distill/selfq_2.jsonl; do
    [ -s /root/hfdl/$f ] || for try in 1 2 3; do hf download $R --include "$f" --local-dir /root/hfdl >/dev/null 2>&1; [ -s /root/hfdl/$f ] && break; sleep 10; done
  done
  if [ -n "${OSEARCHDEMO:-}" ]; then
    [ -s /root/hfdl/$OSEARCHDEMO ] || for try in 1 2 3; do hf download $R --include "$OSEARCHDEMO" --local-dir /root/hfdl >/dev/null 2>&1; [ -s /root/hfdl/$OSEARCHDEMO ] && break; sleep 10; done
  fi
  if [ -n "${OQFILE:-}" ]; then
    [ -s /root/hfdl/$OQFILE ] || for try in 1 2 3; do hf download $R --include "$OQFILE" --local-dir /root/hfdl >/dev/null 2>&1; [ -s /root/hfdl/$OQFILE ] && break; sleep 10; done
    cp /root/hfdl/$OQFILE /root/work/selfq_all.jsonl
  else cat /root/hfdl/pooler_distill/selfq_?.jsonl > /root/work/selfq_all.jsonl; fi; cp /root/hfdl/pooler_distill/chatsft/${ODOLPHIN:-dolphin_v1.jsonl} /root/work/dolphin_v1.jsonl
  # OREASONSRC=gsm8k: reasoning problems with a checkable number, from the public grade-school-math repo; the reward is
  # exact agreement (--reason-verify numeric), so no teacher model is in the loop and length cannot be talked into a pass
  if [ "${OREASONSRC:-}" = "gsm8k" ]; then
    for f in train test; do [ -s /root/work/gsm8k_$f.raw ] || curl -sSL --retry 3 -o /root/work/gsm8k_$f.raw "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/$f.jsonl"; done
    python3 - <<'PG'
import json
for f in ("train", "test"):
    n = 0
    with open(f"/root/work/gsm8k_{f}.jsonl", "w") as o:
        for l in open(f"/root/work/gsm8k_{f}.raw"):
            try: r = json.loads(l)
            except Exception: continue
            gold = r["answer"].split("####")[-1].strip()
            o.write(json.dumps({"q": r["question"].strip(), "reply": gold, "thinking": "", "gold": gold}, ensure_ascii=False) + "\n"); n += 1
    print(f"[gsm8k] {f}: {n} problems")
PG
    REASONF=/root/work/gsm8k_train.jsonl
  elif [ "${OREASONSRC:-}" = "dolphin_rft" ]; then
    # dolphin_v1 minus a fixed held-out hundred (seed 0). The held-out is the reasoning yardstick from here on: never trained on.
    for f in dolphin_v1.jsonl dolphin_v2.jsonl; do [ -s /root/hfdl/pooler_distill/chatsft/$f ] || hf download $R --include "pooler_distill/chatsft/$f" --local-dir /root/hfdl >/dev/null 2>&1; done
    python3 - <<'PD'
import json, random
v1 = [json.loads(l) for l in open("/root/hfdl/pooler_distill/chatsft/dolphin_v1.jsonl") if l.strip()]
v1 = [r for r in v1 if r.get("q") and r.get("reply")]
v2q = set((json.loads(l).get("q") or "").strip() for l in open("/root/hfdl/pooler_distill/chatsft/dolphin_v2.jsonl") if l.strip())
cand = [r for r in v1 if r["q"].strip() not in v2q]; random.Random(0).shuffle(cand)
held = cand[:100]; hq = set(r["q"].strip() for r in held)
with open("/root/work/dolphin_heldout100.jsonl", "w") as o:
    for r in held: o.write(json.dumps({"q": r["q"], "ref": r["reply"]}, ensure_ascii=False) + "\n")
with open("/root/work/dolphin_rft.jsonl", "w") as o:
    n = 0
    for r in v1:
        if r["q"].strip() in hq: continue
        o.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
print(f"[dolphin_rft] {n} training problems, 100 held out")
PD
    REASONF=/root/work/dolphin_rft.jsonl
  else REASONF=/root/hfdl/pooler_distill/chatsft/$OREASON; fi
  QFILE=/root/work/selfq_all.jsonl
  if [ "${OPOOL3Q:-0}" = 1 ]; then
    # the corpus pool lives on the hub as box_recover/corpus.jsonl (boxC fetched it the same way for the search GRPO)
    [ -s /root/work/corpus_box_final.jsonl ] || curl -sSL --retry 3 -o /root/work/corpus_box_final.jsonl "https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/box_recover/corpus.jsonl"
    python3 - <<'PQ'
import json
n=0
with open('/root/work/pool3q.jsonl','w') as o:
    for line in open('/root/work/corpus_box_final.jsonl'):
        try: r=json.loads(line)
        except Exception: continue
        q,g=(r.get('q') or '').strip(),(r.get('gold') or '').strip()
        if q and g and len(g.split())<=6: o.write(json.dumps({'q':q,'gold':g},ensure_ascii=False)+'\n'); n+=1
print('[pool3q]',n,'questions')
PQ
    QFILE=/root/work/pool3q.jsonl
  fi
  # r5: drop replay traces that carry the stray thought markers or a repetition loop (41 of 672 did; replaying them raised the marker rate from 20% to 35% of rollouts in r4)
  REPLAYF=/root/hfdl/pooler_distill/chatsft/${OREPLAY:-replay_v1.jsonl}
  if [ "${OREPLAYFILTER:-0}" = "1" ]; then
    python3 - "$REPLAYF" /root/work/replay_clean.jsonl <<'PYF'
import json, re, sys
degen = re.compile(r"begin_of_thought|end_of_thought|\b(\w+(?:\W+\w+){0,3})\b(?:\W+\1\b){4,}")
keep = [l for l in open(sys.argv[1]) if l.strip() and not degen.search(l)]
open(sys.argv[2], "w").writelines(keep); print(f"[replay filter] kept {len(keep)}")
PYF
    REPLAYF=/root/work/replay_clean.jsonl
  fi
  OUT=/root/online_$ORUN; mkdir -p $OUT
  cat > /root/freeze.sh <<'FZ'
#!/bin/bash
# a checkpoint every two hundred steps: the run is five days long and one overwritten file is no record
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
R=baya1116/hypernet-sp-distill; ORUN=$1; OUT=/root/online_$ORUN
while :; do
  st=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null || echo 0)
  mark=$(( (st / 200) * 200 ))
  snap=$OUT/step$mark.safetensors   # the loop writes this at the exact step; before it did, the mark got whatever "latest" was (g7_step400 held step 470)
  if [ "$mark" -gt 0 ] && [ ! -f /root/.frozen_${ORUN}_$mark ] && { [ -s $snap ] || [ -s $OUT/latest.safetensors ]; }; then
    touch /root/.frozen_${ORUN}_$mark
    if [ -s $snap ]; then
      hf upload $R $snap pooler_distill/chatsft/online/${ORUN}_step$mark/latest.safetensors >/dev/null 2>&1 && rm -f $snap
      [ -s $OUT/step$mark.json ] && hf upload $R $OUT/step$mark.json pooler_distill/chatsft/online/${ORUN}_step$mark/state.json >/dev/null 2>&1 && rm -f $OUT/step$mark.json
      for f in loop.log rollouts.jsonl; do [ -s $OUT/$f ] && hf upload $R $OUT/$f pooler_distill/chatsft/online/${ORUN}_step$mark/$f >/dev/null 2>&1; done
      echo "ONLINE_FROZEN ${ORUN}_step$mark (exact) $(date -u)"
    else
      for f in latest.safetensors state.json loop.log rollouts.jsonl; do
        [ -s $OUT/$f ] && hf upload $R $OUT/$f pooler_distill/chatsft/online/${ORUN}_step$mark/$f >/dev/null 2>&1
      done
      echo "ONLINE_FROZEN ${ORUN}_step$mark (latest, step $st) $(date -u)"
    fi
  fi
  # a restart marker kills the loop for a few seconds; this watcher used to quit on that and stay dead until the next control run
  if pgrep -f "online_loop.p[y]" >/dev/null; then miss=0; else miss=$((${miss:-0} + 1)); [ "$miss" -ge 6 ] && break; fi
  sleep 600
done
FZ
  chmod +x /root/freeze.sh
  # the watcher is per run: the one started for the previous run kept running, never saw this run's marks, and blocked a new one
  pgrep -f "freeze.s[h]" >/dev/null && ! pgrep -f "freeze.sh $ORUN\b" >/dev/null && { pkill -f "freeze.s[h]"; sleep 2; }
  pgrep -f "freeze.sh $ORUN\b" >/dev/null || setsid nohup bash /root/freeze.sh $ORUN >> /proc/1/fd/1 2>&1 < /dev/null &
  # (the one-off step-150 upload of the g3 days is gone: it labelled whatever "latest" was as step 150 - g10_step150 on the hub is really step ~285)
  # every checkpoint is ~3.5 GB and every run keeps two (latest and the guard's healthy copy): six finished
  # runs filled the disk and killed r10 at step 5 with "No space left on device" while it wrote one.
  for d in /root/online_*/; do [ "$d" = "$OUT/" ] || rm -f $d/latest.safetensors $d/good.safetensors $d/*.tmp; done
  rm -f /root/reeval_*.safetensors /root/online_*/latest.safetensors.tmp
  # the merged evaluation copies and the hub downloads of frozen checkpoints are re-creatable: 5.4 GB free on 09-22
  # would not have held the step-600 snapshot beside the next save
  rm -rf /root/reeval_hf_* /root/hfdl/pooler_distill/chatsft/online/*/latest.safetensors
  echo "[disk] $(df -h /root | tail -1 | awk '{print $3" used, "$4" free"}')"
  # once (2026-09-20): g5 ended unwatched; its full step record is still on this disk and the hub copy stops at step 227
  if [ ! -f /root/.up_g5_final ] && [ -s /root/online_g5/rollouts.jsonl ]; then ( for f in rollouts.jsonl loop.log; do hf upload $R /root/online_g5/$f pooler_distill/chatsft/online/g5/$f >/dev/null 2>&1; done; touch /root/.up_g5_final; echo "ONLINE_G5_RECORD uploaded $(date -u)" ) & fi
  # resume from the hub copy of this run (uploaded every 30 min) when this box did not start it
  # start from a frozen checkpoint of another run: g5 broke at step ~225 with no guard in place, so g6
  # picks up its step-200 weights and carries on with the guard and half the rate
  if [ -n "${OSEED:-}" ] && [ ! -s $OUT/state.json ]; then
    for f in latest.safetensors state.json; do
      for try in 1 2 3 4 5 6; do hf download $R --include "pooler_distill/chatsft/online/$OSEED/$f" --local-dir /root/hfdl >/dev/null 2>&1; [ -s /root/hfdl/pooler_distill/chatsft/online/$OSEED/$f ] && break; sleep 20; done
      [ -s /root/hfdl/pooler_distill/chatsft/online/$OSEED/$f ] && cp /root/hfdl/pooler_distill/chatsft/online/$OSEED/$f $OUT/$f
    done
    [ -s $OUT/latest.safetensors ] || { echo "ONLINE_SEED_ABORT $ORUN: $OSEED weights did not arrive; not starting from the bare base under a borrowed step count"; rm -f $OUT/state.json; exit 0; }
    st=$(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null || echo 0)
    touch /root/.frozen_${ORUN}_$st       # the seed itself is already frozen under its own name
    echo "ONLINE_SEED $ORUN from $OSEED (step $st) $(date -u)"
  fi
  if [ "${ORESUME:-0}" = "1" ] && [ ! -s $OUT/state.json ]; then
    for f in latest.safetensors state.json loop.log accepted.jsonl rollouts.jsonl; do
      for try in 1 2 3 4 5 6; do hf download $R --include "pooler_distill/chatsft/online/$ORUN/$f" --local-dir /root/hfdl >/dev/null 2>&1; [ -s /root/hfdl/pooler_distill/chatsft/online/$ORUN/$f ] && break; sleep 20; done
      [ -s /root/hfdl/pooler_distill/chatsft/online/$ORUN/$f ] && cp /root/hfdl/pooler_distill/chatsft/online/$ORUN/$f $OUT/$f
    done
    echo "ONLINE_RESUME $ORUN from step $(python3 -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null) $(date -u)"
  fi
  # one-time: the first run used 8 rollouts per step and 6 GB of a 24 GB card; restart with 16 (the loop resumes from its state file)
  # 16 rollouts per step took 4.5 min against 1.4 min for 8 (per rollout slower, not faster): back to 8, once
  # r4: retention by replaying fixed verified search traces (not by training on the newest own samples, which sharpened into repetition twice); rollouts every 10 steps only to measure the skip rates and top up the replay set
  # once: the measurement questions move from nq_open to the GRPO pool (the loop resumes from its local state)
  if [ ! -f /root/.restart_${ORUN}_q ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_q; echo "ONLINE_RESTART $ORUN questions=$OQFILE $(date -u)"; fi
  # once: the searches trip needs an absolute floor as well (the capped mean sits near 3, and a
  # window holding a couple of capped rollouts would otherwise read as a runaway)
  # once: the guard also watches the share of rollouts cut for searching without end, which is the
  # signal that moves first (5%% at the start of r11, 11%% by step 360 while the mean stayed near 3)
  # once: the judge decides nothing now that every finished rollout is trained on, so stop calling it
  # thinking has grown to about 850 tokens on the search side, which is what chain of thought is for;
  # what it collided with was the budget, so the budget moves rather than the thinking being penalised
  # one row that keeps reading holds the whole batch: the wall-clock ceiling comes down so a step
  # costs about twenty-five minutes at worst, which still leaves a long rollout room to finish
  # past the fifth search the environment returns a notice and no content, so a rollout that issues
  # 181 of them is reading nothing: it is cut at seven, two past the last one that can serve a page
  if [ ! -f /root/.restart_${ORUN}_srch7 ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_srch7; echo "ONLINE_RESTART $ORUN maxsrch 7 $(date -u)"; fi
  if [ ! -f /root/.restart_${ORUN}_budget ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_budget; echo "ONLINE_RESTART $ORUN budget 1500 $(date -u)"; fi
  if [ ! -f /root/.restart_${ORUN}_gen7k ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_gen7k; echo "ONLINE_RESTART $ORUN gen 7000 $(date -u)"; fi
  if [ ! -f /root/.restart_${ORUN}_nojudge ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_nojudge; echo "ONLINE_RESTART $ORUN judge off $(date -u)"; fi
  if [ ! -f /root/.restart_${ORUN}_cut ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_cut; echo "ONLINE_RESTART $ORUN guard cut share $(date -u)"; fi
  if [ ! -f /root/.restart_${ORUN}_nsfloor ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_nsfloor; echo "ONLINE_RESTART $ORUN guard floor $(date -u)"; fi
  # once: a search group that scores zero now learns from that question's own teacher trajectory
  if [ ! -f /root/.restart_${ORUN}_demo ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_demo; echo "ONLINE_RESTART $ORUN search demos $(date -u)"; fi
  # once (2026-09-20): the wheel trigger tested mean == 0, which an all-wrong search group (-0.5 each) never meets; it now tests max <= 0
  if [ ! -f /root/.restart_${ORUN}_wheelfix ] && grep -q "max(rw) <= 0.0" /root/work/online_loop.py; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_wheelfix; echo "ONLINE_RESTART $ORUN wheel trigger $(date -u)"; fi
  # once (2026-09-20): the search side back to the search GRPO's conditions: its own optimizer at 1e-5, temp 0.9, gen 1500, corpus questions, mean/std normalisation
  if [ ! -f /root/.restart_${ORUN}_pool3 ] && grep -q "search-lr" /root/work/online_loop.py; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_pool3; echo "ONLINE_RESTART $ORUN pool3 conditions $(date -u)"; fi
  # once (2026-09-25): the judge is nano again
  if [ ! -f /root/.restart_${ORUN}_nano ] && [ -f /root/.restart_${ORUN}_dsk ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_nano; echo "ONLINE_RESTART $ORUN judge nano $(date -u)"; fi
  # once (2026-09-25): the judge is DeepSeek now
  if [ ! -f /root/.restart_${ORUN}_dsk ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_dsk; echo "ONLINE_RESTART $ORUN judge deepseek $(date -u)"; fi
  # once (2026-09-25): g12 with the 3000-token cap and the judge-failure skip
  if [ ! -f /root/.restart_${ORUN}_cap3k ] && grep -q "judge calls failed" /root/work/online_loop.py; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_cap3k; echo "ONLINE_RESTART $ORUN cap 3000 $(date -u)"; fi
  # once (2026-09-24): KL 0.2 and the Dolphin record off; the rollback count is forgotten
  if [ ! -f /root/.restart_${ORUN}_kl02 ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_kl02
    python3 - "$OUT/state.json" <<'GR5'
import json,sys
p=sys.argv[1]
try: st=json.load(open(p))
except Exception: sys.exit(0)
for k in ('rollbacks','guard_from'): st.pop(k, None)
json.dump(st, open(p,'w'))
GR5
    echo "ONLINE_RESTART $ORUN kl 0.2, dolphin off $(date -u)"; fi
  # once (2026-09-23): the KL run starts from the guard's last healthy copy, not from the drifted step-205 save the rollback had already discarded
  if [ ! -f /root/.restart_${ORUN}_fromgood ] && [ -s $OUT/good.safetensors ] && grep -q "disable_adapter" /root/work/online_loop.py; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_fromgood; cp $OUT/good.safetensors $OUT/latest.safetensors; echo "ONLINE_RESTART $ORUN from good.safetensors $(date -u)"; fi
  # once (2026-09-23): the KL anchor; the one rollback so far is forgotten, the baseline kept
  if [ ! -f /root/.restart_${ORUN}_kl ] && grep -q "disable_adapter" /root/work/online_loop.py; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_kl
    python3 - "$OUT/state.json" <<'GR4'
import json,sys
p=sys.argv[1]
try: st=json.load(open(p))
except Exception: sys.exit(0)
for k in ('rollbacks','guard_from'): st.pop(k, None)
json.dump(st, open(p,'w'))
GR4
    echo "ONLINE_RESTART $ORUN kl anchor $(date -u)"; fi
  # once (2026-09-22): pooler adapter on; the optimizer file changes shape, so it is dropped (one fresh-moments resume)
  if [ ! -f /root/.restart_${ORUN}_pooler ] && grep -q "ride with the search optimizer" /root/work/online_loop.py; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_pooler; rm -f $OUT/opt.pt; echo "ONLINE_RESTART $ORUN pooler adapter $(date -u)"; fi
  # once (2026-09-21): guard window 20 steps, no rate halving, reasoning rate 2e-5; the baseline and the two rollbacks are forgotten
  if [ ! -f /root/.restart_${ORUN}_guard20 ] && grep -q "guard-steps" /root/work/online_loop.py; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_guard20
    python3 - "$OUT/state.json" <<'GR3'
import json,sys
p=sys.argv[1]
try: st=json.load(open(p))
except Exception: sys.exit(0)
for k in ('gbase','rollbacks','guard_from'): st.pop(k, None)
json.dump(st, open(p,'w'))
print('[guard] baseline and rollbacks reset at step', st.get('step'))
GR3
    echo "ONLINE_RESTART $ORUN guard 20 steps, lr 2e-5 $(date -u)"; fi
  # once (2026-09-21): the step-200 check's full replies, for grading
  if [ ! -f /root/.up_g7d200 ] && [ -s /root/work/g7d200_out_0.jsonl ]; then hf upload $R /root/work/g7d200_out_0.jsonl pooler_distill/chatsft/rollouts/g7d200_dolphin.jsonl >/dev/null 2>&1 && touch /root/.up_g7d200 && echo "ONLINE_UPLOADED g7d200 replies"; fi
  # once (2026-09-21): search questions continue pool3's sequence
  if [ ! -f /root/.restart_${ORUN}_poolorder ] && grep -q "pool-order" /root/work/online_loop.py; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_poolorder; echo "ONLINE_RESTART $ORUN pool3 order $(date -u)"; fi
  # once (2026-09-20): the guard also needs the pass rate to fall (g7 tripped at step 40 on one hard question); the false rollback is forgotten
  if [ ! -f /root/.restart_${ORUN}_guardpass ] && grep -q "guard-pass" /root/work/online_loop.py; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_guardpass
    python3 - "$OUT/state.json" <<'GR2'
import json,sys
p=sys.argv[1]
try: st=json.load(open(p))
except Exception: sys.exit(0)
for k in ('rollbacks','guard_from'): st.pop(k, None)
json.dump(st, open(p,'w'))
print('[guard] rollback count reset at step', st.get('step'))
GR2
    echo "ONLINE_RESTART $ORUN guard pass-rate $(date -u)"; fi
  # once (2026-09-20): search temperature back to 0.6, and the guard baseline it learned while the model was not searching is dropped
  if [ ! -f /root/.restart_${ORUN}_temp06 ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_temp06
    python3 - "$OUT/state.json" <<'GR'
import json,sys
p=sys.argv[1]
try: st=json.load(open(p))
except Exception: sys.exit(0)
for k in ('gbase','rollbacks','guard_from'): st.pop(k, None)
json.dump(st, open(p,'w'))
print('[guard] baseline reset at step', st.get('step'))
GR
    rm -f $OUT/good.safetensors; echo "ONLINE_RESTART $ORUN search temp 0.6 $(date -u)"; fi
  # once (2026-09-20): the first pool3 launch had no corpus on this box and started with 0 questions; relaunch once the pool exists
  if [ ! -f /root/.restart_${ORUN}_pool3q ] && [ -s /root/work/pool3q.jsonl ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_pool3q; echo "ONLINE_RESTART $ORUN pool3 questions $(wc -l < /root/work/pool3q.jsonl) $(date -u)"; fi
  # once (2026-09-20): search-side wheels off again
  if [ ! -f /root/.restart_${ORUN}_swoff ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_swoff; echo "ONLINE_RESTART $ORUN search wheels off $(date -u)"; fi
  # once (2026-09-20): LoRA back on layers 20-27 as in the search GRPO that produced this model
  if [ ! -f /root/.restart_${ORUN}_layers ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_layers; echo "ONLINE_RESTART $ORUN lora layers 20-27 $(date -u)"; fi
  # once (2026-09-20): constant-length loss normalisation and no std scaling (the g5 collapse cause, see docs).
  # The first launch died on the flags because the raw fetch of online_loop.py was still the old copy; this re-run refetches.
  if [ ! -f /root/.restart_${ORUN}_drgrpo ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_drgrpo; echo "ONLINE_RESTART $ORUN dr-grpo norm $(date -u)"; fi
  # once: search-side wheels back on (replay traces), after g5 collapsed on the side that had none
  if [ ! -f /root/.restart_${ORUN}_swheel ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_${ORUN}_swheel; echo "ONLINE_RESTART $ORUN search wheels $(date -u)"; fi
  if [ ! -f /root/.restart_$ORUN ]; then pkill -f "online_loop.p[y]"; sleep 8; pkill -9 -f "online_loop.p[y]" 2>/dev/null; touch /root/.restart_$ORUN; echo "ONLINE_RESTART $ORUN dolphin ratio $ODRATIO min $ODMIN $(date -u)"; fi
  if ! pgrep -f "online_loop.p[y]" >/dev/null; then
    cd /root/work && DSK_KEY=$(cat /root/.dsk 2>/dev/null) OAI_KEY=$(cat /root/.oai 2>/dev/null) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/online_loop.py $OHF $OUT \
      --questions $QFILE --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl \
      --b $OB --steps $OSTEPS --temp $OTEMP --lr $OLR --dolphin-ratio ${ODRATIO:-2} --dolphin-min ${ODMIN:-2} --accum ${OACCUM:-1} --replay $REPLAYF --replay-per-step ${OREPLAYN:-2} --rollout-every ${OROLLEVERY:-1} --train-all ${OTRAINALL:-0} --queue ${OQUEUE:-0} --complete-only ${OCOMPLETE:-0} --gen ${OGEN:-1500} --budget ${OBUDGET:-900} --guard ${OGUARD:-0} --maxsrch ${OMAXSRCH:-0} --judge ${OJUDGE:-1} ${OREASON:+--reason $REASONF --reason-verify ${OREASONVERIFY:-judge} --rft ${ORFT:-0} --rft-replay ${ORFTREPLAY:-0.5} --wheel-max-think ${OWHEELMAX:-0} --sft-only ${OSFTONLY:-0} --reason-g ${OREASONG:-8} --reason-stub ${OREASONSTUB:-0} --judge-api ${OJUDGEAPI:-deepseek} --judge-model ${OJUDGEMODEL:-} --search-every ${OSEARCHEVERY:-0} --reason-every ${OREASONEVERY:-0} --kl ${OKL:-0} --dolphin-on ${ODOLPHINON:-all} --w-talk ${OWTALK:-0.5} --wheels ${OWHEELS:-0} --search-wheels ${OSEARCHWHEELS:-0} --pg-norm ${OPGNORM:-mean} --pg-norm-len 1024 --adv-std ${OADVSTD:-1} --search-lr ${OSEARCHLR:-0} --guard-pass ${OGUARDPASS:-0.5} --guard-steps ${OGUARDSTEPS:-10} --guard-halve ${OGUARDHALVE:-1} --pool-order ${OPOOLORDER:-loop} --pool-offset ${OPOOLOFFSET:-0} --search-temp ${OSEARCHTEMP:-0} --search-gen ${OSEARCHGEN:-0} ${OSEARCHDEMO:+--search-demo /root/hfdl/$OSEARCHDEMO}} --pooler ${OPOOLER:-none} --pooler-rank ${OPOOLERRANK:-8} --pooler-lr ${OPOOLERLR:-1e-5} $([ -s $OHF/pooler.safetensors ] && echo --pooler-init $OHF/pooler.safetensors) --lora-rank 16 --lora-layers ${OLAYERS:-all} --gradckpt 1 --save-every 5 --stop eos \
      >> /root/online_$ORUN.log 2>&1 < /dev/null &
  fi
  # once: in reasoning mode the empty search loop printed the end marker at launch, and the keeper
  # read it as the run being over and stopped uploading. Strip it so the keeper runs to the real end.
  if [ ! -f /root/.fixed_marker_$ORUN ]; then sed -i '/^ONLINE_LOOP_DONE$/d' /root/online_$ORUN.log 2>/dev/null; touch /root/.fixed_marker_$ORUN; echo "ONLINE_MARKER_FIX $ORUN $(date -u)"; fi
  # only what this launch wrote counts as its end: a previous run's marker sits in the same log and
  # twice now has made the keeper quit at the first upload, leaving the run saving nothing
  MARK=$(( $(pgrep -f "online_loop.p[y]" >/dev/null && cat /root/.logmark_$ORUN 2>/dev/null || wc -c < /root/online_$ORUN.log 2>/dev/null || echo 0) + 1 ))
  [ -f /root/.logmark_$ORUN ] || echo $((MARK - 1)) > /root/.logmark_$ORUN
  cat > /root/onlinekeep.sh <<OK
#!/bin/bash
ORUN=$ORUN; OUT=$OUT; R=$R; MARK=$MARK
OK
  cat >> /root/onlinekeep.sh <<'OKB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
last=0
while :; do
  now=$(date +%s)
  if [ $((now - last)) -ge 1800 ] || tail -c +$MARK /root/online_$ORUN.log 2>/dev/null | grep -q "ONLINE_LOOP_DONE"; then
    [ -s $OUT/latest.safetensors ] && hf upload $R $OUT/latest.safetensors pooler_distill/chatsft/online/$ORUN/latest.safetensors >/dev/null 2>&1
    for f in loop.log accepted.jsonl rollouts.jsonl state.json; do [ -s $OUT/$f ] && hf upload $R $OUT/$f pooler_distill/chatsft/online/$ORUN/$f >/dev/null 2>&1; done
    hf upload $R /root/online_$ORUN.log pooler_distill/chatsft/online/$ORUN/run.log >/dev/null 2>&1
    last=$now; echo "[online $ORUN $(date -u +%H:%M)] uploaded | $(tail -1 $OUT/loop.log 2>/dev/null | cut -c1-200)"
    /usr/local/bin/s 20 2>/dev/null | sed 's/^/SCORE /' 
    grep -hE "^ONLINE_ROLLBACK|^ONLINE_COLLAPSE|^\[guard\]" /root/online_$ORUN.log 2>/dev/null | tail -3
    tail -c +$MARK /root/online_$ORUN.log 2>/dev/null | grep -q "ONLINE_LOOP_DONE" && { echo "ONLINE_DONE $ORUN $(date -u)"; break; }
    pgrep -f "online_loop.p[y]" >/dev/null || { echo "ONLINE_DIED $ORUN: $(grep -E 'Error|error|Traceback' /root/online_$ORUN.log | tail -2 | cut -c1-160)"; break; }
  fi
  sleep 60
done
OKB
  chmod +x /root/onlinekeep.sh; setsid nohup bash /root/onlinekeep.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  sleep 90; echo "loop script: $(wc -l < /root/work/online_loop.py) lines, wheelfix=$(grep -c "max(rw) <= 0.0" /root/work/online_loop.py)"; grep -E "^\[init\]|^\[cfg\]|ONLINE_ABORT|pool3q" /root/online_$ORUN.log | tail -5 | cut -c1-400; tail -3 /root/online_$ORUN.log | cut -c1-200; echo "ONLINE_LAUNCH_DONE $ORUN $(date -u)"; exit 0
fi

if [ "$MODE" = "reeval" ]; then
  # two control runs fired within a minute of each other on 09-21 and merged the same checkpoint side by side: CUDA OOM for both
  exec 9>/root/.reeval.lock; flock -n 9 || { echo "REEVAL_SKIP: another control run is already doing this"; exit 0; }
  # a push that changes this file re-runs it; the evaluation already under way for the same RRUN is left alone
  if pgrep -f "reevalkee[p].sh" >/dev/null && grep -q "^RRUN=${RRUN:-basefix};.*; RB=${RB:-12};.*; RN=${RN:-999};" /root/reevalkeep.sh 2>/dev/null; then echo "REEVAL_SKIP: $RRUN is already running"; exit 0; fi
  pkill -f "onlinekee[p].sh"; pkill -f "online_loop.p[y]"; sleep 5; pkill -9 -f "online_loop.p[y]" 2>/dev/null
  pkill -f "build_merged.p[y]"; sleep 3; pkill -9 -f "build_merged.p[y]" 2>/dev/null   # a merge that died half-way may still hold the card
  # Re-measure an earlier checkpoint under the fixed conversational stop rule (EOS terminal), so the
  # table compares like with like: RMODEL is "base" (the step-200 student) or an adapter name on the hub.
  RRUN=${RRUN:-basefix}; RMODEL=${RMODEL:-base}   # re-run for the no-search set
  pkill -f "afterkee[p].sh"; pkill -f "evalkee[p].sh"; pkill -f "genkee[p].sh"; pkill -f "sft2kee[p].sh"; pkill -f "reevalkee[p].sh"; sleep 3
  pkill -f "pool_eval.p[y]"; sleep 8; pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 2   # a re-run must not leave the old evaluator holding the card
  export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
  R=baya1116/hypernet-sp-distill
  # the disk hit 97% after the corpus build (its float vectors, 4.9 GB, are not shipped); everything removed here is on the hub or rebuilt
  if [ "$(df -BG /root | awk 'NR==2{print $4}' | tr -d G)" -lt 8 ]; then
    rm -rf /root/wiki_store/emb_f16.bin /root/wiki_store/pq_codes.npy /root/wikidl /root/base_distill /root/reeval_hf_g14_step400m /root/evalrun_* /root/hfdl/pooler_distill/chatsft/online/*/latest.safetensors /root/.cache/pip /root/.cache/huggingface/hub/datasets--wikimedia--wikipedia /root/hfdl/pooler_distill/chatsft/s4_hf /root/reeval_hf_g10m
    for d in /root/sft_hf_q* /root/sft_mlx4_q*; do [ -d "$d" ] && grep -q "QUANT_UPLOADED $(basename $d | sed 's/sft_hf_//; s/sft_mlx4_//') " /root/quant.log 2>/dev/null && rm -rf "$d"; done
    echo "[disk] cleaned: $(df -h /root | awk 'NR==2{print $4" free"}')"
  fi
  RCKPT=${RCKPT:-/root/pooler200.safetensors}
  if [ "${RKIND:-adapter}" = "ckpt" ]; then
    # RMODEL names a run under chatsft/online: its latest.safetensors is a full state dict, so the
    # structure comes from the step-200 directory and every weight is overwritten at load time.
    RHF=/root/eval_hf200; RCKPT=/root/reeval_$RRUN.safetensors
    if [ -s /root/online_$RMODEL/latest.safetensors ]; then cp /root/online_$RMODEL/latest.safetensors $RCKPT
    else
      for try in 1 2 3 4 5 6; do hf download $R --include "pooler_distill/chatsft/online/$RMODEL/latest.safetensors" --local-dir /root/hfdl >/dev/null 2>&1; [ -s /root/hfdl/pooler_distill/chatsft/online/$RMODEL/latest.safetensors ] && break; sleep 20; done
      cp /root/hfdl/pooler_distill/chatsft/online/$RMODEL/latest.safetensors $RCKPT 2>/dev/null
    fi
    [ -s $RCKPT ] || { echo "REEVAL_ABORT $RRUN: online/$RMODEL checkpoint not found"; exit 0; }
  elif [ "${RKIND:-adapter}" = "merge" ]; then
    # RMODEL names a run under chatsft/online whose LoRA sits on a subset of layers: fold it into its own base first,
    # never overlay it on the step-200 structure (that is how g6 ended up with layers 0-19 of the stock model)
    RHF=/root/reeval_hf_${RMODEL}m; MB=/root/hfdl/pooler_distill/chatsft/${RBASE:-s4_hf}
    [ -s $MB/model.safetensors ] || hf download $R --include "pooler_distill/chatsft/${RBASE:-s4_hf}/*" --local-dir /root/hfdl >/dev/null 2>&1
    SRC=/root/online_$RMODEL/latest.safetensors
    if [ ! -s $SRC ]; then
      rm -f /root/hfdl/pooler_distill/chatsft/online/$RMODEL/latest.safetensors
      for try in 1 2 3 4 5 6; do hf download $R --include "pooler_distill/chatsft/online/$RMODEL/latest.safetensors" --local-dir /root/hfdl >/dev/null 2>&1; [ -s /root/hfdl/pooler_distill/chatsft/online/$RMODEL/latest.safetensors ] && break; sleep 20; done
      SRC=/root/hfdl/pooler_distill/chatsft/online/$RMODEL/latest.safetensors
    fi
    [ -s $SRC ] || { echo "REEVAL_ABORT $RRUN: online/$RMODEL checkpoint not found"; exit 0; }
    RCKPT=/root/reeval_${RMODEL}m_pooler.safetensors; rm -rf $RHF
    echo "[merge] $SRC (step $(python3 -c "import json;print(json.load(open('/root/online_$RMODEL/state.json'))['step'])" 2>/dev/null || echo ?)) onto ${RBASE:-s4_hf}, lora r16 layers ${RLAYERS:-20-27}"
    cd /root/work && SP_BASE=$MB SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 python3 /root/work/build_merged.py $SRC $RHF $RCKPT 16 ${RLAYERS:-20-27} 2>&1 | grep -E "^\[merge\]|MERGE_DONE|Error|assert|unexpected" | tail -4
    [ -s $RHF/model.safetensors ] || { echo "REEVAL_ABORT $RRUN: merge of online/$RMODEL failed"; exit 0; }
  elif [ "${RKIND:-adapter}" = "sftdir" ]; then
    # a quant arm's dequantized directory on the box (RMODEL = its QRUN), with its pooler: the way a screened arm gets more shards
    RHF=/root/sft_hf_$RMODEL; RCKPT=/root/reeval_g14m_pooler.safetensors
    [ -s /root/pooler_eval_$RMODEL.safetensors ] && RCKPT=/root/pooler_eval_$RMODEL.safetensors
    # the directory is removed to make room for later arms; its state file rebuilds it - with the run's codes when it did not round the base
    [ "$RMODEL" = q14g ] && [ ! -f /root/sft/q14g.codes ] && echo /root/gptq_state_gq14.pt > /root/sft/q14g.codes
    RCODES=$(cat /root/sft/$RMODEL.codes 2>/dev/null)
    if [ -n "$RCODES" ] && [ -s $RHF/model.safetensors ] && [ ! -f $RHF/.codes_ok ]; then echo "[reeval] $RHF was rebuilt without its codes - rebuilt again"; rm -rf $RHF; fi
    [ -s $RHF/model.safetensors ] || { cd /root/work && python3 /root/work/dequant_state.py --base /root/reeval_hf_g14m --state /root/sft/$RMODEL.pt --out $RHF ${RCODES:+--codes-from $RCODES} 2>&1 | tail -2; [ -n "$RCODES" ] && [ -s $RHF/model.safetensors ] && touch $RHF/.codes_ok; }
    [ -s $RHF/model.safetensors ] || { echo "REEVAL_ABORT $RRUN: $RHF is not on the box and could not be rebuilt"; exit 0; }
  elif [ "${RKIND:-adapter}" = "gptqdir" ]; then
    # a GPTQ run's dequantized directory on the box (RMODEL = its GRUN), with g14's pooler
    RHF=/root/gptq_hf_$RMODEL; RCKPT=/root/reeval_g14m_pooler.safetensors
    [ -s $RHF/model.safetensors ] || { echo "REEVAL_ABORT $RRUN: $RHF is not on the box"; exit 0; }
  elif [ "${RKIND:-adapter}" = "stock" ]; then
    # the untouched R1 distill the whole lineage started from: the reasoning ceiling before any of our training
    RHF=/root/base_distill; RCKPT=/root/pooler200.safetensors
    [ -s $RHF/model.safetensors ] || for try in 1 2 3 4 5 6; do hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir $RHF >/dev/null 2>&1; [ -s $RHF/model.safetensors ] && break; sleep 20; done
    [ -s $RHF/model.safetensors ] || { echo "REEVAL_ABORT $RRUN: stock model download failed"; exit 0; }
  elif [ "$RMODEL" = "base" ]; then RHF=/root/eval_hf200; elif [ "${RKIND:-adapter}" = "dir" ]; then
    RHF=/root/hfdl/pooler_distill/chatsft/$RMODEL   # a merged model directory published by the training box
    for try in $(seq 1 60); do   # it may still be uploading: poll for up to an hour
      hf download $R --include "pooler_distill/chatsft/${RMODEL}/*" --local-dir /root/hfdl >/dev/null 2>&1
      [ -s $RHF/model.safetensors ] && [ -s $RHF/config.json ] && break; sleep 60
    done
    [ -s $RHF/model.safetensors ] || { echo "REEVAL_ABORT $RRUN: $RMODEL not on the hub"; exit 0; }
    [ -s $RHF/pooler.safetensors ] && RCKPT=$RHF/pooler.safetensors   # the directory carries its own trained pooler
  else
    RHF=/root/reeval_hf_$RMODEL
    if [ ! -s $RHF/model.safetensors ]; then
      for try in 1 2 3; do hf download $R --include "pooler_distill/chatsft/${RMODEL}/*" --local-dir /root/hfdl 2>&1 | tail -1; [ "${PIPESTATUS[0]}" -eq 0 ] && break; sleep 10; done
      cd /root/work && python3 - <<PYM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("/root/eval_hf200", torch_dtype=torch.bfloat16)
m = PeftModel.from_pretrained(base, "/root/hfdl/pooler_distill/chatsft/$RMODEL").merge_and_unload()
m.save_pretrained("$RHF", safe_serialization=True); AutoTokenizer.from_pretrained("/root/eval_hf200").save_pretrained("$RHF"); print("MERGED $RMODEL")
PYM
    fi
    [ -s $RHF/model.safetensors ] || { echo "REEVAL_ABORT $RRUN: merge failed"; exit 0; }
  fi
  if [ "${RQSRC:-}" = "dolphin" ]; then
    [ -s /root/hfdl/pooler_distill/chatsft/dolphin_v2.jsonl ] || hf download $R --include "pooler_distill/chatsft/dolphin_v2.jsonl" --local-dir /root/hfdl >/dev/null 2>&1
    python3 - <<PYD
import json
rows = [json.loads(l) for l in open("/root/hfdl/pooler_distill/chatsft/dolphin_v2.jsonl") if l.strip()]
out = open("/root/work/dolphinq.jsonl", "w")
for r in rows[:${RQN:-12}]:
    out.write(json.dumps({"q": r["q"]}, ensure_ascii=False) + "\n")
print("[dolphin questions]", min(len(rows), ${RQN:-12}))
PYD
  fi
  if [ "${RQSRC:-}" = "dolphinh" ]; then
    if [ ! -s /root/work/dolphin_heldout100.jsonl ]; then
      for f in dolphin_v1.jsonl dolphin_v2.jsonl; do [ -s /root/hfdl/pooler_distill/chatsft/$f ] || hf download $R --include "pooler_distill/chatsft/$f" --local-dir /root/hfdl >/dev/null 2>&1; done
      python3 - <<'PD2'
import json, random
v1 = [json.loads(l) for l in open("/root/hfdl/pooler_distill/chatsft/dolphin_v1.jsonl") if l.strip()]
v1 = [r for r in v1 if r.get("q") and r.get("reply")]
v2q = set((json.loads(l).get("q") or "").strip() for l in open("/root/hfdl/pooler_distill/chatsft/dolphin_v2.jsonl") if l.strip())
cand = [r for r in v1 if r["q"].strip() not in v2q]; random.Random(0).shuffle(cand)
with open("/root/work/dolphin_heldout100.jsonl", "w") as o:
    for r in cand[:100]: o.write(json.dumps({"q": r["q"], "ref": r["reply"]}, ensure_ascii=False) + "\n")
print("[dolphin held-out] built, 100 problems")
PD2
    fi
    python3 - <<'PYH2'
import json
rows = [json.loads(l) for l in open("/root/work/dolphin_heldout100.jsonl") if l.strip()]
with open("/root/work/dolphinq.jsonl", "w") as o:
    for r in rows: o.write(json.dumps({"q": r["q"]}, ensure_ascii=False) + "\n")
print("[dolphin held-out questions]", len(rows))
PYH2
  fi
  if [ "${RQSRC:-}" = "gsm" ]; then
    [ -s /root/work/gsm8k_test.raw ] || curl -sSL --retry 3 -o /root/work/gsm8k_test.raw "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    python3 - <<PYG
import json
rows = [json.loads(l) for l in open("/root/work/gsm8k_test.raw") if l.strip()][:${RQN:-100}]
with open("/root/work/dolphinq.jsonl", "w") as o:
    for r in rows: o.write(json.dumps({"q": r["question"].strip(), "gold": r["answer"].split("####")[-1].strip()}, ensure_ascii=False) + "\n")
print("[gsm questions]", len(rows))
PYG
  fi
  [ -s /root/hfdl/pooler_distill/chat_eval60.jsonl ] || hf download $R --include "pooler_distill/chat_eval60.jsonl" --local-dir /root/hfdl 2>&1 | tail -1
  cat > /root/reevalkeep.sh <<RK
#!/bin/bash
RRUN=$RRUN; RHF=$RHF; R=$R; RCKPT=$RCKPT; RQSRC=${RQSRC:-}; RHINT=${RHINT:-}; RFAST=${RFAST:-1}; RB=${RB:-12}; RLOOP=${RLOOP:-}; RBUDGET=${RBUDGET:-2400}; RSHARDS=${RSHARDS:-3}; RTEMP=${RTEMP:-0.9}; RCAP=${RCAP:-600}; RGEN=${RGEN:-1500}; RN=${RN:-999}; RQ4=${RQ4:-}; RQ4SKIP=${RQ4SKIP:-}; RQ4BITS=${RQ4BITS:-4}; RLOCAL=${RLOCAL:-}; RLOCALK=${RLOCALK:-1}; RLOCALCHARS=${RLOCALCHARS:-0}; RLOCALSTORE=${RLOCALSTORE:-/root/wiki_store}; RLOCALMODEL=${RLOCALMODEL:-/root/bge-small}; RQFILE=${RQFILE:-}; RFORCE=${RFORCE:-}; RFORCEK=${RFORCEK:-5}; RLOCALPASSAGE=${RLOCALPASSAGE:-0}
RK
  cat >> /root/reevalkeep.sh <<'RKB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
run_one() {  # $1 questions file, $2 out file, $3 tag
  want=$(wc -l < "$1"); [ "${RN:-999}" -lt "$want" ] && want=${RN:-999}; [ -n "$RFORCE" ] && want=$((want * ${RFORCEK:-5}))
  [ -s "$2" ] && [ "$(wc -l < "$2")" -ge "$want" ] && return 0
  cd /root/work && env ${RLOCAL:+SP_LOCAL_STORE=$RLOCALSTORE SP_LOCAL_MODEL=$RLOCALMODEL SP_LOCAL_GPU=1 SP_LOCAL_K=${RLOCALK:-1} SP_LOCAL_CHARS=${RLOCALCHARS:-0} SP_LOCAL_PASSAGE=${RLOCALPASSAGE:-0}} SP_BASE=$RHF SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/pool_eval.py     $RCKPT "$1" "$2" --n $RN --rw 768 --maxd 384 --samepage 1 --decode plain --temp $RTEMP --gen $RGEN --stop eos --replycap $RCAP ${RQ4:+--q4 $RQ4 --q4skip "${RQ4SKIP:-}" --q4bits ${RQ4BITS:-4}} ${RFORCE:+--force $RFORCE} --tag "[$3]" >> /root/${RRUN}_$3.log 2>&1
  [ -s "$2" ] && [ "$(wc -l < "$2")" -ge "$want" ] || { echo "REEVAL_ABORT $RRUN at $3: $(tail -1 /root/${RRUN}_$3.log | cut -c1-100)"; exit 1; }
  echo "[$RRUN] $3: $(grep -h -m1 "pooler restored\|WARNING: no pooler" /root/${RRUN}_$3.log)"
}
run_fast() {  # $1 questions, $2 out: the training loop's batched rollout, RB questions at a time (the one-at-a-time evaluator took ~2 h per 100)
  cd /root/work && OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/online_loop.py $RHF /root/evalrun_$RRUN \
    --questions /root/work/selfq_all.jsonl --dolphin /root/work/dolphin_v1.jsonl --heldout /root/work/eval300.jsonl --pooler-init $RCKPT \
    --b $RB --gen $RGEN --budget ${RBUDGET:-2400} --temp $RTEMP --maxsrch 7 --pooler none --lora-rank 16 --lora-layers 20-27 --stop eos ${RLOOP:+--loop-break $RLOOP} \
    --eval-file "$1" --eval-out "$2" > /root/${RRUN}_fast.log 2>&1
  grep -q EVAL_DONE /root/${RRUN}_fast.log || { echo "REEVAL_ABORT $RRUN (fast): $(grep -E 'Error|error' /root/${RRUN}_fast.log | tail -1 | cut -c1-160)"; exit 1; }
}
if [ -n "$RQSRC" ]; then
  if [ "$RFAST" = 1 ]; then run_fast /root/work/dolphinq.jsonl /root/work/${RRUN}_out_0.jsonl
  else run_one /root/work/dolphinq.jsonl /root/work/${RRUN}_out_0.jsonl ${RRUN}0; fi
  hf upload $R /root/work/${RRUN}_out_0.jsonl pooler_distill/chatsft/rollouts/${RRUN}_dolphin.jsonl >/dev/null 2>&1   # full replies, the log shows 900 chars
  if [ "$RQSRC" = "dolphinh" ]; then
    OAI_KEY=$(cat /root/.oai 2>/dev/null) DSK_KEY=$(cat /root/.dsk 2>/dev/null) JUDGE_API=${RJUDGE:-openai} python3 - <<'PYJ'
import json, os, glob, urllib.request, time
from concurrent.futures import ThreadPoolExecutor
src = open("/root/work/online_loop.py").read(); i = src.index("REASON_SYS = "); j = src.index('"""', src.index('"""', i) + 3) + 3
ns = {}; exec(src[i:j], ns); SYS = ns["REASON_SYS"]
ref = {json.loads(l)["q"].strip(): json.loads(l)["ref"] for l in open("/root/work/dolphin_heldout100.jsonl") if l.strip()}
rows = [json.loads(l) for l in open(sorted(glob.glob("/root/work/*_out_0.jsonl"), key=os.path.getmtime)[-1]) if l.strip()]
OAI = os.environ.get("JUDGE_API") == "openai"
key = os.environ.get("OAI_KEY" if OAI else "DSK_KEY", "")
URL = "https://api.openai.com/v1/chat/completions" if OAI else "https://api.deepseek.com/chat/completions"
def judge(r):
    t = r["text"]; reply = t.split("</think>")[-1].strip() if "</think>" in t else ""
    if not reply: return 0, "unfinished"
    body = {"model": "gpt-5-nano" if OAI else "deepseek-flash", "messages": [{"role": "system", "content": SYS},
            {"role": "user", "content": f"QUESTION:\n{r['q'][:2000]}\n\nREFERENCE ANSWER:\n{ref.get(r['q'].strip(), '')[:3000]}\n\nASSISTANT ANSWER:\n{reply[:3000]}"}]}
    for _ in range(3):
        try:
            body.update({"max_completion_tokens": 2000} if OAI else {"max_tokens": 2000, "temperature": 0})
            d = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}), timeout=180))
            c = d["choices"][0]["message"].get("content") or ""; v = json.loads(c[c.find("{"): c.rfind("}") + 1])
            return int(all(bool(v.get(k)) for k in ("solves_it", "follows_the_request", "language_english", "clean"))), v
        except Exception as e: time.sleep(3); err = type(e).__name__
    return 0, {"error": err}
with ThreadPoolExecutor(max_workers=8) as ex: res = list(ex.map(judge, rows))
ok = sum(a for a, _ in res); th = sorted(len(r["text"].split("</think>")[0].split()) for r in rows); errs = sum(1 for _, v in res if isinstance(v, dict) and "error" in v)
print(f"DOLPHIN_ACC {100*ok/max(len(rows),1):.1f}% ({ok}/{len(rows)})  judge errors {errs}  think median {th[len(th)//2] if th else 0} words")
with open("/root/work/" + os.path.basename(sorted(glob.glob("/root/work/*_out_0.jsonl"), key=os.path.getmtime)[-1]).replace("_out_0", "_judged"), "w") as o:
    for r, (a, v) in zip(rows, res): o.write(json.dumps({"q": r["q"], "pass": a, "why": v, "text": r["text"]}, ensure_ascii=False) + "\n")
PYJ
    hf upload $R /root/work/${RRUN}_judged.jsonl pooler_distill/chatsft/rollouts/${RRUN}_judged.jsonl >/dev/null 2>&1
    echo "REEVAL_DONE $RRUN $(date -u)"; exit 0
  fi
  if [ "$RQSRC" = "gsm" ]; then
    python3 - <<'PYA'
import json, re
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
def fin(t):
    m = re.findall(r"\\boxed\{([^{}]*)\}", t); c = m[-1] if m else (t.split("####")[-1] if "####" in t else t)
    n = NUM.findall(c)
    try: return float(n[-1].replace(",", "")) if n else None
    except ValueError: return None
gold = {}
for l in open("/root/work/dolphinq.jsonl"):
    r = json.loads(l); gold[r["q"].strip()] = float(r["gold"].replace(",", ""))
rows = [json.loads(l) for l in open(sorted(__import__("glob").glob("/root/work/*_out_0.jsonl"), key=__import__("os").path.getmtime)[-1]) if l.strip()]
ok = fin_n = 0; th = []
for r in rows:
    t = r["text"]; reply = t.split("</think>")[-1] if "</think>" in t else ""
    g = gold.get(r["q"].strip()); a = fin(reply) if reply else None
    ok += int(g is not None and a is not None and abs(a - g) <= 1e-6 * max(1.0, abs(g))); fin_n += bool(reply); th.append(len(t.split("</think>")[0].split()))
th.sort(); print(f"GSM_ACC {100*ok/max(len(rows),1):.1f}% ({ok}/{len(rows)})  finished {100*fin_n/max(len(rows),1):.0f}%  think median {th[len(th)//2] if th else 0} words")
PYA
    echo "REEVAL_DONE $RRUN $(date -u)"; exit 0
  fi
  python3 - <<'PYT'
import json
import glob
for r in [json.loads(l) for l in open(sorted(glob.glob("/root/work/*dolph_out_0.jsonl"))[-1]) if l.strip()]:
    think, _, reply = r["text"].partition("</think>")
    print(f"DTEXT ===== {r['q'][:220]}")
    print(f"DTEXT searches {r['ns']} | think {len(think.split())} words | reply {len(reply.split())} words")
    print("DTEXT reply:", reply.strip()[:900].replace("\n", " "))
PYT
  echo "REEVAL_DONE $RRUN $(date -u)"; exit 0
fi
# RHINT: one instruction put in front of every held-out question (the "knob" test: does a drifted checkpoint search again
# when told to?). The gold and the scoring are untouched; the hinted copies are separate files.
EVP=ev
if [ -n "$RHINT" ]; then
  EVP=evh_${RRUN}
  python3 - "$RHINT" "${RSHARDS:-3}" "$EVP" <<'PYH'
import json, sys
hint, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
for i in range(n):
    with open(f"/root/work/{out}_{i}.jsonl", "w") as o:
        for l in open(f"/root/work/ev_{i}.jsonl"):
            try: r = json.loads(l)
            except Exception: continue
            r["q"] = hint.strip() + " " + r["q"]; o.write(json.dumps(r, ensure_ascii=False) + "\n")
print("[hint]", repr(hint))
PYH
fi
if [ -n "$RQFILE" ]; then   # one custom question file (a reward table: RFORCE pages per question), no chat eval
  run_one $RQFILE /root/work/${RRUN}_out_0.jsonl ${RRUN}0; hf upload $R /root/work/${RRUN}_out_0.jsonl pooler_distill/chatsft/rollouts/${RRUN}_0.jsonl >/dev/null 2>&1; echo "[$RRUN] uploaded $(tail -1 /root/${RRUN}_${RRUN}0.log | cut -c1-110)"
  echo "REEVAL_DONE $RRUN $(date -u)"; exit 0
fi
for i in $(seq 0 $((${RSHARDS:-3} - 1))); do run_one /root/work/${EVP}_$i.jsonl /root/work/${RRUN}_out_$i.jsonl ${RRUN}$i; hf upload $R /root/work/${RRUN}_out_$i.jsonl pooler_distill/chatsft/rollouts/${RRUN}_$i.jsonl >/dev/null 2>&1; echo "[$RRUN] shard $i uploaded $(tail -1 /root/${RRUN}_${RRUN}$i.log | cut -c1-110)"; done
run_one /root/hfdl/pooler_distill/chat_eval60.jsonl /root/work/${RRUN}_chat_out.jsonl ${RRUN}chat
hf upload $R /root/work/${RRUN}_chat_out.jsonl pooler_distill/chatsft/rollouts/${RRUN}_chat.jsonl >/dev/null 2>&1
python3 - <<'PYT'
import json, glob
for f in sorted(glob.glob("/root/work/" + "s4prod" + "_out_*.jsonl")):
    for r in [json.loads(l) for l in open(f) if l.strip()][:6]:
        think, _, reply = r["text"].partition("</think>")
        print(f"TEXT ===== {r['q'][:100]} | gold {r['gold']} | correct {r['correct']} grounded {r['grounded']} searches {r['ns']}")
        print("TEXT queries:", "; ".join(r.get("queries", [])[:8]))
        print("TEXT think tail:", think[-700:].replace("\n", " "))
        print("TEXT reply:", reply.strip()[:900].replace("\n", " "))
PYT
echo "REEVAL_DONE $RRUN $(date -u)"
RKB
  # the keeper must not inherit the lock descriptor: it held it for the whole evaluation and every later control run skipped (the s4gsm baseline never launched)
  chmod +x /root/reevalkeep.sh; setsid nohup bash -c 'bash /root/reevalkeep.sh 2>&1 | tee -a /root/reeval.log' >> /proc/1/fd/1 2>&1 < /dev/null 9>&- &
  sleep 5; echo "REEVAL_LAUNCH_DONE $RRUN model=$RHF $(date -u)"; exit 0
fi

if [ "$MODE" = "sft2" ]; then
  # Supervised fine-tuning of the step-200 student on its own search prefixes continued by a
  # teacher: the thinking after the last result and a conversational reply. Then the held-out
  # measurement, so the reply style is checked on the same 150 questions as everything else.
  SRUN2=${SRUN2:-s2}; SDATA=${SDATA:-pooler_distill/chatsft/search_sft_v2.jsonl}
  pkill -f "afterkee[p].sh"; pkill -f "evalkee[p].sh"; pkill -f "dwqkee[p].sh"; pkill -f "probekee[p].sh"; pkill -f "poolkee[p].sh"; pkill -f "jointkee[p].sh"; pkill -f "sftkee[p].sh"; pkill -f "genkee[p].sh"; sleep 5
  export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
  R=baya1116/hypernet-sp-distill
  for try in 1 2 3; do hf download $R --include "$SDATA" --local-dir /root/hfdl 2>&1 | tail -1 && break; sleep 10; done
  cp /root/hfdl/$SDATA /root/work/sft2_data.jsonl
  # replay: strict single-sentence QA traces of the same student, so the search reflex is not traded away
  if [ "${S2REPLAY:-0}" -gt 0 ]; then
    [ -s /root/hfdl/pooler_distill/grpo_pool3/rollouts.jsonl ] || for try in 1 2 3; do hf download $R --include "pooler_distill/grpo_pool3/rollouts.jsonl" --local-dir /root/hfdl 2>&1 | tail -1 && break; sleep 10; done
    python3 - <<PYR
import json, random
held=set()
for line in open("/root/work/eval300.jsonl"):
    try: held.add(json.loads(line).get("q",""))
    except Exception: pass
rows=[]; seen=set()
for line in open("/root/hfdl/pooler_distill/grpo_pool3/rollouts.jsonl"):
    try: d=json.loads(line)
    except Exception: continue
    if not (d.get("correct") and d.get("grounded") and d.get("landed")): continue
    q,t=d.get("q",""),d.get("text","")
    if not q or not t or q in held or (q,t[:200]) in seen: continue
    seen.add((q,t[:200])); rows.append({"q":q,"text":t})
random.seed(0); random.shuffle(rows); rows=rows[:$S2REPLAY]
with open("/root/work/sft2_data.jsonl","a") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")
print(f"[sft2] replay {len(rows)} strict traces appended")
PYR
  fi
  echo "traces: $(wc -l < /root/work/sft2_data.jsonl)"
  HF2=/root/sft2_hf_$SRUN2; LOG2=/root/sft2_$SRUN2.log
  RUNNING=0
  pgrep -f "sft_lora.p[y]" >/dev/null && { RUNNING=1; echo "already training: $(tail -1 $LOG2)"; }
  [ -s $HF2/model.safetensors ] && { RUNNING=1; echo "sft2 $SRUN2 already finished"; }
  if [ "$RUNNING" = "0" ]; then
    pkill -f "pool_eval.p[y]"; sleep 8; pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 2
    rm -f $LOG2
    cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/sft_lora.py \
      --base /root/eval_hf200 --data /root/work/sft2_data.jsonl --out $HF2 --log $LOG2 \
      --rank ${S2RANK:-16} --lr ${S2LR:-3e-5} --epochs ${S2EPOCHS:-2} --accum ${S2ACCUM:-8} \
      >> /root/sft2_run_$SRUN2.log 2>&1 < /dev/null &
    sleep 20
  fi
  cat > /root/sft2keep.sh <<SK2
#!/bin/bash
SRUN2=$SRUN2; HF2=$HF2; R=$R
SK2
  cat >> /root/sft2keep.sh <<'SK2B'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
until [ -s $HF2/model.safetensors ] && grep -q SFT2_DONE /root/sft2_run_$SRUN2.log 2>/dev/null; do sleep 60; done
pkill -f "sft_lora.p[y]"; sleep 10
hf upload $R ${HF2}_adapter pooler_distill/chatsft/${SRUN2}_adapter >/dev/null 2>&1
hf upload $R /root/sft2_$SRUN2.log pooler_distill/chatsft/logs/sft2_$SRUN2.log >/dev/null 2>&1
echo "[$SRUN2] adapter and log uploaded"
while :; do
  done=1
  for i in 0 1 2; do
    want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
    have=$(wc -l < /root/work/${SRUN2}_out_$i.jsonl 2>/dev/null || echo 0)
    [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
    done=0
    pgrep -f "pool_eval.py .* /root/work/ev_$i.jsonl" >/dev/null && continue
    echo "[$SRUN2-eval $(date -u +%H:%M)] shard $i at $have/$want - starting"
    cd /root/work && SP_BASE=$HF2 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py       /root/pooler200.safetensors /root/work/ev_$i.jsonl /root/work/${SRUN2}_out_$i.jsonl       --n 999 --rw 768 --maxd 384 --samepage 1 --decode plain --temp ${S2TEMP:-0.9} --stop eos --replycap 200 --tag "[$SRUN2$i]"       >> /root/${SRUN2}_$i.log 2>&1 < /dev/null &
    sleep 60
  done
  if [ "$done" = "1" ]; then
    for i in 0 1 2; do hf upload $R /root/work/${SRUN2}_out_$i.jsonl pooler_distill/chatsft/rollouts/${SRUN2}_$i.jsonl >/dev/null 2>&1; done
    echo "SFT2_EVAL_DONE $SRUN2 $(date -u)"; break
  fi
  sleep 120
done
SK2B
  chmod +x /root/sft2keep.sh
  pkill -f "sft2kee[p].sh"; sleep 1
  setsid nohup bash /root/sft2keep.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  sleep 40; tail -3 /root/sft2_run_$SRUN2.log 2>/dev/null | cut -c1-160; echo "SFT2_LAUNCH_DONE $SRUN2 $(date -u)"
  exit 0
fi
if [ "$MODE" = "selfgen" ]; then
  # The no-search side, on-policy: the lineage's base writes candidate replies to real prompts; a judge
  # picks later, off the box. Then hand the card to the nq_open generation.
  export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
  R=baya1116/hypernet-sp-distill
  pkill -f "genkee[p].sh"; sleep 2
  for try in 1 2 3; do hf download $R --include "pooler_distill/chatsft/chat_prompts.jsonl" --local-dir /root/hfdl 2>&1 | tail -1 && break; sleep 10; done
  [ -s /root/base_distill/model.safetensors ] || for try in 1 2 3; do hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir /root/base_distill 2>&1 | tail -1 && break; sleep 10; done
  SGOUT=${SGOUT:-self_cands.jsonl}
  if [ ! -s /root/work/$SGOUT ] || ! grep -q SELFGEN_DONE /root/selfgen.log 2>/dev/null; then
    cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/selfgen_gpu.py \
      --base /root/base_distill --prompts /root/hfdl/pooler_distill/chatsft/chat_prompts.jsonl --out /root/work/$SGOUT \
      --k ${SGK:-4} --batch ${SGBATCH:-8} --maxnew ${SGMAXNEW:-360} > /root/selfgen.log 2>&1
    tail -2 /root/selfgen.log
  fi
  hf upload $R /root/work/$SGOUT pooler_distill/chatsft/$SGOUT 2>&1 | tail -1
  echo "SELFGEN_UPLOADED $(wc -l < /root/work/$SGOUT) candidates $(date -u)"
  [ "${SGCHAIN:-1}" = "1" ] || exit 0
  MODE=gen   # fall through into the nq_open generation with the same card
fi
if [ "$MODE" = "gen" ]; then
  # Data generation for the conversational lineage: the step-200 student answers real questions
  # (nq_open, people's own search queries) through the same environment the evaluator uses, one
  # rollout each. Correct, grounded traces become the search prefixes a teacher continues.
  GRUN=${GRUN:-g1}; GN=${GN:-1500}; GTEMP=${GTEMP:-0.8}; GPAR=${GPAR:-3}   # GPAR: evaluators at once (1 on a 12 GB card)
  pkill -f "afterkee[p].sh"; pkill -f "evalkee[p].sh"; pkill -f "dwqkee[p].sh"; pkill -f "probekee[p].sh"; pkill -f "poolkee[p].sh"; pkill -f "jointkee[p].sh"; pkill -f "sftkee[p].sh"; sleep 5
  export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
  R=baya1116/hypernet-sp-distill
  GQFILE=${GQFILE:-pooler_distill/nq_pool.jsonl}   # any {"q","gold"} jsonl on the HF repo (e.g. a chat-register rewrite pool)
  [ -s /root/hfdl/$GQFILE ] || for try in 1 2 3; do hf download $R --include "$GQFILE" --local-dir /root/hfdl 2>&1 | tail -1 && break; sleep 10; done
  python3 - <<PYG
import json
qs=[l for l in open("/root/hfdl/$GQFILE") if l.strip()][:$GN]
for i in range(3): open(f"/root/work/gq_{i}.jsonl","w").writelines(qs[i::3])
print(f"[gen] {len(qs)} questions -> {[len(qs[i::3]) for i in range(3)]}")
PYG
  for i in 0 1 2; do for try in 1 2 3; do hf download $R --include "pooler_distill/nq_gen/${GRUN}_$i.jsonl" --local-dir /root/hfdl 2>/dev/null | tail -1 && break; sleep 5; done
    [ -s /root/hfdl/pooler_distill/nq_gen/${GRUN}_$i.jsonl ] && cp /root/hfdl/pooler_distill/nq_gen/${GRUN}_$i.jsonl /root/work/${GRUN}_out_$i.jsonl; done
  echo "restored $(cat /root/work/${GRUN}_out_*.jsonl 2>/dev/null | wc -l) rollouts"
  cat > /root/genkeep.sh <<GKQ
#!/bin/bash
GRUN=$GRUN; GTEMP=$GTEMP; R=$R; GPAR=$GPAR
GKQ
  cat >> /root/genkeep.sh <<'GKQ2'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
last_up=0
while :; do
  done=1
  for i in 0 1 2; do
    want=$(wc -l < /root/work/gq_$i.jsonl 2>/dev/null || echo 0)
    have=$(wc -l < /root/work/${GRUN}_out_$i.jsonl 2>/dev/null || echo 0)
    [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
    done=0
    pgrep -f "pool_eval.py .* /root/work/gq_$i.jsonl" >/dev/null && continue
    [ "$(pgrep -fc "pool_eval.p[y]")" -ge "$GPAR" ] && continue
    echo "[$GRUN-gen $(date -u +%H:%M)] shard $i at $have/$want - starting (temp $GTEMP)"
    cd /root/work && SP_BASE=/root/eval_hf200 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py       /root/pooler200.safetensors /root/work/gq_$i.jsonl /root/work/${GRUN}_out_$i.jsonl       --n 9999 --rw 768 --maxd 384 --samepage 1 --decode plain --temp $GTEMP --tag "[$GRUN$i]"       >> /root/${GRUN}_$i.log 2>&1 < /dev/null &
    sleep 60
  done
  now=$(date +%s)
  if [ "$done" = "1" ] || [ $((now - last_up)) -ge 1800 ]; then
    for i in 0 1 2; do [ -s /root/work/${GRUN}_out_$i.jsonl ] && hf upload $R /root/work/${GRUN}_out_$i.jsonl pooler_distill/nq_gen/${GRUN}_$i.jsonl >/dev/null 2>&1; done
    last_up=$now; echo "[$GRUN-gen $(date -u +%H:%M)] uploaded $(cat /root/work/${GRUN}_out_*.jsonl 2>/dev/null | wc -l) rollouts"
  fi
  [ "$done" = "1" ] && { echo "GEN_DONE $GRUN $(date -u)"; break; }
  sleep 120
done
GKQ2
  chmod +x /root/genkeep.sh
  pkill -f "genkee[p].sh"; sleep 1
  setsid nohup bash /root/genkeep.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  sleep 30; echo "GEN_LAUNCH_DONE $GRUN n=$GN $(date -u)"
  exit 0
fi
if [ "$MODE" = "publish2" ]; then
  # Everything worth keeping leaves the box before it is stopped.
  pkill -f "afterkee[p].sh"; pkill -f "evalkee[p].sh"; pkill -f "dwqkee[p].sh"; pkill -f "probekee[p].sh"
  pkill -f "poolkee[p].sh"; pkill -f "jointkee[p].sh"; pkill -f "sftkee[p].sh"; sleep 5
  pkill -f "pool_eval.p[y]"; sleep 8; pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 2
  export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
  R=baya1116/hypernet-sp-distill; D=pooler_distill/grpo_pool3_step200_q4
  # the directory the app would load must unpack to the values the evaluator measured
  [ -s /root/sft_mlx4_s1/model.safetensors ] && python3 /root/work/checkmlx.py /root/sft_mlx4_s1 /root/sft_hf_s1 2>&1 | tail -3
  python3 - <<'PYP' > /root/work/q4_metrics.json
import json, glob, collections, math, os

def load(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            try: rows.append(json.loads(line))
            except Exception: pass
    if not rows: return None
    per = {}
    for k in ("correct", "grounded", "landed", "ns"):
        d = collections.defaultdict(list)
        for r in rows: d[r.get("q", "")].append(float(r.get(k) or 0))
        per[k] = {q: sum(v)/len(v) for q, v in d.items()}
    return {"rollouts": len(rows), "questions": len(per["correct"]),
            "correct": round(100*sum(float(r.get("correct") or 0) for r in rows)/len(rows), 1),
            "grounded": round(100*sum(float(r.get("grounded") or 0) for r in rows)/len(rows), 1),
            "landed": round(100*sum(float(r.get("landed") or 0) for r in rows)/len(rows), 1),
            "searches": round(sum(float(r.get("ns") or 0) for r in rows)/len(rows), 2), "_per": per}

RUNS = [("bf16", "/root/work/ev_out_*.jsonl"), ("4bit_plain", "/root/work/q4_out_*.jsonl"),
        ("4bit_ste_lora", "/root/work/qa_out_*.jsonl"), ("4bit_dwq_temp2", "/root/work/dw_out_*.jsonl"),
        ("4bit_dwq_temp1", "/root/work/d3_out_*.jsonl"), ("4bit_group32", "/root/work/g32_out_*.jsonl"),
        ("4bit_joint_pooler", "/root/work/j1_out_*.jsonl"),
        ("4bit_self_trace_s1", "/root/work/s1_out_*.jsonl"),
        ("bf16_temp06", "/root/work/p_t06bf16_out_*.jsonl"), ("4bit_plain_temp06", "/root/work/p_t06q4_out_*.jsonl"),
        ("4bit_pooler_only", "/root/work/j2_out_*.jsonl"),
        ("4bit_float_embed", "/root/work/p_embfloat_out_*.jsonl")]
L = {n: load(p) for n, p in RUNS}
F = L["bf16"]

def paired(S):
    if not (F and S): return None
    c = sorted(set(F["_per"]["correct"]) & set(S["_per"]["correct"]))
    if not c: return None
    out = {"questions": len(c)}
    for k in ("correct", "grounded", "landed", "ns"):
        d = [S["_per"][k][q] - F["_per"][k][q] for q in c]
        m = sum(d)/len(d); sd = (sum((x-m)**2 for x in d)/len(d))**0.5
        sc = 1 if k == "ns" else 100
        out[k] = {"delta": round(sc*m, 1), "se": round(sc*sd/math.sqrt(len(c)), 1)}
    return out

print(json.dumps({
  "what": "What the phone's 4-bit conversion costs this model, and six attempts to recover it",
  "format": {"scheme": "MLX affine, group 64, 4 bits, scales and biases fp16",
             "tensors": "7 projections x 28 blocks + embed_tokens + lm_head = 198, 1776.9M weights",
             "relative_weight_error": {"overall": 9.51, "embed_tokens": 10.19, "lm_head": 10.35}},
  "heldout": {"file": "eval300.jsonl first 300 lines, 150 distinct questions x 2",
              "settings": {"rw": 768, "maxd": 384, "samepage": 1, "temp": 0.9, "gen": 1500, "maxs": 5}},
  "runs": {n: {k: v for k, v in (L[n] or {}).items() if k != "_per"} for n, _ in RUNS if L[n]},
  "paired_against_bf16": {n: paired(L[n]) for n, _ in RUNS[1:] if L[n]},
  "conclusion": "The loss is about seven points and none of the six attempts separates from the "
                "others or from doing nothing: they span -5.7 to -11.6 with standard errors of 3.2 "
                "to 5.5. Every run closed most of the KL gap to bf16 and recovered grounding; none "
                "recovered the score. See docs/quantization_4bit.md.",
}, ensure_ascii=False, indent=2))
PYP
  cat /root/work/q4_metrics.json
  hf upload $R /root/work/q4_metrics.json $D/q4_metrics.json 2>&1 | tail -1
  for f in /root/dwq_d3.log /root/dwq_d4.log /root/joint_j1.log /root/joint_j2.log /root/qat.log /root/sft_s1.log; do
    [ -s $f ] && hf upload $R $f $D/logs/$(basename $f) >/dev/null 2>&1
  done
  for p in q4 qa d3 d4 g32 j1 j2 p_embfloat p_t06q4 p_t06bf16 s1; do
    for i in 0 1 2; do
      [ -s /root/work/${p}_out_$i.jsonl ] && hf upload $R /root/work/${p}_out_$i.jsonl $D/rollouts/${p}_$i.jsonl >/dev/null 2>&1
    done
  done
  echo "--- deployable directories ---"
  for n in d3 d4; do
    [ -s /root/dwq_mlx4_$n/model.safetensors ] && { echo "dwq_$n $(du -shL /root/dwq_mlx4_$n | cut -f1)"; hf upload $R /root/dwq_mlx4_$n $D/dwq_${n}_mlx4 2>&1 | tail -1; }
  done
  [ -s /root/sft_mlx4_s1/model.safetensors ] && { echo "sft s1 $(du -shL /root/sft_mlx4_s1 | cut -f1)"; hf upload $R /root/sft_mlx4_s1 $D/sft_s1_mlx4 2>&1 | tail -1; }
  [ -s /root/sft/s1.pt ] && hf upload $R /root/sft/s1.pt $D/sft_s1_params.pt >/dev/null 2>&1
  [ -s /root/joint_hf_j1/model.safetensors ] && { echo "joint j1 $(du -shL /root/joint_hf_j1 | cut -f1)"; hf upload $R /root/joint_hf_j1 $D/joint_j1_hf 2>&1 | tail -1; }
  [ -s /root/pooler_joint_j1.safetensors ] && hf upload $R /root/pooler_joint_j1.safetensors $D/pooler_joint_j1.safetensors 2>&1 | tail -1
  # the pooler-only run: its model directory is the untouched 4-bit grid, so only the pooler is new
  [ -s /root/pooler_joint_j2.safetensors ] && hf upload $R /root/pooler_joint_j2.safetensors $D/pooler_only_j2.safetensors 2>&1 | tail -1
  [ -s /root/joint/j2.pt ] && hf upload $R /root/joint/j2.pt $D/pooler_only_j2_params.pt >/dev/null 2>&1
  [ -s /root/dwq/j1.pt ] && hf upload $R /root/dwq/j1.pt $D/joint_j1_params.pt >/dev/null 2>&1
  [ -s /root/joint/j1.pt ] && hf upload $R /root/joint/j1.pt $D/joint_j1_params.pt 2>&1 | tail -1
  [ -s /root/dwq/d4.pt ] && hf upload $R /root/dwq/d4.pt $D/dwq_d4_params.pt 2>&1 | tail -1
  echo "PUBLISH2_DONE $(date -u)"
  exit 0
fi

if [ "$MODE" = "joint" ]; then
  # The quantization parameters and the pooler, trained together against the 16-bit system, through
  # the compressed forward the evaluator actually runs. Everything before this trained on plain
  # contexts, so the pooler - where the compression happens - never saw a gradient at all.
  JRUN=${JRUN:-j1}
  pkill -f "afterkee[p].sh"; pkill -f "evalkee[p].sh"; pkill -f "dwqkee[p].sh"; pkill -f "probekee[p].sh"; pkill -f "poolkee[p].sh"; sleep 5
  HF=/root/joint_hf_$JRUN; POOL=/root/pooler_joint_$JRUN.safetensors; LOG=/root/joint_$JRUN.log
  RUNNING=0
  pgrep -f "jointfit.p[y]" >/dev/null && { RUNNING=1; echo "already running: $(tail -1 $LOG)"; }
  [ -s $HF/model.safetensors ] && { RUNNING=1; echo "joint run $JRUN already finished"; }
  if [ "$RUNNING" = "0" ]; then
    pkill -f "dwq.p[y]"; pkill -f "poolerfit.p[y]"; pkill -f "pool_eval.p[y]"; sleep 8
    pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 2
    python3 /root/work/q4.py || { echo "Q4 SELFTEST FAILED - not launching"; exit 0; }
    rm -f $LOG
    cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/jointfit.py \
      --ckpt /root/pooler200.safetensors --data /root/work/dwq_calib/train.jsonl \
      --out-hf $HF --out-mlx /root/joint_mlx4_$JRUN --out-pooler $POOL \
      --state /root/joint/$JRUN.pt --log $LOG --clip-search ${JCLIP:-1} --lr-q ${JLRQ:-2e-6} \
      --val 8 --val-every 2 --selftest 3 2>&1 | tail -16
    grep -q "^step 3 " $LOG 2>/dev/null || { echo "JOINTFIT SELFTEST FAILED - not launching"; exit 0; }
    rm -f $LOG
    cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/jointfit.py \
      --ckpt /root/pooler200.safetensors --data /root/work/dwq_calib/train.jsonl \
      --out-hf $HF --out-mlx /root/joint_mlx4_$JRUN --out-pooler $POOL \
      --state /root/joint/$JRUN.pt --log $LOG \
      --steps ${JSTEPS:-1200} --lr-q ${JLRQ:-2e-6} --lr-p ${JLRP:-1e-5} --clip-search ${JCLIP:-1} \
      --val 24 --val-every ${JVAL:-50} \
      >> /root/joint_run_$JRUN.log 2>&1 < /dev/null &
    sleep 20
  fi
  cat > /root/jointkeep.sh <<JKP
#!/bin/bash
JRUN=$JRUN; HF=$HF; POOL=$POOL
JKP
  cat >> /root/jointkeep.sh <<'JKP2'
until [ -s $HF/model.safetensors ] && grep -q JOINTFIT_DONE /root/joint_run_$JRUN.log 2>/dev/null; do sleep 60; done
pkill -f "jointfit.p[y]"; sleep 10
while :; do
  for i in 0 1 2; do
    want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
    have=$(wc -l < /root/work/${JRUN}_out_$i.jsonl 2>/dev/null || echo 0)
    [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
    pgrep -f "pool_eval.py .* /root/work/ev_$i.jsonl" >/dev/null && continue
    echo "[$JRUN-eval $(date -u +%H:%M)] shard $i at $have/$want - starting (quantized body + fitted pooler)"
    grep -viE "^\s*$" /root/${JRUN}_$i.log 2>/dev/null | tail -4 | cut -c1-200
    cd /root/work && SP_BASE=$HF SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py \
      $POOL /root/work/ev_$i.jsonl /root/work/${JRUN}_out_$i.jsonl \
      --n 999 --rw 768 --maxd 384 --samepage 1 --decode plain --tag "[$JRUN$i]" \
      >> /root/${JRUN}_$i.log 2>&1 < /dev/null &
    sleep 60
  done
  done=1
  for i in 0 1 2; do
    want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
    have=$(wc -l < /root/work/${JRUN}_out_$i.jsonl 2>/dev/null || echo 0)
    [ "$have" -ge "$want" ] || done=0
  done
  if [ "$done" = "1" ]; then
    # The untouched 4-bit arm is the reference every other number leans on and it stopped at 174
    # rollouts. pool_eval resumes from its own output, so this only runs what is missing.
    for i in 0 1 2; do
      want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
      have=$(wc -l < /root/work/q4_out_$i.jsonl 2>/dev/null || echo 0)
      [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
      pgrep -f "pool_eval.py .* /root/work/ev_$i.jsonl" >/dev/null && continue
      echo "[baseline $(date -u +%H:%M)] plain 4-bit shard $i at $have/$want - filling in"
      cd /root/work && SP_BASE=/root/eval_hf200 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py \
        /root/pooler200.safetensors /root/work/ev_$i.jsonl /root/work/q4_out_$i.jsonl \
        --n 999 --rw 768 --maxd 384 --samepage 1 --decode plain --q4 1 --tag "[q4$i]" \
        >> /root/q4_$i.log 2>&1 < /dev/null &
      sleep 60
    done
  fi
  sleep 120
done
JKP2
  chmod +x /root/jointkeep.sh
  pkill -f "jointkee[p].sh"; sleep 1
  setsid nohup bash /root/jointkeep.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  sleep 30; t; echo "JOINT_LAUNCH_DONE $JRUN $(date -u)"
  exit 0
fi

if [ "$MODE" = "pool" ]; then
  # Fit the pooler to the embedding table the phone actually feeds it. Ships as a pooler file:
  # no model directory, no size change, no app code.
  pkill -f "afterkee[p].sh"; pkill -f "evalkee[p].sh"; pkill -f "dwqkee[p].sh"; pkill -f "probekee[p].sh"; sleep 5
  PRUN=${PRUN:-p_poolfit}
  RUNNING=0
  pgrep -f "poolerfit.p[y]" >/dev/null && { RUNNING=1; echo "already running: $(tail -1 /root/poolerfit.log)"; }
  [ -s /root/pooler_q4.safetensors ] && { RUNNING=1; echo "pooler already fitted"; }
  if [ "$RUNNING" = "0" ]; then
    pkill -f "dwq.p[y]"; pkill -f "pool_eval.p[y]"; sleep 8; pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 2
    cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/poolerfit.py \
      --ckpt /root/pooler200.safetensors --out /root/pooler_q4.safetensors \
      --data /root/work/dwq_calib/train.jsonl --selftest 3 2>&1 | tail -14
    grep -q "^step 3 " /root/poolerfit.log 2>/dev/null || { echo "POOLERFIT SELFTEST FAILED - not launching"; exit 0; }
    cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/poolerfit.py \
      --ckpt /root/pooler200.safetensors --out /root/pooler_q4.safetensors \
      --data /root/work/dwq_calib/train.jsonl --steps ${PSTEPS:-3000} --lr ${PLR:-1e-5} \
      >> /root/poolerfit_run.log 2>&1 < /dev/null &
    sleep 20
  fi
  cat > /root/poolkeep.sh <<PKP
#!/bin/bash
PRUN=$PRUN
PKP
  cat >> /root/poolkeep.sh <<'PKP2'
until [ -s /root/pooler_q4.safetensors ] && grep -q POOLERFIT_DONE /root/poolerfit_run.log 2>/dev/null; do sleep 30; done
pkill -f "poolerfit.p[y]"; sleep 5
while :; do
  for i in 0 1 2; do
    want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
    have=$(wc -l < /root/work/${PRUN}_out_$i.jsonl 2>/dev/null || echo 0)
    [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
    pgrep -f "pool_eval.py .* /root/work/ev_$i.jsonl" >/dev/null && continue
    echo "[$PRUN $(date -u +%H:%M)] shard $i at $have/$want - starting (pooler fitted to the 4-bit table)"
    grep -viE "^\s*$" /root/${PRUN}_$i.log 2>/dev/null | tail -4 | cut -c1-200
    cd /root/work && SP_BASE=/root/eval_hf200 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py \
      /root/pooler_q4.safetensors /root/work/ev_$i.jsonl /root/work/${PRUN}_out_$i.jsonl \
      --n 999 --rw 768 --maxd 384 --samepage 1 --decode plain --q4 1 --tag "[$PRUN$i]" \
      >> /root/${PRUN}_$i.log 2>&1 < /dev/null &
    sleep 60
  done
  sleep 120
done
PKP2
  chmod +x /root/poolkeep.sh
  pkill -f "poolkee[p].sh"; sleep 1
  setsid nohup bash /root/poolkeep.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  sleep 30; t; echo "POOLFIT_LAUNCH_DONE $(date -u)"
  exit 0
fi

if [ "$MODE" = "probe" ]; then
  # Three training attempts have each closed most of the KL gap to bf16 and left the score where it
  # was. Before a fourth, ask a different question: is the format itself the binding constraint?
  # Group 32 halves how many weights share a scale. It needs one line changed in the app and a
  # reconversion, so it is a recommendation rather than a drop-in - but if it recovers the gap, that
  # is the answer, and if it does not, no amount of training the group-64 grid will help either.
  PRUN=${PRUN:-g32}; PG=${PG:-64}; PSKIP=${PSKIP:-}; PTEMP=${PTEMP:-0.9}; PQ4=${PQ4:-1}
  # PRUNS overrides the single run: a space-separated list of tag:q4:temp, worked through in order
  PRUNS=${PRUNS:-"$PRUN:$PQ4:$PTEMP"}
  pkill -f "afterkee[p].sh"; pkill -f "evalkee[p].sh"; pkill -f "dwqkee[p].sh"; pkill -f "qat.p[y]"; sleep 5
  R=baya1116/hypernet-sp-distill; D=pooler_distill/grpo_pool3_step200_q4
  if [ ! -f /root/.d3_published ] && [ -s /root/dwq_mlx4_d3/model.safetensors ]; then
    export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
    echo "--- publishing d3 ($(du -shL /root/dwq_mlx4_d3 | cut -f1)) ---"
    hf upload $R /root/dwq_mlx4_d3 $D/dwq_d3_mlx4 2>&1 | tail -2
    hf upload $R /root/dwq/d3.pt $D/dwq_d3_params.pt 2>&1 | tail -1
    hf upload $R /root/dwq_d3.log $D/dwq_d3.log >/dev/null 2>&1
    for i in 0 1 2; do hf upload $R /root/work/d3_out_$i.jsonl $D/d3_shard$i.jsonl >/dev/null 2>&1; done
    touch /root/.d3_published; echo "d3 published"
  fi
  cat > /root/probekeep.sh <<PKQ
#!/bin/bash
PRUNS="$PRUNS"; PG=$PG; PSKIP="$PSKIP"
PKQ
  cat >> /root/probekeep.sh <<'PKQ2'
# a training run in flight owns the card; let it finish and write its directory first
while pgrep -f "dwq.p[y]" >/dev/null; do sleep 60; done
for spec in $PRUNS; do
  PRUN=${spec%%:*}; rest=${spec#*:}; PQ4=${rest%%:*}; PTEMP=${rest#*:}
  Q4FLAGS=""; [ "$PQ4" = "1" ] && Q4FLAGS="--q4 1 --q4group $PG"
  [ "$PQ4" = "1" ] && [ -n "$PSKIP" ] && Q4FLAGS="$Q4FLAGS --q4skip $PSKIP"
  while :; do
    done=1
    for i in 0 1 2; do
      want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
      have=$(wc -l < /root/work/${PRUN}_out_$i.jsonl 2>/dev/null || echo 0)
      [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
      done=0
      pgrep -f "pool_eval.py .* /root/work/ev_$i.jsonl" >/dev/null && continue
      echo "[$PRUN-probe $(date -u +%H:%M)] shard $i at $have/$want - starting (q4=$PQ4 temp=$PTEMP group $PG)"
      grep -viE "^\s*$" /root/${PRUN}_$i.log 2>/dev/null | tail -4 | cut -c1-200
      cd /root/work && SP_BASE=/root/eval_hf200 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py \
        /root/pooler200.safetensors /root/work/ev_$i.jsonl /root/work/${PRUN}_out_$i.jsonl \
        --n 999 --rw 768 --maxd 384 --samepage 1 --decode plain --temp $PTEMP $Q4FLAGS --tag "[$PRUN$i]" \
        >> /root/${PRUN}_$i.log 2>&1 < /dev/null &
      sleep 60
    done
    [ "$done" = "1" ] && break
    sleep 120
  done
  echo "[$PRUN-probe $(date -u +%H:%M)] complete"
done
PKQ2
  chmod +x /root/probekeep.sh
  pkill -f "probekee[p].sh"; sleep 1
  setsid nohup bash /root/probekeep.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  sleep 40; t; echo "PROBE_LAUNCH_DONE $PRUN group $PG $(date -u)"
  exit 0
fi

if [ "$MODE" = "dwq" ]; then
  # Train the quantization parameters rather than the weights under them, starting from the bf16
  # model that scored 39.7%. RUN names the attempt, so a second one does not overwrite the first.
  RUN=${DRUN:-d1}
  pkill -f "afterkee[p].sh"; pkill -f "evalkee[p].sh"; pkill -f "dwqkee[p].sh"; pkill -f "qat.p[y]"; sleep 5
  [ -s /root/work/dwq_calib/train.jsonl ] || { echo "NOT READY: no calibration set"; exit 0; }
  echo "=== DWQ run $RUN $(date -u) === calibration $(wc -l < /root/work/dwq_calib/train.jsonl) sequences"
  HF=/root/dwq_hf_$RUN; MLX=/root/dwq_mlx4_$RUN; LOG=/root/dwq_$RUN.log
  # a run under a different tag is a superseded configuration, not something to wait for
  if pgrep -af "dwq.p[y]" | grep -qv -- "--ckpt /root/dwq/$RUN.pt"; then
    echo "stopping the run that is not $RUN: $(pgrep -af 'dwq.p[y]' | head -1 | cut -c1-120)"
    pkill -f "dwq.p[y]"; sleep 8; pkill -9 -f "dwq.p[y]" 2>/dev/null; sleep 2
  fi
  RUNNING=0
  pgrep -f "dwq.p[y]" >/dev/null && { RUNNING=1; echo "DWQ already running: $(tail -1 $LOG)"; }
  [ -f $HF/model.safetensors ] && { RUNNING=1; echo "DWQ $RUN already finished"; }
  if [ "$RUNNING" = "0" ]; then
    # Only a launch needs the card to itself. Re-running this file while a measurement is in flight
    # must not kill it - that is how the status command gets fixed without losing an hour of work.
    pkill -f "pool_eval.p[y]"; sleep 8; pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 2
    # The quantizer's own checks first: they cost seconds and they gate an hour of GPU.
    python3 /root/work/q4.py || { echo "Q4 SELFTEST FAILED - not launching"; exit 0; }
    cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/dwq.py \
      --base /root/eval_hf200 --data /root/work/dwq_calib/train.jsonl \
      --out-hf $HF --out-mlx $MLX --ckpt /root/dwq/$RUN.pt --log $LOG \
      --clip-search ${DCLIP:-1} --lr ${DLR:-2e-6} --steps ${DSTEPS:-4000} \
      --accum 4 --len 1024 --kpos 256 --temp ${DTEMP:-1.0} --val 48 --val-every ${DVAL:-100} \
      --policy-only ${DPOL:-0} --focus ${DFOCUS:-0} \
      >> /root/dwq_run_$RUN.log 2>&1 < /dev/null &
    sleep 20
  fi
  # The measurement needs no --q4: the directory already holds the 4-bit values, dequantized.
  cat > /root/dwqkeep.sh <<DKQ
#!/bin/bash
RUN=$RUN; HF=$HF; ET=${DEVALTEMP:-0.9}
DKQ
  cat >> /root/dwqkeep.sh <<'DKQ2'
until [ -s $HF/model.safetensors ] && grep -q DWQ_DONE /root/dwq_run_$RUN.log 2>/dev/null; do sleep 60; done
pkill -f "dwq.p[y]"; sleep 10
python3 /root/work/checkmlx.py ${HF/_hf_/_mlx4_} $HF 2>&1 | tail -8
while :; do
  for i in 0 1 2; do
    want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
    have=$(wc -l < /root/work/${RUN}_out_$i.jsonl 2>/dev/null || echo 0)
    [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
    pgrep -f "pool_eval.py .* /root/work/ev_$i.jsonl" >/dev/null && continue
    echo "[$RUN-eval $(date -u +%H:%M)] shard $i at $have/$want - starting"
    grep -viE "^\s*$" /root/${RUN}_$i.log 2>/dev/null | tail -4 | cut -c1-200
    cd /root/work && SP_BASE=$HF SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py \
      /root/pooler200.safetensors /root/work/ev_$i.jsonl /root/work/${RUN}_out_$i.jsonl \
      --n 999 --rw 768 --maxd 384 --samepage 1 --decode plain --temp $ET --tag "[$RUN$i]" \
      >> /root/${RUN}_$i.log 2>&1 < /dev/null &
    sleep 60
  done
  sleep 120
done
DKQ2
  chmod +x /root/dwqkeep.sh
  setsid nohup bash /root/dwqkeep.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  sleep 40; t; echo "DWQ_LAUNCH_DONE $RUN $(date -u)"
  exit 0
fi

if [ "$MODE" = "publish" ]; then
  # Everything worth keeping leaves the box before it is touched: this host's ssh tunnel never came
  # up, so the next box is a new one, and a stopped instance is not guaranteed to start again.
  pkill -f "afterkee[p].sh"; pkill -f "evalkee[p].sh"; pkill -f "pool_eval.p[y]"; sleep 8
  pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 2
  R=baya1116/hypernet-sp-distill; D=pooler_distill/grpo_pool3_step200_q4
  python3 - <<'PYP' > /root/work/qat_metrics.json
import json, glob, collections, math

def load(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            try: rows.append(json.loads(line))
            except Exception: pass
    if not rows: return None
    by = collections.defaultdict(list)
    for r in rows: by[r.get("q", "")].append(bool(r.get("correct")))
    return {"rollouts": len(rows), "questions": len(by),
            "correct": round(100*sum(1 for r in rows if r.get("correct"))/len(rows), 1),
            "grounded": round(100*sum(1 for r in rows if r.get("grounded"))/len(rows), 1),
            "landed": round(100*sum(1 for r in rows if r.get("landed"))/len(rows), 1),
            "searches_per_rollout": round(sum(r.get("ns", 0) for r in rows)/len(rows), 2),
            "_per_q": {k: sum(v)/len(v) for k, v in by.items()}}

def paired(X, Y):
    if not (X and Y): return None
    c = sorted(set(X["_per_q"]) & set(Y["_per_q"]))
    if not c: return None
    d = [Y["_per_q"][k] - X["_per_q"][k] for k in c]
    m = sum(d)/len(d); sd = (sum((x-m)**2 for x in d)/len(d))**0.5
    return {"delta_pt": round(100*m, 1), "se_pt": round(100*sd/math.sqrt(len(c)), 1), "questions": len(c)}

F, Q, N = load("/root/work/ev_out_*.jsonl"), load("/root/work/q4_out_*.jsonl"), load("/root/work/qa_out_*.jsonl")
out = {
  "what": "Does the phone's 4-bit conversion cost accuracy, and does training against it help?",
  "quantization": {"scheme": "MLX affine, group 64, 4 bits, scales and biases fp16",
                   "modules": "7 projections x 28 blocks + embed_tokens + lm_head = 198, 1776.9M weights",
                   "weight_relative_error": {"projections": 9.51, "embed_tokens": 10.19, "lm_head": 10.35}},
  "training": {"method": "straight-through on the merged weight: forward uses W + (Q(W)-W).detach(), W = base + BA",
               "trainable": "LoRA r32 alpha64 on the 196 projections, 36.9M parameters",
               "objective": "KL to the model's own bf16 self at sampled positions + normalised hidden-state MSE",
               "data": "4344 of the model's own GRPO traces", "steps": 1500, "seconds_per_step": 1.7,
               "kl": {"first_20_steps": 0.0713, "last_20_steps": 0.0229},
               "hidden_mse": {"first_20_steps": 0.0260, "last_20_steps": 0.0110}},
  "heldout": {"file": "eval300.jsonl first 300 lines, 150 distinct questions x 2",
              "settings": {"rw": 768, "maxd": 384, "samepage": 1, "temp": 0.9, "gen": 1500, "maxs": 5, "decode": "plain"},
              "bf16": F and {k: v for k, v in F.items() if k != "_per_q"},
              "4bit_before": Q and {k: v for k, v in Q.items() if k != "_per_q"},
              "4bit_after": N and {k: v for k, v in N.items() if k != "_per_q"}},
  "paired": {"4bit_before_vs_bf16": paired(F, Q), "4bit_after_vs_bf16": paired(F, N),
             "4bit_after_vs_before": paired(Q, N)},
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PYP
  cat /root/work/qat_metrics.json
  echo "--- uploading $(du -shL /root/qat_hf 2>/dev/null | cut -f1) ---"
  hf upload $R /root/qat_hf $D/model 2>&1 | tail -2
  hf upload $R /root/work/qat_metrics.json $D/metrics.json 2>&1 | tail -1
  hf upload $R /root/qat/latest.pt $D/qat_lora.pt 2>&1 | tail -1
  hf upload $R /root/qat.log $D/qat.log >/dev/null 2>&1
  for i in 0 1 2; do
    hf upload $R /root/work/q4_out_$i.jsonl $D/before_shard$i.jsonl >/dev/null 2>&1
    hf upload $R /root/work/qa_out_$i.jsonl $D/after_shard$i.jsonl >/dev/null 2>&1
  done
  # Calibration text for DWQ, in the shape the model actually sees: the harness prompt, the <think>
  # opener, and the model's own trace. mlx-lm reads a folder of jsonl with a "text" field.
  mkdir -p /root/work/dwq_calib
  python3 - <<'PYC'
import json, os
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/root/eval_hf200")
held = set()
for line in open("/root/work/eval300.jsonl"):
    try: held.add(json.loads(line).get("q", ""))
    except Exception: pass
seen, n = set(), 0
with open("/root/work/dwq_calib/train.jsonl", "w") as out:
    for line in open("/root/work/rollouts.jsonl"):
        try: d = json.loads(line)
        except Exception: continue
        q, t = d.get("q", ""), d.get("text", "")
        if not q or not t or q in held:     # the held-out questions never enter any training input
            continue
        head = tok.apply_chat_template([{"role": "user", "content": q}],
                                       add_generation_prompt=True, tokenize=False) + "<think>\n"
        out.write(json.dumps({"text": head + t}, ensure_ascii=False) + "\n"); n += 1
        seen.add(q)
print(f"[calib] {n} sequences over {len(seen)} questions, {len(held)} held-out questions excluded")
PYC
  hf upload $R /root/work/dwq_calib/train.jsonl pooler_distill/dwq_calib/train.jsonl 2>&1 | tail -1
  echo "PUBLISH_DONE $(date -u)"
  exit 0
fi

if [ "$MODE" = "qat" ]; then
  # Quantization-aware training. The held-out measurement is the before-number and stays on disk;
  # what runs now moves the weights so that the same 4-bit conversion stops costing accuracy.
  pkill -f "evalkee[p].sh"; pkill -f "pool_eval.p[y]"; sleep 8
  pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 2
  for i in 0 1 2; do hf upload baya1116/hypernet-sp-distill /root/work/q4_out_$i.jsonl pooler_distill/grpo_pool3_step200/q4_shard$i.jsonl >/dev/null 2>&1; done
  echo "before-number frozen at $(cat /root/work/q4_out_*.jsonl 2>/dev/null | wc -l) rollouts"
  # the model's own traces: on-distribution calibration input, and the only data this needs
  if [ ! -s /root/work/rollouts.jsonl ]; then
    for try in 1 2 3; do hf download baya1116/hypernet-sp-distill --include "pooler_distill/grpo_pool3/rollouts.jsonl" --local-dir /root/hfdl 2>&1 | tail -1 && break; sleep 10; done
    cp /root/hfdl/pooler_distill/grpo_pool3/rollouts.jsonl /root/work/rollouts.jsonl
  fi
  echo "traces: $(wc -l < /root/work/rollouts.jsonl) rollouts"
  RUNNING=0
  pgrep -f "qat.p[y]" >/dev/null && { RUNNING=1; echo "QAT already running: $(tail -1 /root/qat.log)"; }
  [ -f /root/qat_hf/model.safetensors ] && { RUNNING=1; echo "QAT already finished -> /root/qat_hf"; }
  if [ "$RUNNING" = "0" ]; then
  # Two steps first: they print the step-0 loss, which IS the quantization damage while the adapter
  # is still zero, and they prove the memory fits before an hour is committed to it. A crash here is
  # not retried, it is looked at.
  if [ ! -f /root/.qat_selftest_ok ]; then
    rm -f /root/qat.log
    cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/work/qat.py \
      --base /root/eval_hf200 --data /root/work/rollouts.jsonl --selftest 2 2>&1 | tail -22
    grep -q "^step 2 " /root/qat.log 2>/dev/null || { echo "QAT SELFTEST FAILED - not launching"; exit 0; }
    touch /root/.qat_selftest_ok
  fi
  cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/qat.py \
    --base /root/eval_hf200 --data /root/work/rollouts.jsonl --out /root/qat_hf \
    --rank 32 --alpha 64 --lr 1e-4 --steps ${QSTEPS:-1500} --accum 4 --len 640 --kpos 256 \
    >> /root/qat_run.log 2>&1 < /dev/null &
  sleep 20
  fi
  cat > /usr/local/bin/t <<'TTQ'
#!/bin/bash
echo "$(date -u +%H:%M)Z qat $(pgrep -fc 'qat.p[y]')本  $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
grep -E "^\[q4\]|^\[lora\]|^\[data\]" /root/qat_run.log 2>/dev/null | head -5
python3 - <<'PYT' 2>/dev/null
import re
rows=[]
for l in open("/root/qat.log"):
    m=re.match(r"step (\d+) kl=([\d.]+) mse=([\d.]+)", l)
    if m: rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
if rows:
    rows.sort()
    def mean(xs): return sum(xs)/len(xs)
    a, b = rows[:20], rows[-20:]
    print(f"  step {rows[-1][0]}  kl {mean([r[1] for r in a]):.4f} -> {mean([r[1] for r in b]):.4f}"
          f"   hidden mse {mean([r[2] for r in a]):.4f} -> {mean([r[2] for r in b]):.4f}"
          f"   ({100*(1-mean([r[1] for r in b])/max(mean([r[1] for r in a]),1e-9)):.0f}% of the rounding damage removed)")
PYT
tail -2 /root/qat.log 2>/dev/null
python3 - <<'PYQ' 2>/dev/null
import json,glob,collections,math
def summary(pat):
    rows=[]
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            try: rows.append(json.loads(line))
            except Exception: pass
    if not rows: return None
    by=collections.defaultdict(list)
    for r in rows: by[r.get("q","")].append(bool(r.get("correct")))
    return len(rows), 100*sum(1 for r in rows if r.get("correct"))/len(rows), {k:sum(v)/len(v) for k,v in by.items()}
F=summary("/root/work/ev_out_*.jsonl"); Q=summary("/root/work/q4_out_*.jsonl"); N=summary("/root/work/qa_out_*.jsonl")
for tag,S in (("bf16",F),("4bit before",Q),("4bit after",N)):
    if S: print(f"  {tag:12s} {S[1]:5.1f}%  ({S[0]} roll)")
def pair(tag,X,Y):
    if not (X and Y): return
    c=sorted(set(X[2])&set(Y[2]))
    if not c: return
    d=[Y[2][k]-X[2][k] for k in c]; m=sum(d)/len(d)
    sd=(sum((x-m)**2 for x in d)/len(d))**0.5
    print(f"  paired {tag:22s} {100*m:+.1f} pt ±{100*sd/math.sqrt(len(d)):.1f} over {len(c)} questions")
pair("4bit before vs bf16",F,Q); pair("4bit after vs bf16",F,N); pair("4bit after vs before",Q,N)
PYQ
TTQ
  chmod +x /usr/local/bin/t
  cat > /root/status.sh <<'STQ'
#!/bin/bash
t 2>/dev/null
STQ
  chmod +x /root/status.sh
  pkill -f "status_pu[b]"; sleep 1
  cat > /root/status_pub.sh <<'SPQ'
#!/bin/bash
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
while true; do
  { echo "=== $(date -u) === box E (qat)"; t 2>/dev/null; } > /root/work/status.txt 2>&1
  echo "--- STATUS $(date -u +%H:%M) ---"; cat /root/work/status.txt
  sleep 300
done
SPQ
  # When the training finishes cleanly, the same held-out measurement runs against the trained
  # weights, through the same evaluator, the same 150 questions and the same 4-bit grid. This is
  # sequenced rather than retried: it starts only on QAT_DONE plus a saved model, and a shard that
  # dies mid-run is restarted because pool_eval resumes from its own output.
  cat > /root/afterkeep.sh <<'AKQ'
#!/bin/bash
until [ -s /root/qat_hf/model.safetensors ] && grep -q QAT_DONE /root/qat_run.log 2>/dev/null; do sleep 60; done
pkill -f "qat.p[y]"; sleep 10
while :; do
  for i in 0 1 2; do
    want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
    have=$(wc -l < /root/work/qa_out_$i.jsonl 2>/dev/null || echo 0)
    [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
    pgrep -f "pool_eval.py .* /root/work/ev_$i.jsonl" >/dev/null && continue
    echo "[after $(date -u +%H:%M)] shard $i at $have/$want - starting"
    grep -viE "^\s*$" /root/qa_$i.log 2>/dev/null | tail -4 | cut -c1-200
    cd /root/work && SP_BASE=/root/qat_hf SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py \
      /root/pooler200.safetensors /root/work/ev_$i.jsonl /root/work/qa_out_$i.jsonl \
      --n 999 --rw 768 --maxd 384 --samepage 1 --decode plain --q4 1 --tag "[a$i]" \
      >> /root/qa_$i.log 2>&1 < /dev/null &
    sleep 60
  done
  sleep 120
done
AKQ
  chmod +x /root/afterkeep.sh
  pkill -f "afterkee[p].sh"; sleep 1
  setsid nohup bash /root/afterkeep.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  chmod +x /root/status_pub.sh
  setsid nohup bash /root/status_pub.sh >> /proc/1/fd/1 2>&1 < /dev/null &
  sleep 40; t; echo "QAT_LAUNCH_DONE $(date -u)"
  exit 0
fi

# The control loop starts before the assets finish arriving, so wait here rather than exiting: ctl
# only re-runs this file when its content changes, and a one-line "not ready" would be the last thing
# the box ever said.
for w in $(seq 1 40); do
  [ -s /root/eval_hf200/model.safetensors ] && [ -s /root/pooler200.safetensors ] && break
  echo "[wait $w] downloading: $(du -sh /root/hfdl 2>/dev/null | cut -f1) in /root/hfdl, $(tail -1 /root/boot.log 2>/dev/null | cut -c1-100)"
  sleep 60
done
if [ ! -s /root/eval_hf200/model.safetensors ]; then echo "NOT READY: /root/eval_hf200 never arrived"; exit 0; fi
if [ ! -s /root/pooler200.safetensors ]; then echo "NOT READY: pooler never arrived"; exit 0; fi
if [ ! -s /root/work/eval300.jsonl ]; then echo "NOT READY: questions never arrived"; exit 0; fi

# thirty seconds of checking the quantizer is cheaper than two hours of measuring with a broken one
python3 /root/work/q4.py || { echo "Q4 SELFTEST FAILED - not launching"; exit 0; }

# the same first 300 lines every earlier number on this file came from
if [ ! -f /root/work/ev_0.jsonl ]; then
  python3 - <<'PY'
import json
qs=[l for l in open("/root/work/eval300.jsonl") if l.strip()][:300]
S=3
for i in range(S): open(f"/root/work/ev_{i}.jsonl","w").writelines(qs[i::S])
print(f"[shard] first 300 lines, {len({json.loads(l).get('q','') for l in qs})} distinct questions -> {[len(qs[i::S]) for i in range(S)]}")
PY
fi
[ -f /root/.q4_t0 ] || date +%s > /root/.q4_t0

# One-shot: the first launch did the quantization arithmetic on the GPU, which left about 9.5GB
# reserved per process and let only two of the three shards onto the card - the third OOMed and the
# watchdog kept retrying it. The quantizer now works on the CPU in row blocks, so restart the shards
# once to pick that up. pool_eval resumes from its own output, so only the question in flight is lost.
if [ ! -f /root/.q4_cpuquant ]; then
  touch /root/.q4_cpuquant
  pkill -f "evalkee[p].sh"; pkill -f "pool_eval.p[y]"; sleep 10
  pkill -9 -f "pool_eval.p[y]" 2>/dev/null; sleep 3
  echo "shards stopped; they come back on the CPU-side quantizer ($(cat /root/work/q4_out_*.jsonl 2>/dev/null | wc -l) rollouts kept)"
fi

# ctl only re-runs this file when its content changes, so a shard that dies would otherwise sit dead
# until the next edit. This watchdog restarts one when its process is gone and its output is short;
# pool_eval resumes by skipping the questions already in its own output.
cat > /root/evalkeep.sh <<'KG'
#!/bin/bash
while :; do
  for i in 0 1 2; do
    want=$(wc -l < /root/work/ev_$i.jsonl 2>/dev/null || echo 0)
    have=$(wc -l < /root/work/q4_out_$i.jsonl 2>/dev/null || echo 0)
    [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && continue
    pgrep -f "pool_eval.py .* /root/work/ev_$i.jsonl" >/dev/null && continue
    echo "[keep $(date -u +%H:%M)] shard $i is dead at $have/$want - restarting"
    grep -viE "^\s*$" /root/q4_$i.log 2>/dev/null | tail -4 | cut -c1-200
    cd /root/work && SP_BASE=/root/eval_hf200 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 OMP_NUM_THREADS=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 /root/work/pool_eval.py \
      /root/pooler200.safetensors /root/work/ev_$i.jsonl /root/work/q4_out_$i.jsonl \
      --n 999 --rw 768 --maxd 384 --samepage 1 --decode plain --q4 1 --tag "[q$i]" \
      >> /root/q4_$i.log 2>&1 < /dev/null &
    sleep 60          # stagger the model loads; three landing together is what killed a shard last time
  done
  sleep 120
done
KG
chmod +x /root/evalkeep.sh
pkill -f "evalkee[p].sh"; sleep 1
setsid nohup bash /root/evalkeep.sh >> /proc/1/fd/1 2>&1 < /dev/null &

cat > /usr/local/bin/t <<'TT'
#!/bin/bash
python3 - <<'PY'
import json,glob,os,time,math,collections

def read(pat):
    rows=[]
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows

def summary(rows):
    n=len(rows)
    if not n: return None
    by=collections.defaultdict(list)
    for r in rows: by[r.get("q","")].append(bool(r.get("correct")))
    qm={k:sum(v)/len(v) for k,v in by.items()}
    mu=sum(qm.values())/len(qm)
    sd=(sum((x-mu)**2 for x in qm.values())/len(qm))**0.5
    def per_q(key):
        d=collections.defaultdict(list)
        for r in rows: d[r.get("q","")].append(float(r.get(key) or 0))
        return {k:sum(v)/len(v) for k,v in d.items()}
    return dict(n=n, q=len(qm), qm=qm, gq=per_q("grounded"), sq=per_q("ns"),
                c=100*sum(1 for r in rows if r.get("correct"))/n,
                g=100*sum(1 for r in rows if r.get("grounded"))/n,
                l=100*sum(1 for r in rows if r.get("landed"))/n,
                s=sum(r.get("ns",0) for r in rows)/n,
                se=100*sd/math.sqrt(len(qm)))

alive=os.popen("pgrep -fc 'pool_eval.p[y]'").read().strip() or "0"
qh=os.popen("grep '^\\[q4\\]' /root/q4_0.log 2>/dev/null | head -2").read().rstrip()
t0=0
try: t0=int(open("/root/.q4_t0").read().strip())
except Exception: pass
Q=summary(read("/root/work/q4_out_*.jsonl")); F=summary(read("/root/work/ev_out_*.jsonl"))
nq=Q["n"] if Q else 0
el=max(time.time()-t0,1) if t0 else 0
eta=f"  残り~{(300-nq)/(nq/el)/60:.0f}分" if nq and el and nq<300 else ""
print(f"{time.strftime('%H:%M',time.gmtime())}Z q4eval {alive}本  {nq}/300{eta}")
if qh: print("  "+qh.replace("\n","\n  "))
for tag,S in (("bf16",F),("4bit",Q)):
    if S: print(f"  {tag}  correct {S['c']:5.1f}% ±{S['se']:.1f}  gnd {S['g']:3.0f}%  land {S['l']:3.0f}%  srch {S['s']:.1f}  ({S['n']} roll / {S['q']} q)")
if Q and F:
    common=sorted(set(Q["qm"]) & set(F["qm"]))
    if common:
        def pair(key, scale, unit):
            d=[Q[key][k]-F[key][k] for k in common]
            m=sum(d)/len(d); sd=(sum((x-m)**2 for x in d)/len(d))**0.5
            return f"{scale*m:+.1f}{unit} ±{scale*sd/math.sqrt(len(d)):.1f}"
        print(f"  paired 4bit-bf16 over {len(common)} shared questions: "
              f"correct {pair('qm',100,'pt')}  gnd {pair('gq',100,'pt')}  srch {pair('sq',1,'')}")
PY
TT
chmod +x /usr/local/bin/t
cat > /root/status.sh <<'ST'
#!/bin/bash
t 2>/dev/null
ST
chmod +x /root/status.sh
pkill -f "status_pu[b]"; sleep 1
cat > /root/status_pub.sh <<'SP2'
#!/bin/bash
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null)
while true; do
  { echo "=== $(date -u) === box E (4-bit held-out)"; t 2>/dev/null; } > /root/work/status.txt 2>&1
  echo "--- STATUS $(date -u +%H:%M) ---"; cat /root/work/status.txt
  for i in 0 1 2; do hf upload baya1116/hypernet-sp-distill /root/work/q4_out_$i.jsonl pooler_distill/grpo_pool3_step200/q4_shard$i.jsonl >/dev/null 2>&1; done
  sleep 300
done
SP2
chmod +x /root/status_pub.sh
setsid nohup bash /root/status_pub.sh >> /proc/1/fd/1 2>&1 < /dev/null &
sleep 30; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader; t
echo "Q4EVAL_LAUNCH_DONE $(date -u)"
# CTL-END
