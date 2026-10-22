#!/bin/bash
# Box VO (3060, pytorch image): the student's own successes, the lineage's s1 recipe and the NPC one (teach1 copied a
# stronger model's two-search habit and lost 7 points). (A) 2.1.0 answers the use-case bench (301 real prompts) for the
# judge against 4o-mini; (B) 2.1.0 on 600 training questions, one sampled rollout each; (C) the rollouts that are right
# AND grounded learned back into 2.1.0 (memfit ce, the served pages masked, LoRA, mem10's settings) -> rft1;
# (D) rft1 on the search held-out (eval300, two passes) and on the use-case bench. Results to archive/rft/ and archive/uc/.
BOXVO_SERIAL=1
if [ -f /root/.boxvo_serial ] && [ "$(cat /root/.boxvo_serial)" -gt "$BOXVO_SERIAL" ] 2>/dev/null; then echo "BOXVO_STALE $BOXVO_SERIAL"; exit 0; fi
echo $BOXVO_SERIAL > /root/.boxvo_serial
if [ ! -e /root/.vo_pre ]; then touch /root/.vo_pre
  echo "[vo] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader)"
fi
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
mkdir -p /root/work/localsearch /root/work/fft_out /root/work/runtime
for f in pool_eval.py memfit.py web_search.py localsearch/search.py localsearch/store.py localsearch/ivf.py localsearch/terms.py localsearch/pq.py localsearch/embed.py; do
  for try in 1 2 3; do curl -sS -o /root/work/$f "$RAW/$f?nocache=$(date +%s)" && break; sleep 5; done
done
cp /root/work/web_search.py /root/work/runtime/web_search.py 2>/dev/null; touch /root/work/runtime/__init__.py
[ -f /root/.bootstrapped ] || { python3 -m pip install -q "transformers>=4.51,<4.57" accelerate "huggingface_hub[hf_xet]>=0.34,<1.0" safetensors peft zstandard onnxruntime datasets certifi 2>&1 | tail -1; touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"; }
cat > /root/status.sh <<'ST'
grep -E "^\[(vo|hub)\]|MEMFIT_DONE|EVAL_DONE|Error|Traceback|out of memory" /root/vo.log 2>/dev/null | tail -n 4 | cut -c1-300
for f in /root/work/uc_dc210.jsonl /root/work/tr_a.jsonl /root/work/tr_b.jsonl /root/work/rft1_ev0.jsonl /root/work/rft1_ev1.jsonl /root/work/uc_rft1.jsonl; do [ -s $f ] && echo "$(basename $f): $(wc -l < $f) rows"; done
grep -E "^val" /root/memfit_rft1.log 2>/dev/null | tail -n 1
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh
if [ ! -e /root/.vojob ]; then touch /root/.vojob
  cat > /root/vojob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/work
R=baya1116/hypernet-sp-distill; H=https://huggingface.co/$R/resolve/main
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/'+sys.argv[2], repo_id='$R')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
get() { [ -s $2 ] || curl -sSfL -o $2 "$1" || { rm -f $2; echo "[vo] could not get $1"; return 1; }; }
for f in grpo_e2e_torch.py pooler_torch.py; do get $H/box_recover/scripts/$f /root/work/$f; done
python3 - <<'PY'   # transformers 4.56+ rejects a positive crop (keep N); the harness's crop_cache made version-proof
import re
p = "/root/work/grpo_e2e_torch.py"; s = open(p).read()
new = ('def crop_cache(past, length):\n    if hasattr(past, "crop"):\n        L = past.get_seq_length()\n        if L > length: past.crop(length - L)\n        return past\n')
s2 = re.sub(r'def crop_cache\(past, length\):\n(    .*\n)+?        past\.crop\(length\); return past\n', new, s, count=1)
print("[patch] crop_cache", "patched" if s2 != s else "NOT FOUND"); open(p, "w").write(s2)
PY
get $H/fft_out/pooler.pt /root/work/fft_out/pooler.pt
mkdir -p /root/gptq_hf_gq14; for f in config.json generation_config.json special_tokens_map.json tokenizer.json tokenizer_config.json model.safetensors; do get $H/versions/2.0.0/base/$f /root/gptq_hf_gq14/$f; done
get $H/versions/2.1.0/lora.safetensors /root/mtg7_s40.safetensors
get $H/grpo_assets/mus_run/eval_step200/eval_heldout_300q_g2.jsonl /root/work/eval300.jsonl
get $H/box_recover/corpus.jsonl /root/work/corpus.jsonl; get $H/archive/uc/uc_bench.jsonl /root/work/uc_bench.jsonl
mkdir -p /root/wiki_store /root/bge-small
for f in blocks.idx titles.txt meta.json ivf_centroids.npy ivf_offsets.npy ivf_order.npy emb.bin emb_ivf.bin lex.sqlite docs.bin; do get $H/localsearch/wiki_en_20231101/$f /root/wiki_store/$f; done
for f in config.json model_int8.onnx special_tokens_map.json tokenizer.json tokenizer_config.json; do get $H/localsearch/bge-small-en-v1.5/$f /root/bge-small/$f; done
python3 - <<'PY'
import json, random
rows = [json.loads(l) for l in open("/root/work/corpus.jsonl") if l.strip()]; ex = set()
for l in open("/root/work/eval300.jsonl"):
    try: ex.add(json.loads(l)["q"])
    except Exception: pass
items = [(r["q"], r["gold"].strip()) for r in rows if r.get("q") and (r.get("gold") or "").strip() and len(r["gold"].split()) <= 6 and r["q"] not in ex]
random.Random(0).shuffle(items); items = items[:600]
open("/root/work/tr_qa.jsonl", "w").writelines(json.dumps({"q": q, "gold": g}) + "\n" for q, g in items[:450])
open("/root/work/tr_qb.jsonl", "w").writelines(json.dumps({"q": q, "gold": g}) + "\n" for q, g in items[450:])
print(f"[vo] training questions: 450 + 150 written", flush=True)
PY
echo "[vo] assets in place $(date -u)"
ENV="SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 SP_LOCAL_STORE=/root/wiki_store SP_LOCAL_MODEL=/root/bge-small OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
EVARGS="--rw 768 --maxd 384 --samepage 1 --decode plain --temp 0.6 --gen 4000 --stop eos --replycap 600"
( while sleep 600; do for f in uc_dc210 tr_a tr_b; do [ -s /root/work/$f.jsonl ] && put /root/work/$f.jsonl rft/$f.jsonl > /dev/null; done; done ) & MIR=$!
# (A)+(B): two processes - the use-case bench then 150 training questions; 450 training questions
( env $ENV python3 /root/work/pool_eval.py /root/mtg7_s40.safetensors /root/work/uc_bench.jsonl /root/work/uc_dc210.jsonl $EVARGS --n 400 --tag "[uc dc210]" 2>&1 | grep -E "EVAL_DONE|Traceback|Error" | sed "s/^/[vo] /"
  env $ENV python3 /root/work/pool_eval.py /root/mtg7_s40.safetensors /root/work/tr_qb.jsonl /root/work/tr_b.jsonl $EVARGS --n 400 --tag "[tr b]" 2>&1 | grep -E "EVAL_DONE|Traceback|Error" | sed "s/^/[vo] /" ) &
sleep 90
( env $ENV python3 /root/work/pool_eval.py /root/mtg7_s40.safetensors /root/work/tr_qa.jsonl /root/work/tr_a.jsonl $EVARGS --n 500 --tag "[tr a]" 2>&1 | grep -E "EVAL_DONE|Traceback|Error" | sed "s/^/[vo] /" ) &
wait $(jobs -p | grep -v $MIR) 2>/dev/null; kill $MIR 2>/dev/null
put /root/work/uc_dc210.jsonl uc/uc_dc210.jsonl; for f in tr_a tr_b; do put /root/work/$f.jsonl rft/$f.jsonl; done
# (C) the student's own right-and-grounded rollouts -> memfit rows (the prompt as pool_eval renders it, the trace as generated)
python3 - <<'PY'
import json
from tokenizers import Tokenizer
tok = Tokenizer.from_file("/root/gptq_hf_gq14/tokenizer.json"); enc = lambda t: tok.encode(t, add_special_tokens=False).ids
bos = tok.token_to_id("<｜begin▁of▁sentence｜>"); n = k = 0
with open("/root/work/rft_rows.jsonl", "w") as f:
    for fn in ("/root/work/tr_a.jsonl", "/root/work/tr_b.jsonl"):
        for l in open(fn):
            r = json.loads(l); n += 1
            if not (r.get("correct") and r.get("grounded") and r.get("landed")): continue
            q_ids = [bos] + enc("<｜User｜>" + r["q"] + "<｜Assistant｜><think>\n"); gen = enc(r["text"])
            f.write(json.dumps({"q": r["q"], "gold": r["gold"], "dialog": f"r{k}", "turn": 0, "q_ids": q_ids, "gen": gen, "correct": True, "ns": r.get("ns")}) + "\n"); k += 1
print(f"[vo] the student's own successes: {k} of {n} rollouts right and grounded -> rft_rows.jsonl", flush=True)
PY
put /root/work/rft_rows.jsonl rft/rft_rows.jsonl
export SP_BASE=/root/gptq_hf_gq14 SP_RANK=16 SP_NOSYS=1 SP_EPISODIC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 /root/work/memfit.py --objective ce --ce-native 1 --student full --ckpt /root/mtg7_s40.safetensors --data /root/work/rft_rows.jsonl --out /root/rft1.safetensors --log /root/memfit_rft1.log --steps 600 --val-every 100 --lr 0 --lr-lora 5e-6 > /root/memfit_rft1_run.log 2>&1
grep -E "MEMFIT_DONE|Traceback|Error" /root/memfit_rft1_run.log | tail -2 | sed "s/^/[vo] /"; put /root/memfit_rft1.log rft/memfit_rft1.log
[ -s /root/rft1.safetensors ] || { echo "[vo] no rft1 checkpoint"; date -u > /root/DONE_VO; put /root/DONE_VO rft/DONE; exit 1; }
put /root/rft1.safetensors rft/rft1.safetensors
# (D) rft1: the held-out (two passes) and the use-case bench
( while sleep 600; do for f in rft1_ev0 rft1_ev1 uc_rft1; do [ -s /root/work/$f.jsonl ] && put /root/work/$f.jsonl rft/$f.jsonl > /dev/null; done; done ) & MIR=$!
( for i in 0 1; do env $ENV python3 /root/work/pool_eval.py /root/rft1.safetensors /root/work/eval300.jsonl /root/work/rft1_ev$i.jsonl $EVARGS --n 150 --tag "[rft1 ev$i]" 2>&1 | grep -E "EVAL_DONE|Traceback|Error" | sed "s/^/[vo] /"; done ) &
sleep 90
( env $ENV python3 /root/work/pool_eval.py /root/rft1.safetensors /root/work/uc_bench.jsonl /root/work/uc_rft1.jsonl $EVARGS --n 400 --tag "[uc rft1]" 2>&1 | grep -E "EVAL_DONE|Traceback|Error" | sed "s/^/[vo] /" ) &
wait $(jobs -p | grep -v $MIR) 2>/dev/null; kill $MIR 2>/dev/null
for f in rft1_ev0 rft1_ev1; do put /root/work/$f.jsonl rft/$f.jsonl; done; put /root/work/uc_rft1.jsonl uc/uc_rft1.jsonl
put /root/vo.log rft/run.log; date -u > /root/DONE_VO; put /root/DONE_VO rft/DONE; echo "[vo] VOJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vojob.sh >> /root/vo.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VOJOB_LAUNCHED $(date -u)"
fi
# CTL-END
