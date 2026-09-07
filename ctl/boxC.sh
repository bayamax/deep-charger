# box C: finalize the streaming SFT once A has shipped its last rows.
#  1) wait for /root/work/gen_mus200_full.jsonl (A ships it after the final harvest sync)
#  2) let the trainer consume the tail, then kill it, rewind fft_seen to the last saved step (SAVE_EVERY=25 -> step*8 rows)
#  3) relaunch with a short IDLE_EXIT so it re-trains only the unsaved tail and writes the DONE checkpoint
#  4) export a plain HF model dir (pooler tensors dropped; eval protocol never used the pooler at inference)
for i in $(seq 1 120); do [ -f /root/work/gen_mus200_full.jsonl ] && break; sleep 30; done
[ -f /root/work/gen_mus200_full.jsonl ] || { echo "A did not ship full jsonl in 60 min"; exit 0; }
echo "full jsonl present: $(wc -l < /root/work/gen_mus200_full.jsonl) rows; harvest $(wc -l < /root/harvest_sft/gen_mus200.jsonl) rows"
for i in $(seq 1 40); do S=$(wc -l < /root/fft_seen_stream.txt); sleep 30; S2=$(wc -l < /root/fft_seen_stream.txt); [ "$S" = "$S2" ] && grep -q "idle: scan() empty" <(tail -3 /root/sft_stream.log) && break; done
LAST=$(awk 'NF==5 && $1!="#"{s=$1} END{print s+0}' /root/sft_loss_stream.log); SAVED=$(( LAST / 25 * 25 )); echo "trainer at step $LAST, last saved step $SAVED, seen $(wc -l < /root/fft_seen_stream.txt)"
pkill -f "sft_pool_ru[n].py"; sleep 8
head -n $(( SAVED * 8 )) /root/fft_seen_stream.txt > /root/fft_seen_stream.trim && mv /root/fft_seen_stream.trim /root/fft_seen_stream.txt; echo "seen rewound to $(wc -l < /root/fft_seen_stream.txt)"
FREEZE=0 GCKPT=0 IDLE_EXIT=180 bash /root/do_sft_pool.sh stream
for i in $(seq 1 60); do grep -q "^\[fft\] DONE" /root/sft_stream.log && break; sleep 30; done
grep "^\[fft\] DONE\|^\[fft\] step" /root/sft_stream.log | tail -3
ls -la /root/fft_new_stream.safetensors
python3 - <<'PY'
import torch, os
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
sd=load_file("/root/fft_new_stream.safetensors"); md={k:v for k,v in sd.items() if not k.startswith("pooler.")}
m=AutoModelForCausalLM.from_pretrained("/root/fft_hf", torch_dtype=torch.bfloat16)
r=m.load_state_dict(md, strict=False); print("missing", len(r.missing_keys), "unexpected", len(r.unexpected_keys))
m.save_pretrained("/root/work/sft_stream_hf", safe_serialization=True); AutoTokenizer.from_pretrained("/root/fft_hf").save_pretrained("/root/work/sft_stream_hf")
print("EXPORTED /root/work/sft_stream_hf")
PY
du -sh /root/work/sft_stream_hf; echo "C_FINALIZE_DONE $(date -u)"
