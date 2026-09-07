# box C: upload everything that only exists on this box to HF (baya1116/hypernet-sp-distill/pooler_distill/).
# Needs a write token at /root/.hf_token (user puts it there over ssh). Idempotent: hf upload skips identical files.
cd /root/work
if [ ! -s /root/.hf_token ]; then echo "NO_HF_TOKEN at /root/.hf_token"; exit 0; fi
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token)
pip install -q -U huggingface_hub >/dev/null 2>&1; hf --version 2>/dev/null || python3 -m pip install -q "huggingface_hub[cli]" >/dev/null 2>&1
R=baya1116/hypernet-sp-distill; D=pooler_distill
cat > /root/work/NOTES_pooler_distill.txt <<'NT'
pooler_distill: self-distillation of the pooler lineage (fft_final + SP pooler) from the GRPO teacher ckpt_mus200 (mus_run).
fft_new_all.safetensors  = fft_final SFT'd one pass over every landed teacher trajectory (1092 rows, failures included), pooler included. 136 steps.
fft_new_stream.safetensors = earlier variant: one pass over the 499 correct&grounded rows only.
gen_mus200_full.jsonl    = teacher (ckpt_mus200) trajectories on 1276 training-pool questions (held-out 300 excluded).
pooleval_all_c.jsonl     = fft_new_all, held-out first 150 q, SP compression (raw window 768, maxd 384), plain decoding: 32.7%
pooleval_all_nc.jsonl    = fft_new_all, same 150 q, no compression (rw 8000):                                        36.7%
teacher on the same 150 q (eval_step200, no compression): 37.7%
pooleval_post.jsonl      = fft_new_stream, first 114 q, rw 512, harness guard decoding (rep-penalty/no-repeat): 23.7%
sft_all.log / sft_loss_all.log = the all-trajectory SFT logs. Evaluator: deep-charger ctl/pool_eval.py.
NT
up() { for i in 1 2 3; do hf upload "$R" "$1" "$D/$2" >/dev/null 2>&1 && { echo "UPLOADED $2"; return 0; }; sleep 20; done; echo "UPLOAD_FAILED $2"; }
up /root/work/NOTES_pooler_distill.txt NOTES.txt
up /root/work/gen_mus200_full.jsonl gen_mus200_full.jsonl
up /root/work/pooleval_all_c.jsonl pooleval_all_c.jsonl
up /root/work/pooleval_all_nc.jsonl pooleval_all_nc.jsonl
up /root/work/pooleval_post.jsonl pooleval_post.jsonl
up /root/sft_all.log sft_all.log
up /root/sft_loss_all.log sft_loss_all.log
up /root/fft_new_all.safetensors fft_new_all.safetensors
up /root/fft_new_stream.safetensors fft_new_stream.safetensors
echo "HF_UPLOAD_DONE $(date -u)"
