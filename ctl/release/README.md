# Release: the search-and-reason model, 2026-09-28

How to run it on a device (files, prompt, the pooler loop, sampling, the search protocol): `USAGE.md`.

One 1.5B model (DeepSeek-R1-Distill-Qwen-1.5B lineage) that reasons, chats, and searches when it needs to,
with the pooler that compresses what it has read. Three directories, the same model:

| directory | what | size | search held-out (same 300 rollouts) |
|---|---|---|---|
| `g14-bf16/` | the merged 16-bit weights + `pooler.safetensors` | 3.5 GB + 0.3 GB | 45.3% |
| `g14-4bit-gptq/` | MLX affine 4-bit, group 64, codes chosen by GPTQ on the model's own traces + pooler | 1.0 GB + 0.3 GB | 45.0% |
| `g14-4bit-gptq-trained/` | the same codes, the grid's scales and biases then trained on the model's own traces + pooler - **the one the app ships** | 1.0 GB + 0.3 GB | 47.3% |

Held-out: eval300 subset, temperature 0.6, 4000 tokens, the model's own pooler, Wikipedia search.
Paired over the same 150 questions: GPTQ against 16-bit -0.3 +- 2.9, GPTQ with the trained grid
against 16-bit +2.0 +- 3.6 (better on 38 questions, worse on 34, equal on 78). Both 4-bit directories are
indistinguishable from the 16-bit weights; round-to-nearest 4-bit lost about ten points. Other 4-bit arms
(`pooler_distill/chatsft/g14_mlx4*`) are experiments, not releases. Reasoning (Dolphin-R1
held-out, 100): 53%, the level of the untouched distill (56%).

Lineage: chat SFT (s4) -> search GRPO on the pooler-equipped student (g10, layers 20-27, 400 steps)
-> distillation of R1 thinking and replies with search-trace replay (g14, 835 steps). Details in the
repository's `docs/conversational_lineage.md` and `docs/quantization_4bit.md`.

Loading: the 4-bit directories are what the app reads (`SPGenerator` with `QuantizationConfig(groupSize: 64,
bits: 4)`); `pooler.safetensors` holds the pooler's 64 tensors under bare keys. The 16-bit directory is a
plain `transformers` model directory; the same `pooler.safetensors` goes with it.

Local search (no network): `release/local-search/` - the store, the sign index and its IVF layout, the
lexical index, the int8 embedder and the code, with its own README and the measurements (4-bit model with
it: 48.0% at 102 rollouts, 42.3% at 300, against the API's 48.0 / 45.0); `docs/local_search.md`.
