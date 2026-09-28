# Release: the search-and-reason model, 2026-09-28

One 1.5B model (DeepSeek-R1-Distill-Qwen-1.5B lineage) that reasons, chats, and searches when it needs to,
with the pooler that compresses what it has read. Two directories, the same model:

| directory | what | size | search held-out (same 300 rollouts) |
|---|---|---|---|
| `g14-bf16/` | the merged 16-bit weights + `pooler.safetensors` | 3.5 GB + 0.3 GB | 45.3% |
| `g14-4bit-gptq/` | MLX affine 4-bit, group 64, codes chosen by GPTQ on the model's own traces + pooler - **the one the app ships** | 1.0 GB + 0.3 GB | 45.0% |

Held-out: eval300 subset, temperature 0.6, 4000 tokens, the model's own pooler, Wikipedia search.
The 4-bit GPTQ directory is indistinguishable from the 16-bit weights on it (paired difference
-0.3 +- 2.9 over 150 questions); round-to-nearest 4-bit lost about ten points. Other 4-bit arms
(`pooler_distill/chatsft/g14_mlx4*`) are experiments, not releases, until they are measured the same way. Reasoning (Dolphin-R1
held-out, 100): 53%, the level of the untouched distill (56%).

Lineage: chat SFT (s4) -> search GRPO on the pooler-equipped student (g10, layers 20-27, 400 steps)
-> distillation of R1 thinking and replies with search-trace replay (g14, 835 steps). Details in the
repository's `docs/conversational_lineage.md` and `docs/quantization_4bit.md`.

Loading: the 4-bit directories are what the app reads (`SPGenerator` with `QuantizationConfig(groupSize: 64,
bits: 4)`); `pooler.safetensors` holds the pooler's 64 tensors under bare keys. The 16-bit directory is a
plain `transformers` model directory; the same `pooler.safetensors` goes with it.

Local search (no network): `localsearch/wiki_en_20231101/` (the store, the sign index and its IVF layout,
the lexical index) with `localsearch/bge-small-en-v1.5/` (the int8 embedder); `docs/local_search.md`.
