# GPT-2 (124M) — 4-bit GPTQ, MLX

- Blocks (attention and MLP matrices): 4-bit affine, group 64, codes chosen by GPTQ on WikiText-103 (64 × 1024 tokens).
- Token table (tied input/output): 8-bit affine, group 64 — at 4 bits it carried almost all of the loss.
- Position table, LayerNorms, scales and biases: fp32 (GPT-2's residual has outliers in the thousands; fp16 compute drifts).
- Size: 100 MB (original fp32 523 MB).

| WikiText-2 test, 1024-token windows, stride 512 | perplexity | next-token top-1 |
|---|---|---|
| original, PyTorch fp32 (all windows) | 25.17 | 41.2% |
| round-to-nearest 4-bit, everything (all windows) | 405.26 | 21.4% |
| GPTQ 4-bit, everything (all windows) | 36.09 | 35.7% |
| original, PyTorch fp32 (first 120 windows) | 26.81 | |
| this model's weights, PyTorch (first 120 windows) | 27.76 | |
| original, MLX fp16 (first 20 windows) | 27.85 | 39.7% |
| **this model, MLX (first 20 windows)** | **28.83** | **39.2%** |

    pip install mlx-lm
    python -m mlx_lm generate --model ./gpt2-4bit-gptq-e8 --prompt "The capital of France is" --max-tokens 40

Made with ctl/gpt2_gptq.py then ctl/gpt2_embed8.py.
