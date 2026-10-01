# Using the model on the device

What ships: `release/g14-4bit-gptq-trained/` (the 4-bit model, 1.0 GB) with its pooler - the 4-bit
`pooler_4bit.safetensors` (43 MB, section 1a; measured as good as the float one) or the float
`pooler.safetensors` (0.3 GB) - and `pooler_config.json`, plus `release/local-search/` (4.4 GB) when the app searches without a
network. The 16-bit directory is the same model for a Mac or a server. Everything below is what the
evaluation harness does (`ctl/pool_eval.py` in the repository) and what the Swift port
(`Sources/HypernetSP/SPGenerator.swift`, `Pooler.swift`) mirrors; the Japanese implementation spec with the
loop written out is `docs/iphone_agent_v2_spec.md`.

## 1. Files

| file | what |
|---|---|
| `model.safetensors`, `config.json` | Qwen2 1.5B, MLX affine quantization, `{"group_size": 64, "bits": 4, "mode": "affine"}` in `config.json`; loads with MLX (`QuantizationConfig(groupSize: 64, bits: 4)`) or `mlx_lm` |
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` | the DeepSeek-R1-Distill tokenizer and chat template |
| `pooler.safetensors` | the pooler, 64 tensors under bare keys (`query`, `blocks.{0,1,2}.*`, `ln_out.*`, `out_scale`), float32 |
| `pooler_4bit.safetensors` | the same pooler on the model's 4-bit grid, 43 MB - section 1a |
| `pooler_config.json` | `hidden_dim 1536, num_soft_tokens 32, heads 8, layers 3, ffn 2048, out_scale 1.4687` (the trained `out_scale`, also stored in the weights) |

The pooler is shared by the three model directories: it was trained with the 16-bit model and kept
through quantization.

## 1a. The 4-bit pooler (`pooler_4bit.safetensors`, added 2026-10-02)

The pooler's 18 weight matrices on the same grid as the model (MLX affine, 4 bits, group 64, fp16 scale and
bias per group), the codes chosen by GPTQ on the pooler's real inputs (the 4-bit model's embeddings of past
trace tokens, half search traces, half reasoning traces). 43 MB instead of 302 MB; everything else is unchanged
(same config, same loop, the pooler still computes in float32).

Contents, 100 tensors under the same bare keys as `pooler.safetensors`:

- the 18 matrices `blocks.{0,1,2}.{cross,selfa}.in_proj_weight` [4608, 1536], `blocks.{0,1,2}.{cross,selfa}.out_proj.weight`
  [1536, 1536], `blocks.{0,1,2}.ffn.0.weight` [2048, 1536], `blocks.{0,1,2}.ffn.2.weight` [1536, 2048]:
  each stored as `K` uint32 [rows, cols/8] (MLX layout: 8 codes per word, the first column in the lowest 4
  bits) with `K.scales` and `K.biases` float16 [rows, cols/64]. The weight is `code * scale + bias`, per
  row and per group of 64 input columns.
- the other 46 tensors (`query`, the layernorms, every bias, `out_scale`) as float32, exactly as evaluated.

Loading (MLX / Swift): `dequantized(K, scales: K.scales, biases: K.biases, groupSize: 64, bits: 4)` gives
the float matrix; dequantize once at load and the pooler code does not change (0.3 GB resident in float32,
0.15 GB in float16), or keep the codes and use `quantizedMatmul(x, K, scales:, biases:, transpose: true,
groupSize: 64, bits: 4)` for `x @ K.T` (43 MB resident). The packed `in_proj_weight` is quantized row by
row, so its q / k / v thirds (rows 0-1535, 1536-3071, 3072-4607) are row slices of `K`, `K.scales` and
`K.biases`. In Python: `mx.dequantize(w, scales, biases, group_size=64, bits=4)`, or with numpy
`((w[..., None] >> np.arange(0, 32, 4)) & 15).reshape(rows, cols) * repeat(scales, 64) + repeat(biases, 64)`.
The unpacked file reproduces the evaluated weights exactly.

Measured (the 4-bit model, this loop, the same harness as section 7):

| | float pooler | 4-bit pooler |
|---|---|---|
| output error on held-out windows (relative, against the input-driven part) | - | 0.36% (search 0.32, reasoning 0.41; round-to-nearest: 5.1%) |
| search held-out, 102 rollouts (eval300 shards 0-2) | 47.3% at 300 rollouts | 52.9% (55.9 / 64.7 / 38.2) |
| reasoning held-out (Dolphin-R1, 100), two draws each | 54, 48 = 102 / 200 | 47, 51 = 98 / 200 |

On the reasoning held-out the float pooler run twice disagrees with itself on 24 questions; float against
4-bit disagrees on 27 (float better on 20, 4-bit on 18): the 4-bit pooler's difference is the size of one
more draw. Variants that did not help (each 100 questions): the float pooler's empty-past output with
the 4-bit pooler 51, the grid's scales and biases trained against the float pooler 52. Details:
`docs/quantization_4bit.md` in the repository; the files under `pooler_distill/pooler4bit/` on the hub.

## 2. Prompt

No system prompt. The user turn through the chat template, generation prompt added, then `<think>\n`
appended so the model always thinks first:

```
<｜begin▁of▁sentence｜><｜User｜>{question}<｜Assistant｜><think>\n
```

Multi-turn chat: the previous turns as the template renders them (`<｜end▁of▁sentence｜>` closes an
assistant turn), the new user turn, `<｜Assistant｜><think>\n`.

## 3. The pooler loop (the model's context is a soft prompt plus a raw window)

Constants the numbers in this repository were measured with: `RW = 768` (raw window, tokens kept
verbatim), `MAXD = 384` (most past tokens handed to the pooler), `CHUNK = 128` (tokens generated between
rebuilds), `GEN = 4000` (model tokens per reply, injected text not counted), reply cap 600 tokens after
`</think>`, temperature 0.6, stop at end of sequence.

```
prefill the prompt tokens; MQ = their length (the KV cache is cut back to MQ at every rebuild)
gen = []           # every token after the prompt: generated ones and injected information blocks
kept = []; absorbed = 0
loop:
  c0 = len(gen); R = min(c0, RW); nd = c0 - R              # nd tokens have left the raw window
  if nd > absorbed: kept += gen[absorbed:nd]; absorbed = nd
  if len(kept) > MAXD:
      mass = pooler.forward_with_mass(embed(kept))          # per-token attention mass, (len(kept),)
      kept = the MAXD tokens of highest mass, in their original order
  sp = pooler(embed(kept))                                  # (32, 1536), also when kept is empty
  block = concat(sp, embed(gen[c0-R : c0]))                 # inputs_embeds, positions MQ .. MQ+len-1
  cut the KV cache to MQ; run block; last = its final logits
  repeat CHUNK times:
      nx = sample(last)                                     # section 4
      if nx == EOS: stop
      gen.append(nx); if the last 8 tokens are identical: stop (dead)
      if a <search>...</search> just closed: build the information block (section 5), append its tokens to gen, break
      if <more> was written: the next chunk of the current page, same, break
      if "</think>" is present and the reply after it reaches the cap: stop
      run nx alone (position continues); last = its logits
```

`embed` is the model's `embed_tokens`; the pooler adds sinusoidal position encodings itself. The pooler
runs in float32 (its `forward_with_mass` returns the per-token mass used for eviction); the model in the
app's dtype. A rebuild costs a prefill of at most 32 + 768 tokens every 128 generated tokens. The
attention state is bounded: 32 soft tokens + 768 raw tokens + the prompt, whatever the conversation's
length, which is why the KV cache stays around 36 MB.

## 4. Sampling

Temperature sampling (0.6). One rule: the model may never write an information block itself. If the
sampled token would make the recent text contain `<information`, or end with a prefix of it (4 characters
or more), the token is rejected and its logit set to -inf, up to 8 times (the harness's `pick_plain`).
Optionally greedy after `</think>`; the measurements above did not.

## 5. Search protocol

The model writes `<search>query</search>` (the closing tag may be followed by up to 3 stray
characters). The text between the tags is the query; if it contains `||`, the part before it is the
query. The environment answers by appending, as tokens of `gen`:

```
\n<information>\n{chunk}\n[READER] (no extraction)\n</information>\n
```

where `chunk` is the first `PAGE_STEP = 256` tokens of the page text `Title: text` (Wikipedia's search
result's extract, or `LocalSearch.fetch(query)` from `release/local-search/`). Rules:

- `<more>` (also `<more/>`, `</more>`) appends the next 256 tokens of the current page; after `MAXM = 8`
  or when the page is exhausted: `\n<information>(no more of this page - search again or answer)</information>\n`.
- A query that returns the same page as an earlier one serves that page's *next* chunk (the harness's
  `--samepage 1`); a page used up that way: `\n<information>(this page is used up - search a different query or answer)</information>\n`.
- No result: `\n<information>(no results)</information>\n`.
- After `MAXS = 5` searches: `\n<information>(no searches left - answer from what you have read)</information>\n`.
- An empty query gets `(no results)`.

The answer is what follows `</think>`; the harness reads its first sentence. The model decides whether to
search at all: a reasoning or chat prompt usually gets no search.

## 6. Local search

`release/local-search/README.md` has the files and the algorithm; `code/search.py` is the reference
(`LocalSearch(store_dir, embedder_dir).fetch(query)` returns `Title: text`, the same shape the API path
served). On the device it is one process beside the model: 0.3-0.4 GB resident, 0.6-1.4 s a query.
Memory of the whole: 4-bit weights 1.0 GB, pooler 0.3 GB (0.15 in float16; 0.04 GB with the 4-bit pooler
kept quantized, section 1a), KV about 36 MB, runtime about 0.1 GB, search 0.3-0.4 GB: about 1.75 GB
(about 1.5 GB with the 4-bit pooler).

## 7. What to expect

Search held-out (eval300 subset, 300 rollouts, this loop), Wikipedia's API: 47.3% for
`g14-4bit-gptq-trained`, 45.0% for `g14-4bit-gptq`, 45.3% for the 16-bit weights. The local search,
measured with `g14-4bit-gptq`: 42.3% (paired against the API on the same questions -2.7 +- 4.1).
Reasoning held-out (Dolphin-R1, 100): 53%. `release/README.md` has
the table and `docs/` in the repository the measurements.
