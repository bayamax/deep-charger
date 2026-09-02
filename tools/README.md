# tools/eval_heldout_parallel.py

Held-out evaluation for the GRPO search policy, parallel and device-agnostic.

It reuses the episode loop, retrieval and scorer of `grpo_ep_torch.py` unchanged, so
its numbers are directly comparable with the `correct=` / `grounded=` / `landed=`
figures in the training log. Nothing here trains or writes an adapter.

## Why it is parallel

A rollout spends roughly a third of its wall clock waiting on the Wikipedia API and
the rest decoding one token at a time. Measured on the 3060 box: 40 s per rollout,
3.5 cache misses per rollout, GPU utilisation 30%, GPU memory 4 GB of 12 GB. Both the
waiting and the batch-1 decoding leave the device idle, so overlapping rollouts is
close to free. One model is shared by all workers; extra workers cost only KV cache.

Raising `EVAL_THREADS` does **not** raise the Wikipedia request rate — that is capped
globally by `EVAL_RPS`. The User-Agent identifies a real person, so keep it polite.
Concurrent rollouts asking for the same keyword collapse into a single fetch.

## Question selection

`EVAL_MATCH=1` (default) draws a sample whose teacher-search-count distribution
matches the training slice `pool[1:241]`, so held-out and training numbers can be
compared without post-hoc standardisation. The corpus is shuffled with
`random.Random(0)` exactly as training does, and `EVAL_SKIP` keeps the sample clear of
the indices training consumed.

## Sharding

`EVAL_SHARD=i/n` takes every n-th question. Each shard writes its own jsonl, so two
machines can split one question set and the results merge by concatenation. Re-running
a shard skips questions already in its output file, so an interrupted run resumes.

## Usage

```sh
# CUDA box, 6 workers, first half
EVAL_HOME=/root/work GRPO_MODEL=/root/work/bf16_s100 GRPO_RESUME=/root/work/ckpt_best \
EVAL_THREADS=6 EVAL_SHARD=0/2 python3 eval_heldout_parallel.py

# Apple silicon, 2 workers, second half
EVAL_HOME=~/work GRPO_MODEL=~/work/bf16_s100 GRPO_RESUME=~/work/ckpt_step120 \
EVAL_THREADS=2 EVAL_SHARD=1/2 python3 eval_heldout_parallel.py
```

The model forward is serialised behind a lock on every backend. Letting threads call one
shared module concurrently produces NaN logits and the rollout dies in
`torch.multinomial` — measured on CUDA with 6 workers, 47 of 50 rollouts were lost.
Threads therefore buy the retrieval overlap only, worth roughly 1.6x.

For more than that, run **several processes** on different shards, each with its own
copy of the model. On the 3060 box, two processes of 4 threads each raised GPU
utilisation from 30% to 79% and cut the effective cost from 40 s to about 13 s per
rollout, at 8.4 GB of 12 GB. Three processes would not fit.

## Environment

| variable | default | meaning |
| --- | --- | --- |
| `EVAL_DEVICE` | auto | `cuda` / `mps` / `cpu`; auto-detected when unset |
| `EVAL_DTYPE` | `bf16` | `bf16`, `fp16`, `fp32`. Use `fp16` on macOS below 14 |
| `EVAL_HOME` | script dir | where the corpus and `*cache*.jsonl` live |
| `GRPO_MODEL` | `$EVAL_HOME/bf16_s100` | base model |
| `GRPO_RESUME` | none | LoRA adapter directory to evaluate |
| `EVAL_N` | 300 | questions in the whole sample, before sharding |
| `EVAL_G` | 2 | rollouts per question |
| `EVAL_SKIP` | 380 | corpus index the held-out sample starts from |
| `EVAL_MATCH` | 1 | difficulty-match to the training slice; 0 for a plain slice |
| `EVAL_THREADS` | 4 | rollouts in flight |
| `EVAL_RPS` | 3.0 | global Wikipedia request rate, shared by all workers |
| `EVAL_SHARD` | `0/1` | this machine's shard as `i/n` |
| `GRPO_MAXS` | 5 | search cap, same as training |

Rollouts within one question are strongly correlated, so the standard error of the
mean is set by the number of questions, not the number of rollouts. For a fixed time
budget, more questions with `EVAL_G=2` beats fewer questions with `EVAL_G=4`.
