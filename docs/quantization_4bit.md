# What 4-bit costs this model, and six attempts to get it back

The app runs an MLX affine 4-bit conversion of the pooler-distilled student, group size 64. This
records what that conversion costs on the held-out set, what was tried against it on 2026-09-11, and
which of those attempts can be told apart from doing nothing.

## The format, read off the shipped directory

`mlx_lm.convert --hf-path … -q --q-bits 4 --q-group-size 64` produces scales and biases for exactly
198 tensors: the seven projections of all 28 blocks, plus `embed_tokens` and `lm_head`. Scales and
biases are stored fp16. The norms, the attention biases and the pooler stay in float. `ctl/q4.py`
reproduces that grid in torch, including the step that makes each group's larger-magnitude edge
exactly representable, and `ctl/checkmlx.py` confirms that a directory written in the packed format
unpacks to the same numbers the evaluator measured.

Relative weight error of the untrained conversion: 9.51% overall, `embed_tokens` 10.19%,
`lm_head` 10.35% — the two largest, and the two an adapter on the projections cannot reach.

## The numbers

Held-out: the first 300 lines of `eval300.jsonl`, 150 distinct questions each answered twice, raw
window 768, maxd 384, samepage on, temperature 0.9, plain decoding. Every row is paired against the
bf16 run **per question**, which removes the between-question variance that dominates an unpaired
comparison. bf16 scores 39.7% with grounding 64%, landing 98%, 4.7 searches per rollout.

| attempt | correct | grounded | landed | searches | questions |
|---|---|---|---|---|---|
| 4-bit, untrained | -7.5 ± 4.2 | -10.4 | +0.0 | -0.8 | 106 |
| straight-through LoRA (`ctl/qat.py`) | -5.7 ± 3.3 | -3.0 | +0.0 | +0.4 | 150 |
| DWQ, temperature 2 (`ctl/dwq.py`) | -7.7 ± 5.5 | +2.9 | -2.9 | +1.4 | 52 |
| DWQ, temperature 1, clipped init | -6.3 ± 3.2 | -4.7 | +0.7 | +0.1 | 150 |
| group 32 instead of 64 | -11.6 ± 3.7 | -7.7 | -1.1 | +0.6 | 142 |
| quantization + pooler, joint (`ctl/jointfit.py`) | -7.4 ± 3.3 | -1.7 | -0.3 | +0.3 | 149 |

## What can be concluded

**The loss is real and is about seven points.** Every measurement agrees on that.

**None of the six attempts can be told apart from each other, or from doing nothing.** They span
-5.7 to -11.6 with standard errors of 3.2 to 5.5. That is the honest reading, and it is as much a
statement about the instrument as about the methods: at ±3.3 points, an experiment cannot resolve
the three-point differences it was run to find. Six ninety-minute measurements were spent on
comparisons that could not have come out decisive. Resolving three points needs roughly four times
the questions or four times the rollouts per question.

**A finer grid is not the answer.** Group 32 halves how many weights share a scale and measured
worse, not better.

**The training worked; it just did not transfer.** Every run closed most of the KL gap to bf16 - 55%
to 74% on plain contexts, 66% in the compressed configuration - and grounding recovered from -10.4
to -1.7. What did not recover is reading: correct over grounded is 62% for bf16, 61% for untrained
4-bit, and 56% for both trained runs. They find the fact more often and turn it into an answer less
often.

**The pooler's input is not the mechanism.** The pooler compresses `embed_tokens` output, which is
quantized at the largest error of any tensor, and this lineage fitted the pooler against the
unquantized table - so it looked like the culprit. Measured, it is not: the compressed-context KL
starts at 0.0364 against 0.0463 for plain contexts, so the compressed path is if anything less
damaged, and training the pooler jointly recovered grounding without recovering the score.

## Why matching bf16 does not buy accuracy

All six objectives asked the 4-bit model to imitate the bf16 model on stored traces, teacher-forced,
one token ahead. The evaluation is 1500 tokens of free generation with search in the loop. Small
per-token agreement does not constrain a trajectory: the student's own context is not the teacher's
trace, and errors compound along a path the training never visited. Recovering the search behaviour
without recovering the answer is what that looks like from outside.

## What to try next, in order

1. **Make the measurement able to see three points** before running any more comparisons.
2. **On-policy distillation.** Sample trajectories from the 4-bit student in the real environment and
   match the teacher on the contexts the student actually reaches. No reward and no groups, so it
   costs a fraction of GRPO, and it removes the off-policy gap above.
3. **Rotation (QuaRot / SpinQuant).** An orthogonal transform folded into neighbouring weights
   spreads the outliers that make affine 4-bit expensive. Expressible in the weights.
4. **Sensitivity-driven mixed precision.** Quantize one tensor at a time, measure, and spend more
   bits only where it hurts. `embed_tokens` and `lm_head` are the candidates; the app's loader
   quantizes exactly the modules the file says are quantized, so leaving them in float needs no code
   change, only about 350MB.
5. **AWQ-style per-channel scaling** from activation statistics, folded into the preceding norm or
   rows. No training.

The historical lineage adapted the model *while it was quantized*, against the task rather than
against a float teacher. That remains the other way back, and it is what GRPO under fake
quantization would do - at roughly ten minutes a step.
