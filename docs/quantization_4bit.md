# What 4-bit costs this model, and the attempt that got it back

The app runs an MLX affine 4-bit conversion of the pooler-distilled student, group size 64. This
records what that conversion costs on the held-out set, what was tried against it on 2026-09-11 and
2026-09-12, and which of those attempts can be told apart from doing nothing. One can: the last.

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
| 4-bit, untrained | -11.7 ± 3.0 | -11.0 | +0.0 | -0.4 | 150 |
| straight-through LoRA (`ctl/qat.py`) | -5.7 ± 3.3 | -3.0 | +0.0 | +0.4 | 150 |
| DWQ, temperature 2 (`ctl/dwq.py`) | -7.7 ± 5.5 | +2.9 | -2.9 | +1.4 | 52 |
| DWQ, temperature 1, clipped init | -6.3 ± 3.2 | -4.7 | +0.7 | +0.1 | 150 |
| group 32 instead of 64 | -11.6 ± 3.7 | -7.7 | -1.1 | +0.6 | 142 |
| quantization + pooler, joint (`ctl/jointfit.py`) | -8.0 ± 3.3 | -2.3 | -0.3 | +0.3 | 150 |
| pooler only, on the untouched 4-bit grid | -12.3 ± 3.2 | -13.3 | -1.7 | +0.4 | 150 |
| **self-trace fine-tuning under quantization (`ctl/jointfit.py --objective ce`)** | **+2.7 ± 3.4** | +0.3 | +0.0 | +0.5 | 150 |

## What the instrument can and cannot see

Measured from the data (`bf16`'s two rollouts per question against each other, and a bootstrap over
questions and rollouts for every arm): 62% of the 150 questions answer themselves the same way
twice - 31 always right, 62 always wrong - so the paired difference between two arms has a
standard error of 3.9 to 4.8 points, and the smallest difference detectable at 80% power is about
11 points. The four-bit loss itself is about 7; the differences between methods are 0 to 4. Neither
was ever within reach of this experiment. Resolving 3 points needs roughly 13 times the rollouts,
about 28 GPU-hours per arm; resolving 5 points needs about 5 times, 10 GPU-hours per arm.

Pooling the four group-64 arms per question gives a four-bit loss of -7.1 ± 2.7, p = 0.008. No
single arm reaches p < 0.05 against bf16 on its own, and no trained arm differs from the untouched
one: +3.8 ± 3.9, +2.4 ± 3.8, +0.9 ± 4.7, all p > 0.4.

## What can be concluded

**The loss is real and is about ten points.** The untouched arm, filled to all 150 questions,
sits at -11.7 ± 3.0 in the copy on Hugging Face. The same two arms were measured a second time on
another machine by accident - a stopped instance whose start request had been queued came back
hours later and ran the same job - and that copy gave -8.7 ± 3.0 for the untouched arm and
-10.0 ± 3.5 for the pooler-only one. The first 174 rollouts of the untouched arm are shared
between the two copies; the remaining 126 differ, and that alone moves the score three points.
Two full measurements of an identical setup disagreeing by three points is the instrument noise the
power analysis predicts, seen directly.

**None of the six attempts can be told apart from each other, or from doing nothing.** They span
-5.7 to -11.6 with standard errors of 3.2 to 5.5. That is the honest reading, and it is as much a
statement about the instrument as about the methods: at ±3.3 points, an experiment cannot resolve
the three-point differences it was run to find. Six ninety-minute measurements were spent on
comparisons that could not have come out decisive. Resolving three points needs roughly four times
the questions or four times the rollouts per question.

**A finer grid is not the answer.** Group 32 halves how many weights share a scale and measured
worse, not better.

**The training worked; it just did not transfer.** Every run that moved the quantization parameters
closed most of the KL gap to bf16 - 55% to 74% on plain contexts, 66% in the compressed
configuration - and the joint run brought grounding from -10.4 to -2.3. The score did not follow.
Of what bf16 gets and a 4-bit arm loses, about half is the fact never being served and half is the
fact being served and misread, in every arm alike; landing is never the problem. And the measured
difference is a small net of a large two-way exchange: the untouched arm loses 19.5
rollout-equivalents to bf16 and wins 11.5 back, so three to five times the net amount changes hands
in both directions.

**The pooler is neither the mechanism nor the fix.** It compresses `embed_tokens` output, which is
quantized at the largest error of any tensor, and this lineage fitted it against the unquantized
table - so it looked like the culprit. Measured, it is not: the compressed-context KL starts at
0.0364 against 0.0463 for plain contexts, so the compressed path is if anything less damaged.
Trained on its own, with the grid untouched, the pooler could not move the validation KL at all
(0.0345 to 0.0344 over 1200 steps) and did not help: -12.3 against -11.7 in one measurement,
-10.0 against -8.7 in the other, grounding no better in either. The joint run's KL gain came entirely from the quantization parameters. One reason the pooler
run hurt: the loss was computed on the teacher's eviction schedule, so the student pooler's mass
output - which decides what gets evicted at inference - was never constrained.

**The "reading got worse" story was wrong.** An earlier reading of the aggregates said the untouched
4-bit model kept the bf16 reading rate and training destroyed it. The conditional rate, correct given
grounded with a bootstrap over questions, is 61% for bf16 and 50 to 55% for every 4-bit arm,
untouched included. Reading is down for all of them, about equally.

## The app's own temperature

Every row above was sampled at 0.9, the temperature the lineage was trained and evaluated at. The app
samples at 0.6. If the 4-bit damage lived in the tail of the distribution, the colder sampling the
app actually uses would hide most of it, and the right fix would have been to train and measure at
0.6 rather than to chase the 0.9 gap. Both models were therefore measured again at 0.6, untouched,
300 rollouts each on the same 150 questions (2026-09-12, box 50767646).

| arm, temperature 0.6 | correct | grounded | landed | searches | vs bf16 at 0.9 | vs bf16 at 0.6 |
|---|---|---|---|---|---|---|
| bf16 | 45.7% | 65% | 96% | 5.0 | +6.0 ± 3.4 | - |
| 4-bit, untrained | 37.0% | 61% | 97% | 5.7 | -2.7 ± 3.3 | **-8.7 ± 3.0** |

Cooling the sampling lifts both models by about six points. The paired gap between them, read in
the 0.6 frame, is -8.7 ± 3.0: the same size as the two 0.9 measurements of the untouched grid
(-11.7 ± 3.0 and -8.7 ± 3.0). The loss is not in the sampling tail, and temperature is not the fix.

Two things follow for the shipped app rather than for the research question. At the temperature it
runs, the untouched 4-bit conversion scores 37.0% absolute on this set, which is where the 0.9-frame
target of 37% sits, but that number is the wrong comparison: the same app with the float model would
score 45.7%, and the eight or nine points between them are what quantization still costs. And the
grounding difference at 0.6 (-4.0 ± 3.2) is smaller than at 0.9 (-11.0 ± 3.4) while the correctness
difference is not, which is one more reading in which the 4-bit model finds the page and misreads it.

## The one that worked: learning its own behaviour under quantization

Every objective above asked the 4-bit model to imitate the float model. The last one does not. It
takes the traces the lineage itself produced during GRPO that were correct, grounded and landed
(1431 of them, every held-out question excluded by its text), puts them through the compressed
context exactly as the evaluator builds it (chunking, eviction, the pooler's summaries, one block per
step), and minimises the cross-entropy of the trace's own tokens through the 4-bit grid. Only the
scales and biases of the 198 quantized tensors move, at 2e-6 for 1500 steps of about a second each;
the 4-bit codes, the pooler and everything unquantized stay as shipped. The result packs into the
same MLX directory the app loads, and the directory was checked to unpack to the values measured.

| arm, temperature 0.9 | correct | grounded | landed | searches | vs bf16 |
|---|---|---|---|---|---|
| bf16 | 39.7% | 64% | 98% | 4.7 | - |
| 4-bit, untrained | 28.0% | 53% | 98% | 4.3 | -11.7 ± 3.0 |
| 4-bit, self-trace fine-tuned (s1) | **42.3%** | 64% | 98% | 5.2 | **+2.7 ± 3.4** |

On the same 150 questions, paired, the fine-tuned 4-bit model is indistinguishable from bf16 and
about fourteen points above the untouched grid, with grounding and landing back at the float
model's values. Training loss went 0.40 to 0.28 and held-out-trace validation 0.79 to 0.63, still
falling at the last step, so the run was stopped by its budget rather than by convergence. Twenty-
five minutes of training on one RTX 3090, about $0.10.

Why this worked where six imitation objectives did not is the same reason the imitation objectives
could not: the traces are on-policy for this lineage, they were produced by this model in this
environment with this pooler, and the loss is measured in the context the model will actually see.
Matching bf16 one token ahead constrains agreement on the teacher's path; reproducing scored traces
in the compressed context constrains the path itself. It is also, in kind, how the lineage was
trained when it lived in 4-bit.

Caveats, before this becomes the shipped file. One measurement, 300 rollouts, standard error 3.4:
the honest claim is "at bf16's level, not below it", not "above it". The traces come from GRPO
rollouts on the training questions; the held-out set was excluded by question text and was never
in GRPO, but the two draw on the same corpus and the same search index. And the untouched grid's
own two measurements differ by three points, so a replication of s1 on a second box is the next
thing worth its cost.

## Why matching bf16 does not buy accuracy

All six objectives asked the 4-bit model to imitate the bf16 model on stored traces, teacher-forced,
one token ahead. The evaluation is 1500 tokens of free generation with search in the loop. Small
per-token agreement does not constrain a trajectory: the student's own context is not the teacher's
trace, and errors compound along a path the training never visited. Recovering the search behaviour
without recovering the answer is what that looks like from outside.

## What to try next, in order

1. **Ship s1 behind a replication.** Run the same 300 rollouts again on another machine; if it holds
   within noise of bf16, `sft_s1_mlx4` on Hugging Face is the directory to put in the app.
2. **Continue s1.** Validation was still falling at step 1500; a second 1500 steps from the saved
   state costs another $0.10 before the $0.60 measurement.
3. **Make the measurement able to see three points** before comparing s1 variants against each other.
4. **On-policy distillation.** Sample trajectories from the 4-bit student in the real environment and
   match the teacher on the contexts the student actually reaches. No reward and no groups, so it
   costs a fraction of GRPO, and it removes the off-policy gap above.
5. **Rotation (QuaRot / SpinQuant).** An orthogonal transform folded into neighbouring weights
   spreads the outliers that make affine 4-bit expensive. Expressible in the weights.
6. **Sensitivity-driven mixed precision.** Quantize one tensor at a time, measure, and spend more
   bits only where it hurts. `embed_tokens` and `lm_head` are the candidates; the app's loader
   quantizes exactly the modules the file says are quantized, so leaving them in float needs no code
   change, only about 350MB.
7. **AWQ-style per-channel scaling** from activation statistics, folded into the preceding norm or
   rows. No training.

The historical lineage adapted the model *while it was quantized*, against the task rather than
against a float teacher. s1 is the cheap end of that: its own scored traces instead of new
rollouts. GRPO under fake quantization, at roughly ten minutes a step, remains the expensive end.
