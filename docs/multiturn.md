# Multi-turn

The model (g14, 4-bit GPTQ) was trained on single questions. The app carried no history between turns. This
records what the model does with a conversation, what the app should hand it, and the training that follows.

## The protocols (`pool_eval.py --multiturn --mt-mode`)

- **none**: each turn alone (what the app did).
- **full**: the earlier exchanges (user + reply, thinking stripped, as the chat template renders them) verbatim
  in the pinned prompt, the current question last; the pooler only for this turn's own evicted tokens.
- **stream**: only the current question pinned; every earlier token (questions, thinking, search results,
  replies) handed to the pooler and mass-evicted to 384 tokens - the bounded-KV design.
- **mix**: the last exchange as in full, older turns as in stream.

## The held-out dialogues (`mt_gen.py`, teacher: DeepSeek)

From eval300 questions (never trained on), 86 dialogues with gold answers:
- **bridge** (40): a two-hop question split into two turns; turn 2 refers back by pronoun ("that film").
- **memory** (26): a personal remark, an unrelated search question, then "what did I say my dog's name was?".
- **switch** (20): two unrelated questions in a row.

## Measured (2026-09-29, mt2: 4-bit, Wikipedia's search, first 30 dialogues, final turns)

| | bridge | memory | switch |
|---|---|---|---|
| none | ~0 | 0 | - |
| full | 41.7% (5/12) | 40.0% (4/10) | **0/8** |
| stream | **0/12** | **0/10** | 3/8 |

- **full**: the history is used (references resolved, details recalled), but on a topic switch the model
  answers the FIRST question of the conversation - its searches go to the previous topic (asked about The
  Human League, it searched "Josiah Bartlett House"). Trained only on one user message, it anchors on it.
- **stream**: the topic carries through the pooler, the names do not. Asked for the pottery instructor's name
  (Miriam) it searched pottery and named a potter; asked for the dog's name (Rufus) it said "your dog, Pepe";
  asked where the user was moving (Asheville) it said New York City. References to a named entity ("that
  dog", "that dramatic work") fail outright. The switch is better than full (only the current question is
  pinned).

So: names travel as text, gist can travel through the pooler, and the model must learn to answer the latest
question with history in view. The user's trade-off (2026-09-29): no long-range recall is required, a natural
rally is; the topic switch is the bottleneck; existing ability must be kept. Names and facts from earlier in a
conversation are not the model's job - a memory bank retrieves them the proper way - so the bridge and memory
numbers are read as information, and the targets are the switch, the rally's fluency, and no loss elsewhere.

## The training (memfit.py, run mem5)

Self-distillation. The **teacher** is the same model on each turn's self-contained version, alone
("when was Spirited Away released?" for "when was that film released?"; the question itself for a switch; the
remark quoted for a recall) - what it does well. The **student** sees the turn as asked with the earlier
exchanges verbatim in the prompt (`--student full`). The loss is the KL between them on the teacher's own
trajectory, so the student learns to answer the latest question and resolve references from the history,
and nothing moves where the two already agree (the existing ability). Half the examples are topic switches
assembled from the traces (one dialogue's history, another dialogue's self-contained turn). The LoRA and the
pooler train; the model's merged weights do not.

Checks: a switch-only held-out set (60 dialogues) before and after, the first 30 dialogues above (vs mt2
full), and the single-turn search held-out (102 questions, vs 48.0%) for the loss.

## What the app hands the model

The last few exchanges (user message + the reply after `</think>`, thinking and search results stripped)
through the chat template, then the new user message and `<think>\n`; older exchanges dropped. The pooler
loop runs as for a single question. (Written into release/USAGE.md once the trained model is released.)

## Results so far (2026-09-30, the switch set: 60 held-out dialogues, two unrelated questions each)

Turn 1 has no history, so its prompt is identical in every arm that does not change the weights; three runs of the
base model under identical conditions read 41, 33 and 29 of 60 - a 60-dialogue reading moves by several points on
its own, and every comparison below is against the average of the base runs.

| | turn 1 (no history) | turn 2 (after an unrelated exchange) |
|---|---|---|
| base, history as separate turns (2 runs) | 41, 33 | 17, 19 |
| base, **quote** (no training; history quoted inside the one user message) | 29 (the same prompt as base) | **28** |
| mem5 (self-distillation, pooler + LoRA) | 32 | 26 |
| mem6 (+ single-turn anchors, gentler) | 27 | 29 |
| mem7 (rejection-sampled self-training, LoRA only) | 35 | 20 |

Per question against the average of the two base runs, turn 2: quote +16.7 +- 5.5 points, mem6 +18.3 +- 6.3, mem5
+13.3 +- 5.6, mem7 +3.3 +- 6.1. The turn-1 drops read earlier as damage are mostly the base's own spread (mem7 -3.3
+- 5.0, mem5 -8.3 +- 5.4; mem6 -16.7 +- 5.6 the one that may be real). The single-turn search held-out (102):
mem5 45.1%, mem7 50.0%, base 48.0%.

So the protocol alone - the model given one user message, the history quoted as context - recovers as much of the
topic switch as the best training, with no weight change and so no loss anywhere. Next: whether quote keeps the
references and recall the full protocol had (mt_eval), and whether training adds to it (mem6 + quote).

## mem8 (2026-09-30): quote-collected successes trained into the native chat - no measurable effect

Collection: 70 training chains of 4 unrelated questions under quote; correct 29 / 13 / 13 / 22 of 70 by turn (the
switch turns 2-3 at 19%, below the 28% quote read on the switch set), 77 correct turns in all (67 train, 10
validation). memfit ce, native prompt, LoRA only, pooler frozen: validation KL 0.76 -> 0.32 (step 200).

The 30 held-out chains of 4 (the switch set's 60 dialogues joined two by two), native protocol, same questions
paired:

| | turn 1 | turn 2 | turn 3 | turn 4 |
|---|---|---|---|---|
| base | 16 | 10 | 16 | 8 |
| mem8 | 18 | 11 | 14 | 10 |

Turns 2-4: base 34/90, mem8 35/90, +1.1 +- 5.6 points (13 won, 12 lost). The single-turn search held-out (102):
mem8 49.0% (44.1 / 61.8 / 41.2), base 48.0% - no loss, and no gain on the chains either.

On these chains the base's turn 3 (a first question after two unrelated exchanges) reads as high as turn 1; the
drop is on turns 2 and 4, the switch set's second questions. So part of the "switch drop" may be those questions'
own difficulty. mem9 measures it: each chain question alone (twice, for the spread) and quote on the same chains.

## mem9 (2026-10-01): the history does cost the switch turns (first reading, 30 chains)

Same 30 chains, base model, correct of 30 per turn (turn 1 has no history in any arm - its spread is the noise):

| | turn 1 | turn 2 | turn 3 | turn 4 |
|---|---|---|---|---|
| each question alone (no history), run 1 | 20 | 19 | 19 | 11 |
| native history (base-chain) | 16 | 10 | 16 | 8 |
| quote | 17 | 13 | 17 | 9 |
| mem8, native history | 18 | 11 | 14 | 10 |

Paired alone vs history: turn 2 11 vs 2 discordant questions (sign test p ~ 0.02), turns 2-4 21 vs 6 (p < 0.01);
turn 1, identical prompts, 8 vs 4 (chance). Turn 4's questions are hard on their own (11 alone). So the history costs
the first switch most; its size is not settled at 30 chains (the alone run's turn 1 reads 4 above the history run's
identical turn 1). The second alone run finishes the mem9 job.

mem10: a held-out of 100 fresh chains of 4 (400 questions), base with history and alone, side by side with a training
collection 4x mem8's (about 200 chains, each question run alone, one success per distinct question), then memfit
ce-native (LoRA only) and the 100-chain held-out and the single-turn 102 on the result.

The second alone run (2026-10-01): 19 / 18 / 21 / 9. The two alone runs agree: turn 2 alone 19 and 18 vs history 10
(11 vs 2 and 11 vs 3 discordant), turns 2-4 49 and 48 vs 34 of 90. The history costs the switch, most on the first
one; settled enough to act on. mem10's 100-chain base measurement was dropped after the card proved to hold only two
processes (about 6.5 GB each): mem10b runs the training collection (202 chains alone) on two lanes, trains, and
measures mem10 on the same 30 chains as base / mem8 and the single-turn 102.

## mtg1 (queued behind mem10d): multi-turn GRPO with nano's naturalness check

online_loop.py --mt-items: each step is one search turn of a training conversation (rft8 / rft10, the model's own
earlier replies as the history), rolled out 8 times with the history in the prompt exactly as the app sends it,
scored as in the search GRPO (grounded-correct 1.5; nano's sound / natural / clean +0.5, where natural now also fails
a reply that answers or drags in the earlier topic), group-normalised advantage, LoRA all layers r16 at 1e-5, pooler
frozen, collapse guard on. 100 steps from the base, then the 30 held-out chains and the single-turn screen.

## mem10 (2026-10-01): 404 training questions run alone, 141 own successes (106 with history), ce-native, LoRA only

Same 30 chains, correct of 30 per turn:

| | turn 1 | turn 2 | turn 3 | turn 4 |
|---|---|---|---|---|
| base, history | 16 | 10 | 16 | 8 |
| each question alone (two runs) | 20 / 19 | 19 / 18 | 19 / 21 | 11 / 9 |
| mem8 | 18 | 11 | 14 | 10 |
| **mem10** | 18 | **17** | 17 | 12 |

Turns 2-4: base 34, mem10 46 of 90 (23 won, 11 lost; sign test p ~ 0.06). Turn 2, the first switch, goes from 10 to
17 - most of the way to the alone level (18-19). The single-turn 102 decides whether the multi-turn GRPO (mtg1)
starts from mem10 (mean of its three shards >= 45) or from the base.

## mtg1 (2026-10-01/02): multi-turn GRPO from mem10, steps 25 and 100

Same 30 chains, correct of 30 per turn; single-turn 102 (shards 0-2):

| | turn 1 | turn 2 | turn 3 | turn 4 | turns 2-4 | single-turn |
|---|---|---|---|---|---|---|
| base, history | 16 | 10 | 16 | 8 | 34 | 48.0 |
| mem10 (the start) | 18 | 17 | 17 | 12 | 46 | 45.1 |
| mtg1 step 25 | 19 | 15 | 18 | 14 | 47 | 55.9 (shard 0 only) |
| mtg1 step 100 | 19 | 15 | 14 | 8 | 37 | 51.0 (52.9 / 47.1 / 52.9) |

The training reward did not move over the 100 steps (search pass by 20-step block 26 / 21 / 31 / 23 / 26%, a
60-step running mean flat at 24-27%). Step 100 is above the base on both screens but below step 25 / mem10 on the
later turns; with 30 chains these gaps are within noise. The user asked to keep going: mtg1c resumes to 200
(measured) and 300 (measured), same settings.

## mtg1 continued to 300 (2026-10-02): the app candidate

| | turn 1 | turn 2 | turn 3 | turn 4 | turns 2-4 | single-turn 102 |
|---|---|---|---|---|---|---|
| base, history | 16 | 10 | 16 | 8 | 34 | 48.0 |
| mem10 | 18 | 17 | 17 | 12 | 46 | 45.1 |
| mtg1 step 200 | 18 | 18 | 18 | 13 | 49 | 56.9 (50.0 / 61.8 / 58.8) |
| **mtg1 step 300** | **21** | 18 | 17 | **17** | **52** | **57.8** (52.9 / 61.8 / 58.8) |

Step 300 (`pooler_distill/chatsft/multiturn/mtg1_s300.safetensors`, LoRA r16 all layers + pooler, over the 4-bit
g14 base) is the app candidate: turns 2-4 at the level of each question run alone (49-51), single-turn +10 over the
base. Unfinished rollouts during training were all the 7-search guard (the model searching on after "no searches
left"), none a length or time cap; they peaked at 39% around steps 201-240 and fell to 7% by 281-300.

Follow-up (bridge) dialogues, the 40 of mt_eval, turn 1 / turn 2 (the follow-up): base 26 / 19, step 300 23 / 18 -
the switch training did not cost the follow-ups. Next (mix0 -> mtg2): the Dolphin reasoning baseline, then GRPO from
step 300 on switch + ~30% bridge items, measured on all four screens.

## Bounded memory: the history through the pooler (2026-10-03, "win")

The app's memory must not grow with the conversation. `pool_eval.py --mt-mode win` runs the dialogue as one stream,
exactly as one reply runs: the newest 768 tokens (the end of the previous turn - its pages, thinking, reply - and the
new question) stay raw, everything older is in the pooler's 32 soft tokens (from at most 384 kept tokens), only the
new question is pinned. KV is bounded by question + 32 + 768 (+ the 128-token chunk) whatever the number of turns.
Step 300, 40 follow-up dialogues: follow-ups 17/40 under win against 18/40 with the whole history in the prompt; the
30 switch chains turns 2-4: 51/90 under win against 52/90. Win is the spec; the app keeps no history in the prompt.

## mtg2 / mtg3 (2026-10-02/04): GRPO with reasoning mixed in

mtg2 (step 300 on switch + bridge, every step a search turn, lr 1e-5) rolled back at step 56 and lost Dolphin 54 -> 49.
mtg3: one step in four a Dolphin reasoning problem (nano against the R1 reference), both rates 5e-6, from step 300.
The dips in the training pass rate were mostly hard stretches of the item list (mtg2 and mtg3 draw the same items at
the same steps and match there), but past step 100 the run did drift (rollback at 150; step 200 below step 300 on
search). Half of the search groups carry no signal (all eight samples alike, nearly always all wrong); nano judged only
6 of 150 wrong replies "the same answer in another form", so the misses are real, not gold-string strictness.

| screen (win where multi-turn) | step 300 | **mtg3 step 100** | mtg3 step 200 |
|---|---|---|---|
| single-turn, all 300 | 144 | **160** (54 / 54 / 52) | - |
| single-turn, first 102 | 59 | 64 | 55 |
| follow-ups, 40 of mt_eval (turn 1 / follow-up) | 25 / 17 | 23 / 19 | 27 / 14 |
| follow-ups, 97 new (gpt-5-mini, selfq_all seeds) | 39 / 21 | **43 / 31** | - |
| switch chains, turns 2-4 of 30 | 51 | 52 | - |
| Dolphin reasoning 100 (nano) | 54 | 56 | 61 |

**mtg3 step 100 is the app candidate** (`pooler_distill/chatsft/multiturn/mtg3_s100.safetensors`, LoRA r16 all
layers + pooler over the 4-bit g14 base): no screen below step 300, single-turn +16/300, new follow-ups +10/97.
Next (teach): probe 600 training items with step 100, R1 solves the never-solved ones in the student's search
environment (verified, no answer-first queries), supervised steps on those trajectories beside the model's own.
