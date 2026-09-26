# Toward a conversational search agent: the first night

Goal: keep the search skill of the step-200 student (39.7% on the held-out set with compression and
search) and give it back the ability to converse, so the app can be used the way a chat assistant with
search is used. Everything below is from 2026-09-13.

## What the checkpoints can do today (CPU probes, no training)

- The step-200 student treats every input as a search task. "Hi! How's it going?" becomes three
  searches; a request for a haiku becomes a search and "The answer is 'Haiku for autumn rain'".
  A system prompt does not change this, and neither does writing "this is casual conversation, I will
  not search" into the start of its thinking: it writes `<search>` on the next line anyway.
- The plain DeepSeek-R1-Distill-Qwen-1.5B (the lineage's base) chats acceptably, never uses the
  search tool even when told about it, and confabulates facts it does not know ("Spirited Away was
  directed by Michael Mckay"). Given a Wikipedia chunk in its prompt it reads correctly when the fact
  is there (Tokyo Skytree: 2012, 634 m) and confabulates when it is not.
- So the two abilities live in two checkpoints, and the search decision and the multi-hop persistence
  exist only in the trained one.

## The data recipe: student prefix, teacher continuation

A trace is built as  `<think>` [student's own searches and served chunks, cut right after the chunk
that first shows the gold] + [teacher's continuation of the thinking] `</think>` [teacher's reply].
The search part is on-policy; only what happens after reading comes from the teacher (DeepSeek R1
through the API). Four ways of asking for the continuation were compared on the same prefixes:

| method | what happened |
|---|---|
| assistant prefill (R1, V4-pro) | continues in the student's voice, but fabricates new `<search>`/`<information>` blocks in 2 of 6 and drifts into restating rules |
| assistant prefill (deepseek-flash) | best voice match, clean in 4 of 6 |
| R1's own hidden reasoning as the thinking | "The user asks: ... 2-4 sentences ..." - reasoning about the request, not a continuation |
| structured JSON, continuity enforced ("pick up from the last result, never restate the question or the rules") | 6 of 6 coherent; adopted |

Verification is an LLM pass (V4-pro) per sample: commits to an answer, matches the gold, no claim
that is not in the served text or the question, English, naturalness 1-5, continuity 1-5. Two samples
per prefix, best passing one kept. On the 60-question pilot: 87 of 120 samples passed, 49 of 60
questions covered, continuity 4.95, naturalness 4.54. The verifier proved more reliable than my own
reading twice (details I flagged as leaks were in the source text).

Limits found on the way: the 1431 correct GRPO traces cover only 222 distinct questions; the DeepSeek
account ran dry (-$0.23) after the pilot and 162 questions under the earlier prompt, so the corpus
stopped at 184 traces (49 continuity-prompt, 135 earlier structured prompt). A pool of 5990 real
Google questions (nq_open, short Wikipedia-answerable golds, held-out excluded) is on the hub for the
expansion, with a box mode that has the student answer them.

## Where the questions come from, and their register

The nq_open questions are people's own Google queries. Run through the student (shard 0, 499
questions, one rollout each, RTX 3060 at $0.06/h): 15% correct and grounded, 37% grounded, 63%
never grounded. The shallow reader (top page, 256-token chunks, at most 8 more) and dated or noisy
golds account for most of the misses, so the pool yields about 150 usable prefixes per 1000
questions.

The bigger issue is register: a search box gets "who wrote the song ruby don't take your love to
town", a chat assistant gets "any idea who wrote the song You Don't Know Me?". Training only on
the first would teach the search reflex on a phrasing users never type. No public corpus of real
assistant conversations that also carries a Wikipedia-checkable gold was found, so the answerable
questions are rewritten into two chat registers by deepseek-flash, with real assistant prompts
(WildChat, OASST) as style exemplars and a second call that checks the rewrite still asks for
exactly the same fact. Smoke on 30 questions: 24 rewritten, all 24 kept the need, openers diverse.
Shard 0 in full: 73 answerable questions became 131 chat-register turns
(`pooler_distill/para_pool0.jsonl`). A box then runs the student on the rewritten forms
(`ctl/boxM.sh`, gen mode with `GQFILE`), so the search prefix stays on-policy for the phrasing
that will actually be seen; the teacher continuation and the flash verifier follow as before.

Result of that run (131 rewritten forms, one rollout each, RTX 3060, 85 minutes): 50% correct and
grounded, 71% grounded. The polite form ("Who wrote the song ...?") came out at 56%, the casual
form ("I heard ... earlier, who actually wrote it?") at 44%; per original question, 21 were right
in both forms, 20 in one, 25 in neither. Because the originals were selected by a single correct
rollout, part of the drop is sampling: re-rolling the 73 originals on the same box gave 56%, so
the polite chat form costs nothing (56% against 56% on the same 66 questions) and the casual form
costs about 11 points (45% against 55%). The casual phrasings, with their context clauses and
hedges, are exactly what the rewrite corpus is for. The teacher continuation on the 62 gold-bearing prefixes gave 56 verified traces
(`pooler_distill/chatsft/search_sft_para0.jsonl`), naturalness 5.0, continuity at least 4; the
rejections were almost all the teacher adding facts not in the served text.

Search Arena (lmarena's real prompts to search-enabled assistants) was checked as an alternative
source, with the two search-model replies and the cheap judge agreeing on a gold. Of 132 judged
prompts, 15 asked for a short fact and 4 survived the stable / encyclopedic / agreed-gold
filter (about 3%): real search-assistant traffic is mostly news, how-to, recommendations and
comparisons, which this lineage's Wikipedia reader does not serve. The prompts are kept as style
exemplars for the rewrite instead of as questions.

The no-search side was meant to have two sources: R1 answering real prompts (236 verified pairs,
`chat_sft_v1.jsonl`) and the lineage's own checkpoints writing K candidates that the flash judge
scores. The second source is closed after two full runs on an RTX 2060 (about $1.7 of card time):

| checkpoint | candidates | usable (reached `</think>`, ended, clean) | judged pass |
|---|---|---|---|
| plain distill, K=6, 1200 new tokens | 1686 | 836 | 2 of 72 judged (canned "I'm DeepSeek-R1..." self-introductions, markdown lists, "I'm here to help" boilerplate) |
| earlier deep-charger checkpoint (`fft_out/student.pt`), K=3, 1000 new tokens | 843 | 101 | not judged: average 25 words, and the readable ones confabulate ("Obi Watanabe Kenobi ... born October 28, 1970") |

The candidate files stay on the hub (`chatsft/self_cands2.jsonl`, `self_cands3.jsonl`) but the
teacher pairs are the no-search data.

The second rewrite pool (nq shard 1: 71 answerable questions, 142 turns) went through the same
steps on an RTX 2060 in 85 minutes: 61% correct and grounded (polite 68%, casual 54%), 86
prefixes, 73 verified traces (`search_sft_para1.jsonl`). One operational note: since the evening
of 2026-09-14 the API serves `deepseek-reasoner` requests with the model field reading
`deepseek-flash` (about 3 s per continuation, 760 reasoning tokens); the verified quality did not
change, but the teacher for para1 was effectively the cheap model.

## The first fine-tune (s2) and what it broke

LoRA r=32 on all projections, lr 1e-4, 3 epochs over the 184 traces (65 optimizer steps), loss on
the student's prefix, the continuation and the reply; prompt and served chunks masked. Validation
loss 1.09 to 1.00 by step 20. Measured on the held-out set, 201 of 300 rollouts before the credit
ran out (105 questions, paired):

| | bf16 step 200 | s2 |
|---|---|---|
| correct | 39.7% | 34.3% (-6.2 ± 4.1) |
| grounded | 64% | 57% (-10 ± 5) |
| landed | 98% | 99.5% |
| searches per rollout | 4.7 | 3.7 |
| replies starting "The answer is" | 289 / 293 | 8 / 200 |
| zero-search rollouts | 7 / 300 (1 correct) | 13 / 201 (0 correct) |
| `<search>` after `</think>` | 1 / 300 | 21 / 201 |
| replies with CJK characters | - | 31 / 200 |
| reply length, words (median / p90) | one sentence | 82 / 420 |

The style transferred: replies are conversational sentences instead of "The answer is X". But the
model learned the form without the discipline. It answers fluently with nothing behind it ("That film
is The West Coast Five, a 1996 drama about single, unemployed actors", zero searches, wrong), it
sometimes keeps generating after the reply (search tags after `</think>`, repeated sentences, stray
Chinese), and it searches less. Part of the length tail is the evaluator: it stops on "answer is",
which these replies no longer say, so it runs to the token limit; the confabulation and the
zero-search answers are real.

Reading: 184 examples with an aggressive LoRA (r=32, lr 1e-4, 3 epochs) is enough to overwrite the
surface form and to loosen the search reflex, and not enough to teach when the fluent form is
warranted. This is the expected failure of imitating a stronger teacher's prose with a 1.5B model,
and it is why the plan has a second stage: selection and reward on the student's own outputs, where
a fluent wrong answer scores zero.

## The second fine-tune (s3): the style moves, the stop does not

s3 trained the step-200 student on 863 records (search-side 327 teacher traces, no-search 236,
and 300 strict single-sentence traces replayed raw), r=16, lr 3e-5, 2 epochs, 213 steps, 13 min
on an RTX 3060. Validation loss 1.993 -> 1.576, no sign of divergence. Held-out (EOS stop, reply
cap 200, temperature 0.9), first 100 questions: 35% correct and grounded, 56% grounded, 2.9
searches per question, 13% answered without searching, 6% with tags after `</think>`, 15% of
replies with CJK characters. Read side by side with the bf16 baseline (39.7%, single-sentence
answers, 4% zero-search), the search ability is intact within noise and the output has moved:
about half the replies open in the reply register ("It is the Danube River. Wachau is one of the
most prominent tourist destinations of Lower Austria ...") instead of "The answer is X.". But not
one of 52 inspected replies is a clean conversational answer: 45 of 52 run to the token cap,
repeating the point or adding facts from nowhere, and the few that stop early are broken
(a line repeated ten times, an HTML tag). The two reply endings in the data - the strict traces
end after one sentence, the teacher replies after three or four - were mixed without any signal
to tell them apart, and 213 gentle steps were not enough to learn the ending at all.

The conclusion is the plain one: more data of one consistent shape, and a proper training run
from the base with a real validation split. s4 is that run.

## The evaluator was banning the end token

Before s4 was measured, the evaluator's stop handling was read again. Under `--stop eos`, when
the model sampled the end token the inner loop broke without appending it, the outer loop then
checked whether the last token was the end token (it was not) and re-sampled from the same
logits. In effect the end token was banned: every reply ran to the token cap, and everything
written after the intended stop (the repetition, the invented facts, the CJK) was the model being
pushed past its own ending. The strict lineage never saw this because it stops on "the answer is
...". Fixed in `pool_eval.py` (`ended` flag, only under `--stop eos`; the strict rule is
untouched), together with a second fix that lets gold-less prompts through under that rule so a
"should not search" set can be measured.

## s4: one run from the base on 1107 records

Corpus `chatsft/mix_v3.jsonl`: search-side 556 (198 from the GRPO prefixes, 129 + 71 from the
chat-register rewrites, 110 + 48 from the original nq forms), no-search 551 (236 + 315 teacher
pairs on real prompts). No raw strict traces. r=16, lr 5e-5, 3 epochs (378 steps), 100-record
validation every 10 steps with the best adapter kept and early stop on 6 flat validations (never
triggered); validation loss 2.323 -> 1.660; 27 min on an RTX 3060. The merged model is on the hub
(`chatsft/s4_hf`, s3 as `chatsft/s3_hf`).

Everything below is the fixed evaluator, EOS stop, reply cap 200, temperature 0.9, the same 150
held-out questions x 2, all models re-measured under the same rule.

| held-out, 300 rollouts | base (step 200) | s3 | s4 |
|---|---|---|---|
| correct and grounded | 30.7% | 31.7% | 31.7% |
| paired difference to base (150 q) | | | -2.7 pt +- 4.2 |
| grounded | 61.7% | 53.7% | 56.7% |
| searches per question | 4.85 | 3.56 | 4.05 |
| answered without searching | 2.3% | 12.7% | 8.0% |
| tags after `</think>` | 1.0% | 0.7% | 1.0% |
| reply length (words) | 7 ("The answer is X.") | 20 | 41, 2 of 300 at the cap |
| replies with CJK | 1 | 10 | 8 |

So the search ability is unchanged within noise and the reply is now two or three sentences that
end where they should ("System Shock is the 1994 PC game. SHODAN is the main antagonist of it, a
cyberpunk-horror themed video game."). Two weaknesses remain on this set. When the search never
reaches the gold (130 of 300), the reply states an answer anyway: 1 of those 130 hedges. The
corpus has no example of not finding something. And the zero-search rate went from 2% to 8%.

One question raised by the re-measurement, and settled: the base under the EOS rule scored 30.7%
against 39.0% for the same 300 rollouts under the "answer is" rule. The scoring is the same
function for both (re-scoring both files offline gives the same numbers, and cutting the EOS-rule
texts at the answer sentence after the fact changes nothing), so any difference had to be in the
generated text. Sampling at temperature 0.9 hides that, so the same 40 held-out questions were run
with argmax decoding on one card through the old loop and then the new loop: all 40 outputs are
byte-identical, with identical verdicts (20 of 40 correct and grounded under both). On the same
card argmax was deterministic 40 of 40 times. The two loops therefore produce the same text and
the same judgement for the strict model; the only mechanism that can separate them, the end token
being honoured mid-thought, did not occur in 40 argmax questions and occurred 12 times against 7
in the two sampled runs (about 2 points). The 8-point gap between the two sampled runs is
sampling variance at temperature 0.9 (two paired standard errors), and paired runs under one rule
are the only comparisons to quote. A side finding: near-greedy decoding (temperature 0.01) on two
different cards diverged mid-thought on 8 of 14 questions from floating-point near-ties, so
evaluator A/B checks need argmax on one card (`pool_eval.py --greedy 1`).

| "should not search" set, 60 real prompts | base | s4, temp 0.9 | s4, temp 0.6 |
|---|---|---|---|
| searched anyway | 60 of 60 (41 replies "The answer is X.") | 14 | 4 |
| reply answers the message (flash judge) | | 1 of 47 | 9 of 41 |
| naturalness 1-5 (flash judge) | | 1.13 | 1.66 |
| replies with CJK | | 11 | 0 |

The base cannot chat at all; s4 has learnt when not to search (93% at temperature 0.6) but the
free-form replies are weak: coherent at 0.6 ("Building in Minecraft is basically a lot of trial
and error, so the key is to keep trying ..."), mushy or invented at 0.9, with made-up citations
("M. L. Scott's 2010 study") either way. Without served text to lean on, the 1.5B model's own
knowledge is what shows, and 551 examples do not change that. This side needs its own treatment:
sampling settings (temperature, repetition control) measured together with the data, and a
judge-in-the-loop selection rather than more of the same pairs.

## The on-the-fly loop (r1–r11): every form of it spent the search ability

After s4 the plan was one box that alternates every step: sample a packed batch of search
rollouts, keep only the ones that landed on the gold, were grounded and passed the flash judge,
train on them at once, then one step of plain SFT on Dolphin-R1 records, and repeat; the held-out
set only every few dozen steps, the skip rate of the rollouts as the running metric
(`ctl/online_loop.py`, launched by `ctl/boxO.sh` / `ctl/boxP.sh` on one 3060). GPU use of this
pipeline is 0–19%: it is bound by Wikipedia and the judge, not by the card.

### r1, r2: training on the accepted rollouts collapses within tens of steps

Both runs start from s4 and train only on their own accepted rollouts (plus Dolphin). The
measurement below is over the rollouts the loop itself drew (8 per question, the same 8
questions per 20-step block, so the percentages are coarse):

| run | steps | thinking words, p90 | searches / rollout | rollouts with >5 searches | repetition or stray marker | correct & grounded |
|---|---|---|---|---|---|---|
| r1 | 1–20 | 251 | 3.9 | 36 / 376 | 20 / 376 | 24% |
| r1 | 41–60 | 1103 | 8.2 | 40 / 160 | 21 / 160 | 18% |
| r1 | 61–72 | 957 | 5.8 | 19 / 96 | 60 / 96 | 11% |
| r2 | 1–20 | 378 | 5.0 | 20 / 160 | 27 / 160 | 23% |
| r2 | 41–53 | 1062 | 8.4 | 24 / 104 | 42 / 104 | 18% |

The failure is the same in both and does not depend on the Dolphin share (r2 doubled it): the
thinking grows, the search count climbs past the cap, and the text falls into repetition loops.
Selecting the model's own sharpest samples and training on them is the GRPO dynamic without the
group baseline, so it sharpens the same way GRPO collapsed before layer and learning-rate limits
were added. The fix chosen here was not to select at all.

### r4: retention by replay, rollouts only to measure

r4 keeps the alternation but trains on a fixed set instead of on fresh samples: every step
replays two verified search traces (`replay_v1`, 672 traces of the student's own verified
rollouts, `<information>` blocks masked) and two Dolphin records; rollouts run every 10 steps
only to measure. Learning rate 1e-5, gradient accumulation 4, LoRA r16 on all layers, 400 steps
in 84 minutes.

| steps | thinking words, p90 | searches / rollout | >5 searches | repetition or marker | correct & grounded |
|---|---|---|---|---|---|
| 1–100 | 788 | 4.7 | 8 / 80 | 13 / 80 | 12% |
| 101–200 | 876 | 7.4 | 15 / 80 | 15 / 80 | 24% |
| 201–300 | 303 | 2.1 | 4 / 80 | 10 / 80 | 25% |
| 301–400 | 220 | 1.7 | 4 / 80 | 28 / 80 | 50% |

Replay cross-entropy went from 0.75 to 0.48 and Dolphin from 0.79 to 0.46 without the thinking
or the search count running away, which is the retention the replay was for. Two caveats:

- These questions were still the nq_open rewrites, which this model answers at roughly 15% and
  where half the rollouts never reach a page that carries the gold ("page not found" 45–58% of
  the outcomes). They measure the ceiling of the corpus more than the model. The measurement
  moves to the GRPO teacher pool (`pooler_distill/measureq.jsonl`, 2557 questions) from r5.
- The stray `<|begin_of_thought|` marker after `<think>` is in s4 already (20% of its rollouts)
  and 41 of the 672 replay traces carry it; replaying them raised its rate to 35% of the
  rollouts by step 400. r5 drops those traces before training (filter on the box).

### r5–r11: no selection, and it still went

The rule "do not select at all" was the answer to the sharpening, and it was followed through:
r6 onwards trained on every rollout of the step, with the gold and the teacher only counting. To
keep the card busy while the pages were fetched, r7 gave each row of the packed batch its own
question (eight questions a batch, each row left-padded with its own prefix and positions), queued
the eight, and alternated one rollout with one Dolphin record per step. The measurement moved to
the teacher's own pool, where this model answers around 45%, rather than the nq_open rewrites,
where it answers 15%.

| run | what changed | what happened |
|---|---|---|
| r5 | replay with the marker-carrying traces dropped | superseded before it measured |
| r6 | no selection at all, rollouts every step | superseded by r7 |
| r7 | eight questions a batch, queued, one Dolphin record per rollout | no-reply 5% → 59%, self-CE 0.41 → 0.06, correct and grounded 25% → 1% |
| r8 | unfinished rollouts left out of training | stopped at step 60 for r9 |
| r9/r10 | generation budget 1500 → 4000, batch budget 900 s → 2400 s | no-reply held at 1–3%, but searches per rollout went 7 → 22 and thinking p90 550 → 2400 words |
| r11 | a rollout cut at fifteen searches, and left out of training | cut share 5% → 11%, correct and grounded 45% → 36%, self-CE → 0.10 |

Two mechanisms, both worth keeping in mind:

- **Training on truncated text teaches never stopping.** A rollout that runs out of budget has no
  end token and stops mid-sentence; trained on, it makes the next rollout longer, which makes the
  next one more likely to truncate. That is the whole of r7's 5% → 59%: a loop with its own gain.
  Raising the budget and dropping the unfinished ones closed it.
- **With that closed, the drift moved to the searches.** Nothing in the objective prefers a short
  search, so the same self-imitation walked the search count up until the rollouts were 22 queries
  long. Capping and dropping them slowed it; it did not stop it.

What all of r7–r11 have in common is the objective: imitate your own samples. It has no term that
prefers a right answer over a wrong one, so the only thing it can reliably do is sharpen whatever
the model already does most, and on this lineage that is searching. Retention was not bought; the
search ability was spent, by 9 points in r11 and by everything in r7.

## The measurement settings were most of the gap

The held-out table above (base 30.7%, s4 31.7%) was taken at temperature 0.9 with a 1500-token
budget and a 200-word reply cap. The product runs none of those. Twenty held-out questions through
`s4_hf` at temperature 0.6 with a 4000-token budget:

| same 20 held-out questions | temperature 0.9, 1500 tokens, 200-word cap | temperature 0.6, 4000 tokens |
|---|---|---|
| correct and grounded | 31.7% (300 rollouts) | 60.0% |
| reached a page carrying the gold | 56.7% | 80.0% |
| searches per question | 4.05 | 3.45 |

Twenty questions carry about eleven points of error, and the two columns are not the same sample,
so the exact figures are not comparable; the direction is far outside that. The mechanism is in
the queries. At 0.9 the model invents the proper noun it is about to search for:

```
0.9: <search>Northropragh semi retired professional wrestler</search>
     <search>HalpoAtIndex center of mass person who had a civil suit against Euromas</search>
     ... and the same César query six times, 29 searches in one rollout
0.6: <search>semi-retired professional wrestler George Euripides Tragos father</search>
     <search>27th César Awards ceremony</search>          one query, page reached
```

A page lookup is a string match, so one wrong character is a miss, and sampling a name at 0.9
gets characters wrong. That is why 43% of the held-out rollouts never reached the gold. At 0.6
they reach it and the failures move to reading it: the model answers "Ibou Touray plays for
Salford City" and then adds two clubs the page never mentioned. That is a different defect, and a
1.5B-sized one.

The lesson for this lineage: measure at the settings the product runs, and treat every earlier
number here as a lower bound taken under a hotter decode.

## What the no-search side is actually worth

The same question, asked of the 60 real prompts that should not be searched:

| | base (step 200) | s4, temperature 0.6 |
|---|---|---|
| empty reply | 12 of 60 | 4 of 60 |
| under ten words | 28 of 60 | 10 of 60 |
| median reply | 6 words | 18 words |
| searches on a prompt needing none | 9.9 per prompt | 1.4 |

The base was not "1.5B-weak" on these, it was broken: it searched ten times for a chat message and
answered in six words or not at all. s4 fixed that. Sampling settings do not move what is left:
top-p 0.4 and 0.5 and repetition penalty 0.5, all at temperature 0.6, change the empty count from
4 to 1 and nothing else.

Twelve reasoning problems from the Dolphin set, at the product settings, place the remainder. The
model searched zero times on all twelve. Seven are right, including the percentage increase, the
exponent expression and the coin-weight comparison, in two or three conversational sentences. The
five failures are: a constraint it cannot hold ("three titles, each exactly four words" came back
as the same two-word title three times), code that names the right library and will not run
(`nx.Graph(graph_data.json)`), a Malay question answered in English and wrongly, a classification
task it restated instead of doing, and one problem where it thought for 3349 words and never
replied. Those are 1.5B failures, not lineage damage, except the last, which is worth fixing
because it is the worst thing a phone can do.

## The teacher that scores the samples

Twelve generations, hand-graded, then scored by each candidate judge on the same four boxes
(solves it, follows the request, English, clean):

| judge | agreed with the hand grading | cost of one 800-step run at eight samples |
|---|---|---|
| gpt-5-nano | 12 of 12 | $0.65 |
| deepseek-flash | 11 of 12 | $1.65 off-peak, $3.30 peak |
| gpt-4o-mini | 10 of 12 | $1.65 |

The hand grading was wrong once and all three judges caught it ("numbers between 5 and 9 that are
greater than 7" is 8, and the model answered "8 and 9"); the table above is after that correction.
The cheapest judge was also the most accurate, so it is the one wired in. Keys reach the box as
instance environment and are written to a 600 file there; none of them is in this repository.

## g2: on-policy on the reasoning problems, group-normalised

The run now on the card is not the self-imitation that r7–r11 were. Eight samples per Dolphin
problem at temperature 0.6, the teacher scoring each against the reference, the advantage measured
against the group's own mean, so a sample below the mean is pushed down rather than ignored. A
group whose eight samples all score alike carries no signal and is skipped, which is three or four
steps in ten. One Dolphin record of ordinary SFT rides along with each step.

## g3–g6: both sides on-policy, and why g5 lost the search side

g3 (lr 1e-5, 280 steps) barely moved the weights: the adapter came to 0.031% of the base norm and the
behaviour did not change. g5 raised the rate to 1e-4 and by step 200 the adapter was 0.284% and the
model was visibly different (working code, longer thinking, 4/12 on the twelve check problems against
6/12 for its seed). It then lost the search side between steps 220 and 260 with no guard running:
the guard of that time lived in the search-corpus loop, which the GRPO mode never enters.

The full step record says what happened, and it was not more searching. The median search count
stayed at one. What grew was the thinking of the *wrong* search rollouts: 176 words in steps
180–209, 206 in 210–239, 436 in 240–269, while the correct ones stayed at 130–176. Unfinished
rollouts went from 12 to 44 to 56 per 180. The reasoning side, at 75% pass, never moved. The
gradient itself was measured too: on both sides, in every block, the advantage-weighted sum of
length and of search count was negative, so the reward was not asking for length.

The path it runs away along is the loss normalisation. Each rollout's policy-gradient loss is the
mean over its own tokens, so a wrong rollout's negative advantage is spread over however many
tokens it wrote: a long wrong rollout is punished less per token than a short one, and a side
that is wrong two times in three drifts toward being long when wrong (the length bias the Dr.
GRPO paper describes). But the search GRPO that made this model had exactly that normalisation
and did not run away, so the bias is the road, not the driver. What g5 had changed against that
run was the two limits that had stopped the search GRPO's own early collapse: LoRA on all 28
layers instead of 20–27, and a rate of 1e-4 instead of 1e-5. Every online run since r1 had been
on all layers; at 1e-5 that never showed because the weights barely moved.

g6 therefore puts the search side back under the search GRPO's conditions and lets only the
reasoning side differ. Search: LoRA on layers 20–27, its own Adam at 1e-5 with its own moments,
temperature 0.9, a 1500-token cap, questions from the same corpus pool (2857, gold of at most
six words, the 300 held-out removed), the per-rollout mean and std normalisation as before, no
wheels; the original reward plus the 0.5 conversational bonus, and the cut at the seventh
search. Reasoning: the same adapter, its own Adam at 5e-5 which also carries the one Dolphin
record per step, temperature 0.6, a 7000-token cap, the R1 reference when all twelve samples
fail. One optimizer steps per loop step. The constant-length normalisation stays in the code as
a flag, off. The guard runs inside the GRPO loop as the safety net.

## g7–g10: the reasoning side loses the search side however it is held

g7 ran g6's plan from s4: the search side under the search GRPO's own conditions, the reasoning
side on the same adapter at 2e-5. On the held-out it went 35.3% at step 120, 40.2% at 470, 37.3%
at 600 (s4: 38.0%) — level, not up — while the training-time search pass rate sagged over the
later hundreds of steps. g8 moved everything to all-layer LoRA at 1e-5, three search steps per
reasoning step, the Dolphin record only on reasoning steps, and a KL anchor to the base policy
(k3 with the adapters off, 0.04); g9 raised the anchor to 0.2 and dropped the Dolphin record.
Each of them drifted the same way: the thinking got longer, then the search pass rate fell, and
the guard's healthy copy was the only thing worth keeping. Layer set, rate, KL weight, and the
Dolphin record were each varied on their own; none of them changed the direction.

g10 is the control: the search side alone, from s4, under the search GRPO's recipe (layers
20–27, the pooler's rank-8 adapter, 1e-5, temperature 0.9, 1500-token cap, corpus questions,
mean and std normalisation). It is stable and it climbs: 43.1% on the held-out at step 400
against s4's 38.0%. Its step 400 folded into s4 is `chatsft/g10m_hf`, the base of every run
since. So the search GRPO is not the fragile part; anything that trains the reasoning side by
policy gradient alongside it is.

## g12–g14: growing the reasoning side without policy gradient

The reasoning side's own ceiling under RL was the first thing to measure. On the training
problems the twelve samples pass at least once about 65% of the time; rejection-sampling
fine-tuning or GRPO can only sharpen inside that set, and the held-out hundred (dolphin_v1
minus v2, seed 0, never trained on) sat at 56% for s4 under the nano judge (58% under
DeepSeek). The judge itself was audited before it was trusted for this: on the same replies it
agreed with a strict gpt-5-mini reading 91–96% of the time, so the score is not the judge's
noise.

g12 was rejection-sampling fine-tuning on the Dolphin problems from g10m_hf: the shortest
passing sample of twelve as plain SFT, a search trace replayed at weight 0.5 each step, and the
R1 reference as the fallback when all twelve failed. The fallback is what moved it: the R1
references are long (median 718 words of thinking) and the student took on their length —
174 to 612 words by step 116 with 28% of samples unfinished — without taking on their
correctness. g13 limited the fallback to short references and was stopped for the same reason.

g14 dropped the rollouts and distilled directly: the R1 thinking and reply of every dolphin_v1
record outside the held-out hundred, eight records a step, all-layer LoRA r16 at 5e-5, the
search trace replay kept at 0.5, one epoch (835 steps). The distillation loss fell 0.62 → 0.52
and the replay loss 0.79 → 0.55, the latter's drop coming after the 630 traces had been seen
once, so that side is partly memorised. On the held-out, measured with a repetition-loop
breaker that closes the thinking and answers when the generation starts repeating:

| | s4 | g10 step 400 | g14 step 400 | g14 step 835 |
|---|---|---|---|---|
| search held-out (eval300 subset, 102) | 38.0% | 43.1% | – | 48.0% |
| Dolphin held-out (100, nano) | 56% | – | 52% (22 unfinished) | 53% (4 unfinished) |
| thinking, median words | 153 | – | 900 | 576 |

The search side did not pay for the replay's memorisation: 48% is the best number the held-out
has given (the three shards read 59/38/47%, so the noise is about ±5 points). The reasoning side
did not gain: per problem, s4 and g14 both solve 46, s4 alone 10, g14 alone 7, neither 37. The
distillation changed the length of the thinking and not the set of problems the model can do.

The 37 that neither solves say where the target lives. The held-out hundred is 61 maths, 26
code, 13 general; the maths sits at 72–77%, the code (Erlang, Lisp, Rust, Swift, SQL, R) at
23–30%, the general (image prompts, articles, explanations) at 7–23%. An 85% on the whole set
needs the code and general parts near 80%, which is a different question from whether the
maths can be sharpened. Whether a 1.5B model has that at all was then measured directly, the
untouched R1 distill and the step-200 student on the same hundred under the same settings:

| | untouched distill | step-200 student | s4 | g10 step 400 | g14 step 835 |
|---|---|---|---|---|---|
| Dolphin held-out | 56% (1 unfinished) | 46% (12 unfinished) | 56% | 55% | 53% |
| maths / code / general | 45 / 6 / 5 | 40 / 4 / 2 | 47 / 8 / 1 | 44 / 9 / 2 | 44 / 6 / 3 |
| thinking, median words | 701 | 366 | 153 | 149 | 576 |
| search held-out | – | – | 38.0% | 43.1% | 48.0% |

The untouched model is at 56%: our training lost nothing. (The step-200 student's 46% is the
old stop rule leaving twelve replies unfinished, which the chat SFT repaired.) The four models
together solve 68 of the hundred and 32 by none. The reasoning target is therefore outside this
model, not something to recover: 85% on this mix is thirty points above what the 1.5B distill
does on its own, most of them in code and general tasks it does not do at any weight. The
coexistence question, on the other hand, is answered by g14: a single model at 48% search and
base-level reasoning. What 85% needs is a larger reasoning model, and the router-and-two-models
design is then a 1.5B search model beside a 7B reasoning model, not two of the same size.

## Next, in order

1. The untouched 7B distill on the same hundred (a 24 GB card for two hours), to know what a
   larger reasoning model buys on this mix before committing to it.
2. If it is worth it: a task router in front of the 1.5B search model (g14 lineage) and the 7B
   reasoning model, each kept under the conditions that made it.
3. The search trace pool for any further replay comes from g10's verified rollouts, not the 630
   traces again.
4. Only then the 4-bit packing of the chosen checkpoint (`packmlx.py`) and the app.
