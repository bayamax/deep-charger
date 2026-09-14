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

The no-search side has two sources: R1 answering real prompts (236 verified pairs,
`chat_sft_v1.jsonl`) and the lineage's own base writing K candidates that the flash judge scores.
On the first 68 prompts of the K=6 run, 201 of 408 candidates were usable and about one prompt in
ten got a candidate that passed (the base answers small talk with "I'm here to help" boilerplate,
naturalness 2-3), so that path is a supplement, not the main source.

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

## Next, in order

1. Top up the DeepSeek account; finish the 162 questions with the continuity prompt.
2. Expand the questions with nq_open through the box's generation mode (about $1.1 per 1000
   questions), rewritten into chat register first (see above), so the corpus is 1000+ questions of
   the kind people actually type into an assistant.
3. Gentler fine-tune with replay: r=16, lr 3e-5, 2 epochs, and the strict single-sentence QA traces
   mixed in at about 1:1 so the search reflex is preserved; stop by validation loss.
4. Change the evaluator's stop rule for this lineage (stop at EOS, cap the reply) and add two
   discipline metrics to every measurement: zero-search answers and tags after `</think>`.
5. Then rejection sampling with the gold reward plus pairwise judging on the student's own samples,
   and only after that multi-turn.
