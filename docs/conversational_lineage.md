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

## Next, in order

1. Top up the DeepSeek account; finish the 162 questions with the continuity prompt.
2. Expand the questions with nq_open through the box's generation mode (about $1.1 per 1000
   questions), rewritten into chat register first (see above), so the corpus is 1000+ questions of
   the kind people actually type into an assistant.
3. Gentler fine-tune with replay: r=16, lr 3e-5, 2 epochs, and the strict single-sentence QA traces
   mixed in at about 1:1 so the search reflex is preserved; stop by validation loss. Running as s3
   (`ctl/boxS.sh`) on the 563-record mix `chatsft/mix_v1.jsonl` (search v3 198, para0 56, para1 73,
   chat v1 236; held-out overlap checked: none) plus 300 replayed strict traces.
4. Change the evaluator's stop rule for this lineage (stop at EOS, cap the reply) and add two
   discipline metrics to every measurement: zero-search answers and tags after `</think>`.
5. Then rejection sampling with the gold reward plus pairwise judging on the student's own samples,
   and only after that multi-turn.
