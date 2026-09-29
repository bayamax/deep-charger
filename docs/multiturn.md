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
