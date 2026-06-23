# Report — multi-turn retrieval trigger: over-fires + referential follow-ups degrade (for the model/generation owner)

**Date:** 2026-06-12
**From:** app-implementation side (MLX-swift port, on-device iPhone 16e / iOS 26.5)
**Re:** the attention-mass retrieval trigger (`attn_trigger3_head` + `trigger_runtime`) we ported per
`APP_SPEC.md §2/§4`, to enable the "雑談マルチターン" you pointed us to.

This is a **findings + calibration** report. The greeting fix you diagnosed (skip the SP block while
`kept` is empty) is **confirmed working on device** (`こんにちは` → a normal greeting; `Hello` → a
normal greeting, no more `12`). The remaining multi-turn quality issues are below — several look like
they sit on the generation / trigger-training side, so we're handing the data over.

---

## 1. What we implemented (so you can judge the contract)

- Ported `TriggerScorer`: one eager forward over `[BOS | SP | turn-tokens]`, read attention mass at
  the 6 trained `(layer,head)` pairs `[[1,0],[1,3],[21,4],[7,9],[22,9],[9,3]]`, standardize by
  `mu`/`sd`, logistic (`coef`,`intercept`), threshold **0.35**. Mass per pair =
  `mean over turn-rows ( sum over the 32 SP columns of attn )`. (Matches
  `trigger_runtime._masses` structure.)
- **The SP we feed it** = `pooler(conversation_tokens)`, where `conversation_tokens` is the running
  `User: … / Assistant: …` transcript of the session (capped 6000 tok). **This is our choice and is
  probably the crux — see §3.**
- On fire (score ≥ 0.35) we re-inject the recent transcript as context for the turn.

## 2. Observed behaviour on device (real transcript, JP via the translation shim)

| turn (user) | route | trigger | result |
|---|---|---|---|
| 今日の原油価格は？ | lookup/web | 0.92 **FIRE** | $99.29/barrel ✓ |
| え？本当？ | chitchat/↩︎ctx | 1.00 **FIRE** | repeats "$99.29/barrel" (ok-ish) |
| もう少し調べてもらえますか？ | chitchat/↩︎ctx | 1.00 **FIRE** | repeats "$99.29/barrel" (weak) |
| その数字が正しいか検証して | lookup/web | 1.00 **FIRE** | ❌ *"telephone number verification services…"* |
| じゃなくて原油価格の話だったんですが | lookup/web | 0.99 **FIRE** | ❌ *"Heavy crude oil is API gravity < 20°"* |
| 重油じゃなくて原油です。原油の価格。今日のね？ | lookup/web | 0.42 **FIRE** | ❌ *"Gulf Oil"* |

Mac isolation test (clean, English, no web):

| turn | trigger | answer |
|---|---|---|
| "The capital of France is Paris." | 0.000 (no history) | Paris |
| "Are you sure about that?" | **0.990** | "Yes, I am certain that Paris is the capital of France…" ✓ |
| "What is 12 times 3?" | **0.998** | 36 |
| "Is that answer correct?" | **1.000** | "Yes, all the answers are correct…" ✓ |

## 3. Finding A — the trigger over-fires (non-discriminative)

**Almost everything scores 0.9–1.0**, including turns that are NOT referential: a fresh
`What is 12 times 3?` scored **0.998**; a fresh `今日の原油価格は？` scored **0.92**. The operating
table you shipped (`trigger_runtime.__main__`) expected control turns to sit *below* ~0.4 (FPR ≈ 0.10
at th 0.4). Ours don't separate.

Our standardized features are evidently far above `mu` (`mu` ≈ [0.31, 0.71, 0.33, 0.64, 0.52, 0.57]),
so `z` saturates the sigmoid.

**The likely cause is the SP we feed.** `trigger_runtime.score(sp, turn_ids)` takes `sp` as an
argument; the probe set `attn_probe3.npz` paired each example with a specific `sp_i`. **What SP scope
was the head trained on?**
- the *evicted/distant* soft prompts only (i.e. SP over `kept`, excluding the raw window), or
- the SP over the *whole* conversation (what we currently feed)?

If the head was trained on distant-only SP, our whole-conversation SP makes the turn attend to the SP
much more (recent turns are *in* the SP rather than in a separate raw window), which would push every
mass above `mu` and fire on everything. We can switch our SP construction to match — **please confirm
the intended SP scope** (and ideally share `attn_probe3.npz` / `attn_trigger3.joblib` so we can
re-verify masses against your reference; we currently can't, having only the exported
`attn_trigger3_head.npz`).

## 4. Finding B — referential follow-ups web-search the vague text

When a follow-up *does* route to `lookup` ("検証して", "もう少し詳しく", "原油の価格"), the pipeline
runs a fresh web search on the literal (translated) follow-up, which has no standalone meaning, so it
returns junk ("verify the number" → telephone-number-verification; "crude oil price" →
heavy-oil-API-gravity / "Gulf Oil"). `expand_web_query`'s anaphora resolution isn't pulling the topic
("crude oil price") forward from the prior turns.

We tried an app-side patch (referential lookups answer from the conversation instead of the web) but
**it regressed** — *because* the trigger over-fires (§3), it also hijacked genuine new lookups. We
reverted it. A reliable fix needs the trigger to actually discriminate (§3), or stronger anaphora
resolution upstream.

## 5. Finding C — what's confirmed good

- **Greeting fix works** (your `kept`-empty SP-skip): `こんにちは`/`Hello` → normal greeting.
- **Single-shot lookups** translate and answer correctly ($99.29/barrel → 日本語).
- **Referential confirmations** ("Are you sure?", "本当？") *do* pull context and answer about the
  right topic (Mac + device) — the mechanism is sound when the turn stays in chitchat.
- Math unaffected (47×9 → 423, 650−200 → 450).

## 6. Questions for the generation / trigger owner

1. **SP scope for the trigger head**: distant-only (over `kept`) or whole-conversation? (§3 — most
   likely the over-fire root cause.)
2. Can you share `attn_probe3.npz` + `attn_trigger3.joblib` so we can diff our eager-attention masses
   against your reference and confirm fidelity of our QK port?
3. Is the threshold still 0.35 for the deployed head, or should it move given the SP we feed?
4. For vague referential follow-ups, is anaphora/topic-carry expected to be solved by the trigger
   (pull past into the window) alone, or also by a stronger query rewriter before retrieval?

## 7. App-side guarantees (so you can trust the inputs)

- The trigger input is exactly `[embed(BOS) | SP | embed(turn_ids)]`, fp32 attention math, eager
  softmax, layers 0–22, mass read at your 6 `(layer,head)` pairs. Greedy generation otherwise
  unchanged.
- We log per turn `route=… trigger=<score>` and the full EN-core + translated answer; full transcript
  (`convo_log.txt`) available on request for any repro.
- We will wire whatever SP scope / threshold / rewriter contract you specify, verbatim.
