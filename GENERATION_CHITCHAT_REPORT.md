# Report — chitchat / greeting turns generate off-topic content (for the model-generation owner)

**Date:** 2026-06-11
**From:** app-implementation side (MLX-swift port)
**Scope boundary:** This is a **generation-behavior** report, not an app/port bug. The app side has
ruled out translation, routing, retrieval, and the SP-evict port (evidence below). Handing off to
whoever owns model/generation quality.

---

## 0. Symptom

A casual/greeting turn produces unrelated technical content instead of a normal reply.

| Input (as typed) | What the model received | Output |
|---|---|---|
| `こんにちは` (device) | `Hello` (translated, see §2) | a statistics lecture: *"1. **Mean** is used when you need an average value… 2. **Median** is used when the data is skewed…"* |
| `Hello` (Mac, English direct) | `Hello` | **`12.`** |

Both are off-topic. The greeting carries no task, and the model fabricates one (a number, or a
stats explanation).

## 1. The exact app→model contract for this turn

A `chitchat` turn (no memory hit, trigger not fired) is sent to the model **verbatim, unframed**:

- **Prompt fed to the LLM** (firstTurn, empty SP history):
  ```
  <｜User｜>Hello<｜Assistant｜><think>\n
  ```
  (i.e. `aug = userMsg` exactly — no system prompt, no instruction; canonical
  `app_session_torch.turn`’s final `else: aug = user_msg`.)
- **Generation:** SP-evict `_gen_once`, `temp=0.6` inside `<think>`, **greedy (1e-4) after
  `</think>`**, `rw=512`, `maxD=4096`, chunk `C=64`, `genLen=800`, `seed=0`.
- **DecodePolicy:** think-phase temp 0.6; convergence force-close at k=3 repeated asserted value;
  6-gram×4 loop guard; 2-pass salvage if `</think>` never closes.
- **Post-processing:** calculator repair runs on `math`/`command` only (NOT chitchat), so the output
  is the raw model text. `answerOk` only rejects degenerate/garbage strings — `"12."` and the stats
  lecture both pass.

So the model receives a bare greeting and is asked (by the `<think>` opener) to reason before
answering.

## 2. App-side causes ruled out

- **Translation is not involved.** The Mac repro uses the English string `Hello` directly (no
  translation shim) and still yields `12.`. On device, `こんにちは` translated to `Hello` correctly
  (verified) — the off-topic answer is downstream of a correct translation.
- **Routing is correct.** Intent = `chitchat` (appropriate for a greeting). No retrieval is done, so
  no wrong context was injected.
- **The retrieval trigger did NOT fire.** Logged `trigger=0.00` (first turn, empty history) → no
  transcript/context was injected. The output is the model’s own, not injected material.
- **SP-evict history is empty.** The preceding turn was a `lookup` (isolated clean-quote), which does
  not update the SP-evict `genState`; so this turn ran with `firstTurn=true`, empty SP. No stale
  conversation polluted it.
- **The SP-evict port is parity-verified** (separate `GENONCE_DEGENERATION_REPORT.md`): same path
  produces correct math (`47×9 → 423`, `650−200 → 450`) and coherent long CoT. It is not degenerate
  here — the output is *fluent and well-formed*, just **off-topic**.

Net: the model is given a clean `Hello` and freely generates `12.` / a stats lecture.

## 3. Hypothesis (for the generation owner)

`DeepSeek-R1-Distill-Qwen-1.5B` (FFT) is a **math/reasoning distillation**. Handed a greeting with
no problem to solve, the forced `<think>` phase appears to **invent a task** (pick a number; explain
mean vs median) because its training distribution is reasoning/QA, not open-domain small talk. The
output is on-distribution for the *model* (fluent reasoning prose) but off-topic for the *user*.

This is a **generation / training-distribution** matter — candidate levers all sit on your side:
- chitchat/instruction data in the FFT mix, or a small-talk-aware decode policy;
- whether a contentless turn should even enter `<think>` (a “no-reasoning” path for greetings);
- a generation-side system framing (the app deliberately sends none — by design the app does not
  editorialize the model’s contract, per the role split).

The app side will **not** patch this with prompt-engineering hacks (we reverted an experimental
chitchat framing) — that’s your call. If you decide a framing or a route-specific prompt is the fix,
tell us the exact contract and we’ll wire it verbatim.

## 4. Reproduction (model-side, no app needed)

Greedy, no guards, same weights:
```
prompt = "<｜User｜>Hello<｜Assistant｜>" + "<think>\n"
# generate; observe the think + answer
```
Expected: a fabricated numeric/technical answer rather than a greeting. Vary with `temp=0.6` in
think to see the stats-lecture style. Compare against feeding an explicit small-talk framing to
confirm whether the issue is the bare contract or the model itself.

## 5. What the app guarantees (so you can trust the inputs)

- The string reaching the model is exactly `<｜User｜>{user_or_translated_text}<｜Assistant｜><think>\n`.
- For Japanese input, the text is translated EN first (verbatim spans protected); translation is
  verified independently.
- No retrieved context is injected unless a tier hit or the trigger fired (logged per turn:
  `route=…  trigger=<score>`).
