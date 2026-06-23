# RESOLVED — SP-evict `,1!#!!!!` degeneration was a low-temperature sampling overflow in the Swift port

**Date:** 2026-06-11 (updated — root cause found & fixed)
**Component:** MLX-swift port `SPModel.sample` (answer-phase sampling), not the SP-evict machinery
**Status:** ✅ fixed and verified on macOS (Release, Metal GPU)

---

## 0. TL;DR

The `,1!#!!!!` / `)1!!!!!` degeneration is a **Swift-port bug in the sampler**, isolated to the
**greedy-after-`</think>` answer phase**. It is **not** the model, **not** the SP-evict recipe,
and — importantly — **not** the cause the model-dev RESPONSE hypothesised (empty-past pooler
softmax→NaN). My port already guards the empty-past pooler and matches the PyTorch reference
`sp0` to **1.79e-7**.

**Actual bug:** `DecodePolicy.temp` returns `1e-4` for the answer phase (greedy-after-think).
The port routed that through `MLXRandom.categorical(logits / 1e-4)` = `categorical(logits × 1e4)`.
PyTorch's `softmax(logits/1e-4)` is **max-stabilised** → collapses to the argmax. MLX's
`categorical` is **not** max-stabilised at that scale → the softmax overflows to a NaN
distribution → it emits garbage tokens (`)`, `1`, then a `!` lock-on).

**Fix (one line):** treat near-zero temperature as argmax in `sample()`:
```swift
if greedy || temp < 1e-2 { return last[0].argMax().item(Int.self) }   // was: temp <= 0
```
This is the faithful, overflow-safe equivalent of PyTorch's stable `softmax(logits/1e-4)`.

## 1. Why it looked like an SP-evict / model problem (and wasn't)

- It was **prompt-dependent**: `47×9` → `423` (the first answer token happened to dodge the
  overflow / coincide with argmax), but `650−200` → `)1!!!!`. Prompt-dependence wrongly suggested
  the SP compression or the recipe.
- The decisive artifact was the **full body dump** of the real `_gen_once`:
  ```
  <think>
  First, I start with the total amount of $650.
  ... The calculation is $650 - $200 = $450.
  Therefore, after the purchase, I have $450 remaining.
  </think>)1!!!!!!!!...
  ```
  The think is **perfect** (reaches $450, closes `</think>` itself). The garbage is **only**
  after `</think>` — i.e. in the answer phase, which is exactly where `temp` drops to `1e-4`.

## 2. Parity evidence (all measured on macOS, Metal GPU)

Diffed against the model-dev `parity_reference.npz` and staged isolation harnesses:

| Stage | What it exercises | Result |
|-------|-------------------|--------|
| ① empty-past pooler | candidate-#1 NaN suspicion | **NaN=0, norms=1.4685306, max\|Δ vs sp0\|=1.79e-7** ✓ |
| ② SP machinery, prompt **prefilled**, greedy | block-injection / cache-crop / RoPE / mask | **token-identical to torch** (first divergence = none) ✓ |
| ③ `_gen_once` forced-feed structure, greedy | MQ=1, prompt as forced feed after empty-SP | coherent ✓ |
| ④ forced-feed + temp=0.6 | think-phase sampling | coherent ("To solve 47 multiplied by 9, I can use the distributive property…") ✓ |
| ⑤ forced-feed + temp=0.6, **rw=16** | **non-empty-kept** pooling (real-history compression) | coherent ✓ |
| ⑥ "650−200" prompt, temp-only & greedy-after | answer phase with **argmax** | coherent, reaches **$450** ✓ |
| ⑦ "650−200" via **real `_gen_once`** | answer phase via `categorical(logits×1e4)` | **DEGENERATE `)1!!!!` ← the bug** |

Stage ⑥ (argmax answer phase) clean vs stage ⑦ (`categorical(×1e4)` answer phase) garbage, on the
same prompt, pins the fault to the sampler's low-temp path.

## 3. Note for the model-dev side

Two corrections to the earlier RESPONSE, for your records:

1. **The empty-past pooler guard was NOT missing in the Swift port.** `Pooler.run` already has
   `if L > 0 { …cross-attention… }`, and the empty-past SP matches your `sp0` to 1.79e-7 with the
   norms exactly 1.468531. Candidate #1 did not apply here.
2. **The real fault is environmental to MLX**: `MLXRandom.categorical` is not numerically
   equivalent to `torch.softmax(logits/T)+multinomial` when `T` is tiny (logits get scaled by
   `1/T=1e4`). Any future MLX port of a greedy-via-tiny-temperature policy needs `T→argmax`
   clamping (or a max-subtraction before `categorical`). Worth a one-line note in
   `decode_policy` / the porting guide: *"the answer-phase temp of 1e-4 must be realised as
   argmax on MLX, not as categorical(logits/1e-4)."*

`parity_reference.npz` and `sp_evict_parity.py` were instrumental — thank you. The empty-past
`sp0` check (stage ①) ruled out the pooler in one shot and forced the search downstream.

## 4. Verification

`SP_ASK="If I have 650 dollars and I spend 200, how much do I have left?"` → before: `)1!!!!`;
after the fix: `**Solution:** … $650 − $200 = $450 … **Answer:** You will h…` (correct).
`47×9` → 423 still correct.
