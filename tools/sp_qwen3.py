"""deep-charger's SP-evict hypernet, ported to a new base (Qwen3-1.7B): same design, the numbers as arguments.

Design (train_sched_evict.py / pooler_mlx.py, unchanged):
  * AttnPoolSP: S learned queries cross-attend to the token embeddings (+ sinusoidal PE) of the distant buffer, then
    self-attention and an FFN, `layers` blocks; output LayerNorm -> L2-normalised -> x |out_scale| (initialised to the
    mean norm of the model's token embeddings): the soft prompts live on the embedding sphere.
  * distant buffer: tokens that left the raw window, at most maxD, evicted by the pooler's own cross-attention mass.
  * generation / training loop per chunk of C tokens: cache cropped to the prompt, re-prefill [SP | raw window (rw)],
    then the chunk.
  * objective: KL(full-context teacher || SP student) on every token of the chunk, teacher-forced on the model's own
    full-KV traces.
Stage 1 trains the pooler with the model frozen; --ft-layers N also full-fine-tunes the first N decoder layers
(N = all layers: FFT) with fp32 master weights (+ --opt8bit for 24 GB cards); the teacher is the frozen copy.

  python sp_qwen3.py train --traces traces.jsonl --skip 60 --n 400 --S 32 --rw 768 --C 128 --maxd 384 --out sp.pt
  python sp_qwen3.py eval  --traces q17_full.jsonl --ckpt sp.pt          (KL / top-1 agreement vs full KV)
  python sp_qwen3.py grpo  --traces grpo_prompts.jsonl --ckpt spG.pt --ft-layers 28 --opt8bit --grpo-eval math96.jsonl
                     (GRPO under the deployment loop: the reward is a correct final answer, see the --group/--grpo-* flags)
"""
import argparse, json, math, os, random, subprocess, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["train", "eval", "profile", "rollout", "grpo", "probe", "massprof", "oracle", "kvsel"])
ap.add_argument("--kvsel-budgets", default="512", help="kvsel: budgets B of older entries kept beside the newest rw (comma list)")
ap.add_argument("--kvsel-rules", default="recent,random,cum,mean,ema0.9,ema0.98,last,grad", help="kvsel: selection rules compared")
ap.add_argument("--kvsel-every", type=int, default=4, help="kvsel: every k-th chunk boundary of a trace")
ap.add_argument("--kvsel-dump", default="", help="kvsel: save per position the older entries' features and labels (grad, abl) to this .pt - the training set of a learned selector")
ap.add_argument("--kvsel-abl", type=int, default=8, help="kvsel rule abl: exact ablation in blocks of this many consecutive older entries (the KL-based label, costly)")
ap.add_argument("--kvsel-layerwise", type=int, default=1, help="kvsel: also each attention rule with a per-layer choice (each layer keeps its own top B)")
ap.add_argument("--oracle-steps", type=int, default=200, help="oracle: Adam steps on the 32 slots themselves at each position")
ap.add_argument("--oracle-lr", type=float, default=1e-3, help="oracle: Adam lr on the slots' pre-normalisation parameters (unit rows, so a coordinate is ~0.02)")
ap.add_argument("--oracle-pos", default="0.5,0.85", help="oracle: positions per trace as fractions of its chunk starts")
ap.add_argument("--oracle-horizon", type=int, default=0, help="oracle: fit the slots on the next K chunks jointly (the raw window fixed at the "
                "position, the block growing to K chunks, so nothing is missing) and score them on the 2 chunks after those - a chunk-sized fit "
                "encodes the chunk's own text and is worse than the pooler one chunk later; a summary must carry further")
ap.add_argument("--oracle-content", type=int, default=0, help="oracle (horizon mode): also measure how much the pooler's output depends on its "
                "input - cosine to its output on the same tokens shuffled, and on another trace's tokens (the first trace of the file, or the second)")
ap.add_argument("--oracle-chain", type=int, default=0, help="oracle (horizon mode, --oracle-run n): the fit at each position starts from the PREVIOUS "
                "position's optimum (the first from the pooler's output) - the minimal-movement trajectory, one solution per chain instead of an "
                "arbitrary member of the solution set; its neighbour statistics ask whether a lawful memory exists")
ap.add_argument("--oracle-random", type=int, default=1, help="oracle: 0 skips the random-seeded reference fit (half the time; its fields repeat the pooler's)")
ap.add_argument("--oracle-run", type=int, default=0, help="oracle: instead, this many CONSECUTIVE chunk starts from the trace's middle - how the "
                "pooler-seeded optima of one chain of thought relate to each other (cosine, the correction's direction, the previous "
                "optimum's KL on the next chunk): a smooth target a pooler can learn, or unrelated solutions")
ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
ap.add_argument("--traces", required=True)
ap.add_argument("--n", type=int, default=0); ap.add_argument("--skip", type=int, default=0)
ap.add_argument("--max-len", type=int, default=3072, help="trace tokens used after the prompt")
ap.add_argument("--S", type=int, default=32, help="soft prompts")
ap.add_argument("--rw", type=int, default=768, help="raw (exposure) window")
ap.add_argument("--C", type=int, default=128, help="chunk")
ap.add_argument("--maxd", type=int, default=384, help="distant buffer cap (tokens)")
ap.add_argument("--layers", type=int, default=3); ap.add_argument("--heads", type=int, default=8); ap.add_argument("--ffn", type=int, default=2048)
ap.add_argument("--lr", type=float, default=1e-4, help="pooler lr")
ap.add_argument("--ft-layers", type=int, default=0, help="also full fine-tune the first N decoder layers")
ap.add_argument("--ft-lr", type=float, default=1e-5)
ap.add_argument("--opt8bit", action="store_true")
ap.add_argument("--accum", type=int, default=8, help="traces per optimizer step")
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--ckpt", default="")
ap.add_argument("--out", default="")
ap.add_argument("--log", default="")
ap.add_argument("--chunks-per-trace", type=int, default=0, help="train: back-propagate only this many random chunks per trace "
               "(the rest only advance the buffer) - more distinct traces per hour (0 = every chunk)")
ap.add_argument("--par", action="store_true", help="train: one forward of [prompt | SP(buffer at boundary b) | the next rw+C "
               "tokens] scores EVERY next-token prediction in it - window 0 (right after the soft prompts) up to rw+C - instead of "
               "one chunk after a full window; --chunks-per-trace boundaries per trace")
ap.add_argument("--spnext", action="store_true", help="train: window 0 only - [prompt | SP_t] -> token t for (nearly) every "
               "position t of a trace, in batches of --sn-batch positions (memory fixed by the batch, not the trace)")
ap.add_argument("--sn-step", type=int, default=16, help="spnext: buffer snapshot granularity (tokens)")
ap.add_argument("--sn-batch", type=int, default=64, help="spnext: positions per forward")
ap.add_argument("--sn-tokens", type=int, default=16384, help="spnext: cap on batch x (prompt + S) - the prompt KV is "
               "repeated per batch item, so long prompts get smaller batches")
ap.add_argument("--sn-pos", type=int, default=0, help="spnext: positions sampled per trace (0 = all)")
ap.add_argument("--sn-win", default="1", help="spnext curriculum: tokens predicted after SP_t (1 = window 0 only; w = SP_t + the next "
                "w-1 raw tokens, scoring windows 0..w-1), comma list stepped every --sn-win-every traces, e.g. 1,2,4,8,16")
ap.add_argument("--sn-win-every", type=int, default=150, help="spnext curriculum: traces per --sn-win step")
ap.add_argument("--sn-win-mix", action="store_true", help="spnext: instead of a curriculum, draw w from the --sn-win list at random "
                "for every trace - window 0 and the longer windows trained together")
ap.add_argument("--sn-win-plateau", type=float, default=0.0, help="spnext curriculum: step to the next w when the last 100 traces' "
                "KL is not this fraction below the 100 before (0 = fixed --sn-win-every); after the last w plateaus, training stops")
ap.add_argument("--sn-win-block", type=int, default=4, help="spnext plateau rule: logs (25 traces each) per compared block")
ap.add_argument("--sn-win-max", type=int, default=800, help="spnext curriculum with --sn-win-plateau: most traces at one w")
ap.add_argument("--sn-t0", type=int, default=16, help="spnext: first answer position trained")
ap.add_argument("--gen-kl", type=int, default=0, help="eval: also let the student WRITE the next --gen-len tokens itself (deployment sampling) "
                "from its state at G evenly spaced chunk starts per trace, the full-context teacher reading the true text up to the chunk "
                "start and then the student's tokens; KL per position on the student's own path (teacher forcing never shows the "
                "student its own mistakes). Only those G chunks are scored, teacher-forced too")
ap.add_argument("--gen-len", type=int, default=128, help="--gen-kl: tokens the student writes per chunk")
ap.add_argument("--gen-force", action="store_true", help="--gen-kl check: feed the true text instead of samples (the per-position KL "
                "must then equal the teacher-forced KL of the chunk)")
ap.add_argument("--score-from", type=int, default=0, help="count only answer tokens at index >= this (a common region when rw differs)")
ap.add_argument("--pe-rev", action="store_true", help="pooler positions counted from the NEWEST buffer token (0 = the token "
               "just before the window / the predicted position) instead of from the oldest - saved in the checkpoint")
ap.add_argument("--pe-norm", action="store_true", help="scale the pooler's sinusoidal positions to the mean embedding norm "
                "(raw sinusoids have norm sqrt(H/2) = 32 against token embeddings of ~1.5: the content was ~5%% of the key/value signal)")
ap.add_argument("--fresh-pooler", action="store_true", help="train: take only the fine-tuned model weights from --ckpt, start a new pooler")
ap.add_argument("--roll-out", default="", help="rollout: write the model's own SP-evict generations here (jsonl prompt/text) - "
                "the original deep-charger's on-policy (DAgger) data: train on what the bounded model itself produces")
ap.add_argument("--roll-batch", type=int, default=24, help="rollout: streams generated together")
ap.add_argument("--roll-cb", action="store_true", help="rollout: the continuous-batching generator (rollout_cb) - all "
                "prompts in one call, --roll-batch streams at a time, a finished stream's slot refilled at once")
ap.add_argument("--temp", type=float, default=0.6); ap.add_argument("--top-p", type=float, default=0.95); ap.add_argument("--top-k", type=int, default=20)
ap.add_argument("--end-bias", type=float, default=0.0, help="rollout: added to the </think> logit while thinking")
ap.add_argument("--hes-bias", type=float, default=0.0, help="rollout: added to the logits of Wait/But/Alternatively/Hmm "
                "while thinking (negative = less re-checking)")
ap.add_argument("--spin", action="store_true", help="train: SPIN - rows {prompt, chosen, rejected}: chosen = the full-KV model's "
                "generation, rejected = the bounded (SP-evict) model's own; DPO on the sequence log-probs under the deployment loop")
ap.add_argument("--spin-beta", type=float, default=0.1); ap.add_argument("--spin-sft", type=float, default=0.0,
                help="SPIN: + this x the per-token NLL of the chosen sequence")
ap.add_argument("--kl-rows", type=int, default=256, help="--par: rows per KL piece (the vocab-wide fp32 logits of a piece are "
                "the peak: 256 rows ~0.6 GB of temporaries, 64 ~0.15 GB)")
ap.add_argument("--end-train", action="store_true", help="--par: also train short traces and the trace tail (partial last "
                "block), and always one block holding the end of the thought")
ap.add_argument("--end-weight", type=float, default=1.0, help="--par: KL weight of the rows from 32 tokens before </think> "
                "to 8 after")
ap.add_argument("--att-prior", type=float, default=0.0, help="the model's own attention decides where the pooler looks: while a token sits "
                "in the raw window, the attention it receives from later queries (middle layers, summed) is recorded; once it is in the "
                "distant buffer, att-prior x log(its share) is added to the pooler's attention logits (0 = off). The pooler's own attention "
                "collapses onto ~20 sink tokens (spaces, commas, digits, <think>) whatever the length - this pins it to what the model read")
ap.add_argument("--att-layers", default="8-19", help="--att-prior: decoder layers whose attention is summed (a-b inclusive)")
ap.add_argument("--att-cap", type=float, default=0.0, help="--att-prior: bound the pooler's own attention logits to +-cap (cap x tanh(logit / cap)) before the prior is added, so the pooler can move a token's weight at most e^(2 cap) relative to the model's attention and cannot collapse onto a few tokens again (0 = unbounded)")
ap.add_argument("--evict-by", default="pooler", choices=["pooler", "model"], help="--maxd eviction by the pooler's mass or by the model's "
                "attention (needs --att-prior > 0 or --evict-by model to record it)")
ap.add_argument("--drop-top", type=int, default=0, help="distant buffer: drop the N tokens with the MOST pooling mass before pooling (0 = off). "
                "The mass sits on ~20 tokens whatever the length; if they are sinks (punctuation, digits), dropping them should force it onto content")
ap.add_argument("--fifo", type=int, default=0, help="distant buffer: keep only the newest N tokens (oldest dropped first) - "
                "caps the compression ratio without a learned selection; mass eviction (--maxd) still applies on top if set")
ap.add_argument("--par-deploy", action="store_true", help="--par: score only the rows the deployment loop scores - in the block "
                "[SP(tokens before b) | a[b : b+rw+C]] the last C (raw window >= rw), i.e. the chunk after a full raw window; the "
                "rows with a short window (SP alone, never seen in deployment) are left out")
ap.add_argument("--ctl-para", action="store_true", help="train (with --ctl-window): at paragraph openings (the previous token ends "
                "with a blank line) the whole target is the raw-window model's distribution - what to do next (So / Wait / Alternatively "
                "/ Therefore ...) from the bounded model, everything inside a paragraph from the full-KV teacher")
ap.add_argument("--para-window", action="store_true", help="rollout (--kv-mem): every stream runs twice, with its memory and with the "
                "memory masked out; the token opening a paragraph (after a blank line) is sampled from the masked twin - what to do next "
                "from the bounded model, the rest with the memory")
ap.add_argument("--twin-mix", type=float, default=-1.0, help="rollout (--kv-mem): run the memory-masked twin and sample from "
                "(1 - w) x its logits + w x the memory's (w=0: the raw window, w=1: the memory); with --para-window, paragraph openings "
                "still come from the twin alone")
ap.add_argument("--twin-minent", action="store_true", help="rollout (--kv-mem): run the memory-masked twin and, per token, sample from "
                "whichever of the two distributions has the lower entropy")
ap.add_argument("--ctl-window", action="store_true", help="train (chunk mode, --kv-mem): the teacher's distribution with the total "
                "mass on the control tokens (the 'keep checking' openers Wait/But/Alternatively/Hmm, and </think>) set to the raw-window "
                "model's: content from the full-KV teacher, the decision to re-check or stop from the bounded model (the full-KV model "
                "explores because it can see its earlier checks; a student copying that without the recall never converges)")
ap.add_argument("--rkl-weight", type=float, default=0.0, help="train (chunk mode): loss = (1-w) KL(teacher||student) + w KL(student||teacher) "
                "per token. Reverse KL is mode-seeking: on the model's OWN generations (on-policy distillation) it punishes tokens the "
                "full-KV teacher would not write at that state - e.g. a guessed value the soft prompts only blur - where the forward KL "
                "(mode-covering) rewards hedging over every value the lossy past allows, which sampling then turns into confident errors")
ap.add_argument("--sp-null", action="store_true", help="eval/rollout sanity: wherever SP would go, the SP of an EMPTY buffer (a fixed "
                "set of 32 vectors, no content) - does the slot itself hurt, apart from what the soft prompts say?")
ap.add_argument("--sp-swap", action="store_true", help="rollout (--roll-cb) sanity: the SP of ANOTHER stream's distant buffer (a different "
                "problem) - content that is real soft-prompt material but unrelated to this answer")
ap.add_argument("--sp-filler", default="", help="eval/rollout sanity: wherever SP would go, S copies of this REAL token (e.g. a newline) "
                "instead of soft prompts - do soft vectors hurt where ordinary tokens would not?")
ap.add_argument("--probe-n", type=int, default=12, help="probe: traces; 4 chunk states each, after the window has filled")
ap.add_argument("--pin-think", action="store_true", help="eval/rollout/probe: keep the answer's opening '<think>\\n' right after the prompt "
                "once the raw window has moved past it ([prompt | <think> | SP | window]) - otherwise the model writes on mid-thought "
                "without the tag that says it is thinking, and the slot sits exactly where the tag belongs")
ap.add_argument("--kv-evict", action="store_true", help="rollout: plain KV-cache generation that only DROPS old entries - every C "
                "tokens the cache is cut to [prompt | newest rw answer tokens], the kept entries as they were computed (with the whole "
                "past in view) and at their true positions - instead of re-prefilling the window from scratch (no SP). With a huge rw "
                "it is full KV with the same sampling code")
ap.add_argument("--kv-budget", type=int, default=150000, help="--kv-evict: cap on streams x (prompt + kept window + C) cached tokens")
ap.add_argument("--kv-keep", type=int, default=0, help="--kv-evict: a fixed budget of OLDER entries kept beside the newest rw: every C tokens "
                "each stream keeps its prompt, its newest rw answer entries and the --kv-keep older entries the model attended most (the "
                "attention each cached key received from the generated tokens, layers --att-layers, all heads, per step it was cached), "
                "as computed and at their true positions - a bounded REAL KV cache chosen by the model's own attention, no training")
ap.add_argument("--kv-select", default="mean", help="--kv-keep: how the older entries are ranked - mean (attention received per step cached), cum, "
                "ema0.9, ema0.98, last (the last chunk), or mlp:<path> (the learned selector of tools/kvsel_train.py on the same features)")
ap.add_argument("--kv-layerwise", action="store_true", help="--kv-keep: each hooked layer (--att-layers) keeps its own top-B older entries by its own attention "
                "statistic (--kv-select mean / cum / ema0.9 / ema0.98 / last); the other layers keep the set chosen by the summed statistic. "
                "The harness read a per-layer choice ~10% lower in KL")
ap.add_argument("--kv-keep-path", action="store_true", help="--kv-evict: use the --kv-keep code path even with --kv-keep 0 (a check)")
ap.add_argument("--win-teacher", action="store_true", help="train/eval: the SP REPLACES the raw window - teacher = the model reading "
                "[prompt | the last rw answer tokens before chunk start c0 | the chunk] (the raw-window loop, nothing it does not "
                "also hand the student), student = [prompt | SP(those same rw tokens) | the chunk]; KL on the chunk's tokens. A "
                "fixed-ratio compression (rw -> S) with a target the soft prompts can in principle reach")
ap.add_argument("--wt-batch", type=int, default=8, help="--win-teacher: chunk starts per forward")
ap.add_argument("--sp-replace", action="store_true", help="rollout (--roll-cb): the --win-teacher deployment - once rw answer tokens exist, "
                "[prompt | SP(the last rw tokens) | the chunk so far]; before that the plain window")
ap.add_argument("--kv-mem", action="store_true", help="the pooler's S outputs become per-layer KEY/VALUE memory instead of input "
                "embeddings: a low-rank head per layer makes S keys (through that layer's k_norm, then RoPE at the slots right after the "
                "prompt) and S values, put straight into the cache - each layer reads K/V of its own kind instead of re-processing "
                "vectors from layer 0. The model stays frozen; chunk mode (train/eval) and rollout")
ap.add_argument("--kv-rank", type=int, default=512, help="--kv-mem: rank of the per-layer key/value heads")
ap.add_argument("--rmt", action="store_true", help="recurrent memory (RMT style) instead of the pooler: the LLM itself, with a LoRA "
                "switched on only while writing, reads [prompt | previous memory | the next --rmt-block answer tokens | S learned write "
                "tokens]; the write tokens' last hidden states (normalised to the embedding norm) are the new memory. Reading is the "
                "frozen model on [prompt | memory | the raw tokens since the last written block (rw .. rw+block-1) | chunk]")
ap.add_argument("--rmt-block", type=int, default=512, help="--rmt: answer tokens per memory write")
ap.add_argument("--rmt-bptt", type=int, default=2, help="--rmt train: writes back-propagated through before a scored chunk")
ap.add_argument("--rmt-shared", action="store_true", help="--rmt as in the RMT paper: reading and writing share the weights - the LoRA "
                "is on in every pass of the student (the teacher stays the original model)")
ap.add_argument("--rmt-copy", action="store_true", help="--rmt as in the RMT paper: the write block's inputs are a copy of the current "
                "memory ([prompt | M | block | M]) instead of separate learned write tokens")
ap.add_argument("--grad-ckpt", action="store_true", help="train: gradient checkpointing in the decoder layers (activation memory for "
                "full fine-tuning with the RMT passes)")
ap.add_argument("--lora", type=int, default=0, help="train the model through fp32 LoRA adapters of this rank on q/k/v/o and the MLP of "
                "every layer (0 = off; a checkpoint that carries LoRA turns it on at its rank). The bf16 weights cannot take small "
                "updates (one bf16 step is 4e-3 relative; RL updates at lr 1e-6 are 100x smaller and round away) - fp32 adapters can")
ap.add_argument("--grad-sketch", default="", help="train --grpo-groups: count-sketch (2^20 buckets) of each group's LoRA gradient, "
                "summed separately over the even- and the odd-numbered groups, saved to this file; their cosine is logged. Two halves "
                "of the same round under the same weights: a positive cosine is a gradient direction the groups agree on, ~0 is noise "
                "(the cosine of consecutive optimizer steps is not that measure once Adam's momentum carries across rounds)")
ap.add_argument("--opt-state", default="", help="train: load the optimizer state from this file if it exists and save it there at the "
                "end - Adam's momentum then carries across the short rounds of an RL run (a fresh optimizer makes every round's first "
                "step lr x sign(grad) for every parameter, so noise moves as far as signal)")
ap.add_argument("--lora-lr", type=float, default=1e-5, help="--lora: the adapters' learning rate")
ap.add_argument("--rmt-rank", type=int, default=16, help="--rmt: LoRA rank (q/k/v/o and the MLP of every layer)")
ap.add_argument("--const-lr", action="store_true", help="train: constant lr (no warm-up / decay) - for short repeated runs such as the "
                "rounds of an on-policy loop, where a schedule per run would restart every round")
ap.add_argument("--end-stat", action="store_true", help="eval: how much the student and the teacher want to stop thinking")
ap.add_argument("--dump", default="", help="eval: per-token records to this .pt file - per trace the answer positions, the token, "
                "log p of the token under the student and the teacher, forward KL(teacher||student) and reverse KL(student||teacher)")
ap.add_argument("--roll-lp", action="store_true", help="rollout (--roll-cb): also store the generated token ids and each sampled "
                "token's log p under the generator's own (untempered) logits - eval on such a file scores exactly those ids, so "
                "the generation path can be checked against the teacher-forced path token by token")
ap.add_argument("--presence", type=float, default=0.0, help="rollout: presence penalty on the logits of tokens already generated")
ap.add_argument("--no-sp", action="store_true", help="eval/rollout baseline without soft prompts: [raw window | chunk] only "
                "(with a window as long as the trace it is the full-KV model)")
ap.add_argument("--ctx-hidden", type=int, default=0, help="pooler input = the frozen model's layer-N hidden states of the distant "
                "tokens (full-context pass, rescaled to the embedding norm) instead of raw token embeddings - the lever the original "
                "deep-charger INVESTIGATION found (overfit floor 0.16 -> 0.11); chunk mode (train/eval) only")
ap.add_argument("--sp-empty", action="store_true", help="rollout: prepend the SP of an EMPTY buffer while nothing is distant yet "
                "(the first rw tokens) - training never scores those states, so by default they get the plain window")
ap.add_argument("--group", type=int, default=8, help="grpo: samples per problem")
ap.add_argument("--grpo-groups", action="store_true", help="train: GRPO from pre-generated groups - rows {id, prompt, gold, samples: "
                "[texts drawn by the collectors with recent weights], guide: a raw-window text or null}; reward 1 = closed and right; "
                "a right-and-closed guide joins its group with reward 1; advantage = reward - group mean; uniform groups are dropped; "
                "--accum groups per update; sequence log-probs at --temp under the deployment loop")
ap.add_argument("--grpo-wrong", type=float, default=0.0, help="grpo-groups: reward of a closed but wrong answer (right = 1, "
                "unclosed = 0). Negative removes the pay-off of guessing: with 0, answering anything beats thinking to the cap")
ap.add_argument("--grpo-guide", default="", help="grpo: jsonl of guide trajectories (id, text - e.g. the raw window's right and closed "
                "draws): a problem with one gets it as one more group member with reward 1, scored under the policy like the "
                "samples (off-policy guidance: a problem the policy never solves still carries signal)")
ap.add_argument("--grpo-q", type=int, default=16, help="grpo: problems per update")
ap.add_argument("--grpo-updates", type=int, default=200, help="grpo: most updates")
ap.add_argument("--grpo-hours", type=float, default=0.0, help="grpo: stop after this many hours (0 = no limit)")
ap.add_argument("--grpo-slots", type=int, default=32, help="grpo: streams generated together (continuous batching)")
ap.add_argument("--grpo-tok-budget", type=int, default=0, help="grpo/rollout_cb: also cap streams x the longest stream's "
                "re-prefill (prompt + S + rw + C) at this many tokens - long prompts get fewer streams (0 = no cap)")
ap.add_argument("--grpo-best0", type=int, default=-1, help="grpo: the starting weights' check total, when it is known "
                "(instead of --grpo-eval0)")
ap.add_argument("--grpo-lnorm", type=float, default=4096, help="grpo: the summed token log-probs are divided by this constant "
                "(Dr. GRPO: no per-answer length normalisation, so long wrong answers are not pushed down less per token)")
ap.add_argument("--grpo-warmup", type=int, default=3, help="grpo: updates of linear lr warm-up")
ap.add_argument("--grpo-eval", default="", help="grpo: problems (jsonl with prompt/gold, e.g. math96) generated and scored "
                "every --grpo-eval-every updates; the best total is kept at --out")
ap.add_argument("--grpo-eval-every", type=int, default=10); ap.add_argument("--grpo-eval-seeds", type=int, default=1)
ap.add_argument("--grpo-kl", default="", help="grpo: full-KV teacher traces (jsonl prompt/text); at every check the deployment "
                "KL to the teacher over their first 4096 answer tokens (as eval mode) - how far GRPO has moved off the teacher")
ap.add_argument("--grpo-eval0", action="store_true", help="grpo: also score the starting weights")
ap.add_argument("--grpo-patience", type=int, default=3, help="grpo: stop after this many evals without a new best (0 = never)")
ap.add_argument("--grpo-latest", default="", help="grpo: at every check also save the whole training state here (weights, fp32 "
                "masters, optimizer, update count) - and when --ckpt is such a file, carry on from it")
ap.add_argument("--grpo-hook", default="", help="grpo: shell command run (in the background) after each eval with the new "
                "files as arguments - e.g. an upload")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda"); ap.add_argument("--dtype", default="bfloat16")
a = ap.parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
torch.manual_seed(a.seed); random.seed(a.seed)
dev, dt = a.device, getattr(torch, a.dtype)
tok = AutoTokenizer.from_pretrained(a.model)
END_ID = tok.convert_tokens_to_ids("</think>")
THINK_ID = tok.convert_tokens_to_ids("<think>")
FILL = tok.encode(a.sp_filler.encode().decode("unicode_escape"), add_special_tokens=False)[0] if a.sp_filler else None
HES_IDS = sorted({x[0] for w in ("Wait", " Wait", "But", " But", "Alternatively", " Alternatively", "Hmm", " Hmm")
                  for x in [tok.encode(w, add_special_tokens=False)] if len(x) == 1})   # "keep checking" openers
ENDST = {"pt": 0.0, "ps": 0.0, "n": 0, "lt": 0.0, "ls": 0.0, "k": 0, "ht": 0.0, "hs": 0.0, "et": 0.0, "es": 0.0}   # --end-stat: mean p(</think>) all rows; log p at real ends
llm = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, attn_implementation="sdpa").to(dev).eval()
for p_ in llm.parameters(): p_.requires_grad_(False)
embed = llm.get_input_embeddings(); H = llm.config.hidden_size; NL = llm.config.num_hidden_layers
LOGF = open(a.log, "a") if a.log else None
def log(s):
    print(s, flush=True)
    if LOGF: LOGF.write(s + "\n"); LOGF.flush()


def sinusoidal(L, Hd, device):
    pos = torch.arange(L, device=device).float().unsqueeze(1)
    i = torch.arange(0, Hd, 2, device=device).float()
    div = torch.exp(-math.log(10000.0) * i / Hd)
    pe = torch.zeros(L, Hd, device=device)
    pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Block(nn.Module):
    def __init__(self, Hd, heads, ffn):
        super().__init__()
        self.lnq1 = nn.LayerNorm(Hd); self.lnk = nn.LayerNorm(Hd)
        self.cross = nn.MultiheadAttention(Hd, heads, batch_first=True)
        self.lnq2 = nn.LayerNorm(Hd)
        self.selfa = nn.MultiheadAttention(Hd, heads, batch_first=True)
        self.lnq3 = nn.LayerNorm(Hd)
        self.ffn = nn.Sequential(nn.Linear(Hd, ffn), nn.GELU(), nn.Linear(ffn, Hd))

    def forward(self, q, past, return_w=False, kpm=None):
        w = None
        if past is not None and past.size(1) > 0:
            kn = self.lnk(past)
            if kpm is not None and kpm.dtype != torch.bool and a.att_cap > 0:
                a_, w = self.capped_cross(self.lnq1(q), kn, kpm, return_w)
            else:
                a_, w = self.cross(self.lnq1(q), kn, kn, need_weights=return_w, key_padding_mask=kpm)
            q = q + a_
        s, _ = self.selfa(self.lnq2(q), self.lnq2(q), self.lnq2(q), need_weights=False); q = q + s
        q = q + self.ffn(self.lnq3(q))
        return (q, w) if return_w else q

    def capped_cross(self, qn, kn, bias, return_w):
        """self.cross with the pooler's own logits bounded (--att-cap c): logits = c tanh(q.k / sqrt(d) / c) + bias, the bias being
        the model's attention prior (att_bias). The pooler can then move a token's weight by at most e^(2c) relative to the prior,
        so it cannot pile its mass onto a few tokens again. Same weights as self.cross; w averaged over heads as the module's."""
        B, S, H = qn.shape; h = self.cross.num_heads; d = H // h; W, b = self.cross.in_proj_weight, self.cross.in_proj_bias; c = a.att_cap
        qp = F.linear(qn, W[:H], b[:H]).view(B, S, h, d).transpose(1, 2)                      # (B, h, S, d)
        K = F.linear(kn, W[H:2 * H], b[H:2 * H]).view(B, -1, h, d).transpose(1, 2)            # (B, h, L, d)
        V = F.linear(kn, W[2 * H:], b[2 * H:]).view(B, -1, h, d).transpose(1, 2)
        sc = torch.matmul(qp, K.transpose(-1, -2)) / math.sqrt(d)                             # (B, h, S, L)
        w = (c * torch.tanh(sc / c) + bias[:, None, None, :].to(sc.dtype)).softmax(-1)
        o = torch.matmul(w, V).transpose(1, 2).reshape(B, S, H)
        return self.cross.out_proj(o), (w.mean(1) if return_w else None)

    def project_past(self, past):
        """keys and values of ONE past (1, L, H), projected once (the cross-attention's own weights): (heads, L, d) each"""
        H = past.size(-1); h = self.cross.num_heads; d = H // h; W, b = self.cross.in_proj_weight, self.cross.in_proj_bias
        kn = self.lnk(past)[0]
        K = F.linear(kn, W[H:2 * H], b[H:2 * H]).view(-1, h, d).permute(1, 0, 2)
        V = F.linear(kn, W[2 * H:], b[2 * H:]).view(-1, h, d).permute(1, 0, 2)
        return K, V

    def forward_prefix(self, q, K, V, lens):
        """forward() for B query sets over prefixes of one shared past: row b attends to the first lens[b] tokens of K / V
        (projected by project_past), so the past is projected once per trace instead of once per position."""
        B, S, H = q.shape; h = self.cross.num_heads; d = H // h; W, b = self.cross.in_proj_weight, self.cross.in_proj_bias
        L = K.size(1)
        qp = F.linear(self.lnq1(q), W[:H], b[:H]).view(B, S, h, d).permute(2, 0, 1, 3).reshape(h, B * S, d)
        sc = torch.bmm(qp, K.transpose(1, 2)).view(h, B, S, L) / math.sqrt(d)
        beyond = torch.arange(L, device=q.device)[None] >= lens[:, None]; beyond[:, 0] = False   # (B, L)
        w = sc.masked_fill(beyond[None, :, None], float("-inf")).softmax(-1).view(h, B * S, L)
        o = torch.bmm(w, V).view(h, B, S, d).permute(1, 2, 0, 3).reshape(B, S, H)
        q = q + self.cross.out_proj(o)
        s_, _ = self.selfa(self.lnq2(q), self.lnq2(q), self.lnq2(q), need_weights=False); q = q + s_
        return q + self.ffn(self.lnq3(q))


class AttnPoolSP(nn.Module):
    def __init__(self, Hd, n_sp, heads, layers, ffn, target_norm):
        super().__init__()
        self.H = Hd; self.n_sp = n_sp
        self.query = nn.Parameter(torch.randn(n_sp, Hd) * 0.02)
        self.blocks = nn.ModuleList([Block(Hd, heads, ffn) for _ in range(layers)])
        self.ln_out = nn.LayerNorm(Hd)
        self.out_scale = nn.Parameter(torch.tensor(float(target_norm)))
        self._pe = None
        self.pe_rev = False; self.pe_gain = 1.0

    def pe(self, L, device):
        if self._pe is None or self._pe.size(0) < L: self._pe = sinusoidal(max(L, 1024), self.H, device) * self.pe_gain
        return self._pe[:L]

    def _run(self, past_emb, want_mass, kpm=None):
        B, L = past_emb.size(0), past_emb.size(1)
        if L > 0 and self.pe_rev:   # distance from the newest real token of each row (right padding)
            lens = (~kpm).sum(1) if kpm is not None and kpm.dtype == torch.bool else torch.full((B,), L, device=past_emb.device)
            idx = (lens[:, None] - 1 - torch.arange(L, device=past_emb.device)[None]).clamp(min=0)
            past = past_emb + self.pe(L, past_emb.device)[idx]
        elif L > 0:
            past = past_emb + self.pe(L, past_emb.device).unsqueeze(0)
        else:
            past = past_emb
        q = self.query.unsqueeze(0).expand(B, -1, -1).contiguous()
        mass = torch.zeros(B, L, device=past_emb.device) if want_mass else None
        for blk in self.blocks:
            if want_mass:
                q, w = blk(q, past, return_w=True, kpm=kpm)
                if w is not None: mass = mass + w.sum(dim=1)
            else:
                q = blk(q, past, kpm=kpm)
        sp = self.ln_out(q)
        sp = sp / sp.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self.out_scale.abs()   # output normalisation
        return (sp, mass) if want_mass else sp

    def forward(self, past_emb, kpm=None): return self._run(past_emb, False, kpm)   # kpm: (B, L) True = padding

    def forward_prefix(self, past_emb, lens):
        """SP for B prefixes (lengths lens) of ONE past (1, L, H) - the same result as forward() on the padded prefixes
        (positions counted from the start, i.e. not pe_rev), with the past embedded and projected once."""
        L = past_emb.size(1); B = lens.numel()
        past = past_emb + self.pe(L, past_emb.device).unsqueeze(0)
        q = self.query.unsqueeze(0).expand(B, -1, -1).contiguous()
        for blk in self.blocks:
            K, V = blk.project_past(past)
            q = blk.forward_prefix(q, K, V, lens)
        sp = self.ln_out(q)
        return sp / sp.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self.out_scale.abs()
    def forward_with_mass(self, past_emb, kpm=None): return self._run(past_emb, True, kpm)


with torch.no_grad():
    target_norm = embed(torch.randint(0, embed.weight.size(0), (2048,), device=dev)).float().norm(dim=-1).mean().item()
pooler = AttnPoolSP(H, a.S, a.heads, a.layers, a.ffn, target_norm).to(dev)
KVH, KVD = llm.config.num_key_value_heads, getattr(llm.config, "head_dim", H // llm.config.num_attention_heads)


class KVHead(nn.Module):
    """--kv-mem: S pooler outputs -> per-layer keys (pre-norm, pre-RoPE) and values, low rank (shared down map, one up map per
    layer); values start at zero, so the fresh memory adds nothing to the attention outputs"""
    def __init__(self, Hd, nl, kvh, hd, r):
        super().__init__()
        self.down = nn.Linear(Hd, r)
        self.uk = nn.Parameter(torch.randn(nl, r, kvh * hd) / math.sqrt(r)); self.bk = nn.Parameter(torch.zeros(nl, kvh * hd))
        self.uv = nn.Parameter(torch.zeros(nl, r, kvh * hd)); self.bv = nn.Parameter(torch.zeros(nl, kvh * hd))
    def forward(self, sp):
        z = self.down(sp.float())                                                   # (B, S, r)
        return torch.einsum("bsr,lrd->lbsd", z, self.uk) + self.bk[:, None, None], torch.einsum("bsr,lrd->lbsd", z, self.uv) + self.bv[:, None, None]


if a.kv_mem: pooler.kvhead = KVHead(H, NL, KVH, KVD, a.kv_rank).to(dev)
FT_SD = None
if a.ckpt:
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if not (a.fresh_pooler and a.mode == "train"):
        psd = dict(ck["pooler"]); q0 = psd["query"]
        if q0.size(0) < a.S:   # growing S (as the character model grew depth): the new queries start as copies of the old
            g = torch.Generator().manual_seed(a.seed)    # ones plus a little noise, so they split off from their originals
            extra = q0[torch.arange(a.S - q0.size(0)) % q0.size(0)]
            psd["query"] = torch.cat([q0, extra + 0.1 * q0.std() * torch.randn(extra.shape, generator=g)], 0)
            print(f"[sp] S grown {q0.size(0)} -> {a.S} (new queries = copies + noise)", flush=True)
        pooler.load_state_dict(psd)
    if a.mode != "train":   # eval (and grpo) follows the checkpoint
        if (ck.get("args") or {}).get("pe_rev"): a.pe_rev = True
        if (ck.get("args") or {}).get("pe_norm"): a.pe_norm = True
        if (ck.get("args") or {}).get("ctx_hidden"): a.ctx_hidden = ck["args"]["ctx_hidden"]
    if ck.get("ft") and not a.rmt:
        FT_SD = {k: v.detach().to("cpu", copy=True) for k, v in llm.state_dict().items() if k in ck["ft"]}   # frozen copy = teacher
        llm.load_state_dict({k: v.to(dev, dt) for k, v in ck["ft"].items()}, strict=False)
pooler.pe_rev = a.pe_rev
if a.pe_norm: pooler.pe_gain = target_norm / math.sqrt(H / 2)   # sinusoid norm sqrt(H/2) -> the embedding norm
LORA = {"on": False, "default": False}   # default: the state outside memory writes (on for --rmt-shared)


class LoRALinear(nn.Module):
    """a frozen Linear plus a low-rank update that is applied only while LORA['on'] (the memory-writing passes)"""
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base; self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
    def forward(self, x):
        y = self.base(x)
        if LORA["on"]: y = y + ((x.float() @ self.A.t()) @ self.B.t() * self.scale).to(y.dtype)
        return y


class RMTMem(nn.Module):
    """the memory writer's own parameters: the first memory, the write tokens, the output map and scale"""
    def __init__(self, Hd, n, target_norm):
        super().__init__()
        self.m0 = nn.Parameter(torch.randn(n, Hd) * 0.02); self.wtok = nn.Parameter(torch.randn(n, Hd) * 0.02)
        self.proj = nn.Linear(Hd, Hd); nn.init.eye_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        self.out_scale = nn.Parameter(torch.tensor(float(target_norm)))
    def norm_out(self, h):
        m = self.proj(h.float()); return m / m.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self.out_scale.abs()


RMT = None
if a.rmt and a.rmt_shared: LORA["default"] = LORA["on"] = True
if a.rmt:
    for li in range(NL):
        L_ = llm.model.layers[li]
        for mod, names in ((L_.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")), (L_.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for nm in names: setattr(mod, nm, LoRALinear(getattr(mod, nm), a.rmt_rank, 2 * a.rmt_rank))
    RMT = RMTMem(H, a.S, target_norm).to(dev)
    if a.ckpt and ck.get("rmt"):
        RMT.load_state_dict(ck["rmt"]["mem"]); llm.load_state_dict(ck["rmt"]["lora"], strict=False)
    if a.ckpt and ck.get("ft"):   # tuned base weights (names after the wrapping: ...q_proj.base.weight); the original = the teacher
        FT_SD = {k: v.detach().to("cpu", copy=True) for k, v in llm.state_dict().items() if k in ck["ft"]}
        llm.load_state_dict({k: v.to(dev, dt) for k, v in ck["ft"].items()}, strict=False)
RMT_PARAMS = ([p_ for n_, p_ in llm.named_parameters() if n_.endswith((".A", ".B"))] + list(RMT.parameters())) if a.rmt else []
for p_ in RMT_PARAMS: p_.requires_grad_(True)

BASE_FT = ck.get("ft") if a.ckpt else None   # tuned base weights carried through a LoRA run's checkpoints
LORA_RANK = a.lora or ((ck.get("lora_rank") or 0) if a.ckpt else 0)
if LORA_RANK and not a.rmt:
    for li in range(NL):
        L_ = llm.model.layers[li]
        for mod, names in ((L_.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")), (L_.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for nm in names: setattr(mod, nm, LoRALinear(getattr(mod, nm), LORA_RANK, 2 * LORA_RANK))
    LORA["default"] = LORA["on"] = True
    if a.ckpt and ck.get("lora"): llm.load_state_dict({k: v.to(dev) for k, v in ck["lora"].items()}, strict=False)
    print(f"[sp] LoRA rank {LORA_RANK} on all {NL} layers" + (" (loaded)" if a.ckpt and ck.get("lora") else " (fresh: B = 0)"), flush=True)
LORA_PARAMS = [p_ for n_, p_ in llm.named_parameters() if n_.endswith((".A", ".B"))] if (LORA_RANK and not a.rmt) else []
for p_ in LORA_PARAMS: p_.requires_grad_(True)


def rmt_write(m_prev, prompt_ids, block_ids):
    """one memory write: [prompt | m_prev | block | write tokens] through the model with the LoRA on -> new memory (1, S, H)"""
    LORA["on"] = True
    try:
        wt = m_prev.to(dt) if a.rmt_copy else RMT.wtok[None].to(dt)
        x = torch.cat([embed(prompt_ids[None]).to(dt), m_prev.to(dt), embed(block_ids[None]).to(dt), wt], 1)
        h = llm.model(inputs_embeds=x, position_ids=torch.arange(x.size(1), device=dev)[None], use_cache=False).last_hidden_state[:, -a.S:]
    finally:
        LORA["on"] = LORA["default"]
    return RMT.norm_out(h)


n_pool = sum(x.numel() for x in pooler.parameters())
KV_TOK = NL * 2 * llm.config.num_key_value_heads * getattr(llm.config, "head_dim", H // llm.config.num_attention_heads) * 2
log(f"[sp] pooler {n_pool / 1e6:.1f}M params (4-bit ~{n_pool / 2 / 2**20:.0f} MB) | S={a.S} rw={a.rw} C={a.C} maxd={a.maxd} | "
    f"bounded KV {(a.S + a.rw + a.C) * KV_TOK / 2**20:.0f} MB + prompt | target_norm {target_norm:.3f}")


@torch.no_grad()
def evict_topk(ids, mass, maxD):
    if ids.size(1) <= maxD: return ids
    idx = mass.topk(maxD, dim=1).indices.sort(dim=1).values
    return ids.gather(1, idx)


def att_bias(imp):
    """(1, L) importance -> the additive bias on the pooler's attention logits (None when --att-prior is off)"""
    if not a.att_prior or imp is None or imp.numel() == 0: return None
    sh = imp.float().clamp(min=0) + 1e-6; sh = sh / sh.sum(dim=1, keepdim=True)
    return a.att_prior * (sh.log() - sh.log().mean(dim=1, keepdim=True))


@torch.no_grad()
def update_buffer(kept, absorbed, src, nd_end, feat=None, imp=None, src_imp=None):
    """imp: (1, L) the model's attention importance of the kept tokens (--att-prior / --evict-by model); src_imp: (1, T) of src"""
    if nd_end > absorbed:
        kept = torch.cat([kept, src[:, absorbed:nd_end]], 1)
        if imp is not None: imp = torch.cat([imp, src_imp[:, absorbed:nd_end]], 1)
        absorbed = nd_end
        if a.fifo and kept.size(1) > a.fifo:
            kept = kept[:, -a.fifo:]; imp = imp[:, -a.fifo:] if imp is not None else None
        if a.maxd and kept.size(1) > a.maxd:
            if a.evict_by == "model" and imp is not None: mass = imp
            else: _, mass = pooler.forward_with_mass(feat(kept) if feat else embed(kept).float(), att_bias(imp))
            idx = mass.topk(a.maxd, dim=1).indices.sort(dim=1).values
            kept = kept.gather(1, idx); imp = imp.gather(1, idx) if imp is not None else None
        if a.drop_top and kept.size(1) > a.drop_top + 64:
            _, mass = pooler.forward_with_mass(feat(kept) if feat else embed(kept).float(), att_bias(imp))
            keep_ = torch.ones_like(mass, dtype=torch.bool); keep_.scatter_(1, mass.topk(a.drop_top, dim=1).indices, False)
            kept = kept[keep_].view(1, -1); imp = imp[keep_].view(1, -1) if imp is not None else None
    return (kept, absorbed) if imp is None and src_imp is None else (kept, absorbed, imp)


ATT_L = []   # --att-prior / --evict-by model: the decoder layers whose attention is the importance
if a.att_prior > 0 or a.evict_by == "model" or a.kv_keep > 0 or a.mode == "kvsel":
    _lo, _hi = (int(x) for x in a.att_layers.split("-")); ATT_L = list(range(_lo, _hi + 1))
ATTLOG = [False]   # the one-time log of the importance on a first distant buffer
ATTREC = {"rows": 0, "acc": {}}   # teacher-forced passes: per hooked layer, the attention the block's last `rows` query rows paid to each key


def _att_hook(mod, args, kwargs, out):
    """forward hook on the attention of layers ATT_L: recomputes this layer's attention of the block's last ATTREC['rows'] query
    rows over every key (prompt, memory, window and chunk; causal inside the block), summed over heads and rows - what the
    rollout reads from the eager attentions of its decode steps. Written per layer (gradient checkpointing runs a layer twice)."""
    rows = ATTREC["rows"]
    if not rows: return
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
    h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
    cos, sin = kwargs["position_embeddings"] if "position_embeddings" in kwargs else args[1]
    cache = kwargs.get("past_key_values")
    if cache is None: return
    with torch.no_grad():
        B, nb = h.shape[:2]
        q = mod.q_norm(mod.q_proj(h[:, -rows:]).view(B, rows, -1, mod.head_dim)).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, cos[:, -rows:], sin[:, -rows:])
        kb = mod.k_norm(mod.k_proj(h).view(B, nb, -1, mod.head_dim)).transpose(1, 2)
        _, kb = apply_rotary_pos_emb(kb, kb, cos, sin)
        kc = kv_of(cache, mod.layer_idx)[0]   # a frozen PrefixCache holds the prefix only; a DynamicCache the prefix and this block
        pre = kc if getattr(cache, "frozen", False) else kc[:, :, :kc.size(2) - nb]
        k = repeat_kv(torch.cat([pre, kb], 2), q.size(1) // kb.size(1))
        sc = (q.float() @ k.float().transpose(-1, -2)) * mod.scaling   # (B, heads, rows, keys)
        L = k.size(2); t = torch.arange(L, device=h.device)
        sc.masked_fill_((t[None, :] > (L - rows + torch.arange(rows, device=h.device))[:, None])[None, None], float("-inf"))
        am = kwargs.get("attention_mask")   # left padding / finished rows in a batch: the model's own 4-d mask
        if am is not None and am.dim() == 4 and am.size(-1) >= L and am.size(-2) >= rows:
            am = am[:, :, -rows:, :L]
            if am.dtype == torch.bool: sc.masked_fill_(~am, float("-inf"))
            else: sc = sc + am.to(sc.dtype)
        ATTREC["acc"][mod.layer_idx] = sc.softmax(-1).nan_to_num(0.0).sum(1).sum(1)   # (B, keys)


def att_take():
    """the hooked layers' sums since the last att_take, then reset: (keys,) for a single row, (B, keys) for a batch"""
    acc = ATTREC["acc"]; ATTREC["acc"] = {}; ATTREC["rows"] = 0
    if not acc: return None
    x = sum(acc.values()); return x[0] if x.size(0) == 1 else x


for _li in ATT_L: llm.model.layers[_li].self_attn.register_forward_hook(_att_hook, with_kwargs=True)


@torch.no_grad()
def importance_of(ids, P):
    """--att-prior profiling: the importance of the answer tokens ids[P:] from window-only deployment forwards (prompt + raw
    window + chunk, no memory slots) with the hooks - the rollout's quantity up to the memory's share of the keys"""
    a_ids = torch.tensor([ids[P:]], device=dev); La = a_ids.size(1)
    pc = DynamicCache(); llm(input_ids=torch.tensor([ids[:P]], device=dev), past_key_values=pc, use_cache=True)
    pkv = [tuple(kv_of(pc, li)) for li in range(NL)]
    imp = torch.zeros(La, device=dev)
    for c0 in range(0, La, a.C):
        c1 = min(c0 + a.C, La); R = min(c0, a.rw)
        cache = DynamicCache()
        for li, (k_, v_) in enumerate(pkv): cache.update(k_, v_, li)
        blk = a_ids[:, c0 - R:c1]; nb = blk.size(1); pos = torch.arange(P, P + nb, device=dev)[None]
        ATTREC["rows"] = c1 - c0
        llm.model(inputs_embeds=embed(blk).to(dt), attention_mask=torch.ones(1, P + nb, device=dev), past_key_values=cache,
                  position_ids=pos, cache_position=pos[0], use_cache=True)
        imp[c0 - R:c1] += att_take()[P:P + nb]
    imp[:2] = 0
    return imp


def kv_of(cache, li):
    if hasattr(cache, "layers"): return cache.layers[li].keys, cache.layers[li].values
    return cache.key_cache[li], cache.value_cache[li]


SWAP = {"pairs": None, "depth": 0}   # training: (param, pinned frozen copy, fp32 master) - fast in-place swap


class _Frozen:
    """swap the frozen copy of the tuned layers in for the teacher pass (re-entrant: nested uses are no-ops)"""
    def __enter__(self):
        SWAP["depth"] += 1
        if FT_SD is None or SWAP["depth"] > 1: return
        if SWAP["pairs"] is not None:      # training: frozen copy from pinned CPU, restore from the fp32 master (on GPU)
            with torch.no_grad():
                for p_, f_, m_ in SWAP["pairs"]: p_.copy_(f_, non_blocking=True)
            return
        self.cur = {k: v.detach().to("cpu", copy=True) for k, v in llm.state_dict().items() if k in FT_SD}
        llm.load_state_dict(FT_SD, strict=False)
    def __exit__(self, *e):
        SWAP["depth"] -= 1
        if FT_SD is None or SWAP["depth"] > 0: return
        if SWAP["pairs"] is not None:
            with torch.no_grad():
                for p_, f_, m_ in SWAP["pairs"]: p_.copy_(m_.to(p_.dtype))
            return
        llm.load_state_dict(self.cur, strict=False)


@torch.no_grad()
def teacher_hidden(ids, lo, hi):
    """the full-context teacher's last hidden states whose rows predict tokens lo..hi-1 (lm head applied per chunk)"""
    on = LORA["on"]; LORA["on"] = False   # the teacher is the original model (no LoRA, frozen weights)
    try:
        with _Frozen():
            return llm.model(input_ids=ids[None, :hi], use_cache=False).last_hidden_state[0, lo - 1:hi - 1]
    finally:
        LORA["on"] = on


@torch.no_grad()
def teacher_rows(h):
    # the lm head is never tuned, so no weight swap here (swapping in place between the student's forward and backward
    # would invalidate its graph)
    return F.log_softmax(llm.lm_head(h).float(), -1)


def pin_of(ans, lo):
    """--pin-think: the answer's opening '<think>' (+ its newline) as a list of ids, when the raw window starts after it"""
    if not a.pin_think or not len(ans) or int(ans[0]) != THINK_ID: return []
    k = 2 if len(ans) > 1 and tok.decode([int(ans[1])]).strip() == "" else 1
    return [int(x) for x in ans[:k]] if lo >= k else []


def kv_memory(sp, pos):
    """--kv-mem: pooler outputs sp (B, S, H) -> per layer (k, v), each (B, kv heads, S, head dim) in the model's dtype: keys through
    the layer's k_norm and RoPE at positions pos (B, S), exactly like the cache entries of S tokens sitting there"""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    K, V = pooler.kvhead(sp); B, S_ = sp.shape[:2]
    cos, sin = llm.model.rotary_emb(sp.to(dt), pos)
    out = []
    for li in range(NL):
        k = llm.model.layers[li].self_attn.k_norm(K[li].view(B, S_, KVH, KVD).to(dt)).transpose(1, 2)
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        out.append((k, V[li].view(B, S_, KVH, KVD).transpose(1, 2).to(dt)))
    return out


def slot(kept_ids, feat=None):
    """what goes between the prompt and the raw window once something is distant: the soft prompts, or a sanity stand-in"""
    if a.sp_filler: return embed(torch.tensor([FILL] * a.S, device=dev))[None].to(dt)
    if a.sp_null or kept_ids.size(1) == 0: return pooler(torch.zeros(1, 0, H, device=dev)).to(dt)
    return pooler(feat(kept_ids) if feat else embed(kept_ids).float()).to(dt)


RKL = [0.0, 0]   # chunk mode: summed reverse KL(student||teacher) and rows since the last log
DUMP = [] if a.dump else None   # eval --dump: per trace a list of (rows, 7) tensors


CTL_G = None
CTLP = [0, 0]   # --ctl-para: paragraph-opening rows, all rows


PARA_END = None


def ctl_target(tr, pkv, block, P, cur, prev=None):
    """--ctl-window: tr (cur, V) full-KV teacher log-probs -> the same with each control group's total mass (and the rest's)
    replaced by the raw-window model's on the same block (no memory, positions right after the prompt)"""
    global CTL_G, PARA_END
    if CTL_G is None: CTL_G = [torch.tensor(HES_IDS, device=dev), torch.tensor([END_ID], device=dev)]
    if a.ctl_para and PARA_END is None:
        PARA_END = torch.tensor([tok.decode([i]).endswith("\n\n") for i in range(len(tok))], device=dev)
    with torch.no_grad():
        cw = DynamicCache()
        for li, (k_, v_) in enumerate(pkv): cw.update(k_, v_, li)
        nb = block.size(1); pos = torch.arange(P, P + nb, device=dev)[None]
        ow = llm(inputs_embeds=block.detach(), attention_mask=torch.ones(1, P + nb, device=dev), past_key_values=cw,
                 position_ids=pos, cache_position=pos[0], use_cache=True, logits_to_keep=cur + 1)
        wl = F.log_softmax(ow.logits[0, :cur].float(), -1); del ow, cw
        u = torch.cat(CTL_G)
        l1m = lambda x: torch.log(-torch.expm1(x.clamp(max=-1e-6)))
        out = tr + (l1m(torch.logsumexp(wl[:, u], -1, keepdim=True)) - l1m(torch.logsumexp(tr[:, u], -1, keepdim=True)))
        for g in CTL_G:
            out[:, g] = tr[:, g] - torch.logsumexp(tr[:, g], -1, keepdim=True) + torch.logsumexp(wl[:, g], -1, keepdim=True)
        if a.ctl_para and prev is not None:
            m = PARA_END[prev.clamp(max=len(PARA_END) - 1)][:, None]; CTLP[0] += int(m.sum()); CTLP[1] += m.numel()
            out = torch.where(m, wl, out)
    return out


GEN = {"c": 0, "n": 0, "fk": 0.0, "rk": 0.0, "nll": 0.0, "bins": [[0.0, 0] for _ in range(3)], "eos": 0}   # --gen-kl totals


@torch.no_grad()
def gen_chunk(cache, keep, s0, t0, tc, tlen, seed, force=None):
    """--gen-kl: the student writes a.gen_len tokens from its state at a chunk start (cache cropped to prompt + memory + raw
    window, s0 its distribution for the first token); the teacher's cache tc holds the true text, of which the first tlen
    positions are used, then it reads the student's tokens (t0: its distribution for the first one). Adds to GEN."""
    full = cache; cache = DynamicCache()   # the first `keep` positions (a new cache: crop() changed meaning across versions)
    for li in range(NL):
        k_, v_ = kv_of(full, li); cache.update(k_[:, :, :keep], v_[:, :, :keep], li)
    del full; torch.manual_seed(seed & 0x7FFFFFFF)
    ss, toks, s_ = [s0], [], s0
    for j in range(a.gen_len):
        t_ = force[j:j + 1] if force is not None else sample_tok(s_[None])
        toks.append(t_)
        if j == a.gen_len - 1 or (force is None and int(t_) == tok.eos_token_id) or (force is not None and j + 1 >= force.numel()): break
        pos = torch.tensor([[keep + j]], device=dev)
        o = llm(inputs_embeds=embed(t_[None]).to(dt), attention_mask=torch.ones(1, keep + j + 1, device=dev), past_key_values=cache,
                position_ids=pos, cache_position=pos[0], use_cache=True)
        s_ = F.log_softmax(o.logits[0, -1].float(), -1); ss.append(s_)
    g = torch.cat(toks); sl = torch.stack(ss); n = g.numel()
    tl = [t0[None]]
    if n > 1:
        tcache = DynamicCache()
        for li in range(NL):
            k_, v_ = kv_of(tc, li); tcache.update(k_[:, :, :tlen], v_[:, :, :tlen], li)
        on = LORA["on"]; LORA["on"] = False
        try:
            with _Frozen():
                pos = torch.arange(tlen, tlen + n - 1, device=dev)[None]
                ot = llm(input_ids=g[None, :-1], past_key_values=tcache, position_ids=pos, cache_position=pos[0], use_cache=True)
        finally:
            LORA["on"] = on
        tl.append(F.log_softmax(ot.logits[0].float(), -1))
    tl = torch.cat(tl)
    fk = (tl.exp() * (tl - sl)).sum(-1); rk = (sl.exp() * (sl - tl)).sum(-1)
    GEN["c"] += 1; GEN["n"] += n; GEN["fk"] += fk.sum().item(); GEN["rk"] += rk.sum().item()
    GEN["nll"] -= tl.gather(1, g[:, None]).sum().item(); GEN["eos"] += int(n < a.gen_len)
    for b, (lo, hi) in enumerate([(0, 32), (32, 64), (64, 10 ** 9)]):
        GEN["bins"][b][0] += fk[lo:hi].sum().item(); GEN["bins"][b][1] += fk[lo:hi].numel()
    return fk


def gen_report():
    if not GEN["c"]: return ""
    c, n, b = GEN["c"], max(GEN["n"], 1), GEN["bins"]
    return (f" | gen{a.gen_len} ({c} chunks, {'true text' if a.gen_force else f'temp {a.temp}'}): KL summed per chunk {GEN['fk'] / c:.2f} "
            f"(per token {GEN['fk'] / n:.4f}; by position 0-31 {b[0][0] / max(b[0][1], 1):.4f} 32-63 {b[1][0] / max(b[1][1], 1):.4f} "
            f"64+ {b[2][0] / max(b[2][1], 1):.4f}) reverse {GEN['rk'] / c:.2f} | teacher NLL of the student's tokens {GEN['nll'] / n:.3f}"
            f" | ended early {GEN['eos']}")


def _mem_kl(sp, pkv, P, block, cur, tr, off0):
    """KL(teacher || student) on a chunk given the memory sp (1, S, H) placed after the prompt: the chunk's rows"""
    mem = kv_memory(sp, torch.arange(P, P + a.S, device=dev)[None]); cache = DynamicCache()
    for li, (k_, v_) in enumerate(pkv): cache.update(torch.cat([k_, mem[li][0]], 2), torch.cat([v_, mem[li][1]], 2), li)
    nb = block.size(1); off = P + a.S; pos = torch.arange(off, off + nb, device=dev)[None]
    o = llm(inputs_embeds=block, attention_mask=torch.ones(1, off + nb, device=dev), past_key_values=cache, position_ids=pos,
            cache_position=pos[0], use_cache=True, logits_to_keep=cur + 1)
    sl = F.log_softmax(o.logits[0, :cur].float(), -1)
    return (tr.exp() * (tr - sl)).sum(-1)


def _oracle_fit(sp0, z0, pkv, P, block, cur, tr):
    """optimise the slots from z0 (pre-normalisation; sp = z / |z| x out_scale, the pooler's own output manifold) on the chunk's
    KL; returns (sp*, KL per 25 steps)"""
    z = z0.clone().requires_grad_(True); opt = torch.optim.Adam([z], lr=a.oracle_lr); sc = pooler.out_scale.abs().detach(); hist = []
    for it in range(a.oracle_steps):
        sp = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-6) * sc
        kl = _mem_kl(sp, pkv, P, block, cur, tr, 0); loss = kl.mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if it % 25 == 0: hist.append(round(loss.item(), 4))
    with torch.no_grad():
        sp = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-6) * sc
        hist.append(round(_mem_kl(sp, pkv, P, block, cur, tr, 0).mean().item(), 4))
    return sp.detach(), hist


ORC = []   # oracle: one dict per position


def oracle_trace(ids, P):
    """oracle (the user): at a few chunk starts, optimise the 32 memory slots THEMSELVES through the frozen model on the chunk's
    KL - (a) seeded with the pooler's own output: how low can the KL go from where the pooler already is, and how far must
    the slots move (a target a pooler could reproduce); (b) seeded at random: the same floor from an unrelated start, and how
    far that optimum lies from the pooler's. Also each memory's KL on the NEXT chunk with the window slid and the memory kept
    (the deployment would rebuild it), and the window-only KL as the reference. The slots stay on the pooler's output
    manifold (unit rows x out_scale)."""
    T = min(len(ids), P + a.max_len); a_ids = ids[None, P:T]; La = a_ids.size(1); first = a.rw + a.C
    c0s = list(range(first, La - a.C, a.C))
    if len(c0s) < 4: return
    ids = ids.to(dev); a_ids = a_ids.to(dev)
    with torch.no_grad():
        ref_h = teacher_hidden(ids, P + first, T)
        pc = DynamicCache(); llm(input_ids=ids[None, :P], past_key_values=pc, use_cache=True)
        pkv = [tuple(t for t in kv_of(pc, li)) for li in range(NL)]
    if a.oracle_run > 0:
        st_ = max(0, min(len(c0s) - a.oracle_run, len(c0s) // 2)); targets = set(c0s[st_:st_ + a.oracle_run])
    else:
        targets = {c0s[min(len(c0s) - 1, int(float(f) * len(c0s)))] for f in a.oracle_pos.split(",")}
    prev = None   # --oracle-run: (sp0, spA, spB) of the previous position of this trace
    kept = torch.zeros(1, 0, dtype=torch.long, device=dev); absorbed = 0; use_att = bool(ATT_L)
    impT = torch.zeros(La, device=dev); kimp = torch.zeros(1, 0, device=dev)
    for c0 in range(0, La, a.C):
        c1 = min(c0 + a.C, La); cur = c1 - c0
        if cur < 4: break
        R = min(c0, a.rw); nd_end = c0 - R
        if use_att:
            si = impT[None].clone(); si[:, :2] = 0
            kept, absorbed, kimp = update_buffer(kept, absorbed, a_ids, nd_end, None, imp=kimp, src_imp=si)
        else: kept, absorbed = update_buffer(kept, absorbed, a_ids, nd_end, None)
        hit = c0 in targets and kept.size(1) > 0
        if not hit and not use_att: continue
        with torch.no_grad():
            block = torch.cat([embed(a_ids[:, c0 - R:c0]).to(dt), embed(a_ids[:, c0:c1]).to(dt)], 1); nb = block.size(1)
            sp0 = pooler(embed(kept).float(), kpm=att_bias(kimp) if use_att else None)
            if use_att:   # the importance, recorded as in run_trace (memory in place)
                cache = DynamicCache(); mem = kv_memory(sp0, torch.arange(P, P + a.S, device=dev)[None]); off = P + a.S
                for li, (k_, v_) in enumerate(pkv): cache.update(torch.cat([k_, mem[li][0]], 2), torch.cat([v_, mem[li][1]], 2), li)
                pos = torch.arange(off, off + nb, device=dev)[None]; ATTREC["rows"] = cur
                llm(inputs_embeds=block, attention_mask=torch.ones(1, off + nb, device=dev), past_key_values=cache, position_ids=pos,
                    cache_position=pos[0], use_cache=True, logits_to_keep=1)
                impT[c0 - R:c1] += att_take()[off:off + R + cur]
        if not hit: continue
        tr = teacher_rows(ref_h[c0 - first:c1 - first])
        with torch.no_grad():
            kl0 = _mem_kl(sp0, pkv, P, block, cur, tr, 0).mean().item()
            cw = DynamicCache()   # window only
            for li, (k_, v_) in enumerate(pkv): cw.update(k_, v_, li)
            pos = torch.arange(P, P + nb, device=dev)[None]
            ow = llm(inputs_embeds=block, attention_mask=torch.ones(1, P + nb, device=dev), past_key_values=cw, position_ids=pos,
                     cache_position=pos[0], use_cache=True, logits_to_keep=cur + 1)
            klw = (tr.exp() * (tr - F.log_softmax(ow.logits[0, :cur].float(), -1))).sum(-1).mean().item(); del ow, cw
        g = torch.Generator(device=dev).manual_seed(1234 + c0)
        unit = lambda x: x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        spA, hA = _oracle_fit(sp0, unit(sp0.float()), pkv, P, block, cur, tr)
        spB, hB = _oracle_fit(sp0, unit(torch.randn(sp0.shape, generator=g, device=dev)), pkv, P, block, cur, tr) if a.oracle_random else (sp0, [round(kl0, 4)])
        # the next chunk, the window slid, the memory kept
        n0, n1 = c1, min(c1 + a.C, La); nxt = {}
        if n1 - n0 >= 4 and n1 - first <= ref_h.size(0):
            with torch.no_grad():
                Rn = min(n0, a.rw); blk2 = torch.cat([embed(a_ids[:, n0 - Rn:n0]).to(dt), embed(a_ids[:, n0:n1]).to(dt)], 1)
                tr2 = teacher_rows(ref_h[n0 - first:n1 - first])
                for nm, sp_ in (("pooler", sp0), ("seeded", spA), ("random", spB)):
                    nxt[nm] = round(_mem_kl(sp_, pkv, P, blk2, n1 - n0, tr2, 0).mean().item(), 4)
        cs = F.cosine_similarity(spA, sp0, dim=-1).mean().item(); csB = F.cosine_similarity(spB, sp0, dim=-1).mean().item()
        rel = ((spA - sp0).norm() / sp0.norm()).item(); relB = ((spB - sp0).norm() / sp0.norm()).item(); relAB = ((spA - spB).norm() / sp0.norm()).item()
        ent = {"c0": c0, "nd": int(nd_end), "window": klw, "pooler": kl0, "seeded": hA[-1], "random": hB[-1], "next": nxt,
               "rel": rel, "cos": cs, "relB": relB, "cosB": csB, "relAB": relAB}
        if a.oracle_run > 0 and prev is not None:   # this position against the previous one of the same chain of thought
            p0, pA, pB = prev; cosf = lambda x, y: F.cosine_similarity(x, y, dim=-1).mean().item()
            with torch.no_grad():
                kprevA = _mem_kl(pA, pkv, P, block, cur, tr, 0).mean().item(); kprevB = _mem_kl(pB, pkv, P, block, cur, tr, 0).mean().item()
            ent["pair"] = {"cos_pooler": cosf(sp0, p0), "cos_seeded": cosf(spA, pA), "cos_random": cosf(spB, pB),
                           "cos_corr": F.cosine_similarity((spA - sp0).flatten(), (pA - p0).flatten(), dim=0).item(),
                           "kl_prev_seeded": kprevA, "kl_prev_random": kprevB}
            pr_ = ent["pair"]
            log(f"[sp] oracle-run pos {c0} vs {c0 - a.C}: cos pooler {pr_['cos_pooler']:.3f} seeded optima {pr_['cos_seeded']:.3f} random optima "
                f"{pr_['cos_random']:.3f} | corrections (optimum - pooler) cos {pr_['cos_corr']:.3f} | KL on this chunk: pooler {kl0:.4f} own optimum "
                f"{hA[-1]:.4f} previous seeded optimum {kprevA:.4f} previous random optimum {kprevB:.4f} window-only {klw:.4f}")
        if a.oracle_run > 0: prev = (sp0, spA, spB)
        ORC.append(ent)
        log(f"[sp] oracle pos {c0} (distant {nd_end}): KL window-only {klw:.4f} | pooler {kl0:.4f} | slots optimised from the pooler "
            f"{' > '.join(str(x) for x in hA)} | from random {' > '.join(str(x) for x in hB)} | next chunk, memory kept: {nxt} | "
            f"move from the pooler's output: seeded {rel:.3f} (cos {cs:.3f}), random optimum {relB:.3f} (cos {csB:.3f}), between the two optima {relAB:.3f}")


def oracle_h_trace(ids, P):
    """--oracle-horizon K: at consecutive chunk starts (--oracle-run n, else the trace's middle), the slots are optimised on the KL
    of the next K chunks together (block = raw window + K chunks, the memory covering the tokens before the window, as deployed
    at this position) and scored on the 2 chunks after those (block = window + K+2 chunks, the last 2C rows): the pooler's own
    output, the pooler-seeded optimum and a random-seeded one, and window-only. Neighbour statistics as --oracle-run."""
    T = min(len(ids), P + a.max_len); a_ids = ids[None, P:T]; La = a_ids.size(1); first = a.rw + a.C; K = a.oracle_horizon
    span = (K + 2) * a.C; c0s = [c for c in range(first, La, a.C) if c + span <= La]
    if len(c0s) < 2: return
    ids = ids.to(dev); a_ids = a_ids.to(dev)
    with torch.no_grad():
        ref_h = teacher_hidden(ids, P + first, T)
        pc = DynamicCache(); llm(input_ids=ids[None, :P], past_key_values=pc, use_cache=True)
        pkv = [tuple(t for t in kv_of(pc, li)) for li in range(NL)]
    n_ = max(1, a.oracle_run); st_ = max(0, min(len(c0s) - n_, len(c0s) // 2)); targets = set(c0s[st_:st_ + n_])
    kept = torch.zeros(1, 0, dtype=torch.long, device=dev); absorbed = 0; use_att = bool(ATT_L)
    impT = torch.zeros(La, device=dev); kimp = torch.zeros(1, 0, device=dev); prev = None
    for c0 in range(0, La, a.C):
        c1 = min(c0 + a.C, La); cur = c1 - c0
        if cur < 4: break
        R = min(c0, a.rw); nd_end = c0 - R
        if use_att:
            si = impT[None].clone(); si[:, :2] = 0
            kept, absorbed, kimp = update_buffer(kept, absorbed, a_ids, nd_end, None, imp=kimp, src_imp=si)
        else: kept, absorbed = update_buffer(kept, absorbed, a_ids, nd_end, None)
        hit = c0 in targets and kept.size(1) > 0
        if not hit and not use_att: continue
        with torch.no_grad():
            block = torch.cat([embed(a_ids[:, c0 - R:c0]).to(dt), embed(a_ids[:, c0:c1]).to(dt)], 1); nb = block.size(1)
            sp0 = pooler(embed(kept).float(), kpm=att_bias(kimp) if use_att else None)
            if use_att:
                cache = DynamicCache(); mem = kv_memory(sp0, torch.arange(P, P + a.S, device=dev)[None]); off = P + a.S
                for li, (k_, v_) in enumerate(pkv): cache.update(torch.cat([k_, mem[li][0]], 2), torch.cat([v_, mem[li][1]], 2), li)
                pos = torch.arange(off, off + nb, device=dev)[None]; ATTREC["rows"] = cur
                llm(inputs_embeds=block, attention_mask=torch.ones(1, off + nb, device=dev), past_key_values=cache, position_ids=pos,
                    cache_position=pos[0], use_cache=True, logits_to_keep=1)
                impT[c0 - R:c1] += att_take()[off:off + R + cur]
        if not hit: continue
        fitN = K * a.C; allN = span
        blkF = torch.cat([embed(a_ids[:, c0 - R:c0]).to(dt), embed(a_ids[:, c0:c0 + fitN]).to(dt)], 1); trF = teacher_rows(ref_h[c0 - first:c0 + fitN - first])
        blkT = torch.cat([embed(a_ids[:, c0 - R:c0]).to(dt), embed(a_ids[:, c0:c0 + allN]).to(dt)], 1); trT = teacher_rows(ref_h[c0 - first:c0 + allN - first])
        test = lambda sp_: _mem_kl(sp_, pkv, P, blkT, allN, trT, 0)[fitN:].mean().item()   # the 2 chunks after the fitted K
        with torch.no_grad():
            kl0 = test(sp0); klfit0 = _mem_kl(sp0, pkv, P, blkF, fitN, trF, 0).mean().item()
            cw = DynamicCache()
            for li, (k_, v_) in enumerate(pkv): cw.update(k_, v_, li)
            nbT = blkT.size(1); pos = torch.arange(P, P + nbT, device=dev)[None]
            ow = llm(inputs_embeds=blkT, attention_mask=torch.ones(1, P + nbT, device=dev), past_key_values=cw, position_ids=pos,
                     cache_position=pos[0], use_cache=True, logits_to_keep=allN + 1)
            klw = (trT.exp() * (trT - F.log_softmax(ow.logits[0, :allN].float(), -1))).sum(-1)[fitN:].mean().item(); del ow, cw
        unit = lambda x: x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6); g = torch.Generator(device=dev).manual_seed(1234 + c0)
        init = unit(prev[1].float()) if (a.oracle_chain and prev is not None) else unit(sp0.float())
        with torch.no_grad(): kl_init = test(init * pooler.out_scale.abs()) if (a.oracle_chain and prev is not None) else kl0   # the start's own held-out KL
        spA, hA = _oracle_fit(sp0, init, pkv, P, blkF, fitN, trF)
        spB, hB = _oracle_fit(sp0, unit(torch.randn(sp0.shape, generator=g, device=dev)), pkv, P, blkF, fitN, trF) if a.oracle_random else (sp0, [round(klfit0, 4)])
        with torch.no_grad(): tA, tB = test(spA), test(spB)
        cosf = lambda x, y: F.cosine_similarity(x, y, dim=-1).mean().item()
        ent = {"c0": c0, "nd": int(nd_end), "window": klw, "pooler": kl0, "pooler_fit": klfit0, "seeded_fit": hA[-1], "random_fit": hB[-1],
               "seeded": tA, "random": tB, "cos": cosf(spA, sp0), "cosB": cosf(spB, sp0), "cosAB": cosf(spA, spB), "kl_init": kl_init}
        if a.oracle_content:   # does the pooler's output depend on what it reads?
            with torch.no_grad():
                perm = torch.randperm(kept.size(1), generator=g, device=dev)
                sp_sh = pooler(embed(kept[:, perm]).float(), kpm=att_bias(kimp[:, perm]) if use_att else None)
                oth = OTHER[1] if OTHER_KEY[0] == ids[P:P + 8].tolist() else OTHER[0]
                o_ids = oth[:kept.size(1)].to(dev)[None]
                sp_ot = pooler(embed(o_ids).float(), kpm=att_bias(kimp[:, :o_ids.size(1)]) if use_att else None)
                ent["cos_shuffled"] = cosf(sp0, sp_sh); ent["cos_other"] = cosf(sp0, sp_ot)
                tsh, tot_ = test(sp_sh), test(sp_ot)
                ent["kl_shuffled"] = tsh; ent["kl_other"] = tot_
        if "cos_shuffled" in ent:
            log(f"[sp] oracle-content pos {c0} (distant {nd_end}): the pooler's output vs the same tokens shuffled: cos {ent['cos_shuffled']:.3f} (KL on the held-out chunks "
                f"{ent['kl_shuffled']:.4f} vs {kl0:.4f}), vs another trace's tokens: cos {ent['cos_other']:.3f} (KL {ent['kl_other']:.4f}), window-only {klw:.4f}")
        msg = (f"[sp] oracle-h{K} pos {c0} (distant {nd_end}): fit on {fitN} tokens: pooler {klfit0:.4f} seeded {' > '.join(str(x) for x in hA)} random {hB[-1]:.4f}"
               f" | the 2 chunks after: window-only {klw:.4f} pooler {kl0:.4f} seeded optimum {tA:.4f} random optimum {tB:.4f}"
               f" | cos to the pooler's output: seeded {ent['cos']:.3f} random {ent['cosB']:.3f}, between optima {ent['cosAB']:.3f}")
        if prev is not None:
            p0, pA, pB = prev
            with torch.no_grad(): kpA, kpB = test(pA), test(pB)
            ent["pair"] = {"cos_pooler": cosf(sp0, p0), "cos_seeded": cosf(spA, pA), "cos_random": cosf(spB, pB), "move": ((spA - pA).norm() / pA.norm()).item(),
                           "cos_corr": F.cosine_similarity((spA - sp0).flatten(), (pA - p0).flatten(), dim=0).item(), "kl_prev_seeded": kpA, "kl_prev_random": kpB}
            pr_ = ent["pair"]
            msg += (f" | vs the previous position: cos pooler {pr_['cos_pooler']:.3f} seeded optima {pr_['cos_seeded']:.3f} (move {pr_['move']:.3f}) random optima {pr_['cos_random']:.3f}"
                    f" corrections {pr_['cos_corr']:.3f}; its optima here: seeded {kpA:.4f} random {kpB:.4f}" + (f"; the chain's start here {kl_init:.4f}" if a.oracle_chain else ""))
        log(msg); prev = (sp0, spA, spB); ORC.append(ent)


KVS = {}   # kvsel: (rule, B) -> [KL per position]
SELS = {}   # kvsel rule mlp:<path>: the learned selectors, loaded once


def kvsel_model(path):
    if path not in SELS:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from kvsel_feat import Selector
        rs_ = torch.get_rng_state(); ck = torch.load(path, map_location=dev)   # the layers' init must not move the sampling RNG
        m = Selector(ck["vocab"], mu=ck["mu"], sd=ck["sd"]).to(dev); m.load_state_dict(ck["state"]); m.eval(); SELS[path] = m; torch.set_rng_state(rs_)
    return SELS[path]
DUMPS = []   # kvsel --kvsel-dump: one dict per position


def kvsel_trace(ids, P):
    """kvsel (the user: judge the eviction rules by KL, on paper, before any generation): one full-context pass of the trace with the
    attention each key receives from every chunk recorded per layer; then at every k-th chunk boundary each rule keeps B of the
    older entries (answer tokens before the raw window) beside the prompt and the window, the next chunk is read on that pruned
    cache (entries as computed in full context, true positions) and its KL to the full-context teacher is the rule's score.
    'recent' = the window widened by B; 'grad' peeks at the chunk (first-order loss sensitivity to each entry's values: the
    KL-based label a learned selector would be trained on) - an upper bound, not a deployable rule."""
    T = min(len(ids), P + a.max_len); a_ids = ids[None, P:T]; La = a_ids.size(1); first = a.rw + a.C
    nck = La // a.C
    if nck * a.C < first + 2 * a.C: return
    ids = ids.to(dev); a_ids = a_ids.to(dev)
    cache = DynamicCache(); llm(input_ids=ids[None, :P], past_key_values=cache, use_cache=True, logits_to_keep=1)
    att = []   # per chunk: {layer: (keys,)} attention received by every key from the chunk's rows
    for m in range(nck):
        c0, c1 = m * a.C, (m + 1) * a.C; pos = torch.arange(P + c0, P + c1, device=dev)[None]; ATTREC["rows"] = a.C
        llm(input_ids=a_ids[:, c0:c1], past_key_values=cache, position_ids=pos, cache_position=pos[0], use_cache=True, logits_to_keep=1)
        att.append({li: v[0].clone() for li, v in ATTREC["acc"].items()}); ATTREC["acc"] = {}; ATTREC["rows"] = 0
    full = [tuple(kv_of(cache, li)) for li in range(NL)]; del cache
    ref_h = teacher_hidden(ids, P + first, T)
    budgets = [int(x) for x in a.kvsel_budgets.split(",")]; rules = a.kvsel_rules.split(",")
    g = torch.Generator(device=dev).manual_seed(7)
    def read(keep_by_layer, c0, c1):
        """KL of chunk [c0, c1) read on prompt + kept older entries (per layer: a sorted LongTensor of answer indices) + window [c0-rw, c0-1)"""
        cur = c1 - c0; pc = DynamicCache(); win = torch.arange(c0 - a.rw, c0 - 1, device=dev)
        for li in range(NL):
            idx = torch.cat([torch.arange(P, device=dev), P + keep_by_layer[li], P + win])
            k_, v_ = full[li]; pc.update(k_.index_select(2, idx), v_.index_select(2, idx), li)
        pos = torch.arange(P + c0 - 1, P + c1 - 1, device=dev)[None]
        o = llm(input_ids=a_ids[:, c0 - 1:c1 - 1], past_key_values=pc, position_ids=pos, cache_position=pos[0], use_cache=True, logits_to_keep=cur)
        sl = F.log_softmax(o.logits[0].float(), -1); tr = teacher_rows(ref_h[c0 - first:c1 - first])
        return (tr.exp() * (tr - sl)).sum(-1).mean().item()
    for mb in range(first // a.C, nck - 1, a.kvsel_every):
        c0, c1 = mb * a.C, (mb + 1) * a.C; nold = c0 - a.rw
        if nold < max(budgets) + a.C: continue
        older = torch.arange(nold, device=dev)
        # attention history of the older entries: R[m, e] from chunk m (m < mb), per layer and summed
        Rl = {li: torch.zeros(mb, nold, device=dev) for li in ATT_L}
        for m in range(mb):
            for li in ATT_L:
                v = att[m].get(li)
                if v is not None: n_ = min(nold, v.numel() - P); Rl[li][m, :n_] = v[P:P + n_]
        Rs = sum(Rl.values())
        born = older // a.C; nobs = (mb - born).clamp(min=1).float()
        def scores(R):
            w = {"cum": R.sum(0), "mean": R.sum(0) / nobs, "last": R[-1]}
            for lam in (0.9, 0.98):
                dec = lam ** torch.arange(mb - 1, -1, -1, device=dev).float(); w[f"ema{lam}"] = (dec[:, None] * R).sum(0)
            return w
        Ssum = scores(Rs); Slay = {li: scores(Rl[li]) for li in ATT_L} if a.kvsel_layerwise else {}
        fut = torch.zeros(nold, device=dev)   # the attention the older entries WILL receive from the next 4 chunks (an oracle rule, and a label)
        for m in range(mb, min(mb + 4, nck)):
            for li in ATT_L:
                v = att[m].get(li)
                if v is not None: fut += v[P:P + nold]
        grad_imp = None
        if "grad" in rules:   # first-order sensitivity of the chunk's loss to each entry's values, full cache (the KL-based label)
            cur = c1 - c0; pc = DynamicCache(); sc_ = []; win = torch.arange(c0 - a.rw, c0 - 1, device=dev)
            idx = torch.cat([torch.arange(P, device=dev), P + older, P + win])
            with torch.enable_grad():
                for li in range(NL):
                    k_, v_ = full[li]; vv = v_.index_select(2, idx); s_ = torch.ones(1, 1, vv.size(2), 1, device=dev, dtype=vv.dtype, requires_grad=True)
                    sc_.append(s_); pc.update(k_.index_select(2, idx), vv * s_, li)
                pos = torch.arange(P + c0 - 1, P + c1 - 1, device=dev)[None]
                o = llm(input_ids=a_ids[:, c0 - 1:c1 - 1], past_key_values=pc, position_ids=pos, cache_position=pos[0], use_cache=True, logits_to_keep=cur)
                nll = -F.log_softmax(o.logits[0].float(), -1).gather(1, a_ids[0, c0:c1, None]).sum()
                nll.backward()
            gl = {li: (-sc_[li].grad[0, 0, P:P + nold, 0]).float() for li in range(NL)}
            grad_imp = sum(gl.values()); del pc, o, sc_
        full_kl = read({li: older for li in range(NL)}, c0, c1); KVS.setdefault(("full", 0), []).append(full_kl)
        win_kl = read({li: older[:0] for li in range(NL)}, c0, c1); KVS.setdefault(("window", 0), []).append(win_kl)
        abl_imp = None
        if "abl" in rules:   # exact: the KL after removing each block of consecutive older entries (all else kept)
            bs = a.kvsel_abl; abl_imp = torch.zeros(nold, device=dev)
            for b0 in range(0, nold, bs):
                keep_ = torch.cat([older[:b0], older[b0 + bs:]])
                abl_imp[b0:b0 + bs] = read({li: keep_ for li in range(NL)}, c0, c1) - full_kl
        if grad_imp is not None and abl_imp is not None:
            rk = lambda x: x.argsort().argsort().float()
            KVS.setdefault(("spearman grad-abl", -1), []).append(torch.corrcoef(torch.stack([rk(grad_imp), rk(abl_imp)]))[0, 1].item())
        if a.kvsel_dump:
            with torch.no_grad():
                vn = torch.stack([full[li][1][0, :, P:P + nold].norm(dim=-1).mean(0) for li in range(NL)], 1)   # (nold, NL) value norms
                kn = torch.stack([full[li][0][0, :, P:P + nold].norm(dim=-1).mean(0) for li in range(NL)], 1)
                feat = {"sum": torch.stack([Ssum[k] for k in ("cum", "mean", "last", "ema0.9", "ema0.98")] + [Rs.max(0).values], 1),
                        "layer": torch.stack([torch.stack([Slay[li][k] for k in ("cum", "mean", "last", "ema0.9", "ema0.98")], 1) for li in ATT_L], 1) if Slay else None,
                        "nobs": nobs, "age": (mb - born).float(), "rel": older.float() / c0, "tok": a_ids[0, :nold].clone(), "vnorm": vn, "knorm": kn,
                        "grad": grad_imp, "abl": abl_imp, "fut": fut, "c0": c0, "P": P, "full_kl": full_kl, "window_kl": win_kl}
                DUMPS.append({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in feat.items()})
        for B in budgets:
            for r in rules:
                if r == "recent": top = older[-B:]
                elif r == "random": top = older[torch.randperm(nold, generator=g, device=dev)[:B]].sort().values
                elif r == "grad": top = grad_imp.topk(B).indices.sort().values if grad_imp is not None else None
                elif r == "abl": top = abl_imp.topk(B).indices.sort().values if abl_imp is not None else None
                elif r == "fut": top = fut.topk(B).indices.sort().values
                elif r.startswith("mlp:"):
                    from kvsel_feat import numeric
                    vn = torch.stack([full[li][1][0, :, P:P + nold].norm(dim=-1).mean(0) for li in range(NL)], 1)
                    kn = torch.stack([full[li][0][0, :, P:P + nold].norm(dim=-1).mean(0) for li in range(NL)], 1)
                    x6 = torch.stack([Ssum[k] for k in ("cum", "mean", "last", "ema0.9", "ema0.98")] + [Rs.max(0).values], 1)
                    sc = kvsel_model(r[4:])(numeric(x6, nobs, (mb - born).float(), older.float() / c0, vn, kn), a_ids[0, :nold])
                    top = sc.topk(B).indices.sort().values
                else: top = Ssum[r].topk(B).indices.sort().values
                if top is None: continue
                KVS.setdefault((r, B), []).append(read({li: top for li in range(NL)}, c0, c1))
                if a.kvsel_layerwise and r in ("cum", "mean", "last", "ema0.9", "ema0.98"):
                    kb = {li: (Slay[li][r] if li in Slay else Ssum[r]).topk(B).indices.sort().values for li in range(NL)}
                    KVS.setdefault((r + "/layer", B), []).append(read(kb, c0, c1))
                if a.kvsel_layerwise and r == "grad" and grad_imp is not None:
                    kb = {li: gl[li].topk(B).indices.sort().values for li in range(NL)}
                    KVS.setdefault(("grad/layer", B), []).append(read(kb, c0, c1))
        log(f"[sp] kvsel pos {c0} (older {nold}): full {full_kl:.4f} window {win_kl:.4f} | " + " ".join(f"{r}:{B}={KVS[(r, B)][-1]:.4f}" for B in budgets for r in rules if (r, B) in KVS)
            + (f" | spearman grad-abl {KVS[('spearman grad-abl', -1)][-1]:.3f}" if ("spearman grad-abl", -1) in KVS else ""))


def run_trace(ids, P, train, ref_h=None):
    """teacher-forced SP-evict over one trace: returns (sum KL, sum top-1 agree, n tokens) on tokens past the window."""
    T = min(len(ids), P + a.max_len)
    a_ids = ids[None, P:T]; La = a_ids.size(1)
    first = a.rw + a.C          # chunks before this see the whole answer so far in the raw window (no eviction yet)
    if La - first < a.C: return 0.0, 0.0, 0
    ids = ids.to(dev); a_ids = a_ids.to(dev)
    if ref_h is None: ref_h = teacher_hidden(ids, P + first, T)
    c0s = list(range(first, La, a.C))
    pick = set(c0s)
    if train and a.chunks_per_trace and len(c0s) > a.chunks_per_trace:
        pick = set(random.sample(c0s, a.chunks_per_trace))
    gen = not train and a.gen_kl > 0
    if gen:   # G evenly spaced chunk starts, the same for every checkpoint; the teacher's cache of the true text, once
        ng = min(a.gen_kl, len(c0s)); pick = {c0s[int((i + 0.5) * len(c0s) / ng)] for i in range(ng)}
        on = LORA["on"]; LORA["on"] = False
        try:
            with torch.no_grad(), _Frozen():
                tc = DynamicCache(); llm.model(input_ids=ids[None, :T], past_key_values=tc, use_cache=True)
        finally:
            LORA["on"] = on
    with torch.no_grad():
        pc = DynamicCache(); llm(input_ids=ids[None, :P], past_key_values=pc, use_cache=True)
        pkv = [tuple(t for t in kv_of(pc, li)) for li in range(NL)]
        feat, src = None, a_ids
        if a.ctx_hidden:   # contextualised pooler input: buffer holds answer positions, features = layer-N hidden states
            with _Frozen():
                hs = llm.model(input_ids=ids[None, :T], output_hidden_states=True).hidden_states[a.ctx_hidden][0, P:T].float()
            hs = hs / hs.norm(dim=-1, keepdim=True).clamp(min=1e-6) * target_norm
            feat = lambda k: hs[k[0]][None]; src = torch.arange(La, device=dev)[None]
    kept = torch.zeros(1, 0, dtype=torch.long, device=dev); absorbed = 0
    K = A_ = N = 0.0
    use_att = bool(ATT_L)   # the model's attention importance of each answer token, recorded chunk by chunk as in the rollout
    if use_att: impT = torch.zeros(La, device=dev); kimp = torch.zeros(1, 0, device=dev)
    for c0 in range(0 if use_att else first, La, a.C):
        c1 = min(c0 + a.C, La); cur = c1 - c0
        if cur < 4: break
        R = min(c0, a.rw); nd_end = c0 - R
        if use_att:
            si = impT[None].clone(); si[:, :2] = 0   # the answer's first tokens (<think>, a newline) are the model's own sinks
            kept, absorbed, kimp = update_buffer(kept, absorbed, src, nd_end, feat, imp=kimp, src_imp=si)
        else: kept, absorbed = update_buffer(kept, absorbed, src, nd_end, feat)
        imp_only = c0 < first or c0 not in pick   # importance only: no loss, no gradient
        if imp_only and not use_att: continue
        with torch.set_grad_enabled(train and not imp_only):
            pn = pin_of(a_ids[0], c0 - R)
            parts = [embed(torch.tensor([pn], device=dev)).to(dt)] if pn else []
            if not a.no_sp and not a.kv_mem: parts.append(slot(kept, feat))
            block = torch.cat(parts + [embed(a_ids[:, c0 - R:c0]).to(dt), embed(a_ids[:, c0:c1]).to(dt)], 1)
            nb = block.size(1)
            cache = DynamicCache(); off = P
            if a.kv_mem and not a.no_sp and kept.size(1) > 0:   # prompt KV + the memory's per-layer KV, then the block
                sp = pooler(feat(kept) if feat else embed(kept).float(), kpm=att_bias(kimp) if use_att else None)
                if use_att and not ATTLOG[0]:
                    ATTLOG[0] = True; sh = kimp[0].clamp(min=0); sh = sh / sh.sum().clamp(min=1e-9); bb = att_bias(kimp)
                    log(f"[sp] att-prior {a.att_prior} (train): first distant buffer {sh.numel()} tokens, importance share of the top 20 "
                        f"{sh.topk(min(20, sh.numel())).values.sum().item():.2f}, zero-importance tokens {(sh == 0).sum().item()}, "
                        f"bias range {bb.min().item():+.2f}..{bb.max().item():+.2f}, pooler logits capped at {a.att_cap or 'inf'}")
                mem = kv_memory(sp, torch.arange(P, P + a.S, device=dev)[None])
                for li, (k_, v_) in enumerate(pkv): cache.update(torch.cat([k_, mem[li][0]], 2), torch.cat([v_, mem[li][1]], 2), li)
                off = P + a.S
            else:
                for li, (k_, v_) in enumerate(pkv): cache.update(k_, v_, li)
            pos = torch.arange(off, off + nb, device=dev)[None]
            if use_att: ATTREC["rows"] = cur
            o = llm(inputs_embeds=block, attention_mask=torch.ones(1, off + nb, device=dev), past_key_values=cache,
                    position_ids=pos, cache_position=pos[0], use_cache=True, logits_to_keep=1 if imp_only else cur + 1)
            if use_att:
                w_ = att_take(); npin = len(pn)
                impT[c0 - R:c1] += w_[off + npin:off + npin + R + cur]
            if imp_only: continue
            # kept rows: the last window token + the chunk's tokens but its last -> they predict answer tokens c0 .. c1-1
            sl = F.log_softmax(o.logits[0, :cur].float(), -1)
            tr = teacher_rows(ref_h[c0 - first:c1 - first])     # teacher row j predicts answer token first + j
            if a.ctl_window and train: tr = ctl_target(tr, pkv, block, P, cur, a_ids[0, c0 - 1:c1 - 1])
            kl = (tr.exp() * (tr - sl)).sum(-1)
            rk = (sl.exp() * (sl - tr)).sum(-1)
            RKL[0] += rk.sum().item(); RKL[1] += rk.numel()
            if train: (((1 - a.rkl_weight) * kl + a.rkl_weight * rk).sum() / 1000.0).backward()
            if gen:
                gk = gen_chunk(cache, off + nb - cur, sl[0], tr[0], tc, P + c0, int(ids[P:P + 8].sum()) * 1009 + c0,
                               a_ids[0, c0:c1] if a.gen_force else None)
                if a.gen_force: GEN.setdefault("diff", []).append((gk - kl[:gk.numel()]).abs().max().item())
        if c0 >= a.score_from:
            K += kl.sum().item(); A_ += (sl.argmax(-1) == tr.argmax(-1)).float().sum().item(); N += kl.numel()
            if DUMP is not None and not train:
                tk = a_ids[0, c0:c1]
                DUMP[-1].append(torch.stack([torch.arange(c0, c1, device=dev).float(), tk.float(), sl.gather(1, tk[:, None])[:, 0],
                                             tr.gather(1, tk[:, None])[:, 0], kl, (sl.exp() * (sl - tr)).sum(-1),
                                             (sl.argmax(-1) == tr.argmax(-1)).float()], 1).cpu())
            if a.end_stat:
                ENDST["pt"] += tr[:, END_ID].exp().sum().item(); ENDST["ps"] += sl[:, END_ID].exp().sum().item(); ENDST["n"] += kl.numel()
                ENDST["ht"] += tr[:, HES_IDS].exp().sum().item(); ENDST["hs"] += sl[:, HES_IDS].exp().sum().item()
                ENDST["et"] -= (tr.exp() * tr).sum().item(); ENDST["es"] -= (sl.exp() * sl).sum().item()
                hit = (a_ids[0, c0:c1] == END_ID).nonzero().flatten()
                if hit.numel(): ENDST["lt"] += tr[hit, END_ID].sum().item(); ENDST["ls"] += sl[hit, END_ID].sum().item(); ENDST["k"] += hit.numel()
    return K, A_, N


class PrefixCache(DynamicCache):
    """a cache that, once frozen, prepends its stored prefix (prompt KV + memory KV) to each layer's new keys/values without
    storing them - so a layer recomputed by gradient checkpointing sees the same keys as the first time"""
    frozen = False

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.frozen:
            pk, pv = kv_of(self, layer_idx)
            return torch.cat([pk, key_states], 2), torch.cat([pv, value_states], 2)
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


def seq_logp_kv(ids, P, coef=None, temp=1.0):
    """seq_logp under the --kv-mem deployment loop (rollout_kvmem): every C tokens the distant tokens are pooled into the
    per-layer memory KV (none while nothing is distant), the cache is [prompt KV | memory KV at P..P+S-1 | raw window]
    and the chunk's tokens are scored; the first answer token (predicted by the prompt alone) is left out."""
    T = min(len(ids), P + a.max_len); ids = ids.to(dev); a_ids = ids[None, P:T]; La = a_ids.size(1)
    was = llm.training; llm.eval()   # the prompt prefill fills a cache: not in train mode (gradient checkpointing drops caches)
    with torch.no_grad():
        pc = DynamicCache(); llm(input_ids=ids[None, :P], past_key_values=pc, use_cache=True)
        pkv = [tuple(t for t in kv_of(pc, li)) for li in range(NL)]
    llm.train(was)
    kept = torch.zeros(1, 0, dtype=torch.long, device=dev); absorbed = 0; tot = 0.0; n = 0
    use_att = bool(ATT_L)
    if use_att: impT = torch.zeros(La, device=dev); kimp = torch.zeros(1, 0, device=dev)
    for c0 in range(0, La, a.C):
        c1 = min(c0 + a.C, La); R = min(c0, a.rw); lo = c0 - R
        if use_att:
            si = impT[None].clone(); si[:, :2] = 0
            kept, absorbed, kimp = update_buffer(kept, absorbed, a_ids, lo, imp=kimp, src_imp=si)
        else: kept, absorbed = update_buffer(kept, absorbed, a_ids, lo)
        use = kept.size(1) > 0 and not a.no_sp; t0 = c0 if c0 > 0 else 1
        if c1 - t0 < 1: continue
        trn = coef is not None and (use or a.ft_layers > 0 or bool(LORA_PARAMS))   # no memory yet: nothing to train here unless the model itself is tuned
        with torch.set_grad_enabled(trn):
            cache = PrefixCache(); off = P
            if use:
                mem = kv_memory(pooler(embed(kept).float(), kpm=att_bias(kimp) if use_att else None), torch.arange(P, P + a.S, device=dev)[None])
                for li, (k_, v_) in enumerate(pkv): cache.update(torch.cat([k_, mem[li][0]], 2), torch.cat([v_, mem[li][1]], 2), li)
                off = P + a.S
            else:
                for li, (k_, v_) in enumerate(pkv): cache.update(k_, v_, li)
            cache.frozen = True
            blk = a_ids[:, lo:c1 - 1]; nb = blk.size(1); pos = torch.arange(off, off + nb, device=dev)[None]
            if use_att: ATTREC["rows"] = nb if c0 == 0 else c1 - c0   # this chunk's rows and the last of the one before: each token a query once
            hs = llm.model(inputs_embeds=embed(blk).to(dt), attention_mask=torch.ones(1, off + nb, device=dev), past_key_values=cache,
                           position_ids=pos, cache_position=pos[0], use_cache=True).last_hidden_state[0]
            if use_att: impT[lo:c1 - 1] += att_take()[off:off + nb]
            h0 = t0 - lo - 1   # block row r predicts answer token lo + r + 1
            lp = F.log_softmax(llm.lm_head(hs[h0:h0 + c1 - t0]).float() / temp, -1).gather(1, a_ids[0, t0:c1, None]).sum()
            if trn: (coef * lp).backward()
            lp = lp.detach()
        tot += lp.item(); n += c1 - t0
    return tot, n


def seq_logp(ids, P, coef=None, temp=1.0):
    """log-prob of the answer ids[P:] under the deployment loop (the rollout's): every C tokens the distant tokens are pooled
    (mass-evicted at maxd) into SP - none while nothing is distant - and [prompt | SP | raw window] re-prefilled; the chunk's
    tokens are scored. coef: backward of coef x each chunk's log-prob (the gradient of a loss linear in the sum). The first
    answer token (predicted by the prompt alone) is left out. temp: log-probs of softmax(logits / temp) - the sampling
    temperature, so a policy gradient scores the distribution the samples came from. Returns (sum log p, tokens)."""
    if a.kv_mem: return seq_logp_kv(ids, P, coef, temp)
    T = min(len(ids), P + a.max_len); ids = ids.to(dev); a_ids = ids[None, P:T]; La = a_ids.size(1)
    with torch.no_grad():
        pc = DynamicCache(); llm(input_ids=ids[None, :P], past_key_values=pc, use_cache=True)
        pkv = [tuple(t for t in kv_of(pc, li)) for li in range(NL)]
    kept = torch.zeros(1, 0, dtype=torch.long, device=dev); absorbed = 0; tot = 0.0; n = 0
    for c0 in range(0, La, a.C):
        c1 = min(c0 + a.C, La); R = min(c0, a.rw); lo = c0 - R
        kept, absorbed = update_buffer(kept, absorbed, a_ids, lo)
        use_sp = kept.size(1) > 0; nsp = a.S if use_sp else 0
        t0 = c0 if (use_sp or c0 > lo) else 1
        if c1 - t0 < 1: continue
        with torch.set_grad_enabled(coef is not None):
            parts = [pooler(embed(kept).float()).to(dt)] if use_sp else []
            block = torch.cat(parts + [embed(a_ids[:, lo:c1 - 1]).to(dt)], 1); nb = block.size(1)
            cache = DynamicCache()
            for li, (k_, v_) in enumerate(pkv): cache.update(k_, v_, li)
            pos = torch.arange(P, P + nb, device=dev)[None]
            hs = llm.model(inputs_embeds=block, attention_mask=torch.ones(1, P + nb, device=dev), past_key_values=cache,
                           position_ids=pos, cache_position=pos[0], use_cache=True).last_hidden_state[0]
            h0 = nsp + (t0 - lo) - 1
            lp = F.log_softmax(llm.lm_head(hs[h0:h0 + c1 - t0]).float() / temp, -1).gather(1, a_ids[0, t0:c1, None]).sum()
            if coef is not None: (coef * lp).backward()
        tot += lp.item(); n += c1 - t0
    return tot, n


SPIN_REF = {}


def run_group(st):
    """one pre-generated GRPO group (--grpo-groups): backward of -(reward - mean) x sequence log-prob for each member"""
    P, seqs, rs = GROUPS[st % len(GROUPS)]; m_ = sum(rs) / len(rs)
    for ids, r_ in zip(seqs, rs):
        seq_logp(ids, P, coef=-(r_ - m_) / (len(rs) * a.accum * a.grpo_lnorm), temp=a.temp)
    return sum(rs[:len(rs)]) / len(rs), 1.0, 1


def run_pair_spin(st):
    """one SPIN (DPO) pair: margins from a no-grad pass, then the backward with the pair's coefficients"""
    ids_c, ids_r, P = SPIN[st % len(SPIN)]; rc, rr = SPIN_REF[st % len(SPIN)]
    with torch.no_grad():
        lc, nc = seq_logp(ids_c, P); lr_, nr = seq_logp(ids_r, P)
    m = a.spin_beta * ((lc - rc) - (lr_ - rr)); sg = 1.0 / (1.0 + math.exp(m))      # sigmoid(-m)
    cc = (-a.spin_beta * sg - a.spin_sft / max(nc, 1)) / a.accum; cr = a.spin_beta * sg / a.accum
    seq_logp(ids_c, P, coef=cc); seq_logp(ids_r, P, coef=cr)
    loss = math.log1p(math.exp(-m)) if m > -30 else -m
    return loss, float(m > 0), 1


from torch.utils.checkpoint import checkpoint


def _kl_rows(h, tr, w=None):
    """KL(teacher rows || student rows from hidden h) summed (row weights w), plus top-1 agreements (logits rebuilt in
    backward)"""
    lp = F.log_softmax(llm.lm_head(h).float(), -1)
    kl = (tr.exp() * (tr - lp)).sum(-1)
    return (kl if w is None else kl * w).sum(), (lp.argmax(-1) == tr.argmax(-1)).float().sum().detach()


def run_trace_par(ids, P, ref_h=None):
    """--par training on one trace: buffer snapshots at chunk boundaries (sequential, no grad - eviction never had a
    gradient), then for a few boundaries b one forward of [prompt | SP_b | a[b : b+rw+C]] scoring all rw+C predictions."""
    T = min(len(ids), P + a.max_len)
    ids = ids.to(dev); a_ids = ids[None, P:T]; La = a_ids.size(1)
    L = a.rw + a.C
    bounds = [b for b in range(a.C, La - L + 1, a.C)]
    ends = (a_ids[0] == END_ID).nonzero().flatten().tolist() if a.end_train or a.end_weight != 1 else []
    if a.end_train:   # blocks may be cut by the trace's end: short traces and the tail are trained too
        bounds = [b for b in range(a.C, La - 16, a.C)]
    if not bounds: return 0.0, 0.0, 0
    if ref_h is None: ref_h = teacher_hidden(ids, P + a.C, T)     # row j predicts answer token C + j
    pick = set(random.sample(bounds, min(len(bounds), a.chunks_per_trace or len(bounds))))
    if a.end_train and ends:   # the block whose window holds the end of the thought (and what follows it)
        e = ends[0]; cand = [b for b in bounds if e - L + 65 <= b <= e]; pick.add(cand[0] if cand else bounds[-1])
    with torch.no_grad():
        pc = DynamicCache(); llm(input_ids=ids[None, :P], past_key_values=pc, use_cache=True)
        pkv = [tuple(t for t in kv_of(pc, li)) for li in range(NL)]
    kept = torch.zeros(1, 0, dtype=torch.long, device=dev); absorbed = 0
    K = A_ = N = 0.0
    if a.par_deploy:   # a block's scored chunk must exist: a[b+rw ...]
        bounds = [b for b in bounds if b + a.rw < La]; pick = {b for b in pick if b + a.rw < La}
        if not pick: return 0.0, 0.0, 0
    if a.end_train and La > a.C + 16 and not a.par_deploy: bounds, pick = [0] + bounds, pick | {0}   # + the plain window from the start (the
    # deployment's first chunks: nothing distant yet, no SP) - scored from token C, where the teacher rows begin
    for b in bounds:
        kept, absorbed = update_buffer(kept, absorbed, a_ids, b)
        if b not in pick: continue
        Lb = min(L, La - b)
        if b == 0:
            block = embed(a_ids[:, :Lb - 1]).to(dt); h0, t0 = a.C - 1, a.C        # row C-1 predicts a[C]
        else:
            sp = pooler(embed(kept).float()) if kept.size(1) > 0 else pooler(torch.zeros(1, 0, H, device=dev))
            block = torch.cat([sp.to(dt), embed(a_ids[:, b:b + Lb - 1]).to(dt)], 1); h0, t0 = a.S - 1, b   # predicts a[b .. b+Lb-1]
            if a.par_deploy: h0, t0 = h0 + a.rw, b + a.rw   # only a[b+rw ..]: the chunk after a full raw window
        nb = block.size(1); nr = b + Lb - t0
        cache = DynamicCache()
        for li, (k_, v_) in enumerate(pkv): cache.update(k_, v_, li)
        pos = torch.arange(P, P + nb, device=dev)[None]
        hs = llm.model(inputs_embeds=block, attention_mask=torch.ones(1, P + nb, device=dev), past_key_values=cache,
                       position_ids=pos, cache_position=pos[0], use_cache=True).last_hidden_state[0, h0:]   # (nr, H)
        loss = 0.0
        wv = None
        if ends and a.end_weight != 1:
            tpos = torch.arange(t0, t0 + nr, device=dev); wv = torch.ones(nr, device=dev)
            for e in ends: wv[(tpos >= e - 32) & (tpos <= e + 8)] = a.end_weight
        KR = a.kl_rows; kb = 0.0
        for i in range(0, nr, KR):
            tr = teacher_rows(ref_h[t0 - a.C + i:t0 - a.C + min(i + KR, nr)])
            kl, ag = checkpoint(_kl_rows, hs[i:i + KR], tr, None if wv is None else wv[i:i + KR], use_reentrant=False)
            loss = loss + kl; K += kl.item(); A_ += ag.item(); N += tr.shape[0]; kb += kl.item()
        if b > 0:   # the past outside the window = tokens before b
            bk = sum(b >= e for e in DB_EDGES); v = DBK.setdefault(bk, [0.0, 0]); v[0] += kb; v[1] += nr
        (loss / 1000.0).backward()
    return K, A_, N


WTPOS = [[0.0, 0] for _ in range(4)]   # --win-teacher eval: KL by chunk position (0, 1-7, 8-63, 64+)


def run_trace_wt(ids, P, train):
    """--win-teacher on one trace: for chunk starts c0 >= rw (all, or --chunks-per-trace at random) the teacher reads
    [prompt | a[c0-rw : c0+C-1]] and the student [prompt | SP(a[c0-rw : c0]) | a[c0 : c0+C-1]] (--no-sp: [prompt | chunk]);
    forward KL on a[c0 .. c0+C-1]. Teacher passes first (frozen weights), student passes after, batched by --wt-batch."""
    T = min(len(ids), P + a.max_len); ids = ids.to(dev); a_ids = ids[None, P:T]; La = a_ids.size(1)
    c0s = list(range(a.rw, La - a.C + 1, a.C))
    if not c0s: return 0.0, 0.0, 0
    if train and a.chunks_per_trace and len(c0s) > a.chunks_per_trace: c0s = sorted(random.sample(c0s, a.chunks_per_trace))
    Bt = max(1, a.wt_batch); groups = [c0s[i:i + Bt] for i in range(0, len(c0s), Bt)]

    def prompt_cache(Pn):
        pc = DynamicCache(); llm(input_ids=ids[None, :Pn], past_key_values=pc, use_cache=True)
        return [tuple(kv_of(pc, li)) for li in range(NL)]

    def cache_for(pkv, B):
        c = DynamicCache()
        for li, (k_, v_) in enumerate(pkv): c.update(k_.expand(B, -1, -1, -1), v_.expand(B, -1, -1, -1), li)
        return c
    with torch.no_grad():
        with _Frozen():
            pkv_t = prompt_cache(P); TH = []
            for cs in groups:
                B = len(cs); x = torch.stack([a_ids[0, c - a.rw:c + a.C - 1] for c in cs]); L = x.size(1)
                pos = torch.arange(P, P + L, device=dev)[None].expand(B, -1)
                TH.append(llm.model(input_ids=x, attention_mask=torch.ones(B, P + L, device=dev), past_key_values=cache_for(pkv_t, B),
                                    position_ids=pos, cache_position=pos[0], use_cache=True).last_hidden_state[:, a.rw - 1:])
            del pkv_t
        P0 = P - 1 if a.no_sp else P            # --no-sp: the block starts with the prompt's last token (its row predicts a[c0])
        pkv_s = prompt_cache(P0)
    K = A_ = N = 0.0
    for cs, th in zip(groups, TH):
        B = len(cs)
        with torch.set_grad_enabled(train):
            ch = embed(torch.stack([a_ids[0, c:c + a.C - 1] for c in cs])).to(dt)
            if a.no_sp: xin = torch.cat([embed(ids[P - 1:P])[None].expand(B, -1, -1).to(dt), ch], 1); h0 = 0
            else:
                sp = pooler(embed(torch.stack([a_ids[0, c - a.rw:c] for c in cs])).float()).to(dt)
                xin = torch.cat([sp, ch], 1); h0 = a.S - 1
            L = xin.size(1); pos = torch.arange(P0, P0 + L, device=dev)[None].expand(B, -1)
            hs = llm.model(inputs_embeds=xin, attention_mask=torch.ones(B, P0 + L, device=dev), past_key_values=cache_for(pkv_s, B),
                           position_ids=pos, cache_position=pos[0], use_cache=True).last_hidden_state[:, h0:].reshape(-1, H)
            thr = th.reshape(-1, H); loss = 0.0
            for j in range(0, hs.size(0), a.kl_rows):
                tr = teacher_rows(thr[j:j + a.kl_rows])
                if train: kl, ag = checkpoint(_kl_rows, hs[j:j + a.kl_rows], tr, None, use_reentrant=False)
                else:   # eval: the KL by position in the chunk as well (0 = right after the soft prompts)
                    lp = F.log_softmax(llm.lm_head(hs[j:j + a.kl_rows]).float(), -1); kr = (tr.exp() * (tr - lp)).sum(-1)
                    kl, ag = kr.sum(), (lp.argmax(-1) == tr.argmax(-1)).float().sum()
                    for q, v in zip(range(j, j + kr.numel()), kr.tolist()):
                        w = q % a.C; b_ = 0 if w == 0 else 1 if w < 8 else 2 if w < 64 else 3
                        WTPOS[b_][0] += v; WTPOS[b_][1] += 1
                loss = loss + kl; K += kl.item(); A_ += ag.item(); N += tr.shape[0]
            if train: (loss / 1000.0).backward()
    return K, A_, N


def run_trace_rmt(ids, P, train):
    """--rmt on one trace: memory chain over the answer's blocks (no grad), then for chunk starts c0 (>= rw + block: at least
    one block written) the frozen model reads [prompt | M_n | a[n*block : c0] | chunk], n = (c0 - rw) // block, and the KL
    to the full-KV teacher is taken on the chunk's tokens; in training the last --rmt-bptt writes are redone with grad.
    --no-sp: the same raw part without the memory (the floor); --sp-null: the first memory m0 in place of M_n."""
    T = min(len(ids), P + a.max_len); ids = ids.to(dev); a_ids = ids[P:T]; La = a_ids.numel(); blk = a.rmt_block
    c0s = list(range(a.rw + blk, La - 8, a.C))
    if not c0s: return 0.0, 0.0, 0
    pick = sorted(random.sample(c0s, a.chunks_per_trace)) if train and a.chunks_per_trace and len(c0s) > a.chunks_per_trace else c0s
    ref_h = teacher_hidden(ids, P + c0s[0], T)            # row j predicts answer token c0s[0] + j
    pids = ids[:P]; nmax = (pick[-1] - a.rw) // blk
    with torch.no_grad():
        Ms = [RMT.m0[None]]
        for k in range(nmax): Ms.append(rmt_write(Ms[-1], pids, a_ids[k * blk:(k + 1) * blk]))
    K = A_ = N = 0.0
    for c in pick:
        n = (c - a.rw) // blk; c1 = min(c + a.C, La); cur = c1 - c
        with torch.set_grad_enabled(train):
            if a.no_sp: mem = None
            elif a.sp_null: mem = RMT.m0[None]
            elif train:
                mem = Ms[max(0, n - a.rmt_bptt)].detach()
                for k in range(max(0, n - a.rmt_bptt), n): mem = rmt_write(mem, pids, a_ids[k * blk:(k + 1) * blk])
            else: mem = Ms[n]
            # the reading pass, prompt included (no cache: the same in training with gradient checkpointing and in generation)
            parts = [embed(pids[None]).to(dt)] + ([mem.to(dt)] if mem is not None else []) + [embed(a_ids[None, n * blk:c1 - 1]).to(dt)]
            x = torch.cat(parts, 1)
            h0 = P + (0 if mem is None else a.S) + (c - 1 - n * blk)   # the row of a[c-1], which predicts a[c]
            hs = llm.model(inputs_embeds=x, position_ids=torch.arange(x.size(1), device=dev)[None],
                           use_cache=False).last_hidden_state[0, h0:]   # (cur, H)
            tr = teacher_rows(ref_h[c - c0s[0]:c1 - c0s[0]])
            lp = F.log_softmax(llm.lm_head(hs).float(), -1); kr = (tr.exp() * (tr - lp)).sum(-1)
            if train: (kr.sum() / 1000.0).backward()
        K += kr.sum().item(); A_ += (lp.argmax(-1) == tr.argmax(-1)).float().sum().item(); N += kr.numel()
        if not train:
            for w, v in enumerate(kr.tolist()):
                b_ = 0 if w == 0 else 1 if w < 8 else 2 if w < 64 else 3
                WTPOS[b_][0] += v; WTPOS[b_][1] += 1
    return K, A_, N


SNW = {"w": 1, "k0": 0.0, "n0": 0}
DBK = {}   # --par: training KL by how much past lies outside the window (bucket -> [sum KL, rows]), logged and reset
DB_EDGES = [1024, 2048, 4096, 8192]   # spnext: the current curriculum window and the window-0 KL since the last log


def run_trace_spnext(ids, P, ref_h=None):
    """--spnext training on one trace: the buffer before every position t (snapshots every sn_step tokens, mass-evicted,
    no grad) -> SP_t -> [prompt | SP_t | a[t : t+w-1]] predicts answer tokens t .. t+w-1 (windows 0 .. w-1, w = the
    curriculum step, 1 = SP alone); positions batched, backward per batch."""
    w = SNW["w"]
    T = min(len(ids), P + a.max_len)
    ids = ids.to(dev); a_ids = ids[None, P:T]; La = a_ids.size(1); t0 = a.sn_t0
    if La - t0 - w + 1 < a.sn_batch // 4: return 0.0, 0.0, 0
    if ref_h is None: ref_h = teacher_hidden(ids, P + t0, T)          # row j predicts answer token t0 + j
    with torch.no_grad():
        pc = DynamicCache(); llm(input_ids=ids[None, :P], past_key_values=pc, use_cache=True)
        pkv = [tuple(t for t in kv_of(pc, li)) for li in range(NL)]
        snaps = {}; kept = torch.zeros(1, 0, dtype=torch.long, device=dev); absorbed = 0
        for g in ([] if (a.maxd == 0 and not a.fifo and not pooler.pe_rev) else range(0, La, a.sn_step)):
            kept, absorbed = update_buffer(kept, absorbed, a_ids, g); snaps[g] = kept[0]
    shared = a.maxd == 0 and not a.fifo and not pooler.pe_rev        # the buffer before t is just a[:t]: project the past once
    past_full = embed(a_ids).float() if shared else None               # (1, La, H)
    pos = list(range(t0, La - w + 1))
    npos = max(16, a.sn_pos // w) if a.sn_pos else 0                  # about the same scored tokens per trace at every w
    if npos and len(pos) > npos: pos = sorted(random.sample(pos, npos))
    K = A_ = N = 0.0
    bsz = max(1, min(a.sn_batch, a.sn_tokens // (P + a.S + w), max(1, 256 // min(w, 256))))   # <= ~256 scored rows (B x w x vocab) per forward
    for i in range(0, len(pos), bsz):
        ts = pos[i:i + bsz]; B = len(ts)
        if shared:
            sp = pooler.forward_prefix(past_full, torch.tensor(ts, device=dev))   # (B, S, H)
        else:
            bufs = []
            for t in ts:
                g = (t // a.sn_step) * a.sn_step
                bufs.append(torch.cat([snaps[g], a_ids[0, g:t]]))             # snapshot + the raw tokens since it
            Lb = max(1, max(len(x) for x in bufs))
            pad = torch.zeros(B, Lb, dtype=torch.long, device=dev); kpm = torch.ones(B, Lb, dtype=torch.bool, device=dev)
            for j, x in enumerate(bufs):
                pad[j, :len(x)] = x; kpm[j, :len(x)] = False
            kpm[:, 0] = False                                                  # an empty buffer still attends to one slot
            sp = pooler(embed(pad).float(), kpm=kpm)                          # (B, S, H)
        x_in = sp.to(dt)
        if w > 1:                                                          # the raw window after SP_t: a[t : t+w-1]
            raw = torch.stack([a_ids[0, t:t + w - 1] for t in ts])
            x_in = torch.cat([x_in, embed(raw).to(dt)], 1)
        nb = a.S + w - 1
        cache = DynamicCache()
        for li, (k_, v_) in enumerate(pkv): cache.update(k_.expand(B, -1, -1, -1), v_.expand(B, -1, -1, -1), li)
        pp = torch.arange(P, P + nb, device=dev)[None].expand(B, -1)
        h = llm.model(inputs_embeds=x_in, attention_mask=torch.ones(B, P + nb, device=dev), past_key_values=cache,
                      position_ids=pp, cache_position=pp[0], use_cache=True).last_hidden_state[:, a.S - 1:]   # (B, w, H)
        rows = (torch.tensor(ts, device=dev) - t0)[:, None] + torch.arange(w, device=dev)[None]          # (B, w)
        if w > 256:   # wide windows: score window 0 + 255 random windows (B x w x vocab logits would not fit)
            sel = torch.tensor([0] + sorted(random.sample(range(1, w), 255)), device=dev)
            h, rows = h[:, sel], rows[:, sel]
        wk = rows.size(1)
        tr = teacher_rows(ref_h[rows.reshape(-1)]).view(B, wk, -1)
        lp = F.log_softmax(llm.lm_head(h).float(), -1)
        kl = (tr.exp() * (tr - lp)).sum(-1)                                # (B, w)
        (kl.sum() / 1000.0).backward()
        K += kl.sum().item(); A_ += (lp.argmax(-1) == tr.argmax(-1)).float().sum().item(); N += B * wk
        SNW["k0"] += kl[:, 0].sum().item(); SNW["n0"] += B
        for t, v in zip(ts, kl[:, 0].tolist()):   # window-0 KL by the length of the past (as --par logs it)
            bk = sum(t >= e for e in DB_EDGES); d_ = DBK.setdefault(bk, [0.0, 0]); d_[0] += v; d_[1] += 1
    return K, A_, N


@torch.no_grad()
def profile_trace(ids, P, acc):
    """KL by window length: one [prompt | SP_b | a[b : b+rw+C]] forward per boundary b (as --par training sees it)."""
    T = min(len(ids), P + a.max_len)
    ids = ids.to(dev); a_ids = ids[None, P:T]; La = a_ids.size(1)
    L = a.rw + a.C
    bounds = list(range(a.C, La - L + 1, a.C))
    if not bounds: return
    ref_h = teacher_hidden(ids, P + a.C, T)
    P0 = P - 1 if a.no_sp else P   # --no-sp: the block starts with the prompt's last token, whose row predicts a[b]
    pc = DynamicCache(); llm(input_ids=ids[None, :P0], past_key_values=pc, use_cache=True)
    pkv = [tuple(t for t in kv_of(pc, li)) for li in range(NL)]
    kept = torch.zeros(1, 0, dtype=torch.long, device=dev); absorbed = 0
    for b in bounds:
        kept, absorbed = update_buffer(kept, absorbed, a_ids, b)
        if a.no_sp:   # the same window, nothing for the past: prompt + the chunk so far
            block = torch.cat([embed(ids[None, P - 1:P]).to(dt), embed(a_ids[:, b:b + L - 1]).to(dt)], 1); h0 = 0
        else:
            sp = pooler(embed(kept).float()) if kept.size(1) > 0 else pooler(torch.zeros(1, 0, H, device=dev))
            block = torch.cat([sp.to(dt), embed(a_ids[:, b:b + L - 1]).to(dt)], 1); h0 = a.S - 1
        nb = block.size(1)
        cache = DynamicCache()
        for li, (k_, v_) in enumerate(pkv): cache.update(k_, v_, li)
        pos = torch.arange(P0, P0 + nb, device=dev)[None]
        hs = llm.model(inputs_embeds=block, attention_mask=torch.ones(1, P0 + nb, device=dev), past_key_values=cache,
                       position_ids=pos, cache_position=pos[0], use_cache=True).last_hidden_state[0, h0:]
        for i in range(0, L, 256):
            tr = teacher_rows(ref_h[b - a.C + i:b - a.C + min(i + 256, L)])
            lp = F.log_softmax(llm.lm_head(hs[i:i + 256]).float(), -1)
            kl = (tr.exp() * (tr - lp)).sum(-1)
            for j, v in enumerate(kl.tolist()):
                w = i + j
                key = "0" if w == 0 else "1-7" if w < 8 else "8-63" if w < 64 else "64-511" if w < 512 else "512+"
                acc.setdefault(key, [0.0, 0]); acc[key][0] += v; acc[key][1] += 1


@torch.no_grad()
def rollout(prompts):
    """the deployment SP-evict loop as a generator, N streams in one batch: every C tokens the distant tokens (older than the
    last rw) are pooled (mass-evicted at maxd) into SP, the cache is cut back to the prompt and [SP | raw window] re-prefilled,
    then C tokens are sampled. Returns the generated token lists."""
    N = len(prompts); mqs = [len(x) for x in prompts]; MQ = max(mqs)
    stop = {i for i in [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")] if i is not None and i >= 0}
    inp = torch.zeros((N, MQ), device=dev, dtype=torch.long); pmask = torch.zeros((N, MQ), device=dev, dtype=torch.long)
    ppos = torch.zeros((N, MQ), device=dev, dtype=torch.long)
    for b, x in enumerate(prompts):
        inp[b, MQ - len(x):] = torch.tensor(x, device=dev); pmask[b, MQ - len(x):] = 1; ppos[b, MQ - len(x):] = torch.arange(len(x), device=dev)
    cache = DynamicCache()
    plast = llm(input_ids=inp, attention_mask=pmask, position_ids=ppos, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1]
    gen = [[] for _ in range(N)]; fin = [False] * N
    kept = [torch.zeros(1, 0, dtype=torch.long, device=dev) for _ in range(N)]; absorbed = [0] * N

    seen = torch.zeros((N, llm.config.vocab_size), device=dev, dtype=torch.bool) if a.presence else None
    think = torch.ones(N, device=dev)   # 1 while the stream has not closed its thought (--end-bias applies only then)

    def sample(lg):
        lg = lg.float()
        if a.presence: lg = lg - a.presence * seen[:, :lg.size(-1)].float()
        if a.end_bias: lg[:, END_ID] += a.end_bias * think
        if a.hes_bias: lg[:, HES_IDS] += a.hes_bias * think[:, None]
        lg = lg / a.temp
        if a.top_k: lg = lg.masked_fill(lg < lg.topk(a.top_k, -1).values[:, -1:], float("-inf"))
        pr = torch.softmax(lg, -1); sp_, si = pr.sort(-1, descending=True)
        drop = sp_.cumsum(-1) - sp_ > a.top_p; sp_ = sp_.masked_fill(drop, 0)
        return si.gather(-1, torch.multinomial(sp_ / sp_.sum(-1, keepdim=True), 1)).squeeze(-1)

    while not all(fin):
        blocks, Rs = [], []
        for b in range(N):
            c0 = len(gen[b]); R = min(c0, a.rw); nd = c0 - R
            if nd > absorbed[b]:
                g_ = torch.tensor([gen[b]], device=dev)
                kept[b], absorbed[b] = update_buffer(kept[b], absorbed[b], g_, nd)
            if a.no_sp or (kept[b].size(1) == 0 and not a.sp_empty): parts = []   # nothing distant yet: the plain window (as trained)
            else: parts = [(pooler(embed(kept[b]).float()) if kept[b].size(1) > 0 else pooler(torch.zeros(1, 0, H, device=dev))).to(dt)]
            parts += [embed(torch.tensor([gen[b][c0 - R:c0]], device=dev)).to(dt)] if R > 0 else []
            if not parts: parts = [torch.zeros(1, 0, H, device=dev, dtype=dt)]   # no-SP, first chunk: nothing to re-prefill
            blocks.append(torch.cat(parts, 1)); Rs.append(R)
        lens = [x.size(1) for x in blocks]; Lmax = max(lens)
        block = torch.zeros((N, Lmax, H), device=dev, dtype=dt); amask = torch.zeros((N, MQ + Lmax), device=dev, dtype=torch.long)
        amask[:, :MQ] = pmask; pos = torch.zeros((N, Lmax), device=dev, dtype=torch.long)
        for b in range(N):   # right-pad the shorter blocks; their next token goes right after their own last real position
            block[b, :lens[b]] = blocks[b][0]; amask[b, MQ:MQ + lens[b]] = 1; pos[b] = mqs[b] + torch.arange(Lmax, device=dev)
        L0 = cache.get_seq_length()
        if L0 > MQ: cache.crop(MQ - L0) if hasattr(cache, "crop") else None
        if Lmax == 0:   # no-SP first chunk: continue straight from the prompt
            amask = pmask.clone(); last = plast; cur = MQ; nxtpos = list(mqs)
        else:
          hs = llm.model(inputs_embeds=block, attention_mask=amask, position_ids=pos, past_key_values=cache,
                       cache_position=torch.arange(MQ, MQ + Lmax, device=dev), use_cache=True).last_hidden_state
          last = llm.lm_head(hs[torch.arange(N, device=dev), torch.tensor(lens, device=dev) - 1])   # each stream's own last row
          cur = MQ + Lmax; nxtpos = [mqs[b] + lens[b] for b in range(N)]
        for _ in range(a.C):
            nx = sample(last).tolist()
            for b in range(N):
                if fin[b]: continue
                if nx[b] in stop: fin[b] = True; continue
                gen[b].append(nx[b])
                if a.presence: seen[b, nx[b]] = True
                if nx[b] == END_ID: think[b] = 0.0
                if len(gen[b]) >= a.max_len: fin[b] = True
            if all(fin): break
            amask = torch.cat([amask, torch.ones((N, 1), device=dev, dtype=torch.long)], 1)
            o = llm(input_ids=torch.tensor([[g[-1] if g else 0] for g in gen], device=dev), attention_mask=amask,
                    position_ids=torch.tensor([[x] for x in nxtpos], device=dev), past_key_values=cache,
                    cache_position=torch.tensor([cur], device=dev), use_cache=True)
            last = o.logits[:, -1]; cur += 1; nxtpos = [x + 1 for x in nxtpos]
    return gen


def sample_tok(lg):
    """the deployment sampling (temperature, top-k, top-p) of a (N, vocab) logit batch"""
    lg = lg.float() / a.temp
    if a.top_k: lg = lg.masked_fill(lg < lg.topk(a.top_k, -1).values[:, -1:], float("-inf"))
    pr = torch.softmax(lg, -1); sp_, si = pr.sort(-1, descending=True)
    sp_ = sp_.masked_fill(sp_.cumsum(-1) - sp_ > a.top_p, 0)
    return si.gather(-1, torch.multinomial(sp_ / sp_.sum(-1, keepdim=True), 1)).squeeze(-1)


RMS = {}   # rollout_cb --rmt: prompt index -> [memory (1, S, H), blocks written]
LPG = {}   # rollout_cb --roll-lp: prompt index -> log p of each sampled token under the generator's logits


@torch.no_grad()
def rollout_cb(prompts, nslot):
    """the deployment SP-evict loop (as rollout()) with continuous batching: up to nslot streams; at every chunk boundary
    each stream's [prompt | SP | raw window] is prefilled afresh (left-padded, positions from 0 per stream), so the slot
    of a finished stream takes the next prompt at once instead of idling until the longest stream ends.
    Returns (generated token lists, ended-by-stop-token flags), in the order of prompts."""
    stop = {i for i in [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")] if i is not None and i >= 0}
    todo = list(range(len(prompts)))[::-1]; act = []
    gen = [[] for _ in prompts]; ended = [False] * len(prompts); kept = [None] * len(prompts); absorbed = [0] * len(prompts)
    while todo or act:
        full = lambda j: len(prompts[j]) + a.S + a.rw + a.C + (a.rmt_block if a.rmt else 0)   # the stream's largest re-prefill + chunk
        while todo and len(act) < nslot and (not act or not a.grpo_tok_budget
                                             or (len(act) + 1) * max(full(x) for x in act + [todo[-1]]) <= a.grpo_tok_budget):
            j = todo.pop(); act.append(j); kept[j] = torch.zeros(1, 0, dtype=torch.long, device=dev)
        blocks = []
        for j in act:
            c0 = len(gen[j]); R = min(c0, a.rw); nd = c0 - R
            if a.rmt:   # [prompt | memory | raw since the last written block | chunk]
                blk = a.rmt_block; nbk = max(0, (c0 - a.rw) // blk); st_ = RMS.setdefault(j, [RMT.m0[None], 0])
                pt_ = torch.tensor(prompts[j], device=dev)
                while st_[1] < nbk:
                    k = st_[1]; st_[0] = rmt_write(st_[0], pt_, torch.tensor(gen[j][k * blk:(k + 1) * blk], device=dev)); st_[1] += 1
                parts = [embed(pt_[None]).to(dt)] + ([st_[0].to(dt)] if nbk > 0 and not a.no_sp else [])
                if c0 > nbk * blk: parts.append(embed(torch.tensor([gen[j][nbk * blk:c0]], device=dev)).to(dt))
                blocks.append(torch.cat(parts, 1)); continue
            if a.sp_replace:   # [prompt | SP(last rw) | chunk] once rw tokens exist, else the plain window
                parts = [embed(torch.tensor([prompts[j]], device=dev)).to(dt)]
                if c0 >= a.rw: parts.append(pooler(embed(torch.tensor([gen[j][c0 - a.rw:c0]], device=dev)).float()).to(dt))
                elif c0 > 0: parts.append(embed(torch.tensor([gen[j]], device=dev)).to(dt))
                blocks.append(torch.cat(parts, 1)); continue
            if nd > absorbed[j]: kept[j], absorbed[j] = update_buffer(kept[j], absorbed[j], torch.tensor([gen[j]], device=dev), nd)
            parts = [embed(torch.tensor([prompts[j] + pin_of(gen[j], c0 - R)], device=dev)).to(dt)]
            if kept[j].size(1) > 0 and not a.no_sp:
                if a.sp_null or a.sp_filler: parts.append(slot(kept[j]))
                elif a.sp_swap:
                    oth = [x for x in act if x != j and kept[x] is not None and kept[x].size(1) > 0]
                    src_ = kept[oth[0]] if oth else torch.zeros(1, 0, dtype=torch.long, device=dev)
                    parts.append((pooler(embed(src_).float()) if src_.size(1) else pooler(torch.zeros(1, 0, H, device=dev))).to(dt))
                else: parts.append(pooler(embed(kept[j]).float()).to(dt))
            if R > 0: parts.append(embed(torch.tensor([gen[j][c0 - R:]], device=dev)).to(dt))
            blocks.append(torch.cat(parts, 1))
        N = len(act); lens = [x.size(1) for x in blocks]; Lmax = max(lens)
        block = torch.zeros((N, Lmax, H), device=dev, dtype=dt); amask = torch.zeros((N, Lmax), device=dev, dtype=torch.long)
        pos = torch.zeros((N, Lmax), device=dev, dtype=torch.long)
        for b in range(N):
            block[b, Lmax - lens[b]:] = blocks[b][0]; amask[b, Lmax - lens[b]:] = 1; pos[b, Lmax - lens[b]:] = torch.arange(lens[b], device=dev)
        del blocks
        cache = DynamicCache()
        hs = llm.model(inputs_embeds=block, attention_mask=amask, position_ids=pos, past_key_values=cache, use_cache=True).last_hidden_state
        last = llm.lm_head(hs[:, -1]); del hs, block
        nxt = torch.tensor(lens, device=dev)[:, None]; fin = [False] * N
        for _ in range(a.C):
            nx_t = sample_tok(last); nx = nx_t.tolist()
            if a.roll_lp: lpn = F.log_softmax(last.float(), -1).gather(1, nx_t[:, None])[:, 0].tolist()
            for b, j in enumerate(act):
                if fin[b]: continue
                if nx[b] in stop: fin[b] = True; ended[j] = True; continue
                gen[j].append(nx[b])
                if a.roll_lp: LPG.setdefault(j, []).append(lpn[b])
                if len(gen[j]) >= a.max_len: fin[b] = True
            if all(fin): break
            amask = torch.cat([amask, torch.ones((N, 1), device=dev, dtype=torch.long)], 1)
            o = llm(input_ids=torch.tensor([[gen[j][-1] if gen[j] else 0] for j in act], device=dev), attention_mask=amask,
                    position_ids=nxt, past_key_values=cache, use_cache=True, logits_to_keep=1)
            last = o.logits[:, -1]; nxt = nxt + 1
        del cache
        act = [j for b, j in enumerate(act) if not fin[b]]
    return gen, ended


@torch.no_grad()
def rollout_kv(prompts, nslot):
    """--kv-evict generator: continuous batching over a left-padded KV cache. Every C tokens (and when streams join or
    leave) the cache is rebuilt from each stream's kept entries - its prompt plus its newest rw answer tokens, exactly as
    they were computed - so nothing is re-encoded; new tokens take their true positions (prompt length + index).
    Returns (generated token lists, ended-by-stop-token flags)."""
    stop = {i for i in [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")] if i is not None and i >= 0}
    todo = list(range(len(prompts)))[::-1]; act = []
    gen = [[] for _ in prompts]; ended = [False] * len(prompts)
    cache = None; pad = []; nans = []; lastlog = None   # per row: left padding, answer entries cached
    per = lambda j: len(prompts[j]) + min(a.rw, a.max_len) + a.C
    while todo or act:
        new = []
        while todo and len(act) + len(new) < nslot and (len(act) + len(new) + 1) * max(per(x) for x in act + new + [todo[-1]]) <= a.kv_budget:
            new.append(todo.pop())
        newkv, newlog = {}, {}
        for j in new:   # prompt prefill, alone (no padding)
            pc = DynamicCache(); o = llm(input_ids=torch.tensor([prompts[j]], device=dev), past_key_values=pc, use_cache=True, logits_to_keep=1)
            newkv[j] = [kv_of(pc, li) for li in range(NL)]; newlog[j] = o.logits[0, -1].float()
        # what each stream keeps: (source, start, length) - continuing streams from the old batch, new ones from their prefill
        keep = []
        for b, j in enumerate(act):
            P = len(prompts[j]); n = nans[b]; k = min(a.rw, n)
            keep.append((j, b, pad[b], P, n - k, k))           # old row b: prompt [pad, pad+P), answer entries [pad+P, pad+P+n) = the newest n tokens
        for j in new: keep.append((j, None, 0, len(prompts[j]), 0, 0))
        Ln = max(P + k for (_, _, _, P, _, k) in keep)
        B = len(keep); pad = [Ln - (P + k) for (_, _, _, P, _, k) in keep]
        nc = DynamicCache()
        for li in range(NL):
            ref = kv_of(cache, li)[0] if cache is not None else newkv[new[0]][li][0]
            K = torch.zeros((B, ref.size(1), Ln, ref.size(3)), device=dev, dtype=ref.dtype); V = torch.zeros_like(K)
            ok, ov = kv_of(cache, li) if cache is not None else (None, None)
            for r, (j, b, p0, P, s0, k) in enumerate(keep):
                if b is None: K[r, :, pad[r]:], V[r, :, pad[r]:] = newkv[j][li][0][0], newkv[j][li][1][0]
                else:
                    K[r, :, pad[r]:pad[r] + P], V[r, :, pad[r]:pad[r] + P] = ok[b, :, p0:p0 + P], ov[b, :, p0:p0 + P]
                    if k: K[r, :, pad[r] + P:], V[r, :, pad[r] + P:] = ok[b, :, p0 + P + s0:p0 + P + s0 + k], ov[b, :, p0 + P + s0:p0 + P + s0 + k]
            nc.update(K, V, li)
        cache = nc; del newkv
        lastlog = torch.stack([lastlog[b] if b is not None else newlog[j] for (j, b, _, _, _, _) in keep])
        act = [j for (j, _, _, _, _, _) in keep]; nans = [k for (_, _, _, _, _, k) in keep]
        amask = torch.zeros((B, Ln), device=dev, dtype=torch.long)
        for r in range(B): amask[r, pad[r]:] = 1
        fin = [False] * B
        for _ in range(a.C):
            nx_t = sample_tok(lastlog); nx = nx_t.tolist()
            if a.roll_lp: lpn = F.log_softmax(lastlog, -1).gather(1, nx_t[:, None])[:, 0].tolist()
            for r, j in enumerate(act):
                if fin[r]: continue
                if nx[r] in stop: fin[r] = True; ended[j] = True; continue
                gen[j].append(nx[r])
                if a.roll_lp: LPG.setdefault(j, []).append(lpn[r])
                if len(gen[j]) >= a.max_len: fin[r] = True
            if all(fin): break
            amask = torch.cat([amask, torch.ones((B, 1), device=dev, dtype=torch.long)], 1)
            pos = torch.tensor([[len(prompts[j]) + max(len(gen[j]) - 1, 0)] for j in act], device=dev)
            o = llm(input_ids=torch.tensor([[gen[j][-1] if gen[j] else 0] for j in act], device=dev), attention_mask=amask,
                    position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1)
            lastlog = o.logits[:, -1].float(); nans = [x + 1 for x in nans]   # every row got one entry (finished rows: dummies)
        # entries of a stream that finished mid-chunk are fed dummies afterwards; such streams leave now
        keep_rows = [r for r in range(B) if not fin[r]]
        if len(keep_rows) < B:
            sel = torch.tensor(keep_rows, device=dev, dtype=torch.long)
            if keep_rows:
                c2 = DynamicCache()
                for li in range(NL):
                    k_, v_ = kv_of(cache, li); c2.update(k_.index_select(0, sel), v_.index_select(0, sel), li)
                cache = c2; lastlog = lastlog.index_select(0, sel)
            else: cache = None; lastlog = None
            act = [act[r] for r in keep_rows]; pad = [pad[r] for r in keep_rows]; nans = [nans[r] for r in keep_rows]
    return gen, ended


@torch.no_grad()
def rollout_kvkeep(prompts, nslot):
    """--kv-evict --kv-keep B (the user: a fixed KV budget of real entries, no synthesis): rollout_kv, but every C tokens each
    stream keeps its prompt, its newest rw answer entries AND the B older answer entries the model attended most - the attention
    each cached key received from the generated tokens (layers ATT_L, all heads, the forward hook), divided by the steps it
    was cached - all as computed, at their true positions. Returns (generated token lists, ended flags)."""
    stop = {i for i in [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")] if i is not None and i >= 0}
    todo = list(range(len(prompts)))[::-1]; act = []
    gen = [[] for _ in prompts]; ended = [False] * len(prompts)
    cache = None; pad = []; cols = []; lastlog = None; IMP = CNT = None   # per row: left padding, the answer index of each cached answer entry; (B, Ln) attention received / steps observed
    ST = None   # (6, B, Ln) chunk-level statistics per cached column: cumulative, last chunk, EMA 0.9, EMA 0.98, max over chunks, chunks observed; CH: this chunk's sum
    CH = None; sel_model = kvsel_model(a.kv_select[4:]) if a.kv_select.startswith("mlp:") else None
    if sel_model is not None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from kvsel_feat import numeric as _numeric
    per = lambda j: len(prompts[j]) + a.kv_keep + min(a.rw, a.max_len) + a.C
    KEEPLOG = [0, 0, 0]   # kept older entries: how many, their mean age in chunks, rebuilds
    while todo or act:
        new = []
        while todo and len(act) + len(new) < nslot and (len(act) + len(new) + 1) * max(per(x) for x in act + new + [todo[-1]]) <= a.kv_budget:
            new.append(todo.pop())
        newkv, newlog = {}, {}
        for j in new:
            pc = DynamicCache(); o = llm(input_ids=torch.tensor([prompts[j]], device=dev), past_key_values=pc, use_cache=True, logits_to_keep=1)
            newkv[j] = [kv_of(pc, li) for li in range(NL)]; newlog[j] = o.logits[0, -1].float()
        keep = []   # (j, old row, old pad, P, the old row's answer-entry indices to keep)
        for b, j in enumerate(act):
            P = len(prompts[j]); n = len(cols[b]); k = min(a.rw, n); older = list(range(n - k)); newest = list(range(n - k, n))
            if a.kv_keep <= 0: older = []
            elif len(older) > a.kv_keep:
                base = pad[b] + P; sl_ = slice(base, base + n - k)
                if sel_model is not None:
                    nob = ST[5, b, sl_].clamp(min=1)
                    x6 = torch.stack([ST[0, b, sl_], ST[0, b, sl_] / nob, ST[1, b, sl_], ST[2, b, sl_], ST[3, b, sl_], ST[4, b, sl_]], 1)
                    vn = torch.stack([kv_of(cache, li)[1][b, :, sl_].norm(dim=-1).mean(0) for li in range(NL)], 1)
                    kn = torch.stack([kv_of(cache, li)[0][b, :, sl_].norm(dim=-1).mean(0) for li in range(NL)], 1)
                    e_ = torch.tensor(cols[b][:n - k], device=dev); age = (len(gen[j]) - e_).float() / a.C
                    score = sel_model(_numeric(x6, nob, age, e_.float() / max(len(gen[j]), 1), vn, kn), torch.tensor([gen[j][i] for i in cols[b][:n - k]], device=dev))
                elif a.kv_select == "cum": score = ST[0, b, sl_]
                elif a.kv_select == "ema0.9": score = ST[2, b, sl_]
                elif a.kv_select == "ema0.98": score = ST[3, b, sl_]
                elif a.kv_select == "last": score = ST[1, b, sl_]
                else: score = IMP[b, sl_] / CNT[b, sl_].clamp(min=1)
                older = score.topk(a.kv_keep).indices.sort().values.tolist()
            if older: KEEPLOG[0] += len(older); KEEPLOG[1] += sum(len(gen[j]) - cols[b][i] for i in older) / len(older) / a.C; KEEPLOG[2] += 1
            keep.append((j, b, pad[b], P, older + newest))
        for j in new: keep.append((j, None, 0, len(prompts[j]), []))
        Ln = max(P + len(idx) for (_, _, _, P, idx) in keep); B = len(keep); pad = [Ln - (P + len(idx)) for (_, _, _, P, idx) in keep]
        nc = DynamicCache(); IMPn = torch.zeros(B, Ln, device=dev); CNTn = torch.zeros(B, Ln, device=dev); STn = torch.zeros(6, B, Ln, device=dev)
        srcs = [torch.tensor(list(range(p0, p0 + P)) + [p0 + P + i for i in idx], device=dev, dtype=torch.long) if b is not None else None for (j, b, p0, P, idx) in keep]
        for li in range(NL):
            ref = kv_of(cache, li)[0] if cache is not None else newkv[new[0]][li][0]
            K = torch.zeros((B, ref.size(1), Ln, ref.size(3)), device=dev, dtype=ref.dtype); V = torch.zeros_like(K)
            ok, ov = kv_of(cache, li) if cache is not None else (None, None)
            for r, (j, b, p0, P, idx) in enumerate(keep):
                if b is None: K[r, :, pad[r]:], V[r, :, pad[r]:] = newkv[j][li][0][0], newkv[j][li][1][0]
                else:
                    K[r, :, pad[r]:] = ok[b].index_select(1, srcs[r]); V[r, :, pad[r]:] = ov[b].index_select(1, srcs[r])
                    if li == 0: IMPn[r, pad[r]:] = IMP[b, srcs[r]]; CNTn[r, pad[r]:] = CNT[b, srcs[r]]; STn[:, r, pad[r]:] = ST[:, b, srcs[r]]
            nc.update(K, V, li)
            if cache is not None:   # the old layer is not needed any more: free it (the rebuild otherwise holds two full caches)
                if hasattr(cache, "layers"): cache.layers[li].keys = cache.layers[li].values = None
                else: cache.key_cache[li] = cache.value_cache[li] = None
            del ok, ov
        cache = nc; IMP, CNT, ST = IMPn, CNTn, STn; CH = torch.zeros(B, Ln, device=dev); del newkv
        lastlog = torch.stack([lastlog[b] if b is not None else newlog[j] for (j, b, _, _, _) in keep])
        cols = [[cols[b][i] for i in idx] if b is not None else [] for (j, b, _, _, idx) in keep]
        act = [j for (j, _, _, _, _) in keep]
        amask = torch.zeros((B, Ln), device=dev, dtype=torch.long)
        for r in range(B): amask[r, pad[r]:] = 1
        fin = [False] * B
        for _ in range(a.C):
            nx_t = sample_tok(lastlog); nx = nx_t.tolist()
            if a.roll_lp: lpn = F.log_softmax(lastlog, -1).gather(1, nx_t[:, None])[:, 0].tolist()
            for r, j in enumerate(act):
                if fin[r]: continue
                if nx[r] in stop: fin[r] = True; ended[j] = True; continue
                gen[j].append(nx[r])
                if a.roll_lp: LPG.setdefault(j, []).append(lpn[r])
                if len(gen[j]) >= a.max_len: fin[r] = True
            if all(fin): break
            amask = torch.cat([amask, torch.ones((B, 1), device=dev, dtype=torch.long)], 1)
            pos = torch.tensor([[len(prompts[j]) + max(len(gen[j]) - 1, 0)] for j in act], device=dev)
            ATTREC["rows"] = 1
            o = llm(input_ids=torch.tensor([[gen[j][-1] if gen[j] else 0] for j in act], device=dev), attention_mask=amask,
                    position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1)
            lastlog = o.logits[:, -1].float()
            w = att_take()
            IMP = torch.cat([IMP, torch.zeros(B, 1, device=dev)], 1); CNT = torch.cat([CNT, torch.zeros(B, 1, device=dev)], 1)
            CH = torch.cat([CH, torch.zeros(B, 1, device=dev)], 1); ST = torch.cat([ST, torch.zeros(6, B, 1, device=dev)], 2)
            if w is not None:
                if w.dim() == 1: w = w[None]
                live = torch.tensor([0.0 if f else 1.0 for f in fin], device=dev)[:, None]
                IMP += w * live; CNT += amask.float() * live; CH += w * live
            for r, j in enumerate(act): cols[r].append(len(gen[j]) - 1)   # the new entry (a finished row's dummy leaves with it)
        ST[0] += CH; ST[1] = CH; ST[2] = 0.9 * ST[2] + CH; ST[3] = 0.98 * ST[3] + CH; ST[4] = torch.maximum(ST[4], CH); ST[5] += amask.float()
        keep_rows = [r for r in range(B) if not fin[r]]
        if len(keep_rows) < B:
            sel = torch.tensor(keep_rows, device=dev, dtype=torch.long)
            if keep_rows:
                c2 = DynamicCache()
                for li in range(NL):
                    k_, v_ = kv_of(cache, li); c2.update(k_.index_select(0, sel), v_.index_select(0, sel), li)
                cache = c2; lastlog = lastlog.index_select(0, sel); IMP = IMP.index_select(0, sel); CNT = CNT.index_select(0, sel); ST = ST.index_select(1, sel)
            else: cache = None; lastlog = None; IMP = CNT = ST = None
            act = [act[r] for r in keep_rows]; pad = [pad[r] for r in keep_rows]; cols = [cols[r] for r in keep_rows]
    if KEEPLOG[2]: log(f"[sp] kv-keep {a.kv_keep}: {KEEPLOG[2]} rebuilds kept {KEEPLOG[0] / KEEPLOG[2]:.0f} older entries each, mean age {KEEPLOG[1] / KEEPLOG[2]:.1f} chunks")
    return gen, ended


@torch.no_grad()
def rollout_kvlayer(prompts, nslot):
    """--kv-evict --kv-keep B --kv-layerwise: rollout_kvkeep with a per-layer choice - every hooked layer ranks the older entries by
    its OWN attention statistic and keeps its own B; the other layers keep the set of the summed statistic. All layers keep the same
    number of entries per stream, so the cache stays rectangular and one mask serves every layer. Groups: 'sum' and each hooked
    layer; per group its own column -> answer-index map and chunk statistics (cumulative, last chunk, EMA 0.9, EMA 0.98, max, chunks)."""
    stop = {i for i in [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")] if i is not None and i >= 0}
    groups = ["sum"] + list(ATT_L); grp = lambda li: li if li in ATT_L else "sum"
    todo = list(range(len(prompts)))[::-1]; act = []
    gen = [[] for _ in prompts]; ended = [False] * len(prompts)
    cache = None; pad = []; lastlog = None
    cols = {g: [] for g in groups}; ST = {g: None for g in groups}; CH = {g: None for g in groups}
    per = lambda j: len(prompts[j]) + a.kv_keep + min(a.rw, a.max_len) + a.C
    sidx = {"mean": None, "cum": 0, "last": 1, "ema0.9": 2, "ema0.98": 3}[a.kv_select if a.kv_select in ("mean", "cum", "last", "ema0.9", "ema0.98") else "mean"]
    while todo or act:
        new = []
        while todo and len(act) + len(new) < nslot and (len(act) + len(new) + 1) * max(per(x) for x in act + new + [todo[-1]]) <= a.kv_budget:
            new.append(todo.pop())
        newkv, newlog = {}, {}
        for j in new:
            pc = DynamicCache(); o = llm(input_ids=torch.tensor([prompts[j]], device=dev), past_key_values=pc, use_cache=True, logits_to_keep=1)
            newkv[j] = [kv_of(pc, li) for li in range(NL)]; newlog[j] = o.logits[0, -1].float()
        keep = []   # (j, old row, old pad, P, {group: indices of the old row's answer columns to keep})
        for b, j in enumerate(act):
            P = len(prompts[j]); n = len(cols["sum"][b]); k = min(a.rw, n); sel = {}
            for g in groups:
                older = list(range(n - k)); newest = list(range(n - k, n))
                if a.kv_keep <= 0: older = []
                elif len(older) > a.kv_keep:
                    base = pad[b] + P; sl_ = slice(base, base + n - k); st = ST[g]
                    score = st[0, b, sl_] / st[5, b, sl_].clamp(min=1) if sidx is None else st[sidx, b, sl_]
                    older = score.topk(a.kv_keep).indices.sort().values.tolist()
                sel[g] = older + newest
            keep.append((j, b, pad[b], P, sel))
        for j in new: keep.append((j, None, 0, len(prompts[j]), {g: [] for g in groups}))
        Ln = max(P + len(sel["sum"]) for (_, _, _, P, sel) in keep); B = len(keep); pad = [Ln - (P + len(sel["sum"])) for (_, _, _, P, sel) in keep]
        nc = DynamicCache(); STn = {g: torch.zeros(6, B, Ln, device=dev) for g in groups}
        srcs = {g: [torch.tensor(list(range(p0, p0 + P)) + [p0 + P + i for i in sel[g]], device=dev, dtype=torch.long) if b is not None else None for (j, b, p0, P, sel) in keep] for g in groups}
        for li in range(NL):
            g = grp(li); ref = kv_of(cache, li)[0] if cache is not None else newkv[new[0]][li][0]
            K = torch.zeros((B, ref.size(1), Ln, ref.size(3)), device=dev, dtype=ref.dtype); V = torch.zeros_like(K)
            ok, ov = kv_of(cache, li) if cache is not None else (None, None)
            for r, (j, b, p0, P, sel) in enumerate(keep):
                if b is None: K[r, :, pad[r]:], V[r, :, pad[r]:] = newkv[j][li][0][0], newkv[j][li][1][0]
                else: K[r, :, pad[r]:] = ok[b].index_select(1, srcs[g][r]); V[r, :, pad[r]:] = ov[b].index_select(1, srcs[g][r])
            nc.update(K, V, li)
            if cache is not None:
                if hasattr(cache, "layers"): cache.layers[li].keys = cache.layers[li].values = None
                else: cache.key_cache[li] = cache.value_cache[li] = None
            del ok, ov
        for g in groups:
            for r, (j, b, p0, P, sel) in enumerate(keep):
                if b is not None: STn[g][:, r, pad[r]:] = ST[g][:, b, srcs[g][r]]
            cols[g] = [[cols[g][b][i] for i in sel[g]] if b is not None else [] for (j, b, _, _, sel) in keep]
            CH[g] = torch.zeros(B, Ln, device=dev)
        cache = nc; ST = STn; del newkv
        lastlog = torch.stack([lastlog[b] if b is not None else newlog[j] for (j, b, _, _, _) in keep])
        act = [j for (j, _, _, _, _) in keep]
        amask = torch.zeros((B, Ln), device=dev, dtype=torch.long)
        for r in range(B): amask[r, pad[r]:] = 1
        fin = [False] * B
        for _ in range(a.C):
            nx_t = sample_tok(lastlog); nx = nx_t.tolist()
            if a.roll_lp: lpn = F.log_softmax(lastlog, -1).gather(1, nx_t[:, None])[:, 0].tolist()
            for r, j in enumerate(act):
                if fin[r]: continue
                if nx[r] in stop: fin[r] = True; ended[j] = True; continue
                gen[j].append(nx[r])
                if a.roll_lp: LPG.setdefault(j, []).append(lpn[r])
                if len(gen[j]) >= a.max_len: fin[r] = True
            if all(fin): break
            amask = torch.cat([amask, torch.ones((B, 1), device=dev, dtype=torch.long)], 1)
            pos = torch.tensor([[len(prompts[j]) + max(len(gen[j]) - 1, 0)] for j in act], device=dev)
            ATTREC["rows"] = 1
            o = llm(input_ids=torch.tensor([[gen[j][-1] if gen[j] else 0] for j in act], device=dev), attention_mask=amask,
                    position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1)
            lastlog = o.logits[:, -1].float()
            acc = ATTREC["acc"]; ATTREC["acc"] = {}; ATTREC["rows"] = 0
            live = torch.tensor([0.0 if f else 1.0 for f in fin], device=dev)[:, None]
            for g in groups:
                CH[g] = torch.cat([CH[g], torch.zeros(B, 1, device=dev)], 1); ST[g] = torch.cat([ST[g], torch.zeros(6, B, 1, device=dev)], 2)
                w = sum(acc.values()) if g == "sum" else acc.get(g)
                if w is not None: CH[g] += w * live
                for r, j in enumerate(act): cols[g][r].append(len(gen[j]) - 1)
        for g in groups:
            st = ST[g]; st[0] += CH[g]; st[1] = CH[g]; st[2] = 0.9 * st[2] + CH[g]; st[3] = 0.98 * st[3] + CH[g]; st[4] = torch.maximum(st[4], CH[g]); st[5] += amask.float()
        keep_rows = [r for r in range(B) if not fin[r]]
        if len(keep_rows) < B:
            sel_ = torch.tensor(keep_rows, device=dev, dtype=torch.long)
            if keep_rows:
                c2 = DynamicCache()
                for li in range(NL):
                    k_, v_ = kv_of(cache, li); c2.update(k_.index_select(0, sel_), v_.index_select(0, sel_), li)
                cache = c2; lastlog = lastlog.index_select(0, sel_); ST = {g: ST[g].index_select(1, sel_) for g in groups}
            else: cache = None; lastlog = None; ST = {g: None for g in groups}
            act = [act[r] for r in keep_rows]; pad = [pad[r] for r in keep_rows]; cols = {g: [cols[g][r] for r in keep_rows] for g in groups}
    return gen, ended


PARA_END_L = None
PWN = [0, 0]   # --para-window: openings sampled from the masked twin, all sampled steps


RT = {"prompt": 0.0, "memory": 0.0, "window": 0.0, "decode": 0.0, "chunks": 0, "streams": 0, "wtok": 0, "ptok": 0}   # where rollout_kvmem's time goes


def _now():
    if dev == "cuda": torch.cuda.synchronize()
    return time.time()


@torch.no_grad()
def rollout_kvmem(prompts, nslot):
    """--kv-mem generation: continuous batching as rollout_cb, but every chunk the cache is [prompt KV (left-padded) | the
    stream's S memory slots (its pooled distant past; masked out while nothing is distant) | its raw window (left-padded)],
    each part at the stream's own true positions, then C tokens are sampled. Returns (generated token lists, ended flags)."""
    stop = {i for i in [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")] if i is not None and i >= 0}
    global PARA_END_L
    TW = 2 if (a.para_window or a.twin_mix >= 0 or a.twin_minent) else 1; dup = (lambda x: torch.cat([x] * TW, 0)) if TW > 1 else (lambda x: x)
    if TW > 1 and PARA_END_L is None: PARA_END_L = [tok.decode([i]).endswith("\n\n") for i in range(len(tok))]
    todo = list(range(len(prompts)))[::-1]; act = []
    gen = [[] for _ in prompts]; ended = [False] * len(prompts); kept = [None] * len(prompts); absorbed = [0] * len(prompts)
    use_att = a.att_prior > 0 or a.evict_by == "model"   # the model's attention on each generated token, while it sits in the window
    if use_att:
        impT = [torch.zeros(0, device=dev) for _ in prompts]; kimp = [None] * len(prompts)
        assert TW == 1, "--att-prior with the masked twin is not implemented"
    while todo or act:
        full = lambda j: len(prompts[j]) + a.S + a.rw + a.C
        while todo and len(act) < nslot and (not act or not a.grpo_tok_budget
                                             or TW * (len(act) + 1) * max(full(x) for x in act + [todo[-1]]) <= a.grpo_tok_budget):
            j = todo.pop(); act.append(j); kept[j] = torch.zeros(1, 0, dtype=torch.long, device=dev)
            if use_att: kimp[j] = torch.zeros(1, 0, device=dev)
        N = len(act); Ps = [len(prompts[j]) for j in act]; Pm = max(Ps)
        # 1) the prompts, left-padded
        pin = torch.zeros((N, Pm), dtype=torch.long, device=dev); pm = torch.zeros((N, Pm), dtype=torch.long, device=dev)
        ppos = torch.zeros((N, Pm), dtype=torch.long, device=dev)
        for b, j in enumerate(act):
            pin[b, Pm - Ps[b]:] = torch.tensor(prompts[j], device=dev); pm[b, Pm - Ps[b]:] = 1; ppos[b, Pm - Ps[b]:] = torch.arange(Ps[b], device=dev)
        pin, pm, ppos = dup(pin), dup(pm), dup(ppos)
        t_ = _now(); RT["chunks"] += 1; RT["streams"] += N; RT["ptok"] += N * Pm
        cache = DynamicCache()
        plast = llm(input_ids=pin, attention_mask=pm, position_ids=ppos, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1]
        t1_ = _now(); RT["prompt"] += t1_ - t_; t_ = t1_
        # 2) the memory slots
        Rs, has = [], []
        for b, j in enumerate(act):
            c0 = len(gen[j]); R = min(c0, a.rw); nd = c0 - R
            if nd > absorbed[j]:
                if use_att:
                    si = impT[j][None].clone(); si[:, :2] = 0   # the first generated tokens (<think>, a newline) are the model's own sinks
                    kept[j], absorbed[j], kimp[j] = update_buffer(kept[j], absorbed[j], torch.tensor([gen[j]], device=dev), nd, imp=kimp[j], src_imp=si)
                else: kept[j], absorbed[j] = update_buffer(kept[j], absorbed[j], torch.tensor([gen[j]], device=dev), nd)
            Rs.append(R); has.append(kept[j].size(1) > 0 and not a.no_sp)
        mm = torch.tensor([[1 if h else 0] * a.S for h in has] + [[0] * a.S for _ in has] * (TW - 1), dtype=torch.long, device=dev)
        if any(has):
            sp = torch.cat([pooler(embed(kept[j]).float(), kpm=att_bias(kimp[j]) if use_att else None) if h else torch.zeros(1, a.S, H, device=dev) for j, h in zip(act, has)], 0)
            if use_att and not ATTLOG[0]:   # once: what the model's attention importance looks like on the first distant buffer
                j0 = next(j for j, h in zip(act, has) if h); sh = kimp[j0][0].clamp(min=0); sh = sh / sh.sum().clamp(min=1e-9)
                ATTLOG[0] = True; log(f"[sp] att-prior {a.att_prior}: first distant buffer {sh.numel()} tokens, importance share of the top 20 {sh.topk(min(20, sh.numel())).values.sum().item():.2f}, "
                                      f"zero-importance tokens {(sh == 0).sum().item()}, bias range {att_bias(kimp[j0]).min().item():+.2f}..{att_bias(kimp[j0]).max().item():+.2f}")
            mem = kv_memory(sp, torch.tensor(Ps, device=dev)[:, None] + torch.arange(a.S, device=dev)[None])
            mem = [(dup(k_), dup(v_)) for k_, v_ in mem]
            nc = DynamicCache()
            for li in range(NL):
                k_, v_ = kv_of(cache, li); nc.update(torch.cat([k_, mem[li][0]], 2), torch.cat([v_, mem[li][1]], 2), li)
        else:
            nc = DynamicCache()
            for li in range(NL):
                k_, v_ = kv_of(cache, li); z = torch.zeros(TW * N, k_.size(1), a.S, k_.size(3), device=dev, dtype=k_.dtype)
                nc.update(torch.cat([k_, z], 2), torch.cat([v_, z], 2), li)
        cache = nc
        t1_ = _now(); RT["memory"] += t1_ - t_; t_ = t1_
        # 3) the raw windows, left-padded, at each stream's positions after its prompt (+ S when it has a memory)
        Wm = max(Rs); amask = torch.cat([pm, mm], 1)
        if Wm > 0:
            win = torch.zeros((N, Wm), dtype=torch.long, device=dev); wm = torch.zeros((N, Wm), dtype=torch.long, device=dev)
            wpos = torch.zeros((N, Wm), dtype=torch.long, device=dev)
            for b, j in enumerate(act):
                R = Rs[b]
                if R:
                    st = Ps[b] + (a.S if has[b] else 0)
                    win[b, Wm - R:] = torch.tensor(gen[j][len(gen[j]) - R:], device=dev); wm[b, Wm - R:] = 1
                    wpos[b, Wm - R:] = torch.arange(st, st + R, device=dev)
            win, wm, wpos = dup(win), dup(wm), dup(wpos)
            amask = torch.cat([amask, wm], 1)
            o = llm(input_ids=win, attention_mask=amask, position_ids=wpos, past_key_values=cache, use_cache=True, logits_to_keep=1)
            last = torch.where(dup(torch.tensor([R > 0 for R in Rs], device=dev))[:, None], o.logits[:, -1], plast)
        else:
            last = plast
        t1_ = _now(); RT["window"] += t1_ - t_; t_ = t1_; RT["wtok"] += N * Wm
        nxt = dup(torch.tensor([[Ps[b] + (a.S if has[b] else 0) + Rs[b]] for b in range(N)], device=dev)); fin = [False] * N
        if use_att: c0s = [len(gen[j]) for j in act]; W0 = Pm + a.S + Wm   # key index where this chunk's new tokens start
        for _ in range(a.C):
            if TW > 1:   # the memory-masked twin: paragraph openings / a mix / the lower-entropy side
                lm_, lw_ = last[:N].float(), last[N:].float()
                if a.twin_mix >= 0: sel = (1 - a.twin_mix) * lw_ + a.twin_mix * lm_
                elif a.twin_minent:
                    em = -(F.softmax(lm_, -1) * F.log_softmax(lm_, -1)).sum(-1); ew = -(F.softmax(lw_, -1) * F.log_softmax(lw_, -1)).sum(-1)
                    sel = torch.where((ew < em)[:, None], lw_, lm_); PWN[0] += int((ew < em).sum()); PWN[1] += N
                else: sel = lm_
                if a.para_window:
                    pw = torch.tensor([bool(gen[j]) and bool(PARA_END_L[gen[j][-1]]) for j in act], device=dev)
                    PWN[0] += int(pw.sum()); PWN[1] += N
                    sel = torch.where(pw[:, None], lw_, sel)
                last = sel
            nx_t = sample_tok(last); nx = nx_t.tolist()
            if a.roll_lp: lpn = F.log_softmax(last.float(), -1).gather(1, nx_t[:, None])[:, 0].tolist()
            for b, j in enumerate(act):
                if fin[b]: continue
                if nx[b] in stop: fin[b] = True; ended[j] = True; continue
                gen[j].append(nx[b])
                if use_att: impT[j] = torch.cat([impT[j], torch.zeros(1, device=dev)])
                if a.roll_lp: LPG.setdefault(j, []).append(lpn[b])
                if len(gen[j]) >= a.max_len: fin[b] = True
            if all(fin): break
            amask = torch.cat([amask, torch.ones((TW * N, 1), device=dev, dtype=torch.long)], 1)
            if use_att: llm.set_attn_implementation("eager")
            o = llm(input_ids=dup(torch.tensor([[gen[j][-1] if gen[j] else 0] for j in act], device=dev)), attention_mask=amask,
                    position_ids=nxt, past_key_values=cache, use_cache=True, logits_to_keep=1, output_attentions=use_att)
            if use_att:
                llm.set_attn_implementation("sdpa")
                sc_ = sum(o.attentions[l_][:, :, 0, :].float().sum(1) for l_ in ATT_L)   # (N, keys): the attention the new token paid to each key
                for b, j in enumerate(act):
                    if fin[b]: continue
                    R = Rs[b]; g0 = c0s[b] - R; na = len(gen[j]) - c0s[b]
                    if R: impT[j][g0:g0 + R] += sc_[b, W0 - R:W0]
                    if na: impT[j][c0s[b]:c0s[b] + na] += sc_[b, W0:W0 + na]
            last = o.logits[:, -1]; nxt = nxt + 1
        del cache; RT["decode"] += _now() - t_
        act = [j for b, j in enumerate(act) if not fin[b]]
    return gen, ended


def run_grpo(opt, FTP, MASTER):
    """GRPO under the deployment loop (Dr. GRPO form): per update, --grpo-q problems x --group samples from rollout_cb (the
    deployment sampling); reward 1 for a correct final answer (sp_mathcheck: \\boxed after </think> vs gold, letters for
    multiple choice), else 0 - an answer that never closes its thought scores 0 too. Advantage = reward - the group mean
    (no std division); loss = -sum A x (summed token log-probs at the sampling temperature) / (group x q x lnorm), one
    on-policy step per batch (no ratio clipping needed). Groups with one reward for all carry no signal and are skipped.
    While generating, the fp32 masters and the optimizer state wait on the CPU so more streams fit."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sp_mathcheck import correct
    stops = {i for i in [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")] if i is not None and i >= 0}
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    base_lr = [g_["lr"] for g_ in opt.param_groups]

    def offload(cpu):
        tg = "cpu" if cpu else dev
        for m_ in MASTER: m_.data = m_.data.to(tg)
        for st_ in opt.state.values():
            for k_, v_ in list(st_.items()):
                if torch.is_tensor(v_): st_[k_] = v_.to(tg)
        if dev == "cuda": torch.cuda.empty_cache()

    def save_to(path):
        ck = {"pooler": {k: v.detach().cpu() for k, v in pooler.state_dict().items()}, "args": vars(a)}
        if a.ft_layers: ck["ft"] = {k: v.detach().cpu() for k, v in llm.state_dict().items() if k in FT_SD}
        torch.save(ck, path)

    def hook(*files):
        if a.grpo_hook: subprocess.Popen(a.grpo_hook + " " + " ".join(files), shell=True)

    EV = [json.loads(l) for l in open(a.grpo_eval)] if a.grpo_eval else []
    GUIDE = {}
    for l in (open(a.grpo_guide) if a.grpo_guide else []):
        r = json.loads(l)
        if "</think>" in r["text"] and correct(r): GUIDE.setdefault(r["id"], r["text"])
    if a.grpo_guide: log(f"[grpo] guides: {len(GUIDE)} right-and-closed trajectories from {a.grpo_guide}")
    NG = [0, 0]   # groups that had a guide, groups informative only because of it
    KLD = []
    for l in (open(a.grpo_kl) if a.grpo_kl else []):
        r = json.loads(l); pid = tok.encode(r["prompt"], add_special_tokens=False)
        KLD.append((torch.tensor(pid + tok.encode(r["text"], add_special_tokens=False)), len(pid)))

    def kl_check():
        ml = a.max_len; a.max_len = 4096; K = N = 0.0; t0 = time.time()
        with torch.no_grad():
            for ids, P in KLD: k, _, n = run_trace(ids, P, False); K += k; N += n
        a.max_len = ml
        return K / max(N, 1), time.time() - t0
    stem = a.out[:-3] if a.out.endswith(".pt") else a.out

    def evaluate(u):
        tot = cl = 0; files = []
        offload(True); pooler.eval()
        for sd in range(1, a.grpo_eval_seeds + 1):
            torch.manual_seed(1000 * sd); t0 = time.time()
            gens, _ = gen_safe([tok.encode(r["prompt"], add_special_tokens=False) for r in EV])
            rs = [{"src": r.get("src", ""), "id": r.get("id", ""), "gold": r.get("gold", ""), "prompt": r["prompt"],
                   "text": tok.decode(g)} for r, g in zip(EV, gens)]
            f_ = f"{stem}_eval_u{u}_s{sd}.jsonl"
            with open(f_, "w") as fo: fo.writelines(json.dumps(r) + "\n" for r in rs)
            ok = sum(correct(r) for r in rs); c_ = sum("</think>" in r["text"] for r in rs); tot += ok; cl += c_; files.append(f_)
            log(f"[grpo] eval u{u} seed {sd}: {ok}/{len(rs)} correct | closed think {c_} | {sum(len(g) for g in gens)} tokens ({time.time() - t0:.0f}s)")
        if KLD: kl, dt_ = kl_check(); log(f"[grpo] eval u{u}: deployment KL to the teacher {kl:.4f} ({len(KLD)} traces, 4096, {dt_:.0f}s)")
        offload(False); torch.manual_seed(a.seed + 7919 * (u + 1))
        return tot, cl, files

    order = list(range(len(rows))); random.Random(a.seed).shuffle(order)
    best, bad, t_start, u0, POS = a.grpo_best0, 0, time.time(), 1, {"n": 0}   # n: problems used so far
    if a.ckpt and "master" in ck:   # resume: the exact fp32 masters and the optimizer state, then the next update
        with torch.no_grad():
            for m_, v_ in zip(MASTER, ck["master"]): m_.copy_(v_)
        opt.load_state_dict(ck["opt"]); u0, best, bad = ck["u"] + 1, ck["best"], ck["bad"]
        POS["n"] = ck.get("pos", ck["u"] * ck["args"]["grpo_q"])   # the next unused problem, even when --grpo-q changed
        torch.manual_seed(a.seed + 7919 * u0)
        log(f"[grpo] resumed from {os.path.basename(a.ckpt)} after update {ck['u']} (best {best}, {bad} checks since)")
        ck["master"] = ck["opt"] = ck["ft"] = None   # (the weights are in the model already) - the RAM is needed for the next save

    def save_latest(u):
        if not a.grpo_latest: return
        st_ = {"pooler": {k: v.detach().cpu() for k, v in pooler.state_dict().items()}, "args": vars(a),
               "ft": {k: v.detach().cpu() for k, v in llm.state_dict().items() if k in FT_SD},
               "master": [m_.detach().cpu() for m_ in MASTER], "opt": opt.state_dict(), "u": u, "best": best, "bad": bad, "pos": POS["n"]}
        if os.path.exists(a.grpo_latest): os.remove(a.grpo_latest)   # ~14 GB at 1.7B: no room on the box for two copies
        torch.save(st_, a.grpo_latest); del st_

    def gen_safe(pr):   # out of memory while generating: fewer streams, generate the whole batch again
        while True:
            try: return rollout_kvmem(pr, a.grpo_slots) if a.kv_mem else rollout_cb(pr, a.grpo_slots)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); a.grpo_slots = max(4, a.grpo_slots * 3 // 4)
                if a.grpo_tok_budget: a.grpo_tok_budget = a.grpo_tok_budget * 3 // 4
                log(f"[grpo] out of memory while generating: now {a.grpo_slots} streams, budget {a.grpo_tok_budget} tokens")
    log(f"[grpo] {tag} {os.path.basename(a.ckpt)}: {len(rows)} problems, {a.grpo_q} x {a.group} samples per update, "
        f"lr pooler {a.lr} / model {a.ft_lr}, temp {a.temp}, cap {a.max_len}, slots {a.grpo_slots}")
    if EV and a.grpo_eval0:
        best, _, files = evaluate(0); save_to(a.out); hook(*files, a.out)
    HIST = {}
    if os.path.exists(stem + "_train.jsonl"):   # resumed: keep the updates before u0 (a killed run may have logged one more)
        keep = [l for l in open(stem + "_train.jsonl") if json.loads(l)["u"] < u0]
        with open(stem + "_train.jsonl", "w") as fo: fo.writelines(keep)
        for l in keep:
            r = json.loads(l); HIST.setdefault(r["u"], []).append(r["reward"])
    HIST = {k: sum(v) / len(v) for k, v in HIST.items()}
    trlog = open(stem + "_train.jsonl", "a")
    for u in range(u0, a.grpo_updates + 1):
        batch = [rows[order[(POS["n"] + k) % len(order)]] for k in range(a.grpo_q)]; POS["n"] += a.grpo_q
        pids = [tok.encode(r["prompt"], add_special_tokens=False) for r in batch]
        t0 = time.time(); offload(True); pooler.eval()
        res = torch.cuda.memory_allocated() / 2**30 if dev == "cuda" else 0.0   # what stays on the GPU while generating
        gens, ended = gen_safe([p_ for p_ in pids for _ in range(a.group)])
        offload(False); pooler.train(); t1 = time.time()
        R = [float("</think>" in t_ and correct({"text": t_, "gold": batch[i]["gold"]}))   # a \\boxed inside an unclosed
             for i in range(a.grpo_q) for k in range(a.group) for t_ in [tok.decode(gens[i * a.group + k])]]   # thought earns 0
        for i in range(a.grpo_q):
            for k in range(a.group):
                j = i * a.group + k
                trlog.write(json.dumps({"u": u, "src": batch[i].get("src", ""), "id": batch[i].get("id", ""), "gold": batch[i]["gold"],
                                        "reward": R[j], "ended": ended[j], "tokens": len(gens[j]), "text": tok.decode(gens[j])}) + "\n")
        trlog.flush()
        n_inf = n_bw = oom = 0
        for i in range(a.grpo_q):
            rs = R[i * a.group:(i + 1) * a.group]
            seqs = [pids[i] + gens[i * a.group + k] + ([im_end] if ended[i * a.group + k] else []) for k in range(a.group)]
            g_txt = GUIDE.get(batch[i].get("id", ""))
            if g_txt is not None:   # the guide joins the group with reward 1
                NG[0] += 1; NG[1] += int(max(rs) == min(rs) == 0.0)
                rs = rs + [1.0]; seqs.append(pids[i] + tok.encode(g_txt, add_special_tokens=False) + [im_end])
            if max(rs) == min(rs): continue
            n_inf += 1; m_ = sum(rs) / len(rs)
            for k in range(len(rs)):
                A = rs[k] - m_; ids = torch.tensor(seqs[k])
                try: seq_logp(ids, len(pids[i]), coef=-A / (len(rs) * a.grpo_q * a.grpo_lnorm), temp=a.temp); n_bw += 1
                except torch.cuda.OutOfMemoryError:
                    oom += 1; torch.cuda.empty_cache()
        if n_inf:
            f_ = min(1.0, u / max(1, a.grpo_warmup))
            for g_, lr_ in zip(opt.param_groups, base_lr): g_["lr"] = lr_ * f_
            gn = torch.nn.utils.clip_grad_norm_(list(pooler.parameters()) + MASTER, 1.0).item()
            opt.step()
            if FTP:
                with torch.no_grad():
                    for p_, m_ in zip(FTP, MASTER): p_.copy_(m_.to(p_.dtype))
        else: gn = 0.0
        opt.zero_grad(set_to_none=True)
        by = {}
        for i in range(a.grpo_q):
            s_ = batch[i].get("src", "").split("-")[0]; v_ = by.setdefault(s_, [0.0, 0])
            v_[0] += sum(R[i * a.group:(i + 1) * a.group]); v_[1] += a.group
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0.0
        if dev == "cuda": torch.cuda.reset_peak_memory_stats()
        HIST[u] = sum(R) / len(R); ks = sorted(HIST)
        tr_ = (f" | reward of the last 6 updates {sum(HIST[k] for k in ks[-6:]) / len(ks[-6:]):.3f}, first 6 known "
               f"{sum(HIST[k] for k in ks[:6]) / len(ks[:6]):.3f}") if len(ks) >= 6 else ""
        log(f"[grpo] u{u}: reward {sum(R) / len(R):.3f} (" + " ".join(f"{k} {v[0] / v[1]:.2f}" for k, v in sorted(by.items()))
            + f") | ended {sum(ended) / len(ended):.2f} | tokens {sum(len(g) for g in gens) / len(gens):.0f} | informative groups "
            f"{n_inf}/{a.grpo_q}" + (f" (with a guide so far {NG[0]}, informative only through it {NG[1]})" if GUIDE else "") + f" | grad norm {gn:.3f}{f' | {oom} OOM' if oom else ''} | gen {t1 - t0:.0f}s, update {time.time() - t1:.0f}s"
            f" | peak {mem:.1f} GiB, resident while generating {res:.1f} GiB, {a.grpo_slots} streams" + tr_)
        late = a.grpo_hours and time.time() - t_start > 3600 * a.grpo_hours
        if u % a.grpo_eval_every == 0 or late or u == a.grpo_updates:   # a check: eval (if any), teacher KL, state saved
            if EV:
                tot, cl, files = evaluate(u)
                if tot > best:
                    best, bad = tot, 0; save_to(a.out); hook(*files, a.out); log(f"[grpo] u{u}: new best {tot}")
                else:
                    bad += 1; hook(*files); log(f"[grpo] u{u}: no gain ({tot} vs best {best}, {bad} in a row)")
            elif KLD:
                offload(True); pooler.eval(); kl, dt_ = kl_check(); offload(False)
                log(f"[grpo] u{u}: deployment KL to the teacher {kl:.4f} ({len(KLD)} traces, 4096, {dt_:.0f}s)")
            save_latest(u); hook(*([a.grpo_latest] if a.grpo_latest else []), stem + "_train.jsonl")
            if EV and a.grpo_patience and bad >= a.grpo_patience: log(f"[grpo] stop: {bad} evals without a new best"); break
        if late: log(f"[grpo] stop: {a.grpo_hours} h"); break
    if not EV: save_to(a.out)
    log(f"[grpo] done: best {best} ({time.time() - t_start:.0f}s)")


rows = [json.loads(l) for f_ in a.traces.split(",") for l in open(f_)]
if not all("prompt" in r for r in rows):
    if not all("q" in r for r in rows if "prompt" not in r):
        from datasets import load_dataset
        PROB = {x["unique_id"]: x["problem"] for x in load_dataset("HuggingFaceH4/MATH-500", split="test")}
    SUF = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    for r in rows:
        if "prompt" in r: continue
        r["prompt"] = tok.apply_chat_template([{"role": "user", "content": r["q"] if "q" in r else PROB[r["id"]] + SUF}],
                                              add_generation_prompt=True, tokenize=False, enable_thinking=True)
rows = rows[a.skip:]
if a.n: rows = rows[:a.n]
data = []; SPIN = []; GROUPS = []
if a.grpo_groups:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sp_mathcheck import correct as _correct
    _im_end = tok.convert_tokens_to_ids("<|im_end|>"); _gs = [0, 0, 0]
    for r in rows:
        pid = tok.encode(r["prompt"], add_special_tokens=False); seqs, rs = [], []
        for t_ in r.get("samples", []):
            ans = tok.encode(t_, add_special_tokens=False)
            seqs.append(torch.tensor(pid + ans + ([_im_end] if len(ans) < a.max_len - 1 else [])))
            rs.append(1.0 if "</think>" in t_ and _correct({"text": t_, "gold": r["gold"]}) else (a.grpo_wrong if "</think>" in t_ else 0.0))
        g_ = r.get("guide")
        if g_ and "</think>" in g_ and _correct({"text": g_, "gold": r["gold"]}):
            seqs.append(torch.tensor(pid + tok.encode(g_, add_special_tokens=False) + [_im_end])); rs.append(1.0); _gs[2] += 1
        _gs[0] += 1
        if rs and max(rs) > min(rs): GROUPS.append((len(pid), seqs, rs)); _gs[1] += 1
    rows = []
    log(f"[sp] grpo groups: {_gs[0]} problems, {_gs[1]} informative (a right guide in {_gs[2]}), {sum(len(g[1]) for g in GROUPS)} sequences")
    data = [(g[1][0], g[0]) for g in GROUPS]
for r in rows:
    pid = tok.encode(r["prompt"], add_special_tokens=False)
    if a.spin:
        SPIN.append(tuple(torch.tensor(pid + tok.encode(r[k], add_special_tokens=False)) for k in ("chosen", "rejected")) + (len(pid),))
    ans = r["ids"] if "ids" in r else tok.encode(r.get("text", r.get("chosen", "")), add_special_tokens=False)
    data.append((torch.tensor(pid + ans), len(pid)))   # CPU; moved per trace

tag = f"S{a.S} rw{a.rw} C{a.C} maxd{a.maxd} L{a.layers} ffn{a.ffn}" + (" par" if a.par and a.mode == "train" else "") + (f" spin b{a.spin_beta}" if a.spin and a.mode == "train" else "") + (" grpo-groups" if a.grpo_groups and a.mode == "train" else "") + (" spnext" + (f" win{a.sn_win}/{a.sn_win_every}" if a.sn_win != "1" else "") if a.spnext and a.mode == "train" else "") + (" no-SP" if a.no_sp else "") + (" SP-null" if a.sp_null else "") + (" SP-swap" if a.sp_swap else "") + (f" filler{FILL}" if a.sp_filler else "") + (" pin-think" if a.pin_think else "") + (" kv-evict" if a.kv_evict else "") + (" para-window" if a.para_window else "") + (f" twin-mix{a.twin_mix}" if a.twin_mix >= 0 else "") + (" twin-minent" if a.twin_minent else "") + (" win-teacher" if a.win_teacher else "") + (" sp-replace" if a.sp_replace else "") + (f" rmt{a.rmt_block}" if a.rmt else "") + (f" kvmem{a.kv_rank}" if a.kv_mem else "") + (f" ctxL{a.ctx_hidden}" if a.ctx_hidden else "") + (" pe-rev" if a.pe_rev else "") + (" pe-norm" if a.pe_norm else "") + (f" FT{a.ft_layers}" if a.ft_layers else "") + (f" lora{LORA_RANK}" if LORA_RANK and not a.rmt else "") + (f" from{a.score_from}" if a.score_from else "") + (f" rkl{a.rkl_weight}" if a.rkl_weight and a.mode == "train" else "") + (" ctl-window" if a.ctl_window and a.mode == "train" else "") + (" ctl-para" if a.ctl_para and a.mode == "train" else "")
if a.mode == "massprof":   # where the pooler's attention mass sits over a long distant past (the user: evict by mass; how many tokens
    # to keep is a number to measure, not to guess). --traces: rollout texts. For each text and distant length nd (the window
    # holding the last rw tokens), the mass over the first nd generated tokens: the share of it on the top-k tokens, and the
    # number of tokens that carry 50 / 80 / 90 / 95% of it. Seconds per text.
    import collections as _c
    pooler.eval(); Ks = [256, 512, 1024, 2048, 4096]; cov = _c.defaultdict(list); need = _c.defaultdict(list); n_ = _c.Counter()
    tokmass = _c.Counter(); tokcnt = _c.Counter(); totmass = 0.0   # which tokens carry the mass (share of the total, over all texts at nd=2048)
    posmass = _c.Counter()   # and where they sit
    top20 = _c.Counter(); top20pos = _c.Counter()   # the tokens at the 20 heaviest positions of each text (the ~20 that carry half the mass)
    cov2 = {k_: _c.defaultdict(list) for k_ in ("importance", "effective")}; need2 = {k_: _c.defaultdict(list) for k_ in cov2}
    tokmass2 = _c.Counter(); tot2 = 0.0   # --att-prior: the model's attention importance itself, and the pooler's mass with the prior added
    with torch.no_grad():
        for r in rows:
            ids_ = tok.encode(r["text"], add_special_tokens=False)
            imp_ = None
            if ATT_L and len(ids_) >= 1024 + a.rw:
                pr_ = tok.encode(r["prompt"], add_special_tokens=False); imp_ = importance_of(pr_ + ids_, len(pr_))
            for nd in (1024, 2048, 4096, 8192):
                if len(ids_) < nd + a.rw: continue
                emb_ = embed(torch.tensor([ids_[:nd]], device=dev)).float()
                _, m_ = pooler.forward_with_mass(emb_); m_ = m_[0].float()
                s_ = m_.sort(descending=True).values; c_ = s_.cumsum(0) / s_.sum(); n_[nd] += 1
                if imp_ is not None:
                    _, me_ = pooler.forward_with_mass(emb_, att_bias(imp_[None, :nd]))
                    for nm_, x_ in (("importance", imp_[:nd]), ("effective", me_[0].float())):
                        sx_ = x_.sort(descending=True).values; cx_ = sx_.cumsum(0) / sx_.sum().clamp(min=1e-9)
                        for k in Ks:
                            if k < nd: cov2[nm_][(nd, k)].append(cx_[k - 1].item())
                        for p_ in (0.5, 0.8, 0.9, 0.95): need2[nm_][(nd, p_)].append(int((cx_ < p_).sum().item()) + 1)
                    if nd == 2048:
                        shi_ = (imp_[:nd] / imp_[:nd].sum().clamp(min=1e-9)).tolist(); tot2 += 1.0
                        for t_, w_ in zip(ids_[:nd], shi_): tokmass2[t_] += w_
                if nd == 2048:
                    sh_ = (m_ / m_.sum()).tolist(); totmass += 1.0
                    for t_, w_ in zip(ids_[:nd], sh_): tokmass[t_] += w_; tokcnt[t_] += 1
                    for (lo_, hi_) in ((0, 4), (4, 16), (16, 64), (64, 256), (256, 1024), (1024, 2048)): posmass[(lo_, hi_)] += sum(sh_[lo_:hi_])
                    for p_ in m_.topk(20).indices.tolist():
                        top20[tok.decode([ids_[p_]])] += 1; top20pos[min(p_ // 256, 7)] += 1
                for k in Ks:
                    if k < nd: cov[(nd, k)].append(c_[k - 1].item())
                for p_ in (0.5, 0.8, 0.9, 0.95): need[(nd, p_)].append(int((c_ < p_).sum().item()) + 1)
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")
    for nd in (1024, 2048, 4096, 8192):
        if not n_[nd]: continue
        log(f"[sp] massprof distant {nd} ({n_[nd]} texts): mass on the top " + " ".join(f"{k}: {100 * med(cov[(nd, k)]):.0f}%" for k in Ks if k < nd)
            + " | tokens for " + " ".join(f"{int(100 * p_)}%: {med(need[(nd, p_)])}" for p_ in (0.5, 0.8, 0.9, 0.95)) + " (medians)")
    if totmass:
        top = tokmass.most_common(12)
        log("[sp] massprof top tokens by mass (share of a text's mass, mean over texts at distant 2048; count = occurrences in the 2048): "
            + " | ".join(f"{tok.decode([t_])!r} {100 * w_ / totmass:.1f}% (x{tokcnt[t_] / totmass:.0f})" for t_, w_ in top))
        log("[sp] massprof mass by position in the distant 2048: " + " | ".join(f"{lo_}-{hi_}: {100 * posmass[(lo_, hi_)] / totmass:.1f}%" for (lo_, hi_) in sorted(posmass)))
        log("[sp] massprof the 20 heaviest positions per text - tokens: " + " | ".join(f"{t_!r} x{c_}" for t_, c_ in top20.most_common(16))
            + " || by 256-token block: " + " ".join(f"{k_}: {v_}" for k_, v_ in sorted(top20pos.items())))
    for nm_ in cov2:
        for nd in (1024, 2048, 4096, 8192):
            if not need2[nm_][(nd, 0.5)]: continue
            log(f"[sp] massprof {nm_} (att-prior {a.att_prior}, layers {a.att_layers}) distant {nd} ({len(need2[nm_][(nd, 0.5)])} texts): mass on the top "
                + " ".join(f"{k}: {100 * med(cov2[nm_][(nd, k)]):.0f}%" for k in Ks if k < nd)
                + " | tokens for " + " ".join(f"{int(100 * p_)}%: {med(need2[nm_][(nd, p_)])}" for p_ in (0.5, 0.8, 0.9, 0.95)) + " (medians)")
    if tot2:
        log("[sp] massprof importance top tokens (share of a text's importance, mean over texts at distant 2048): "
            + " | ".join(f"{tok.decode([t_])!r} {100 * w_ / tot2:.1f}%" for t_, w_ in tokmass2.most_common(12)))
    sys.exit(0)
if a.mode == "profile":
    pooler.eval(); acc = {}
    for ids, P in data: profile_trace(ids, P, acc)
    tot = sum(v[0] for v in acc.values()); n = sum(v[1] for v in acc.values())
    log(f"[sp] profile {tag} {os.path.basename(a.ckpt) or 'untrained'}: mean KL over all windows {tot / max(n, 1):.4f}")
    for key in ["0", "1-7", "8-63", "64-511", "512+"]:
        if key in acc: log(f"[sp] profile   window {key:>7}: KL {acc[key][0] / acc[key][1]:.4f}  share of the summed KL {100 * acc[key][0] / max(tot, 1e-9):.1f}%  ({acc[key][1]} tokens)")
elif a.mode == "probe":   # where the window's attention goes with a slot between the prompt and the window
    pooler.eval(); llm.set_attn_implementation("eager")
    acc = {}; ns = 0
    with torch.no_grad():
        for ids, P in data[:a.probe_n]:
            T = min(len(ids), P + a.max_len); ids = ids.to(dev); a_ids = ids[None, P:T]; La = a_ids.size(1)
            c0s = list(range(a.rw + a.C, La - a.C, a.C))
            if not c0s: continue
            for c0 in [c0s[int(i * (len(c0s) - 1) / 3)] for i in range(4)]:
                R = a.rw; kept = a_ids[:, :c0 - R]
                pn = pin_of(a_ids[0], c0 - R)
                parts = [embed(torch.cat([ids[:P], torch.tensor(pn, dtype=ids.dtype, device=dev)])[None]).to(dt)] + ([] if a.no_sp else [slot(kept)]) + [embed(a_ids[:, c0 - R:c0 + 64]).to(dt)]
                x = torch.cat(parts, 1); ns_ = 0 if a.no_sp else a.S; Pq = P + len(pn)
                o = llm.model(inputs_embeds=x, position_ids=torch.arange(x.size(1), device=dev)[None], output_attentions=True, output_hidden_states=True)
                for li, att in enumerate(o.attentions):
                    q = att[0, :, -64:].float()                                    # (heads, 64 queries, keys)
                    v = acc.setdefault(li, [0.0] * 6)
                    v[0] += q[:, :, Pq:Pq + ns_].sum(-1).mean().item() if ns_ else 0.0     # mass on the slot
                    v[1] += q[:, :, 0].mean().item()                                       # mass on the first prompt token
                    v[2] += q[:, :, Pq + ns_:Pq + ns_ + 32].sum(-1).mean().item()           # mass on the window's first 32 tokens
                    v[3] += q[:, :, -96:].sum(-1).mean().item()                            # mass on the newest 96 positions
                    hs = o.hidden_states[li + 1][0].float().norm(dim=-1)
                    v[4] += hs[Pq:Pq + ns_].mean().item() if ns_ else 0.0; v[5] += hs[Pq + ns_:].median().item()
                ns += 1
    log(f"[sp] probe {tag} {os.path.basename(a.ckpt) or 'untrained'}: {ns} states; per layer: attention mass of the last 64 window queries "
        f"on the slot / first prompt token / first 32 window tokens / newest 96, hidden norm slot / window median")
    for li in sorted(acc):
        v = [x / max(ns, 1) for x in acc[li]]
        log(f"[sp] probe L{li:02d}: slot {v[0]:.3f} first {v[1]:.3f} win32 {v[2]:.3f} new96 {v[3]:.3f} | norm slot {v[4]:.1f} window {v[5]:.1f}")
elif a.mode == "rollout":
    pooler.eval(); t0 = time.time(); n_tok = 0
    with open(a.roll_out, "a") as fo:
        step_ = len(rows) if (a.roll_cb or a.kv_evict or a.kv_mem) else a.roll_batch
        for i in range(0, len(rows), step_):
            rb = rows[i:i + step_]
            pr_ = [tok.encode(r["prompt"], add_special_tokens=False) for r in rb]
            LPG.clear()
            gens = rollout_kvmem(pr_, a.roll_batch)[0] if a.kv_mem else (rollout_kvlayer if a.kv_layerwise else rollout_kvkeep if (a.kv_keep > 0 or a.kv_keep_path) else rollout_kv)(pr_, a.roll_batch)[0] if a.kv_evict else rollout_cb(pr_, a.roll_batch)[0] if a.roll_cb else rollout(pr_)
            for k_, (r, g) in enumerate(zip(rb, gens)):
                rec = {"src": r.get("src", ""), "id": r.get("id", ""), "gold": r.get("gold", ""), "prompt": r["prompt"], "text": tok.decode(g)}
                if a.roll_lp: rec["ids"] = g; rec["lp"] = [round(x, 5) for x in LPG.get(k_, [])]
                fo.write(json.dumps(rec) + "\n"); n_tok += len(g)
            fo.flush()
            log(f"[sp] rollout {tag} {os.path.basename(a.ckpt) or 'untrained'}: {i + len(rb)}/{len(rows)} prompts, {n_tok} tokens ({time.time() - t0:.0f}s)"
                + (f" | steps taken from the masked twin {100 * PWN[0] / max(PWN[1], 1):.2f}%" if (a.para_window or a.twin_minent) else "")
                + (" | time " + " ".join(f"{k} {100 * RT[k] / max(sum(RT[x] for x in ('prompt', 'memory', 'window', 'decode')), 1e-9):.0f}%" for k in ("prompt", "memory", "window", "decode"))
                   + f" (chunks {RT['chunks']}, streams {RT['streams'] / max(RT['chunks'], 1):.1f}, prompt {RT['ptok'] / max(RT['streams'], 1):.0f} + window {RT['wtok'] / max(RT['streams'], 1):.0f} tokens re-read per stream-chunk)" if a.kv_mem else ""))
elif a.mode == "kvsel":
    t0 = time.time()
    with torch.no_grad():
        for ids, P in data: kvsel_trace(ids, P)
    if a.kvsel_dump and DUMPS: torch.save(DUMPS, a.kvsel_dump); log(f"[sp] kvsel dump: {len(DUMPS)} positions -> {a.kvsel_dump}")
    if KVS:
        n_ = len(KVS[("window", 0)]); rows_ = sorted(KVS, key=lambda k: (k[1], sum(KVS[k]) / len(KVS[k])))
        log(f"[sp] kvsel {tag}: {n_} positions in {len(data)} traces, rw {a.rw} | mean KL to the full-context teacher on the next chunk: "
            + " | ".join(f"{r}{'' if B == 0 else ' B=' + str(B)} {sum(v) / len(v):.4f}" for (r, B), v in ((k, KVS[k]) for k in rows_)) + f" ({time.time() - t0:.0f}s)")
elif a.mode == "oracle":
    pooler.eval(); t0 = time.time()
    for p_ in pooler.parameters(): p_.requires_grad_(False)
    OTHER = [data[0][0][data[0][1]:], data[1][0][data[1][1]:] if len(data) > 1 else data[0][0][data[0][1]:]]   # --oracle-content: answer tokens of two traces
    OTHER_KEY = [data[0][0][data[0][1]:data[0][1] + 8].tolist()]
    for ids, P in data: (oracle_h_trace if a.oracle_horizon > 0 else oracle_trace)(ids, P)
    if ORC and a.oracle_horizon > 0:
        import statistics as st_
        m_ = lambda k: st_.median(r[k] for r in ORC); pairs = [r["pair"] for r in ORC if "pair" in r]
        log(f"[sp] oracle-h{a.oracle_horizon} {tag} {os.path.basename(a.ckpt) or 'untrained'}: {len(ORC)} positions in {len(data)} chains, {a.oracle_steps} steps | "
            f"median KL on the fitted {a.oracle_horizon} chunks: pooler {m_('pooler_fit'):.4f} seeded {m_('seeded_fit'):.4f} random {m_('random_fit'):.4f} | "
            f"on the 2 chunks after (held out): window-only {m_('window'):.4f} pooler {m_('pooler'):.4f} seeded optimum {m_('seeded'):.4f} random optimum "
            f"{m_('random'):.4f} | median cos to the pooler's output: seeded {m_('cos'):.3f} random {m_('cosB'):.3f}, between optima {m_('cosAB'):.3f}"
            + (f" | content: pooler output vs shuffled tokens cos {m_('cos_shuffled'):.3f} (KL {m_('kl_shuffled'):.4f}), vs another trace cos {m_('cos_other'):.3f} (KL {m_('kl_other'):.4f})" if "cos_shuffled" in ORC[0] else "")
            + (f" | neighbours ({len(pairs)} pairs{', chained' if a.oracle_chain else ''}): cos pooler {st_.median(p['cos_pooler'] for p in pairs):.3f} seeded optima "
               f"{st_.median(p['cos_seeded'] for p in pairs):.3f} (move {st_.median(p['move'] for p in pairs):.3f}) random optima {st_.median(p['cos_random'] for p in pairs):.3f} corrections "
               f"{st_.median(p['cos_corr'] for p in pairs):.3f}; the previous position's optima on these held-out chunks: seeded "
               f"{st_.median(p['kl_prev_seeded'] for p in pairs):.4f} random {st_.median(p['kl_prev_random'] for p in pairs):.4f}" if pairs else "")
            + f" ({time.time() - t0:.0f}s)")
    elif ORC:
        import statistics as st_
        m_ = lambda k: st_.median(r[k] for r in ORC); mn = lambda k: sum(r[k] for r in ORC) / len(ORC)
        nx = lambda k: st_.median(r["next"][k] for r in ORC if r["next"])
        log(f"[sp] oracle {tag} {os.path.basename(a.ckpt) or 'untrained'}: {len(ORC)} positions, {a.oracle_steps} steps | median KL: window-only "
            f"{m_('window'):.4f} pooler {m_('pooler'):.4f} slots optimised from the pooler {m_('seeded'):.4f} from random {m_('random'):.4f} | "
            f"next chunk, memory kept (median): pooler {nx('pooler'):.4f} seeded {nx('seeded'):.4f} random {nx('random'):.4f} | mean move from the "
            f"pooler's output: seeded {mn('rel'):.3f} (cos {mn('cos'):.3f}), random optimum {mn('relB'):.3f} (cos {mn('cosB'):.3f}), between optima "
            f"{mn('relAB'):.3f} ({time.time() - t0:.0f}s)")
        pairs = [r["pair"] for r in ORC if "pair" in r]
        if pairs:
            md = lambda k: st_.median(p_[k] for p_ in pairs); mp = lambda k: st_.median(r[k] for r in ORC if "pair" in r)
            log(f"[sp] oracle-run {tag}: {len(pairs)} consecutive pairs ({a.C} tokens apart) in {len(data)} chains | median cosine between neighbours: pooler "
                f"outputs {md('cos_pooler'):.3f} seeded optima {md('cos_seeded'):.3f} random optima {md('cos_random'):.3f} | corrections "
                f"{md('cos_corr'):.3f} | median KL on a chunk: pooler {mp('pooler'):.4f} own optimum {mp('seeded'):.4f} the previous chunk's seeded "
                f"optimum {md('kl_prev_seeded'):.4f} its random optimum {md('kl_prev_random'):.4f} window-only {mp('window'):.4f}")
elif a.mode == "eval":
    pooler.eval(); t0 = time.time(); K = A_ = N = 0
    for ids, P in data:
        if DUMP is not None: DUMP.append([])
        k, g, n = run_trace_rmt(ids, P, False) if a.rmt else run_trace_wt(ids, P, False) if a.win_teacher else run_trace(ids, P, False); K += k; A_ += g; N += n
    if DUMP is not None:
        torch.save([torch.cat(x) if x else torch.zeros(0, 7) for x in DUMP], a.dump)
    log(f"[sp] eval {tag} {os.path.basename(a.ckpt) or 'untrained'}: {len(data)} traces {int(N)} tokens  KL {K / max(N, 1):.4f}  "
        f"top1 agree {100 * A_ / max(N, 1):.2f}%  reverse KL {RKL[0] / max(RKL[1], 1):.4f}  ({time.time() - t0:.0f}s)"
        + (" | by position (0, 1-7, 8-63, 64+): " + " ".join(f"{v[0] / max(v[1], 1):.4f}" for v in WTPOS) if (a.win_teacher or a.rmt) else "")
        + gen_report() + (f" | forced vs teacher-forced max diff {max(GEN['diff']):.2e}" if GEN.get("diff") else ""))
    if a.end_stat:
        e = ENDST; log(f"[sp] end-stat {tag}: mean p(</think>) teacher {e['pt'] / max(e['n'], 1):.3e} student {e['ps'] / max(e['n'], 1):.3e} | "
                       f"at the {e['k']} real ends: log p teacher {e['lt'] / max(e['k'], 1):.3f} student {e['ls'] / max(e['k'], 1):.3f} | "
                       f"mean p(Wait/But/Alternatively/Hmm) teacher {e['ht'] / max(e['n'], 1):.4f} student {e['hs'] / max(e['n'], 1):.4f} | "
                       f"mean entropy (nats) teacher {e['et'] / max(e['n'], 1):.4f} student {e['es'] / max(e['n'], 1):.4f}")
else:
    pooler.train()
    groups = [{"params": list(pooler.parameters()) if not a.rmt else RMT_PARAMS, "lr": a.lr}]
    if LORA_PARAMS: groups.append({"params": LORA_PARAMS, "lr": a.lora_lr})
    FTP, MASTER = [], []
    if a.ft_layers:
        if FT_SD is None:   # fresh run: the frozen original is the teacher; resuming from an FT ckpt keeps the original taken at load
            FT_SD = {}
        for k, v in llm.state_dict().items():   # layers tuned now but not in the checkpoint are still the original: add them
            if k.startswith("model.layers.") and int(k.split(".")[2]) < a.ft_layers and k not in FT_SD and not k.endswith((".A", ".B")):
                FT_SD[k] = v.detach().to("cpu", copy=True)
        FTP = [p_ for li in range(a.ft_layers) for n_, p_ in llm.model.layers[li].named_parameters() if not n_.endswith((".A", ".B"))]
        for p_ in FTP: p_.requires_grad_(True)
        MASTER = [p_.detach().float().clone().requires_grad_(True) for p_ in FTP]
        def _acc(m_):
            def hook(p_):
                if m_.grad is None: m_.grad = p_.grad.float()
                else: m_.grad.add_(p_.grad.float())
                p_.grad = None
            return hook
        for p_, m_ in zip(FTP, MASTER): p_.register_post_accumulate_grad_hook(_acc(m_))
        groups.append({"params": MASTER, "lr": a.ft_lr})
    if a.opt8bit:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(groups, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(groups, weight_decay=0.0)
    if a.opt_state and os.path.exists(a.opt_state):
        try:
            want_ = [g_["lr"] for g_ in opt.param_groups]
            opt.load_state_dict(torch.load(a.opt_state, map_location="cpu", weights_only=False))
            for g_, w_ in zip(opt.param_groups, want_): g_["lr"] = w_; g_.pop("initial_lr", None)   # load_state_dict brings back the saved lrs: keep this run's
            log(f"[sp] optimizer state loaded from {a.opt_state} (lrs {', '.join(f'{x:g}' for x in want_)})")
        except Exception as e_: log(f"[sp] optimizer state NOT loaded ({str(e_)[:100]}) - fresh optimizer")
    steps = int(len(data) * a.epochs); n_upd = max(1, steps // a.accum)
    if a.mode == "train":
        sched = torch.optim.lr_scheduler.LambdaLR(opt, (lambda s: 1.0) if a.const_lr else
                                                  (lambda s: min(1.0, (s + 1) / max(2, n_upd // 15)) * max(0.1, 1 - s / n_upd)))
    if a.mode == "grpo": run_grpo(opt, FTP, MASTER); sys.exit(0)
    log(f"[sp] train {tag}: {steps} traces, {a.accum} per update ({n_upd} updates), pooler lr {a.lr}"
        + (f", FT layers 0-{a.ft_layers - 1} lr {a.ft_lr} ({sum(p_.numel() for p_ in FTP) / 1e6:.0f}M)" if a.ft_layers else "")
        + (f", LoRA rank {LORA_RANK} lr {a.lora_lr} ({sum(p_.numel() for p_ in LORA_PARAMS) / 1e6:.1f}M, fp32)" if LORA_PARAMS else ""))

    def save():
        if not a.out: return
        ck = {"pooler": {k: v.detach().cpu() for k, v in pooler.state_dict().items()}, "args": vars(a)}
        if a.ft_layers: ck["ft"] = {k: v.detach().cpu() for k, v in llm.state_dict().items() if k in FT_SD}
        if a.rmt: ck["rmt"] = {"mem": {k: v.detach().cpu() for k, v in RMT.state_dict().items()},
                               "lora": {k: v.detach().cpu() for k, v in llm.state_dict().items() if k.endswith((".A", ".B"))}}
        if LORA_PARAMS:
            ck["lora"] = {k: v.detach().cpu() for k, v in llm.state_dict().items() if k.endswith((".A", ".B"))}; ck["lora_rank"] = LORA_RANK
            if not a.ft_layers and BASE_FT is not None: ck["ft"] = BASE_FT   # the tuned base the adapters sit on
        torch.save(ck, a.out)
        if a.opt_state: torch.save(opt.state_dict(), a.opt_state)

    t0 = time.time(); K = A_ = N = 0
    GSK = None
    if a.grad_sketch and a.grpo_groups and LORA_PARAMS:
        GSK_P = [LORA_PARAMS, [p_ for p_ in pooler.parameters() if p_.requires_grad]]   # the hypernet's gradient is there even at lr 0
        def _gsk():
            return torch.stack([_gsk1(ps_, 1000 + 100000 * k_) for k_, ps_ in enumerate(GSK_P)])
        def _gsk1(params, seed0):
            out = torch.zeros(1 << 20, device=dev, dtype=torch.float64)
            with torch.no_grad():
                for i_, p_ in enumerate(params):
                    if p_.grad is None: continue
                    g_ = torch.Generator(device=p_.device).manual_seed(seed0 + i_)
                    idx = torch.randint(0, 1 << 20, (p_.numel(),), generator=g_, device=p_.device)
                    sg = torch.randint(0, 2, (p_.numel(),), generator=g_, device=p_.device).double() * 2 - 1
                    out.index_add_(0, idx, p_.grad.detach().flatten().double() * sg)
            return out
        GSK = [torch.zeros(2, 1 << 20, device=dev, dtype=torch.float64) for _ in range(3)] + [[0, 0]]
    if a.grad_ckpt:
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); llm.train()
    if FTP:   # pinned frozen copies in the same order as the tuned params, for the fast swap
        names = [n_ for li in range(a.ft_layers) for n_, _ in llm.model.layers[li].named_parameters(prefix=f"model.layers.{li}") if not n_.endswith((".A", ".B"))]
        SWAP["pairs"] = [(p_, FT_SD[n_].pin_memory() if dev == "cuda" else FT_SD[n_], m_) for p_, n_, m_ in zip(FTP, names, MASTER)]
    refs = {}; WINS = [int(x) for x in a.sn_win.split(",")]
    _on_dev = lambda r: None if r is None else r.to(dev, non_blocking=True)
    WI = {"i": 0, "from": 0, "kl": []}; STOP = {"now": False}
    if a.spin:   # the reference: the starting weights, same loop
        with torch.no_grad():
            for i_, (ids_c, ids_r, P_) in enumerate(SPIN): SPIN_REF[i_] = (seq_logp(ids_c, P_)[0], seq_logp(ids_r, P_)[0])
        log(f"[sp] spin: reference log-probs of {len(SPIN)} pairs ({time.time() - t0:.0f}s)")
    for st in range(steps):
        ids, P = data[st % len(data)]
        if FTP and st % a.accum == 0 and not a.spin and not a.grpo_groups and not a.win_teacher and not a.rmt:   # the teacher pass for the next group of traces in ONE weight swap
            refs = {}
            with _Frozen():
                for j in range(st, min(st + a.accum, steps)):
                    ids_j, P_j = data[j % len(data)]
                    T_j = min(len(ids_j), P_j + a.max_len)
                    lo_j = a.sn_t0 if a.spnext else (a.C if a.par else a.rw + a.C)
                    if T_j - P_j - lo_j >= (a.sn_batch // 4 if a.spnext else (a.rw + a.C if a.par else a.C)):
                        refs[j] = teacher_hidden(ids_j.to(dev), P_j + lo_j, T_j).to("cpu")   # parked on the CPU until its turn
            if dev == "cuda": torch.cuda.empty_cache()   # long traces' teacher passes leave large odd-sized blocks behind
        try:
            if a.spnext:
                if a.sn_win_mix: SNW["w"] = random.choice(WINS)
                elif a.sn_win_plateau: SNW["w"] = WINS[WI["i"]]
                else: SNW["w"] = WINS[min(st // a.sn_win_every, len(WINS) - 1)]
                k, g, n = run_trace_spnext(ids, P, ref_h=_on_dev(refs.pop(st, None)))
            elif a.spin: k, g, n = run_pair_spin(st)
            elif a.grpo_groups: k, g, n = run_group(st)
            elif a.rmt: k, g, n = run_trace_rmt(ids, P, True)
            elif a.win_teacher: k, g, n = run_trace_wt(ids, P, True)
            elif a.par: k, g, n = run_trace_par(ids, P, ref_h=_on_dev(refs.pop(st, None)))
            else: k, g, n = run_trace(ids, P, True, ref_h=_on_dev(refs.pop(st, None)))
            if GSK is not None:   # this group's gradient = the accumulated gradient now minus before it (the sketch is linear)
                s_ = _gsk(); GSK[st % 2] += s_ - GSK[2]; GSK[2] = s_; GSK[3][st % 2] += 1
        except torch.cuda.OutOfMemoryError as e_:   # a long-prompt trace: drop it (and its half-built grads), keep training
            if GSK is not None: GSK[2] = torch.zeros_like(GSK[2])
            refs.pop(st, None); pooler.zero_grad(set_to_none=True)
            for p_ in FTP + LORA_PARAMS: p_.grad = None
            for m_ in MASTER: m_.grad = None
            ma, mr = torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30; torch.cuda.empty_cache()
            log(f"[sp] trace {st + 1} ({len(ids) - P} tokens): out of memory, skipped | {ma:.1f} / {mr:.1f} GiB allocated / reserved | "
                f"{str(e_).splitlines()[0][:120]}"); k = g = n = 0
        K += k; A_ += g; N += n
        if (st + 1) % a.accum == 0 or st + 1 == steps:
            torch.nn.utils.clip_grad_norm_((RMT_PARAMS if a.rmt else list(pooler.parameters())) + MASTER + LORA_PARAMS, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True); sched.step()
            if GSK is not None: GSK[2] = torch.zeros_like(GSK[2])
            if FTP:
                with torch.no_grad():
                    for p_, m_ in zip(FTP, MASTER): p_.copy_(m_.to(p_.dtype))
        if (st + 1) % 25 == 0 or st + 1 == steps:
            x = f" | w{SNW['w']} window-0 KL {SNW['k0'] / max(SNW['n0'], 1):.4f}" if a.spnext and len(WINS) > 1 else ""
            if a.spin: log(f"[sp] train {tag} pair {st + 1}/{steps}: DPO loss {K / max(N, 1):.4f} chosen ahead {100 * A_ / max(N, 1):.0f}% ({time.time() - t0:.0f}s)")
            elif a.grpo_groups: log(f"[sp] train {tag} grpo group {st + 1}/{steps}: mean reward {K / max(N, 1):.3f} ({time.time() - t0:.0f}s)")
            else:
                lab = ["<1k", "1-2k", "2-4k", "4-8k", "8k+"]
                x += (" | KL by past outside the window: " + " ".join(f"{lab[k]} {v[0] / max(v[1], 1):.4f}" for k, v in sorted(DBK.items()))) if DBK else ""
                DBK.clear()
                if RKL[1]: x += f" | reverse KL {RKL[0] / RKL[1]:.4f}"; RKL[0] = 0.0; RKL[1] = 0
                if CTLP[1]: x += f" | paragraph openings {100 * CTLP[0] / CTLP[1]:.1f}%"
                log(f"[sp] train {tag} trace {st + 1}/{steps}: KL {K / max(N, 1):.4f} top1 {100 * A_ / max(N, 1):.2f}%{x} ({time.time() - t0:.0f}s)")
            if a.spnext and a.sn_win_plateau and N:   # curriculum by plateau: 100-trace means at this w, last vs the one before
                WI["kl"].append(K / N); n_at = st + 1 - WI["from"]; hist = WI["kl"]
                b_ = a.sn_win_block
                flat = len(hist) >= 2 * b_ and sum(hist[-b_:]) / b_ > (1 - a.sn_win_plateau) * sum(hist[-2 * b_:-b_]) / b_
                if flat or n_at >= a.sn_win_max:
                    log(f"[sp] curriculum: w{WINS[WI['i']]} {'plateau' if flat else 'cap'} after {n_at} traces "
                        f"(last {25 * len(hist[-a.sn_win_block:])} KL {sum(hist[-a.sn_win_block:]) / len(hist[-a.sn_win_block:]):.4f})")
                    if WI["i"] + 1 >= len(WINS): STOP["now"] = True
                    else: WI["i"] += 1; WI["from"] = st + 1; WI["kl"] = []
            K = A_ = N = 0; SNW["k0"] = 0.0; SNW["n0"] = 0
        if (st + 1) % 200 == 0: save()
        if STOP["now"]: break
    if GSK is not None:
        cs_ = [(GSK[0][k_] @ GSK[1][k_] / (GSK[0][k_].norm() * GSK[1][k_].norm() + 1e-30)).item() for k_ in range(2)]
        torch.save({"even": GSK[0][0].cpu(), "odd": GSK[1][0].cpu(), "even_h": GSK[0][1].cpu(), "odd_h": GSK[1][1].cpu(), "n": GSK[3],
                    "norm": [GSK[0][k_].norm().item() + GSK[1][k_].norm().item() for k_ in range(2)]}, a.grad_sketch)
        log(f"[sp] gradient halves: LoRA cos {cs_[0]:+.3f}, hypernet cos {cs_[1]:+.3f} (even {GSK[3][0]} groups, odd {GSK[3][1]}) -> {a.grad_sketch}")
    save(); log(f"[sp] trained {tag} -> {a.out}")
