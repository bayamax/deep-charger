#!/usr/bin/env python3
"""On-the-fly alternating loop for the conversational lineage (derived from grpo_pool.py, same rollout
machinery under compression):

  each step : one question, B rollouts decoded in lockstep (temperature --temp, EOS terminal); the ones that
              land, read the gold and state it, carry no tags and pass the cheap judge are the positives;
              SFT step on them (-mean log p over the policy's own tokens, information blocks masked), then an
              SFT step on the same number of Dolphin-R1 records (plain sequences, prompt masked)
  logging   : per step the skip reasons (no reply / page not found / wrong / tags / judge) and the acceptance
  saving    : latest.safetensors every --save-every steps (base+LoRA merged view + pooler), loadable by pool_eval
  usage     : python3 online_loop.py <model dir> <outdir> --questions q.jsonl --dolphin d.jsonl [--b 8] [--steps 400]

GRPO for the pooler lineage, in the TEACHER's recipe (grpo_ep_more.py) but with compression on.

  environment : grpo_ep_more verbatim (kw||ask -> top-1 page, 256-token head, <more/> paging, MAXS 5, MAXM 8, GEN 1500, temp 0.9,
                plain sampling with only the <information ban)   -- same code as pool_eval.py
                + samepage (default on): a search that lands on a page already shown in this rollout serves the NEXT chunk of it
                  instead of the head again, and says "(this page is used up ...)" once the page is exhausted
  reward      : grounded-correct = 1.0, everything else 0.0 (no partial credit), group-normalised advantage, G=12, one question/step
  loss        : -adv/G * mean log p over the policy's own tokens, computed under compression by rebuilding every block exactly
                as the policy saw it (the pooled set after mass eviction is recorded during the rollout); Adam lr 1e-5 on the
                LoRA adapters + the pooler
  inference   : rollouts run with the deployed compression (RW 768, MAXD 384, chunk 128, mass eviction)
  data        : corpus_box_final.jsonl (the teacher's 2857-question pool), gold <= 6 words, shuffled with seed 0, sequential;
                the held-out eval300 questions are removed if present
  usage       : python3 grpo_pool.py <init.safetensors> <outdir> [--steps 200] [--g 12] [--rw 768] [--maxd 384] [--lr 1e-5]
  resume      : if <outdir>/latest.safetensors + state.json exist, continues from there
"""
import os, sys, json, time, re, ssl, random, argparse, threading, collections, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser()
ap.add_argument("init"); ap.add_argument("outdir")
ap.add_argument("--steps", type=int, default=200); ap.add_argument("--g", type=int, default=12)
ap.add_argument("--rw", type=int, default=768); ap.add_argument("--maxd", type=int, default=384)
ap.add_argument("--chunk", type=int, default=128); ap.add_argument("--temp", type=float, default=0.9)
ap.add_argument("--gen", type=int, default=1500); ap.add_argument("--budget", type=int, default=900, help="wall-clock seconds for one packed rollout batch; a row still generating when it runs out is unfinished, and --complete-only drops it"); ap.add_argument("--maxs", type=int, default=5); ap.add_argument("--maxm", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--pooler-lr", type=float, default=1e-5)
ap.add_argument("--corpus", default="/root/work/corpus_box_final.jsonl"); ap.add_argument("--heldout", default="/root/work/eval300.jsonl")
ap.add_argument("--save-every", type=int, default=20)
ap.add_argument("--maxsrch", type=int, default=0, help=">0: stop a rollout once it has issued this many <search> tags (loop guard; the rollout ends unlanded, reward 0). 0 = teacher recipe (only the 1500-token cap)")
ap.add_argument("--phantom", type=float, default=0.0, help=">0: add one phantom rollout with this reward to every group's statistics, so all-wrong / all-right groups still get a (uniform) advantage instead of being skipped. 0 = plain GRPO")
ap.add_argument("--phantom-scale", type=float, default=0.5, help="advantage multiplier for groups whose real rewards are all identical (they only learn through the phantom)")
ap.add_argument("--gradckpt", type=int, default=1, help="1: gradient checkpointing through the transformer during the policy-gradient pass (16GB cards)")
ap.add_argument("--lora-rank", type=int, default=16, help="LoRA rank on the LLM (teacher used 16; the SFT lineage used 128)")
ap.add_argument("--lora-layers", default="20-27", help="transformer layers the LoRA covers, e.g. 20-27 (teacher) or 'all'")
ap.add_argument("--pooler", default="lora", choices=["none", "ln", "lora"],
                help="none: frozen | ln: query + layernorms + out_scale only | lora: that plus a low-rank adapter on the pooler's big matrices")
ap.add_argument("--pooler-rank", type=int, default=8); ap.add_argument("--pooler-scale", type=float, default=2.0)
ap.add_argument("--pooler-init", default="", help="safetensors holding the pooler tensors (when the model comes from a merged HF dir)")
ap.add_argument("--fetchers", type=int, default=8, help="how many Wikipedia lookups a batched step may have in flight. The searches were the last serial part of a batched rollout: two API round trips each with a 0.6 s courtesy sleep, run one row at a time. 1 restores the serial behaviour")
ap.add_argument("--backprop", type=int, default=0, help="how many of the group's rollouts the gradient actually replays. 0 = all of them. A smaller number keeps the update as cheap as it was while the group itself grows: with a 0/1 reward every rollout in a reward stratum carries the same advantage, so which members are replayed is a free choice, and each is reweighted so the group's gradient keeps its original scale")
ap.add_argument("--select", default="random", choices=["random", "grounded"], help="how the replayed rollouts are drawn inside each reward stratum. random keeps the estimate unbiased; grounded prefers zero-reward rollouts that DID read the gold, which aims the negative gradient at the reading failure rather than at the search failure (deliberately biased)")
ap.add_argument("--batch", type=int, default=1, help="rollouts decoded in lockstep. 1 = the original one-at-a-time path. >1 batches the per-token decode, which is where ~89%% of wall time goes at 14%% GPU utilisation")
ap.add_argument("--selftest-batch", type=int, default=0, help="run the greedy equivalence check between the single and batched rollout, print the verdict and exit")
ap.add_argument("--questions", default="/root/work/selfq_all.jsonl"); ap.add_argument("--dolphin", default="/root/work/dolphin_v1.jsonl")
ap.add_argument("--b", type=int, default=8, help="rollouts per question, decoded together")
ap.add_argument("--stop", default="eos", choices=["eos", "answer"]); ap.add_argument("--judge", type=int, default=1)
ap.add_argument("--dolphin-ratio", type=float, default=2.0, help="Dolphin records per accepted search rollout in the same step"); ap.add_argument("--dolphin-min", type=int, default=2, help="Dolphin records in a step with no accepted rollout")
ap.add_argument("--maxlen", type=int, default=4096)
ap.add_argument("--replay", default="", help="jsonl of verified search traces {q,text}; each step trains on --replay-per-step of them (the retention data)")
ap.add_argument("--guard", type=int, default=0, help="1: watch the share of rollouts that write no reply over a sliding window; when it runs away from the opening baseline, reload the last healthy checkpoint, halve the learning rate and carry on (three times, then stop)"); ap.add_argument("--guard-window", type=int, default=80); ap.add_argument("--guard-floor", type=float, default=0.20, help="no trip below this absolute rate"); ap.add_argument("--guard-mult", type=float, default=2.0, help="trip at this multiple of the opening baseline"); ap.add_argument("--guard-cut", type=float, default=3.0, help="trip at this multiple of the opening share of rollouts cut for searching without end"); ap.add_argument("--guard-cut-floor", type=float, default=0.20); ap.add_argument("--guard-ns-floor", type=float, default=6.0, help="no searches trip below this absolute mean"); ap.add_argument("--guard-ns", type=float, default=1.8, help="trip at this multiple of the opening searches per rollout"); ap.add_argument("--guard-rollbacks", type=int, default=3); ap.add_argument("--complete-only", type=int, default=0, help="1: train only on rollouts that actually finished their turn (EOS reached, no repetition death). Not a quality filter: a rollout cut off by the token cap is an unfinished fragment, and training on it teaches the model not to stop -- r7 went from 5%% to 59%% no-reply that way"); ap.add_argument("--reason", default="", help="jsonl of reasoning problems {q, reply}: the loop leaves the search corpus and runs GRPO on these instead, the teacher scoring each sample against the reference"); ap.add_argument("--reason-g", type=int, default=8, help="samples per problem"); ap.add_argument("--search-every", type=int, default=0, help=">0: every Nth step is a search question from the pool instead of a reasoning problem, scored the same way but on its own reward"); ap.add_argument("--w-talk", type=float, default=0.5, help="what the teacher can add to a grounded-correct search rollout for a reply that reads conversationally"); ap.add_argument("--search-wheels", type=int, default=0, help="1: a search group that scores zero also learns from a demonstration. Off by default: the search side already works at the product settings, and a teacher trajectory carries the old register and the old thinking style"); ap.add_argument("--search-demo", default="", help="jsonl of {q, traj}: the teacher's own trajectory for each search question, trained on (searches only, reply discarded) when a group scores zero"); ap.add_argument("--wheels", type=int, default=0, help="1: when all samples of a reasoning group score zero there is no signal, so train on the reference answer instead (the teacher demonstrates the problem the model cannot do)"); ap.add_argument("--judge-api", default="deepseek", choices=["deepseek", "openai"], help="which teacher scores the reasoning samples"); ap.add_argument("--judge-model", default="", help="model name for that teacher; empty picks the default for the api"); ap.add_argument("--reason-stub", type=int, default=0, help="1: score by shape alone, no teacher call (for a smoke run with no credit)"); ap.add_argument("--queue", type=int, default=0, help="1: each rollout batch takes --b different questions; the rows queue up and every step trains on one of them (no selection) plus Dolphin; a new batch runs when the queue is empty"); ap.add_argument("--train-all", type=int, default=0, help="1: train on every rollout of the step, no selection (the gold and the judge only measure)"); ap.add_argument("--replay-per-step", type=int, default=2); ap.add_argument("--rollout-every", type=int, default=1, help="do the measurement rollout only every N steps (accepted ones join the replay set)")
ap.add_argument("--pg-norm", default="mean", choices=["mean", "const"], help="how a rollout's policy-gradient loss is averaged over its tokens. mean: by its own length (a long wrong rollout is then punished less per token than a short one, so wrong rollouts grow: g5 search side 176 -> 436 words, 7%% -> 36%% unfinished). const: by --pg-norm-len, so every token of a wrong rollout costs the same and a long one costs more in total.")
ap.add_argument("--guard-steps", type=int, default=10, help="search steps per guard window (baseline = the first window). 10 steps = 120 rollouts was noisy enough to trip on hard stretches; 20 halves that")
ap.add_argument("--guard-halve", type=int, default=1, help="1: a rollback also halves both learning rates. 0: weights only")
ap.add_argument("--guard-pass", type=float, default=0.5, help="the guard trips only when, besides the unfinished or search-count rise, the window's pass rate has fallen to this fraction of the baseline pass rate (one hard question in a window is not a collapse)")
ap.add_argument("--save-lora-only", type=int, default=0, help="1: checkpoints hold only the trained parameters (LoRA) and the pooler, not the 3.5 GB of base weights the base directory already has")
ap.add_argument("--mt-items", default="", help="multi-turn GRPO: jsonl of {q, gold, hist} where hist is the earlier exchanges of a conversation as chat messages (user / assistant, replies without thinking). Every step is a search step on one of these, rolled out with the history in the prompt as the app sends it, scored as a search rollout (the teacher's naturalness check also sees the history)")
ap.add_argument("--eval-file", default="", help="evaluation only: generate one reply per question in this jsonl ({\"q\": ...}) with the batched rollout, --b questions at a time, write {q, text, ns} to --eval-out and exit")
ap.add_argument("--eval-out", default="")
ap.add_argument("--probe-out", default="", help="with --mt-items: no training - every item (or the first --probe-n) is rolled out --probe-g times in the batched rollout (several items per batch), scored by the gold string only, and written as {q, gold, hist, n, pass, texts of the passing samples}; then exit")
ap.add_argument("--probe-g", type=int, default=4); ap.add_argument("--probe-n", type=int, default=0)
ap.add_argument("--demo-sft", default="", help="jsonl of {q, hist, traj}: supervised steps on teacher trajectories (thinking and searches trained, information blocks and the reply masked), --demo-per-step a step, beside --replay-per-step of the model's own verified traces {q, hist, text}")
ap.add_argument("--demo-per-step", type=int, default=4)
ap.add_argument("--demo-on-fail", type=float, default=0.0, help=">0 (with --search-demo): teacher-mixed GRPO - a search group none of whose samples passes also takes one supervised step on that question's teacher trajectory (searching trained, pages and reply masked) at this weight, beside its policy gradient")
ap.add_argument("--followup", type=float, default=0.0, help="with --mt-items: after a search step on a question without history that at least one sample passed, with this probability the teacher writes the user's follow-up (refers to the exchange by pronoun, asks one new fact with a short answer an English Wikipedia article states; the page is fetched and the answer checked to be on it); it becomes the next search step's item, the exchange as its history - follow-ups on the fly, never seen before")
ap.add_argument("--search-g", type=int, default=0, help=">0: samples per search question on the multi-turn path (the original search GRPO used 12; the multi-turn runs used --reason-g for both kinds); 0 = --reason-g")
ap.add_argument("--cot-on-fail", type=float, default=0.0, help=">0: a reasoning group none of whose samples passes takes one supervised step on the problem's own reference thinking and answer (Dolphin's CoT), at this weight - the teacher shows the way where the model found none")
ap.add_argument("--r1-on-fail", type=int, default=0, help="1 (with --demo-on-fail): a search group none of whose samples passes, with no teacher trajectory on file, has R1 solve the question now in the student's environment (r1_traj.py, up to 5 searches, kept at <= 3); a verified trajectory is shown at once (as --demo-on-fail) and kept for the run. Follow-ups written on the fly get theirs this way")
ap.add_argument("--loop-break", default="", choices=["", "stop", "answer"], help="a thinking span whose last 256 tokens are under 25%% distinct is a repetition loop (g14 step 400: 22 of 100 held-out replies never finished, tail repetition 0.84). stop: end the row there; answer: close the thinking and let it answer")
ap.add_argument("--rft", type=int, default=0, help="1: rejection-sampling fine-tuning instead of the policy gradient on reasoning steps: of the G samples the teacher passes, the one with the shortest thinking is trained on as plain SFT; none passing falls back to --wheels. No advantage, no std, no length pressure.")
ap.add_argument("--sft-only", type=int, default=0, help=">0: pure distillation, no rollouts and no judge: each step trains this many reasoning records (their R1 thinking and reply) as plain SFT, plus one verified search trace at --rft-replay weight")
ap.add_argument("--wheel-max-think", type=int, default=0, help=">0: the R1 reference is trained on only when its thinking is at most this many words (the dolphin_v1 references run 718 words at the median; g12 took on their length: thinking 174 -> 612 words and 28%% unfinished by step 116)")
ap.add_argument("--rft-replay", type=float, default=0.5, help="in --rft mode, one verified search trace (from --replay) is trained on every step at this weight, so the search side is rehearsed while the reasoning side learns")
ap.add_argument("--reason-verify", default="judge", choices=["judge", "numeric"], help="numeric: the reasoning reward is exact agreement of the final number with the reference (GSM8K-style), no teacher; judge: the teacher model's four boxes")
ap.add_argument("--kl", type=float, default=0.0, help=">0: GRPO's anchor to the reference policy, the base with the adapters disabled (k3 estimator per policy token, this coefficient). g7 and g8 had no anchor at all: the reasoning reward favours longer thinking, and with nothing holding the policy near the base that length leaked into the search side within 200 steps on all layers")
ap.add_argument("--reason-every", type=int, default=0, help=">0: a reasoning step every N steps and a search step otherwise (3 search : 1 reasoning at 4); overrides --search-every")
ap.add_argument("--dolphin-on", default="all", choices=["all", "reason"], help="reason: the Dolphin SFT record rides only on reasoning steps, so the search steps carry search signal alone")
ap.add_argument("--pool-order", default="loop", choices=["loop", "pool3"], help="pool3: walk the search questions in the order the search GRPO (grpo_pool) walked them: shuffle the whole gold<=6 pool with seed 0, then drop the held-out")
ap.add_argument("--pool-offset", type=int, default=0, help="the pool index the first search step of the run maps to (pool3 stopped at 200)")
ap.add_argument("--search-lr", type=float, default=0.0, help=">0: the search side gets its own Adam at this rate (the pool3 search GRPO ran 1e-5), the reasoning side and the Dolphin SFT keep --lr. Implies --accum 1.")
ap.add_argument("--search-temp", type=float, default=0.0, help=">0: sampling temperature for the search rollouts (pool3: 0.9); 0 = --temp")
ap.add_argument("--search-gen", type=int, default=0, help=">0: generation cap for the search rollouts (pool3: 1500); 0 = --gen")
ap.add_argument("--pg-norm-len", type=int, default=1024, help="the fixed divisor for --pg-norm const")
ap.add_argument("--adv-std", type=int, default=1, help="1: advantage = (r - mean) / std within the group. 0: r - mean only, so an all-wrong group with one slightly-less-wrong rollout does not get blown up into a +3 sigma push toward that rollout.")
ap.add_argument("--accum", type=int, default=1, help="steps whose gradients are accumulated before one optimizer update (both the search-side and the Dolphin part)")
ap.add_argument("--judge-correct", type=int, default=0, help="1: a search reply the gold string does not match is also shown to the teacher with the gold; when it gives the same answer in another form (Claire for Claire Casey, Jamie Dornan for James \"Jamie\" Dornan) it counts as correct - all or nothing, no partial credit. The evaluation keeps the strict string match.")
ap.add_argument("--samepage", type=int, default=1, help="1: a search whose top page was already shown in this rollout serves the NEXT chunk of that page (and says so when the page is used up); 0: teacher environment (always the head)")
A = ap.parse_args()
os.makedirs(A.outdir, exist_ok=True)
os.environ["SP_RANK"] = str(A.lora_rank); os.environ.setdefault("SP_NOSYS", "1"); os.environ.setdefault("SP_EPISODIC", "1")
os.environ["SP_TRAIN_POOLER"] = "0"           # the pooler is wired up below, not by the harness
os.environ["SP_LR"] = str(A.lr); os.environ["SP_POOLER_LR"] = str(A.pooler_lr)
LAYERS = None if A.lora_layers == "all" else list(range(int(A.lora_layers.split("-")[0]), int(A.lora_layers.split("-")[1]) + 1))
# resume decides which weights the harness prefix loads
STATE_F = os.path.join(A.outdir, "state.json"); LATEST = os.path.join(A.outdir, "latest.safetensors")
state = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {"step": 0}
init_path = LATEST if (state["step"] > 0 and os.path.exists(LATEST)) else A.init
if os.path.isdir(A.init):                     # merged HF model dir: the weights ARE the base
    os.environ["SP_BASE"] = A.init            # also under a resume: whatever the checkpoint does not cover must come from
                                              # this base, never from the stock model the harness would otherwise fetch
if not os.path.isdir(init_path):
    os.environ["SP_INIT_FULL"] = init_path    # the run's own checkpoint, overlaid on that base
import torch  # noqa: E402
import numpy as np  # noqa: E402

# ---- model + pooler + optimizer + pg_grad_backward via the harness prefix (same wrapping as the evaluator) ----
F = "/root/work/grpo_e2e_torch.py"
src = open(F).read().split("\n")
cut = next(i for i, l in enumerate(src) if l.startswith("if MULTI > 1:"))
sys.argv = ["grpo_e2e_torch.py", "0", str(A.g), "0", str(A.gen)]
sys.path.insert(0, "/root/work")
ns = {"__name__": "grpo_pool", "__file__": F, "_SP_LAYERS": LAYERS}
prefix = "\n".join(src[:cut]).replace(
    'target_modules=TARGETS, bias="none", task_type="CAUSAL_LM")',
    'target_modules=TARGETS, bias="none", task_type="CAUSAL_LM", layers_to_transform=_SP_LAYERS)', 1)
exec(compile(prefix, F, "exec"), ns)
model, tok, pooler = ns["model"], ns["tok"], ns["pooler"]
emb, sp, crop_cache, pick, _ngrams = ns["emb"], ns["sp"], ns["crop_cache"], ns["pick"], ns["_ngrams"]
pg_grad_backward, clear = ns["pg_grad_backward"], ns["clear"]
DEV, eos = ns["DEV"], ns["eos"]
ns["TEMP"] = A.temp; ns["MAXD"] = A.maxd; ns["C"] = A.chunk; ns["RWG"] = A.rw; ns["GREEDY"] = False
from transformers import DynamicCache  # noqa: E402
from safetensors.torch import save_file, load_file  # noqa: E402
_lb = sum(float(p.detach().float().norm()) ** 2 for n, p in model.named_parameters() if "lora_B" in n) ** 0.5
print(f"[init] weights <- {init_path} (resume step {state['step']}) lora r={A.lora_rank} layers={A.lora_layers} base={os.environ.get('SP_BASE')} lora_B_norm={_lb:.3f}", flush=True)
if os.environ.get("SP_INIT_FULL"):
    # a checkpoint only resumes under the LoRA layout it was saved with: with any other layout the harness loads what
    # matches and silently leaves the rest at the base (g6 ran layers 0-19 of the stock model for hours this way)
    from safetensors import safe_open as _so
    _ck = [k for k in _so(os.environ["SP_INIT_FULL"], "pt").keys() if not k.startswith("pooler.")]
    _mk = set(model.state_dict().keys()); _bad = [k for k in _ck if k not in _mk]
    if _bad:
        print(f"ONLINE_ABORT layout mismatch: {len(_bad)} of {len(_ck)} checkpoint tensors have no place under lora layers={A.lora_layers} (e.g. {_bad[0]}); "
              f"resume under the layout the checkpoint was saved with, or merge it into a base first", flush=True)
        sys.exit(1)
if A.pooler_init and os.path.isdir(init_path) and os.path.exists(A.pooler_init):   # resuming a .safetensors already carries its pooler
    from safetensors.torch import load_file as _lf
    n = pooler.load_sd(_lf(A.pooler_init)); print(f"[init] pooler <- {A.pooler_init} ({n} tensors)", flush=True)

if os.environ.get("SP_EMPTY_FROM"):
    # The 32 vectors the pooler emits for an EMPTY past are a constant every prompt starts with, long before anything is
    # compressed; take them from the float pooler (SP_EMPTY_FROM), so a quantized pooler only acts once there is a past.
    from safetensors.torch import load_file as _lfe
    _cur = {k: v.detach().clone() for k, v in pooler.A.items()}
    _fl = {(k[len("pooler."):] if k.startswith("pooler.") else k): v for k, v in _lfe(os.environ["SP_EMPTY_FROM"]).items()
           if not k.startswith(("model.", "lm_head", "base_model"))}
    pooler.load_sd(_fl)
    with torch.no_grad():
        _E = sp([]).detach().clone()
    pooler.load_sd(_cur)
    _sp_inner = sp

    def sp(kept, _E=_E, _inner=_sp_inner):
        return _E if len(kept) == 0 else _inner(kept)
    print(f"[load] empty-past soft prompt from {os.environ['SP_EMPTY_FROM']} ({tuple(_E.shape)})", flush=True)


class PoolerAdapter:
    """Keeps the pooler's weights frozen and adds a small trainable part, addressed by the same keys.

    ln   : the query vectors, every layernorm and out_scale become parameters (~0.1% of the pooler)
    lora : that, plus W + scale*(B@A) on each 2-D matrix (attention projections and the FFN)
    Reading self.A[k] returns the effective tensor, so the pooler's forward is untouched, and
    items() yields merged weights so a checkpoint stays loadable by the plain pooler."""

    def __init__(self, base, mode, rank, scale):
        self.frozen, self.param, self.lo, self.scale = {}, {}, {}, scale
        for k, v in base.items():
            small = (v.ndim <= 1) or k == "query"
            if small:
                self.param[k] = torch.nn.Parameter(v.detach().clone())
            elif mode == "lora":
                self.frozen[k] = v.detach()
                a = torch.nn.Parameter(torch.randn(rank, v.shape[1], device=v.device, dtype=v.dtype) * 0.01)
                b = torch.nn.Parameter(torch.zeros(v.shape[0], rank, device=v.device, dtype=v.dtype))
                self.lo[k] = (a, b)
            else:
                self.frozen[k] = v.detach()

    def __getitem__(self, k):
        if k in self.param: return self.param[k]
        v = self.frozen[k]
        ab = self.lo.get(k)
        return v + self.scale * (ab[1] @ ab[0]) if ab else v

    def __contains__(self, k): return k in self.param or k in self.frozen
    def keys(self): return list(self.param) + list(self.frozen)
    def values(self): return [self[k] for k in self.keys()]
    def items(self): return [(k, self[k]) for k in self.keys()]        # merged: plain-pooler compatible
    def trainable(self): return list(self.param.values()) + [p for ab in self.lo.values() for p in ab]


if A.pooler == "none":
    pooler_params = []
else:
    pooler.A = PoolerAdapter(pooler.A, A.pooler, A.pooler_rank, A.pooler_scale)
    pooler_params = pooler.A.trainable()
# the pooler params ride with the search optimizer when there is one: the pooler is search machinery, and the search
# GRPO trained it (rank-8 adapter, 1e-5) from the search rollouts alone
opt = torch.optim.Adam(
    [{"params": [p for p in model.parameters() if p.requires_grad], "lr": A.lr}]
    + ([{"params": pooler_params, "lr": A.pooler_lr}] if pooler_params and A.search_lr <= 0 else []))
opt_s = None
if A.search_lr > 0:
    # the search side steps its own optimizer, with its own moments, at the rate the search GRPO used;
    # each loop step then steps exactly one optimizer, so accumulation across steps is off
    opt_s = torch.optim.Adam([{"params": [p for p in model.parameters() if p.requires_grad], "lr": A.search_lr}]
                             + ([{"params": pooler_params, "lr": A.pooler_lr}] if pooler_params else []))
    A.accum = 1
BASE_TEMP, BASE_GEN = A.temp, A.gen
OPT_F = os.path.join(A.outdir, "opt.pt")
def save_opt():
    """Adam's moments next to the weights: a resume that rebuilds them from zero takes full-size sign steps for a while
    (g7's two dips, at 121-140 and 212-230, both followed a resume)"""
    torch.save({"opt": opt.state_dict(), "opt_s": opt_s.state_dict() if opt_s is not None else None}, OPT_F + ".tmp"); os.replace(OPT_F + ".tmp", OPT_F)
if state["step"] > 0 and os.path.exists(OPT_F):
    try:
        _o = torch.load(OPT_F, map_location=DEV)
        opt.load_state_dict(_o["opt"])
        if opt_s is not None and _o.get("opt_s") is not None: opt_s.load_state_dict(_o["opt_s"])
        for g in opt.param_groups: g["lr"] = A.lr            # the rates always come from the flags, not from the file
        if opt_s is not None:
            for g in opt_s.param_groups: g["lr"] = A.search_lr
        print(f"[init] optimizer moments <- {OPT_F}", flush=True)
    except Exception as e:
        print(f"[init] optimizer moments not restored ({type(e).__name__}: {str(e)[:120]}); starting them fresh", flush=True)
CLM = model.base_model.model            # peft -> causal LM
BODY, HEAD = CLM.model, CLM.lm_head     # transformer body, lm_head
if A.gradckpt:
    CLM.config.use_cache = False
    CLM.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("[init] gradient checkpointing ON for the policy-gradient pass", flush=True)
nT = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
nP = sum(p.numel() for p in pooler_params) / 1e6
print(f"[cfg] G={A.g} sft_only={A.sft_only} rft={A.rft}/{A.rft_replay} wheel_max_think={A.wheel_max_think} reason_verify={A.reason_verify} kl={A.kl} reason_every={A.reason_every} dolphin_on={A.dolphin_on} pool_order={A.pool_order}+{A.pool_offset} pg_norm={A.pg_norm}/{A.pg_norm_len} adv_std={A.adv_std} search_lr={A.search_lr} search_temp={A.search_temp} search_gen={A.search_gen} accum={A.accum} steps={A.steps} budget={A.budget} rw={A.rw} maxd={A.maxd} chunk={A.chunk} temp={A.temp} gen={A.gen} maxs={A.maxs} maxm={A.maxm} samepage={A.samepage} maxsrch={A.maxsrch} phantom={A.phantom}x{A.phantom_scale} "
      f"lr={A.lr} pooler_lr={A.pooler_lr} pooler={A.pooler}(r={A.pooler_rank}) trainable lora={nT:.1f}M pooler={nP:.2f}M", flush=True)
# ---- environment: verbatim grpo_ep_more serve() ----
WAPI = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "deep-charger-grpo-ep/1.0 (research; bayamax@icloud.com)"}
CTX = ssl.create_default_context()
try:
    import certifi; CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    pass
PAGE_STEP = 256
NOTICE = "(no searches left - answer from what you have read)"
NOMORE = "(no more of this page - search again or answer)"
EXHAUSTED = "(this page is used up - search a different query or answer)"
cache = {}
CACHE_F = "/root/work/pool_eval_cache_local.jsonl" if os.environ.get("SP_LOCAL_STORE") else "/root/work/pool_eval_cache.jsonl"   # local pages are not Wikipedia pages
if os.path.exists(CACHE_F):
    for line in open(CACHE_F):
        try:
            d = json.loads(line); cache[d["kw"]] = d["page"]
        except Exception:
            pass
cache_fh = open(CACHE_F, "a")


def api(params, tries=3):
    url = WAPI + "?" + urllib.parse.urlencode({**params, "maxlag": 5, "format": "json"})
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=20, context=CTX) as resp:
                out = json.loads(resp.read().decode())
            if isinstance(out, dict) and out.get("error", {}).get("code") == "maxlag":
                time.sleep(5 * (i + 1)); continue
            time.sleep(0.6)
            return out
        except Exception:
            time.sleep(3)
    return {}


def fetch(kw):
    sd_ = api({"action": "query", "list": "search", "srsearch": kw, "srlimit": 3})
    hits = [h["title"] for h in sd_.get("query", {}).get("search", [])][:3]
    if not hits:
        return ""
    d = api({"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
             "exlimit": "max", "redirects": 1, "titles": "|".join(hits)})
    pages = {}
    for p in d.get("query", {}).get("pages", {}).values():
        t, ex = p.get("title", ""), (p.get("extract", "") or "")
        if ex:
            pages[t] = ex[:40000]
    for t in hits:
        if t in pages:
            full = api({"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "titles": t})
            for p in full.get("query", {}).get("pages", {}).values():
                if p.get("extract"):
                    return f"{t}: {p['extract'][:40000]}"
            return f"{t}: {pages[t]}"
    return ""


_page_guard = threading.Lock()
_page_locks = {}


if os.environ.get("SP_LOCAL_STORE"):
    # the on-device search instead of Wikipedia's: the same "Title: text" the loop serves, from the local store
    sys.path.insert(0, "/root/work/localsearch")
    from search import LocalSearch as _LocalSearch  # noqa: E402
    _LS = _LocalSearch(os.environ["SP_LOCAL_STORE"], os.environ.get("SP_LOCAL_MODEL", "/root/bge-small"))
    print(f"[search] local store {os.environ['SP_LOCAL_STORE']} ({_LS.st.n_docs} articles)", flush=True)

    def fetch(kw):
        return _LS.fetch(kw)


def get_page(kw):
    """Thread-safe: rows of a batched rollout look pages up concurrently, and two rows asking for the same
    keyword at the same moment must still cost one fetch, not two."""
    if kw in cache:
        return cache[kw]
    with _page_guard:
        lk = _page_locks.setdefault(kw, threading.Lock())
    with lk:
        if kw in cache:
            return cache[kw]
        page = fetch(kw)
        with _page_guard:
            cache[kw] = page
            cache_fh.write(json.dumps({"kw": kw, "page": page}, ensure_ascii=False) + "\n"); cache_fh.flush()
    return page


def norm(s): return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()
def has(t, g): return (" " + norm(g) + " ") in (" " + norm(t) + " ")
ABBR = {"st", "mr", "mrs", "ms", "dr", "jr", "sr", "mt", "vs", "no", "inc", "ltd", "co"}


def head_sentence(a):
    h = (a or "").strip().split("\n")[0]
    for m in re.finditer(r"[.!?](?=\s|$)", h):
        t = re.split(r"[\s(\"]", h[:m.start()])[-1]
        if len(t) == 1 or t.lower() in ABBR:
            continue
        return h[:m.end()][:120]
    return h[:120]


def answer_complete(seg):
    m = re.search(r"answer is (.+)", seg, re.I | re.S)
    if not m:
        return False
    for mm in re.finditer(r"[.!?](?=\s|$)", m.group(1)):
        t = re.split(r"[\s(\"]", m.group(1)[:mm.start()])[-1]
        if len(t) == 1 or t.lower() in ABBR:
            continue
        return True
    return False


CLOSE_RE = re.compile(r"<search>(.*?)</\s*search\s*[^\w<]{0,3}$", re.S)
MORE_RE = re.compile(r"<\s*/?\s*more\s*/?\s*>\s*$", re.I)


TAG = "<information"
GREEDY_PICK = False          # the equivalence self-test needs a deterministic policy; training never sets this


def _banned(lg, t, tail):
    cand = tail + tok.decode([t])
    return TAG in cand or any(cand.endswith(TAG[:k]) for k in range(4, len(TAG) + 1))


@torch.no_grad()
def pick_row(lg, gen):
    """verbatim grpo_ep_more.pick() for one row of logits; the policy may never write its own information block."""
    lg = lg.float().clone(); tail = tok.decode(gen[-16:]) if gen else ""
    for _ in range(8):
        t = int(torch.argmax(lg).item()) if GREEDY_PICK else int(torch.multinomial(torch.softmax(lg / A.temp, dim=-1), 1).item())
        if _banned(lg, t, tail):
            lg[t] = -1e9; continue
        return t
    return int(torch.argmax(lg).item())


@torch.no_grad()
def pick_plain(logits, gen):
    return pick_row(logits[0], gen)



@torch.no_grad()
def rollout(question):
    """pool_eval.rollout + a policy mask (1 = token the policy chose, 0 = injected information block)."""
    model.eval()
    q_ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": question}],
                                               add_generation_prompt=True, tokenize=False) + "<think>\n")
    past = DynamicCache()
    model(input_ids=torch.tensor([q_ids], device=DEV), past_key_values=past, use_cache=True)
    MQ = past.get_seq_length()
    gen, msk, kept, absorbed, segs = [], [], [], 0, []
    n_model, ns_, nm, nmt = 0, 0, 0, 0
    served, queries, page_ids, page_off = [], [], [], 0
    seen_pages, cur_key, nrep = {}, None, 0     # samepage: per-rollout read offset of every page shown so far
    t0 = time.time(); dead = False; cut = False

    def inject(text):
        ids = tok.encode(text, add_special_tokens=False)
        gen.extend(ids); msk.extend([0] * len(ids))

    while n_model < A.gen and time.time() - t0 < A.budget:
        c0 = len(gen); R = min(c0, A.rw); nd = c0 - R
        if nd > absorbed:
            kept.extend(gen[absorbed:nd]); absorbed = nd
            if len(kept) > A.maxd:
                _, mass = pooler.forward_with_mass(emb(kept).to(torch.float32))
                mm = mass[0].float().cpu().numpy(); kept = [kept[i] for i in np.sort(np.argsort(mm)[-A.maxd:])]
        spv = sp(kept); segs.append([c0, None, list(kept)])     # what the policy saw for this block: pooled set + raw window gen[c0-R:c0]
        parts = [spv] + ([emb(gen[c0 - R:c0])] if R > 0 else []); block = torch.cat(parts, dim=1)
        crop_cache(past, MQ)
        Lb = block.shape[1]; pos = torch.arange(MQ, MQ + Lb, device=DEV)
        out = model(inputs_embeds=block, past_key_values=past, attention_mask=torch.ones(1, MQ + Lb, device=DEV),
                    position_ids=pos.unsqueeze(0), cache_position=pos, use_cache=True)
        last = out.logits[:, -1, :]; npos = MQ + Lb
        brk = False
        for _ in range(A.chunk):
            nx = pick_plain(last, gen)
            if nx == eos:
                brk = True; break
            gen.append(nx); msk.append(1); n_model += 1
            if len(gen) >= 8 and len(set(gen[-8:])) == 1:
                dead = True; brk = True; break
            txt = tok.decode(gen)
            si = txt.rfind("<search>")
            mclose = CLOSE_RE.search(txt, si) if si >= 0 else None
            if mclose and txt.count("<search>") > ns_:
                ns_ += 1
                body = mclose.group(1).strip()
                kw, ask = ([x.strip() for x in body.split("||", 1)] if "||" in body else (body, body))
                queries.append(kw)
                if A.maxsrch and ns_ >= A.maxsrch:
                    cut = True; brk = True; break           # loop guard: the rollout ends here (unlanded)
                if not kw:
                    blk = "\n<information>(no results)</information>\n"
                elif ns_ > A.maxs:
                    blk = f"\n<information>{NOTICE}</information>\n"
                else:
                    pg = get_page(kw)
                    if not pg:
                        chunk, page_ids, page_off, cur_key = "(no results)", [], 0, None
                    else:
                        page_ids = tok.encode(pg, add_special_tokens=False); key = pg[:120]
                        if A.samepage and key in seen_pages:
                            page_off = seen_pages[key]; nrep += 1
                            nxt = page_ids[page_off:page_off + PAGE_STEP]
                            chunk = tok.decode(nxt) if nxt else None; page_off += len(nxt)
                        else:
                            chunk = tok.decode(page_ids[:PAGE_STEP]); page_off = PAGE_STEP
                        seen_pages[key] = page_off; cur_key = key
                    if chunk is None:
                        blk = f"\n<information>{EXHAUSTED}</information>\n"
                    else:
                        served.append(chunk)
                        blk = f"\n<information>\n{chunk}\n[READER] (no extraction)\n</information>\n"
                inject(blk); brk = True; break
            if MORE_RE.search(txt) and len(re.findall(r"<\s*/?\s*more\s*/?\s*>", txt, re.I)) > nmt:
                nmt += 1
                nxt = page_ids[page_off:page_off + PAGE_STEP] if nm < A.maxm else []
                if not nxt:
                    blk = f"\n<information>{NOMORE}</information>\n"
                else:
                    nm += 1; page_off += len(nxt); chunk = tok.decode(nxt); served.append(chunk)
                    if cur_key is not None: seen_pages[cur_key] = page_off
                    blk = f"\n<information>\n{chunk}\n[READER] (no extraction)\n</information>\n"
                inject(blk); brk = True; break
            if "</think>" in txt and answer_complete(txt.split("</think>")[-1]):
                brk = True; break
            out = model(inputs_embeds=emb([nx]), past_key_values=past, attention_mask=torch.ones(1, npos + 1, device=DEV),
                        position_ids=torch.tensor([[npos]], device=DEV), cache_position=torch.tensor([npos], device=DEV), use_cache=True)
            npos += 1; last = out.logits[:, -1, :]
        segs[-1][1] = len(gen)
        txt = tok.decode(gen)
        if dead or cut or (brk and gen and gen[-1] == eos):
            break
        if "</think>" in txt and answer_complete(txt.split("</think>")[-1]):
            break
    txt = tok.decode(gen)
    landed = (not cut) and "</think>" in txt and bool(txt.split("</think>")[-1].strip())
    ans = head_sentence(txt.split("</think>")[-1].strip()) if landed else ""
    return dict(q_ids=q_ids, gen=gen, msk=msk, segs=[x for x in segs if x[1] is not None], text=txt, answer=ans, ns=ns_, more=nm,
                rep=nrep, cut=cut, served=served, queries=queries, landed=landed, dead=dead)


def last_logits(**kw):
    """Forward through the transformer body and run the lm_head on the final position only.

    Going through the causal-LM wrapper computes logits for every position and casts them to float32: at batch 48
    that is a 9 GiB allocation for one row of numbers we actually read. peft hides the wrapper's signature, so
    asking transformers to keep only the last logits is not reliable across versions -- calling the body and the
    head separately is."""
    h = BODY(**kw).last_hidden_state[:, -1:, :]
    return HEAD(h)[:, -1, :]


EOS_IDS = tok.encode("<｜end▁of▁sentence｜>", add_special_tokens=False)


def seed_from_hist(hist):
    """earlier exchanges given as chat messages (an items file): the stream they would have been, nothing absorbed yet"""
    ids = tok.encode(tok.apply_chat_template(list(hist), tokenize=False), add_special_tokens=False)
    if ids and ids[0] == tok.bos_token_id: ids = ids[1:]
    return {"gen": ids, "kept": []}


def seed_from_roll(roll):
    """the stream after a rollout, as pool_eval's win mode carries it to the next turn: the pooler's kept tokens as they
    stand, the unabsorbed tail (plus the end-of-turn token) raw; a first turn's question, pinned only, goes into the stream"""
    q_part = roll["q_ids"][1:] if roll["q_ids"] and roll["q_ids"][0] == tok.bos_token_id else list(roll["q_ids"])
    wkept, wgen = list(roll.get("kept") or []), list(roll["gen"][roll.get("absorbed", 0):]) + EOS_IDS
    if roll.get("g0", 0) == 0:
        if roll.get("absorbed", 0): wkept = q_part + wkept
        else: wgen = q_part + wgen
    return {"gen": wgen, "kept": wkept}


@torch.no_grad()
def rollout_batch(question, B, seed=None):
    """B independent rollouts decoded in lockstep: one question for all rows (str), or one question per row (list).
    seed (multi-turn, the win scheme the app runs): {gen, kept} - the conversation so far as one stream; its newest --rw
    tokens start in the raw window, the rest is absorbed by the pooler, and only this question is pinned. Memory is then
    bounded whatever the history (question + SP + raw window + block), exactly as pool_eval --mt-mode win measures it.

    Semantics match rollout() exactly: every row keeps its own generation, pooled set, page offsets and stop
    conditions, and each block is rebuilt from that row's own state against a freshly prefilled question cache.
    Only the per-token decode is shared -- that is the part that leaves the GPU at ~14% utilisation when the
    rollouts are run one at a time. Rows that stop early are carried along with filler tokens; the cache is
    rebuilt from scratch every block, so their pollution never reaches a row that is still generating."""
    model.eval()
    qs = list(question) if isinstance(question, (list, tuple)) else [question] * B
    B = len(qs)
    QID = [tok.encode(tok.apply_chat_template([{"role": "user", "content": q}] if seed else chat_msgs(q), add_generation_prompt=True, tokenize=False) + "<think>\n")
           for q in qs]
    seed_gen = []
    if seed:   # the stream so far, then this question as the chat renders it (the same header the pinned prompt holds)
        hdr = QID[0][1:] if QID[0] and QID[0][0] == tok.bos_token_id else list(QID[0])
        seed_gen = list(seed["gen"]) + hdr
    MQs = [len(x) for x in QID]; MQ = max(MQs)                       # MQ = cache slots of the question prefix (left-padded)
    PAD = tok.pad_token_id if tok.pad_token_id is not None else eos
    q_in = torch.full((B, MQ), PAD, dtype=torch.long, device=DEV); q_am = torch.zeros(B, MQ, device=DEV)
    q_pos = torch.zeros(B, MQ, dtype=torch.long, device=DEV)
    for b, ids in enumerate(QID):                                    # every row's own tokens end on slot MQ-1, positions 0..len-1
        q_in[b, MQ - len(ids):] = torch.tensor(ids, device=DEV); q_am[b, MQ - len(ids):] = 1
        q_pos[b, MQ - len(ids):] = torch.arange(len(ids), device=DEV)
    S = [dict(gen=list(seed_gen), msk=[0] * len(seed_gen), kept=list(seed["kept"]) if seed else [], absorbed=0, g0=len(seed_gen),
              segs=[], n_model=0, ns_=0, nm=0, nmt=0, served=[], queries=[],
              page_ids=[], page_off=0, seen_pages={}, cur_key=None, nrep=0, dead=False, cut=False, done=False)
         for _ in range(B)]
    # Wall clock, not work: the rows run together, so the batch needs about what one rollout needed. 600 s was
    # nonetheless too tight while the page lookups were serial, and cutting rows off mid answer fed the gradient
    # failures the policy had not caused. 900 s with the lookups overlapped leaves room without hiding a stall.
    budget = A.budget
    t0 = time.time()

    def inject(st, text):
        ids = tok.encode(text, add_special_tokens=False)
        st["gen"].extend(ids); st["msk"].extend([0] * len(ids))

    def advance(st, nx):
        """one decoded token for one row; returns True when this row's block ends here (verbatim rollout() order)"""
        if nx == eos:
            if A.stop == "eos": st["ended"] = True      # the conversational lineage ends its reply here
            return True
        st["gen"].append(nx); st["msk"].append(1); st["n_model"] += 1
        gen = st["gen"]
        if len(gen) >= 8 and len(set(gen[-8:])) == 1:
            st["dead"] = True; return True
        txt = tok.decode(gen[st["g0"]:])
        if A.loop_break and st["n_model"] % 64 == 0 and len(gen) - st["g0"] >= 256 and "</think>" not in txt and len(set(gen[-256:])) < 64:
            st["loops"] = st.get("loops", 0) + 1
            if A.loop_break == "stop" or st["loops"] > 1:
                st["dead"] = True; return True
            inject(st, "\n</think>\n\n"); return True
        si = txt.rfind("<search>")
        mclose = CLOSE_RE.search(txt, si) if si >= 0 else None
        if mclose and txt.count("<search>") > st["ns_"]:
            st["ns_"] += 1
            body = mclose.group(1).strip()
            kw, ask = ([x.strip() for x in body.split("||", 1)] if "||" in body else (body, body))
            st["queries"].append(kw)
            if A.maxsrch and st["ns_"] >= A.maxsrch:
                st["cut"] = True; return True
            if not kw:
                blk = "\n<information>(no results)</information>\n"
            elif st["ns_"] > A.maxs:
                blk = f"\n<information>{NOTICE}</information>\n"
            else:
                pg = get_page(kw)
                if not pg:
                    chunk, st["page_ids"], st["page_off"], st["cur_key"] = "(no results)", [], 0, None
                else:
                    st["page_ids"] = tok.encode(pg, add_special_tokens=False); key = pg[:120]
                    if A.samepage and key in st["seen_pages"]:
                        st["page_off"] = st["seen_pages"][key]; st["nrep"] += 1
                        nxt = st["page_ids"][st["page_off"]:st["page_off"] + PAGE_STEP]
                        chunk = tok.decode(nxt) if nxt else None; st["page_off"] += len(nxt)
                    else:
                        chunk = tok.decode(st["page_ids"][:PAGE_STEP]); st["page_off"] = PAGE_STEP
                    st["seen_pages"][key] = st["page_off"]; st["cur_key"] = key
                if chunk is None:
                    blk = f"\n<information>{EXHAUSTED}</information>\n"
                else:
                    st["served"].append(chunk)
                    blk = f"\n<information>\n{chunk}\n[READER] (no extraction)\n</information>\n"
            inject(st, blk); return True
        if MORE_RE.search(txt) and len(re.findall(r"<\s*/?\s*more\s*/?\s*>", txt, re.I)) > st["nmt"]:
            st["nmt"] += 1
            nxt = st["page_ids"][st["page_off"]:st["page_off"] + PAGE_STEP] if st["nm"] < A.maxm else []
            if not nxt:
                blk = f"\n<information>{NOMORE}</information>\n"
            else:
                st["nm"] += 1; st["page_off"] += len(nxt); chunk = tok.decode(nxt); st["served"].append(chunk)
                if st["cur_key"] is not None: st["seen_pages"][st["cur_key"]] = st["page_off"]
                blk = f"\n<information>\n{chunk}\n[READER] (no extraction)\n</information>\n"
            inject(st, blk); return True
        if A.stop == "answer" and "</think>" in txt and answer_complete(txt.split("</think>")[-1]):
            return True
        return False

    while True:
        for st in S:
            if not st["done"] and (st["n_model"] >= A.gen or time.time() - t0 >= budget):
                st["done"] = True
        act = [b for b in range(B) if not S[b]["done"]]
        if not act:
            break
        blocks, Ls = [], []
        for b in act:                                  # one block per row, built exactly as rollout() builds it
            st = S[b]; gen = st["gen"]; c0 = len(gen); R = min(c0, A.rw); nd = c0 - R
            if nd > st["absorbed"]:
                st["kept"].extend(gen[st["absorbed"]:nd]); st["absorbed"] = nd
            if len(st["kept"]) > A.maxd:
                _, mass = pooler.forward_with_mass(emb(st["kept"]).to(torch.float32))
                mm = mass[0].float().cpu().numpy()
                st["kept"] = [st["kept"][i] for i in np.sort(np.argsort(mm)[-A.maxd:])]
            spv = sp(st["kept"]); st["segs"].append([c0, None, list(st["kept"])])
            parts = [spv] + ([emb(gen[c0 - R:c0])] if R > 0 else [])
            blk = torch.cat(parts, dim=1); blocks.append(blk); Ls.append(blk.shape[1])
        nA = len(act); Lmax = max(Ls); H = blocks[0].shape[-1]
        emb_b = torch.zeros(nA, Lmax, H, device=DEV, dtype=blocks[0].dtype)
        posv = torch.zeros(nA, Lmax, dtype=torch.long, device=DEV)
        amask = torch.zeros(nA, MQ + Lmax, device=DEV)
        amask[:, :MQ] = q_am[act]
        for i, blk in enumerate(blocks):               # left-pad: every row's last real token lands on index -1
            L = Ls[i]; mq = MQs[act[i]]
            emb_b[i, Lmax - L:] = blk[0]
            posv[i, Lmax - L:] = torch.arange(mq, mq + L, device=DEV)
            amask[i, MQ + Lmax - L:] = 1
        past = DynamicCache()
        BODY(input_ids=q_in[act], attention_mask=q_am[act], position_ids=q_pos[act], past_key_values=past, use_cache=True)
        cpos = torch.arange(MQ, MQ + Lmax, device=DEV)
        last = last_logits(inputs_embeds=emb_b, past_key_values=past, attention_mask=amask,
                           position_ids=posv, cache_position=cpos, use_cache=True)
        npos = [MQs[b] + L for b, L in zip(act, Ls)]
        alive = [True] * nA
        for _ in range(A.chunk):
            # sampling stays serial (it is microseconds of GPU work); advancing does not, because a row that has
            # just written a search tag blocks on Wikipedia and the other rows have no reason to wait for it
            toks = [pick_row(last[i], S[b]["gen"]) if alive[i] else None for i, b in enumerate(act)]
            live = [i for i in range(nA) if toks[i] is not None]
            if A.fetchers > 1 and len(live) > 1:
                with ThreadPoolExecutor(max_workers=min(A.fetchers, len(live))) as ex:
                    ended = list(ex.map(lambda i: advance(S[act[i]], toks[i]), live))
            else:
                ended = [advance(S[act[i]], toks[i]) for i in live]
            for j, i in enumerate(live):
                if ended[j]:
                    alive[i] = False
            step_tok = [t if t is not None else eos for t in toks]
            if not any(alive):
                break
            nxt_emb = torch.cat([emb([t]) for t in step_tok], dim=0)
            amask = torch.cat([amask, torch.ones(nA, 1, device=DEV)], dim=1)
            pid = torch.tensor([[p] for p in npos], device=DEV)
            cp = torch.tensor([amask.shape[1] - 1], device=DEV)
            last = last_logits(inputs_embeds=nxt_emb, past_key_values=past, attention_mask=amask,
                               position_ids=pid, cache_position=cp, use_cache=True)
            npos = [p + 1 for p in npos]
        for b in act:
            st = S[b]; st["segs"][-1][1] = len(st["gen"])
            txt = tok.decode(st["gen"][st["g0"]:])
            if st["dead"] or st["cut"] or st.get("ended") or (st["gen"] and st["gen"][-1] == eos):
                st["done"] = True
            elif A.stop == "answer" and "</think>" in txt and answer_complete(txt.split("</think>")[-1]):
                st["done"] = True
        del past, last; clear()

    cut_by_time = sum(1 for st in S if st["n_model"] < A.gen and not st["cut"] and not st["dead"]
                      and "</think>" not in tok.decode(st["gen"][st["g0"]:]))
    if cut_by_time:
        print(f"[warn] {cut_by_time}/{B} rollouts hit the {budget}s batch budget before answering", flush=True)
    outs = []
    for b, st in enumerate(S):
        txt = tok.decode(st["gen"][st["g0"]:])
        landed = (not st["cut"]) and "</think>" in txt and bool(txt.split("</think>")[-1].strip())
        ans = head_sentence(txt.split("</think>")[-1].strip()) if landed else ""
        outs.append(dict(q_ids=QID[b], q=qs[b], gen=st["gen"], msk=st["msk"], ended=bool(st.get("ended")), kept=st["kept"], absorbed=st["absorbed"], g0=st["g0"],
                         segs=[x for x in st["segs"] if x[1] is not None], text=txt, answer=ans,
                         ns=st["ns_"], more=st["nm"], rep=st["nrep"], cut=st["cut"], served=st["served"],
                         queries=st["queries"], landed=landed, dead=st["dead"]))
    return outs


def pg_backward(r, coef):
    """coef * (-mean log p over the policy's own tokens), backward. Each block is rebuilt EXACTLY as the policy saw it during
    the rollout: [query, SP(recorded pooled set), raw window, block tokens] -- same mass-ordered eviction, same boundaries."""
    qe = emb(r["q_ids"]); gen, msk = r["gen"], r["msk"]
    ntot = sum(msk)
    if ntot == 0: return 0.0
    tot = 0.0
    for c0, c1, kept in r["segs"]:
        if sum(msk[c0:c1]) == 0: continue
        R = min(c0, A.rw)
        spv = sp(kept) if kept else torch.zeros((1, 0, ns["H"]), device=DEV, dtype=ns["MDTYPE"])
        parts = [qe, spv] + ([emb(gen[c0 - R:c0])] if R > 0 else []) + [emb(gen[c0:c1])]
        block = torch.cat(parts, dim=1); L = block.shape[1]; cur = c1 - c0
        # lm_head only on the positions that predict this block's tokens (a full-block float32 logits tensor is ~800 MB)
        h = BODY(inputs_embeds=block, use_cache=False).last_hidden_state[:, L - cur - 1:L - 1, :]
        pr = HEAD(h).float()
        tgt = torch.tensor([gen[c0:c1]], device=DEV); tm = torch.tensor([msk[c0:c1]], device=DEV, dtype=torch.float32)
        ce = torch.nn.functional.cross_entropy(pr.reshape(-1, pr.shape[-1]), tgt.reshape(-1), reduction="none")
        denom = (A.pg_norm_len if A.pg_norm == "const" else ntot)
        loss = (ce * tm.reshape(-1)).sum() / denom
        if A.kl > 0:
            # the same block under the base policy (LoRA off; the pooled set is the rollout's own either way), then
            # k3 = exp(ref - cur) - (ref - cur) - 1 per policy token: zero at the base, always positive, gradient through cur only
            with torch.no_grad(), model.disable_adapter():
                hr = BODY(inputs_embeds=block, use_cache=False).last_hidden_state[:, L - cur - 1:L - 1, :]
                ref_ce = torch.nn.functional.cross_entropy(HEAD(hr).float().reshape(-1, pr.shape[-1]), tgt.reshape(-1), reduction="none")
                del hr
            d = ce - ref_ce                       # = logp_ref - logp_cur
            k3 = torch.exp(d) - d - 1.0
            loss = loss + A.kl * (k3 * tm.reshape(-1)).sum() / denom
        (coef * loss).backward()
        tot += float(loss.item()); del h, pr, ce, loss, block; clear()
    return tot


@torch.no_grad()
def logit_drift(question, N):
    """How far apart are the logits a row sees at batch 1 and at batch N, for the identical input?

    Token-level equality under a greedy policy cannot survive a change of batch size: the reduction order in the
    matmuls changes, the logits move in their last bits, and a near tie eventually flips. What matters is whether
    the sampling distribution moves, so measure that directly."""
    q_ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": question}],
                                               add_generation_prompt=True, tokenize=False) + "<think>\n")
    MQ = len(q_ids); block = sp([])
    def run(n):
        past = DynamicCache()
        BODY(input_ids=torch.tensor([q_ids] * n, device=DEV), past_key_values=past, use_cache=True)
        L = block.shape[1]
        pos = torch.arange(MQ, MQ + L, device=DEV)
        return last_logits(inputs_embeds=block.expand(n, -1, -1).contiguous(), past_key_values=past,
                           attention_mask=torch.ones(n, MQ + L, device=DEV),
                           position_ids=pos.unsqueeze(0).expand(n, -1), cache_position=pos, use_cache=True)[0].float()
    a, b = run(1), run(N)
    pa = torch.softmax(a / A.temp, -1); pb = torch.softmax(b / A.temp, -1)
    kl = float((pa * (pa.clamp_min(1e-12).log() - pb.clamp_min(1e-12).log())).sum())
    top = 200
    ia = a.topk(top).indices; ib = b.topk(top).indices
    return dict(maxabs=float((a - b).abs().max()), kl=kl, argmax_same=bool(ia[0] == ib[0]),
                top200_same=int((ia == ib).sum()), mass=float(pa[ia[0]]))


def choose_replay(rews, gnds, m):
    """Pick which of the group's rollouts the gradient replays, with a weight that keeps the sum unbiased.

    With a 0/1 reward the advantage is constant inside a reward stratum, so replaying a subset costs nothing in
    fidelity as long as each stratum keeps its share: weighting a drawn member by (stratum size / drawn) makes the
    subset sum an unbiased estimate of the whole group's."""
    n = len(rews)
    if m <= 0 or m >= n:
        return [(i, 1.0) for i in range(n)]
    by = {}
    for i, r in enumerate(rews):
        by.setdefault(round(r, 6), []).append(i)
    out = []
    for r, idxs in sorted(by.items()):
        k = max(1, min(len(idxs), int(round(m * len(idxs) / n))))
        if A.select == "grounded" and r <= 0:
            pool_g = [i for i in idxs if gnds[i]]                  # read the gold and still got it wrong
            pick = random.sample(pool_g, min(k, len(pool_g)))
            if len(pick) < k:
                rest = [i for i in idxs if i not in set(pick)]
                pick += random.sample(rest, k - len(pick))
        else:
            pick = random.sample(idxs, k)
        w = len(idxs) / len(pick)
        out.extend((i, w) for i in pick)
    return out


def price(r, gold):
    """teacher's price(): grounded-correct is 1.0, everything else 0.0."""
    correct = r["landed"] and has(r["answer"], gold)
    grounded = any(has(s, gold) for s in r["served"])
    return (1.0 if (correct and grounded) else 0.0), bool(correct), bool(grounded)


def save_ckpt(path):
    if A.save_lora_only:   # the base is the --init directory every loader already has; the LoRA and the pooler are what training changed
        sd = {n: p.detach().to(torch.bfloat16).cpu().contiguous() for n, p in model.named_parameters() if p.requires_grad}
    else:
        sd = {n: p.detach().to(torch.bfloat16).cpu().contiguous() for n, p in model.named_parameters()}   # full model (base + LoRA): loads standalone like the SFT ckpt
    sd.update({"pooler." + k: v.detach().float().cpu().contiguous() for k, v in pooler.A.items()})   # merged
    save_file(sd, path + ".tmp"); os.replace(path + ".tmp", path)


# ---- data ----
held = set()
for line in open(A.heldout):
    try: held.add((json.loads(line).get("q") or "").strip())
    except Exception: pass
pool = []; _allq = []
for line in open(A.questions):
    try: r = json.loads(line)
    except Exception: continue
    q, gold = (r.get("q") or "").strip(), (r.get("gold") or "").strip()
    if q and gold: _allq.append({"q": q, "gold": gold})
    if q and gold and q not in held: pool.append({"q": q, "gold": gold})
dol = []
for line in open(A.dolphin):
    try: r = json.loads(line)
    except Exception: continue
    if r.get("q") and r.get("thinking") and r.get("reply"): dol.append(r)
rep = []
if A.replay:
    for line in open(A.replay):
        try: r = json.loads(line)
        except Exception: continue
        if r.get("q") and r.get("text"): rep.append(r)
rng = random.Random(0)
if A.pool_order == "pool3":
    rng.shuffle(list(range(len(pool))))       # consume what the plain shuffle would have, so dol and rep keep their order
    random.Random(0).shuffle(_allq); pool = [x for x in _allq if x["q"] not in held]   # grpo_pool: shuffle first, then drop the held-out
else:
    rng.shuffle(pool)
rng.shuffle(dol); rng.shuffle(rep)
HIST = {}
if A.mt_items:
    pool = []
    for line in open(A.mt_items):
        try: r = json.loads(line)
        except Exception: continue
        q, gold = (r.get("q") or "").strip(), (r.get("gold") or "").strip()
        if q and gold and q not in held:
            pool.append({"q": q, "gold": gold}); HIST[q] = list(r.get("hist") or [])
    random.Random(1).shuffle(pool)
    print(f"[data] multi-turn: {len(pool)} search turns, {sum(1 for x in pool if HIST.get(x['q']))} with history in the prompt", flush=True)


def chat_msgs(q):
    """the prompt as the app sends it: the earlier exchanges of the conversation (if any), then the new question"""
    return HIST.get(q, []) + [{"role": "user", "content": q}]
FOLLOW = collections.deque()
FOLLOW_SYS = "You write the next thing a user says in a chat with an assistant. Output JSON only."
FOLLOW_USER = """The chat so far:
USER: {q}
ASSISTANT: {reply}

Write the user's natural follow-up question. It asks for ONE new fact closely related to what the assistant just said; it refers to things from the exchange only by a pronoun or a phrase like "that film" / "he" / "that city" (never by name); it is not answerable from the exchange itself. The answer must be a short name, term, date or number (1-4 words) that an English Wikipedia article states plainly.
Output: {{"q": "...", "gold": "<the answer>", "page": "<the title of the Wikipedia article that states it>"}} - or {{"skip": true}} if no good follow-up exists."""


def make_followup(step, q, gold, roll):
    """the teacher's follow-up to a passing exchange, verified on its page, queued for the next search step"""
    reply = roll["text"].split("</think>")[-1].strip()[:1500]
    v = ask_teacher(FOLLOW_SYS, FOLLOW_USER.format(q=q[:1500], reply=reply), "followup")
    fq, fg, pg = (v.get("q") or "").strip(), (v.get("gold") or "").strip(), (v.get("page") or "").strip()
    why = None
    if "error" in v or v.get("skip") or not fq or not fg or not pg: why = v.get("error") or "skip"
    elif len(fg.split()) > 4 or not any(c.isalnum() for c in fg): why = "gold shape"
    elif has(fq, fg) or has(q, fg) or has(reply, fg): why = "answer already in the exchange"
    elif has(fq, gold): why = "names the first answer"
    elif not has(get_page(pg) or "", fg): why = "answer not on the page"
    if why: print(f"[followup] not used ({why}): {fq[:80]!r} / {fg!r}", flush=True); return
    hist = [{"role": "user", "content": q}, {"role": "assistant", "content": reply}]
    HIST[fq] = hist; FOLLOW.append({"q": fq, "gold": fg, "seed": seed_from_roll(roll)})
    with open(os.path.join(A.outdir, "followups.jsonl"), "a") as f:
        f.write(json.dumps({"step": step, "q": q, "fq": fq, "gold": fg, "page": pg, "hist": hist}, ensure_ascii=False) + "\n")
    print(f"[followup] {fq[:90]!r} -> {fg!r} ({pg})", flush=True)


def r1_on_the_fly(q, gold, hist):
    """R1 on one question now: r1_traj.py as a subprocess; the trajectory (student format) or None"""
    import subprocess, tempfile
    d = A.outdir; fi = os.path.join(d, "r1_fly_in.jsonl"); fo = os.path.join(d, "r1_fly_out.jsonl")
    open(fi, "w").write(json.dumps({"q": q, "gold": gold, "hist": hist, "pass": 0}, ensure_ascii=False) + "\n")
    if os.path.exists(fo): os.remove(fo)
    t_ = time.time()
    try:
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "r1_traj.py"), "--probe", fi, "--out", fo,
                        "--max-pass", "0", "--workers", "1", "--maxsrch", "5", "--tok", os.environ.get("SP_BASE", "/root/gptq_hf_gq14")],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
    except Exception as e:
        print(f"[r1] failed: {type(e).__name__}", flush=True); return None
    rows = [json.loads(l) for l in open(fo)] if os.path.exists(fo) else []
    r = next((x for x in rows if x.get("traj")), None)
    ok = r is not None and r.get("ns", 9) <= 3
    print(f"[r1] {q[:80]!r}: {'verified, ' + str(r['ns']) + ' searches' if r else 'no verified trajectory'}{'' if ok or not r else ' (too many searches, not used)'} ({time.time()-t_:.0f} s)", flush=True)
    if ok:
        with open(os.path.join(d, "r1_onfly.jsonl"), "a") as f: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return r["traj"]
    return None


def pool_item(step):
    """the search question of this step: loop order indexes by step (odd slots), pool3 order counts search steps from --pool-offset;
    a follow-up written on the fly (--followup) comes first"""
    if FOLLOW: return FOLLOW.popleft()
    if A.pool_order == "pool3":
        n_search = (step - step // A.reason_every) if A.reason_every else step // max(A.search_every, 1)   # search steps so far, this one included
        return pool[(A.pool_offset + n_search - 1) % len(pool)]
    return pool[(step - 1) % len(pool)]
INFO_RE = re.compile(r"(<information>.*?</information>\n?)", re.S)
def ce_backward(h, tgt, tm, coef, slice_len=256):
    """cross-entropy through the lm_head without ever holding the whole logits tensor.

    A 4096-token record against a 151k vocabulary is 2.5 GB of float logits, and with the rollouts
    now running to 4000 tokens there is no longer room for that beside them: r10 died of it at step
    103. The body runs once; the head runs on one slice at a time, each slice's gradient lands on a
    detached copy of the hidden states, and one backward carries the sum into the body. Same
    gradient, a fraction of the memory."""
    hd = h.detach().requires_grad_(True)
    n = hd.shape[1]; tot = 0.0; denom = tm.sum().clamp(min=1.0)
    for a in range(0, n, slice_len):
        b = min(a + slice_len, n)
        if float(tm[:, a:b].sum()) == 0: continue
        pr = HEAD(hd[:, a:b, :]).float()
        ce = torch.nn.functional.cross_entropy(pr.reshape(-1, pr.shape[-1]), tgt[:, a:b].reshape(-1), reduction="none")
        part = (ce * tm[:, a:b].reshape(-1)).sum() / denom
        (coef * part).backward(retain_graph=True); tot += float(part.item())
        del pr, ce, part
    if hd.grad is not None: h.backward(gradient=hd.grad)
    del hd; clear(); return tot


def replay_backward(rec, coef):
    """a verified search trace as plain SFT: prompt masked, information blocks masked, EOS trained"""
    head_t = tok.apply_chat_template(list(rec.get("hist") or []) + [{"role": "user", "content": rec["q"]}], add_generation_prompt=True, tokenize=False)
    if not head_t.rstrip().endswith("<think>"): head_t += "<think>\n"
    ids = tok.encode(head_t, add_special_tokens=False); msk = [0] * len(ids)
    body = rec["text"]; body = body[len("<think>"):].lstrip("\n") if body.startswith("<think>") else body
    for piece in INFO_RE.split(body):
        if not piece: continue
        t = tok.encode(piece, add_special_tokens=False); ids += t; msk += [0 if piece.startswith("<information>") else 1] * len(t)
    ids.append(eos); msk.append(1); ids, msk = ids[:A.maxlen], msk[:A.maxlen]
    if sum(msk[1:]) == 0: return 0.0
    x = torch.tensor([ids], device=DEV)
    h = BODY(input_ids=x, use_cache=False).last_hidden_state[:, :-1, :]
    tgt = x[:, 1:]; tm = torch.tensor([msk[1:]], device=DEV, dtype=torch.float32)
    v = ce_backward(h, tgt, tm, coef); del h; clear(); return v
print(f"[data] {len(pool)} questions, {len(dol)} dolphin records, {len(rep)} replay traces", flush=True)

# ---- the cheap judge (key from the environment, never from the repo) ----
DSK = os.environ.get("DSK_KEY", "").strip()
OAI = os.environ.get("OAI_KEY", "").strip()
JUDGE_SYS = """You are a strict data-quality checker for a small assistant that searches Wikipedia and then replies in a conversational way. You get the user question, the reference answer, the assistant's whole private thinking (its searches and the results it read), and its reply. Return ONLY a JSON object:
{"commits_to_answer": true/false, "matches_reference": true/false, "unsupported_claims": ["..."], "language_english": true/false, "clean": true/false, "natural": 1-5}
where unsupported_claims lists every fact in the reply that is NOT in the results or the question (dates, numbers, names, titles, roles, characterizations; empty if none), clean means no tool tags, no leftover thinking, no repetition, no cut-off sentence, and natural is 5 when the reply reads like a knowledgeable friend answering the question and 1 when robotic or awkward."""
def judge(q, gold, text):
    if not DSK or not A.judge: return {"skip": True}
    reply = text.split("</think>")[-1].strip(); think = text.split("</think>")[0][-6000:]
    body = {"model": "deepseek-flash", "messages": [{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": f"QUESTION:\n{q}\n\nREFERENCE ANSWER:\n{gold}\n\nTHINKING (searches and results):\n{think}\n\nREPLY:\n{reply}"}], "max_tokens": 2000, "temperature": 0.0, "response_format": {"type": "json_object"}}
    for a in range(3):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request("https://api.deepseek.com/chat/completions", data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + DSK, "Content-Type": "application/json"}), timeout=120))
        except Exception: time.sleep(3); continue
        if "choices" not in d: return {"error": "no choices"}
        c = d["choices"][0]["message"].get("content") or ""
        try: return json.loads(c[c.find("{"): c.rfind("}") + 1])
        except Exception: return {"error": "unparsable"}
    return {"error": "network"}
REASON_SYS = """You are scoring one answer from a small assistant against a reference answer. Reply with JSON only:
{"solves_it": true/false, "follows_the_request": true/false, "language_english": true/false, "clean": true/false}
solves_it means the assistant reaches the same result as the reference (the wording may differ, and a different but equally valid result counts); follows_the_request means it does what was asked, including any count, length or format the question specifies; clean means no tool tags, no repetition loop and no text left unfinished."""


SEARCH_SYS = """You are scoring the reply of a small assistant that just searched an encyclopedia and answered. The reply is already known to contain the right answer; you are judging how it is written. Reply with JSON only:
{"sound": true/false, "natural": true/false, "clean": true/false}
sound means every claim in the reply is the kind of thing the search would have supported, with nothing obviously invented and nothing self-contradictory; natural means it reads as a person speaking, two or three sentences, not a template or a bare fragment; clean means no tool tags, no repetition loop, and nothing left unfinished."""


CORRECT_SYS = """You check whether a small assistant's reply gives the same answer as a reference answer. Reply with JSON only:
{"same_answer": true/false, "on_the_pages": true/false}
same_answer is true only when the reply commits to one answer and that answer is the same entity, date, number or fact as the reference: a shorter or longer form of the same name, another spelling, the same date or quantity written differently all count. A different person, place, title, year or number does not; an answer that hedges between candidates or does not commit does not; a reply that only repeats the question does not. on_the_pages is true when the search results shown state that answer in some form."""


MT_SYS = """
The question comes after an earlier exchange, shown for context. natural also requires that the reply answers the NEW question as asked: it does not answer or revisit the earlier question, and it does not drag the earlier topic in where the new question does not refer to it."""


def ask_teacher(system, user, tag):
    """one scored call to whichever teacher is configured; returns the parsed JSON or an error note"""
    oai = A.judge_api == "openai"
    key = OAI if oai else DSK
    if not key: return {"error": "no key"}
    model = A.judge_model or ("gpt-5-nano" if oai else "deepseek-flash")
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if model.startswith("gpt-5"): body["max_completion_tokens"] = 2000
    else: body["max_tokens"] = 2000; body["temperature"] = 0
    url = "https://api.openai.com/v1/chat/completions" if oai else "https://api.deepseek.com/chat/completions"
    for _ in range(3):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(
                url, data=json.dumps(body).encode(),
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}), timeout=180))
        except Exception:
            time.sleep(3); continue
        if "choices" not in d: return {"error": "no choices"}
        c = d["choices"][0]["message"].get("content") or ""
        try: return json.loads(c[c.find("{"): c.rfind("}") + 1])
        except Exception: return {"error": "unparsable"}
    return {"error": "network"}


# the reward the base model was trained under (grpo_e2e_torch.py, the needs-search branch), kept as it
# stands. Note what it does NOT do: it never charges for the number of searches. Reading the same page
# on and on costs nothing while the fact is still missing; what it charges for is carrying on searching
# AFTER the fact has been surfaced, at 0.1 a step and capped at five. On top of it, and only for a
# rollout that already earns the grounded-correct 1.5, the teacher adds up to W_TALK for a reply that
# reads like a person rather than a template.
W_ACC, W_GND, W_UNGND, W_WRONG, W_RETR, ATTEMPT_T = 1.0, 0.5, 1.0, 0.5, 0.4, 1.0
W_LATE, LATECAP = 0.1, 5


def score_search(r, gold):
    reply = r["text"].split("</think>")[-1].strip() if "</think>" in r["text"] else ""
    landed = bool(r["landed"] and reply)
    _, correct, grounded = price(r, gold)
    judged = False
    if A.judge_correct and not correct and landed and not any(t in reply for t in TAGS):
        h0 = HIST.get(r.get("q", ""), [])
        ctx0 = ("EARLIER IN THE CONVERSATION:\n" + "\n".join(("USER: " if m["role"] == "user" else "ASSISTANT: ") + m["content"][:600] for m in h0) + "\n\n") if h0 else ""
        v0 = ask_teacher(CORRECT_SYS, f"{ctx0}QUESTION:\n{r.get('q', '')}\n\nREFERENCE ANSWER:\n{gold}\n\nWHAT THE SEARCH RETURNED:\n" + "\n\n".join(r.get("served", []))[-4000:] + f"\n\nREPLY:\n{reply[:2000]}", "correct")
        if "error" not in v0 and v0.get("same_answer"):
            correct, judged = True, True; grounded = grounded or bool(v0.get("on_the_pages"))
    # which search first surfaced the fact, and how many came after it
    hit = next((i for i, sv in enumerate(r.get("served", [])) if has(sv, gold)), None)
    late = (len(r.get("served", [])) - 1 - hit) if hit is not None else 0
    info_hit = hit is not None
    if correct:
        base = (W_ACC + W_GND) if grounded else W_UNGND
    else:
        base = -W_WRONG + W_RETR * (1.0 if info_hit else 0.0) * (ATTEMPT_T if landed else 0.5)
    base -= W_LATE * min(late, LATECAP)
    why = {"correct": int(correct), "judged": int(judged), "grounded": int(grounded), "late": late,
           "unfinished": not landed, "tags": any(t in reply for t in TAGS)}
    if not (correct and grounded and landed) or why["tags"]:
        return base, why
    if A.reason_stub:
        return base + A.w_talk, {**why, "stub": True}
    served = "\n\n".join(r.get("served", []))[-4000:]
    h = HIST.get(r.get("q", ""), [])
    if h:   # multi-turn: the reply must answer the NEW message on its own terms, not carry the earlier topic over
        ctx = "\n".join(("USER: " if m["role"] == "user" else "ASSISTANT: ") + m["content"][:600] for m in h)
        v = ask_teacher(SEARCH_SYS + MT_SYS, f"EARLIER IN THE CONVERSATION:\n{ctx}\n\nNEW QUESTION:\n{r.get('q', '')}\n\nWHAT THE SEARCH RETURNED:\n{served}\n\nREPLY:\n{reply[:2000]}", "search")
    else:
        v = ask_teacher(SEARCH_SYS, f"QUESTION:\n{r.get('q', '')}\n\nWHAT THE SEARCH RETURNED:\n{served}\n\nREPLY:\n{reply[:2000]}", "search")
    if "error" in v: return base, {**why, **v}
    talk = A.w_talk if all(bool(v.get(k)) for k in ("sound", "natural", "clean")) else 0.0
    return base + talk, {**why, **v}


def demo_backward(q, traj, coef, hist=None):
    """the teacher's trajectory as a demonstration of the searching, not of the reply.

    2829 of the 2857 trajectories end "The answer is X.", the register this lineage was fine-tuned
    out of, so everything after </think> is masked along with the information blocks: the model is
    shown which queries reach the page, and nothing about how to word the answer."""
    head_t = tok.apply_chat_template(list(hist if hist is not None else HIST.get(q, [])) + [{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False)
    if not head_t.rstrip().endswith("<think>"): head_t += "<think>\n"
    ids = tok.encode(head_t, add_special_tokens=False); msk = [0] * len(ids)
    body = traj.split("</think>")[0]
    body = body[len("<think>"):].lstrip("\n") if body.startswith("<think>") else body
    for piece in INFO_RE.split(body):
        if not piece: continue
        t = tok.encode(piece, add_special_tokens=False); ids += t
        msk += [0 if piece.startswith("<information>") else 1] * len(t)
    ids, msk = ids[:A.maxlen], msk[:A.maxlen]
    if sum(msk[1:]) == 0: return 0.0
    x = torch.tensor([ids], device=DEV)
    h = BODY(input_ids=x, use_cache=False).last_hidden_state[:, :-1, :]
    tgt = x[:, 1:]; tm = torch.tensor([msk[1:]], device=DEV, dtype=torch.float32)
    v = ce_backward(h, tgt, tm, coef); del h; clear(); return v


def judge_reason(q, ref, text):
    """teacher score for one reasoning sample; the reward is 1.0 only when every box is ticked"""
    reply = text.split("</think>")[-1].strip() if "</think>" in text else ""   # no </think>: unfinished - the thinking is not a reply
    if not reply: return 0.0, {"unfinished": True}                                   # (until 2026-10-06 it was judged as one, and could pass)
    if A.reason_stub:                                     # shape only, for a smoke run with no credit
        ok = 8 <= len(reply.split()) <= 400 and not any(t in reply for t in TAGS)
        return (1.0 if ok else 0.0), {"stub": True}
    v = ask_teacher(REASON_SYS, f"QUESTION:\n{q[:2000]}\n\nREFERENCE ANSWER:\n{ref[:3000]}\n\nASSISTANT ANSWER:\n{reply[:3000]}", "reason")
    if "error" in v: return 0.0, v
    return (1.0 if all(bool(v.get(k)) for k in ("solves_it", "follows_the_request", "language_english", "clean")) else 0.0), v


NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
def final_number(text):
    """the answer a reply commits to: \\boxed{...} first, then '####', then the last number in the reply"""
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    cand = m[-1] if m else (text.split("####")[-1] if "####" in text else text)
    nums = NUM_RE.findall(cand)
    if not nums: return None
    try: return float(nums[-1].replace(",", ""))
    except ValueError: return None
def verify_numeric(ref, text):
    """verifiable reasoning reward: 1.0 when the reply's final number equals the reference number"""
    reply = text.split("</think>")[-1].strip() if "</think>" in text else ""
    if not reply: return 0.0, {"unfinished": True}
    g = final_number(ref); a = final_number(reply)
    ok = g is not None and a is not None and abs(a - g) <= 1e-6 * max(1.0, abs(g))
    return (1.0 if ok else 0.0), {"solves_it": int(ok), "answer": a, "gold": g}


def judge_ok(v):
    if v.get("skip"): return True
    return ("error" not in v) and all(bool(v.get(x)) for x in ("commits_to_answer", "matches_reference", "language_english", "clean")) and not v.get("unsupported_claims") and float(v.get("natural") or 0) >= 4

# ---- plain SFT step for a Dolphin record (no compression: the whole sequence is one block) ----
def plain_backward(rec, coef):
    head_t = tok.apply_chat_template([{"role": "user", "content": rec["q"]}], add_generation_prompt=True, tokenize=False)
    if not head_t.rstrip().endswith("<think>"): head_t += "<think>\n"
    hid = tok.encode(head_t, add_special_tokens=False)
    body = tok.encode(rec["thinking"].strip() + "\n</think>\n" + rec["reply"].strip(), add_special_tokens=False) + [eos]
    ids = (hid + body)[:A.maxlen]; msk = ([0] * len(hid) + [1] * len(body))[:A.maxlen]
    if sum(msk) == 0: return 0.0
    x = torch.tensor([ids], device=DEV)
    h = BODY(input_ids=x, use_cache=False).last_hidden_state[:, :-1, :]
    tgt = x[:, 1:]; tm = torch.tensor([msk[1:]], device=DEV, dtype=torch.float32)
    v = ce_backward(h, tgt, tm, coef); del h; clear(); return v

noom = [0]


def guarded(fn, *a):
    """one record that does not fit costs that record, not the run"""
    try:
        return fn(*a)
    except torch.OutOfMemoryError:
        noom[0] += 1; print(f"[warn] out of memory in {fn.__name__}, record skipped ({noom[0]} so far)", flush=True)
        clear(); return 0.0


TAGS = ("<search>", "<information>", "</think>", "<think>", "<more")
log = open(os.path.join(A.outdir, "loop.log"), "a")
roll_fh = open(os.path.join(A.outdir, "rollouts.jsonl"), "a")
acc_fh = open(os.path.join(A.outdir, "accepted.jsonl"), "a")
cum = state.get("cum", {}); t0 = time.time(); di = state.get("di", 0); ri = state.get("ri", 0)
queue = []; qi = state.get("qi", 0)
demos = {}
if A.search_demo:
    for line in open(A.search_demo):
        try:
            r = json.loads(line)
            if r.get("q") and r.get("traj"): demos[r["q"].strip()] = r["traj"]
        except Exception: pass
    print(f"[data] {len(demos)} teacher trajectories for the search side", flush=True)
gw = collections.deque(maxlen=A.guard_steps); gbase = tuple(state["gbase"]) if state.get("gbase") and len(state["gbase"]) == 3 else None
reason = []
if A.reason:
    for line in open(A.reason):
        try:
            r = json.loads(line)
            if r.get("q") and r.get("reply"): reason.append({"q": r["q"], "ref": r["reply"], "thinking": r.get("thinking", "")})
        except Exception: pass
    random.Random(0).shuffle(reason)
    print(f"[data] {len(reason)} reasoning problems, {A.reason_g} samples each, teacher "
          f"{'stub' if A.reason_stub else (A.judge_model or ('gpt-5-nano' if A.judge_api == 'openai' else 'deepseek-flash'))}", flush=True)
GOOD = os.path.join(A.outdir, "good.safetensors")
recent = collections.deque(maxlen=A.guard_window)      # sliding window of outcomes, for the collapse guard
recent_ns = collections.deque(maxlen=A.guard_window)   # and of the searches each rollout issued
base_rate = state.get("base_rate"); base_ns = state.get("base_ns"); base_cut = state.get("base_cut", 0.0); nrb = state.get("rollbacks", 0); guard_from = state.get("guard_from", 0)


def noreply_rate(xs):
    return sum(1 for w in xs if w == "no reply") / max(len(xs), 1)


def cut_rate(xs):
    return sum(1 for w in xs if w == "search loop") / max(len(xs), 1)


def reload_good():
    """last healthy weights back into the live parameters (same objects, so the optimizer keeps pointing at them)"""
    sd = load_file(GOOD)
    missing = model.load_state_dict({k: v.to(DEV) for k, v in sd.items() if not k.startswith("pooler.")}, strict=False)
    opt.zero_grad(set_to_none=True)
    if A.guard_halve:
        for g in opt.param_groups: g["lr"] = g["lr"] / 2
    if opt_s is not None:
        opt_s.zero_grad(set_to_none=True)
        if A.guard_halve:
            for g in opt_s.param_groups: g["lr"] = g["lr"] / 2
    return len(missing.unexpected_keys)

if A.probe_out:
    # triage: which items does this model solve sometimes, always, never (gold string only, no teacher)
    items = pool[:A.probe_n] if A.probe_n else pool
    done_q = set()
    if os.path.exists(A.probe_out):
        done_q = set(json.loads(l)["q"] for l in open(A.probe_out) if l.strip())
    todo = [it for it in items if it["q"] not in done_q]
    per = max(1, A.b // A.probe_g)
    print(f"[probe] {len(items)} items, {len(todo)} to go, {A.probe_g} samples each, {per} items a batch", flush=True)
    with open(A.probe_out, "a") as fo:
        for k in range(0, len(todo), per):
            chunk = todo[k:k + per]
            try: outs = rollout_batch([it["q"] for it in chunk for _ in range(A.probe_g)], per * A.probe_g if len(chunk) == per else len(chunk) * A.probe_g)
            except Exception as e:
                print(f"[probe] batch {k} failed: {type(e).__name__}: {str(e)[:120]}", flush=True); clear(); continue
            for j, it in enumerate(chunk):
                rs = outs[j * A.probe_g:(j + 1) * A.probe_g]
                ok = [o for o in rs if price({**o, "answer": o.get("answer", "")}, it["gold"])[0] >= 1.0]
                fo.write(json.dumps({"q": it["q"], "gold": it["gold"], "hist": HIST.get(it["q"], []), "n": len(rs), "pass": len(ok),
                                     "texts": [o["text"] for o in ok][:2]}, ensure_ascii=False) + "\n")
            fo.flush(); clear()
            print(f"[probe] {min(k + per, len(todo))}/{len(todo)} ({(time.time()-t0)/60:.0f} min)", flush=True)
    print("PROBE_DONE", flush=True); sys.exit(0)
if A.demo_sft:
    # supervised steps: teacher trajectories for what the model never solves, its own verified traces for the rest
    DM = [json.loads(l) for l in open(A.demo_sft) if l.strip()]
    random.Random(3).shuffle(DM); random.Random(4).shuffle(rep)
    print(f"[demo-sft] {len(DM)} teacher trajectories, {len(rep)} own traces, {A.demo_per_step} + {A.replay_per_step} a step, {A.steps} steps", flush=True)
    for step in range(state["step"] + 1, A.steps + 1):
        opt.zero_grad(set_to_none=True); model.train(); dl, rl_ = [], []
        for k in range(A.demo_per_step):
            d = DM[((step - 1) * A.demo_per_step + k) % len(DM)]
            dl.append(guarded(demo_backward, d["q"], d["traj"], 1.0 / (A.demo_per_step + A.replay_per_step), d.get("hist") or [])); clear()
        for k in range(A.replay_per_step if rep else 0):
            rl_.append(guarded(replay_backward, rep[((step - 1) * A.replay_per_step + k) % len(rep)], 1.0 / (A.demo_per_step + A.replay_per_step))); clear()
        opt.step(); opt.zero_grad(set_to_none=True); clear()
        line = f"[step {step}] demo ce={sum(dl)/max(len(dl),1):.3f} own ce={sum(rl_)/max(len(rl_),1):.3f} | epoch {step * A.demo_per_step / max(len(DM),1):.2f} | {(time.time()-t0)/60:.0f} min"
        print(line, flush=True); log.write(line + "\n"); log.flush()
        if step % A.save_every == 0 or step == A.steps:
            save_ckpt(LATEST); json.dump({"step": step, "cum": cum, "di": di, "ri": ri, "qi": qi}, open(STATE_F, "w")); save_opt()
            import shutil; shutil.copyfile(LATEST, os.path.join(A.outdir, f"step{step}.safetensors"))
            print(f"[save] step {step}", flush=True)
    print("ONLINE_LOOP_DONE", flush=True); sys.exit(0)
if A.eval_file:
    # the batched rollout that training uses, so a 100-question evaluation costs minutes instead of hours
    evq = [json.loads(l) for l in open(A.eval_file) if l.strip()]
    done_q = set()
    if os.path.exists(A.eval_out):
        done_q = set(json.loads(l)["q"] for l in open(A.eval_out) if l.strip())
    todo = [r for r in evq if r["q"] not in done_q]
    print(f"[eval] {len(evq)} questions, {len(todo)} to go, {A.b} at a time, gen {A.gen}, temp {A.temp}", flush=True)
    with open(A.eval_out, "a") as fo:
        for k in range(0, len(todo), A.b):
            chunk = todo[k:k + A.b]
            try: outs = rollout_batch([r["q"] for r in chunk], len(chunk))
            except Exception as e:
                print(f"[eval] batch {k} failed: {type(e).__name__}: {str(e)[:120]}", flush=True); clear(); continue
            for r, o in zip(chunk, outs):
                fo.write(json.dumps({"q": r["q"], "text": o["text"], "ns": o["ns"]}, ensure_ascii=False) + "\n")
            fo.flush(); clear()
            print(f"[eval] {min(k + A.b, len(todo))}/{len(todo)} ({(time.time()-t0)/60:.0f} min)", flush=True)
    print("EVAL_DONE", flush=True); sys.exit(0)
for step in range(state["step"] + 1, A.steps + 1) if not (reason or A.mt_items) else []:
    item = pool[(step - 1) % len(pool)]
    if (step - 1) % A.accum == 0: opt.zero_grad(set_to_none=True)
    rolls = []
    if A.queue:
        if not queue:                                   # B different questions, one rollout each, all of them queued
            items = [pool[(qi + j) % len(pool)] for j in range(A.b)]; qi += A.b
            try:
                for it, r in zip(items, rollout_batch([it["q"] for it in items], A.b)):
                    r["item"] = it; queue.append(r)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                print(f"[warn] batch dropped: {type(e).__name__}: {str(e)[:120]}", flush=True); clear()
        if queue:
            rolls = [queue.pop(0)]; item = rolls[0]["item"]
    elif step % A.rollout_every == 0:
        try:
            rolls = rollout_batch(item["q"], A.b)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            print(f"[warn] batch dropped: {type(e).__name__}: {str(e)[:120]}", flush=True); clear()
    reasons = []; positives = []
    for r in rolls:
        _, c, g = price(r, item["gold"]); reply = r["text"].split("</think>")[-1] if "</think>" in r["text"] else ""
        if r.get("cut"): why = "search loop"
        elif not r["landed"]: why = "no reply"
        elif not g: why = "page not found"
        elif not c: why = "wrong"
        elif any(t in reply for t in TAGS): why = "tags"
        else:
            v = judge(item["q"], item["gold"], r["text"])
            why = "accepted" if judge_ok(v) else ("judge error" if "error" in v else "judge")
            if why == "accepted": positives.append(r)
        reasons.append(why); cum[why] = cum.get(why, 0) + 1
        roll_fh.write(json.dumps({"step": step, "q": item["q"], "gold": item["gold"], "why": why, "ns": r["ns"], "text": r["text"]}, ensure_ascii=False) + "\n")
    roll_fh.flush()
    losses = []; dl = []
    trainset = rolls if (A.train_all or A.queue) else positives
    ncut = 0
    if A.complete_only:
        keep = [r for r in trainset if r.get("ended") and not r.get("dead")]
        ncut = len(trainset) - len(keep); trainset = keep   # no selection: every sample of the step is a target
    if trainset:
        model.train()
        for r in trainset:
            losses.append(guarded(pg_backward, r, 1.0 / (len(trainset) * A.accum))); clear()
        for r in positives:
            acc_fh.write(json.dumps({"q": item["q"], "gold": item["gold"], "text": r["text"], "step": step}, ensure_ascii=False) + "\n")
            rep.append({"q": item["q"], "text": r["text"], "src": "accepted"})   # joins the retention data
        acc_fh.flush()
    rl = []
    if rep and A.replay_per_step > 0:
        model.train()
        for _ in range(A.replay_per_step):
            rl.append(guarded(replay_backward, rep[ri % len(rep)], 1.0 / (A.replay_per_step * A.accum))); ri += 1
    nd = max(A.dolphin_min, int(round(A.dolphin_ratio * len(trainset))))
    if dol:
        model.train()
        for _ in range(nd):
            dl.append(guarded(plain_backward, dol[di % len(dol)], 1.0 / (nd * A.accum))); di += 1
    if step % A.accum == 0:
        opt.step(); opt.zero_grad(set_to_none=True); clear()
    tot = sum(cum.values()); acc = cum.get("accepted", 0)
    line = (f"[step {step}] " + (" ".join(f"{k}={sum(1 for x in reasons if x == k)}" for k in ("accepted", "wrong", "page not found", "no reply", "search loop", "tags", "judge", "judge error") if any(x == k for x in reasons)) if rolls else "no rollout")
            + (f" queue={len(queue)}" if A.queue else "") + (f" unfinished={ncut}" if ncut else "") + f" | sft_ce={sum(losses)/max(len(losses),1):.3f} replay_ce={sum(rl)/max(len(rl),1):.3f} dolphin_ce={sum(dl)/max(len(dl),1):.3f} | cumulative accept {acc}/{tot} ({100*acc/max(tot,1):.1f}%) "
            + " ".join(f"{k}:{100*v/max(tot,1):.0f}%" for k, v in sorted(cum.items())) + f" | {(time.time()-t0)/60:.0f} min")
    print(line, flush=True); log.write(line + "\n"); log.flush()
    if A.guard:
        recent.extend(reasons); recent_ns.extend(r["ns"] for r in rolls)
        if base_rate is None and len(recent) == recent.maxlen:
            base_rate = noreply_rate(recent); base_ns = sum(recent_ns) / max(len(recent_ns), 1); base_cut = cut_rate(recent)
            print(f"[guard] baseline no-reply {100*base_rate:.0f}%, cut {100*base_cut:.0f}%, {base_ns:.1f} searches per rollout, over the first {recent.maxlen} rollouts", flush=True)
        elif base_rate is not None and len(recent) == recent.maxlen and step > guard_from + 20:
            cur = noreply_rate(recent); cur_ns = sum(recent_ns) / max(len(recent_ns), 1); cur_cut = cut_rate(recent)
            trip = (cur >= max(A.guard_floor, A.guard_mult * base_rate)
                    or (base_ns and cur_ns >= max(A.guard_ns_floor, A.guard_ns * base_ns))
                    or cur_cut >= max(A.guard_cut_floor, A.guard_cut * base_cut))
            if trip:
                if nrb >= A.guard_rollbacks or not os.path.exists(GOOD):
                    print(f"ONLINE_COLLAPSE step {step}: no-reply {100*cur:.0f}% vs {100*base_rate:.0f}%, searches {cur_ns:.1f} vs {base_ns:.1f}, cut {100*cur_cut:.0f}% vs {100*base_cut:.0f}%, "
                          + ("no healthy checkpoint" if not os.path.exists(GOOD) else f"{nrb} rollbacks spent"), flush=True)
                    save_ckpt(LATEST); break
                nrb += 1; nu = reload_good()
                print(f"ONLINE_ROLLBACK {nrb} at step {step}: no-reply {100*cur:.0f}% vs {100*base_rate:.0f}%, searches {cur_ns:.1f} vs {base_ns:.1f}, cut {100*cur_cut:.0f}% vs {100*base_cut:.0f}%, "
                      f"reloaded {GOOD} ({nu} unexpected), lr now {opt.param_groups[0]['lr']:.2g}", flush=True)
                recent.clear(); recent_ns.clear(); guard_from = step
            elif step % A.save_every == 0 and cur <= max(base_rate * 1.3, base_rate + 0.03) and cur_ns <= base_ns * 1.2 and cur_cut <= max(base_cut * 1.5, 0.10):
                save_ckpt(GOOD)                              # this window still looks like the opening one
    if step % A.save_every == 0 or step == A.steps:
        save_ckpt(LATEST); json.dump({"step": step, "cum": cum, "di": di, "ri": ri, "qi": qi, "base_rate": base_rate, "base_ns": base_ns, "base_cut": base_cut, "rollbacks": nrb, "guard_from": guard_from}, open(STATE_F, "w")); print(f"[save] step {step}", flush=True)
if not (reason or A.mt_items): print("ONLINE_LOOP_DONE", flush=True)


# ---- GRPO, on the reasoning problems and on the search pool in turn ----
# Not the self-imitation that r7-r11 were: every sample of a group is used, its advantage measured
# against the group's own mean, so a sample below the mean is pushed down rather than ignored. A
# group whose samples all score alike carries no signal; when they all score zero the model cannot
# do the problem at all, so with --wheels the teacher's own answer is trained on instead.
for step in range(state["step"] + 1, A.steps + 1) if (reason or A.mt_items) else []:
    if A.sft_only:
        # distillation: the teacher's own thinking and answer, no sampling. The search trace keeps the other side rehearsed.
        opt.zero_grad(set_to_none=True); model.train(); dl = []
        for k in range(A.sft_only):
            pr_ = reason[((step - 1) * A.sft_only + k) % len(reason)]
            dl.append(guarded(plain_backward, {"q": pr_["q"], "thinking": pr_.get("thinking", ""), "reply": pr_["ref"]}, 1.0 / A.sft_only)); clear()
        rl_ = guarded(replay_backward, rep[ri % len(rep)], A.rft_replay) if (rep and A.rft_replay > 0) else 0.0; ri += 1; clear()
        opt.step(); opt.zero_grad(set_to_none=True); clear()
        line = (f"[step {step}] sft {A.sft_only} records ce={sum(dl)/max(len(dl),1):.3f} replay_ce={rl_:.3f} | epoch {((step) * A.sft_only) / len(reason):.2f} | {(time.time()-t0)/60:.0f} min")
        print(line, flush=True); log.write(line + "\n"); log.flush()
        if step % A.save_every == 0 or step == A.steps:
            save_ckpt(LATEST); json.dump({"step": step, "cum": cum, "di": di, "ri": ri, "qi": qi}, open(STATE_F, "w")); save_opt()
            if step % 200 == 0:
                import shutil; shutil.copyfile(LATEST, os.path.join(A.outdir, f"step{step}.safetensors")); shutil.copyfile(STATE_F, os.path.join(A.outdir, f"step{step}.json"))
        continue
    # --mt-items alone: every step is a search turn; with --reason and --reason-every, every Nth step is a reasoning problem instead
    searching = ((step % A.reason_every != 0) if (A.reason_every and reason) else True) if A.mt_items \
        else (step % A.reason_every != 0) if A.reason_every else (bool(A.search_every) and step % A.search_every == 0)
    if searching:
        item = pool_item(step); qtext, ref = item["q"], None
    else:
        prob = reason[(step - 1) % len(reason)]; qtext, ref = prob["q"], prob["ref"]
    if (step - 1) % A.accum == 0: opt.zero_grad(set_to_none=True)
    A.temp = ns["TEMP"] = A.search_temp if (searching and A.search_temp > 0) else BASE_TEMP
    A.gen = A.search_gen if (searching and A.search_gen > 0) else BASE_GEN
    seed = None
    if searching:
        seed = item.get("seed") or (seed_from_hist(HIST[qtext]) if HIST.get(qtext) else None)
    try:
        rolls = rollout_batch(qtext, A.search_g if (searching and A.search_g > 0) else A.reason_g, seed=seed)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        print(f"[warn] batch dropped: {type(e).__name__}: {str(e)[:120]}", flush=True); clear(); continue
    for r in rolls: r.setdefault("q", qtext)
    rw, notes = [], []
    score = ((lambda x: score_search(x, item["gold"])) if searching
             else (lambda x: verify_numeric(ref, x["text"])) if A.reason_verify == "numeric"
             else (lambda x: judge_reason(qtext, ref, x["text"])))
    with ThreadPoolExecutor(max_workers=min(8, len(rolls))) as ex:
        for r, (sc, v) in zip(rolls, ex.map(score, rolls)):
            rw.append(sc); notes.append(v)
            roll_fh.write(json.dumps({"step": step, "kind": "search" if searching else "reason", "q": qtext,
                                      "reward": sc, "why": v, "ns": r["ns"], "text": r["text"]},
                                     ensure_ascii=False) + "\n")
    roll_fh.flush()
    if searching and A.followup > 0 and A.mt_items and not HIST.get(qtext) and rw and max(rw) >= 1.0 and random.random() < A.followup:
        try: make_followup(step, qtext, item["gold"], rolls[max(range(len(rw)), key=lambda i_: rw[i_])])
        except Exception as e_: print(f"[followup] failed: {type(e_).__name__}: {str(e_)[:100]}", flush=True)
    mu = sum(rw) / max(len(rw), 1)
    sd = (sum((x - mu) ** 2 for x in rw) / max(len(rw), 1)) ** 0.5
    losses = []; wheel = 0.0
    n_err = sum(1 for v in notes if isinstance(v, dict) and "error" in v)
    if n_err * 2 > len(notes):
        print(f"[warn] step {step}: {n_err}/{len(notes)} judge calls failed ({[v.get('error') for v in notes if isinstance(v, dict) and 'error' in v][:1]}); nothing trained", flush=True)
    elif A.rft and not searching:
        # rejection sampling: the passing sample with the shortest thinking becomes a plain SFT record
        model.train()
        good = [r for r, x in zip(rolls, rw) if x >= 1.0 and "</think>" in r["text"]]
        if good:
            best = min(good, key=lambda r: len(r["text"].split("</think>")[0]))
            th, rp = best["text"].split("</think>", 1)
            losses.append(guarded(plain_backward, {"q": qtext, "thinking": th.replace("<think>", "").strip(), "reply": rp.strip()}, 1.0 / A.accum)); clear()
        elif A.wheels and (not A.wheel_max_think or len(prob.get("thinking", "").split()) <= A.wheel_max_think):
            wheel = guarded(plain_backward, {"q": qtext, "thinking": prob.get("thinking", ""), "reply": ref}, 1.0 / A.accum); clear()
        if rep and A.rft_replay > 0:
            losses.append(guarded(replay_backward, rep[ri % len(rep)], A.rft_replay / A.accum)); ri += 1; clear()
    elif sd > 1e-6:
        model.train()
        for r, x in zip(rolls, rw):
            losses.append(guarded(pg_backward, r, (x - mu) / (sd if A.adv_std else 1.0) / (len(rolls) * A.accum))); clear()
    elif max(rw) <= 0.0 and A.wheels:      # every rollout failed (0 on the reasoning side, -0.5 or a page-hit -0.1 on the search side)
        # nothing of its own to learn from: the teacher demonstrates instead
        model.train()
        if searching and not A.search_wheels:
            pass                                          # nothing to demonstrate: this side already works
        elif searching:
            traj = demos.get(qtext.strip())
            if traj: wheel = guarded(demo_backward, qtext, traj, 1.0 / A.accum)
            elif rep: wheel = guarded(replay_backward, rep[ri % len(rep)], 1.0 / A.accum); ri += 1
        else:
            wheel = guarded(plain_backward, {"q": qtext, "thinking": prob.get("thinking", ""), "reply": ref}, 1.0 / A.accum)
        clear()
    if A.demo_on_fail > 0 and searching and rw and max(rw) < 1.0 and demos.get(qtext.strip()):
        model.train()   # nothing of its own passed: the teacher shows this question once, beside whatever the group's gradient said
        wheel = guarded(demo_backward, qtext, demos[qtext.strip()], A.demo_on_fail / A.accum); clear()
    elif A.r1_on_fail and A.demo_on_fail > 0 and searching and rw and max(rw) < 1.0:
        traj = r1_on_the_fly(qtext, item["gold"], HIST.get(qtext) or [])
        if traj:
            demos[qtext.strip()] = traj; model.train()
            wheel = guarded(demo_backward, qtext, traj, A.demo_on_fail / A.accum); clear()
    if A.cot_on_fail > 0 and not searching and rw and max(rw) < 1.0 and prob.get("thinking"):
        model.train()   # the problem's own reference thinking and answer, once
        wheel = guarded(plain_backward, {"q": qtext, "thinking": prob["thinking"], "reply": ref}, A.cot_on_fail / A.accum); clear()
    if opt_s is not None and searching:
        opt_s.step(); opt_s.zero_grad(set_to_none=True); opt.zero_grad(set_to_none=True); clear()
    dl = []
    if dol and A.dolphin_min and not (A.dolphin_on == "reason" and searching):
        model.train()
        for _ in range(A.dolphin_min):
            dl.append(guarded(plain_backward, dol[di % len(dol)], 1.0 / (A.dolphin_min * A.accum))); di += 1
    if step % A.accum == 0:
        opt.step(); opt.zero_grad(set_to_none=True); clear()
        if opt_s is not None: opt_s.zero_grad(set_to_none=True)   # pooler grads from a reasoning rollout are not search signal
    k = "search" if searching else "reason"
    cum[k + " pass"] = cum.get(k + " pass", 0) + sum(1 for x in rw if x >= 1.0); cum[k + " n"] = cum.get(k + " n", 0) + len(rw)
    cum[k + " sum"] = cum.get(k + " sum", 0.0) + sum(rw)
    unfin = sum(1 for v in notes if v.get("unfinished")); err = sum(1 for v in notes if v.get("error"))
    line = (f"[step {step}] {k} pass {sum(1 for x in rw if x >= 1.0)}/{len(rw)} mean {mu:+.2f}"
            + (f" unfinished={unfin}" if unfin else "") + (f" judge-error={err}" if err else "")
            + (" wheels" if wheel else "")
            + f" | ce={sum(losses)/max(len(losses),1):.3f} wheel_ce={wheel:.3f} dolphin_ce={sum(dl)/max(len(dl),1):.3f}"
            + f" | cumulative reason {100*cum.get('reason pass',0)/max(cum.get('reason n',0),1):.0f}% (mean {cum.get('reason sum',0)/max(cum.get('reason n',0),1):+.2f}) of {cum.get('reason n',0)}"
            + f", search {100*cum.get('search pass',0)/max(cum.get('search n',0),1):.0f}% (mean {cum.get('search sum',0)/max(cum.get('search n',0),1):+.2f}) of {cum.get('search n',0)}"
            + f" | {(time.time()-t0)/60:.0f} min")
    print(line, flush=True); log.write(line + "\n"); log.flush()
    if A.guard and searching:
        # what g5 did at step ~225: searches per rollout 1 -> 5, thinking 174 -> 424 words, half the
        # rollouts never answering. Watched over the last ten search steps against the first ten.
        gw.append((sum(1 for v in notes if v.get("unfinished")) / max(len(notes), 1),
                   sum(r["ns"] for r in rolls) / max(len(rolls), 1),
                   sum(1 for x in rw if x >= 1.0) / max(len(rw), 1)))
        if gbase is None and len(gw) == gw.maxlen:
            gbase = tuple(sum(x[i] for x in gw) / len(gw) for i in range(3))
            print(f"[guard] baseline over the first {gw.maxlen} search steps: unfinished {100*gbase[0]:.0f}%, {gbase[1]:.1f} searches per rollout, pass {100*gbase[2]:.0f}%", flush=True)
        elif gbase is not None and len(gw) == gw.maxlen and step > guard_from + 2 * gw.maxlen:
            cu, cn, cp = (sum(x[i] for x in gw) / len(gw) for i in range(3))
            worse = cu >= max(A.guard_floor, A.guard_mult * gbase[0]) or cn >= max(A.guard_ns_floor, A.guard_ns * gbase[1])
            trip = worse and cp <= A.guard_pass * gbase[2]      # g7 step 40: one 7-unfinished question tripped it at 43% pass, same as the baseline
            if worse and not trip: print(f"[guard] window unfinished {100*cu:.0f}%, searches {cn:.1f}, but pass {100*cp:.0f}% vs {100*gbase[2]:.0f}%: not a collapse", flush=True)
            if trip:
                if nrb >= A.guard_rollbacks or not os.path.exists(GOOD):
                    print(f"ONLINE_COLLAPSE step {step}: unfinished {100*cu:.0f}% vs {100*gbase[0]:.0f}%, searches {cn:.1f} vs {gbase[1]:.1f}, "
                          + ("no healthy checkpoint" if not os.path.exists(GOOD) else f"{nrb} rollbacks spent"), flush=True)
                    save_ckpt(LATEST); break
                nrb += 1; nu = reload_good()
                print(f"ONLINE_ROLLBACK {nrb} at step {step}: unfinished {100*cu:.0f}% vs {100*gbase[0]:.0f}%, searches {cn:.1f} vs {gbase[1]:.1f}, pass {100*cp:.0f}% vs {100*gbase[2]:.0f}%, "
                      f"reloaded {GOOD} ({nu} unexpected), lr now {opt.param_groups[0]['lr']:.2g}", flush=True)
                gw.clear(); guard_from = step
            elif cu <= max(gbase[0] * 1.5, gbase[0] + 0.05) and cn <= max(gbase[1] * 1.5, gbase[1] + 0.5) and step % A.save_every == 0:
                save_ckpt(GOOD)                                  # this window still looks like the opening one
    if step % A.save_every == 0 or step == A.steps:
        save_ckpt(LATEST); json.dump({"step": step, "cum": cum, "di": di, "ri": ri, "qi": qi, "gbase": gbase, "rollbacks": nrb, "guard_from": guard_from}, open(STATE_F, "w"))
        save_opt(); print(f"[save] step {step}", flush=True)
        if step % 200 == 0:   # the exact-step snapshot the freeze uploads (the 200-step marks used to get whatever "latest" was when a control run noticed)
            import shutil; shutil.copyfile(LATEST, os.path.join(A.outdir, f"step{step}.safetensors")); shutil.copyfile(STATE_F, os.path.join(A.outdir, f"step{step}.json"))
if reason or A.mt_items: print("ONLINE_LOOP_DONE", flush=True)
