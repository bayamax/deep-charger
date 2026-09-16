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
ap.add_argument("--guard", type=int, default=0, help="1: watch the share of rollouts that write no reply over a sliding window; when it runs away from the opening baseline, reload the last healthy checkpoint, halve the learning rate and carry on (three times, then stop)"); ap.add_argument("--guard-window", type=int, default=80); ap.add_argument("--guard-floor", type=float, default=0.20, help="no trip below this absolute rate"); ap.add_argument("--guard-mult", type=float, default=2.0, help="trip at this multiple of the opening baseline"); ap.add_argument("--guard-cut", type=float, default=3.0, help="trip at this multiple of the opening share of rollouts cut for searching without end"); ap.add_argument("--guard-cut-floor", type=float, default=0.20); ap.add_argument("--guard-ns-floor", type=float, default=6.0, help="no searches trip below this absolute mean"); ap.add_argument("--guard-ns", type=float, default=1.8, help="trip at this multiple of the opening searches per rollout"); ap.add_argument("--guard-rollbacks", type=int, default=3); ap.add_argument("--complete-only", type=int, default=0, help="1: train only on rollouts that actually finished their turn (EOS reached, no repetition death). Not a quality filter: a rollout cut off by the token cap is an unfinished fragment, and training on it teaches the model not to stop -- r7 went from 5%% to 59%% no-reply that way"); ap.add_argument("--reason", default="", help="jsonl of reasoning problems {q, reply}: the loop leaves the search corpus and runs GRPO on these instead, the teacher scoring each sample against the reference"); ap.add_argument("--reason-g", type=int, default=8, help="samples per problem"); ap.add_argument("--judge-api", default="deepseek", choices=["deepseek", "openai"], help="which teacher scores the reasoning samples"); ap.add_argument("--judge-model", default="", help="model name for that teacher; empty picks the default for the api"); ap.add_argument("--reason-stub", type=int, default=0, help="1: score by shape alone, no teacher call (for a smoke run with no credit)"); ap.add_argument("--queue", type=int, default=0, help="1: each rollout batch takes --b different questions; the rows queue up and every step trains on one of them (no selection) plus Dolphin; a new batch runs when the queue is empty"); ap.add_argument("--train-all", type=int, default=0, help="1: train on every rollout of the step, no selection (the gold and the judge only measure)"); ap.add_argument("--replay-per-step", type=int, default=2); ap.add_argument("--rollout-every", type=int, default=1, help="do the measurement rollout only every N steps (accepted ones join the replay set)")
ap.add_argument("--accum", type=int, default=1, help="steps whose gradients are accumulated before one optimizer update (both the search-side and the Dolphin part)")
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
if os.path.isdir(init_path):                  # merged HF model dir: the weights ARE the base, nothing to overlay
    os.environ["SP_BASE"] = init_path
else:
    os.environ["SP_INIT_FULL"] = init_path
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
print(f"[init] weights <- {init_path} (resume step {state['step']}) lora r={A.lora_rank} layers={A.lora_layers}", flush=True)
if A.pooler_init and os.path.isdir(init_path) and os.path.exists(A.pooler_init):   # resuming a .safetensors already carries its pooler
    from safetensors.torch import load_file as _lf
    n = pooler.load_sd(_lf(A.pooler_init)); print(f"[init] pooler <- {A.pooler_init} ({n} tensors)", flush=True)


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
opt = torch.optim.Adam(
    [{"params": [p for p in model.parameters() if p.requires_grad], "lr": A.lr}]
    + ([{"params": pooler_params, "lr": A.pooler_lr}] if pooler_params else []))
CLM = model.base_model.model            # peft -> causal LM
BODY, HEAD = CLM.model, CLM.lm_head     # transformer body, lm_head
if A.gradckpt:
    CLM.config.use_cache = False
    CLM.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("[init] gradient checkpointing ON for the policy-gradient pass", flush=True)
nT = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
nP = sum(p.numel() for p in pooler_params) / 1e6
print(f"[cfg] G={A.g} steps={A.steps} budget={A.budget} rw={A.rw} maxd={A.maxd} chunk={A.chunk} temp={A.temp} gen={A.gen} maxs={A.maxs} maxm={A.maxm} samepage={A.samepage} maxsrch={A.maxsrch} phantom={A.phantom}x{A.phantom_scale} "
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
CACHE_F = "/root/work/pool_eval_cache.jsonl"
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


@torch.no_grad()
def rollout_batch(question, B):
    """B independent rollouts decoded in lockstep: one question for all rows (str), or one question per row (list).

    Semantics match rollout() exactly: every row keeps its own generation, pooled set, page offsets and stop
    conditions, and each block is rebuilt from that row's own state against a freshly prefilled question cache.
    Only the per-token decode is shared -- that is the part that leaves the GPU at ~14% utilisation when the
    rollouts are run one at a time. Rows that stop early are carried along with filler tokens; the cache is
    rebuilt from scratch every block, so their pollution never reaches a row that is still generating."""
    model.eval()
    qs = list(question) if isinstance(question, (list, tuple)) else [question] * B
    B = len(qs)
    QID = [tok.encode(tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False) + "<think>\n")
           for q in qs]
    MQs = [len(x) for x in QID]; MQ = max(MQs)                       # MQ = cache slots of the question prefix (left-padded)
    PAD = tok.pad_token_id if tok.pad_token_id is not None else eos
    q_in = torch.full((B, MQ), PAD, dtype=torch.long, device=DEV); q_am = torch.zeros(B, MQ, device=DEV)
    q_pos = torch.zeros(B, MQ, dtype=torch.long, device=DEV)
    for b, ids in enumerate(QID):                                    # every row's own tokens end on slot MQ-1, positions 0..len-1
        q_in[b, MQ - len(ids):] = torch.tensor(ids, device=DEV); q_am[b, MQ - len(ids):] = 1
        q_pos[b, MQ - len(ids):] = torch.arange(len(ids), device=DEV)
    S = [dict(gen=[], msk=[], kept=[], absorbed=0, segs=[], n_model=0, ns_=0, nm=0, nmt=0, served=[], queries=[],
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
        txt = tok.decode(gen)
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
            txt = tok.decode(st["gen"])
            if st["dead"] or st["cut"] or st.get("ended") or (st["gen"] and st["gen"][-1] == eos):
                st["done"] = True
            elif A.stop == "answer" and "</think>" in txt and answer_complete(txt.split("</think>")[-1]):
                st["done"] = True
        del past, last; clear()

    cut_by_time = sum(1 for st in S if st["n_model"] < A.gen and not st["cut"] and not st["dead"]
                      and "</think>" not in tok.decode(st["gen"]))
    if cut_by_time:
        print(f"[warn] {cut_by_time}/{B} rollouts hit the {budget}s batch budget before answering", flush=True)
    outs = []
    for b, st in enumerate(S):
        txt = tok.decode(st["gen"])
        landed = (not st["cut"]) and "</think>" in txt and bool(txt.split("</think>")[-1].strip())
        ans = head_sentence(txt.split("</think>")[-1].strip()) if landed else ""
        outs.append(dict(q_ids=QID[b], q=qs[b], gen=st["gen"], msk=st["msk"], ended=bool(st.get("ended")),
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
        loss = (ce * tm.reshape(-1)).sum() / ntot
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
    sd = {n: p.detach().to(torch.bfloat16).cpu().contiguous() for n, p in model.named_parameters()}   # full model (base + LoRA): loads standalone like the SFT ckpt
    sd.update({"pooler." + k: v.detach().float().cpu().contiguous() for k, v in pooler.A.items()})   # merged
    save_file(sd, path + ".tmp"); os.replace(path + ".tmp", path)


# ---- data ----
held = set()
for line in open(A.heldout):
    try: held.add((json.loads(line).get("q") or "").strip())
    except Exception: pass
pool = []
for line in open(A.questions):
    try: r = json.loads(line)
    except Exception: continue
    q, gold = (r.get("q") or "").strip(), (r.get("gold") or "").strip()
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
rng = random.Random(0); rng.shuffle(pool); rng.shuffle(dol); rng.shuffle(rep)
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
    head_t = tok.apply_chat_template([{"role": "user", "content": rec["q"]}], add_generation_prompt=True, tokenize=False)
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


def judge_reason(q, ref, text):
    """teacher score for one reasoning sample; the reward is 1.0 only when every box is ticked"""
    reply = text.split("</think>")[-1].strip()
    if not reply: return 0.0, {"unfinished": True}
    if A.reason_stub:                                     # shape only, for a smoke run with no credit
        ok = 8 <= len(reply.split()) <= 400 and not any(t in reply for t in TAGS)
        return (1.0 if ok else 0.0), {"stub": True}
    oai = A.judge_api == "openai"
    key = OAI if oai else DSK
    if not key: return 0.0, {"error": "no key"}
    model = A.judge_model or ("gpt-5-nano" if oai else "deepseek-flash")
    body = {"model": model,
            "messages": [{"role": "system", "content": REASON_SYS},
                         {"role": "user", "content": f"QUESTION:\n{q[:2000]}\n\nREFERENCE ANSWER:\n{ref[:3000]}\n\nASSISTANT ANSWER:\n{reply[:3000]}"}]}
    # both teachers think before answering, and a tight cap comes back as an empty message rather than an error
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
        if "choices" not in d: return 0.0, {"error": "no choices"}
        c = d["choices"][0]["message"].get("content") or ""
        try: v = json.loads(c[c.find("{"): c.rfind("}") + 1])
        except Exception: return 0.0, {"error": "unparsable"}
        return (1.0 if all(bool(v.get(k)) for k in ("solves_it", "follows_the_request", "language_english", "clean")) else 0.0), v
    return 0.0, {"error": "network"}


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
reason = []
if A.reason:
    for line in open(A.reason):
        try:
            r = json.loads(line)
            if r.get("q") and r.get("reply"): reason.append({"q": r["q"], "ref": r["reply"]})
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
    for g in opt.param_groups: g["lr"] = g["lr"] / 2
    return len(missing.unexpected_keys)

for step in range(state["step"] + 1, A.steps + 1) if not reason else []:
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
print("ONLINE_LOOP_DONE", flush=True)


# ---- GRPO on reasoning problems, the teacher scoring each sample against the reference ----
# Not the positive-only self-SFT that collapsed in tens of steps: every sample of a group is used,
# its advantage measured against the group's own mean, so a bad sample is pushed down rather than
# ignored. A group whose samples all score the same carries no signal and is skipped.
for step in range(state["step"] + 1, A.steps + 1) if reason else []:
    prob = reason[(step - 1) % len(reason)]
    if (step - 1) % A.accum == 0: opt.zero_grad(set_to_none=True)
    try:
        rolls = rollout_batch(prob["q"], A.reason_g)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        print(f"[warn] batch dropped: {type(e).__name__}: {str(e)[:120]}", flush=True); clear(); continue
    rw, notes = [], []
    with ThreadPoolExecutor(max_workers=min(8, len(rolls))) as ex:
        for r, (sc, v) in zip(rolls, ex.map(lambda x: judge_reason(prob["q"], prob["ref"], x["text"]), rolls)):
            rw.append(sc); notes.append(v)
            roll_fh.write(json.dumps({"step": step, "q": prob["q"], "reward": sc, "why": v, "ns": r["ns"],
                                      "text": r["text"]}, ensure_ascii=False) + "\n")
    roll_fh.flush()
    mu = sum(rw) / max(len(rw), 1)
    sd = (sum((x - mu) ** 2 for x in rw) / max(len(rw), 1)) ** 0.5
    losses = []
    if sd > 1e-6:
        model.train()
        for r, x in zip(rolls, rw):
            losses.append(guarded(pg_backward, r, (x - mu) / sd / (len(rolls) * A.accum))); clear()
    dl = []
    if dol and A.dolphin_min:
        model.train()
        for _ in range(A.dolphin_min):
            dl.append(guarded(plain_backward, dol[di % len(dol)], 1.0 / (A.dolphin_min * A.accum))); di += 1
    if step % A.accum == 0:
        opt.step(); opt.zero_grad(set_to_none=True); clear()
    cum["pass"] = cum.get("pass", 0) + sum(rw); cum["n"] = cum.get("n", 0) + len(rw)
    unfin = sum(1 for v in notes if v.get("unfinished")); err = sum(1 for v in notes if v.get("error"))
    line = (f"[step {step}] pass {sum(rw):.0f}/{len(rw)}"
            + (f" unfinished={unfin}" if unfin else "") + (f" judge-error={err}" if err else "")
            + f" | ce={sum(losses)/max(len(losses),1):.3f} dolphin_ce={sum(dl)/max(len(dl),1):.3f}"
            + f" | cumulative pass {100*cum['pass']/max(cum['n'],1):.1f}% of {cum['n']}"
            + f" | {(time.time()-t0)/60:.0f} min")
    print(line, flush=True); log.write(line + "\n"); log.flush()
    if step % A.save_every == 0 or step == A.steps:
        save_ckpt(LATEST); json.dump({"step": step, "cum": cum, "di": di, "ri": ri, "qi": qi}, open(STATE_F, "w"))
        print(f"[save] step {step}", flush=True)
if reason: print("ONLINE_LOOP_DONE", flush=True)
