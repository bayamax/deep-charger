#!/usr/bin/env python3
"""GRPO for the pooler lineage, in the TEACHER's recipe (grpo_ep_more.py) but with compression on.

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
import os, sys, json, time, re, ssl, random, argparse, threading, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser()
ap.add_argument("init"); ap.add_argument("outdir")
ap.add_argument("--steps", type=int, default=200); ap.add_argument("--g", type=int, default=12)
ap.add_argument("--rw", type=int, default=768); ap.add_argument("--maxd", type=int, default=384)
ap.add_argument("--chunk", type=int, default=128); ap.add_argument("--temp", type=float, default=0.9)
ap.add_argument("--gen", type=int, default=1500); ap.add_argument("--maxs", type=int, default=5); ap.add_argument("--maxm", type=int, default=8)
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
from safetensors.torch import save_file  # noqa: E402
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
print(f"[cfg] G={A.g} steps={A.steps} rw={A.rw} maxd={A.maxd} chunk={A.chunk} temp={A.temp} gen={A.gen} maxs={A.maxs} maxm={A.maxm} samepage={A.samepage} maxsrch={A.maxsrch} phantom={A.phantom}x{A.phantom_scale} "
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

    while n_model < A.gen and time.time() - t0 < 600:
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
    """B independent rollouts of the same question, decoded in lockstep.

    Semantics match rollout() exactly: every row keeps its own generation, pooled set, page offsets and stop
    conditions, and each block is rebuilt from that row's own state against a freshly prefilled question cache.
    Only the per-token decode is shared -- that is the part that leaves the GPU at ~14% utilisation when the
    rollouts are run one at a time. Rows that stop early are carried along with filler tokens; the cache is
    rebuilt from scratch every block, so their pollution never reaches a row that is still generating."""
    model.eval()
    q_ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": question}],
                                               add_generation_prompt=True, tokenize=False) + "<think>\n")
    MQ = len(q_ids)
    S = [dict(gen=[], msk=[], kept=[], absorbed=0, segs=[], n_model=0, ns_=0, nm=0, nmt=0, served=[], queries=[],
              page_ids=[], page_off=0, seen_pages={}, cur_key=None, nrep=0, dead=False, cut=False, done=False)
         for _ in range(B)]
    # Wall clock, not work: the rows run together, so the batch needs about what one rollout needed. 600 s was
    # nonetheless too tight while the page lookups were serial, and cutting rows off mid answer fed the gradient
    # failures the policy had not caused. 900 s with the lookups overlapped leaves room without hiding a stall.
    budget = 900
    t0 = time.time()

    def inject(st, text):
        ids = tok.encode(text, add_special_tokens=False)
        st["gen"].extend(ids); st["msk"].extend([0] * len(ids))

    def advance(st, nx):
        """one decoded token for one row; returns True when this row's block ends here (verbatim rollout() order)"""
        if nx == eos:
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
        if "</think>" in txt and answer_complete(txt.split("</think>")[-1]):
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
        amask[:, :MQ] = 1
        for i, blk in enumerate(blocks):               # left-pad: every row's last real token lands on index -1
            L = Ls[i]
            emb_b[i, Lmax - L:] = blk[0]
            posv[i, Lmax - L:] = torch.arange(MQ, MQ + L, device=DEV)
            amask[i, MQ + Lmax - L:] = 1
        past = DynamicCache()
        BODY(input_ids=torch.tensor([q_ids] * nA, device=DEV), past_key_values=past, use_cache=True)
        cpos = torch.arange(MQ, MQ + Lmax, device=DEV)
        last = last_logits(inputs_embeds=emb_b, past_key_values=past, attention_mask=amask,
                           position_ids=posv, cache_position=cpos, use_cache=True)
        npos = [MQ + L for L in Ls]
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
            if st["dead"] or st["cut"] or (st["gen"] and st["gen"][-1] == eos):
                st["done"] = True
            elif "</think>" in txt and answer_complete(txt.split("</think>")[-1]):
                st["done"] = True
        del past, last; clear()

    cut_by_time = sum(1 for st in S if st["n_model"] < A.gen and not st["cut"] and not st["dead"]
                      and "</think>" not in tok.decode(st["gen"]))
    if cut_by_time:
        print(f"[warn] {cut_by_time}/{B} rollouts hit the {budget}s batch budget before answering", flush=True)
    outs = []
    for st in S:
        txt = tok.decode(st["gen"])
        landed = (not st["cut"]) and "</think>" in txt and bool(txt.split("</think>")[-1].strip())
        ans = head_sentence(txt.split("</think>")[-1].strip()) if landed else ""
        outs.append(dict(q_ids=q_ids, gen=st["gen"], msk=st["msk"],
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


# ---- data: the teacher's pool, minus the held-out questions ----
held = set()
for line in open(A.heldout):
    try: held.add((json.loads(line).get("q") or "").strip())
    except Exception: pass
pool = []
for line in open(A.corpus):
    try: r = json.loads(line)
    except Exception: continue
    q, gold = (r.get("q") or "").strip(), (r.get("gold") or "").strip()
    if q and gold and len(gold.split()) <= 6:
        pool.append({"q": q, "gold": gold})
rng = random.Random(0); rng.shuffle(pool)
n_all = len(pool); pool = [x for x in pool if x["q"] not in held]
print(f"[data] {n_all} questions, {n_all - len(pool)} held-out removed -> {len(pool)}", flush=True)

if A.selftest_batch:
    # Greedy makes the policy deterministic, so the batched rollout must reproduce the single one token for token.
    # B=1 is the discriminator: it exercises every line of the batched path with no padding and no cross-row
    # arithmetic, so a mismatch there is a bug in the code and a mismatch only at B>1 is padding or numerics.
    GREEDY_PICK = True
    allok = True
    for qi in range(2):
        q = pool[qi]["q"]
        t = time.time(); a = rollout(q); t1 = time.time() - t
        one = rollout_batch(q, 1)[0]
        same1 = one["gen"] == a["gen"]
        d1 = next((i for i in range(min(len(a["gen"]), len(one["gen"]))) if a["gen"][i] != one["gen"][i]),
                  min(len(a["gen"]), len(one["gen"])))
        print(f"  q{qi} B=1 identical: {same1}" + ("" if same1 else f" (first difference at token {d1} of {len(a['gen'])}/{len(one['gen'])})"), flush=True)
        if not same1:
            print(f"    single: {tok.decode(a['gen'][max(0, d1 - 15):d1 + 25])!r}", flush=True)
            print(f"    batch : {tok.decode(one['gen'][max(0, d1 - 15):d1 + 25])!r}", flush=True)
        t = time.time(); bs = rollout_batch(q, A.selftest_batch); t2 = time.time() - t
        match = sum(1 for b in bs if b["gen"] == a["gen"])
        if match < len(bs):
            b = next(b for b in bs if b["gen"] != a["gen"])
            n = min(len(a["gen"]), len(b["gen"]))
            d = next((i for i in range(n) if a["gen"][i] != b["gen"][i]), n)
            print(f"    B={A.selftest_batch} first difference at token {d} of {len(a['gen'])}/{len(b['gen'])}: "
                  f"{tok.decode([a['gen'][d]])!r} vs {tok.decode([b['gen'][d]])!r}", flush=True)
        dr = logit_drift(q, A.selftest_batch)
        print(f"    logit drift batch 1 vs {A.selftest_batch}: max |delta| {dr['maxabs']:.4f} | KL at temp {A.temp} "
              f"{dr['kl']:.2e} nats | same argmax {dr['argmax_same']} | same top-200 order {dr['top200_same']}/200", flush=True)
        # The code is correct when a batch of one reproduces the single path exactly. Beyond that, only the size of
        # the numerical difference matters. 1e-2 nats is a sampled-probability ratio of 1.01 between the batch the
        # rollout drew from and the batch of one the gradient scores it with, against the 1.2 that PPO-style
        # methods normally allow.
        allok &= same1 and dr["kl"] < 1e-2
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"  q{qi}: single {t1:.1f}s | batch x{A.selftest_batch} {t2:.1f}s ({A.selftest_batch * t1 / max(t2, 1e-9):.1f}x) "
              f"| rows matching the single rollout {match}/{len(bs)} | peak {peak:.1f} GiB", flush=True)
        torch.cuda.reset_peak_memory_stats(); clear()
    print("BATCH_SELFTEST " + ("PASS" if allok else "FAIL"), flush=True)
    raise SystemExit(0 if allok else 1)

log = open(os.path.join(A.outdir, "grpo.log"), "a")
roll_fh = open(os.path.join(A.outdir, "rollouts.jsonl"), "a")
hist = list(state.get("hist", []))
t0 = time.time()
for step in range(state["step"] + 1, A.steps + 1):
    item = pool[step % len(pool)]
    opt.zero_grad(set_to_none=True)
    rolls = []
    if A.batch > 1:
        while len(rolls) < A.g:
            k = min(A.batch, A.g - len(rolls))
            try:
                got = rollout_batch(item["q"], k)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                print(f"[warn] batch dropped ({k}): {type(e).__name__}: {str(e)[:120]}", flush=True); clear(); break
            rolls.extend(got)
            if not got:
                break
    else:
        for _ in range(A.g):
            try:
                rolls.append(rollout(item["q"]))
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                print(f"[warn] rollout dropped: {type(e).__name__}: {str(e)[:120]}", flush=True); clear()
    if not rolls:
        continue
    rews, infos = [], []
    for r in rolls:
        rw_, c, g = price(r, item["gold"]); rews.append(rw_); infos.append((r["ns"], g, r["landed"], c, r["more"], r.get("rep", 0)))
        roll_fh.write(json.dumps({"step": step, "q": item["q"], "gold": item["gold"], "queries": r["queries"], "answer": r["answer"][:120],
                                  "grounded": g, "landed": r["landed"], "correct": c, "more": r["more"], "rep": r.get("rep", 0), "cut": r.get("cut", False), "dead": r["dead"],
                                  "text": r["text"]}, ensure_ascii=False) + "\n")
    roll_fh.flush()
    stat = rews + ([A.phantom] if A.phantom > 0 else [])          # phantom sample: shifts the baseline for degenerate groups
    mu = sum(stat) / len(stat); sd = (sum((x - mu) ** 2 for x in stat) / len(stat)) ** 0.5
    degenerate = (max(rews) - min(rews)) < 1e-6
    scale = A.phantom_scale if degenerate else 1.0
    skipped = sd < 1e-6; gnorm = 0.0; losses = []
    if not skipped:
        model.train()
        for i, w in choose_replay(rews, [x[1] for x in infos], A.backprop):
            adv = scale * (rews[i] - mu) / (sd + 1e-6)
            if abs(adv) < 1e-6:
                continue
            losses.append(pg_backward(rolls[i], w * adv / A.g))
            clear()
        params = [p for p in model.parameters() if p.requires_grad and p.grad is not None] + [p for p in pooler_params if p.grad is not None]
        gnorm = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params)).item()) if params else 0.0
        opt.step(); opt.zero_grad(set_to_none=True); clear()
    n = len(infos)
    corr = sum(1 for i in infos if i[3]) / n; gnd = sum(1 for i in infos if i[1]) / n; land = sum(1 for i in infos if i[2]) / n
    hist.append(corr)
    line = (f"[step {step}] correct={corr:.0%} grounded={gnd:.0%} landed={land:.0%} ema={sum(hist[-25:])/max(len(hist[-25:]),1):.0%} "
            f"reward={mu:+.2f} srch={sum(i[0] for i in infos)/n:.1f} more={sum(i[4] for i in infos)/n:.2f} rep={sum(i[5] for i in infos)/n:.2f} "
            f"ce={sum(losses)/max(len(losses),1):.3f} |grad|={gnorm:.4f} g={n} bp={len(losses)} skip={int(skipped)} deg={int(degenerate)} elapsed={(time.time()-t0)/60:.0f}m")
    print(line, flush=True); log.write(line + "\n"); log.flush()
    if step % A.save_every == 0 or step == A.steps:
        save_ckpt(LATEST)
        if step % 100 == 0 or step == A.steps:
            save_ckpt(os.path.join(A.outdir, f"ckpt_step{step}.safetensors"))
        json.dump({"step": step, "hist": hist[-200:]}, open(STATE_F, "w"))
        print(f"[save] step {step}", flush=True)
print("GRPO_POOL_DONE", flush=True)
