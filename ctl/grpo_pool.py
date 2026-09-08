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
import os, sys, json, time, re, ssl, random, argparse, urllib.parse, urllib.request
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
ap.add_argument("--gradckpt", type=int, default=1, help="1: gradient checkpointing through the transformer during the policy-gradient pass (16GB cards)")
ap.add_argument("--samepage", type=int, default=1, help="1: a search whose top page was already shown in this rollout serves the NEXT chunk of that page (and says so when the page is used up); 0: teacher environment (always the head)")
A = ap.parse_args()
os.makedirs(A.outdir, exist_ok=True)
os.environ.setdefault("SP_RANK", "128"); os.environ.setdefault("SP_NOSYS", "1"); os.environ.setdefault("SP_EPISODIC", "1")
os.environ["SP_TRAIN_POOLER"] = "1"; os.environ["SP_LR"] = str(A.lr); os.environ["SP_POOLER_LR"] = str(A.pooler_lr)
# resume decides which weights the harness prefix loads
STATE_F = os.path.join(A.outdir, "state.json"); LATEST = os.path.join(A.outdir, "latest.safetensors")
state = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {"step": 0}
init_path = LATEST if (state["step"] > 0 and os.path.exists(LATEST)) else A.init
os.environ["SP_INIT_FULL"] = init_path
import torch  # noqa: E402
import numpy as np  # noqa: E402

# ---- model + pooler + optimizer + pg_grad_backward via the harness prefix (same wrapping as the evaluator) ----
F = "/root/work/grpo_e2e_torch.py"
src = open(F).read().split("\n")
cut = next(i for i, l in enumerate(src) if l.startswith("if MULTI > 1:"))
sys.argv = ["grpo_e2e_torch.py", "0", str(A.g), "0", str(A.gen)]
sys.path.insert(0, "/root/work")
ns = {"__name__": "grpo_pool", "__file__": F}
exec(compile("\n".join(src[:cut]), F, "exec"), ns)
model, tok, pooler, opt = ns["model"], ns["tok"], ns["pooler"], ns["opt"]
emb, sp, crop_cache, pick, _ngrams = ns["emb"], ns["sp"], ns["crop_cache"], ns["pick"], ns["_ngrams"]
pg_grad_backward, clear = ns["pg_grad_backward"], ns["clear"]
DEV, eos = ns["DEV"], ns["eos"]
ns["TEMP"] = A.temp; ns["MAXD"] = A.maxd; ns["C"] = A.chunk; ns["RWG"] = A.rw; ns["GREEDY"] = False
from transformers import DynamicCache  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
print(f"[init] weights <- {init_path} (resume step {state['step']})", flush=True)
CLM = model.base_model.model            # peft -> causal LM
BODY, HEAD = CLM.model, CLM.lm_head     # transformer body, lm_head
if A.gradckpt:
    CLM.config.use_cache = False
    CLM.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("[init] gradient checkpointing ON for the policy-gradient pass", flush=True)
nT = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
nP = sum(v.numel() for v in pooler.A.values()) / 1e6
print(f"[cfg] G={A.g} steps={A.steps} rw={A.rw} maxd={A.maxd} chunk={A.chunk} temp={A.temp} gen={A.gen} maxs={A.maxs} maxm={A.maxm} samepage={A.samepage} maxsrch={A.maxsrch} "
      f"lr={A.lr} pooler_lr={A.pooler_lr} trainable lora={nT:.1f}M pooler={nP:.1f}M", flush=True)
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


def get_page(kw):
    if kw in cache:
        return cache[kw]
    page = fetch(kw); cache[kw] = page
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
@torch.no_grad()
def pick_plain(logits, gen):
    """verbatim grpo_ep_more.pick(): temperature sampling; the policy may never write its own information block."""
    lg = logits.float()[0].clone(); tail = tok.decode(gen[-16:]) if gen else ""
    for _ in range(8):
        t = int(torch.multinomial(torch.softmax(lg / A.temp, dim=-1), 1).item())
        cand = tail + tok.decode([t])
        if TAG in cand or any(cand.endswith(TAG[:k]) for k in range(4, len(TAG) + 1)):
            lg[t] = -1e9; continue
        return t
    return int(torch.argmax(lg).item())



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


def price(r, gold):
    """teacher's price(): grounded-correct is 1.0, everything else 0.0."""
    correct = r["landed"] and has(r["answer"], gold)
    grounded = any(has(s, gold) for s in r["served"])
    return (1.0 if (correct and grounded) else 0.0), bool(correct), bool(grounded)


def save_ckpt(path):
    sd = {n: p.detach().to(torch.bfloat16).cpu().contiguous() for n, p in model.named_parameters()}   # full model (base + LoRA): loads standalone like the SFT ckpt
    sd.update({"pooler." + k: v.detach().float().cpu().contiguous() for k, v in pooler.A.items()})
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

log = open(os.path.join(A.outdir, "grpo.log"), "a")
roll_fh = open(os.path.join(A.outdir, "rollouts.jsonl"), "a")
hist = list(state.get("hist", []))
t0 = time.time()
for step in range(state["step"] + 1, A.steps + 1):
    item = pool[step % len(pool)]
    opt.zero_grad(set_to_none=True)
    rolls = []
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
    mu = sum(rews) / len(rews); sd = (sum((x - mu) ** 2 for x in rews) / len(rews)) ** 0.5
    skipped = sd < 1e-6; gnorm = 0.0; losses = []
    if not skipped:
        model.train()
        for r, rw_ in zip(rolls, rews):
            adv = (rw_ - mu) / (sd + 1e-6)
            if abs(adv) < 1e-6:
                continue
            losses.append(pg_backward(r, adv / A.g))
            clear()
        params = [p for p in model.parameters() if p.requires_grad and p.grad is not None] + [p for p in pooler.parameters() if p.grad is not None]
        gnorm = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params)).item()) if params else 0.0
        opt.step(); opt.zero_grad(set_to_none=True); clear()
    n = len(infos)
    corr = sum(1 for i in infos if i[3]) / n; gnd = sum(1 for i in infos if i[1]) / n; land = sum(1 for i in infos if i[2]) / n
    hist.append(corr)
    line = (f"[step {step}] correct={corr:.0%} grounded={gnd:.0%} landed={land:.0%} ema={sum(hist[-25:])/max(len(hist[-25:]),1):.0%} "
            f"reward={mu:+.2f} srch={sum(i[0] for i in infos)/n:.1f} more={sum(i[4] for i in infos)/n:.2f} rep={sum(i[5] for i in infos)/n:.2f} "
            f"ce={sum(losses)/max(len(losses),1):.3f} |grad|={gnorm:.4f} skip={int(skipped)} elapsed={(time.time()-t0)/60:.0f}m")
    print(line, flush=True); log.write(line + "\n"); log.flush()
    if step % A.save_every == 0 or step == A.steps:
        save_ckpt(LATEST)
        if step % 100 == 0 or step == A.steps:
            save_ckpt(os.path.join(A.outdir, f"ckpt_step{step}.safetensors"))
        json.dump({"step": step, "hist": hist[-200:]}, open(STATE_F, "w"))
        print(f"[save] step {step}", flush=True)
print("GRPO_POOL_DONE", flush=True)
