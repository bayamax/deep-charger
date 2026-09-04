---SRC-BEGIN---
#!/usr/bin/env python3
"""CUDA port of grpo_ep.py — whole-episode GRPO in the deployed configuration.

Same environment, reward, and guards as the MLX original; only the model/gradient layer moves to
torch. The base model is bf16_s100: mlx4_final dequantized (so the weights are numerically what the
MLX run trained on) with the step-100 adapter already folded in. Fresh LoRA (r16, top 8 layers,
scale 20 to match mlx) trains on top.

Positions are explicit on every forward: HF's auto-derivation after cache manipulation caused the
June RoPE-drift bug, and explicit positions are simply always safe.

  python3 grpo_ep_torch.py [steps]
"""
import json
import os
import random
import re
import ssl
import sys
import threading
import time
import urllib.parse
import urllib.request

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
MODEL = os.environ.get("GRPO_MODEL", os.path.join(HERE, "bf16_s100"))
OUTD = os.environ.get("GRPO_OUT", os.path.join(HERE, "ep_grpo"))
RESUME = os.environ.get("GRPO_RESUME", "")            # peft adapter dir, or empty
G = int(os.environ.get("GRPO_G", "12"))
B = int(os.environ.get("GRPO_B", "1"))
MAXS = int(os.environ.get("GRPO_MAXS", "5"))
GEN = int(os.environ.get("GRPO_GEN", "1500"))
TEMP = float(os.environ.get("GRPO_TEMP", "0.9"))
LR = float(os.environ.get("GRPO_LR", "1e-5"))
PAGE_STEP = 256
CHUNK = 256
DEV = "cuda"
os.makedirs(OUTD, exist_ok=True)

WAPI = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "deep-charger-grpo-ep/1.0 (research; bayamax@icloud.com)"}
CTX = ssl.create_default_context()
try:
    import certifi
    CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    pass


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()


def has(t, g):
    return (" " + norm(g) + " ") in (" " + norm(t) + " ")


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
    a = m.group(1)
    for mm in re.finditer(r"[.!?](?=\s|$)", a):
        t = re.split(r"[\s(\"]", a[:mm.start()])[-1]
        if len(t) == 1 or t.lower() in ABBR:
            continue
        return True
    return False


# ---------------- retrieval -----------------------------------------------------------------
CACHE_F = os.path.join(HERE, "grpo_ep_cache.jsonl")
CACHE_CAP = 6000
cache, cache_lock = {}, threading.Lock()
for f in [p for p in os.listdir(HERE) if "cache" in p and p.endswith(".jsonl")]:
    for line in open(os.path.join(HERE, f)):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "page" in r:
            cache[r["kw"]] = r["page"]
cache_fh = open(CACHE_F, "a")
while len(cache) > CACHE_CAP:
    cache.pop(next(iter(cache)))
print(f"[cache] {len(cache)} pages warm (capped)", flush=True)


def api(params, tries=3):
    url = WAPI + "?" + urllib.parse.urlencode({**params, "maxlag": 5, "format": "json"})
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=20, context=CTX) as resp:
                out = json.loads(resp.read().decode())
            if isinstance(out, dict) and out.get("error", {}).get("code") == "maxlag":
                time.sleep(5 * (i + 1))
                continue
            time.sleep(1.1)
            return out
        except Exception:
            time.sleep(3)
    return {}


def fetch(kw):
    sd = api({"action": "query", "list": "search", "srsearch": kw, "srlimit": 3})
    hits = [h["title"] for h in sd.get("query", {}).get("search", [])][:3]
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
            full = api({"action": "query", "prop": "extracts", "explaintext": 1,
                        "redirects": 1, "titles": t})
            for p in full.get("query", {}).get("pages", {}).values():
                if p.get("extract"):
                    return f"{t}: {p['extract'][:40000]}"
            return f"{t}: {pages[t]}"
    return ""


def get_page(kw):
    if kw in cache:
        return cache[kw]
    page = fetch(kw)
    with cache_lock:
        cache[kw] = page
        while len(cache) > CACHE_CAP:
            cache.pop(next(iter(cache)))
        cache_fh.write(json.dumps({"kw": kw, "page": page}, ensure_ascii=False) + "\n")
        cache_fh.flush()
    return page


def serve(kw, ask):
    """Four-arm test settled it: the page's head chunk, no reader aiming."""
    pg = get_page(kw)
    if not pg:
        return "(no results)", "(no extraction)", ""
    ids = tokenizer.encode(pg, add_special_tokens=False)
    return tokenizer.decode(ids[:PAGE_STEP]), "(no extraction)", pg


# ---------------- policy --------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV)
model.config.use_cache = True
for p in model.parameters():
    p.requires_grad_(False)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
if RESUME:
    model = PeftModel.from_pretrained(model, RESUME, is_trainable=True)
    print(f"[resume] {RESUME}", flush=True)
else:
    lcfg = LoraConfig(r=16, lora_alpha=320, lora_dropout=0.0, bias="none",
                      target_modules=TARGETS, layers_to_transform=list(range(20, 28)),
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[model] {MODEL} + LoRA, {n_train/1e6:.1f}M trainable", flush=True)
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
EOS = tokenizer.eos_token_id
NOTICE = "(no searches left - answer from what you have read)"
CLOSE = re.compile(r"<search>(.*?)</\s*search\s*[^\w<]{0,3}$", re.S)
TAG = "<information"


def pick(logits, txt_tail, temp):
    """Sample, but the policy may never write its own information block."""
    lg = logits.float()
    for _ in range(8):
        p = torch.softmax(lg / temp, dim=-1)
        t = int(torch.multinomial(p, 1).item())
        cand = txt_tail + tokenizer.decode([t])
        if TAG in cand or any(cand.endswith(TAG[:k]) for k in range(4, len(TAG) + 1)):
            lg[t] = -1e9
            continue
        return t
    return int(torch.argmax(lg).item())


@torch.no_grad()
def episode(prompt_ids, question, gold):
    past = DynamicCache()
    ids, mask = list(prompt_ids), [0] * len(prompt_ids)
    arr = torch.tensor([ids], device=DEV)
    pos = torch.arange(0, len(ids), device=DEV)
    out = model(input_ids=arr, past_key_values=past, position_ids=pos.unsqueeze(0),
                cache_position=pos, use_cache=True)
    logits = out.logits[0, -1, :]
    npos = len(ids)
    txt, ns, served, hints, queries_out, page_gold = "", 0, [], [], [], []
    n_gen = 0
    while n_gen < GEN:
        tok_id = pick(logits, txt[-16:] if txt else "", TEMP)
        if tok_id == EOS:
            break
        ids.append(tok_id)
        mask.append(1)
        n_gen += 1
        txt = tokenizer.decode(ids[len(prompt_ids):])
        _si = txt.rfind("<search>")
        mclose = CLOSE.search(txt, _si) if _si >= 0 else None
        if mclose and ns == len(queries_out) and txt.count("<search>") > ns:
            ns += 1
            body = mclose.group(1).strip()
            kw, ask = ([x.strip() for x in body.split("||", 1)] if "||" in body
                       else (body, body))
            queries_out.append(kw)
            if not kw:
                blk = "\n<information>(no results)</information>\n"
            elif ns > MAXS:
                blk = f"\n<information>{NOTICE}</information>\n"
            else:
                chunk, span, page_full = serve(kw, ask)
                served.append(chunk)
                hints.append(span)
                page_gold.append(has(page_full, gold))
                blk = f"\n<information>\n{chunk}\n[READER] {span}\n</information>\n"
            new = tokenizer.encode(blk, add_special_tokens=False)
            ids += new
            mask += [0] * len(new)
            narr = torch.tensor([new], device=DEV)
            npo = torch.arange(npos, npos + len(new), device=DEV)
            out = model(input_ids=narr, past_key_values=past, position_ids=npo.unsqueeze(0),
                        cache_position=npo, use_cache=True)
            logits = out.logits[0, -1, :]
            npos += len(new)
            continue
        if "</think>" in txt and answer_complete(txt.split("</think>")[-1]):
            break
        narr = torch.tensor([[tok_id]], device=DEV)
        npo = torch.tensor([npos], device=DEV)
        out = model(input_ids=narr, past_key_values=past, position_ids=npo.unsqueeze(0),
                    cache_position=npo, use_cache=True)
        logits = out.logits[0, -1, :]
        npos += 1

    grounded = any(has(s + " " + h, gold) for s, h in zip(served, hints))
    landed = "</think>" in txt and bool(txt.split("</think>")[-1].strip())
    ans = head_sentence(txt.split("</think>")[-1].strip()) if landed else ""
    correct = landed and has(ans, gold)
    bridge = any(page_gold[1:])
    return ids, mask, (ns, grounded, landed, correct, bridge), \
        {"queries": queries_out, "answer": (ans if landed else ""), "hints": hints[:5],
         "text": tokenizer.decode(ids[len(prompt_ids):])}


def price(group):
    """Grounded-correct is 1.0. Everything else is 0.0. No partial credit, under any condition."""
    return [1.0 if (g and c) else 0.0 for _, g, _, c, _ in group]


BODY = model.base_model.model.model          # peft -> causal lm -> transformer body
HEAD = model.base_model.model.lm_head


def logp_masked(ids, mask, n_prompt):
    """Mean log p over the tokens the policy chose; lm_head applied in chunks."""
    arr = torch.tensor([ids], device=DEV)
    pos = torch.arange(0, len(ids) - 1, device=DEV).unsqueeze(0)
    h = BODY(input_ids=arr[:, :-1], position_ids=pos, use_cache=False).last_hidden_state
    total, count = None, 0
    for start in range(n_prompt - 1, len(ids) - 1, CHUNK):
        end = min(start + CHUNK, len(ids) - 1)
        sel = [i for i in range(start, end) if mask[i + 1]]
        if not sel:
            continue
        logits = HEAD(h[:, start:end, :]).float()
        targets = torch.tensor([ids[start + 1:end + 1]], device=DEV)
        lse = torch.logsumexp(logits, dim=-1)
        picked = torch.gather(logits, 2, targets.unsqueeze(-1))[..., 0]
        lp = (picked - lse)[0]
        keep = torch.tensor([1.0 if mask[i + 1] else 0.0 for i in range(start, end)],
                            device=DEV)
        part = (lp * keep).sum()
        total = part if total is None else total + part
        count += len(sel)
    if total is None:
        return torch.zeros((), device=DEV, requires_grad=True)
    return total / max(count, 1)


# ---------------- data ----------------------------------------------------------------------
pool = []
for line in open(os.path.join(HERE, "corpus_box_final.jsonl")):
    try:
        r = json.loads(line)
    except Exception:
        continue
    q, gold = (r.get("q") or "").strip(), (r.get("gold") or "").strip()
    if q and gold and len(gold.split()) <= 6:
        pool.append({"q": q, "gold": gold})
rng = random.Random(0)
rng.shuffle(pool)
print(f"[data] {len(pool)} questions", flush=True)

log = open(os.path.join(OUTD, "grpo.log"), "a")
ROLL_F = os.path.join(OUTD, "rollouts.jsonl")
ROLL_CAP = int(os.environ.get("GRPO_ROLL_CAP_MB", "400")) * 1024 * 1024


def rotate(fh):
    if fh.tell() < ROLL_CAP:
        return fh
    fh.close()
    os.replace(ROLL_F, ROLL_F + ".prev")
    return open(ROLL_F, "a")


roll_fh = open(ROLL_F, "a")
hist = []
t0 = time.time()
for step in range(1, STEPS + 1):
    step_r, step_info = [], []
    skipped = 0
    opt.zero_grad(set_to_none=True)
    any_grad = False
    for b in range(B):
        item = pool[(step * B + b) % len(pool)]
        head = tokenizer.apply_chat_template([{"role": "user", "content": item["q"]}],
                                             add_generation_prompt=True, tokenize=False)
        if not head.rstrip().endswith("<think>"):
            head += "<think>\n"
        prompt_ids = tokenizer.encode(head)
        eps = []
        for _ in range(G):
            try:
                eps.append(episode(prompt_ids, item["q"], item["gold"]))
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                print(f"[warn] rollout dropped: {type(e).__name__}: {e}", flush=True)
        if not eps:
            continue
        info = [e[2] for e in eps]
        for (_, _, inf, trace) in eps:
            roll_fh.write(json.dumps({
                "step": step, "q": item["q"], "gold": item["gold"],
                "queries": trace["queries"], "answer": trace["answer"][:120],
                "hints": trace["hints"], "grounded": inf[1], "landed": inf[2],
                "correct": inf[3], "trusted": inf[4],
                "text": trace["text"]}, ensure_ascii=False) + "\n")
        roll_fh.flush()
        roll_fh = rotate(roll_fh)
        rews = price(info)
        step_r += rews
        step_info += info
        mu = sum(rews) / len(rews)
        sd = (sum((x - mu) ** 2 for x in rews) / len(rews)) ** 0.5
        if sd < 1e-6:
            skipped += 1
            continue
        for (ids, mask, _, _), adv in zip(eps, [(x - mu) / (sd + 1e-6) for x in rews]):
            if abs(adv) < 1e-6:
                continue
            loss = (-adv / (G * B)) * logp_masked(ids, mask, len(prompt_ids))
            loss.backward()
            any_grad = True
            torch.cuda.empty_cache()
    gnorm = 0.0
    if any_grad:
        gnorm = float(torch.sqrt(sum((p.grad.float() ** 2).sum()
                                     for p in model.parameters()
                                     if p.requires_grad and p.grad is not None)).item())
        opt.step()
    n = max(len(step_info), 1)
    corr = sum(1 for i in step_info if i[3]) / n
    gnd = sum(1 for i in step_info if i[1]) / n
    land = sum(1 for i in step_info if i[2]) / n
    bridge_n = sum(1 for i in step_info if i[4])
    hist.append(corr)
    line = (f"[step {step}] correct={corr:.0%} grounded={gnd:.0%} landed={land:.0%} "
            f"bridge={bridge_n} "
            f"ema={sum(hist[-25:])/max(len(hist[-25:]),1):.0%} "
            f"reward={sum(step_r)/max(len(step_r),1):+.2f} "
            f"srch={sum(i[0] for i in step_info)/n:.1f} |grad|={gnorm:.4f} "
            f"skip={skipped}/{B} cache={len(cache)} "
            f"elapsed={(time.time()-t0)/60:.0f}m")
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()
    if step % 20 == 0:
        model.save_pretrained(os.path.join(OUTD, "adapter"))
        print(f"[save] adapter at step {step}", flush=True)

print("GRPO_EP_DONE", flush=True)
---SRC-END---
