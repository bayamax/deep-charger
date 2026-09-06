#!/usr/bin/env python3
"""Parallel held-out evaluation for the GRPO search policy.

Runs the *same* episode loop, retrieval and scorer as grpo_ep_torch.py, but with no
gradient, several rollouts in flight at once, and a shard switch so more than one
machine can work on the same question set.

Why threads help here: a single rollout spends roughly a third of its wall clock
waiting on the Wikipedia API and the rest decoding one token at a time, which leaves
the GPU at about 30% utilisation. Overlapping rollouts fills both gaps. One model is
shared by every worker, so extra workers cost only their KV cache.

The API rate is capped globally rather than per call, so raising EVAL_THREADS does not
raise the request rate. Keep EVAL_RPS polite: the User-Agent identifies a real person.

    # CUDA box, 6 workers, first half of the questions
    EVAL_THREADS=6 EVAL_SHARD=0/2 python3 eval_heldout_parallel.py

    # Apple silicon, 2 workers, second half
    EVAL_THREADS=2 EVAL_SHARD=1/2 python3 eval_heldout_parallel.py

Each shard writes its own jsonl; merge them by concatenation. Re-running a shard skips
questions already present in its output file, so an interrupted run can be resumed.
"""
import json
import os
import queue
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

HERE = os.environ.get("EVAL_HOME", os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("GRPO_MODEL", os.path.join(HERE, "bf16_s100"))
ADAPTER = os.environ.get("GRPO_RESUME", "")
CORPUS = os.environ.get("EVAL_CORPUS", os.path.join(HERE, "corpus_box_final.jsonl"))
OUT_DIR = os.environ.get("EVAL_OUT", HERE)

MAXS = int(os.environ.get("GRPO_MAXS", "5"))
GEN = int(os.environ.get("GRPO_GEN", "1500"))
TEMP = float(os.environ.get("GRPO_TEMP", "0.9"))
PAGE_STEP = 256

N_Q = int(os.environ.get("EVAL_N", "300"))
G = int(os.environ.get("EVAL_G", "2"))
SKIP = int(os.environ.get("EVAL_SKIP", "380"))
MATCH = os.environ.get("EVAL_MATCH", "1") != "0"
THREADS = int(os.environ.get("EVAL_THREADS", "4"))
RPS = float(os.environ.get("EVAL_RPS", "3.0"))
SHARD = os.environ.get("EVAL_SHARD", "0/1")


def pick_device():
    want = os.environ.get("EVAL_DEVICE", "")
    if want:
        return want
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEV = pick_device()
DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
    os.environ.get("EVAL_DTYPE", "bf16")]

si, sn = (int(x) for x in SHARD.split("/"))
assert 0 <= si < sn, "EVAL_SHARD must be i/n with 0 <= i < n"
# EVAL_ALL=1: walk the whole shuffled corpus in order (a generation pass, not a held-out probe);
# EVAL_EXCLUDE: jsonl/txt of questions that must never be generated for (the held-out set).
ALL = os.environ.get("EVAL_ALL", "0") == "1"
EXCLUDE_F = os.environ.get("EVAL_EXCLUDE", "")
TAG = os.environ.get("EVAL_TAG", "heldout")
OUT_F = os.path.join(OUT_DIR, f"eval_{TAG}_p{si}of{sn}.jsonl")

# ---------------- scoring (verbatim from grpo_ep_torch.py) ---------------------------


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


# ---------------- retrieval ---------------------------------------------------------
WAPI = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "deep-charger-grpo-ep/1.0 (research; bayamax@icloud.com)"}
CTX = ssl.create_default_context()
try:
    import certifi
    CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    pass

CACHE_F = os.path.join(HERE, "grpo_ep_cache.jsonl")
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
print(f"[cache] {len(cache)} pages warm", flush=True)


class RateLimiter:
    """Global cap on request starts. Concurrency does not raise the request rate."""

    def __init__(self, rps):
        self.gap = 1.0 / max(rps, 0.01)
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self):
        with self.lock:
            t = max(time.monotonic(), self.next_at)
            self.next_at = t + self.gap
        d = t - time.monotonic()
        if d > 0:
            time.sleep(d)


limiter = RateLimiter(RPS)


def api(params, tries=3):
    url = WAPI + "?" + urllib.parse.urlencode({**params, "maxlag": 5, "format": "json"})
    for i in range(tries):
        limiter.wait()
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=20, context=CTX) as resp:
                out = json.loads(resp.read().decode())
            if isinstance(out, dict) and out.get("error", {}).get("code") == "maxlag":
                time.sleep(5 * (i + 1))
                continue
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


inflight, inflight_lock = {}, threading.Lock()


def get_page(kw):
    """One fetch per keyword even when several rollouts want it at the same time."""
    with cache_lock:
        if kw in cache:
            return cache[kw]
    with inflight_lock:
        ev = inflight.get(kw)
        mine = ev is None
        if mine:
            ev = inflight[kw] = threading.Event()
    if not mine:
        ev.wait(120)
        with cache_lock:
            return cache.get(kw, "")
    try:
        page = fetch(kw)
        with cache_lock:
            cache[kw] = page
            cache_fh.write(json.dumps({"kw": kw, "page": page}, ensure_ascii=False) + "\n")
            cache_fh.flush()
        return page
    finally:
        with inflight_lock:
            inflight.pop(kw, None)
        ev.set()


def serve(kw, ask):
    """Four-arm test settled it: the page's head chunk, no reader aiming."""
    pg = get_page(kw)
    if not pg:
        return "(no results)", "(no extraction)", ""
    ids = tokenizer.encode(pg, add_special_tokens=False)
    return tokenizer.decode(ids[:PAGE_STEP]), "(no extraction)", pg


# ---------------- policy ------------------------------------------------------------
print(f"[device] {DEV} dtype={DTYPE}", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE).to(DEV)
model.config.use_cache = True
for p in model.parameters():
    p.requires_grad_(False)
if ADAPTER:
    model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=False)
    print(f"[adapter] {ADAPTER}", flush=True)
else:
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=320, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=list(range(20, 28)), task_type="CAUSAL_LM"))
    print("[adapter] none (fresh LoRA)", flush=True)
model.eval()

EOS = tokenizer.eos_token_id
NOTICE = "(no searches left - answer from what you have read)"
CLOSE = re.compile(r"<search>(.*?)</\s*search\s*[^\w<]{0,3}$", re.S)
TAG = "<information"
# Concurrent forward passes on one shared module produce NaN logits and rollouts die in
# torch.multinomial. Measured on CUDA with 6 workers: 47 of 50 rollouts lost. So the
# forward is serialised on every backend, not just mps. Threads still overlap the
# retrieval waits, which is where the idle time is; for more than that, run several
# processes, each with its own copy of the model, on different shards.
gpu_lock = threading.Lock()


def fwd(**kw):
    with gpu_lock:
        return model(**kw)


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
    ids = list(prompt_ids)
    arr = torch.tensor([ids], device=DEV)
    pos = torch.arange(0, len(ids), device=DEV)
    out = fwd(input_ids=arr, past_key_values=past, position_ids=pos.unsqueeze(0),
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
            narr = torch.tensor([new], device=DEV)
            npo = torch.arange(npos, npos + len(new), device=DEV)
            out = fwd(input_ids=narr, past_key_values=past, position_ids=npo.unsqueeze(0),
                      cache_position=npo, use_cache=True)
            logits = out.logits[0, -1, :]
            npos += len(new)
            continue
        if "</think>" in txt and answer_complete(txt.split("</think>")[-1]):
            break
        narr = torch.tensor([[tok_id]], device=DEV)
        npo = torch.tensor([npos], device=DEV)
        out = fwd(input_ids=narr, past_key_values=past, position_ids=npo.unsqueeze(0),
                  cache_position=npo, use_cache=True)
        logits = out.logits[0, -1, :]
        npos += 1

    grounded = any(has(s + " " + h, gold) for s, h in zip(served, hints))
    landed = "</think>" in txt and bool(txt.split("</think>")[-1].strip())
    ans = head_sentence(txt.split("</think>")[-1].strip()) if landed else ""
    correct = landed and has(ans, gold)
    return {"ns": ns, "grounded": grounded, "landed": landed, "correct": correct,
            "bridge": any(page_gold[1:]), "queries": queries_out,
            "answer": ans, "text": txt}


# ---------------- question selection ------------------------------------------------
full = []
for line in open(CORPUS):
    try:
        r = json.loads(line)
    except Exception:
        continue
    q, gold = (r.get("q") or "").strip(), (r.get("gold") or "").strip()
    if q and gold and len(gold.split()) <= 6:
        full.append(r)
random.Random(0).shuffle(full)


def bucket(r):
    t = r.get("srch")
    return min(int(t), 5) if isinstance(t, (int, float)) else 0


excl = set()
if EXCLUDE_F:
    for line in open(EXCLUDE_F):
        line = line.strip()
        if not line:
            continue
        try:
            excl.add((json.loads(line).get("q") or "").strip())
        except Exception:
            excl.add(line)
    print(f"[data] excluding {len(excl)} questions from {EXCLUDE_F}", flush=True)
if ALL:
    sel = [r for r in full if r.get("q", "").strip() not in excl][:N_Q]
    MATCH = False
elif MATCH:
    train_slice = full[1:241]
    tgt = {}
    for r in train_slice:
        tgt[bucket(r)] = tgt.get(bucket(r), 0) + 1
    want = {k: int(round(N_Q * v / len(train_slice))) for k, v in tgt.items()}
    avail = {}
    for r in full[SKIP:]:
        avail.setdefault(bucket(r), []).append(r)
    sel = []
    print("  stratum : train%  ->  wanted / taken", flush=True)
    for k in sorted(want):
        take = avail.get(k, [])[:want[k]]
        sel += take
        print("   srch=%-3s %6.1f%%  ->  %4d / %4d" %
              (str(k) + ("+" if k == 5 else ""), 100.0 * tgt[k] / len(train_slice),
               want[k], len(take)), flush=True)
    tm = sum(min(int(r.get("srch") or 0), 9) for r in train_slice) / len(train_slice)
    em = sum(min(int(r.get("srch") or 0), 9) for r in sel) / max(len(sel), 1)
    print("  mean teacher srch: train=%.3f eval=%.3f" % (tm, em), flush=True)
elif not ALL:
    sel = full[SKIP:SKIP + N_Q]
if not ALL:
    random.Random(7).shuffle(sel)
sel = [r for r in sel if r.get("q", "").strip() not in excl]
shard = sel[si::sn]

done = set()
if os.path.exists(OUT_F):
    for line in open(OUT_F):
        try:
            done.add(json.loads(line)["q"])
        except Exception:
            pass
todo = [r for r in shard if r["q"] not in done]
print(f"[shard {si}/{sn}] {len(shard)} questions, {len(done)} already done, "
      f"{len(todo)} to run, G={G}, threads={THREADS}, rps={RPS}", flush=True)

# ---------------- run ---------------------------------------------------------------
out_lock = threading.Lock()
out_fh = open(OUT_F, "a")
stat = {"n": 0, "c": 0, "g": 0, "l": 0, "s": 0, "q": 0, "qc": 0}
t0 = time.time()
work = queue.Queue()
for r in todo:
    work.put(r)


def worker():
    while True:
        try:
            item = work.get_nowait()
        except queue.Empty:
            return
        head = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["q"]}],
            add_generation_prompt=True, tokenize=False)
        if not head.rstrip().endswith("<think>"):
            head += "<think>\n"
        prompt_ids = tokenizer.encode(head)
        anyc = False
        for _ in range(G):
            try:
                res = episode(prompt_ids, item["q"], item["gold"])
            except BaseException as e:
                print(f"[warn] rollout dropped: {type(e).__name__}: {e}", flush=True)
                continue
            anyc = anyc or res["correct"]
            rec = {"q": item["q"], "gold": item["gold"], "tsrch": item.get("srch"),
                   **{k: res[k] for k in
                      ("ns", "grounded", "landed", "correct", "answer", "queries", "text")}}
            with out_lock:
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_fh.flush()
                stat["n"] += 1
                stat["c"] += res["correct"]
                stat["g"] += res["grounded"]
                stat["l"] += res["landed"]
                stat["s"] += res["ns"]
        with out_lock:
            stat["q"] += 1
            stat["qc"] += anyc
            n = max(stat["n"], 1)
            el = time.time() - t0
            print(f"[{stat['q']}/{len(todo)}] rolls={stat['n']} "
                  f"correct={stat['c']/n:.1%} grounded={stat['g']/n:.1%} "
                  f"landed={stat['l']/n:.1%} srch={stat['s']/n:.1f} "
                  f"anyq={stat['qc']}/{stat['q']} "
                  f"{el/n:.1f}s/roll eta={(len(todo)-stat['q'])*G*el/n/60:.0f}m",
                  flush=True)
        work.task_done()


ths = [threading.Thread(target=worker, daemon=True) for _ in range(THREADS)]
for t in ths:
    t.start()
for t in ths:
    t.join()
n = max(stat["n"], 1)
print(f"EVAL_DONE shard={si}/{sn} rolls={stat['n']} correct={stat['c']/n:.1%} "
      f"grounded={stat['g']/n:.1%} landed={stat['l']/n:.1%} "
      f"anyq={stat['qc']}/{max(stat['q'],1)} wall={(time.time()-t0)/60:.0f}m", flush=True)
