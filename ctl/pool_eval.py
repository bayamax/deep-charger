#!/usr/bin/env python3
"""Held-out evaluation WITH compression for the pooler lineage.

The pooler is the point: tokens that scroll out of the raw window are pooled into 32 SP vectors
(mass-based eviction keeps at most MAXD of them), exactly as sp_rollout does in the training harness.
The search environment, however, is the one the self-distillation data came from (grpo_ep_more):
  <search>kw || ask</search> -> top-1 Wikipedia page, first 256 tokens, "[READER] (no extraction)" line
  <more/>                     -> next 256-token slice of the same page (up to MAXM)
  scoring                     -> first answer sentence must contain gold; grounded = gold in a served chunk

  SP_BASE=/root/fft_hf SP_RANK=128 SP_NOSYS=1 SP_EPISODIC=1 \
  python3 pool_eval.py <ckpt.safetensors> <questions.jsonl> <out.jsonl> [--rw 512] [--maxd 384] [--n 300] [--temp 0.9]
"""
import argparse, json, os, re, sys, time, urllib.parse, urllib.request, ssl

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("questions"); ap.add_argument("out")
ap.add_argument("--rw", type=int, default=512, help="raw window (tokens kept verbatim)")
ap.add_argument("--maxd", type=int, default=384, help="max pooled tokens kept for SP")
ap.add_argument("--n", type=int, default=300); ap.add_argument("--temp", type=float, default=0.9)
ap.add_argument("--gen", type=int, default=1500); ap.add_argument("--maxs", type=int, default=5)
ap.add_argument("--maxm", type=int, default=8); ap.add_argument("--chunk", type=int, default=128)
ap.add_argument("--tag", default="")
ap.add_argument("--samepage", type=int, default=0, help="1: a repeated page serves its next chunk (+ used-up notice), as in grpo_pool.py; 0: teacher environment")
ap.add_argument("--decode", default="plain", choices=["plain", "guard"], help="plain = teacher environment (temp sampling, only the <information ban); guard = harness pick() with rep-penalty/no-repeat")
ap.add_argument("--q4", type=int, default=0, help="1: round every weight the phone quantizes onto the 4-bit affine grid (group 64) before evaluating")
ap.add_argument("--q4group", type=int, default=64); ap.add_argument("--q4bits", type=int, default=4)
A = ap.parse_args()

os.environ.setdefault("SP_HOTPOT2", "0"); os.environ.setdefault("SP_BASE", "/root/fft_hf")
os.environ.setdefault("SP_RANK", "128"); os.environ.setdefault("SP_NOSYS", "1"); os.environ.setdefault("SP_EPISODIC", "1")
import torch  # noqa: E402
import numpy as np  # noqa: E402

# ---- model + pooler via the trainer's own prefix (same wrapping, same tokenizer) ----
F = "/root/work/grpo_e2e_torch.py"
src = open(F).read().split("\n")
cut = next(i for i, l in enumerate(src) if l.startswith("if MULTI > 1:"))
sys.argv = ["grpo_e2e_torch.py", "0", "6", "0", str(A.gen)]
sys.path.insert(0, "/root/work")
ns = {"__name__": "pool_eval", "__file__": F}
exec(compile("\n".join(src[:cut]), F, "exec"), ns)
model, tok, pooler = ns["model"], ns["tok"], ns["pooler"]
emb, sp, crop_cache, pick, _ngrams = ns["emb"], ns["sp"], ns["crop_cache"], ns["pick"], ns["_ngrams"]
DEV, eos = ns["DEV"], ns["eos"]
ns["TEMP"] = A.temp; ns["MAXD"] = A.maxd; ns["C"] = A.chunk; ns["GREEDY"] = False
from safetensors.torch import load_file  # noqa: E402
from transformers import DynamicCache  # noqa: E402
sd = load_file(A.ckpt)
pl = {k[len("pooler."):]: v for k, v in sd.items() if k.startswith("pooler.")}
md = {k: v for k, v in sd.items() if not k.startswith("pooler.")}
r = model.load_state_dict(md, strict=False)
print(f"[load] {A.ckpt}: {len(md)} tensors, {len(r.unexpected_keys)} unexpected", flush=True)
if pl:
    pooler.load_sd(pl); print(f"[load] pooler restored ({len(pl)} tensors)", flush=True)
model.eval()
if A.q4:
    # What ships is the 4-bit conversion of these weights, so measure that and not the bf16 parent.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import q4  # noqa: E402
    q4.quantize_model(model, group=A.q4group, bits=A.q4bits)
print(f"[cfg] rw={A.rw} maxd={A.maxd} chunk={A.chunk} temp={A.temp} gen={A.gen} maxs={A.maxs} maxm={A.maxm} decode={A.decode} samepage={A.samepage} q4={A.q4}", flush=True)

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
    """sp_rollout mechanics (SP + raw window, mass eviction) with the grpo_ep_more environment."""
    q_ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": question}],
                                               add_generation_prompt=True, tokenize=False) + "<think>\n")
    past = DynamicCache()
    model(input_ids=torch.tensor([q_ids], device=DEV), past_key_values=past, use_cache=True)
    MQ = past.get_seq_length()
    gen, kept, absorbed = [], [], 0
    exempt = set(q_ids); allowed_ng = _ngrams(q_ids)
    n_model, ns_, nm, nmt = 0, 0, 0, 0
    served, queries, page_ids, page_off = [], [], [], 0
    seen_pages, cur_key, nrep = {}, None, 0
    t0 = time.time(); dead = False

    def inject(text):
        ids = tok.encode(text, add_special_tokens=False)
        gen.extend(ids); exempt.update(ids); allowed_ng.update(_ngrams(ids))

    while n_model < A.gen and time.time() - t0 < 600:
        c0 = len(gen); R = min(c0, A.rw); nd = c0 - R
        if nd > absorbed:
            kept.extend(gen[absorbed:nd]); absorbed = nd
            if len(kept) > A.maxd:
                _, mass = pooler.forward_with_mass(emb(kept).to(torch.float32))
                mm = mass[0].float().cpu().numpy(); kept = [kept[i] for i in np.sort(np.argsort(mm)[-A.maxd:])]
        spv = sp(kept)
        parts = [spv] + ([emb(gen[c0 - R:c0])] if R > 0 else []); block = torch.cat(parts, dim=1)
        crop_cache(past, MQ)
        Lb = block.shape[1]; pos = torch.arange(MQ, MQ + Lb, device=DEV)
        out = model(inputs_embeds=block, past_key_values=past, attention_mask=torch.ones(1, MQ + Lb, device=DEV),
                    position_ids=pos.unsqueeze(0), cache_position=pos, use_cache=True)
        last = out.logits[:, -1, :]; npos = MQ + Lb
        brk = False
        for _ in range(A.chunk):
            txt_tail = tok.decode(gen[-40:])
            greedy = "</think>" in txt_tail
            nx = pick_plain(last, gen) if A.decode == "plain" else pick(last, gen, greedy, exempt, allowed_ng)
            if nx == eos:
                brk = True; break
            gen.append(nx); n_model += 1
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
        txt = tok.decode(gen)
        if dead or (brk and gen and gen[-1] == eos):
            break
        if "</think>" in txt and answer_complete(txt.split("</think>")[-1]):
            break
    txt = tok.decode(gen)
    landed = "</think>" in txt and bool(txt.split("</think>")[-1].strip())
    ans = head_sentence(txt.split("</think>")[-1].strip()) if landed else ""
    return txt, ans, ns_, nm, served, queries, landed, dead


qs = []
for line in open(A.questions):
    try:
        d = json.loads(line)
    except Exception:
        continue
    q, g = (d.get("q") or "").strip(), (d.get("gold") or "").strip()
    if q and g and q not in {x[0] for x in qs}:
        qs.append((q, g))
qs = qs[:A.n]
done = set()
if os.path.exists(A.out):
    for line in open(A.out):
        try: done.add(json.loads(line)["q"])
        except Exception: pass
todo = [x for x in qs if x[0] not in done]
print(f"[eval{A.tag}] {len(qs)} questions, {len(done)} done, {len(todo)} to run", flush=True)
stat = {"n": len(done), "c": 0, "g": 0, "l": 0, "s": 0, "m": 0}
for line in (open(A.out) if os.path.exists(A.out) else []):
    try:
        r0 = json.loads(line); stat["c"] += bool(r0.get("correct")); stat["g"] += bool(r0.get("grounded"))
        stat["l"] += bool(r0.get("landed")); stat["s"] += r0.get("ns", 0); stat["m"] += r0.get("more", 0)
    except Exception:
        pass
t0 = time.time(); k0 = stat["n"]
with open(A.out, "a") as fh:
    for q, g in todo:
        txt, ans, ns_, nm, served, queries, landed, dead = rollout(q)
        correct = landed and has(ans, g)
        grounded = any(has(s, g) for s in served)
        rec = {"q": q, "gold": g, "correct": correct, "grounded": grounded, "landed": landed, "dead": dead,
               "ns": ns_, "more": nm, "answer": ans, "queries": queries, "text": txt, "rw": A.rw, "maxd": A.maxd, "decode": A.decode, "samepage": A.samepage}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
        stat["n"] += 1; stat["c"] += correct; stat["g"] += grounded; stat["l"] += landed; stat["s"] += ns_; stat["m"] += nm
        n = stat["n"]; el = time.time() - t0; per = el / max(n - k0, 1)
        print(f"[{n}/{len(qs)}] correct={100*stat['c']/n:.1f}% grounded={100*stat['g']/n:.1f}% landed={100*stat['l']/n:.1f}% "
              f"srch={stat['s']/n:.1f} more={stat['m']/n:.2f} {per:.0f}s/roll eta={per*(len(qs)-n)/60:.0f}m", flush=True)
n = max(stat["n"], 1)
print(f"EVAL_DONE{A.tag} n={stat['n']} correct={100*stat['c']/n:.1f}% grounded={100*stat['g']/n:.1f}% landed={100*stat['l']/n:.1f}% "
      f"srch={stat['s']/n:.2f} more={stat['m']/n:.2f} rw={A.rw} maxd={A.maxd}", flush=True)
