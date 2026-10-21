#!/usr/bin/env python3
"""Parallel SP-evict: N bounded reasoning streams as one GPU batch, against one stream at a time and against full-KV
generation. The claim under test: with the KV footprint O(1) per stream (32 soft tokens + a raw window), many streams
run at once in a fixed memory, and the task can be split into sub-questions that run in parallel.

  modes   sp     the release loop (USAGE.md section 3): prefill the prompt, then every CHUNK tokens crop the KV to the
                 prompt and re-feed [SP(32) of the distant tokens + the last RW raw tokens]; MAXD distant tokens kept
                 by pooler mass
          full   plain generation, the KV grows with the length (the usual way)
  arms    decomp a task of K independent questions (par_bundles.py): --planner none, one stream solves the whole task;
                 g14, the model itself splits the task into sub-questions (a planning stream), the parts run as parallel
                 streams and the answers are merged in order; api, the split made by a large model (in the bundles file)
          gsm    GSM8K questions, N streams at once: tokens/s, peak memory, accuracy, by N
          split  three-part problems ((a)(b)(c) made of three GSM8K questions): one stream for all three, against the
                 three parts as three parallel streams; accuracy per part and wall time

  python3 par_sp.py --model g14 --arm gsm --mode sp --n 32 --questions 64 --gen 1024
  python3 par_sp.py --model g14 --arm split --mode sp --problems 24 --gen 2048
"""
import argparse, json, math, os, re, sys, time

import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="the 16-bit model directory (with pooler.safetensors)")
ap.add_argument("--arm", default="gsm", choices=["gsm", "split", "decomp"]); ap.add_argument("--mode", default="sp", choices=["sp", "full"])
ap.add_argument("--n", type=int, default=8, help="streams per batch"); ap.add_argument("--questions", type=int, default=32); ap.add_argument("--problems", type=int, default=16)
ap.add_argument("--gen", type=int, default=1024, help="model tokens per stream"); ap.add_argument("--gen-split", type=int, default=0, help="per sub-stream (default gen/2)")
ap.add_argument("--rw", type=int, default=768); ap.add_argument("--maxd", type=int, default=384); ap.add_argument("--chunk", type=int, default=128)
ap.add_argument("--temp", type=float, default=0.6); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--gsm", default="gsm8k_test.parquet")
ap.add_argument("--out", default=""); ap.add_argument("--tag", default="")
ap.add_argument("--bundles", default="bundles.jsonl", help="decomp: par_bundles.py's bundles"); ap.add_argument("--planner", default="none", choices=["none", "g14", "api"])
ap.add_argument("--gen-plan", type=int, default=600, help="decomp: model tokens for the planning stream (planner g14)")
A = ap.parse_args()
torch.manual_seed(A.seed)
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.bfloat16 if DEV == "cuda" else torch.float32
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

tok = AutoTokenizer.from_pretrained(A.model)
try: model = AutoModelForCausalLM.from_pretrained(A.model, dtype=DT)
except TypeError: model = AutoModelForCausalLM.from_pretrained(A.model, torch_dtype=DT)
model = model.to(DEV).eval()
EMB = model.get_input_embeddings(); H = EMB.weight.shape[1]; EOS = tok.eos_token_id; PAD = tok.pad_token_id if tok.pad_token_id is not None else EOS
import inspect  # noqa: E402
LTK = "logits_to_keep" if "logits_to_keep" in inspect.signature(model.forward).parameters else ("num_logits_to_keep" if "num_logits_to_keep" in inspect.signature(model.forward).parameters else None)
WEIGHT_BYTES = sum(p.numel() * p.element_size() for p in model.parameters())


# ---- the pooler (TrainablePoolerTorch of the training code, the weights from the release's safetensors) --------------
def _ln(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdim=True); var = ((x - mu) ** 2).mean(-1, keepdim=True); return (x - mu) / torch.sqrt(var + eps) * w + b


def _mha(q, k, v, P, heads, want_w):
    B, Lq, Hd = q.shape; Lk = k.shape[1]; hd = Hd // heads; Wi, bi, Wo, bo = P
    qp = q @ Wi[:Hd].T + bi[:Hd]; kp = k @ Wi[Hd:2 * Hd].T + bi[Hd:2 * Hd]; vp = v @ Wi[2 * Hd:].T + bi[2 * Hd:]
    sp_ = lambda t, L: t.reshape(B, L, heads, hd).permute(0, 2, 1, 3)
    attn = torch.softmax((sp_(qp, Lq) @ sp_(kp, Lk).transpose(-1, -2)) / math.sqrt(hd), -1)
    out = (attn @ sp_(vp, Lk)).permute(0, 2, 1, 3).reshape(B, Lq, Hd) @ Wo.T + bo
    return out, (attn.mean(1) if want_w else None)


class Pooler:
    def __init__(s, path, device):
        s.A = {k: v.float().to(device) for k, v in load_file(path).items()}; s.dev = device
        s.n_sp, s.H = s.A["query"].shape; s.heads = 8; s.layers = sum(1 for k in s.A if k.endswith(".lnq1.weight"))
        pos = np.arange(8192)[:, None].astype(np.float32); i = np.arange(0, s.H, 2).astype(np.float32); div = np.exp(-math.log(10000.0) * i / s.H)
        pe = np.zeros((8192, s.H), np.float32); pe[:, 0::2] = np.sin(pos * div); pe[:, 1::2] = np.cos(pos * div); s.pe = torch.from_numpy(pe).to(device)

    def P(s, i, w): p = f"blocks.{i}.{w}."; return (s.A[p + "in_proj_weight"], s.A[p + "in_proj_bias"], s.A[p + "out_proj.weight"], s.A[p + "out_proj.bias"])
    def wb(s, i, n): return s.A[f"blocks.{i}.{n}.weight"], s.A[f"blocks.{i}.{n}.bias"]

    @torch.no_grad()
    def run(s, past, want_mass=False):
        """past (B, L, H) float32 -> (B, 32, H) and, when asked, the per-token mass (B, L)"""
        B, L = past.shape[0], past.shape[1]; past = past + s.pe[:L][None] if L > 0 else past
        q = s.A["query"][None].expand(B, s.n_sp, s.H).contiguous(); mass = torch.zeros((B, L), device=s.dev) if want_mass else None
        for i in range(s.layers):
            if L > 0:
                kn = _ln(past, *s.wb(i, "lnk")); a, w = _mha(_ln(q, *s.wb(i, "lnq1")), kn, kn, s.P(i, "cross"), s.heads, want_mass); q = q + a
                if want_mass: mass = mass + w.sum(1)
            qn = _ln(q, *s.wb(i, "lnq2")); q = q + _mha(qn, qn, qn, s.P(i, "selfa"), s.heads, False)[0]
            h = _ln(q, *s.wb(i, "lnq3")); ff = F.gelu(h @ s.A[f"blocks.{i}.ffn.0.weight"].T + s.A[f"blocks.{i}.ffn.0.bias"])
            q = q + ff @ s.A[f"blocks.{i}.ffn.2.weight"].T + s.A[f"blocks.{i}.ffn.2.bias"]
        spv = _ln(q, s.A["ln_out.weight"], s.A["ln_out.bias"]); spv = spv / torch.clamp(spv.norm(dim=-1, keepdim=True), min=1e-6) * torch.abs(s.A["out_scale"])
        return (spv, mass) if want_mass else spv


pooler = Pooler(os.path.join(A.model, "pooler.safetensors"), DEV)
SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."
prompt_ids = lambda q: tok.encode(tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False) + "<think>\n")


def crop(cache, n):
    """cut the KV back to its first n positions (a negative count for transformers' crop: it is the same in 4.x and 5.x)"""
    if hasattr(cache, "crop"):
        L = cache.get_seq_length()
        if L > n: cache.crop(n - L)
        return
    for i in range(len(cache.key_cache)): cache.key_cache[i] = cache.key_cache[i][:, :, :n]; cache.value_cache[i] = cache.value_cache[i][:, :, :n]


def cache_bytes(cache):
    try: return sum(l.keys.numel() * l.keys.element_size() + l.values.numel() * l.values.element_size() for l in cache.layers)
    except Exception:
        try: return sum(k.numel() * k.element_size() * 2 for k in cache.key_cache)
        except Exception: return -1


def fwd(**kw):
    if LTK: kw[LTK] = 1
    return model(**kw)


@torch.no_grad()
def generate(prompts, gen_cap):
    """prompts: token id lists; N streams as one batch. Returns (texts, model tokens per stream, peak KV bytes, steps)."""
    N = len(prompts); mqs = [len(p) for p in prompts]; MQ = max(mqs)
    inp = torch.full((N, MQ), PAD, device=DEV, dtype=torch.long); pmask = torch.zeros((N, MQ), device=DEV, dtype=torch.long); ppos = torch.zeros((N, MQ), device=DEV, dtype=torch.long)
    for b, ids in enumerate(prompts): L = len(ids); inp[b, MQ - L:] = torch.tensor(ids, device=DEV); pmask[b, MQ - L:] = 1; ppos[b, MQ - L:] = torch.arange(L, device=DEV)
    cache = DynamicCache()
    out = fwd(input_ids=inp, attention_mask=pmask, position_ids=ppos, past_key_values=cache, use_cache=True)
    last = out.logits[:, -1, :]
    gen = [[] for _ in range(N)]; kept = [[] for _ in range(N)]; absorbed = [0] * N; fin = [False] * N; landed = [False] * N
    kv_peak = cache_bytes(cache); steps = 0; amask = pmask

    def pick(b, lg):
        if landed[b]: return int(lg.argmax())
        return int(torch.multinomial(torch.softmax(lg.float() / A.temp, -1), 1))

    if A.mode == "full":
        cur = MQ
        while not all(fin) and steps < gen_cap:
            nx = [pick(b, last[b]) if not fin[b] else EOS for b in range(N)]
            for b in range(N):
                if fin[b]: continue
                if nx[b] == EOS: fin[b] = True; continue
                gen[b].append(nx[b])
                if len(gen[b]) >= 8 and len(set(gen[b][-8:])) == 1: fin[b] = True
                if not landed[b] and "</think>" in tok.decode(gen[b][-6:]): landed[b] = True
            if all(fin): break
            amask = torch.cat([amask, torch.ones((N, 1), device=DEV, dtype=torch.long)], 1)
            pos = torch.tensor([[mqs[b] + len(gen[b]) - 1 + (1 if fin[b] else 0)] for b in range(N)], device=DEV)
            out = fwd(input_ids=torch.tensor([[gen[b][-1] if gen[b] else EOS] for b in range(N)], device=DEV), attention_mask=amask, position_ids=pos,
                      past_key_values=cache, cache_position=torch.tensor([cur], device=DEV), use_cache=True)
            last = out.logits[:, -1, :]; cur += 1; steps += 1; kv_peak = max(kv_peak, cache_bytes(cache))
    else:
        while not all(fin) and steps < gen_cap:
            blocks, Rs = [], []
            for b in range(N):   # the bounded block of each stream: SP of the distant tokens + the raw window
                c0 = len(gen[b]); R = min(c0, A.rw); nd = c0 - R
                if nd > absorbed[b]:
                    kept[b].extend(gen[b][absorbed[b]:nd]); absorbed[b] = nd
                    if len(kept[b]) > A.maxd:
                        _, mass = pooler.run(EMB(torch.tensor([kept[b]], device=DEV)).float(), True)
                        mm = mass[0].cpu().numpy(); kept[b] = [kept[b][i] for i in np.sort(np.argsort(mm)[-A.maxd:])]
                spv = pooler.run(EMB(torch.tensor([kept[b]], device=DEV)).float() if kept[b] else torch.zeros((1, 0, H), device=DEV)).to(DT)
                parts = [spv] + ([EMB(torch.tensor([gen[b][c0 - R:c0]], device=DEV))] if R > 0 else [])
                blocks.append(torch.cat(parts, 1)); Rs.append(R)
            lens = [x.shape[1] for x in blocks]; Lmax = max(lens)
            block = torch.zeros((N, Lmax, H), device=DEV, dtype=DT); amask = torch.zeros((N, MQ + Lmax), device=DEV, dtype=torch.long); amask[:, :MQ] = pmask
            pos = torch.zeros((N, Lmax), device=DEV, dtype=torch.long)
            for b in range(N): block[b, :lens[b]] = blocks[b][0]; amask[b, MQ:MQ + lens[b]] = 1; pos[b] = mqs[b] + torch.arange(Lmax, device=DEV)
            crop(cache, MQ)
            out = model(inputs_embeds=block, past_key_values=cache, attention_mask=amask, position_ids=pos, cache_position=torch.arange(MQ, MQ + Lmax, device=DEV), use_cache=True)
            last = torch.stack([out.logits[b, lens[b] - 1] for b in range(N)]); cur = MQ + Lmax
            for _ in range(A.chunk):
                nx = [pick(b, last[b]) if not fin[b] else EOS for b in range(N)]
                for b in range(N):
                    if fin[b]: continue
                    if nx[b] == EOS: fin[b] = True; continue
                    gen[b].append(nx[b])
                    if len(gen[b]) >= 8 and len(set(gen[b][-8:])) == 1: fin[b] = True
                    if not landed[b] and "</think>" in tok.decode(gen[b][-6:]): landed[b] = True
                steps += 1
                if all(fin) or steps >= gen_cap: break
                amask = torch.cat([amask, torch.ones((N, 1), device=DEV, dtype=torch.long)], 1)
                pos1 = torch.tensor([[mqs[b] + 32 + Rs[b] + (cur - (MQ + Lmax))] for b in range(N)], device=DEV)
                out = fwd(input_ids=torch.tensor([[gen[b][-1] if gen[b] else EOS] for b in range(N)], device=DEV), attention_mask=amask, position_ids=pos1,
                          past_key_values=cache, cache_position=torch.tensor([cur], device=DEV), use_cache=True)
                last = out.logits[:, -1, :]; cur += 1
            kv_peak = max(kv_peak, cache_bytes(cache))
    return [tok.decode(g) for g in gen], [len(g) for g in gen], kv_peak, steps


def final_number(txt):
    """the model's answer: the last \\boxed{} after </think>, else the last number there"""
    a = txt.split("</think>")[-1] if "</think>" in txt else txt[-400:]
    m = re.findall(r"\\boxed\{([^{}]*)\}", a); s = m[-1] if m else a
    n = re.findall(r"-?\d[\d,]*\.?\d*", s); return n[-1].replace(",", "").rstrip(".") if n else ""


def same(a, b):
    try: return abs(float(a) - float(b)) < 1e-6
    except Exception: return False


import pyarrow.parquet as pq  # noqa: E402
rows = pq.read_table(A.gsm).to_pylist(); rng = np.random.RandomState(1); order = rng.permutation(len(rows))
gold = lambda r: r["answer"].split("####")[-1].strip().replace(",", "")
res = {"arm": A.arm, "mode": A.mode, "n": A.n, "gen": A.gen, "rw": A.rw, "maxd": A.maxd, "chunk": A.chunk, "tag": A.tag, "weights_mb": WEIGHT_BYTES / 1e6}
if DEV == "cuda": torch.cuda.reset_peak_memory_stats()
T0 = time.time(); toks = 0; kvp = 0; correct = 0; n_done = 0; texts_out = []
if A.arm == "gsm":
    qs = [rows[i] for i in order[:A.questions]]
    for a in range(0, len(qs), A.n):
        batch = qs[a:a + A.n]; t1 = time.time()
        texts, lens, kv, steps = generate([prompt_ids(r["question"] + SUFFIX) for r in batch], A.gen)
        toks += sum(lens); kvp = max(kvp, kv)
        for r, t in zip(batch, texts):
            ok = same(final_number(t), gold(r)); correct += ok; n_done += 1; texts_out.append({"q": r["question"][:80], "gold": gold(r), "pred": final_number(t), "ok": bool(ok), "len": len(t), "text": t[:400]})
        print(f"[par] batch {a // A.n}: {len(batch)} streams, {sum(lens)} tokens in {time.time() - t1:.1f}s ({sum(lens) / (time.time() - t1):.1f} tok/s), kv {kv / 1e6:.0f} MB, correct so far {correct}/{n_done}", flush=True)
elif A.arm == "split":
    gs = A.gen_split or A.gen // 2; probs = []
    for i in range(A.problems): probs.append([rows[j] for j in order[3 * i:3 * i + 3]])
    # one stream: the three parts in one prompt
    t1 = time.time(); one_ok = [0, 0, 0]; one_toks = 0; one_kv = 0
    for a in range(0, len(probs), A.n):
        pb = probs[a:a + A.n]
        ps = [prompt_ids("Answer these three questions.\n(a) " + p[0]["question"] + "\n(b) " + p[1]["question"] + "\n(c) " + p[2]["question"] +
                         "\nPlease reason step by step, then give the three final answers as (a) \\boxed{}, (b) \\boxed{}, (c) \\boxed{}.") for p in pb]
        texts, lens, kv, _ = generate(ps, A.gen); one_toks += sum(lens); one_kv = max(one_kv, kv)
        for p, t in zip(pb, texts):
            ans = t.split("</think>")[-1] if "</think>" in t else t[-600:]; boxed = re.findall(r"\\boxed\{([^{}]*)\}", ans)
            for k in range(3):
                pred = re.findall(r"-?\d[\d,]*\.?\d*", boxed[k])[-1].replace(",", "") if k < len(boxed) and re.findall(r"-?\d[\d,]*\.?\d*", boxed[k]) else ""
                one_ok[k] += same(pred, gold(p[k]))
    one_time = time.time() - t1
    print(f"[par] one stream per problem: parts correct {one_ok} of {len(probs)} each = {sum(one_ok)}/{3 * len(probs)}, {one_toks} tokens, {one_time:.0f}s, kv {one_kv / 1e6:.0f} MB", flush=True)
    # split: the three parts as three streams, all problems' parts batched N at a time
    t1 = time.time(); parts = [(i, k, probs[i][k]) for i in range(len(probs)) for k in range(3)]; sp_ok = [0, 0, 0]; sp_toks = 0; sp_kv = 0
    for a in range(0, len(parts), A.n):
        pb = parts[a:a + A.n]
        texts, lens, kv, _ = generate([prompt_ids(r["question"] + SUFFIX) for _, _, r in pb], gs); sp_toks += sum(lens); sp_kv = max(sp_kv, kv)
        for (_, k, r), t in zip(pb, texts): sp_ok[k] += same(final_number(t), gold(r))
    sp_time = time.time() - t1
    print(f"[par] split into parallel sub-streams ({A.n} at a time): parts correct {sp_ok} = {sum(sp_ok)}/{3 * len(probs)}, {sp_toks} tokens, {sp_time:.0f}s, kv {sp_kv / 1e6:.0f} MB", flush=True)
    res.update(one={"ok": one_ok, "tokens": one_toks, "sec": one_time, "kv_mb": one_kv / 1e6}, split={"ok": sp_ok, "tokens": sp_toks, "sec": sp_time, "kv_mb": sp_kv / 1e6}, problems=len(probs))
if A.arm == "decomp":
    bundles = [json.loads(l) for l in open(A.bundles)]; gs = A.gen_split or A.gen // 2
    task_text = lambda b: "Solve this task:\n" + "\n".join(f"({j + 1}) {q}" for j, q in enumerate(b["qs"]))
    PLAN = ("Split the task below into independent sub-questions that can each be solved on their own, in the task's order. Each sub-question must be "
            "self-contained: copy every number and name it needs. Do not solve anything. After thinking, write only a numbered list, one sub-question per line.\n\n")
    nums = lambda t: re.findall(r"-?\d[\d,]*\.?\d*", t)
    per_part = [0] * 5; whole = 0; whole_set = 0; plan_ok = 0; plan_toks = 0; solve_toks = 0; t_plan = 0.0; t_solve = 0.0; kvp = 0; details = []
    if A.planner == "none":
        t1 = time.time()
        for a in range(0, len(bundles), A.n):
            bb = bundles[a:a + A.n]
            ps = [prompt_ids(task_text(b) + "\nPlease reason step by step, then give the final answers in order, each within its own \\boxed{}.") for b in bb]
            texts, lens, kv, _ = generate(ps, A.gen); solve_toks += sum(lens); kvp = max(kvp, kv)
            for b, t in zip(bb, texts):
                ans = t.split("</think>")[-1] if "</think>" in t else t[-800:]; boxed = [x for x in re.findall(r"\\boxed\{([^{}]*)\}", ans)]
                preds = [(nums(x)[-1].replace(",", "") if nums(x) else "") for x in boxed]
                oks = [k < len(preds) and same(preds[k], b["golds"][k]) for k in range(b["k"])]
                for k, o in enumerate(oks): per_part[k] += o
                whole += all(oks); whole_set += sorted(preds[:b["k"]]) == sorted(b["golds"]) if len(preds) >= b["k"] else False
                details.append({"id": b["id"], "preds": preds[:b["k"]], "golds": b["golds"], "ok": oks})
        t_solve = time.time() - t1
    else:
        plans = {}
        if A.planner == "api":
            for b in bundles: plans[b["id"]] = b.get("api_sub") or []
        else:
            t1 = time.time()
            for a in range(0, len(bundles), A.n):
                bb = bundles[a:a + A.n]
                texts, lens, kv, _ = generate([prompt_ids(PLAN + task_text(b)) for b in bb], A.gen_plan); plan_toks += sum(lens); kvp = max(kvp, kv)
                for b, t in zip(bb, texts):
                    ans = t.split("</think>")[-1] if "</think>" in t else ""
                    subs = [m.group(2).strip() for m in re.finditer(r"^\s*(?:\(?\d+[.)]|[-*])\s*(.+)$", ans, re.M) for m in [m] if len(m.group(1)) > 15] if False else \
                           [m.group(1).strip() for m in re.finditer(r"^\s*(?:\(?\d+[.)]|[-*•])\s*(.{15,})$", ans, re.M)]
                    plans[b["id"]] = subs
            t_plan = time.time() - t1
        parts = []
        for b in bundles:
            subs = plans.get(b["id"]) or []; ok_ = len(subs) == b["k"]; plan_ok += ok_
            if ok_:
                for k, sq in enumerate(subs): parts.append((b["id"], k, sq))
        t1 = time.time(); preds = {}
        for a in range(0, len(parts), A.n):
            pb = parts[a:a + A.n]
            texts, lens, kv, _ = generate([prompt_ids(sq + SUFFIX) for _, _, sq in pb], gs); solve_toks += sum(lens); kvp = max(kvp, kv)
            for (bid, k, _), t in zip(pb, texts): preds[(bid, k)] = final_number(t)
        t_solve = time.time() - t1
        for b in bundles:
            pr = [preds.get((b["id"], k), "") for k in range(b["k"])]; oks = [same(pr[k], b["golds"][k]) for k in range(b["k"])]
            for k, o in enumerate(oks): per_part[k] += o
            whole += all(oks); whole_set += sorted(pr) == sorted(b["golds"])
            details.append({"id": b["id"], "plan": (plans.get(b["id"]) or [])[:5], "preds": pr, "golds": b["golds"], "ok": oks})
    nb = len(bundles); K = max(b["k"] for b in bundles); npart = sum(b["k"] for b in bundles)
    print(f"[par] decomp planner={A.planner} K={sorted(set(b['k'] for b in bundles))}: {nb} tasks; plan with the right count {plan_ok if A.planner != 'none' else '-'}/{nb}; "
          f"parts correct {sum(per_part)}/{npart} ({100 * sum(per_part) / npart:.0f}%) by position {per_part[:K]}; whole task correct {whole}/{nb} (as a set {whole_set}); "
          f"plan {t_plan:.0f}s + solve {t_solve:.0f}s; tokens plan {plan_toks}, solve {solve_toks}; KV peak {kvp / 1e6:.0f} MB", flush=True)
    res.update(planner=A.planner, tasks=nb, plan_ok=plan_ok, parts_ok=sum(per_part), parts=npart, by_pos=per_part[:K], whole=whole, whole_set=whole_set,
               t_plan=t_plan, t_solve=t_solve, plan_tokens=plan_toks, solve_tokens=solve_toks, kv_peak_mb=kvp / 1e6, details=details[:12])
sec = time.time() - T0
res.update(tokens=toks, sec=sec, tok_per_s=toks / max(sec, 1e-9), kv_peak_mb=kvp / 1e6, correct=correct, done=n_done,
           peak_alloc_mb=(torch.cuda.max_memory_allocated() / 1e6 if DEV == "cuda" else None), samples=texts_out[:8])
if A.arm == "gsm": print(f"[par] {A.mode} n={A.n}: {n_done} questions, {correct} correct ({100 * correct / max(1, n_done):.0f}%), {toks} tokens in {sec:.0f}s = {toks / sec:.1f} tok/s; "
                         f"KV peak {kvp / 1e6:.0f} MB; GPU peak {res['peak_alloc_mb'] or 0:.0f} MB (weights {WEIGHT_BYTES / 1e6:.0f} MB)", flush=True)
if A.out:
    with open(A.out, "a") as f: f.write(json.dumps(res) + "\n")
