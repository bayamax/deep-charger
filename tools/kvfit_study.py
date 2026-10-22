"""How well can a bounded KV cache reproduce the full model's generation? A GPTQ-style screen for the new pooler.

GPTQ fits each layer so that its OUTPUT on calibration inputs matches the full-precision layer. Here the "layer" is one
attention head's read of the evicted span E: when E leaves the exposure window, m slots per (layer, kv head) replace it,
fitted so that the attention output of calibration queries is unchanged. The calibration queries are those of the
recent window tokens - they exist at eviction time on the phone too (no peeking at the future).

  slot keys   : the m keys of E that the window queries attend to most (top attention mass)
  slot biases : beta_c = log w_c, w >= 0 by NNLS, so the slots carry E's total attention mass (an additive mask column)
  slot values : ridge least squares, so the slots carry E's attention-weighted values
Memory of a fitted cache = prompt + m + window tokens of KV; the fit itself has no weights (pooler size 0 <= KV).

Methods per (exposure window W, slots m):
  drop       : prompt + last W tokens only (m = 0; the floor)
  select     : + the m top-mass keys of E with their own values (H2O-like selection, no fitting)
  fit        : + the m slots fitted as above on the window's queries
  fit_past   : the same fit on every past query (headroom if the phone kept a query summary)
  fit_oracle : the same fit on the next N tokens' queries (capacity of m slots; not a method)
Score: teacher-forced on the model's own full-KV reasoning traces. At a cut t the full model's next-token distributions
on x[t:t+N] are the reference; KL(full || compressed) and top-1 agreement are reported (single eviction event).

  python kvfit_study.py --traces kv_full.jsonl --model Qwen/Qwen3-1.7B --windows 128,256,512,1024 --slots 32,128
"""
import argparse, json, math, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import nnls

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
ap.add_argument("--traces", required=True, help="jsonl from kvwin_eval.py --engine vllm (text = the model's own trace)")
ap.add_argument("--n-traces", type=int, default=16)
ap.add_argument("--cuts", default="1024,2048", help="cut positions counted from the end of the prompt")
ap.add_argument("--N", type=int, default=256, help="tokens scored after each cut")
ap.add_argument("--windows", default="128,256,512,1024")
ap.add_argument("--slots", default="32,128")
ap.add_argument("--ridge", type=float, default=1e-3)
ap.add_argument("--methods", default="select,fit,fit_bank,fit_past,fit_oracle")
ap.add_argument("--bank", type=int, default=512, help="fit_bank: calibration queries per layer kept from EARLIER traces")
ap.add_argument("--max-cal", type=int, default=2048, help="calibration queries per kv head (subsampled)")
ap.add_argument("--device", default="cuda")
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--out", default="kvfit.jsonl")
a = ap.parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import rotate_half

dev = a.device
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), attn_implementation="eager").to(dev).eval()
cfg = model.config
NL, H, G = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
D = getattr(cfg, "head_dim", cfg.hidden_size // H); REP = H // G; SC = D ** -0.5
KV_BYTES = NL * 2 * G * D * 2   # bytes of fp16 KV per token
WINS = [int(x) for x in a.windows.split(",")]; SLOTS = [int(x) for x in a.slots.split(",")]; WMAX = max(WINS)

# capture post-RoPE queries of the last WMAX positions before the cut
QCAP = {}
def make_hook(li):
    def hook(mod, args, kwargs):
        if "q_range" not in QCAP: return
        hs = kwargs["hidden_states"]; cos, sin = kwargs["position_embeddings"]
        lo, hi = QCAP["q_range"]
        h = hs[:, lo:hi]
        q = mod.q_norm(mod.q_proj(h).view(1, hi - lo, H, D)).transpose(1, 2)
        c, s = cos[:, lo:hi].unsqueeze(1), sin[:, lo:hi].unsqueeze(1)
        QCAP[li] = (q * c + rotate_half(q) * s).half()    # (1,H,n,D), positions lo..hi-1
    return hook
for li, layer in enumerate(model.model.layers):
    layer.self_attn.register_forward_pre_hook(make_hook(li), with_kwargs=True)


def kv_of(cache, li):
    if hasattr(cache, "layers"): return cache.layers[li].keys, cache.layers[li].values
    return cache.key_cache[li], cache.value_cache[li]


@torch.no_grad()
def build(cache, P, t, W, m, method, qlo):
    """-> list of (K,V) per layer (1,G,P+m+W,D) and beta (NL,G,m) or None.
    Calibration queries (captured from position qlo on), each attending to the whole cache [0,t) "as if asked now":
      fit        : the exposure window's queries [t-W, t)       (available on the phone at eviction time)
      fit_past   : every past query [P, t)                      (would have to be kept / summarised on the phone)
      fit_oracle : the next N tokens' queries [t, t+N)           (the future: a capacity bound, not a method)
      fit_bank   : the window's queries + BANK[layer] (queries of earlier traces, no leakage)"""
    out, betas = [], torch.zeros(NL, G, m, device=dev)
    lo, hi = {"select": (t - W, t), "fit": (t - W, t), "fit_bank": (t - W, t), "fit_past": (P, t), "fit_oracle": (t, t + a.N)}.get(method, (t - W, t))
    gen = torch.Generator(device="cpu").manual_seed(0)
    for li in range(NL):
        K, V = kv_of(cache, li); K, V = K[:, :, :t], V[:, :, :t]
        if m == 0 or method == "drop":
            out.append((torch.cat([K[:, :, :P], K[:, :, t - W:t]], 2), torch.cat([V[:, :, :P], V[:, :, t - W:t]], 2))); continue
        Qa = QCAP[li][0, :, lo - qlo:hi - qlo].float()                # (H, n, D)
        Ks, Vs = [], []
        for g in range(G):
            q = Qa[g * REP:(g + 1) * REP].reshape(-1, D)
            if method == "fit_bank" and li in BANK: q = torch.cat([q, BANK[li][g * REP:(g + 1) * REP].float().reshape(-1, D)])
            if q.shape[0] > a.max_cal: q = q[torch.randperm(q.shape[0], generator=gen)[:a.max_cal].to(dev)]
            k = K[0, g].float(); v = V[0, g].float()
            s = (q @ k.T) * SC                                         # (Q, t)
            lZ = torch.logsumexp(s, -1, keepdim=True)
            p = torch.exp(s - lZ)
            e0, e1 = P, t - W
            pE = p[:, e0:e1]
            idx = pE.sum(0).topk(m).indices + e0
            kc, vc = k[idx], v[idx]
            if method == "select":
                Ks.append(kc); Vs.append(vc); continue
            A = torch.exp(s[:, idx] - lZ)
            f = pE.sum(-1)
            w, _ = nnls(A.double().cpu().numpy(), f.double().cpu().numpy(), maxiter=50 * m)
            w = torch.tensor(w, device=dev, dtype=torch.float32).clamp(min=1e-8)
            B = A * w[None]
            n = pE @ v[e0:e1]
            BtB = B.T @ B; lam = a.ridge * BtB.diagonal().mean().clamp(min=1e-12)
            vfit = torch.linalg.solve(BtB + lam * torch.eye(m, device=dev), B.T @ n + lam * vc)
            Ks.append(kc); Vs.append(vfit); betas[li, g] = torch.log(w)
        Kc = torch.stack(Ks)[None].to(K.dtype); Vc = torch.stack(Vs)[None].to(V.dtype)
        out.append((torch.cat([K[:, :, :P], Kc, K[:, :, t - W:t]], 2), torch.cat([V[:, :, :P], Vc, V[:, :, t - W:t]], 2)))
    return out, (betas if method.startswith("fit") and m else None)


@torch.no_grad()
def score(kvs, beta, P, m, ids, t, ref_logp):
    c = DynamicCache()
    for li, (k, v) in enumerate(kvs): c.update(k, v, li)
    Lc = kvs[0][0].shape[2]; N = ref_logp.shape[0]
    mask = torch.zeros(1, H, N, Lc + N, device=dev, dtype=model.dtype)
    mask[:, :, :, Lc:] = torch.triu(torch.full((N, N), float("-inf"), device=dev), 1).to(model.dtype)
    BETA["v"] = beta; BETA["lo"] = P; BETA["m"] = m
    lg = model(input_ids=ids[None, t:t + N], position_ids=torch.arange(t, t + N, device=dev)[None],
               past_key_values=c, attention_mask=mask, use_cache=True).logits[0].float()
    BETA["v"] = None
    lp = F.log_softmax(lg, -1)
    kl = (ref_logp.exp() * (ref_logp - lp)).sum(-1)
    agree = (lp.argmax(-1) == ref_logp.argmax(-1)).float()
    return kl, agree


# per-layer slot biases: the attention mask is shared by all layers, so add beta inside each layer's attention
BETA = {"v": None}
BANK = {}   # layer -> (H, bank, D) fp16 queries from earlier traces
import transformers.models.qwen3.modeling_qwen3 as qm
_eager = qm.eager_attention_forward
def eager_with_beta(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    if BETA["v"] is not None and attention_mask is not None:
        b = BETA["v"][module.layer_idx]                                        # (G, m)
        bh = b.repeat_interleave(REP, 0)                                       # (H, m)
        attention_mask = attention_mask.clone()
        attention_mask[0, :, :, BETA["lo"]:BETA["lo"] + BETA["m"]] += bh[:, None, :].to(attention_mask.dtype)
    return _eager(module, query, key, value, attention_mask, scaling, dropout=dropout, **kw)
qm.eager_attention_forward = eager_with_beta
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
if "eager" in ALL_ATTENTION_FUNCTIONS: ALL_ATTENTION_FUNCTIONS["eager"] = eager_with_beta


rows = [json.loads(l) for l in open(a.traces)][:a.n_traces]
if not all("prompt" in r for r in rows):   # kvwin_eval rows: rebuild the prompt from MATH-500 by id
    from datasets import load_dataset
    PROB = {x["unique_id"]: x["problem"] for x in load_dataset("HuggingFaceH4/MATH-500", split="test")}
    SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    for r in rows:
        if "prompt" not in r:
            r["prompt"] = tok.apply_chat_template([{"role": "user", "content": PROB[r["id"]] + SUFFIX}], add_generation_prompt=True,
                                                  tokenize=False, enable_thinking=True)
CUTS = [int(x) for x in a.cuts.split(",")]
res = {}
fo = open(a.out, "w")
t0 = time.time()
for ri, r in enumerate(rows):
    prompt = r["prompt"]
    pid = tok.encode(prompt, add_special_tokens=False); P = len(pid)
    ids = torch.tensor(pid + tok.encode(r["text"], add_special_tokens=False), device=dev)
    for cut in CUTS:
        t = P + cut
        if t + a.N > len(ids) or cut - WMAX < 64: continue
        QCAP.clear(); QCAP["q_range"] = (P, t + a.N)
        cache = DynamicCache()
        with torch.no_grad():
            lg = model(input_ids=ids[None, :t + a.N], past_key_values=cache, use_cache=True).logits[0, t:t + a.N].float()
        del QCAP["q_range"]
        ref = F.log_softmax(lg, -1)
        # sanity: the full cache through the same path must reproduce the reference
        full = [(kv_of(cache, li)[0][:, :, :t], kv_of(cache, li)[1][:, :, :t]) for li in range(NL)]
        kl, ag = score(full, None, P, 0, ids, t, ref)
        res.setdefault("full", []).append((kl.mean().item(), ag.mean().item()))
        for W in WINS:
            for m, meth in [(0, "drop")] + [(m, x) for m in SLOTS for x in a.methods.split(",")]:
                kvs, beta = build(cache, P, t, W, m, meth, P)
                kl, ag = score(kvs, beta, P, m, ids, t, ref)
                key = f"W{W}_m{m}_{meth}"
                res.setdefault(key, []).append((kl.mean().item(), ag.mean().item()))
                fo.write(json.dumps({"trace": ri, "cut": cut, "cfg": key, "kl": kl.mean().item(), "agree": ag.mean().item(),
                                     "kl_first64": kl[:64].mean().item(), "kl_rest": kl[64:].mean().item()}) + "\n"); fo.flush()
        last_q = {li: QCAP[li][0] for li in range(NL)}
        del cache, full; torch.cuda.empty_cache() if dev == "cuda" else None
    if 'last_q' in dir() and last_q:   # this trace's queries join the bank only after its own cuts are scored
        gb = torch.Generator(device="cpu").manual_seed(ri)
        for li, qq in last_q.items():
            pool = torch.cat([BANK[li], qq], 1) if li in BANK else qq
            BANK[li] = pool[:, torch.randperm(pool.shape[1], generator=gb)[:a.bank].to(pool.device)]
        last_q = {}
    print(f"[kvfit] trace {ri + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)

print(f"[kvfit] {a.model}: {len(res.get('full', []))} cuts, N={a.N}; KV per token {KV_BYTES / 1024:.0f} KB", flush=True)
for key, v in res.items():
    kl = np.mean([x[0] for x in v]); ag = np.mean([x[1] for x in v])
    if key == "full": mem = "-"
    else:
        W, m = int(key.split("_")[0][1:]), int(key.split("_")[1][1:])
        mem = f"{(W + m) * KV_BYTES / 2**20:.0f} MB (+prompt)"
    print(f"[kvfit] {key:22s} KL {kl:.4f}  top1 agree {100 * ag:.1f}%  cache {mem}", flush=True)
