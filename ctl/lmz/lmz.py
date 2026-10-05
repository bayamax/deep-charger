#!/usr/bin/env python3
"""Lossless neural compression of Wikipedia with a small language model, and the same model's sentence-end states.

A causal LM (SmolLM2-135M, Apache-2.0) gives the probability of every next token; an arithmetic coder turns the
actual token into -log2 p bits. Decoding replays the same model token by token and reads the tokens back, so the
text comes back byte for byte. Both directions run the identical incremental computation (one token at a time on a
KV cache, float32) so the probabilities match bit for bit; the probabilities are quantised to integer frequencies
(every token at least 1) before coding.

  bpc        bits per character over held-out articles (parallel forward, the code length the coder would reach),
             with zlib / lzma on the same text for reference
  roundtrip  real encode -> bits -> decode on a few articles; asserts the text comes back exactly
  vec        each sentence alone through the LM; the final-layer state at its last token, L2-normalised, as the
             sentence vector, standardised per dimension (the raw states share one dominant direction: any two
             sentences sit at cosine 0.85-0.95) and L2-normalised - written as vec_XXX.npy (int8) + scl_XXX.npy + off_XXX.npy next to docs_XXX.jsonl,
             the layout the sentence BART reads (sentbart/train.py). Alone, not in its article: a state read in
             context carries every earlier sentence, so a hidden sentence would leak into the ones after it.

  python3 lmz.py bpc --docs /root/lz/docs/docs_001.jsonl --n 300
  python3 lmz.py roundtrip --docs /root/lz/docs/docs_001.jsonl --n 10
  python3 lmz.py vec --dir /root/lz/docs --shards 000,001
"""
import argparse, glob, json, lzma, math, os, random, sys, time, zlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["bpc", "roundtrip", "vec"])
ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
ap.add_argument("--docs", default=""); ap.add_argument("--dir", default=""); ap.add_argument("--shards", default="")
ap.add_argument("--n", type=int, default=300); ap.add_argument("--ctx", type=int, default=2048)
ap.add_argument("--batch", type=int, default=512); ap.add_argument("--maxlen", type=int, default=96)
ap.add_argument("--out", default="")
A = ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(A.model)
dtype = torch.float32 if A.mode == "roundtrip" else torch.bfloat16
lm = AutoModelForCausalLM.from_pretrained(A.model, torch_dtype=dtype).to(DEV).eval()
BOS = tok.bos_token_id if tok.bos_token_id is not None else 0


def articles(path, n, seed=0):
    rows = [json.loads(l) for l in open(path)]
    random.Random(seed).shuffle(rows)
    return [(r.get("title", ""), "\n".join(r["sents"])) for r in rows[:n]]


@torch.no_grad()
def bits_parallel(text):
    """code length in bits of text under the LM (BOS-primed, windows of --ctx tokens, each window re-primed)"""
    ids = tok.encode(text, add_special_tokens=False); total = 0.0
    for s in range(0, len(ids), A.ctx - 1):
        w = [BOS] + ids[s:s + A.ctx - 1]
        x = torch.tensor([w], device=DEV)
        lp = torch.log_softmax(lm(x).logits[0, :-1].float(), -1)
        total += -lp.gather(1, x[0, 1:, None]).sum().item() / math.log(2)
    return total, len(ids)


# ---- arithmetic coder (Witten-Neal-Cleary, Python integers) ----
PREC = 62; FULL = 1 << PREC; HALF = FULL >> 1; QUART = FULL >> 2; TOTAL = 1 << 24


def freqs(logits):
    p = torch.softmax(logits.double(), -1).cpu().numpy()
    f = np.floor(p * (TOTAL - len(p))).astype(np.int64) + 1
    c = np.concatenate([[0], np.cumsum(f)]); return c, int(c[-1])


class Enc:
    def __init__(s): s.lo, s.hi, s.pend, s.bits = 0, FULL - 1, 0, []
    def put(s, b):
        s.bits.append(b); s.bits.extend([1 - b] * s.pend); s.pend = 0
    def code(s, lo_c, hi_c, tot):
        r = s.hi - s.lo + 1
        s.hi = s.lo + r * hi_c // tot - 1; s.lo = s.lo + r * lo_c // tot
        while True:
            if s.hi < HALF: s.put(0)
            elif s.lo >= HALF: s.put(1); s.lo -= HALF; s.hi -= HALF
            elif s.lo >= QUART and s.hi < 3 * QUART: s.pend += 1; s.lo -= QUART; s.hi -= QUART
            else: break
            s.lo *= 2; s.hi = s.hi * 2 + 1
    def finish(s):
        s.pend += 1; s.put(0 if s.lo < QUART else 1); return s.bits


class Dec:
    def __init__(s, bits):
        s.bits, s.i, s.lo, s.hi, s.v = bits, 0, 0, FULL - 1, 0
        for _ in range(PREC): s.v = 2 * s.v + s.nb()
    def nb(s):
        b = s.bits[s.i] if s.i < len(s.bits) else 0; s.i += 1; return b
    def code(s, c, tot):
        r = s.hi - s.lo + 1
        t = ((s.v - s.lo + 1) * tot - 1) // r
        k = int(np.searchsorted(c, t, side="right")) - 1
        s.hi = s.lo + r * int(c[k + 1]) // tot - 1; s.lo = s.lo + r * int(c[k]) // tot
        while True:
            if s.hi < HALF: pass
            elif s.lo >= HALF: s.lo -= HALF; s.hi -= HALF; s.v -= HALF
            elif s.lo >= QUART and s.hi < 3 * QUART: s.lo -= QUART; s.hi -= QUART; s.v -= QUART
            else: break
            s.lo *= 2; s.hi = s.hi * 2 + 1; s.v = 2 * s.v + s.nb()
        return k


@torch.no_grad()
def stepper():
    """the incremental LM both directions share: feed one token, get the next-token logits"""
    state = {"past": None}
    def step(t):
        out = lm(torch.tensor([[t]], device=DEV), past_key_values=state["past"], use_cache=True)
        state["past"] = out.past_key_values; return out.logits[0, -1]
    return step


@torch.no_grad()
def encode(ids):
    st = stepper(); e = Enc(); lg = st(BOS)
    for t in ids:
        c, tot = freqs(lg); e.code(int(c[t]), int(c[t + 1]), tot); lg = st(t)
    return e.finish()


@torch.no_grad()
def decode(bits, n):
    st = stepper(); d = Dec(bits); lg = st(BOS); out = []
    for _ in range(n):
        c, tot = freqs(lg); t = d.code(c, tot); out.append(t); lg = st(t)
    return out


if A.mode == "bpc":
    arts = articles(A.docs, A.n); B = C = T = 0; Z = X = 0
    t0 = time.time()
    for _, text in arts:
        b, nt = bits_parallel(text); raw = text.encode("utf-8")
        B += b; C += len(raw); T += nt; Z += len(zlib.compress(raw, 9)); X += len(lzma.compress(raw, preset=9))
    print(f"[bpc] {len(arts)} articles, {C/1e6:.2f} MB, {T} tokens: LM {B/C:.3f} bits/byte ({B/8/C*100:.1f}% of raw), "
          f"zlib-9 {8*Z/C:.3f} ({Z/C*100:.1f}%), lzma-9 {8*X/C:.3f} ({X/C*100:.1f}%), {B/T:.2f} bits/token, {time.time()-t0:.0f} s", flush=True)
    print(f"BPC_DONE lm={B/C:.4f} zlib={8*Z/C:.4f} lzma={8*X/C:.4f}", flush=True)

elif A.mode == "roundtrip":
    arts = articles(A.docs, A.n, seed=1); ok = 0; B = C = 0; t0 = time.time()
    for title, text in arts:
        text = text[:3000]   # a page's first part keeps the token-by-token Python coder to minutes
        ids = tok.encode(text, add_special_tokens=False)
        bits = encode(ids); back = tok.decode(decode(bits, len(ids)))
        same = back == text; ok += same; B += len(bits); C += len(text.encode("utf-8"))
        print(f"[roundtrip] {title[:40]!r}: {len(text.encode('utf-8'))} bytes -> {len(bits)} bits ({len(bits)/8/len(text.encode('utf-8'))*100:.1f}%), exact {same}", flush=True)
    print(f"ROUNDTRIP_DONE {ok}/{len(arts)} exact, {B/C:.3f} bits/byte overall, {time.time()-t0:.0f} s", flush=True)

else:
    hid = lm.config.hidden_size
    shards = A.shards.split(",") if A.shards else [os.path.basename(f)[5:-6] for f in sorted(glob.glob(os.path.join(A.dir, "docs_*.jsonl")))]

    @torch.no_grad()
    def embed(sents):
        order = np.argsort([len(s) for s in sents]); out = np.zeros((len(sents), hid), np.float32)
        for i in range(0, len(sents), A.batch):
            idx = order[i:i + A.batch]
            enc = [[BOS] + tok.encode(sents[j], add_special_tokens=False)[:A.maxlen - 1] for j in idx]
            L = max(len(e) for e in enc)
            x = torch.zeros((len(enc), L), dtype=torch.long); m = torch.zeros((len(enc), L), dtype=torch.long)
            for r, e in enumerate(enc): x[r, :len(e)] = torch.tensor(e); m[r, :len(e)] = 1   # right-padded
            h = lm(x.to(DEV), attention_mask=m.to(DEV), output_hidden_states=True).hidden_states[-1].float()
            last = m.sum(1).to(DEV) - 1
            out[idx] = h[torch.arange(len(enc), device=DEV), last].cpu().numpy()
        return out

    STATS = os.path.join(A.dir, "lm_stats.npz")
    def standardise(v):
        if not os.path.exists(STATS):   # mean / std per dimension from the first chunk seen, shared by every shard
            np.savez(STATS, mu=v.mean(0), sd=v.std(0) + 1e-6); print(f"[vec] stats from {len(v)} sentences", flush=True)
        st_ = np.load(STATS); v = (v - st_["mu"]) / st_["sd"]
        return v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-8)

    for k in shards:
        src = os.path.join(A.dir, f"docs_{k}.jsonl"); vf = os.path.join(A.dir, f"vec_{k}.npy")
        if os.path.exists(vf): print(f"[vec] {k} already done", flush=True); continue
        t0 = time.time(); off = [0]; chunks_q, chunks_s = [], []; buf = []
        def flush():
            v = standardise(embed(buf)); s = np.abs(v).max(1).clip(1e-8)
            chunks_q.append(np.round(v / s[:, None] * 127).astype(np.int8)); chunks_s.append(s.astype(np.float16)); buf.clear()
        for line in open(src):
            d = json.loads(line); buf.extend(d["sents"]); off.append(off[-1] + len(d["sents"]))
            if len(buf) >= 200000: flush()
        if buf: flush()
        np.save(vf + ".tmp.npy", np.concatenate(chunks_q)); os.replace(vf + ".tmp.npy", vf)
        np.save(os.path.join(A.dir, f"scl_{k}.npy"), np.concatenate(chunks_s)); np.save(os.path.join(A.dir, f"off_{k}.npy"), np.array(off, np.int64))
        print(f"[vec] {k}: {len(off)-1} documents, {off[-1]} sentences, dim {hid}, {(time.time()-t0)/60:.1f} min", flush=True)
    print("VEC_DONE", flush=True)
