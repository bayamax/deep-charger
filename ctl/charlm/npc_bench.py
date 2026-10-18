#!/usr/bin/env python3
"""The character model's replies to npc_gen.py's held-out probes, for the judge; and the report once they are graded.

  reply   for each probe, N sampled replies in one batch (KV cache); keeps the first sample and the likeliest of the N
          (mean log-probability per character), with rule checks: the answer's words present, AI talk, an echo of the
          player's line, length
  report  per model and probe kind from the judged rows: usable %, the four scores, answer hit %

  python3 npc_bench.py reply --model npc1/model_final.pt --tag npc1 --bench npc_bench.jsonl --out replies.jsonl
  python3 npc_bench.py report --inp judged.jsonl
"""
import argparse, collections, json, re, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["reply", "report"])
ap.add_argument("--model", default=""); ap.add_argument("--tag", default=""); ap.add_argument("--bench", default=""); ap.add_argument("--out", default="")
ap.add_argument("--inp", default=""); ap.add_argument("--n", type=int, default=4); ap.add_argument("--max-chars", type=int, default=300)
ap.add_argument("--temp", type=float, default=0.7); ap.add_argument("--top-p", type=float, default=0.9); ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
AI_TALK = re.compile(r"\b(as an ai|language model|i'm an ai|i am an ai|virtual assistant|ai assistant|text-based|real-time)\b", re.I)


def words(s): return re.findall(r"[a-z0-9]+", s.lower())


def hit(reply, answer):
    """every word of the answer (numbers and names) in the reply"""
    r = set(words(reply)); a = [w for w in words(answer) if w not in ("the", "a", "an", "of", "at", "in", "on", "to", "and")]
    return bool(a) and all(w in r for w in a)


def echo(reply, line):
    a, b = words(reply), words(line)
    return bool(a) and len(set(a) & set(b)) / max(1, len(set(a))) > 0.8 and len(a) <= len(b) + 2


if A.mode == "reply":
    import torch, torch.nn.functional as F
    sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
    import charlm_fold as FM
    CHARS = FM.CHARS; UNK = len(CHARS); EOD = UNK + 1; USR = EOD + 1; AST = USR + 1
    DEV = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(A.seed)
    ck = torch.load(A.model, map_location="cpu", weights_only=False); a = ck["args"]; L = ck["layers"]
    if "state" in ck:   # the packed 4-bit file: dequantize (codes * scale + bias, groups along the input)
        st, W, sh = ck["state"], {}, torch.arange(0, 32, 4, dtype=torch.int64)
        for k, v in st.items():
            if k.endswith(".scales") or k.endswith(".biases"): continue
            key = k[:-7]
            if k.endswith(".weight") and key + ".scales" in st:
                q = (((v.to(torch.int64) & 0xFFFFFFFF).unsqueeze(-1) >> sh) & 15).reshape(v.shape[0], -1).float()
                s, b = st[key + ".scales"].float(), st[key + ".biases"].float(); g = q.shape[1] // s.shape[1]
                W[k] = (q.reshape(q.shape[0], -1, g) * s.unsqueeze(-1) + b.unsqueeze(-1)).reshape(q.shape)
            else: W[k] = v.float()
    else: W = {k: v.float() for k, v in ck["model"].items()}
    DT = torch.bfloat16 if DEV == "cuda" else torch.float32
    W = {k: v.to(DEV, DT) for k, v in W.items()}; D, H = a["d"], a["heads"]; HD = D // H; QK = bool(a.get("qknorm", 0))

    def ln(x, k): return F.layer_norm(x, (x.shape[-1],), W[k + ".weight"], W[k + ".bias"], 1e-5)
    def lin(x, k): return F.linear(x, W[k + ".weight"], W[k + ".bias"])

    @torch.no_grad()
    def step(ids, cache):
        """ids [B, T] new symbols; cache: per layer (k, v) or None. Logits of the last position [B, V]."""
        B, T = ids.shape; p0 = 0 if cache[0] is None else cache[0][0].shape[2]
        x = W["emb.weight"][ids] + W["pos.weight"][p0:p0 + T]
        for i in range(L):
            b = f"blocks.{i}"; q, k, v = lin(ln(x, b + ".ln1"), b + ".qkv").split(D, -1)
            q, k, v = (z.view(B, T, H, HD) for z in (q, k, v))
            if QK: q, k = ln(q, b + ".qn"), ln(k, b + ".kn")
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if cache[i] is not None: k = torch.cat([cache[i][0], k], 2); v = torch.cat([cache[i][1], v], 2)
            cache[i] = (k, v)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=p0 == 0 and T > 1)
            x = x + lin(o.transpose(1, 2).reshape(B, T, D), b + ".proj")
            x = x + lin(F.gelu(lin(ln(x, b + ".ln2"), b + ".fc"), approximate="tanh"), b + ".fc2")
        return lin(ln(x[:, -1], "ln_f"), "head").float()

    @torch.no_grad()
    def sample(ids, n):
        cache = [None] * L; lg = step(torch.tensor([ids] * n, device=DEV), cache)
        outs = [[] for _ in range(n)]; lp = torch.zeros(n, device=DEV); done = torch.zeros(n, dtype=torch.bool, device=DEV)
        for _ in range(A.max_chars):
            raw = F.log_softmax(lg, -1); pr = F.softmax(lg / A.temp, -1); sp, si = pr.sort(-1, descending=True)
            sp = torch.where(sp.cumsum(-1) - sp < A.top_p, sp, torch.zeros_like(sp)); nxt = si.gather(1, torch.multinomial(sp, 1)).squeeze(1)
            lp += torch.where(done, torch.zeros_like(lp), raw.gather(1, nxt[:, None]).squeeze(1))
            stop = (nxt == USR) | (nxt == EOD)
            for j in range(n):
                if not done[j] and not stop[j]: outs[j].append(int(nxt[j]))
            done |= stop
            if bool(done.all()): break
            lg = step(nxt[:, None], cache)
        txt = ["".join(CHARS[i] if i < UNK else "" for i in o).strip() for o in outs]
        return txt, (lp / torch.tensor([len(o) + 1 for o in outs], device=DEV)).tolist()

    rows = [json.loads(l) for l in open(A.bench)]; T0 = time.time(); n = 0
    with open(A.out, "a") as f:
        for r in rows:
            for p in r["probes"]:
                ids = []
                for i, t in enumerate(p["turns"]):
                    ids += [USR if i % 2 == 0 else AST] + FM.encode((r["card"] + "\n\n" + t) if i == 0 else t).tolist()
                txt, lps = sample(ids + [AST], A.n); best = max(range(A.n), key=lambda j: lps[j])
                for pick, j in (("first", 0), ("best", best)):
                    s = txt[j]
                    f.write(json.dumps({"model": A.tag, "pick": pick, "kind": p["kind"], "card": r["card"], "turns": p["turns"], "answer": p.get("answer", ""),
                                        "reply": s, "hit": hit(s, p["answer"]) if p.get("answer") else None, "ai_talk": bool(AI_TALK.search(s)),
                                        "echo": echo(s, p["turns"][-1]), "chars": len(s)}) + "\n")
                n += 1
    print(f"[bench] {A.tag}: {n} probes x {A.n} samples [{(time.time() - T0) / 60:.1f} min] -> {A.out}", flush=True)
else:
    rows = [json.loads(l) for l in open(A.inp)]; G = collections.defaultdict(list)
    for r in rows:
        G[(r["model"], r["pick"], "ALL")].append(r); G[(r["model"], r["pick"], r["kind"])].append(r)
    print(f"{'model':<14}{'pick':<6}{'kind':<10}{'n':>5}{'usable':>8}{'answers':>8}{'consist':>8}{'in-char':>8}{'fluent':>8}{'hit':>6}{'AI':>5}{'echo':>6}{'chars':>6}")
    for (m, pk, k), rs in sorted(G.items(), key=lambda z: (z[0][0], z[0][1], z[0][2] != "ALL", z[0][2])):
        j = [r["judge"] for r in rs]; mean = lambda key: sum((x.get(key) or 0) for x in j) / len(j)
        hs = [r["hit"] for r in rs if r["hit"] is not None]
        print(f"{m:<14}{pk:<6}{k:<10}{len(rs):>5}{100 * sum(bool(x.get('usable')) for x in j) / len(j):>7.0f}%{mean('answers'):>8.2f}{mean('consistent'):>8.2f}"
              f"{mean('in_character'):>8.2f}{mean('fluent'):>8.2f}{(f'{100 * sum(hs) / len(hs):.0f}%' if hs else '-'):>6}"
              f"{100 * sum(r['ai_talk'] for r in rs) / len(rs):>4.0f}%{100 * sum(r['echo'] for r in rs) / len(rs):>5.0f}%{sum(r['chars'] for r in rs) / len(rs):>6.0f}")
