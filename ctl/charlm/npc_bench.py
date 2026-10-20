#!/usr/bin/env python3
"""The character model's replies to npc_gen.py's held-out probes, for the judge; and the report once they are graded.

  reply   for each probe, N sampled replies in one batch (KV cache); keeps the first sample and the likeliest of the N
          (mean log-probability per character), with rule checks: the answer's words present, AI talk, an echo of the
          player's line, length
  report  per model and probe kind from the judged rows: usable %, the four scores, answer hit %
  sample  for situations cut from the training dialogues (the card, the turns so far, the player's line), N replies each,
          all kept, for the judge
  pick    the judged samples -> the best usable reply per situation as a training dialogue (rejection sampling)
  turns   every assistant turn of conversations (e.g. agent_gen.py's) answered once, the earlier turns as written, for
          scoring against the written answer ("gold")
  rank    a reranker for the game (no judge there): logistic regression on the model's own view of a reply (its last
          hidden state, the mean over the reply, the reply's log-probability and length), fit to the judge's 'usable' on
          the judged samples; tested on held-out situations as "pick the top of N" against the likeliest and the first;
          with --ranker, reply also writes the ranker's pick

  python3 npc_bench.py reply --model npc1/model_final.pt --tag npc1 --bench npc_bench.jsonl --out replies.jsonl
  python3 npc_bench.py report --inp judged.jsonl
  python3 npc_bench.py sample --model npc2/model_final.pt --tag npc2 --bench train_3.jsonl --situations 3000 --n 6 --out samples.jsonl
  python3 npc_bench.py pick --inp samples_judged.jsonl --out rft.jsonl
  python3 npc_bench.py rank --model npc2/model_final.pt --inp samples_judged.jsonl --out rank.pt
"""
import argparse, collections, json, re, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["reply", "report", "sample", "pick", "rank", "turns"])
ap.add_argument("--ranker", default="", help="reply: a rank.pt; its pick of the N is written too")
ap.add_argument("--model", default=""); ap.add_argument("--tag", default=""); ap.add_argument("--bench", default=""); ap.add_argument("--out", default="")
ap.add_argument("--inp", default=""); ap.add_argument("--n", type=int, default=4); ap.add_argument("--max-chars", type=int, default=300)
ap.add_argument("--situations", type=int, default=3000); ap.add_argument("--temp", type=float, default=0.7); ap.add_argument("--top-p", type=float, default=0.9); ap.add_argument("--seed", type=int, default=0)
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


if A.mode in ("reply", "sample", "rank", "turns"):
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
    def feats(ctx, reply_ids, lp):
        """the model's view of a reply: its final hidden state (after the reply and the turn's end), the mean over the
        reply, the mean log-probability per character and the length"""
        seq = (ctx + reply_ids + [USR])[-W["pos.weight"].shape[0]:]; ids = torch.tensor([seq], device=DEV); T = ids.shape[1]; n = len(reply_ids) + 1
        x = W["emb.weight"][ids] + W["pos.weight"][:T]
        for i in range(L):
            b = f"blocks.{i}"; q, k, v = lin(ln(x, b + ".ln1"), b + ".qkv").split(D, -1)
            q, k, v = (z.view(1, T, H, HD) for z in (q, k, v))
            if QK: q, k = ln(q, b + ".qn"), ln(k, b + ".kn")
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
            x = x + lin(o.transpose(1, 2).reshape(1, T, D), b + ".proj")
            x = x + lin(F.gelu(lin(ln(x, b + ".ln2"), b + ".fc"), approximate="tanh"), b + ".fc2")
        h = ln(x[0], "ln_f").float()
        return torch.cat([h[-1], h[-n:].mean(0), torch.tensor([lp, len(reply_ids) / 100.0], device=DEV)])

    RK = torch.load(A.ranker, map_location=DEV) if A.ranker else None
    def rank_score(f): return float(((f - RK["mu"]) / RK["sd"]) @ RK["w"] + RK["b"])

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

    def conv_turn_ids(turns):   # turns as written (the first one carries the card already)
        ids = []
        for i, t in enumerate(turns): ids += [USR if i % 2 == 0 else AST] + FM.encode(t).tolist()
        return ids + [AST]

    def conv_ids(card, turns):
        ids = []
        for i, t in enumerate(turns): ids += [USR if i % 2 == 0 else AST] + FM.encode((card + "\n\n" + t) if i == 0 else t).tolist()
        return ids + [AST]

    T0 = time.time(); n = 0
    if A.mode == "reply":
        rows = [json.loads(l) for l in open(A.bench)]
        with open(A.out, "a") as f:
            for r in rows:
                for p in r["probes"]:
                    ctx = conv_ids(r["card"], p["turns"]); txt, lps = sample(ctx, A.n); best = max(range(A.n), key=lambda j: lps[j])
                    picks = [("first", 0), ("best", best)]
                    if RK is not None: picks.append(("rank", max(range(A.n), key=lambda j: rank_score(feats(ctx, FM.encode(txt[j]).tolist(), lps[j])))))
                    for pick, j in picks:
                        s = txt[j]
                        f.write(json.dumps({"model": A.tag, "pick": pick, "kind": p["kind"], "card": r["card"], "turns": p["turns"], "answer": p.get("answer", ""),
                                            "reply": s, "hit": hit(s, p["answer"]) if p.get("answer") else None, "ai_talk": bool(AI_TALK.search(s)),
                                            "echo": echo(s, p["turns"][-1]), "chars": len(s)}) + "\n")
                    n += 1
    elif A.mode == "turns":
        with open(A.out, "w") as f:
            for l in open(A.bench):
                ms = json.loads(l)["messages"]
                for k in range(1, len(ms), 2):
                    ctx = [m["content"] for m in ms[:k]]; txt, _ = sample(conv_turn_ids(ctx), 1)
                    f.write(json.dumps({"model": A.tag, "turns": ctx, "gold": ms[k]["content"], "reply": txt[0]}) + "\n"); n += 1
    elif A.mode == "rank":
        rows = [json.loads(l) for l in open(A.inp)]; X = []; y = []; sid = []
        for r in rows:
            X.append(feats(conv_ids(r["card"], r["turns"]), FM.encode(r["reply"]).tolist(), r["lp"])); y.append(float(bool(r["judge"].get("usable")))); sid.append(r["sid"])
        X = torch.stack(X); y = torch.tensor(y, device=DEV); sid = torch.tensor(sid, device=DEV)
        test = (sid % 5) == 0; tr = ~test; mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-4; Z = (X - mu) / sd
        w = torch.zeros(Z.shape[1], device=DEV, requires_grad=True); b0 = torch.zeros((), device=DEV, requires_grad=True)
        opt = torch.optim.Adam([w, b0], lr=0.01)
        for it in range(600):
            opt.zero_grad(); z = Z[tr] @ w + b0
            loss = F.binary_cross_entropy_with_logits(z, y[tr]) + 1e-3 * (w ** 2).sum(); loss.backward(); opt.step()
        with torch.no_grad():
            sc = Z @ w + b0; lpv = X[:, -2]; res = collections.Counter(); nsit = 0
            for k in torch.unique(sid[test]).tolist():
                m = (sid == k).nonzero().flatten()
                if len(m) < 2: continue
                nsit += 1; res["first"] += float(y[m[0]]); res["likeliest"] += float(y[m[lpv[m].argmax()]]); res["ranker"] += float(y[m[sc[m].argmax()]])
                res["any"] += float(y[m].max()); res["mean"] += float(y[m].mean())
        torch.save({"w": w.detach(), "b": b0.detach(), "mu": mu, "sd": sd, "features": "last hidden, mean over reply, lp, chars/100"}, A.out)
        print(f"[rank] {len(rows)} judged replies ({100 * float(y.mean()):.0f}% usable); held-out {nsit} situations, usable % of the one picked: "
              + ", ".join(f"{k} {100 * v / max(1, nsit):.1f}" for k, v in res.items()) + f" -> {A.out}", flush=True)
    else:   # situations: a training dialogue cut before one of its NPC turns (a random one), N replies each
        import random; rng = random.Random(A.seed); dl = [json.loads(l) for l in open(A.bench)]; rng.shuffle(dl)
        with open(A.out, "w") as f:
            for k, d in enumerate(dl[:A.situations]):
                m = d["messages"]; card = m[0]["content"]; turns = [x["content"] for x in m[1:]]
                cut = rng.randrange(len(turns) // 2) * 2 + 1; ctx = turns[:cut]
                if len(card) + sum(len(t) for t in ctx) > 3600: continue
                txt, lps = sample(conv_ids(card, ctx), A.n)
                for j in range(A.n):
                    f.write(json.dumps({"model": A.tag, "sid": k, "j": j, "lp": lps[j], "card": card, "turns": ctx, "reference": turns[cut], "reply": txt[j],
                                        "ai_talk": bool(AI_TALK.search(txt[j])), "echo": echo(txt[j], ctx[-1]), "chars": len(txt[j])}) + "\n")
                n += 1
    if A.mode != "rank": print(f"[bench] {A.mode} {A.tag}: {n} {'probes' if A.mode == 'reply' else 'turns' if A.mode == 'turns' else 'situations'} x {A.n} samples [{(time.time() - T0) / 60:.1f} min] -> {A.out}", flush=True)
elif A.mode == "pick":
    rows = [json.loads(l) for l in open(A.inp)]; S = collections.defaultdict(list)
    for r in rows: S[r["sid"]].append(r)
    score = lambda r: (sum((r["judge"].get(k) or 0) for k in ("answers", "consistent", "in_character", "fluent")), r["lp"])
    n_sit = len(S); kept = 0; usable = sum(bool(r["judge"].get("usable")) for r in rows)
    with open(A.out, "w") as f:
        for sid, rs in S.items():
            ok = [r for r in rs if r["judge"].get("usable") and not r["ai_talk"] and not r["echo"] and r["reply"]]
            if not ok: continue
            b = max(ok, key=score); msgs = [{"role": "system", "content": b["card"]}]
            for i, t in enumerate(b["turns"] + [b["reply"]]): msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": t})
            f.write(json.dumps({"messages": msgs}) + "\n"); kept += 1
    print(f"[pick] {len(rows)} samples, {100 * usable / max(1, len(rows)):.0f}% usable; {kept} of {n_sit} situations have one -> {A.out}", flush=True)
else:
    rows = [json.loads(l) for l in open(A.inp)]; G = collections.defaultdict(list)
    for r in rows:
        G[(r["model"], r["pick"], "ALL")].append(r); G[(r["model"], r["pick"], r["kind"])].append(r)
    nat = lambda x: (x.get("answers") or 0) >= 1 and (x.get("fluent") or 0) >= 1 and (x.get("consistent") or 0) >= 1   # a sensible, coherent reply
    print(f"{'model':<14}{'pick':<6}{'kind':<10}{'n':>5}{'natural':>8}{'usable':>8}{'answers':>8}{'consist':>8}{'in-char':>8}{'fluent':>8}{'hit':>6}{'AI':>5}{'echo':>6}{'chars':>6}")
    for (m, pk, k), rs in sorted(G.items(), key=lambda z: (z[0][0], z[0][1], z[0][2] != "ALL", z[0][2])):
        j = [r["judge"] for r in rs]; mean = lambda key: sum((x.get(key) or 0) for x in j) / len(j)
        hs = [r["hit"] for r in rs if r["hit"] is not None]
        print(f"{m:<14}{pk:<6}{k:<10}{len(rs):>5}{100 * sum(nat(x) for x in j) / len(j):>7.0f}%{100 * sum(bool(x.get('usable')) for x in j) / len(j):>7.0f}%{mean('answers'):>8.2f}{mean('consistent'):>8.2f}"
              f"{mean('in_character'):>8.2f}{mean('fluent'):>8.2f}{(f'{100 * sum(hs) / len(hs):.0f}%' if hs else '-'):>6}"
              f"{100 * sum(r['ai_talk'] for r in rs) / len(rs):>4.0f}%{100 * sum(r['echo'] for r in rs) / len(rs):>5.0f}%{sum(r['chars'] for r in rs) / len(rs):>6.0f}")
