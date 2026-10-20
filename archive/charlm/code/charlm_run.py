"""The character chat model for use from code: load the float or the packed 4-bit file, answer a conversation (greedy or
sampled, KV cache), on the GPU when there is one.

  m = Model("npc4/model_final.pt"); m.reply(["card ...\\n\\nHello!"])                  # turns: user, assistant, user, ...
"""
import os, sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import charlm_fold as FM  # noqa: E402

CHARS = FM.CHARS; UNK = len(CHARS); EOD = UNK + 1; USR = EOD + 1; AST = USR + 1


def load_state(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "state" not in ck: return {k: v.float() for k, v in ck["model"].items()}, ck
    st, W, sh = ck["state"], {}, torch.arange(0, 32, 4, dtype=torch.int64)   # packed 4-bit: codes * scale + bias, groups along the input
    for k, v in st.items():
        if k.endswith(".scales") or k.endswith(".biases"): continue
        key = k[:-7]
        if k.endswith(".weight") and key + ".scales" in st:
            q = (((v.to(torch.int64) & 0xFFFFFFFF).unsqueeze(-1) >> sh) & 15).reshape(v.shape[0], -1).float()
            s, b = st[key + ".scales"].float(), st[key + ".biases"].float(); g = q.shape[1] // s.shape[1]
            W[k] = (q.reshape(q.shape[0], -1, g) * s.unsqueeze(-1) + b.unsqueeze(-1)).reshape(q.shape)
        else: W[k] = v.float()
    return W, ck


class Model:
    def __init__(s, path, device=None):
        s.dev = device or ("cuda" if torch.cuda.is_available() else "cpu"); W, ck = load_state(path); a = ck["args"]
        dt = torch.bfloat16 if s.dev == "cuda" else torch.float32
        s.W = {k: v.to(s.dev, dt) for k, v in W.items()}; s.L = ck["layers"]; s.D, s.H = a["d"], a["heads"]; s.HD = s.D // s.H
        s.qk = bool(a.get("qknorm", 0)); s.seq = s.W["pos.weight"].shape[0]

    def _ln(s, x, k): return F.layer_norm(x, (x.shape[-1],), s.W[k + ".weight"], s.W[k + ".bias"], 1e-5)
    def _lin(s, x, k): return F.linear(x, s.W[k + ".weight"], s.W[k + ".bias"])

    @torch.no_grad()
    def _step(s, ids, cache):
        T = len(ids); p0 = 0 if cache[0] is None else cache[0][0].shape[2]
        x = (s.W["emb.weight"][torch.tensor(ids, device=s.dev)] + s.W["pos.weight"][p0:p0 + T])[None]
        for i in range(s.L):
            b = f"blocks.{i}"; q, k, v = s._lin(s._ln(x, b + ".ln1"), b + ".qkv").split(s.D, -1)
            q, k, v = (z.view(1, T, s.H, s.HD) for z in (q, k, v))
            if s.qk: q, k = s._ln(q, b + ".qn"), s._ln(k, b + ".kn")
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if cache[i] is not None: k = torch.cat([cache[i][0], k], 2); v = torch.cat([cache[i][1], v], 2)
            cache[i] = (k, v)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=p0 == 0 and T > 1)
            x = x + s._lin(o.transpose(1, 2).reshape(1, T, s.D), b + ".proj")
            x = x + s._lin(F.gelu(s._lin(s._ln(x, b + ".ln2"), b + ".fc"), approximate="tanh"), b + ".fc2")
        return s._lin(s._ln(x[0, -1], "ln_f"), "head").float()

    @torch.no_grad()
    def bits(s, messages):
        """bits per assistant character of a {"messages": [...]} conversation (a system message goes in front of the first user
        turn, as in charlm_sft.py), and how many characters were scored"""
        ids, mask, sys_ = [], [], ""
        for m in messages:
            if m["role"] == "system": sys_ += m["content"].strip() + "\n\n"; continue
            c = m["content"]
            if m["role"] == "user" and sys_: c = sys_ + c; sys_ = ""
            a = FM.encode(c.strip()).tolist()
            ids += [USR if m["role"] == "user" else AST] + a; mask += [0] + [m["role"] == "assistant"] * len(a)
        ids, mask = ids[:s.seq], mask[:s.seq]; T = len(ids)
        x = (s.W["emb.weight"][torch.tensor(ids, device=s.dev)] + s.W["pos.weight"][:T])[None]
        for i in range(s.L):
            b = f"blocks.{i}"; q, k, v = s._lin(s._ln(x, b + ".ln1"), b + ".qkv").split(s.D, -1)
            q, k, v = (z.view(1, T, s.H, s.HD) for z in (q, k, v))
            if s.qk: q, k = s._ln(q, b + ".qn"), s._ln(k, b + ".kn")
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
            x = x + s._lin(o.transpose(1, 2).reshape(1, T, s.D), b + ".proj")
            x = x + s._lin(F.gelu(s._lin(s._ln(x, b + ".ln2"), b + ".fc"), approximate="tanh"), b + ".fc2")
        lg = s._lin(s._ln(x[0, :-1], "ln_f"), "head").float(); nll = F.cross_entropy(lg, torch.tensor(ids[1:], device=s.dev), reduction="none")
        m = torch.tensor(mask[1:], dtype=torch.bool, device=s.dev)
        return float(nll[m].sum()) / 0.6931471805599453, int(m.sum())

    def ids(s, turns):
        out = []
        for i, t in enumerate(turns): out += [USR if i % 2 == 0 else AST] + FM.encode(t).tolist()
        return (out + [AST])[-(s.seq - 300):]

    @torch.no_grad()
    def reply(s, turns, n=300, temp=0.0, top_p=0.9):
        cache = [None] * s.L; lg = s._step(s.ids(turns), cache); out = []
        for _ in range(n):
            if temp <= 0: nxt = int(lg.argmax())
            else:
                pr = F.softmax(lg / temp, -1); sp, si = pr.sort(descending=True); keep = sp.cumsum(0) - sp < top_p
                nxt = int(si[keep][torch.multinomial(sp[keep] / sp[keep].sum(), 1)])
            if nxt in (USR, EOD): break
            out.append(nxt); lg = s._step([nxt], cache)
        return "".join(CHARS[i] if i < UNK else "" for i in out)
