#!/usr/bin/env python3
"""Talk to a fine-tuned character model (charlm_sft.py output): a character card and a player's line, the reply sampled
(temperature, top-p, a repetition penalty), N candidates and the one the model itself finds likeliest per character.

  python3 charlm_chat.py --model npc1/model_final.pt --n 4 [--prompts prompts.json]  (a list of [card, line] pairs)
"""
import argparse, json, os, sys
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import charlm_fold as FM

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--n", type=int, default=4); ap.add_argument("--prompts", default="")
ap.add_argument("--temp", type=float, default=0.8); ap.add_argument("--top-p", type=float, default=0.9); ap.add_argument("--rep", type=float, default=1.15)
ap.add_argument("--max-chars", type=int, default=200); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--tag", default="chat")
A = ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CK = torch.load(A.model, map_location="cpu", weights_only=False); BA = CK["args"]
D, HEADS, SEQ = BA["d"], BA["heads"], BA["seq"]; FF = BA.get("ff") or 4 * D
CHARS = FM.CHARS; UNK = len(CHARS); EOD = UNK + 1; USR = EOD + 1; AST = USR + 1; V = AST + 1


class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.ln1 = nn.LayerNorm(D); s.qkv = nn.Linear(D, 3 * D); s.proj = nn.Linear(D, D); s.ln2 = nn.LayerNorm(D); s.fc = nn.Linear(D, FF); s.fc2 = nn.Linear(FF, D)
        s.qk = bool(BA.get("qknorm", 0))
        if s.qk: s.qn = nn.LayerNorm(D // HEADS); s.kn = nn.LayerNorm(D // HEADS)

    def forward(s, x):
        B, T, _ = x.shape; q, k, v = s.qkv(s.ln1(x)).split(D, -1); sh = lambda z: z.view(B, T, HEADS, D // HEADS); q, k, v = sh(q), sh(k), sh(v)
        if s.qk: q, k = s.qn(q), s.kn(k)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + s.proj(a); return x + s.fc2(F.gelu(s.fc(s.ln2(x)), approximate="tanh"))


class CharLM(nn.Module):
    def __init__(s, L):
        super().__init__(); s.emb = nn.Embedding(V, D); s.pos = nn.Embedding(SEQ, D); s.blocks = nn.ModuleList([Block() for _ in range(L)]); s.ln_f = nn.LayerNorm(D); s.head = nn.Linear(D, V)

    def forward(s, ids):
        x = s.emb(ids) + s.pos(torch.arange(ids.shape[1], device=ids.device))
        for b in s.blocks: x = b(x)
        return s.head(s.ln_f(x))


m = CharLM(CK["layers"]); m.load_state_dict(CK["model"]); m.to(DEV).eval(); torch.manual_seed(A.seed)


@torch.no_grad()
def gen(ids):
    x = torch.tensor([ids], device=DEV); out = []; lp = 0.0
    for _ in range(A.max_chars):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            lg = m(x[:, -SEQ:])[0, -1].float()
        raw = F.log_softmax(lg, -1)
        for t in set(out[-60:]): lg[t] = lg[t] / A.rep if lg[t] > 0 else lg[t] * A.rep
        pr = F.softmax(lg / A.temp, -1); sp, si = pr.sort(descending=True); keep = sp.cumsum(0) - sp < A.top_p
        nxt = int(si[keep][torch.multinomial(sp[keep] / sp[keep].sum(), 1)]); lp += float(raw[nxt])
        if nxt in (USR, EOD): break
        out.append(nxt); x = torch.cat([x, torch.tensor([[nxt]], device=DEV)], 1)
    return "".join(CHARS[i] if i < UNK else "" for i in out).strip(), lp / max(1, len(out) + 1)


P = json.load(open(A.prompts)) if A.prompts else [
    ["You are Bram, a grumpy blacksmith in a small fantasy village. You hate wasting time but you are honest.", "Hey, can you fix my sword?"],
    ["You are Lina, a cheerful innkeeper at the Silver Goose inn.", "Hi! Do you have a room for tonight?"],
    ["You are Old Tom, a fisherman in a quiet seaside town. You talk slowly and love the sea.", "What's the weather going to be like tomorrow?"],
    ["You are Kael, a nervous town guard at the north gate.", "Where can I find the mayor?"],
    ["You are Mira, a shy herbalist who lives near the forest.", "Do you know anything about the dragon in the mountains?"],
    ["You are Pip, a cheeky street kid in a busy port town.", "What's your name?"],
    ["You are Greta, a baker who wakes up very early.", "What do you recommend today?"],
    ["You are Sir Aldric, a proud knight of the royal guard.", "I challenge you to a duel!"]]
for card, line in P:
    ids = [USR] + FM.encode(card + "\n\n" + line).tolist() + [AST]
    c = [gen(ids) for _ in range(A.n)]; best = max(c, key=lambda z: z[1])
    print(f"[{A.tag}] {card.split(',')[0]} | {line}", flush=True)
    print(f"[{A.tag}]   one sample : {json.dumps(c[0][0][:220])}", flush=True)
    print(f"[{A.tag}]   best of {A.n}  : {json.dumps(best[0][:220])}", flush=True)
