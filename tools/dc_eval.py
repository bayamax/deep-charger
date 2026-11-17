"""The app's model (Deep Charger 1.0.2: hypernet-sp-distill latest/, 4-bit Qwen2 1.5B from DeepSeek-R1-Distill + the pooler) on the
AI-assistant check (ctl/assist_prompts.jsonl), with the app's own loop (release/USAGE.md section 3): no system prompt (a row's
system message goes in front of the first user turn), "<think>\n" appended, RW 768 raw tokens + 32 soft tokens from the pooler
(MAXD 384 kept by mass), rebuilt every 128 tokens, temperature 0.6, the "<information" ban, stop at end of sequence / 8 identical
tokens / reply cap after </think>. Searching: there is no search backend here, so every <search> gets "(no results)" (the same
footing as the other models in the check, which have no search either). CPU, float32; the 4-bit weights are dequantised
(the same numbers the phone computes).
  python tools/dc_eval.py <latest dir> --out dc.jsonl [--gen 2000] [--replycap 600]
Scored by tools/assist_grade.py (the answer is the text after </think>).
"""
import argparse, json, math, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, DynamicCache, Qwen2Config, Qwen2ForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("--out", required=True); ap.add_argument("--rows", default="ctl/assist_prompts.jsonl")
ap.add_argument("--rw", type=int, default=768); ap.add_argument("--maxd", type=int, default=384); ap.add_argument("--chunk", type=int, default=128)
ap.add_argument("--gen", type=int, default=2000, help="model tokens per reply (the app: 4000)"); ap.add_argument("--replycap", type=int, default=600)
ap.add_argument("--wiki", default="", help="answer <search> from Wikipedia's API (tools/wiki_env.py), page cache at this path")
ap.add_argument("--questions", default="", help="search questions (jsonl with q, gold) instead of --rows; scored: gold in the answer's first sentence")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--temp", type=float, default=0.6); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--only", default="")
a = ap.parse_args()
torch.set_num_threads(os.cpu_count() or 4); torch.manual_seed(a.seed)
D = a.dir; tok = AutoTokenizer.from_pretrained(D)
cfg = json.load(open(os.path.join(D, "config.json"))); cfg.pop("quantization", None); cfg.pop("quantization_config", None)
model = Qwen2ForCausalLM(Qwen2Config(**cfg)).eval()
raw = load_file(os.path.join(D, "model.safetensors")); P_ = dict(model.named_parameters()); P_.update(dict(model.named_buffers())); seen = set()
with torch.no_grad():
    for k in list(raw):   # one tensor at a time into the float model (memory: the float model + the 4-bit file)
        if k.endswith((".scales", ".biases")) or k not in P_: continue
        v = raw.pop(k); base = k[:-len(".weight")]
        if base + ".scales" in raw:   # MLX affine 4-bit: 8 codes per uint32, the first column in the lowest bits
            w = v.to(torch.int64) & 0xFFFFFFFF; codes = torch.stack([(w >> (4 * j)) & 15 for j in range(8)], -1).view(w.size(0), -1).float()
            sc, bi = raw.pop(base + ".scales").float(), raw.pop(base + ".biases").float(); g = codes.size(1) // sc.size(1)
            v = codes * sc.repeat_interleave(g, 1) + bi.repeat_interleave(g, 1); del w, codes
        P_[k].copy_(v.to(P_[k].dtype)); seen.add(k)
miss = [k for k in P_ if k not in seen]; del raw
print(f"[dc] model loaded, not in the file: {miss}", flush=True)
emb = model.get_input_embeddings(); H = cfg["hidden_size"]


def sinusoidal(L, Hd):
    pos = torch.arange(L).float()[:, None]; i = torch.arange(0, Hd, 2).float(); div = torch.exp(-math.log(10000.0) * i / Hd)
    pe = torch.zeros(L, Hd); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div); return pe


class Block(nn.Module):
    def __init__(s, Hd, heads, ffn):
        super().__init__(); s.lnq1 = nn.LayerNorm(Hd); s.lnk = nn.LayerNorm(Hd); s.cross = nn.MultiheadAttention(Hd, heads, batch_first=True)
        s.lnq2 = nn.LayerNorm(Hd); s.selfa = nn.MultiheadAttention(Hd, heads, batch_first=True); s.lnq3 = nn.LayerNorm(Hd)
        s.ffn = nn.Sequential(nn.Linear(Hd, ffn), nn.GELU(), nn.Linear(ffn, Hd))
    def forward(s, q, past):
        w = None
        if past.size(1) > 0:
            kn = s.lnk(past); a_, w = s.cross(s.lnq1(q), kn, kn, need_weights=True); q = q + a_
        x = s.lnq2(q); q = q + s.selfa(x, x, x, need_weights=False)[0]
        return q + s.ffn(s.lnq3(q)), w


class Pooler(nn.Module):
    def __init__(s, c):
        super().__init__(); Hd = c["hidden_dim"]; s.query = nn.Parameter(torch.zeros(c["num_soft_tokens"], Hd))
        s.blocks = nn.ModuleList([Block(Hd, c["heads"], c["ffn"]) for _ in range(c["layers"])]); s.ln_out = nn.LayerNorm(Hd)
        s.out_scale = nn.Parameter(torch.tensor(float(c["out_scale"]))); s.pe = sinusoidal(4096, Hd)
    def forward(s, past):   # past (1, L, H) -> soft prompt (1, S, H), per-token mass (L,)
        L = past.size(1); p = past + s.pe[:L][None] if L else past; q = s.query[None]; mass = torch.zeros(L)
        for b in s.blocks:
            q, w = b(q, p)
            if w is not None: mass = mass + w[0].sum(0)
        sp = s.ln_out(q); return sp / sp.norm(dim=-1, keepdim=True).clamp(min=1e-6) * s.out_scale.abs(), mass


pooler = Pooler(json.load(open(os.path.join(D, "pooler_config.json")))).eval()
print(f"[dc] pooler: {pooler.load_state_dict(load_file(os.path.join(D, 'pooler.safetensors')), strict=False)}", flush=True)
EOS = tok.eos_token_id; INFO = "<information"


def ids(t): return tok.encode(t, add_special_tokens=False)


def prompt_of(msgs):
    ms, pre = [], ""
    for m in msgs:
        if m["role"] == "system": pre += m["content"] + "\n\n"
        else: ms.append(dict(m))
    ms[0]["content"] = pre + ms[0]["content"]
    p = tok.apply_chat_template(ms, add_generation_prompt=True, tokenize=False)
    return p if p.rstrip().endswith("<think>") else p + "<think>\n"


@torch.no_grad()
def run(msgs):
    P = tok.encode(prompt_of(msgs), add_special_tokens=False); cache = DynamicCache()
    if ENV: ENV.reset()
    model(input_ids=torch.tensor([P]), past_key_values=cache, use_cache=True); MQ = len(P)
    gen, kept, absorbed, nmod, nsearch, done, last_s, inj_end = [], [], 0, 0, 0, False, -99, 0
    while not done:
        c0 = len(gen); R = min(c0, a.rw); nd = c0 - R
        if nd > absorbed: kept += gen[absorbed:nd]; absorbed = nd
        if len(kept) > a.maxd:
            _, mass = pooler(emb(torch.tensor([kept])))
            keep = sorted(mass.topk(a.maxd).indices.tolist()); kept = [kept[i] for i in keep]
        sp, _ = pooler(emb(torch.tensor([kept])) if kept else torch.zeros(1, 0, H))
        block = torch.cat([sp, emb(torch.tensor([gen[c0 - R:c0]]))], 1) if R else sp
        extra = cache.get_seq_length() - MQ
        if extra > 0: cache.crop(-extra)   # back to the prompt
        pos = MQ + block.size(1)
        last = model(inputs_embeds=block, position_ids=torch.arange(MQ, pos)[None], past_key_values=cache, use_cache=True).logits[0, -1]
        for _ in range(a.chunk):
            lg = last.float().clone(); nx = None
            for _t in range(8):   # the model may never write an information block itself
                c = int(torch.multinomial(F.softmax(lg / a.temp, -1), 1))
                tail = tok.decode(gen[max(inj_end, len(gen) - 12):] + [c])   # only what the model wrote since the last injected block
                if INFO in tail or any(tail.endswith(INFO[:j]) for j in range(4, len(INFO))): lg[c] = -float("inf"); continue
                nx = c; break
            if nx is None or nx == EOS: done = True; break
            gen.append(nx); nmod += 1
            if len(gen) >= 8 and len(set(gen[-8:])) == 1: done = True; break
            inj = None
            if "</search" in tok.decode(gen[-6:]) and len(gen) - last_s > 6:
                last_s = len(gen); nsearch += 1
                own = tok.decode(gen[inj_end:]); q = own.rsplit("<search>", 1)[-1].split("</search")[0].strip() if "<search>" in own else ""
                if ENV: inj = ENV.search(q)
                else: inj = "\n<information>(no results)</information>\n" if nsearch <= 5 else "\n<information>(no searches left - answer from what you have read)</information>\n"
            elif tok.decode(gen[-4:]).rstrip().endswith(("<more>", "<more/>", "</more>")):
                inj = ENV.more() if ENV else "\n<information>(no more of this page - search again or answer)</information>\n"
            if inj: gen += ids(inj); inj_end = len(gen); break
            full = tok.decode(gen)
            if "</think>" in full and len(ids(full.split("</think>", 1)[1])) >= a.replycap: done = True; break
            if nmod >= a.gen: done = True; break
            last = model(input_ids=torch.tensor([[nx]]), position_ids=torch.tensor([[pos]]), past_key_values=cache, use_cache=True).logits[0, -1]; pos += 1
    return tok.decode(gen, skip_special_tokens=True), nmod, nsearch


ENV = None
if a.wiki:
    from wiki_env import WikiEnv
    ENV = WikiEnv(tok, a.wiki)
if a.questions:   # search questions: distinct (q, gold) in file order, the first --n
    rows, seen = [], set()
    for l in open(a.questions):
        r = json.loads(l)
        if r["q"] in seen: continue
        seen.add(r["q"]); rows.append({"id": f"s{len(rows)}", "cat": "search", "gold": r["gold"], "messages": [{"role": "user", "content": r["q"]}]})
    rows = rows[:a.n]
else: rows = [json.loads(l) for l in open(a.rows)]


def first_sentence(t):
    t = t.split("</think>")[-1].strip(); m = re.split(r"(?<=[.!?。])\s", t, maxsplit=1); return m[0]


done_ids = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
todo = [r for r in rows if r["id"] not in done_ids and (not a.only or r["cat"] in a.only.split(","))]
todo.sort(key=lambda r: r["cat"] == "math")   # everything else first
out = open(a.out, "a"); t0 = time.time(); ntok = 0; nc = 0
for k, r in enumerate(todo):
    text, n, ns = run(r["messages"]); ntok += n; rec = {"id": r["id"], "cat": r["cat"], "think": True, "n_tok": n, "searches": ns, "text": text}
    if "gold" in r:
        rec.update(gold=r["gold"], correct=r["gold"].lower() in first_sentence(text).lower(), page_has_gold=ENV.grounded(r["gold"]) if ENV else None, api_failed=ENV.failed if ENV else None)
        nc += rec["correct"]
    out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
    print(f"[as] {k + 1}/{len(todo)} rows, {ntok} tokens, {ntok / (time.time() - t0):.1f} tok/s" + (f", correct {nc}/{k + 1}" if "gold" in r else ""), flush=True)
