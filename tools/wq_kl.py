"""Loss of 4-bit weights measured on conversations nothing was tuned on: the 16-bit model's own full-context outputs on the eval rows
(archive/sp3/mt/mt_full_s1.jsonl, docc_full_s1.jsonl), read teacher-forced by the 16-bit model and by each 4-bit version. Prints
the mean KL(16-bit || 4-bit) per token, the worst conversation, and how often the most likely next token agrees.
  python tools/wq_kl.py mt_full_s1.jsonl,docc_full_s1.jsonl wq_rtn.pt wq_gptq.pt wq_dwq.pt ... [--device cpu --n-conv 40 --maxt 2048]
"""
import json, sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse, random
ap = argparse.ArgumentParser(); ap.add_argument("rows"); ap.add_argument("wq", nargs="+")
ap.add_argument("--device", default="cuda"); ap.add_argument("--maxt", type=int, default=4096)
ap.add_argument("--n-conv", type=int, default=0, help="a fixed random subset of this many conversations (0: all)")
a = ap.parse_args(); dev = a.device; MAXT = a.maxt
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
rows = [json.loads(l) for f in a.rows.split(",") for l in open(f)]
if a.n_conv: rows = random.Random(0).sample(rows, a.n_conv)
X = [torch.tensor(tok.encode((r.get("prompt") or "") + r["text"], add_special_tokens=False)[:MAXT]) for r in rows]
T = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
S = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
W0 = {k: v.detach().cpu().clone() for k, v in S.state_dict().items()}


def put(path):
    S.load_state_dict(W0)
    d = torch.load(path, map_location="cpu"); g = d["group"]
    for k, v in d.items():
        if not isinstance(v, dict): continue
        w = S.get_submodule(k).weight; p = v["q"].to(dev); q = torch.stack([p & 15, p >> 4], 2).view(p.size(0), -1).float()
        w.data.copy_((q * v["s"].to(dev).float().repeat_interleave(g, 1) + v["b"].to(dev).float().repeat_interleave(g, 1)).to(w.dtype))


print(f"[wq] eval conversations {len(X)}, {sum(len(x) for x in X)} tokens (each cut at {MAXT})", flush=True)
for path in a.wq:
    put(path); kl = ag = n = 0.0; worst = 0.0; per = {}
    with torch.no_grad():
        for r, x in zip(rows, X):
            x = x[None].to(dev); Lt = T(input_ids=x).logits[0, :-1]; Ls = S(input_ids=x).logits[0, :-1]; ks = []
            for c in range(0, Lt.size(0), 512):   # in pieces: a 4096 x 152k float tensor is 2.5 GB
                lt = F.log_softmax(Lt[c:c + 512].float(), -1); ls = F.log_softmax(Ls[c:c + 512].float(), -1)
                ks.append((lt.exp() * (lt - ls)).sum(-1)); ag += float((lt.argmax(-1) == ls.argmax(-1)).float().sum())
            k = torch.cat(ks); kl += float(k.sum()); n += k.numel(); worst = max(worst, float(k.mean()))
            t = r.get("task", "?"); per.setdefault(t, []).append(float(k.mean())); del Lt, Ls
    print(f"[wq] {path}: eval KL {kl / n:.4f} (worst conversation {worst:.4f}) | top-1 agreement {ag / n:.4f} | per task "
          + " ".join(f"{t} {sum(v) / len(v):.4f}" for t, v in sorted(per.items())), flush=True)
