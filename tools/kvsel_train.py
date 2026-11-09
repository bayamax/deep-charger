"""Train the learned KV selector (option 5: which older entries to keep, judged by KL) on a kvsel dump.

  python kvsel_train.py DUMP.pt OUT.pt [--epochs 30] [--vocab 151936]
Each position of the dump holds the older entries' features and the labels: 'fut' (the attention the entry receives from the next 4 chunks) and 'abl' (the KL increase on the next
chunk when the entry's block is removed). Pairwise ranking loss (RankNet) between entries of different blocks of a position,
weighted by the label difference. Prints the held-out (last 10% of positions) rank correlation with the label and the share of the
label's top-B mass the selector's top-B covers, against the mean-attention rule."""
import argparse, math, os, random, sys
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvsel_feat import Selector, numeric, N_NUM

ap = argparse.ArgumentParser(); ap.add_argument("dump"); ap.add_argument("out"); ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--vocab", type=int, default=151936); ap.add_argument("--pairs", type=int, default=4096); ap.add_argument("--B", type=int, default=512)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--label", default="fut", choices=["fut", "abl"], help="fut: attention the entry receives from the next 4 chunks (block 1); abl: exact block ablation"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--block", type=int, default=16)
a = ap.parse_args(); torch.manual_seed(a.seed); random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
D = [d for d in torch.load(a.dump) if d.get(a.label) is not None]
if a.label == "fut": a.block = 1
random.shuffle(D); nh = max(1, len(D) // 10); held, train = D[:nh], D[nh:]
print(f"{len(D)} positions: {len(train)} train, {len(held)} held-out | entries per position {sum(d['abl'].numel() for d in D) // len(D)}")
X = torch.cat([numeric(d["sum"], d["nobs"], d["age"], d["rel"], d["vnorm"], d["knorm"]) for d in train])
mu, sd = X.mean(0), X.std(0).clamp(min=1e-6)
model = Selector(a.vocab, mu=mu, sd=sd).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
prep = lambda d: (numeric(d["sum"], d["nobs"], d["age"], d["rel"], d["vnorm"], d["knorm"]).to(dev), d["tok"].to(dev), d[a.label].to(dev).float())


def evaluate(ds):
    rs, cov_m, cov_s, cov_r = [], [], [], []
    with torch.no_grad():
        for d in ds:
            x, t, y = prep(d); s = model(x, t); n = y.numel(); B = min(a.B, n // 2)
            rk = lambda v: v.argsort().argsort().float()
            rs.append(torch.corrcoef(torch.stack([rk(s), rk(y)]))[0, 1].item())
            tot = y.clamp(min=0).sum().clamp(min=1e-9)
            cov_s.append(y.clamp(min=0)[s.topk(B).indices].sum().item() / tot.item())
            cov_m.append(y.clamp(min=0)[d["sum"][:, 1].to(dev).topk(B).indices].sum().item() / tot.item())   # mean attention rule
            cov_r.append(y.clamp(min=0)[-B:].sum().item() / tot.item())   # recent
    f = lambda v: sum(v) / len(v)
    return f(rs), f(cov_s), f(cov_m), f(cov_r)


for ep in range(a.epochs):
    model.train(); tot = 0.0; random.shuffle(train)
    for d in train:
        x, t, y = prep(d); n = y.numel()
        if n < 2 * a.block: continue
        s = model(x, t)
        i = torch.randint(0, n, (a.pairs,), device=dev); j = torch.randint(0, n, (a.pairs,), device=dev)
        keep = (i // a.block) != (j // a.block); i, j = i[keep], j[keep]
        dy = y[i] - y[j]; w = dy.abs(); w = w / w.mean().clamp(min=1e-9)
        loss = (w * F.softplus(-torch.sign(dy) * (s[i] - s[j]))).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    if ep % 5 == 4 or ep == a.epochs - 1:
        model.eval(); r, cs, cm, cr = evaluate(held)
        print(f"epoch {ep + 1}: loss {tot / max(len(train), 1):.4f} | held-out: spearman {r:.3f} | top-{a.B} label mass covered: selector {cs:.3f} mean-attention {cm:.3f} recent {cr:.3f}")
torch.save({"state": model.state_dict(), "mu": mu, "sd": sd, "vocab": a.vocab}, a.out); print("saved", a.out)
