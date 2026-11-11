"""Learned recall for the fixed KV (the user: let the model decide when to recall, with training if needed - not rules).
Data: tools/sp_qwen3.py rollout --kv-lmk --lmk-dump (at every check of the bounded rollout: the hidden state, the stored lines
that are out of the cache, the landmark mass; afterwards the same dialogue read with full context gives the attention the
query at that check pays to every stored line). No hand labels: the model's own full-context attention is the teacher.
  what - a head maps the hidden state into the bge space; trained so that its cosine over the out-of-cache lines matches the
         full-context attention over them (soft cross-entropy). Replaces the query rule (own last 3 lines + the user's '?' line).
  when - a detector from the hidden state and the landmark mass predicts whether full context would put at least --need of
         its attention on lines that are out of the cache. Replaces the fixed threshold.
  python tools/lmk_train.py lmk_train.pt ctl/mt_train.jsonl out.pt [--need 0.05]
Writes out.pt: TorchScript query head ("q"), detector ("det"), its threshold ("thr") and the hidden layers ("hl").
"""
import argparse, io, json, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("dump"); ap.add_argument("rows"); ap.add_argument("out")
ap.add_argument("--need", type=float, default=0.05, help="detector label: full-context attention on out-of-cache lines at least this")
ap.add_argument("--hl", default="13,19"); ap.add_argument("--hidden", type=int, default=1024)
ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--min-mass", type=float, default=0.005, help="query head: only checks whose full-context mass on out-of-cache lines is at least this")
ap.add_argument("--val", type=float, default=0.15); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)

D = torch.load(a.dump, map_location="cpu", weights_only=False)
rows = [json.loads(l) for l in open(a.rows)]
dl = sorted({d["j"] for d in D}); random.shuffle(dl); nv = max(1, int(len(dl) * a.val)); VAL = set(dl[:nv])

Q = {"tr": [], "va": []}; T = {"tr": [], "va": []}   # query items (h, cand bge (n, 384), target (n,), gold index or -1, task); detector items (h, out, y, task)
for d in D:
    sp = "va" if d["j"] in VAL else "tr"; r = rows[d["j"]]; gold = str(r.get("gold", ""))
    gl = [i for i, t in enumerate(d["lines"]) if gold and gold in t and r.get("task", "").startswith("doc")]
    for c, f in zip(d["checks"], d["full"]):
        cand = c["cand"]; fc = f[cand].float() if cand else torch.zeros(0)
        T[sp].append((c["h"].float(), c["out"], float(fc.sum()), r.get("task", "?")))
        if cand and d["semb"] is not None and float(fc.sum()) >= a.min_mass:
            gi = next((k for k, i in enumerate(cand) if i in gl), -1)
            Q[sp].append((c["h"].float(), d["semb"][cand].float(), fc / fc.sum(), gi, r.get("task", "?")))
print(f"dialogues {len(dl)} (val {nv}) | detector items tr {len(T['tr'])} va {len(T['va'])} | query items tr {len(Q['tr'])} va {len(Q['va'])}")
H = torch.stack([x[0] for x in T["tr"]]); MU = H.mean(0); SD = H.std(0).clamp(min=1e-3); DIM = H.size(1)


class QHead(nn.Module):
    def __init__(self, mu, sd, hid):
        super().__init__(); self.register_buffer("mu", mu); self.register_buffer("sd", sd)
        self.net = nn.Sequential(nn.Linear(mu.numel(), hid), nn.GELU(), nn.Linear(hid, 384)) if hid else nn.Linear(mu.numel(), 384)
    def forward(self, h): return self.net((h - self.mu) / self.sd)


class Det(nn.Module):
    def __init__(self, mu, sd, hid):
        super().__init__(); self.register_buffer("mu", mu); self.register_buffer("sd", sd)
        self.net = nn.Sequential(nn.Linear(mu.numel() + 1, hid), nn.GELU(), nn.Linear(hid, 1)) if hid else nn.Linear(mu.numel() + 1, 1)
    def forward(self, h, out):
        return torch.sigmoid(self.net(torch.cat([(h - self.mu) / self.sd, torch.log(out.clamp(min=1e-5)) / 3.0], 1)))[:, 0]


def q_metrics(m, items, tau):
    m.eval(); top1 = hit3 = n = gh = gn = 0; lm = 0.0; base_g = 0
    with torch.no_grad():
        for h, E, t, gi, _ in items:
            s = F.normalize(m(h[None])[0], dim=-1) @ F.normalize(E, dim=-1).T; k = min(3, len(t))
            tk = s.topk(k).indices.tolist(); n += 1; top1 += int(tk[0] == int(t.argmax())); hit3 += int(int(t.argmax()) in tk)
            lm += float(-(t * F.log_softmax(s / tau, 0)).sum())
            if gi >= 0: gn += 1; gh += int(gi in tk); base_g += int(gi in t.topk(k).indices.tolist())
    return dict(n=n, top1=top1 / max(n, 1), hit3=hit3 / max(n, 1), ce=lm / max(n, 1), gold3=f"{gh}/{gn}", teacher_gold3=f"{base_g}/{gn}")


qm = QHead(MU, SD, a.hidden); ltau = nn.Parameter(torch.tensor(math.log(0.05)))
opt = torch.optim.AdamW(list(qm.parameters()) + [ltau], lr=a.lr, weight_decay=a.wd)
for ep in range(a.epochs):
    qm.train(); random.shuffle(Q["tr"]); tot = 0.0
    for i in range(0, len(Q["tr"]), 64):
        b = Q["tr"][i:i + 64]; qv = F.normalize(qm(torch.stack([x[0] for x in b])), dim=-1); loss = 0.0
        for k, (_, E, t, _, _) in enumerate(b):
            loss = loss - (t * F.log_softmax((F.normalize(E, dim=-1) @ qv[k]) / ltau.exp(), 0)).sum()
        loss = loss / len(b); opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)
    if ep % 10 == 9 or ep == a.epochs - 1:
        print(f"query ep {ep + 1}: train ce {tot / max(len(Q['tr']), 1):.3f} tau {ltau.exp().item():.3f} | val {q_metrics(qm, Q['va'], ltau.exp().item())}")

y_tr = torch.tensor([float(x[2] >= a.need) for x in T["tr"]]); y_va = torch.tensor([float(x[2] >= a.need) for x in T["va"]])
o_tr = torch.tensor([[x[1]] for x in T["tr"]]); o_va = torch.tensor([[x[1]] for x in T["va"]])
H_va = torch.stack([x[0] for x in T["va"]]) if T["va"] else torch.zeros(0, DIM)


def auc(p, y):
    pos = p[y > 0.5]; neg = p[y < 0.5]
    if not len(pos) or not len(neg): return float("nan")
    return float((pos[:, None] > neg[None]).float().mean() + 0.5 * (pos[:, None] == neg[None]).float().mean())


print(f"detector: need = full-context mass on out-of-cache lines >= {a.need}: positives tr {int(y_tr.sum())}/{len(y_tr)} va {int(y_va.sum())}/{len(y_va)}"
      f" | landmark mass alone, val AUC {auc(o_va[:, 0], y_va):.3f}")
dm = Det(MU, SD, 256); dopt = torch.optim.AdamW(dm.parameters(), lr=a.lr, weight_decay=a.wd)
pw = ((1 - y_tr).sum() / y_tr.sum().clamp(min=1)).clamp(max=20)
for ep in range(a.epochs):
    dm.train(); perm = torch.randperm(len(y_tr))
    for i in range(0, len(perm), 128):
        ix = perm[i:i + 128]; p = dm(H[ix], o_tr[ix]).clamp(1e-6, 1 - 1e-6)
        loss = -(pw * y_tr[ix] * p.log() + (1 - y_tr[ix]) * (1 - p).log()).mean(); dopt.zero_grad(); loss.backward(); dopt.step()
    if ep % 10 == 9 or ep == a.epochs - 1:
        dm.eval()
        with torch.no_grad(): pv = dm(H_va, o_va)
        print(f"detector ep {ep + 1}: val AUC {auc(pv, y_va):.3f}")
dm.eval()
with torch.no_grad(): pv = dm(H_va, o_va)
best = (0.0, 0.5)   # threshold: best F1 on the held-out dialogues
for th in [x / 100 for x in range(5, 96)]:
    f_ = (pv > th).float(); tp = float((f_ * y_va).sum()); prec = tp / max(float(f_.sum()), 1); rec = tp / max(float(y_va.sum()), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9); best = max(best, (f1, th))
f_ = (pv > best[1]).float(); tasks = [x[3] for x in T["va"]]
print(f"detector threshold {best[1]:.2f} (val F1 {best[0]:.3f}); fires per task on val: " + " ".join(
    f"{t} {int(sum(f_[i] for i in range(len(tasks)) if tasks[i] == t))}/{sum(1 for x in tasks if x == t)} (need {int(sum(y_va[i] for i in range(len(tasks)) if tasks[i] == t))})" for t in sorted(set(tasks))))


def ts(m):
    b = io.BytesIO(); torch.jit.save(torch.jit.script(m), b); return b.getvalue()


torch.save({"q": ts(qm), "det": ts(dm), "thr": best[1], "hl": a.hl, "tau": ltau.exp().item()}, a.out)
print("saved", a.out)
