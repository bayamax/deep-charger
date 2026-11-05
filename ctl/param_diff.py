"""Exact relative movement between two checkpoints: python3 param_diff.py <a.pt> <b.pt>
per part (the 28 tuned layers, the hypernet): ||b - a|| / ||a||, and per layer for the model (attention / MLP)."""
import sys, torch
A = torch.load(sys.argv[1], map_location="cpu", weights_only=False); B = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
def rel(da, db, keys):
    n = d = 0.0
    for k in keys: x = da[k].float(); y = db[k].float(); n += (x * x).sum().item(); d += ((y - x) ** 2).sum().item()
    return (d ** 0.5) / max(n ** 0.5, 1e-30), d ** 0.5
fa, fb = A.get("ft") or {}, B.get("ft") or {}; keys = sorted(set(fa) & set(fb))
r, a = rel(fa, fb, keys); print(f"model (28 layers, {len(keys)} tensors): relative change {r:.2e} (abs {a:.3g})")
pa, pb = A["pooler"], B["pooler"]; pk = sorted(set(pa) & set(pb))
r, a = rel(pa, pb, pk); print(f"hypernet ({len(pk)} tensors): relative change {r:.2e} (abs {a:.3g})")
for part in ["kvhead", "layers", "q", "out"]:
    ks = [k for k in pk if part in k]
    if ks: r, a = rel(pa, pb, ks); print(f"  hypernet[{part}] ({len(ks)}): relative change {r:.2e}")
for li in [0, 7, 14, 21, 27]:
    for part in ["self_attn", "mlp"]:
        ks = [k for k in keys if f"layers.{li}.{part}" in k]
        if ks: r, _ = rel(fa, fb, ks); print(f"  model layer {li} {part}: relative change {r:.2e}")
