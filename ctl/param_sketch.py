"""Parameter-movement tracking for the GRPO run (the user: measure how far and in which direction the weights move - circling
in place calls for a lower lr; directed movement should not be making things worse).
  python3 param_sketch.py sketch <ckpt> <out.pt>    count-sketch of the tuned weights (the 28 decoder layers + the hypernet)
                                                    into 2^20 buckets: norms and dot products of differences are preserved
                                                    (Johnson-Lindenstrauss), so any two versions can be compared from 8 MB
  python3 param_sketch.py report <v1.pt> <v2.pt> ...  in version order: step sizes, cosine of consecutive steps, distance from
                                                    the first, straightness = distance / path length (1 = straight line,
                                                    ~1/sqrt(k) = a random walk)"""
import sys, zlib, torch

D = 1 << 20


def sketch(path, out):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    parts = {"model": ck.get("ft") or {}, "hypernet": ck["pooler"]}
    sk = {}
    for name, sd in parts.items():
        s = torch.zeros(D, dtype=torch.float64)
        for k in sorted(sd):
            v = sd[k].detach().float().flatten()
            g = torch.Generator().manual_seed(zlib.crc32(k.encode()))
            idx = torch.randint(0, D, (v.numel(),), generator=g)
            sg = torch.randint(0, 2, (v.numel(),), generator=g).float() * 2 - 1
            s.index_add_(0, idx, (v * sg).double())
        sk[name] = s
    torch.save(sk, out)


def report(paths):
    S = [torch.load(p) for p in paths]
    for part in S[0]:
        X = [s[part] for s in S]
        steps = [X[i] - X[i - 1] for i in range(1, len(X))]
        path_len = 0.0
        out = []
        for i, st in enumerate(steps):
            n = st.norm().item(); path_len += n
            cos = (st @ steps[i - 1] / (n * steps[i - 1].norm() + 1e-30)).item() if i else float("nan")
            dist = (X[i + 1] - X[0]).norm().item()
            out.append(f"{paths[i + 1].split('/')[-1].replace('.pt', '')}: step {n:.3g}, cos with the previous step {cos:+.2f}, "
                       f"from the first {dist:.3g}, straightness {dist / max(path_len, 1e-30):.2f}")
        print(f"[{part}] " + " | ".join(out))


if __name__ == "__main__":
    if sys.argv[1] == "sketch": sketch(sys.argv[2], sys.argv[3])
    else: report(sys.argv[2:])
