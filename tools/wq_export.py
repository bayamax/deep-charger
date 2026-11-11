"""Export a tools/wq.py result (wq_<method>.pt) as an MLX 4-bit model folder (mlx-lm loads it as is): the parametrisation is the
same (W = q * scale + bias per group of 64 along the input dimension), so nothing is re-quantised. Linears of the decoder layers:
"<name>.weight" uint32 (8 codes per word, lowest bits first), "<name>.scales", "<name>.biases"; the embedding (tied with the output
head) and the norms stay 16-bit (mlx-lm quantises only the modules that have scales in the file).
  python tools/wq_export.py wq_dwq.pt out_dir [--model Qwen/Qwen3-1.7B]
"""
import argparse, json, os, shutil
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
a = ap.parse_args()
base = snapshot_download(a.model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt", "LICENSE*"])
d = torch.load(a.src, map_location="cpu"); g, bits = d["group"], d["bits"]
assert bits == 4, "packing below is for 4-bit codes"
W = {}
for f in sorted(os.listdir(base)):
    if f.endswith(".safetensors"): W.update(load_file(os.path.join(base, f)))
W = {k: v.to(torch.float16) for k, v in W.items()}
n = 0
for k, v in d.items():
    if not isinstance(v, dict): continue
    p = v["q"]; q = torch.stack([p & 15, p >> 4], 2).view(p.size(0), -1).to(torch.int64)
    assert W[k + ".weight"].shape == q.shape, k
    q8 = q.view(q.size(0), -1, 8); word = torch.zeros(q8.shape[:2], dtype=torch.int64)
    for j in range(8): word |= q8[..., j] << (4 * j)
    W[k + ".weight"] = word.to(torch.uint32); W[k + ".scales"] = v["s"].to(torch.float16); W[k + ".biases"] = v["b"].to(torch.float16); n += 1
os.makedirs(a.out, exist_ok=True)
save_file(W, os.path.join(a.out, "model.safetensors"), metadata={"format": "mlx"})
cfg = json.load(open(os.path.join(base, "config.json")))
cfg["quantization"] = {"group_size": g, "bits": bits}; cfg["quantization_config"] = cfg["quantization"]; cfg["torch_dtype"] = "float16"
json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
for f in os.listdir(base):
    if f not in ("config.json", "model.safetensors.index.json") and not f.endswith(".safetensors"): shutil.copy(os.path.join(base, f), a.out)
shutil.copy(a.src, os.path.join(a.out, os.path.basename(a.src)))
print(f"[wq] exported {n} linears ({bits}-bit, group {g}) -> {a.out}")
