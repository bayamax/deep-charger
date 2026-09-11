#!/usr/bin/env python3
"""4-bit affine fake quantization: the arithmetic the phone actually runs, in torch.

The deployed model is produced on the Mac by

    mlx_lm.convert --hf-path <merged> --mlx-path <out> -q --q-bits 4 --q-group-size 64

and the weight index of the resulting directory says exactly what that touched: scales/biases
exist for the seven projections of every block plus embed_tokens and lm_head, and for nothing
else.  The norms and the attention biases stay in fp16, and so does the pooler, which the app
loads separately and keeps in float32.  Scales and biases are stored fp16, so this rounds them
to fp16 too.

quant_dequant is a port of affine_quantize() in mlx/ops.cpp at the version the app pins: the
group's larger-magnitude edge is made exactly representable (q0 = round(edge/scale), then scale
is redefined as edge/q0), the opposite edge becomes the bias, and the grid is
    w ~= q * scale + bias,  q in 0..15.
The only knowing difference is the tie-break of round(): C++ rounds a half away from zero and
torch rounds it to even.  That can only bite a value landing exactly on .5, and it moves such a
value by one step of a 16-level grid.

  python3 q4.py                 # self-check on random and on a real weight-like tensor
  python3 q4.py --hf <dir>      # per-module error report for a model directory
"""
import torch
import torch.nn as nn

# module suffixes that carry scales in the deployed directory
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj",
           "lm_head", "embed_tokens")


def affine_params(w, group=64, bits=4, store_dtype=torch.float16):
    """The three things the file stores: integer codes, per-group scale, per-group bias.

    Returns q [.., n_groups, group] as float, and scales/biases [.., n_groups, 1], such that
    q * scale + bias is what the runtime computes.
    """
    if w.shape[-1] % group:
        raise ValueError(f"last dim {w.shape[-1]} is not a multiple of the group size {group}")
    x = w.float().reshape(-1, w.shape[-1] // group, group)
    n_bins = float((1 << bits) - 1)
    w_max = x.amax(-1, keepdim=True)
    w_min = x.amin(-1, keepdim=True)
    mask = w_min.abs() > w_max.abs()                      # which edge is further from zero
    scales = torch.clamp((w_max - w_min) / n_bins, min=1e-7)
    scales = torch.where(mask, scales, -scales)
    edge = torch.where(mask, w_min, w_max)
    q0 = torch.round(edge / scales)
    scales = torch.where(q0 != 0, edge / q0, scales)
    biases = torch.where(q0 == 0, torch.zeros_like(edge), edge)
    # the file keeps these in fp16, so the runtime dequantizes with the rounded values
    scales = scales.to(store_dtype).float()
    biases = biases.to(store_dtype).float()
    q = torch.clamp(torch.round((x - biases) / scales), 0.0, n_bins)
    return q, scales, biases


def quant_dequant(w, group=64, bits=4, store_dtype=torch.float16, ste=False):
    """Round w onto the 4-bit affine grid and back. Shape preserved, dtype preserved."""
    q, scales, biases = affine_params(w, group, bits, store_dtype)
    deq = (q * scales + biases).reshape(w.shape).to(w.dtype)
    return w + (deq - w).detach() if ste else deq


def _leaf(name):
    """peft renames a wrapped Linear's weight holder to '<target>.base_layer'."""
    parts = [p for p in name.split(".") if p != "base_layer"]
    return parts[-1] if parts else ""


def quantizable(model):
    """[(name, module)] for every module the deployed conversion would have quantized, once each."""
    out, seen = [], set()
    for name, mod in model.named_modules():
        if not isinstance(mod, (nn.Linear, nn.Embedding)):
            continue
        if _leaf(name) not in TARGETS:
            continue
        w = getattr(mod, "weight", None)
        if w is None or w.shape[-1] % 64:
            continue
        if w.data_ptr() in seen:          # tied lm_head / embed_tokens: one grid, not two
            continue
        seen.add(w.data_ptr())
        out.append((name, mod))
    return out


def check_adapters_idle(model):
    """A LoRA that is mounted but untrained must be zero, or the base weight is not the whole weight."""
    bad = []
    for name, p in model.named_parameters():
        if "lora_B" in name and p.detach().abs().max().item() > 0:
            bad.append(name)
    return bad


@torch.no_grad()
def quantize_model(model, group=64, bits=4, store_dtype=torch.float16, verbose=True, chunk=1 << 24):
    """Replace every deployed-quantized weight by its dequantized 4-bit value, in place.

    The arithmetic runs on the CPU in row blocks. Doing it on the GPU costs several float32
    temporaries of the output head at once, and the allocator keeps those blocks reserved
    afterwards - enough to stop a third evaluator process from fitting on the card.
    """
    bad = check_adapters_idle(model)
    if bad:
        raise RuntimeError(f"{len(bad)} lora_B are non-zero (e.g. {bad[0]}); merge before quantizing")
    mods, tot_n, tot_se, tot_ss = quantizable(model), 0, 0.0, 0.0
    worst = []
    for name, mod in mods:
        w = mod.weight
        rows = max(1, chunk // w.shape[-1])
        se = ss = 0.0
        for i in range(0, w.shape[0], rows):
            blk = w.data[i:i + rows].to("cpu", torch.float32)
            q = quant_dequant(blk, group, bits, store_dtype)
            se += (q - blk).pow(2).sum().item(); ss += blk.pow(2).sum().item()
            w.data[i:i + rows].copy_(q.to(w.dtype))
        rel = (se / max(ss, 1e-30)) ** 0.5
        worst.append((rel, name, tuple(w.shape)))
        tot_n += w.numel(); tot_se += se; tot_ss += ss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    worst.sort(reverse=True)
    if verbose:
        print(f"[q4] {len(mods)} modules, {tot_n/1e6:.1f}M weights, group={group} bits={bits} "
              f"store={store_dtype}", flush=True)
        print(f"[q4] overall relative error {100*(tot_se/max(tot_ss,1e-30))**0.5:.2f}%", flush=True)
        for rel, name, shape in worst[:5]:
            print(f"[q4]   worst {100*rel:5.2f}%  {name} {shape}", flush=True)
    return {"modules": len(mods), "weights": tot_n,
            "rel_err": (tot_se / max(tot_ss, 1e-30)) ** 0.5}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="", help="model directory to report on")
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--bits", type=int, default=4)
    A = ap.parse_args()
    if not A.hf:
        torch.manual_seed(0)
        for tag, w in [("uniform", torch.rand(256, 640) * 2 - 1),
                       ("normal", torch.randn(256, 640) * 0.02),
                       ("outlier", torch.cat([torch.randn(256, 639) * 0.02,
                                              torch.randn(256, 1) * 2], 1))]:
            q = quant_dequant(w, A.group, A.bits)
            rel = (q - w).pow(2).sum().sqrt() / w.pow(2).sum().sqrt()
            lv = len(torch.unique(q[0, :A.group]))
            print(f"{tag:8s} rel {100*rel:.2f}%  levels in first group {lv} (<= {(1<<A.bits)})")
        # What the scheme guarantees is that the group's larger-magnitude edge lands on the grid
        # exactly (bias = edge) and that zero does too (scale is redefined so edge/scale is an
        # integer). It does NOT guarantee a second pass is a no-op: re-quantizing re-derives the
        # grid from the new extremes, so values move again.
        w = (torch.randn(64, 128) * 0.02).reshape(-1, 2, 64)
        q = quant_dequant(w.reshape(64, 128)).reshape(-1, 2, 64)
        far = w.abs().argmax(-1, keepdim=True)
        de = (w.gather(-1, far) - q.gather(-1, far)).abs().max()
        print(f"edge preserved: max|w_edge - q(w)_edge| = {de:.3e}  (fp16 rounding of the stored bias)")
    else:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(A.hf, torch_dtype=torch.float32)
        quantize_model(m, A.group, A.bits)
