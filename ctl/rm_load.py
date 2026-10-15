"""Loading a reward model for the GRPO scripts (2026-10-15). <hub name>[@8bit|@4bit]; a 4-bit model saved under
/root/rm_q/<org>__<name> (rm_quant.py: the bf16 download is dropped once it is saved quantised - the box's disk is small)
is loaded from there."""
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
QDIR = "/root/rm_q"


def local_dir(name): return os.path.join(QDIR, name.replace("/", "__"))


def fix_pad(tok, m):
    """a batch needs a pad token in the tokenizer and the model's config (the Tulu-3 reward model has none)"""
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if m.config.pad_token_id is None: m.config.pad_token_id = tok.pad_token_id
    return tok, m


def load_rm(spec, dev="cuda"):
    return fix_pad(*_load(spec, dev))


def _load(spec, dev="cuda"):
    name, _, q = spec.partition("@")
    kw = {"num_labels": 1}
    if q == "4bit" and os.path.exists(os.path.join(local_dir(name), "config.json")):
        m = AutoModelForSequenceClassification.from_pretrained(local_dir(name), device_map={"": 0}, torch_dtype=torch.bfloat16, **kw)
        return AutoTokenizer.from_pretrained(local_dir(name)), m.eval()
    if q in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True) if q == "8bit" else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}; kw["torch_dtype"] = torch.bfloat16
        m = AutoModelForSequenceClassification.from_pretrained(name, **kw)
    else:
        m = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.bfloat16 if str(dev).startswith("cuda") else torch.float32, **kw).to(dev)
    return AutoTokenizer.from_pretrained(name), m.eval()
