# the fine-tuned bge-small as model.onnx + model_int8.onnx (the device path search.py's Embedder loads)
import sys, os, torch
from transformers import AutoModel
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
m = AutoModel.from_pretrained(src).eval()
ids = torch.ones(1, 8, dtype=torch.long); am = torch.ones(1, 8, dtype=torch.long); tt = torch.zeros(1, 8, dtype=torch.long)
class W(torch.nn.Module):
    def __init__(s): super().__init__(); s.m = m
    def forward(s, input_ids, attention_mask, token_type_ids): return s.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
torch.onnx.export(W(), (ids, am, tt), os.path.join(dst, "model.onnx"), input_names=["input_ids", "attention_mask", "token_type_ids"], output_names=["last_hidden_state"],
                  dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "token_type_ids": {0: "b", 1: "s"}, "last_hidden_state": {0: "b", 1: "s"}}, opset_version=17, dynamo=False)
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(os.path.join(dst, "model.onnx"), os.path.join(dst, "model_int8.onnx"), weight_type=QuantType.QInt8)
import shutil
for f in ("tokenizer.json", "tokenizer_config.json", "config.json", "vocab.txt", "special_tokens_map.json"):
    if os.path.abspath(src) != os.path.abspath(dst) and os.path.exists(os.path.join(src, f)): shutil.copy(os.path.join(src, f), dst)
print("EXPORT_DONE", dst, {f: os.path.getsize(os.path.join(dst, f)) // 1000000 for f in ("model.onnx", "model_int8.onnx")}, "MB")
