# usage: python sp_oracle.py TRACES MODEL N_TRACES S1,S2,...  (GPU)
"""Window-0 ceiling for a copying pooler: [prompt | the embeddings of the last S answer tokens] -> token t, KL vs the
full-context model. Variants: raw embeddings, and rescaled to one fixed norm (what the pooler's output normalisation gives)."""
import copy, json, sys, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
path, M, NTR, S = sys.argv[1], sys.argv[2], int(sys.argv[3]), [int(x) for x in sys.argv[4].split(",")]
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(M); llm = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16).cuda().eval()
emb = llm.get_input_embeddings()
tn = emb.weight.norm(dim=-1).mean().item()
rows = [json.loads(l) for l in open(path)][:NTR]
acc = {}
for r in rows:
    p = tok(r["prompt"], return_tensors="pt").input_ids[0].cuda(); a_ = tok(r["text"], add_special_tokens=False, return_tensors="pt").input_ids[0][:1100].cuda()
    ids = torch.cat([p, a_]); P = len(p)
    lp_t = F.log_softmax(llm(ids[None]).logits[0].float(), -1)          # row i predicts ids[i+1]
    pc = DynamicCache(); llm(ids[None, :P], past_key_values=pc, use_cache=True)
    ts = list(range(200, len(a_), 60))
    for s in S:
        for mode in ("raw", "norm"):
            for t in ts:
                e = emb(a_[t - s:t])[None]
                if mode == "norm": e = e / e.norm(dim=-1, keepdim=True) * tn
                c = copy.deepcopy(pc)
                pos = torch.arange(P, P + s, device="cuda")[None]
                h = llm(inputs_embeds=e, past_key_values=c, position_ids=pos, cache_position=pos[0], use_cache=True).logits[0, -1].float()
                lq = F.log_softmax(h, -1); lt = lp_t[P + t - 1]
                kl = (lt.exp() * (lt - lq)).sum().item()
                k = f"S{s} {mode}"; acc.setdefault(k, []).append(kl)
    print({k: round(sum(v) / len(v), 3) for k, v in acc.items()}, flush=True)
