#!/usr/bin/env python3
"""Can a dedicated reward model tell Deep Charger's better replies from its worse ones? (2026-10-14, the user: GRPO with a
judgment model's 0-1 score as the reward.) Before any training, the reward model scores the GRPO rollouts already on the
hub (the mtg7-mtg10 rounds: 12 rollouts per question, each with its correctness and, for correct search replies, the nano
judge's sound / natural / clean; reason items carry the teacher's solves_it / follows_the_request). Only the final reply
(after the last </think>) goes to the reward model, with the question as the user turn.

Reported per reward model, all within a question's group of 12 (what GRPO's advantages compare):
  - pairwise accuracy on correctness (search) and on solves_it (reason): of the pairs where one is right and one is not,
    how often the right one scores higher (0.5 = chance);
  - among correct search replies, pairwise accuracy on the nano judge's verdict (sound and natural and clean);
  - the groups where all 12 share one correctness (GRPO's binary reward gives them no signal): the reward model's spread;
  - the within-group Spearman correlation of the score with the reply's length (a length bias is what GRPO would exploit).

  python3 rm_scan.py --rollouts a.jsonl b.jsonl --models Skywork/Skywork-Reward-V2-Qwen3-0.6B ... --out rm_scan.json
"""
import argparse, itertools, json, math, os, random, time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--rollouts", nargs="+", required=True); ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--max-len", type=int, default=2048); ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--limit", type=int, default=0, help="groups per file (0: all) - for a quick test")
X = ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_rm(spec, dev):
    """<hub name>[@8bit|@4bit]: a quantised reward model (bitsandbytes) when asked, bf16 on the card otherwise"""
    name, _, q = spec.partition("@")
    kw = {"num_labels": 1}
    if q in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True) if q == "8bit" else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}; kw["torch_dtype"] = torch.bfloat16
        m = AutoModelForSequenceClassification.from_pretrained(name, **kw)
    else:
        m = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.bfloat16 if str(dev).startswith("cuda") else torch.float32, **kw).to(dev)
    return AutoTokenizer.from_pretrained(name), m.eval()


rows = []
for f in X.rollouts:
    tag = os.path.basename(os.path.dirname(f)) or os.path.basename(f)
    rs = [json.loads(l) for l in open(f) if l.strip()]
    groups = {}
    for r in rs: groups.setdefault((r["step"], r["q"]), []).append(r)
    keys = list(groups)[: X.limit] if X.limit else list(groups)
    for gk in keys:
        for r in groups[gk]:
            t = r.get("text") or ""; w = r.get("why") or {}
            reply = t.rsplit("</think>", 1)[1].strip() if "</think>" in t else ""
            if r["kind"] == "search": right = bool(w.get("correct"))
            else: right = bool(w.get("solves_it"))
            good = None
            if r["kind"] == "search" and right and w.get("sound") is not None:
                good = bool(w.get("sound")) and bool(w.get("natural")) and bool(w.get("clean"))
            rows.append({"g": f"{tag}|{gk[0]}|{gk[1][:200]}", "kind": r["kind"], "q": r["q"], "reply": reply, "right": right, "good": good,
                         "finished": bool(reply), "reward": r.get("reward"), "len": len(reply)})
print(f"[rm_scan] {len(rows)} rollouts in {len({r['g'] for r in rows})} groups; finished {sum(r['finished'] for r in rows)}; "
      f"search {sum(r['kind'] == 'search' for r in rows)}, reason {sum(r['kind'] != 'search' for r in rows)}; device {DEV}", flush=True)


def rank(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and xs[o[j + 1]] == xs[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3: return None
    ra, rb = rank(a), rank(b); ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb)); den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else None


def pairwise(sel, label):
    """within groups, over pairs that differ in label: share where the positive scores higher (ties count half)"""
    by = {}
    for r in sel: by.setdefault(r["g"], []).append(r)
    win = n = 0.0
    for g, rs in by.items():
        for a, b in itertools.combinations(rs, 2):
            if label(a) == label(b): continue
            p, q_ = (a, b) if label(a) else (b, a); n += 1
            win += 1.0 if p["s"] > q_["s"] else 0.5 if p["s"] == q_["s"] else 0.0
    return (round(win / n, 3) if n else None), int(n)


res = {"args": vars(X), "n_rows": len(rows), "models": {}}
for name in X.models:
    t0 = time.time()
    tok, model = load_rm(name, DEV)
    todo = [r for r in rows if r["finished"]]
    texts = [tok.apply_chat_template([{"role": "user", "content": r["q"]}, {"role": "assistant", "content": r["reply"]}], tokenize=False) for r in todo]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    scores = [0.0] * len(texts)
    with torch.no_grad():
        for b in range(0, len(order), X.batch):
            idx = order[b:b + X.batch]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=X.max_len, add_special_tokens=False).to(model.device)
            out = model(**enc).logits[:, 0].float().cpu().tolist()
            for i, s in zip(idx, out): scores[i] = s
            if (b // X.batch) % 100 == 0: print(f"[rm_scan] {name}: {b + len(idx)}/{len(order)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    for r in rows: r["s"] = -1e9
    for r, s in zip(todo, scores): r["s"] = s
    fin = [r for r in rows if r["finished"]]
    srch = [r for r in fin if r["kind"] == "search"]; reas = [r for r in fin if r["kind"] != "search"]
    m = {}
    m["search_correct_pairwise"], m["search_pairs"] = pairwise(srch, lambda r: r["right"])
    m["reason_solves_pairwise"], m["reason_pairs"] = pairwise(reas, lambda r: r["right"])
    m["judge_good_pairwise_among_correct"], m["judge_pairs"] = pairwise([r for r in srch if r["right"] and r["good"] is not None], lambda r: r["good"])
    m["finished_vs_unfinished_pairwise"], _ = pairwise(rows, lambda r: r["finished"])
    by = {}
    for r in fin: by.setdefault(r["g"], []).append(r)
    same = [rs for rs in by.values() if len(rs) >= 4 and len({r["right"] for r in rs}) == 1]
    m["groups_one_correctness"] = len(same); m["groups_total"] = len(by)
    sp = [math.sqrt(sum((r["s"] - sum(x["s"] for x in rs) / len(rs)) ** 2 for r in rs) / len(rs)) for rs in same]
    allsd = [math.sqrt(sum((r["s"] - sum(x["s"] for x in rs) / len(rs)) ** 2 for r in rs) / len(rs)) for rs in by.values() if len(rs) >= 4]
    m["score_sd_within_same_groups"] = round(sum(sp) / len(sp), 3) if sp else None; m["score_sd_within_all_groups"] = round(sum(allsd) / len(allsd), 3) if allsd else None
    ls = [spearman([r["s"] for r in rs], [r["len"] for r in rs]) for rs in by.values() if len(rs) >= 4]
    ls = [v for v in ls if v is not None]; m["len_spearman_within_groups"] = round(sum(ls) / len(ls), 3) if ls else None
    m["score_mean"] = round(sum(scores) / len(scores), 3); m["minutes"] = round((time.time() - t0) / 60, 1)
    res["models"][name] = m; json.dump(res, open(X.out, "w"), indent=1)
    print(f"[rm_scan] {name}: " + "  ".join(f"{k} {v}" for k, v in m.items()), flush=True)
    del model; torch.cuda.empty_cache() if DEV == "cuda" else None
print("RM_SCAN_DONE", flush=True)
