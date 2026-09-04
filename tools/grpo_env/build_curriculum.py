import json, os, re
from collections import Counter
H="/root/work"; OUT=os.path.join(H,"curriculum.jsonl")
SR=re.compile(r"<search>(.*?)</\s*search", re.S)
def norm(s): return re.sub(r"[^a-z0-9 ]"," ",(s or "").lower()).strip()
def has(t,g): return g and (" "+norm(g)+" ") in (" "+norm(t)+" ")
pages={}
for l in open(os.path.join(H,"reader_data.jsonl"), errors="replace"):
    try: d=json.loads(l)
    except Exception: continue
    if d.get("page"): pages[(d["q"],d["kw"])]=d["page"]
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(os.path.join(H,"bf16_mus_pure"))
STEP=256
def first_gold_slice(pg,g):
    ids=tok.encode(pg, add_special_tokens=False)
    for k in range(0, min(len(ids), STEP*40), STEP):
        if has(tok.decode(ids[k:k+STEP]), g): return k//STEP
    return None
n=0; dist=Counter()
with open(OUT,"w") as fh:
    for l in open(os.path.join(H,"corpus_box_final.jsonl")):
        try: r=json.loads(l)
        except Exception: continue
        q,g=(r.get("q") or "").strip(),(r.get("gold") or "").strip()
        if not q or not g or len(g.split())>6: continue
        seq=[]
        for b in SR.findall(r.get("traj") or ""):
            b=b.strip(); kw=(b.split("||",1)[0] if "||" in b else b).strip()
            if kw: seq.append(kw)
        pgs=[pages.get((q,kw)) for kw in seq]
        if not seq or not all(pgs): continue
        pos=[p for p in (first_gold_slice(pg,g) for pg in pgs) if p is not None]
        if not pos: continue
        s=max(pos); n+=1; dist[min(s,5)]+=1
        fh.write(json.dumps({"q":q,"gold":g,"score":s,"srch":r.get("srch"),"hops":len(seq)},ensure_ascii=False)+"\n")
print(f"[curriculum] wrote {n} scored questions -> {OUT}; score dist {sorted(dist.items())}")
