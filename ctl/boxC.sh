# box C: both evals finished -> failure breakdown of the compressed run (RW=768) and of the plain run, plus the pairing
cd /root/work
grep EVAL_DONE /root/pooleval_all_c.log /root/pooleval_all_nc.log
echo "=== ALL_C"; python3 /root/work/analyze_pool.py /root/work/teacher600.jsonl /root/work/pooleval_all_c.jsonl 768 2>&1 | grep -v Warning | grep -v "think tail"
echo "=== ALL_NC"; python3 /root/work/analyze_pool.py /root/work/teacher600.jsonl /root/work/pooleval_all_nc.jsonl 8000 2>&1 | grep -v Warning | grep -v "think tail\|^--- \|^    gold"
python3 - <<'PY'
import json
def rows(p):
    out=[]
    for l in open(p):
        try: out.append(json.loads(l))
        except Exception: pass
    return out
c={r["q"]:r for r in rows("/root/work/pooleval_all_c.jsonl")}; n={r["q"]:r for r in rows("/root/work/pooleval_all_nc.jsonl")}
common=[q for q in c if q in n]
b=sum(c[q]["correct"] and n[q]["correct"] for q in common); co=sum(c[q]["correct"] and not n[q]["correct"] for q in common); no=sum(n[q]["correct"] and not c[q]["correct"] for q in common)
print(f"=== PAIR c-vs-nc n={len(common)} both={b} c-only={co} nc-only={no}")
print(f"mean searches c={sum(c[q]['ns'] for q in common)/len(common):.2f} nc={sum(n[q]['ns'] for q in common)/len(common):.2f} | grounded c={sum(c[q]['grounded'] for q in common)} nc={sum(n[q]['grounded'] for q in common)} | landed c={sum(c[q]['landed'] for q in common)} nc={sum(n[q]['landed'] for q in common)}")
PY
echo "FINAL_ANALYSIS_DONE $(date -u)"
