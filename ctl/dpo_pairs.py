"""DPO pairs from per-problem draws: python3 dpo_pairs.py <out.jsonl> <raw files...> -- <memory files...>
A problem gives up to two pairs: the raw window's right-and-closed draw over a failed memory draw (unclosed or wrong), and the
memory's right-and-closed draw over a failed raw-window draw. Prints one summary line."""
import json, sys
sys.path.insert(0, '.')
from sp_mathcheck import correct

out, rest = sys.argv[1], sys.argv[2:]
k_ = rest.index('--'); rawf, memf = rest[:k_], rest[k_ + 1:]


def load(fs):
    d = {}
    for f in fs:
        for l in open(f):
            r = json.loads(l); d.setdefault(r['id'], []).append(r)
    return d


raw, mem = load(rawf), load(memf)
ok = lambda r: '</think>' in r['text'] and correct(r)
pairs = []
for k in raw:
    if k not in mem: continue
    rg, rb = [r for r in raw[k] if ok(r)], [r for r in raw[k] if not ok(r)]
    mg, mb = [r for r in mem[k] if ok(r)], [r for r in mem[k] if not ok(r)]
    if rg and mb:
        pairs.append({'id': k, 'prompt': rg[0]['prompt'], 'chosen': rg[0]['text'], 'rejected': mb[0]['text'],
                      'why': 'unclosed' if '</think>' not in mb[0]['text'] else 'wrong'})
    if mg and rb:
        pairs.append({'id': k, 'prompt': mg[0]['prompt'], 'chosen': mg[0]['text'], 'rejected': rb[0]['text'], 'why': 'memory-only'})
both = [k for k in raw if k in mem]
print(f"problems {len(both)}: raw right (any draw) {sum(any(map(ok, raw[k])) for k in both)}, memory right (any draw) "
      f"{sum(any(map(ok, mem[k])) for k in both)}, memory first draw right {sum(ok(mem[k][0]) for k in both)} closed "
      f"{sum('</think>' in mem[k][0]['text'] for k in both)} | pairs {len(pairs)} (raw over memory unclosed "
      f"{sum(p['why'] == 'unclosed' for p in pairs)} / wrong {sum(p['why'] == 'wrong' for p in pairs)}; memory over raw "
      f"{sum(p['why'] == 'memory-only' for p in pairs)})")
open(out, 'w').writelines(json.dumps(p) + '\n' for p in pairs)
