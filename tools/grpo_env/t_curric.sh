#!/bin/sh
# GRPO curriculum status, phone-width (same layout as the MUS-RUN t). Read-only.
echo "CURRIC $(date -u +%H:%M)Z"
R=$(pgrep -f "grpo_ep_curri[c]" | head -1)
[ -n "$R" ] && echo "run  ok (pid $R)" || echo "run  STOPPED (watchdog relaunches)"
L=$(grep "^\[step" /root/grpo_curric.log | tail -1)
echo "$L" | sed "s/ bridge=[0-9]*//;s/ |grad|=[0-9.]*//;s/ skip=[0-9\/]*//;s/ cache=[0-9]*//;s/ band=.*//" | cut -c1-78
echo "$L" | grep -o "lvl=[0-9] solve[0-9]*=[0-9.]*" | sed "s/lvl=/lvl /;s/ solve20=/  solve20 /;s/$/  (up 0.60 \/ dn 0.20)/"
python3 - <<'PYEOF' 2>/dev/null
import json, re
from collections import defaultdict
MO = re.compile(r"<\s*/?\s*more\s*/?\s*>", re.I)
rows = [json.loads(l) for l in open("/root/work/ep_curric/rollouts.jsonl") if l.strip()]
by = defaultdict(list)
for r in rows: by[r["step"]].append(r)
steps = sorted(by); mx = steps[-1]; mid = mx // 2
def win(lo, hi): return [r for s in steps if lo <= s <= hi for r in by[s]]
def rate(rs, k="correct"): return sum(1 for r in rs if r.get(k)) / max(len(rs), 1)
def more_line(tag, rs):
    t = [r for r in rs if MO.search(r.get("text") or "")]
    s = [r for r in rs if (r.get("more") or 0) > 0]
    return (f"{tag} tag {len(t)} ({100*len(t)/max(len(rs),1):.0f}%)  served {len(s)}  "
            f"correct {sum(1 for r in s if r.get('correct'))}")
A = win(1, mx); L = win(max(1, mx-24), mx)
print(f"all    {rate(A):5.1%}  gnd {rate(A,'grounded'):4.0%}  (n={len(A)})")
print(f"1st-half {rate(win(1,mid)):5.1%} / 2nd-half {rate(win(mid+1,mx)):5.1%}")
print(f"last25 {rate(L):5.1%}  gnd {rate(L,'grounded'):4.0%}   step {mx}/200")
print(more_line("moreAll", A))
print(more_line("more25 ", L))
PYEOF
echo "disk $(df -h / | tail -1 | awk "{print \$4}") free"
