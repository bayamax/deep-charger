"""grpo_ep_curric.py = grpo_ep_more.py with the question sampler replaced by an adaptive
curriculum. Reward, parser, environment and optimizer are untouched. Asserts on each edit."""
import ast
S=open("/root/work/grpo_ep_more.py").read()
def sub(old,new,tag):
    global S
    assert old in S, "MISSING: "+tag
    S=S.replace(old,new,1)
sub('print(f"[data] {len(pool)} questions", flush=True)',
'''print(f"[data] {len(pool)} questions", flush=True)

# ---------------- adaptive curriculum ------------------------------------------------------
# score = deepest 256-token slice at which gold appears on the exemplar pages (0 = head).
# The band served rises when the policy solves >= CURR_UP of the last CURR_WINDOW groups
# on it, and falls when it solves <= CURR_DOWN. Easy (score 0) questions are mixed in at
# CURR_EASY_MIX so the head-chunk skill is not forgotten; they do not move the band.
CURR_F = os.environ.get("GRPO_CURRICULUM", os.path.join(HERE, "curriculum.jsonl"))
EASY_MIX = float(os.environ.get("CURR_EASY_MIX", "0.2"))
MAX_BAND = int(os.environ.get("CURR_MAX_BAND", "4"))
START_BAND = int(os.environ.get("CURR_START_BAND", "1"))
WIN = int(os.environ.get("CURR_WINDOW", "20"))
UP = float(os.environ.get("CURR_UP", "0.6"))
DOWN = float(os.environ.get("CURR_DOWN", "0.2"))
bands = {}
for _line in open(CURR_F):
    _r = json.loads(_line)
    bands.setdefault(min(int(_r["score"]), 5), []).append({"q": _r["q"], "gold": _r["gold"]})
crng = random.Random(1)
for _v in bands.values():
    crng.shuffle(_v)
CUR = {"band": START_BAND, "hist": [], "rate": 0.0, "last_band": 0, "ptr": {}}


def curric_pick(step):
    b = 0 if (EASY_MIX > 0 and bands.get(0) and crng.random() < EASY_MIX) else CUR["band"]
    while b > 0 and not bands.get(b):
        b -= 1
    lst = bands[b]
    i = CUR["ptr"].get(b, 0)
    CUR["ptr"][b] = i + 1
    CUR["last_band"] = b
    return lst[i % len(lst)]


def curric_update(corr):
    if CUR["last_band"] == 0:
        return
    CUR["hist"] = (CUR["hist"] + [1.0 if corr > 0 else 0.0])[-WIN:]
    CUR["rate"] = sum(CUR["hist"]) / len(CUR["hist"])
    if len(CUR["hist"]) < WIN:
        return
    if CUR["rate"] >= UP and CUR["band"] < MAX_BAND:
        CUR["band"] += 1
        CUR["hist"] = []
    elif CUR["rate"] <= DOWN and CUR["band"] > 1:
        CUR["band"] -= 1
        CUR["hist"] = []


print("[curriculum] bands " + ", ".join(f"{k}:{len(v)}" for k, v in sorted(bands.items()))
      + f"  start={START_BAND} max={MAX_BAND} easy_mix={EASY_MIX} window={WIN} up={UP} down={DOWN}",
      flush=True)''', "curriculum block")
sub('        item = pool[(step * B + b) % len(pool)]',
    '        item = curric_pick(step)', "sampler")
sub('    corr = sum(1 for i in step_info if i[3]) / n\n',
    '    corr = sum(1 for i in step_info if i[3]) / n\n    curric_update(corr)\n', "update hook")
sub('"correct": inf[3], "trusted": inf[4], "more": inf[5],',
    '"correct": inf[3], "trusted": inf[4], "more": inf[5], "band": CUR["last_band"],', "record")
sub('            f"skip={skipped}/{B} cache={len(cache)} "',
    '            f"skip={skipped}/{B} cache={len(cache)} "\n            f"band={CUR[\'last_band\']} lvl={CUR[\'band\']} solve{WIN}={CUR[\'rate\']:.2f} "', "log line")
ast.parse(S)
open("/root/work/grpo_ep_curric.py","w").write(S)
print("PATCH OK ->", len(S.splitlines()), "lines")
