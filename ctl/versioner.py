#!/usr/bin/env python3
"""Deep Charger's model versions on the hub, the way software is versioned (the user, 2026-10-09: too many models to tell
apart). MAJOR.MINOR.PATCH:
  MAJOR - the base weights change (a new base model or a new quantisation of it): earlier adapters do not carry over
  MINOR - an adopted training step on the same base (a GRPO/SFT adapter that beat the previous version on the screens)
  PATCH - the same model repackaged (quantised, merged, a pooler swapped) with no training of its own
  X.Y.Z-mtgN.sS - a candidate: round mtgN's copy at step S, not (yet) adopted; the adopted one becomes X.Y.Z
Every version lives under versions/<version>/ (copied server side, nothing re-uploaded) with version.json (what it is, its
parent, its base, its screens); versions/INDEX.md lists them all. The old paths stay as they were (the app and the boxes
read them). Run once to lay down the history, then --watch on box G2 follows the GRPO rounds: each copy becomes a candidate
version, each accepted copy the next minor.
  python3 versioner.py [--watch]
"""
import argparse, json, os, re, sys, time
from huggingface_hub import HfApi, CommitOperationCopy, CommitOperationAdd, hf_hub_download
ap = argparse.ArgumentParser(); ap.add_argument("--watch", action="store_true"); ap.add_argument("--repo", default="baya1116/hypernet-sp-distill")
A = ap.parse_args()
TOKEN = open("/root/.hf_token").read().strip(); api = HfApi(token=TOKEN); R = A.repo
STATE = "/root/version_state.json"; V = "versions"
MT = "pooler_distill/chatsft/multiturn"
S100_SCREENS = "single 160/300, br3 43/31, Dolphin 56 (with search) - on the original base"

HISTORY = [
    ("1.0.0", {"what": "g14 (step 835), bf16 - the lineage's first release", "status": "released", "parent": None, "dir": "release/g14-bf16"}),
    ("1.0.1", {"what": "1.0.0 quantised to 4 bits (GPTQ, group 64, MLX)", "status": "released", "parent": "1.0.0", "dir": "release/g14-4bit-gptq"}),
    ("1.0.2", {"what": "1.0.1 with the pooler retrained for the 4-bit weights - the model in the app", "status": "in the app", "parent": "1.0.1", "dir": "release/g14-4bit-gptq-trained"}),
    ("1.1.0", {"what": "multi-turn GRPO (mtg3, step 100: 's100') - a LoRA on 1.0.1's quantisation", "status": "adopted", "parent": "1.0.1",
               "files": {f"{MT}/mtg3_s100.safetensors": "lora.safetensors"}, "base": "1.0.1 (the same 4-bit grid; its HF form was on box G only)", "screens": S100_SCREENS}),
    ("1.1.1", {"what": "1.1.0 merged into the weights and quantised again to 4 bits (MLX)", "status": "not shipped (single 147/300)", "parent": "1.1.0",
               "dir": "pooler_distill/chatsft/s100_mlx4g"}),
    ("2.0.0", {"what": "a new 4-bit base: g14 quantised again from its own traces on box G2 (the original base was lost with box G), with 1.1.0's LoRA",
               "status": "baseline of 2.x", "parent": "1.1.0", "dir": "pooler_distill/chatsft/gq14_hf", "dir_to": "base",
               "files": {f"{MT}/mtg3_s100.safetensors": "lora.safetensors"}, "screens": "shard 0 49, single 146/300, br3 33/30, Dolphin 41 (no search)"}),
    ("2.1.0", {"what": "GRPO round mtg7 (TriviaQA + MuSiQue two-hop), step 40, on 2.0.0's base", "status": "adopted - the rounds' first start", "parent": "2.0.0",
               "files": {f"{MT}/mtg7_s40.safetensors": "lora.safetensors"}, "base": "2.0.0/base", "screens": "shard 0 56, single 162/300, br3 35/31, Dolphin 49 (no search)"}),
]


def listing(d):
    return [f for f in api.list_repo_files(R) if f.startswith(d.rstrip("/") + "/")]


def copy_files(pairs, msg):
    """pairs: [(src_in_repo, dst_in_repo)] - LFS files copied server side, small ones read and written again"""
    ops = []
    for src, dst in pairs:
        info = api.get_paths_info(R, [src])
        if not info: print(f"[versions] missing {src}", flush=True); continue
        if getattr(info[0], "lfs", None): ops.append(CommitOperationCopy(src_path_in_repo=src, path_in_repo=dst))
        else: ops.append(CommitOperationAdd(path_in_repo=dst, path_or_fileobj=open(hf_hub_download(R, src, token=TOKEN), "rb").read()))
    if ops: api.create_commit(R, operations=ops, commit_message=msg)
    return len(ops)


def put_json(path, obj, msg):
    api.create_commit(R, operations=[CommitOperationAdd(path_in_repo=path, path_or_fileobj=json.dumps(obj, indent=1, ensure_ascii=False).encode())], commit_message=msg)


def write_index(st):
    rows = ["# Deep Charger model versions", "", "MAJOR = new base weights, MINOR = an adopted training step on the same base, PATCH = the same model repackaged; "
            "X.Y.Z-mtgN.sS = a candidate (round mtgN, step S). Each version's files and version.json are under versions/<version>/.", "",
            "| version | status | what | screens |", "|---|---|---|---|"]
    for v, m in sorted(st["versions"].items(), key=lambda kv: [int(x) if x.isdigit() else x for x in re.split(r"[.\-]", kv[0])]):
        rows.append(f"| {v} | {m.get('status','')} | {m.get('what','')} | {m.get('screens','')} |")
    api.create_commit(R, operations=[CommitOperationAdd(path_in_repo=f"{V}/INDEX.md", path_or_fileobj=("\n".join(rows) + "\n").encode())], commit_message="versions: index")


def save(st): json.dump(st, open(STATE + ".tmp", "w"), indent=1); os.replace(STATE + ".tmp", STATE)


st = json.load(open(STATE)) if os.path.exists(STATE) else {"versions": {}, "current": "2.1.0", "seen": []}
changed = False
for v, m in HISTORY:
    if v in st["versions"]: continue
    pairs = []
    if m.get("dir"): pairs += [(f, f"{V}/{v}/" + (m["dir_to"] + "/" if m.get("dir_to") else "") + f[len(m["dir"]) + 1:]) for f in listing(m["dir"])]
    pairs += [(s, f"{V}/{v}/{d}") for s, d in (m.get("files") or {}).items()]
    n = copy_files(pairs, f"versions: {v}")
    meta = {k: m[k] for k in ("what", "status", "parent", "base", "screens") if m.get(k)}; meta.update({"version": v, "from": [s for s, _ in pairs]})
    put_json(f"{V}/{v}/version.json", meta, f"versions: {v} metadata"); st["versions"][v] = meta; changed = True
    print(f"[versions] {v}: {n} files", flush=True); save(st)


def bump_minor(v): a, b, _ = v.split("."); return f"{a}.{int(b) + 1}.0"


def follow():
    """the GRPO rounds' log: copies -> candidates, the full screens -> their scores, ACCEPTED -> the next minor"""
    global changed
    if not os.path.exists("/root/rounds.log"): return
    for line in open("/root/rounds.log"):
        line = line.strip()
        if not line or line in st["seen"]: continue
        nxt = bump_minor(st["current"])
        m = re.match(r"\[mtg(\d+)\] copy at step (\d+) \(on the hub\)", line)
        if m:
            n, s = m.groups(); v = f"{nxt}-mtg{n}.s{s}"
            copy_files([(f"{MT}/mtg{n}_s{s}.safetensors", f"{V}/{v}/lora.safetensors")], f"versions: candidate {v}")
            meta = {"version": v, "what": f"GRPO round mtg{n}, step {s}, from {st['current']}", "status": "candidate", "parent": st["current"],
                    "base": "2.0.0/base", "from": [f"{MT}/mtg{n}_s{s}.safetensors"]}
            st["versions"][v] = meta; put_json(f"{V}/{v}/version.json", meta, f"versions: {v} metadata"); changed = True
        m = re.match(r"\[mtg(\d+)\] mtg\d+_s(\d+) single shard 0: (\d+)/100", line)
        if m:
            n, s, c = m.groups(); v = f"{nxt}-mtg{n}.s{s}"
            if v in st["versions"]: st["versions"][v]["screens"] = f"shard 0 {c}"; changed = True
        m = re.match(r"\[mtg(\d+)\] mtg\d+_s(\d+): single (\d+)/300 .*br3 (\d+)/(\d+) .*Dolphin (\d+)", line)
        if m:
            n, s, c3, b1, b2, dl = m.groups(); v = f"{nxt}-mtg{n}.s{s}"
            if v in st["versions"]: st["versions"][v]["screens"] = st["versions"][v].get("screens", "") + f", single {c3}/300, br3 {b1}/{b2}, Dolphin {dl} (no search)"; changed = True
        m = re.match(r"\[mtg(\d+)\] ACCEPTED mtg\d+_s(\d+)", line)
        if m:
            n, s = m.groups(); cand = f"{nxt}-mtg{n}.s{s}"
            copy_files([(f"{V}/{cand}/lora.safetensors", f"{V}/{nxt}/lora.safetensors")], f"versions: {nxt}")
            meta = dict(st["versions"].get(cand, {})); meta.update({"version": nxt, "status": "adopted", "candidate": cand})
            st["versions"][nxt] = meta; st["versions"].setdefault(cand, {})["status"] = f"adopted as {nxt}"
            put_json(f"{V}/{nxt}/version.json", meta, f"versions: {nxt} metadata"); st["current"] = nxt; changed = True
            print(f"[versions] {cand} adopted as {nxt}", flush=True)
        if re.match(r"\[mtg(\d+)\] start kept", line):
            for v, meta in st["versions"].items():
                if meta.get("status") == "candidate" and v.startswith(nxt + "-"): meta["status"] = "not adopted"; changed = True
        st["seen"].append(line); save(st)


while True:
    follow()
    if changed:
        write_index(st); changed = False; save(st)
        print(f"[versions] index written; current {st['current']} ({time.strftime('%H:%M', time.gmtime())})", flush=True)
    if not A.watch: break
    time.sleep(600)
