#!/usr/bin/env python3
"""The e2e ladder: raise the sentence BART as far as it goes, one change at a time, each run to its plateau.

The user (2026-10-05): adjust the negatives -> add bge layers -> add BART layers, round and round, each step taken when
the previous one has levelled off; keep the best per capacity. Each rung is a train_e2e.py run from the current best
checkpoint with ONE change, stopped when --patience evaluations bring no gain (--min-gain); kept if it beats the
current best, otherwise the next change is tried from the same best. The ladder ends when a whole round of the three
changes brings nothing, or when every change is used up.

  negatives : (batch, page queue) one level up: (32,16) -> (32,64) -> (64,64) -> (64,256) -> (96,256)
  bge       : two more layers (middle layers duplicated), 12 -> 14 -> 16 -> ...  (--max-enc)
  BART      : layers doubled with the new ones as the identity, 8 -> 16 -> 32  (--max-layers)

Selection metric: page top-1 among --pool held-out articles (harder than the 2000, closer to use); the 2000's top-1 is
logged beside it. Guard (2026-10-06): e2e1 raised that metric while the app model's real queries got WORSE (its bge
drifted on the query path), so every rung trains with the query-form anchor (--anchor-q) and is kept only if, besides
the pool gain, the real-query search on a dev split of the app queries (the first 300 keyword queries; the rest stay
untouched for the final reading) does not fall: page recall@50 and the lead sentence's recall@10 within 2 points. Per capacity (bge layers, BART layers) the best checkpoint is kept and uploaded to the hub
(sentbart/e2e/cap_bge{n}_bart{m}/), with ladder.json the record of every rung.

  python3 e2e_ladder.py --start /root/sb/e2e1/model_best.pt --work /root/sb/ladder
"""
import argparse, json, os, re, shutil, subprocess, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--start", required=True); ap.add_argument("--work", required=True)
ap.add_argument("--vec", default="/root/sb/data/docs"); ap.add_argument("--text", default="/root/sb/data/e2e")
ap.add_argument("--eval-text", default="/root/sb/data/docs/docs_001.jsonl")
ap.add_argument("--pool", type=int, default=20000); ap.add_argument("--steps", type=int, default=30000)
ap.add_argument("--patience", type=int, default=3); ap.add_argument("--min-gain", type=float, default=0.003)
ap.add_argument("--max-enc", type=int, default=20); ap.add_argument("--max-layers", type=int, default=32)
ap.add_argument("--repo", default="baya1116/hypernet-sp-distill")
ap.add_argument("--anchor-q", type=int, default=256)
ap.add_argument("--qfile", default="", help="qgen.py's queries.jsonl: train_e2e --queries (the query -> article losses)")
ap.add_argument("--select", default="pool_top1", help="the metric rungs are selected on (pool_top1, or pool_qh_top1 with --qfile)")
ap.add_argument("--queries", default="/root/sb/se2/dl/sentbart/searcheval/dcq.jsonl"); ap.add_argument("--dev", type=int, default=300)
ap.add_argument("--guard", type=int, default=1, help="0: no real-query search check between rungs (documents that are not Wikipedia articles)")
ap.add_argument("--hub", default="sentbart/e2e", help="where the rungs' logs, the state and the best per capacity go on the hub")
A = ap.parse_args()
os.makedirs(A.work, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
NEG = [(32, 16), (32, 64), (64, 64), (64, 256), (96, 256)]
REC = os.path.join(A.work, "ladder.json")


def log(m): print(f"[ladder] {m} {time.strftime('%H:%M', time.gmtime())}", flush=True)


def upload(src, dst):
    subprocess.run(["hf", "upload", A.repo, src, dst], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run(name, init, layers, enc_layers, neg, grow=False, steps=None):
    out = os.path.join(A.work, name)
    cmd = [sys.executable, os.path.join(HERE, "train_e2e.py"), "--vec", A.vec, "--text", A.text, "--eval-text", A.eval_text,
           "--out", out, "--steps", str(A.steps if steps is None else steps), "--batch", str(neg[0]), "--page-queue", str(neg[1]),
           "--layers", str(layers), "--enc-layers", str(enc_layers), "--pool", str(A.pool),
           "--patience", str(A.patience), "--min-gain", str(A.min_gain), "--anchor-q", str(A.anchor_q), "--select", A.select] \
          + (["--queries", A.qfile] if A.qfile else []) + [("--grow" if grow else "--init"), init]
    log(f"{name}: batch {neg[0]} queue {neg[1]}, bge {enc_layers} layers, BART {layers}+{layers}{' (grown)' if grow else ''}")
    with open(os.path.join(A.work, name + ".out"), "w") as fo:
        rc = subprocess.run(cmd, stdout=fo, stderr=subprocess.STDOUT).returncode
    tl = os.path.join(out, "train.log"); best, p2k, step = -1.0, -1.0, 0
    if os.path.exists(tl):
        for line in open(tl):
            m = re.match(r"\[eval (\d+)\] .*page_top1 ([0-9.]+).*" + re.escape(A.select) + r" ([0-9.]+)", line)
            if m and float(m.group(3)) > best: best, p2k, step = float(m.group(3)), float(m.group(2)), int(m.group(1))
        upload(tl, f"{A.hub}/ladder/{name}/train.log")
    tail = open(os.path.join(A.work, name + ".out")).read()[-400:].replace("\n", " | ")
    if rc != 0: log(f"{name}: exit {rc}: {tail[-250:]}")
    return {"name": name, "rc": rc, "pool_top1": best, "page_top1": p2k, "step": step,
            "ckpt": os.path.join(out, "model_best.pt"), "layers": layers, "enc_layers": enc_layers, "neg": list(neg)}


DEV = os.path.join(A.work, "dcq_dev.jsonl")
if A.guard and not os.path.exists(DEV):
    rows = [json.loads(l) for l in open(A.queries) if l.strip()]
    rows = [{"idx": r["idx"], "q_api": r["q_api"]} for r in rows if r.get("q_api")][:A.dev]
    open(DEV, "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def guard(r):
    """the real-query check: the rung's best on the dev queries (two-stage search over all 118,398 held-out articles)"""
    if not A.guard: r["dev_page50"] = r["dev_lead10"] = r["dev_page10"] = -1.0; return r
    o = os.path.join(A.work, r["name"] + "_dev.json")
    with open(os.path.join(A.work, r["name"] + "_dev.out"), "w") as fo:
        subprocess.run([sys.executable, os.path.join(HERE, "search_cascade.py"), "--vec", A.vec, "--text", A.eval_text, "--ckpt", r["ckpt"],
                        "--queries", DEV, "--out", o], stdout=fo, stderr=subprocess.STDOUT)
    if not os.path.exists(o): r["dev_page50"] = r["dev_lead10"] = -1.0; return r
    d = json.load(open(o)); r["dev_page50"] = d["q_api|page"]["recall@50"]; r["dev_lead10"] = d["q_api|lead"]["recall@10"]
    r["dev_page10"] = d["q_api|page"]["recall@10"]; upload(o, f"{A.hub}/ladder/{r['name']}_dev.json")
    log(f"{r['name']} dev queries: page recall@10 {r['dev_page10']:.3f} @50 {r['dev_page50']:.3f}, lead recall@10 {r['dev_lead10']:.3f}")
    return r


rec = json.load(open(REC)) if os.path.exists(REC) else {"rungs": [], "cur": None, "cap": {}}
def save_rec(): json.dump(rec, open(REC, "w"), indent=1); upload(REC, f"{A.hub}/ladder/ladder.json")


def keep_cap(r):
    key = f"bge{r['enc_layers']}_bart{r['layers']}"
    if r["pool_top1"] > rec["cap"].get(key, {}).get("pool_top1", -1) and os.path.exists(r["ckpt"]):
        d = os.path.join(A.work, "cap_" + key); os.makedirs(d, exist_ok=True)
        try: shutil.copy(r["ckpt"], os.path.join(d, "model_best.pt"))
        except OSError as e: log(f"could not keep {key} locally ({e}); the hub copy follows"); upload(r["ckpt"], f"{A.hub}/cap_{key}/model_best.pt"); return
        rec["cap"][key] = {k: r[k] for k in ("name", "pool_top1", "page_top1", "step", "neg")}
        upload(os.path.join(d, "model_best.pt"), f"{A.hub}/cap_{key}/model_best.pt")
        log(f"best for {key}: pool {r['pool_top1']:.3f} (2000: {r['page_top1']:.3f}) from {r['name']}")


if rec["cur"] is None:   # the start measured (no training), then rung 0: e2e from it with the base settings and the query anchor
    r = guard(run("r00_ref", A.start, 8, 12, NEG[0], steps=0)); rec["rungs"].append(r)
    log(f"reference (the start, no e2e): pool top-1 {r['pool_top1']:.3f}, 2000 top-1 {r['page_top1']:.3f}")
    ref = r
    r = guard(run("r01_e2e", A.start, 8, 12, NEG[0]))
    assert r["pool_top1"] >= 0, "rung 0 could not be measured"
    if r["dev_page50"] < ref["dev_page50"] - 0.02 or r["dev_lead10"] < ref["dev_lead10"] - 0.02:
        log(f"warning: e2e with the query anchor still lowers the real-query search (page@50 {ref['dev_page50']:.3f} -> {r['dev_page50']:.3f}, lead@10 {ref['dev_lead10']:.3f} -> {r['dev_lead10']:.3f}); the ladder goes on from it")
    cur_ck = os.path.join(A.work, "cur.pt"); shutil.copy(r["ckpt"], cur_ck)
    rec["cur"] = {**r, "ckpt": cur_ck, "neg_i": 0, "turn": 0, "fails": 0}; rec["rungs"].append(r); keep_cap(r); save_rec()

ORDER = ["neg", "bge", "bart"]
while rec["cur"]["fails"] < 3:
    cur = rec["cur"]; kind = ORDER[cur["turn"] % 3]; n = len(rec["rungs"])
    neg_i, layers, enc_layers, grow = cur["neg_i"], cur["layers"], cur["enc_layers"], False
    if kind == "neg":
        if neg_i + 1 >= len(NEG): log("negatives used up"); cur["turn"] += 1; cur["fails"] += 1; save_rec(); continue
        neg_i += 1
    elif kind == "bge":
        if enc_layers + 2 > A.max_enc: log("bge layers used up"); cur["turn"] += 1; cur["fails"] += 1; save_rec(); continue
        enc_layers += 2
    else:
        if layers * 2 > A.max_layers: log("BART layers used up"); cur["turn"] += 1; cur["fails"] += 1; save_rec(); continue
        layers *= 2; grow = True
    r = run(f"r{n:02d}_{kind}", cur["ckpt"], layers, enc_layers, NEG[neg_i], grow=grow)
    if r["rc"] == 0 and r["pool_top1"] > cur["pool_top1"] + A.min_gain: r = guard(r)
    rec["rungs"].append(r); keep_cap(r)
    if r["rc"] == 0 and r["pool_top1"] > cur["pool_top1"] + A.min_gain and r.get("dev_page50", -1) >= cur.get("dev_page50", -1) - 0.02 \
            and r.get("dev_lead10", -1) >= cur.get("dev_lead10", -1) - 0.02:
        shutil.copy(r["ckpt"], cur["ckpt"])
        rec["cur"] = {**r, "ckpt": cur["ckpt"], "neg_i": neg_i, "turn": cur["turn"] + 1, "fails": 0}
        log(f"{r['name']} kept: pool {cur['pool_top1']:.3f} -> {r['pool_top1']:.3f} (2000: {r['page_top1']:.3f})")
    else:
        cur["turn"] += 1; cur["fails"] += 1
        log(f"{r['name']} not kept: pool {r['pool_top1']:.3f} vs {cur['pool_top1']:.3f}")
    for f in ("state_e2e.pt", "model_latest.pt", "model_best.pt"):   # what is kept lives in cur.pt and cap_*/ (the disk is ~13 GB)
        p = os.path.join(A.work, r["name"], f)
        if os.path.exists(p): os.remove(p)
    save_rec()
log(f"done: a whole round without a gain; best pool top-1 {rec['cur']['pool_top1']:.3f} ({rec['cur']['name']})")
print("LADDER_DONE", flush=True)
