# box A: STOP generation (loss plateaued by the 3-way split rule), final sync, ship full jsonl to C
pkill -f "eval_heldout_paralle[l].py"; sleep 5; pgrep -af "eval_heldout_paralle[l]" || echo "gen stopped"
pkill -f "/root/sync.sh"
T="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -i /root/.ssh/id_sync -P 30732"
python3 - <<'PY'
import json, os
src="/root/work/eval_gen_mus200_p0of1.jsonl"; dst="/root/work/gen_corpus.jsonl"; n=k=0
with open(dst,"w") as fh:
    for l in open(src):
        try: r=json.loads(l)
        except Exception: continue
        n+=1
        if not (r.get("correct") and r.get("grounded")): continue
        k+=1; fh.write(json.dumps({"q":r["q"],"gold":r.get("gold",""),"ok":True,"gnd":True,"srch":r.get("ns"),"traj":r["text"]}, ensure_ascii=False)+"\n")
print("FINAL rows=%d keep=%d" % (n,k))
PY
scp $T -q /root/work/gen_corpus.jsonl root@ssh9.vast.ai:/root/harvest_sft/gen_mus200.jsonl && echo "final harvest synced"
scp $T -q /root/work/eval_gen_mus200_p0of1.jsonl root@ssh9.vast.ai:/root/work/gen_mus200_full.jsonl && echo "full jsonl shipped"
scp $T -q /root/gen_mus200.log root@ssh9.vast.ai:/root/work/gen_mus200.log
echo "A_FINAL_SYNC_DONE $(date -u)"
