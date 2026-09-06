# box A: (1) ssh key for pushing rows to box C, printed so it can be attached  (2) sync loop every 5 min
[ -f /root/.ssh/id_sync ] || ssh-keygen -q -t ed25519 -N "" -f /root/.ssh/id_sync -C boxA-sync
echo "BOXA_PUBKEY $(cat /root/.ssh/id_sync.pub)"
cat > /root/sync.sh <<'SY'
#!/bin/bash
# Convert finished rollouts (correct & grounded) into the SFT harness corpus format and push to box C.
# The harness dedups by question, so overwriting the whole file each time is safe.
T="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -i /root/.ssh/id_sync -p 30732"
while :; do
  python3 - <<'PY'
import json, os
src="/root/work/eval_gen_mus200_p0of1.jsonl"; dst="/root/work/gen_corpus.jsonl"
n=k=0
if os.path.exists(src):
    with open(dst+".tmp","w") as fh:
        for l in open(src):
            try: r=json.loads(l)
            except Exception: continue
            n+=1
            if not (r.get("correct") and r.get("grounded")): continue
            k+=1
            fh.write(json.dumps({"q":r["q"],"gold":r.get("gold",""),"ok":True,"gnd":True,"srch":r.get("ns"),"traj":r["text"]}, ensure_ascii=False)+"\n")
    os.replace(dst+".tmp", dst)
open("/root/sync_state","w").write("rows=%d keep=%d\n" % (n,k))
PY
  if [ -s /root/work/gen_corpus.jsonl ]; then
    scp $T -q /root/work/gen_corpus.jsonl root@ssh9.vast.ai:/root/harvest_sft/gen_mus200.jsonl && echo "$(date -u +%H:%M) synced $(cat /root/sync_state)" >> /root/sync.log || echo "$(date -u +%H:%M) scp failed" >> /root/sync.log
  fi
  sleep 300
done
SY
chmod +x /root/sync.sh
pkill -f "/root/sync.sh"; sleep 1; (setsid nohup bash /root/sync.sh > /root/sync_boot.log 2>&1 < /dev/null &)
sleep 2; pgrep -af "/root/sync.sh" | head -2; tail -2 /root/sync.log 2>/dev/null
bash /root/do_gen.sh
