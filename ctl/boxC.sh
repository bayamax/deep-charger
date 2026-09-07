# box C: retry fetching the analysis script (previous curl died with an SSL EOF) and run it on post so far.
cd /root/work
RAW=https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl
for i in 1 2 3 4 5; do curl -sS -L --retry 3 -o /root/work/analyze_pool.py "$RAW/analyze_pool.py" && python3 -m py_compile /root/work/analyze_pool.py && break; sleep 10; done
python3 /root/work/analyze_pool.py /root/work/teacher600.jsonl /root/work/pooleval_post.jsonl 512 2>&1 | grep -v Warning
echo "ANALYZE2_DONE $(date -u)"
