# box C: preserve the teacher trajectories (1276 rows) through the container log (gzip+base64), nothing else on the box can reach outside
cd /root/work
ls -la gen_mus200_full.jsonl pooleval_all_c.jsonl pooleval_all_nc.jsonl eval300.jsonl /root/fft_new_all.safetensors
gzip -9c gen_mus200_full.jsonl > /root/gen_full.gz; md5sum gen_mus200_full.jsonl /root/gen_full.gz; wc -c /root/gen_full.gz
echo "BLOB_BEGIN"; base64 -w0 /root/gen_full.gz; echo; echo "BLOB_END"
echo "PRESERVE_DONE $(date -u)"
