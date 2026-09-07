# box C: the log store truncates lines at 500 chars -> print the base64 in 400-char lines, part by part
P=${PART:-1}; L=1500   # lines per part (400 chars each -> ~600KB)
[ -f /root/gen_full.b64 ] || base64 -w400 /root/gen_full.gz > /root/gen_full.b64
T=$(wc -l < /root/gen_full.b64); echo "B64 lines $T part $P of $(( (T+L-1)/L ))"
echo "PART_BEGIN $P"; sed -n "$(( (P-1)*L+1 )),$(( P*L ))p" /root/gen_full.b64; echo "PART_END $P"
