# box C: raw window for the compressed eval of the new model (read by /root/after_sft.sh at launch time)
echo 768 > /root/rw.txt; echo "rw set to $(cat /root/rw.txt)"; t
echo "RW_SET $(date -u)"
