#!/bin/bash
# Pull-based control loop for a Vast box. No inbound SSH needed:
#  - every 60s fetch ctl/<BOX>.sh from the repo branch; when its hash changes, run it once
#    (output -> /root/ctl.log and the container log)
#  - every 5 min print a one-line status to the container log (/proc/1/fd/1), readable via the
#    Vast request_logs API from anywhere.
BOX=${1:?box name}
RAW="https://raw.githubusercontent.com/bayamax/deep-charger/claude/vast-ai-key-sharing-h0725i/ctl"
LOG=/proc/1/fd/1
LAST=""
n=0
while :; do
  f=$(curl -sS --max-time 20 -H "Cache-Control: no-cache" "$RAW/$BOX.sh?$(date +%s)" 2>/dev/null)
  if [ -n "$f" ] && ! echo "$f" | grep -q "^404"; then
    h=$(echo "$f" | md5sum | cut -c1-12)
    if [ "$h" != "$LAST" ]; then
      LAST=$h
      echo "$f" > /root/ctl_cmd.sh
      echo "=== CTL $(date -u +%H:%M) run $BOX.sh ($h) ===" | tee -a /root/ctl.log >> $LOG
      ( bash /root/ctl_cmd.sh 2>&1; echo "=== CTL done ($h) ===" ) | tee -a /root/ctl.log >> $LOG &
    fi
  fi
  if [ $((n % 5)) -eq 0 ] && [ -x /root/status.sh ]; then
    { echo "--- STATUS $(date -u +%H:%M) ---"; bash /root/status.sh 2>&1; } >> $LOG
  fi
  n=$((n+1)); sleep 60
done
