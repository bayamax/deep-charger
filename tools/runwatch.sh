#!/bin/sh
# Show what the training box is doing, from a phone.
#
# iSH (Alpine) has curl and sh; it does not need python. Put the Vast key in
# ~/.vastkey (chmod 600) and the instance id in ~/.vastbox, or pass the id as $1.
#
#   apk add curl
#   printf '%s' '<vast api key>' > ~/.vastkey && chmod 600 ~/.vastkey
#   printf '%s' '51230932'       > ~/.vastbox
#   . path/to/runwatch.sh            # defines: run, runq, runfull
#
# run      the last few step lines and the latest upload
# score    the mean reward per block, both sides, as of the box's last half-hourly print
# runq     one line: where the run is, and both pass rates
# runfull  the whole tail, for when something looks wrong

_vlog() {
  key=$(tr -d '[:space:]' < "${VASTKEY:-$HOME/.vastkey}")
  box=${1:-$(tr -d '[:space:]' < "${VASTBOX:-$HOME/.vastbox}")}
  tail=${2:-6000}
  url=$(curl -sS -X PUT "https://console.vast.ai/api/v0/instances/request_logs/$box/" \
        -H "Authorization: Bearer $key" -H 'Content-Type: application/json' \
        -d "{\"tail\":\"$tail\"}" | sed -n 's/.*"result_url"[ ]*:[ ]*"\([^"]*\)".*/\1/p')
  [ -n "$url" ] || { echo "no log url (key or box id wrong?)"; return 1; }
  n=0
  while [ $n -lt 20 ]; do
    body=$(curl -sS "$url" 2>/dev/null)
    [ -n "$body" ] && { printf '%s\n' "$body"; return 0; }
    n=$((n + 1)); sleep 3
  done
  echo "log not ready"; return 1
}

run() {
  _vlog "$@" | grep -E '^\[step|^\[online|^ONLINE_|^\[save\]' | tail -12
}

runq() {
  _vlog "$@" | grep -E '^\[step' | tail -1 | sed 's/ | / /g'
}

runfull() {
  _vlog "$@" | tail -60
}

# score: the run's mean reward per block, as the box last printed it (every half hour)
score() {
  _vlog "$@" 12000 | grep '^SCORE' | awk '/through step/{n=NR} {a[NR]=$0} END{for(i=n;i<=NR;i++) print substr(a[i],7)}'
}

# The box also mirrors its own logs to the (public) hub every ten minutes, so these work without the Vast key.
HUBLOG=https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/pooler_distill/chatsft/audit/boxlog.txt

# hub      the last step lines, events and evaluation progress, from the hub mirror
hub() {
  curl -sSL "$HUBLOG" > /tmp/boxlog.txt || { echo "hub mirror not reachable"; return 1; }
  head -1 /tmp/boxlog.txt
  grep -E '^\[step|^ONLINE_|^REEVAL_|^DOLPHIN_ACC|^GSM_ACC|^\[warn\]' /tmp/boxlog.txt | awk '!seen[$0]++' | tail -${1:-12} | cut -c1-150
  sed -n '/^--- eval progress ---/,/^--- processes ---/p' /tmp/boxlog.txt | grep -v '^---' | head -2
}

# hubscore the score table from the hub mirror (same table as score)
hubscore() {
  curl -sSL "$HUBLOG" | grep '^SCORE' | awk '/through step/{n=NR} {a[NR]=$0} END{for(i=n;i<=NR;i++) print substr(a[i],7)}'
}
