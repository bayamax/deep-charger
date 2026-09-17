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
