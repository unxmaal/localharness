#!/usr/bin/env bash
# Starting and stopping the services on Windows, on demand only.
#
# THE DEFAULT IS OFF, AND THERE IS NOTHING TO TURN OFF. scripts/launchd.sh
# installs units with RunAtLoad and KeepAlive, which is right for a mini whose
# job is to serve. This machine's job is to play games and to run this
# sometimes, so nothing here is registered with Windows at all: no scheduled
# task, no Run key, no startup shortcut. A reboot leaves the card empty, and
# `stop` leaves nothing behind that could start again on its own.
#
#   ./scripts/services.sh start          # gateway, text, audio
#   ./scripts/services.sh start llamacpp # just one
#   ./scripts/services.sh stop           # all of them, and the card is free
#   ./scripts/services.sh status         # what is up, and what the card holds
#
# The text server also gives the card back by itself after
# $LLAMACPP_SLEEP_IDLE seconds, so a session left running overnight is not
# holding 8 GB by morning. `stop` is the immediate version of the same thing.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

LH_HOME="${LOCALHARNESS_HOME:-$HOME/localharness}"
RUN="$LH_HOME/run"
LOGS="$LH_HOME/logs"
mkdir -p "$RUN" "$LOGS"

#: name -> the launcher that serves it.
declare -A LAUNCHER=(
  [gateway]="scripts/serve-gateway.sh"
  [llamacpp]="scripts/serve-llamacpp.sh"
  [audio]="scripts/serve-audio-cuda.sh"
)
ALL=(gateway llamacpp audio)

usage() {
  echo "usage: $0 {start|stop|status} [${ALL[*]}]" >&2
  exit 2
}

pidfile() { printf '%s\n' "$RUN/$1.pid"; }

# PowerShell cannot take a POSIX path for a working directory or a redirect,
# and Git Bash reports every path that way. Passing $PWD straight through made
# Start-Process fail before it launched anything, which arrived as an empty
# error log rather than as a message.
winpath() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -w "$1"
  else
    printf '%s\n' "$1"
  fi
}

alive() {
  local pid="$1"
  [ -n "$pid" ] || return 1
  # tasklist rather than kill -0: a Windows pid is not in bash's process table,
  # so kill -0 reports every one of them as gone.
  tasklist //FI "PID eq $pid" //NH 2>/dev/null | grep -q "$pid"
}

start_one() {
  local name="$1" script="${LAUNCHER[$1]:-}" pid bash_exe work out err pidpath
  [ -n "$script" ] || { echo "unknown service $name" >&2; return 1; }
  pid="$(cat "$(pidfile "$name")" 2>/dev/null || true)"
  if alive "$pid"; then
    echo "$name already up (pid $pid)"
    return 0
  fi

  bash_exe="$(winpath "$(command -v bash)")"
  work="$(winpath "$PWD")"
  out="$(winpath "$LOGS/$name.log")"
  err="$(winpath "$LOGS/$name.err.log")"
  pidpath="$(winpath "$(pidfile "$name")")"

  # Start-Process rather than `&`: a background job in this shell dies with the
  # terminal, and the point is to close the terminal and go and play something.
  #
  # THE PID GOES TO A FILE RATHER THAN TO STDOUT. Capturing it with $( ) hangs
  # forever: the process being launched inherits the pipe, so the substitution
  # waits for an end-of-file that arrives only when the server it just started
  # exits, which is the opposite of starting something in the background.
  rm -f "$(pidfile "$name")"
  powershell -NoProfile -Command \
    "(Start-Process -FilePath '$bash_exe' -ArgumentList '$script'\
     -WorkingDirectory '$work' -WindowStyle Hidden -PassThru\
     -RedirectStandardOutput '$out' -RedirectStandardError '$err').Id\
     | Set-Content -Path '$pidpath'" >/dev/null 2>&1

  pid="$(tr -d ' \r\n' < "$(pidfile "$name")" 2>/dev/null || true)"
  case "$pid" in
    "" | *[!0-9]*)
      echo "$name failed to start; see $LOGS/$name.err.log" >&2
      rm -f "$(pidfile "$name")"
      return 1 ;;
  esac
  echo "$name started (pid $pid)"
}

stop_one() {
  local name="$1" pid
  pid="$(cat "$(pidfile "$name")" 2>/dev/null || true)"
  if ! alive "$pid"; then
    rm -f "$(pidfile "$name")"
    echo "$name not running"
    return 0
  fi
  # /T for the tree. The pid recorded is bash; the server holding the card is
  # its child, and ending the parent alone leaves the model resident.
  taskkill //F //T //PID "$pid" >/dev/null 2>&1
  rm -f "$(pidfile "$name")"
  echo "$name stopped"
}

status() {
  local name pid
  for name in "${ALL[@]}"; do
    pid="$(cat "$(pidfile "$name")" 2>/dev/null || true)"
    if alive "$pid"; then
      printf '%-9s up    pid %s\n' "$name" "$pid"
    else
      printf '%-9s down\n' "$name"
    fi
  done
  # The question before starting a game is what the card is holding, which is
  # a different question from which of these is running: a sleeping text server
  # is up and holding nothing.
  if command -v nvidia-smi >/dev/null 2>&1; then
    printf 'vram      %s\n' "$(nvidia-smi --query-gpu=memory.used,memory.total \
      --format=csv,noheader 2>/dev/null | head -1)"
  fi
}

[ $# -ge 1 ] || usage
verb="$1"; shift
targets=("$@")
[ ${#targets[@]} -gt 0 ] || targets=("${ALL[@]}")

case "$verb" in
  start)  for s in "${targets[@]}"; do start_one "$s"; done ;;
  stop)   for s in "${targets[@]}"; do stop_one "$s"; done ;;
  status) status ;;
  *)      usage ;;
esac
