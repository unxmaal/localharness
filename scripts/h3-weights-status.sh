#!/usr/bin/env bash
# Is the MiniMax-H3 FL2VA download finished?
#
# Exit code is the contract, so this is scriptable:
#   0  complete
#   1  running
#   2  stopped short (re-run the fetcher; it resumes)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

DEST="${H3_MODEL_DIR:-/Volumes/Models/MiniMax-H3}"
PIDFILE="${H3_DOWNLOAD_PIDFILE:-.logs/h3-weights.pid}"
TARGET_GIB="${H3_TARGET_GIB:-134.2}"

# Liveness comes from a pidfile written by fetch-h3-weights.sh, not from
# `pgrep -f MiniMax-H3`. That pattern matches ANY process mentioning the path:
# an editor, a du, a grep, a test harness. A false RUNNING makes a caller wait
# forever on a download that already died.
RUNNING=no
if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE" 2>/dev/null || true)"
  # A stale pidfile from a crashed run must not read as running.
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    RUNNING=yes
  fi
fi

BYTES=$(du -sk "$DEST" 2>/dev/null | cut -f1)
BYTES=${BYTES:-0}
GIB=$(awk -v k="$BYTES" 'BEGIN{printf "%.1f", k/1024/1024}')
PCT=$(awk -v g="$GIB" -v t="$TARGET_GIB" 'BEGIN{printf "%.0f", (t>0? g/t*100 : 0)}')

# .incomplete files exist only while a transfer is mid-flight.
PARTIAL=$(find "$DEST" -name '*.incomplete' 2>/dev/null | wc -l | tr -d ' ')

printf 'downloaded   %s / %s GiB  (%s%%)\n' "$GIB" "$TARGET_GIB" "$PCT"
printf 'in progress  %s partial file(s)\n' "$PARTIAL"
printf 'process      %s\n' "$RUNNING"

if [ "$RUNNING" = no ] && [ "$PARTIAL" -eq 0 ] && [ "$PCT" -ge 99 ]; then
  printf '\nSTATUS: COMPLETE.\n'
  exit 0
elif [ "$RUNNING" = no ]; then
  printf '\nSTATUS: STOPPED SHORT. Re-run scripts/fetch-h3-weights.sh; it resumes.\n'
  exit 2
fi

LOG=.logs/h3-weights.log
if [ -f "$LOG" ]; then
  ELAPSED=$(( $(date +%s) - $(stat -f %B "$LOG") ))
  if [ "$ELAPSED" -gt 60 ]; then
    awk -v g="$GIB" -v t="$TARGET_GIB" -v e="$ELAPSED" 'BEGIN{
      rate = g / e
      if (rate > 0) printf "rate         %.0f MB/s\neta          %.0f min remaining\n",
        rate * 1024, (t - g) / rate / 60 }'
  fi
fi
printf '\nSTATUS: RUNNING.\n'
exit 1
