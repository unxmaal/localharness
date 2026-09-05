#!/usr/bin/env bash
# Is the MiniMax-H3 FL2VA download finished? Run this any time.
set -uo pipefail
cd "$(dirname "$0")/.."
DEST="${H3_MODEL_DIR:-/Volumes/Models/MiniMax-H3}"
TARGET_GIB=134.2

if pgrep -qf 'MiniMax-H3'; then RUNNING=yes; else RUNNING=no; fi

BYTES=$(du -sk "$DEST" 2>/dev/null | cut -f1)
BYTES=${BYTES:-0}
GIB=$(awk -v k="$BYTES" 'BEGIN{printf "%.1f", k/1024/1024}')
PCT=$(awk -v g="$GIB" -v t="$TARGET_GIB" 'BEGIN{printf "%.0f", g/t*100}')

# .incomplete files exist only while a transfer is mid-flight
PARTIAL=$(find "$DEST" -name '*.incomplete' 2>/dev/null | wc -l | tr -d ' ')

printf 'downloaded   %s / %s GiB  (%s%%)\n' "$GIB" "$TARGET_GIB" "$PCT"
printf 'in progress  %s partial file(s)\n' "$PARTIAL"
printf 'process      %s\n' "$RUNNING"

if [ "$RUNNING" = no ] && [ "$PARTIAL" -eq 0 ] && [ "${PCT:-0}" -ge 99 ]; then
  echo ""
  echo "STATUS: COMPLETE. Next: scripts/quiesce.sh, then a --ssd-streaming --profile run."
  exit 0
elif [ "$RUNNING" = no ]; then
  echo ""
  echo "STATUS: STOPPED SHORT. Re-run scripts/fetch-h3-weights.sh; it resumes."
  exit 2
else
  # rough ETA from how much has landed since the log's first write
  LOG=.logs/h3-weights.log
  if [ -f "$LOG" ]; then
    ELAPSED=$(( $(date +%s) - $(stat -f %B "$LOG") ))
    if [ "$ELAPSED" -gt 60 ] && [ "$(printf '%.0f' "$GIB")" -gt 0 ]; then
      awk -v g="$GIB" -v t="$TARGET_GIB" -v e="$ELAPSED" 'BEGIN{
        rate=g/e; if(rate>0){ printf "rate         %.0f MB/s\neta          %.0f min remaining\n",
          rate*1024, (t-g)/rate/60 }}'
    fi
  fi
  echo ""
  echo "STATUS: RUNNING."
  exit 1
fi
