#!/usr/bin/env bash
# Record a spoken prompt, transcribe it locally, and type it into the Claude
# Code pane. Bound to a wezterm key; see voice/wezterm-snippet.lua.
#
# Toggle, not push-to-talk: the first invocation records until you stop talking,
# a second invocation while recording stops it early. Holding a key down is
# awkward in a terminal that is also receiving the keystroke.
set -uo pipefail
source "$(dirname "$0")/config.sh"

REC_PID_FILE="$VOICE_RUN/rec.pid"
WAV="$VOICE_RUN/listen.wav"

# Second press while recording: stop and let the first invocation continue.
if [ -f "$REC_PID_FILE" ] && kill -0 "$(cat "$REC_PID_FILE")" 2>/dev/null; then
  kill -INT "$(cat "$REC_PID_FILE")" 2>/dev/null
  exit 0
fi

# Do not talk over the user.
pkill -f 'afplay .*claude-voice' 2>/dev/null

# sox stops on its own after ~1.8s of silence, so the common case needs no
# second keypress. 16 kHz mono is what Parakeet wants; resampling later costs
# quality for nothing.
#   silence 1 0.1 2%   : start recording once sound exceeds 2%
#           1 1.8 2%   : stop after 1.8s below 2%
rec -q -c 1 -r 16000 -b 16 "$WAV" \
    silence 1 0.1 2% 1 1.8 2% trim 0 30 &
echo $! > "$REC_PID_FILE"
wait $! 2>/dev/null
rm -f "$REC_PID_FILE"

[ -s "$WAV" ] || exit 0

TEXT=$(uv run --no-project --with parakeet-mlx python - "$WAV" "$PARAKEET_MODEL" <<'PY' 2>/dev/null
import sys
from parakeet_mlx import from_pretrained
model = from_pretrained(sys.argv[2])
print(model.transcribe(sys.argv[1]).text.strip())
PY
)
rm -f "$WAV"
[ -z "$TEXT" ] && exit 0

# Type it into the pane that invoked us. --no-paste sends it as keystrokes so it
# lands in the prompt for review; you press Enter, not this script. Putting words
# in your mouth AND hitting send is a bridge too far for a speech recognizer.
if [ -n "${WEZTERM_PANE:-}" ]; then
  wezterm cli send-text --pane-id "$WEZTERM_PANE" --no-paste "$TEXT"
else
  printf '%s\n' "$TEXT"
fi
