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

# sox stderr goes to a log rather than the terminal: it emits a harmless
# "trim: Last 1 position(s) not reached" on every gated recording, which would
# otherwise print into the pane you are trying to type into. Real failures are
# still recoverable from $VOICE_RUN/rec.log.
#
# sox stops on its own after VOICE_HANG seconds of silence, so the common case
# needs no second keypress.
#   silence 1 0.1 <thr>      : start capturing once the signal exceeds <thr>
#           1 <hang> <thr>   : stop after <hang> seconds below it
#   trim 0 <max>             : hard cap, applied after the silence gate
#
# channels/rate are EFFECTS, not device flags. Asking the device for 16k mono
# fails on hardware that will not do it (a Yeti is 48k stereo) and sox only
# warns: "can't set sample rate 16000; using 48000". As effects they are applied
# in software and always hold.
rec -q -b 16 "$WAV" \
    channels 1 rate 16000 \
    silence 1 0.1 "$VOICE_THRESHOLD" 1 "$VOICE_HANG" "$VOICE_THRESHOLD" \
    trim 0 "$VOICE_MAX_SECONDS" 2>>"$VOICE_RUN/rec.log" &
REC_PID=$!
echo "$REC_PID" > "$REC_PID_FILE"

# Watchdog: if nothing crosses the threshold, sox writes no bytes and waits
# forever with the microphone open. Give up rather than hang.
(
  for _ in $(seq 1 "$VOICE_ONSET_TIMEOUT"); do
    sleep 1
    kill -0 "$REC_PID" 2>/dev/null || exit 0
    [ -s "$WAV" ] && exit 0        # capture started, the silence gate owns it now
  done
  kill -INT "$REC_PID" 2>/dev/null
) &
WATCHDOG=$!

wait "$REC_PID" 2>/dev/null
kill "$WATCHDOG" 2>/dev/null
rm -f "$REC_PID_FILE"

[ -s "$WAV" ] || exit 0

# Transcribe through the same mlx_audio.server that does TTS. It holds Parakeet
# resident, so this costs ~0.26s instead of ~0.8s with a per-call model load, and
# it is one fewer moving part than a second process.
TEXT=$(curl -s --max-time 60 "$TTS_HOST/v1/audio/transcriptions" \
         -F "file=@$WAV" -F "model=$PARAKEET_MODEL" 2>/dev/null \
       | jq -r '.text // empty' | sed 's/^ *//;s/ *$//')

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
