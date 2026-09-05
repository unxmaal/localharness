#!/usr/bin/env bash
# Claude Code `Stop` hook: speak a short summary of the response just finished.
#
# Registered in ~/.claude/settings.json under hooks.Stop. Receives the hook
# payload as JSON on stdin. Uses `last_assistant_message` rather than reading
# transcript_path, which the docs note can lag behind.
#
# Everything slow happens in a detached child so the hook returns immediately
# and never delays the turn.
set -uo pipefail
source "$(dirname "$0")/config.sh"

INPUT=$(cat)

# stop_hook_active guards against a Stop hook that triggers another turn, which
# would fire Stop again. This hook only plays audio and cannot cause a turn, but
# the guard costs nothing and keeps that true if the script grows.
[ "$(jq -r '.stop_hook_active // false' <<<"$INPUT")" = "true" ] && exit 0
[ -f "$VOICE_MUTE_FLAG" ] && exit 0

MSG=$(jq -r '.last_assistant_message // empty' <<<"$INPUT")
[ -z "$MSG" ] && exit 0

(
  # Only one voice at a time: a new response cuts off the previous one rather
  # than queueing behind it, which is what you want in conversation.
  [ -f "$VOICE_RUN/play.pid" ] && kill "$(cat "$VOICE_RUN/play.pid")" 2>/dev/null
  pkill -f 'afplay .*claude-voice' 2>/dev/null

  # Mechanical strip first, so the summarizer spends its context on prose rather
  # than on code it is only going to discard.
  CLEAN=$(python3 - <<'PY' <<<"$MSG"
import re, sys
t = sys.stdin.read()
t = re.sub(r'```.*?```', ' (code) ', t, flags=re.S)     # fenced code
t = re.sub(r'`[^`]+`', ' ', t)                           # inline code
t = re.sub(r'^\s*\|.*\|\s*$', '', t, flags=re.M)         # table rows
t = re.sub(r'https?://\S+', ' a link ', t)
t = re.sub(r'(/[\w.\-]+){2,}', ' a path ', t)            # unix paths
t = re.sub(r'[#*_>`]', '', t)                            # md punctuation
t = re.sub(r'\s+', ' ', t).strip()
print(t[:6000])
PY
)
  [ -z "$CLEAN" ] && exit 0

  # Summarize through the local gateway. If it is down, fall back to the first
  # couple of sentences rather than going silent: a degraded voice beats none.
  SUMMARY=$(curl -s --max-time 20 "$GATEWAY/v1/chat/completions" \
      -H 'Content-Type: application/json' -H 'Authorization: Bearer sk-local' \
      -d "$(jq -n --arg m "$SUMMARY_MODEL" --arg c "$CLEAN" --argjson n "$SUMMARY_MAX_TOKENS" '{
            model:$m, max_tokens:$n, temperature:0.2,
            messages:[
              {role:"system",content:"Condense the assistant message into at most two spoken sentences. Say what happened or what was concluded. Plain speech, no markdown, no lists, no code, no file paths. If it ends by asking the user something, make the question the last sentence."},
              {role:"user",content:$c}]}')" \
    | jq -r '.choices[0].message.content // empty' 2>/dev/null)

  if [ -z "$SUMMARY" ]; then
    SUMMARY=$(python3 -c 'import re,sys; t=sys.stdin.read(); print(" ".join(re.split(r"(?<=[.!?]) ", t)[:2])[:400])' <<<"$CLEAN")
  fi
  [ -z "$SUMMARY" ] && exit 0

  WAV="$VOICE_RUN/say-$$.wav"
  curl -s --max-time 60 "$TTS_HOST/v1/audio/speech" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --arg m "$TTS_MODEL" --arg v "$TTS_VOICE" --arg i "$SUMMARY" --argjson s "$TTS_SPEED" \
          '{model:$m,voice:$v,input:$i,response_format:"wav",speed:$s}')" \
    -o "$WAV" 2>/dev/null

  if [ -s "$WAV" ]; then
    afplay "$WAV" & echo $! > "$VOICE_RUN/play.pid"
    wait $! 2>/dev/null
  fi
  rm -f "$WAV"
) >/dev/null 2>&1 &

exit 0
