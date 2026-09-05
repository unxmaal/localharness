#!/usr/bin/env bash
# Claude Code `Stop` hook: speak a short summary of the response just finished.
#
# Registered in ~/.claude/settings.json under hooks.Stop. Receives the hook
# payload as JSON on stdin. Uses `last_assistant_message` rather than reading
# transcript_path, which the docs note can lag behind.
#
# Everything slow happens in a detached child so the hook returns immediately
# and never delays the turn. Measured at 0.024s to return.
set -uo pipefail
source "$(dirname "$0")/config.sh"

INPUT=$(cat)

# stop_hook_active guards against a Stop hook that triggers another turn, which
# would fire Stop again. This hook only plays audio and cannot cause a turn, but
# the guard costs nothing and keeps that true if the script grows.
[ "$(jq -r '.stop_hook_active // false' <<<"$INPUT" 2>/dev/null)" = "true" ] && exit 0
[ -f "$VOICE_MUTE_FLAG" ] && exit 0

MSG=$(jq -r '.last_assistant_message // empty' <<<"$INPUT" 2>/dev/null)
[ -z "$MSG" ] && exit 0

(
  # Only one voice at a time: a new response cuts off the previous one rather
  # than queueing behind it, which is what you want in conversation.
  [ -f "$VOICE_RUN/play.pid" ] && kill "$(cat "$VOICE_RUN/play.pid")" 2>/dev/null
  pkill -f 'afplay .*claude-voice' 2>/dev/null

  # Mechanical strip first, so the summarizer spends its context on prose rather
  # than on code it is only going to discard.
  PARSED=$(printf '%s' "$MSG" | python3 "$VOICE_DIR/strip.py")
  CLEAN=$(jq -r '.body // empty' <<<"$PARSED")
  QUESTION=$(jq -r '.question // empty' <<<"$PARSED")
  # A message that is nothing but a question still gets spoken: the question is
  # reattached below, so an empty body is not an empty utterance.
  [ -z "$CLEAN" ] && [ -z "$QUESTION" ] && exit 0

  # Summarize through the local gateway. If it is down, fall back to the first
  # couple of sentences rather than going silent: a degraded voice beats none.
  REQ=$(jq -n --arg m "$SUMMARY_MODEL" --arg c "$CLEAN" --arg sys "$SUMMARY_SYSTEM" \
             --argjson n "$SUMMARY_MAX_TOKENS" '{
          model:$m, max_tokens:$n, temperature:0.1,
          messages:[{role:"system",content:$sys},
                    {role:"user",content:("TRANSCRIPT:\n" + $c)}]}')

  if [ -n "$CLEAN" ]; then
  SUMMARY=$(curl -s --max-time 20 "$GATEWAY/v1/chat/completions" \
              -H 'Content-Type: application/json' -H 'Authorization: Bearer sk-local' \
              -d "$REQ" | jq -r '.choices[0].message.content // empty' 2>/dev/null)

  fi
  if [ -z "$SUMMARY" ] && [ -n "$CLEAN" ]; then
    SUMMARY=$(printf '%s' "$CLEAN" | python3 -c \
      'import re,sys; t=sys.stdin.read(); print(" ".join(re.split(r"(?<=[.!?]) ", t)[:2])[:400])')
  fi
  # Reattach the closing question verbatim. It never went to the model, so it
  # cannot have been answered, inverted, or turned into a commitment.
  if [ -n "$QUESTION" ]; then
    SUMMARY=$(printf '%s %s' "${SUMMARY%%[[:space:]]}" "$QUESTION" | sed 's/^ *//')
  fi
  [ -z "$SUMMARY" ] && exit 0

  # Record what was actually spoken. Without this there is no way to tell a
  # summarizer that said something wrong from one that never ran.
  printf '%s  %s\n' "$(date +%H:%M:%S)" "$SUMMARY" >> "$VOICE_RUN/spoken.log"

  WAV="$VOICE_RUN/say-$$.wav"
  # mlx_audio.server returns HTTP 200 with an EMPTY BODY when synthesis fails,
  # so the file size is the only reliable success check.
  curl -s --max-time 60 "$TTS_HOST/v1/audio/speech" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --arg m "$TTS_MODEL" --arg v "$TTS_VOICE" --arg i "$SUMMARY" \
          --argjson s "$TTS_SPEED" \
          '{model:$m,voice:$v,input:$i,response_format:"wav",speed:$s}')" \
    -o "$WAV" 2>/dev/null

  if [ -s "$WAV" ]; then
    afplay "$WAV" & echo $! > "$VOICE_RUN/play.pid"
    wait $! 2>/dev/null
  fi
  rm -f "$WAV"
) >/dev/null 2>&1 &

exit 0
