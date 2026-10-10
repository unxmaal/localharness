#!/usr/bin/env bash
# Asserts the behaviours the architecture depends on. Run after any
# mlx-lm or litellm upgrade. Non-zero exit means the seam broke.
set -uo pipefail
# SMOKE_HOST so this can be run against another machine: when the Studio serves
# and the Apple Silicon machine is a client, the seam to check is the Studio's.
H="${SMOKE_HOST:-127.0.0.1}"
G="http://$H:${GATEWAY_PORT:-4000}"
E="http://$H:${MLX_PORT:-8081}"
FAIL=0
# The gateway demands its master key (#482). From another machine, set SOHOT_GATEWAY_KEY.
KEY="${SOHOT_GATEWAY_KEY:-$(uv run python -m harness.gateway_key show 2>/dev/null || true)}"
if [ -z "$KEY" ]; then
  echo "FAIL  no gateway key: run \`soh gateway key\` on the gateway's machine and set SOHOT_GATEWAY_KEY"
  exit 1
fi
# Ask by model id, never a nickname (#670): this machine's build of each.
build(){ uv run python -c "from harness.models import build_here; print(build_here('$1'))" 2>/dev/null || echo "$1"; }
SMALL="${SMOKE_SMALL:-$(build mlx-community/Qwen2.5-0.5B-Instruct-4bit)}"
MID="${SMOKE_MID:-$(build mlx-community/Qwen2.5-1.5B-Instruct-4bit)}"
ok(){ printf 'PASS  %s\n' "$1"; }
no(){ printf 'FAIL  %s\n' "$1"; FAIL=1; }

# LiteLLM's /health/readiness returns 200 before the proxy can actually serve.
# Gate on a real completion, or these assertions produce false failures and, worse,
# false attributions of a fix. See docs/validation-log.md.
#
# The delay is load-bearing: curl fails INSTANTLY on connection-refused, so an
# unpaced loop burns all its attempts inside the startup window and reports a
# ready service as dead.
READY_TIMEOUT="${READY_TIMEOUT:-120}"
printf 'wait  gateway accepting real requests'
ready=0
for i in $(seq 1 "$READY_TIMEOUT"); do
  if curl -sf --max-time 10 "$G/v1/chat/completions" -H 'Content-Type: application/json' \
       -H "Authorization: Bearer $KEY" \
       -d '{"model":"'"$SMALL"'","messages":[{"role":"user","content":"hi"}],"max_tokens":2}' \
       2>/dev/null | grep -q '"content"'; then
    printf ' ready after ~%ss\n' "$i"; ready=1; break
  fi
  printf '.'
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  printf '\nFAIL  gateway not serving after %ss\n' "$READY_TIMEOUT"
  exit 1
fi

# The engine binds 127.0.0.1 behind the gateway (#482), so only a local run can ask it.
LOCAL=0
case "$H" in 127.0.0.1|localhost) LOCAL=1 ;; esac
if [ "$LOCAL" -eq 1 ]; then
  grep -q . <<<"$(curl -sf "$E/v1/models")" && ok "engine /v1/models" || no "engine /v1/models"
else
  printf 'SKIP  engine checks: the engine listens on the serving machine only\n'
fi

curl -sf "$G/v1/chat/completions" -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $KEY" \
  -d '{"model":"'"$SMALL"'","messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":10}' \
  | grep -q '"content"' && ok "gateway openai chat" || no "gateway openai chat"

# The reply names the model that answered, not the alias asked for (#670).
curl -sf -D - -o /dev/null "$G/v1/chat/completions" -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $KEY" \
  -d '{"model":"sohot-extract","messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":5}' \
  | grep -qi '^x-sohot-model: ' && ok "gateway names the served model" || no "gateway names the served model"

curl -sfN "$G/v1/chat/completions" -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $KEY" \
  -d '{"model":"'"$SMALL"'","messages":[{"role":"user","content":"count 1 2 3"}],"max_tokens":20,"stream":true}' \
  | grep -q 'data: \[DONE\]' && ok "gateway sse streaming" || no "gateway sse streaming"

# The regression this exists to catch.
curl -sf "$G/v1/messages" -H 'Content-Type: application/json' \
  -H "x-api-key: $KEY" -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"'"$SMALL"'","max_tokens":20,"messages":[{"role":"user","content":"Reply with exactly: OK"}]}' \
  | grep -q '"type":"message"' && ok "gateway anthropic /v1/messages" || no "gateway anthropic /v1/messages"

curl -sf "$G/v1/chat/completions" -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $KEY" \
  -d '{"model":"'"$MID"'","messages":[{"role":"user","content":"Weather in Paris? Use the tool."}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":80}' \
  | grep -q '"tool_calls"' && ok "openai tool calling" || no "openai tool calling"

curl -sf "$G/v1/messages" -H 'Content-Type: application/json' \
  -H "x-api-key: $KEY" -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"'"$MID"'","max_tokens":80,"messages":[{"role":"user","content":"Weather in Paris? Use the tool."}],"tools":[{"name":"get_weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]}' \
  | grep -q '"type": *"tool_use"' && ok "anthropic tool calling" || no "anthropic tool calling"

# The decide lane's requests carry a schema, so its alias must take one; a 200 without it proves nothing. #572.
curl -sf --max-time 300 "$G/v1/chat/completions" -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $KEY" \
  -d '{"model":"sohot-decide","messages":[{"role":"user","content":"Is the sky blue on a clear day? A: yes, B: no. Reply as JSON."}],"max_tokens":20,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"response_format":{"type":"json_schema","json_schema":{"name":"decide","strict":true,"schema":{"type":"object","properties":{"answer":{"type":"string","enum":["A","B"]}},"required":["answer"],"additionalProperties":false}}}}' \
  | grep -q 'answer' && ok "sohot-decide answers under its schema" || no "sohot-decide answers under its schema"

# The claims lane sends its tuple schema too. #654.
curl -sf --max-time 300 "$G/v1/chat/completions" -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $KEY" \
  -d '{"model":"sohot-claims","messages":[{"role":"user","content":"[1] member-a: The fan runs at 12V."}],"max_tokens":100,"temperature":0,"response_format":{"type":"json_schema","json_schema":{"name":"claims","strict":true,"schema":{"type":"object","properties":{"c":{"type":"array","items":{"type":"array","prefixItems":[{"type":"string"},{"type":"string","maxLength":220},{"type":"array","items":{"type":"integer"},"minItems":1}],"minItems":3,"maxItems":3}}},"required":["c"],"additionalProperties":false}}}}' \
  | grep -q '\\"c\\"' && ok "sohot-claims answers under its schema" || no "sohot-claims answers under its schema"

# An alias with no weights behind it must FAIL rather than be answered with
# whatever is loaded. gateway/config.cuda.yaml names GGUF files by filename
# stem, so a typo there would measure the resident model under another
# candidate's name and report the two as a tie, with nothing saying so.
# llama-server answers 400 "model not found"; mlx_lm.server cannot fetch an
# unknown repo under HF_HUB_OFFLINE and fails as well.
if [ "$LOCAL" -eq 1 ]; then
  missing="$(curl -s --max-time 30 "$E/v1/chat/completions"   -H 'Content-Type: application/json'   -d '{"model":"localharness-no-such-model","messages":[{"role":"user","content":"hi"}],"max_tokens":2}' 2>/dev/null)"
  if printf '%s' "$missing" | grep -q '"content"'; then
    no "engine answered for a model that does not exist"
  else
    ok "an unknown model is refused rather than served"
  fi
fi

# ---- audio ------------------------------------------------------------------
#
# Half the goal, and until now nothing guarded it the way the text seam is
# guarded. Both round trips are sub-second, so there is no reason to skip them.
#
# The status code is NOT sufficient: mlx_audio answers 200 with an EMPTY BODY
# when misaki is missing, and it answers 200 then closes mid-stream when the
# requested voice is not in the local cache. Assert on the bytes.
A="http://$H:${TTS_PORT:-8890}/v1"

# WHICH MODEL TO ASK FOR IS A PROPERTY OF THE MACHINE. The Mac transcribes with
# parakeet under MLX; the CUDA box runs faster-whisper and refuses a parakeet
# request rather than answering it with Whisper under another name. Reading the
# defaults from harness/audio.py keeps one source of truth: hardcoding them
# here made this check fail on the machine whose lane was working.
SMOKE_TTS_MODEL="${SMOKE_TTS_MODEL:-$(uv run python -c   'from harness import audio; print(audio.DEFAULT_TTS_MODEL)' 2>/dev/null || echo mlx-community/Kokoro-82M-bf16)}"
SMOKE_STT_MODEL="${SMOKE_STT_MODEL:-$(uv run python -c   'from harness import audio; print(audio.DEFAULT_STT_MODEL)' 2>/dev/null || echo mlx-community/parakeet-tdt-0.6b-v2)}"
# A full template, not `-t prefix`: BSD mktemp invents the suffix and GNU
# refuses without XXXXXX, so the bare form fails on Git Bash, prints "too
# few X's" and leaves $WAV as ".wav" in the working directory.
WAV="$(mktemp "${TMPDIR:-/tmp}/smoke-tts.XXXXXX")"
mv "$WAV" "$WAV.wav" && WAV="$WAV.wav"
trap 'rm -f "$WAV"' EXIT

if curl -sf --max-time 60 "$A/audio/speech" -H 'Content-Type: application/json' \
     -d "{\"model\":\"$SMOKE_TTS_MODEL\",\"input\":\"the gateway is up\",\"voice\":\"${TTS_VOICE:-bm_george}\",\"response_format\":\"wav\"}" \
     -o "$WAV" 2>/dev/null; then
  # 8000 bytes is a fifth of a second at 16k mono. A bare 44-byte WAV header
  # passes `test -s` and plays as silence.
  BYTES=$(wc -c < "$WAV" | tr -d ' ')
  if [ "$BYTES" -ge 8000 ]; then
    ok "tts speech ($BYTES bytes)"
  else
    no "tts speech returned $BYTES bytes; check the server log for an ImportError or a missing voice"
  fi
else
  no "tts speech"
fi

if [ -s "$WAV" ]; then
  curl -sf --max-time 60 "$A/audio/transcriptions" \
    -F "file=@$WAV" -F "model=$SMOKE_STT_MODEL" \
    | grep -qi 'gateway' && ok "stt transcription round trip" \
    || no "stt transcription round trip"
else
  no "stt transcription round trip (no audio to transcribe)"
fi

exit $FAIL
