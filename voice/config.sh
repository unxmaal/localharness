# Shared voice settings. Sourced by speak.sh and listen.sh.

# TTS: mlx_audio.server speaks an OpenAI-compatible /v1/audio/speech, running
# Kokoro-82M on the GPU via MLX. No docker.
#
# Kept behind an HTTP API rather than an in-process library call for the same
# reason the LLM gateway is: a persistent server holds the model resident, so a
# spoken reply does not pay a cold model load every turn, and the engine behind
# the endpoint stays swappable.
TTS_HOST="${TTS_HOST:-http://127.0.0.1:8083}"
TTS_MODEL="${TTS_MODEL:-mlx-community/Kokoro-82M-bf16}"
TTS_VOICE="${TTS_VOICE:-am_adam}"
TTS_SPEED="${TTS_SPEED:-1.15}"

# Summarizer: the local gateway from this repo. Long responses are condensed to
# a couple of spoken sentences; reading a response with code blocks and tables
# aloud verbatim is unusable.
GATEWAY="${GATEWAY:-http://127.0.0.1:4000}"
SUMMARY_MODEL="${SUMMARY_MODEL:-local-mid}"
SUMMARY_MAX_TOKENS="${SUMMARY_MAX_TOKENS:-90}"

# STT
PARAKEET_MODEL="${PARAKEET_MODEL:-mlx-community/parakeet-tdt-0.6b-v2}"

# Kill switch. `touch` this file to go quiet without editing settings.json.
VOICE_MUTE_FLAG="${VOICE_MUTE_FLAG:-$HOME/.claude/voice-mute}"

VOICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICE_RUN="${VOICE_RUN:-${TMPDIR:-/tmp}/claude-voice}"
mkdir -p "$VOICE_RUN"
