# Shared voice settings. Sourced by speak.sh and listen.sh.

# TTS: an OpenAI-compatible /v1/audio/speech endpoint. Kokoro today. Kept as an
# API rather than a library call for the same reason the LLM gateway is: the
# engine behind it should be swappable without touching callers.
KOKORO_HOST="${KOKORO_HOST:-http://127.0.0.1:8880}"
KOKORO_VOICE="${KOKORO_VOICE:-am_echo}"
KOKORO_SPEED="${KOKORO_SPEED:-1.15}"

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
