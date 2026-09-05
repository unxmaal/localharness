#!/usr/bin/env bash
# Kokoro TTS on the GPU via mlx-audio. Speaks an OpenAI-compatible
# /v1/audio/speech, so voice/speak.sh talks to an endpoint, not a library.
#
# Port 8083 keeps clear of the MLX LLM engine (8081) and the gateway (4000).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p .logs
exec uv run --no-project --with mlx-audio \
  python -m mlx_audio.server --host 127.0.0.1 --port "${TTS_PORT:-8083}"
