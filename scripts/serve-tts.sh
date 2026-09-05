#!/usr/bin/env bash
# Kokoro TTS on the GPU via mlx-audio. Speaks an OpenAI-compatible
# /v1/audio/speech, so voice/speak.sh talks to an endpoint, not a library.
#
# Port 8083 keeps clear of the MLX LLM engine (8081) and the gateway (4000).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p .logs
# mlx-audio does not declare its server dependencies, so mlx_audio.server dies
# with ModuleNotFoundError on a clean install. The full set, read off the import
# block at the top of mlx_audio/server.py rather than discovered one crash at a
# time: uvicorn, webrtcvad, fastapi (+ python-multipart for its form parsing).
#
# misaki is Kokoro's grapheme-to-phoneme frontend. Without it the endpoint
# returns HTTP 200 with an EMPTY BODY and logs the ImportError server-side, so
# callers must check that the audio file is non-empty rather than trusting the
# status code. voice/speak.sh does.
#
# The setuptools pin is RULE #143: webrtcvad still imports pkg_resources, uv does
# not install setuptools into venvs on py3.12+, and setuptools >=81 removed
# pkg_resources outright. Unpinned resolves to 84.x and fails the same way.
exec uv run --no-project \
  --with mlx-audio --with uvicorn --with webrtcvad \
  --with fastapi --with python-multipart --with 'setuptools>=70,<81' \
  --with 'misaki[en]' \
  python -m mlx_audio.server --host 127.0.0.1 --port "${TTS_PORT:-8085}"
