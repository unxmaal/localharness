#!/usr/bin/env bash
# Kokoro TTS and Parakeet STT on the GPU via mlx-audio, speaking an
# OpenAI-compatible /v1/audio/speech and /v1/audio/transcriptions.
#
# Port 8890 is deliberate: voicemode's provider_discovery.py classifies that
# port as "mlx-audio" and then stops sending it "whisper-1", which is the whole
# reason the old stt_shim existed. Native detection replaces 120 lines of proxy.
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
  python -m mlx_audio.server --host 127.0.0.1 --port "${TTS_PORT:-8890}"
