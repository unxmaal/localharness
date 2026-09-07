#!/usr/bin/env bash
# Kokoro TTS and Parakeet STT on the GPU via mlx-audio, speaking an
# OpenAI-compatible /v1/audio/speech and /v1/audio/transcriptions.
#
# Port 8890 is deliberate: voicemode's provider_discovery.py classifies that
# port as "mlx-audio" and then stops sending it "whisper-1", which is the whole
# reason the old stt_shim existed. Native detection replaces 120 lines of proxy.
# Binds every interface by default. Deliberate: this is a house LAN, the models
# are local, and the point of the machine is that other machines on it can use
# the GPU. It is also the shape the M5 Studio needs, with the Studio serving and
# the mini as a client. There is NO AUTHENTICATION -- set TTS_HOST=127.0.0.1 on an
# untrusted network.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
source scripts/versions.sh
LH_LOGS="${LOCALHARNESS_HOME:-$HOME/localharness}/logs"
mkdir -p "$LH_LOGS"
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
  --with "$MLX_AUDIO_PIN" --with "$UVICORN_PIN" --with "$WEBRTCVAD_PIN" \
  --with "$FASTAPI_PIN" --with "$MULTIPART_PIN" --with "$SETUPTOOLS_PIN" \
  --with "$MISAKI_PIN" \
  python -m mlx_audio.server \
  --host "${TTS_HOST:-0.0.0.0}" --port "${TTS_PORT:-8890}"
