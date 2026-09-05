#!/usr/bin/env bash
# Rewrites the STT model id so voicemode's hardcoded "whisper-1" reaches
# Parakeet on mlx_audio.server. See voice/stt_shim.py for why.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p .logs
exec uv run --no-project --with fastapi --with uvicorn --with httpx \
  --with python-multipart \
  uvicorn --app-dir voice stt_shim:app --host 127.0.0.1 --port "${STT_SHIM_PORT:-8083}"
