#!/usr/bin/env bash
# The audio lanes on a machine with an NVIDIA card: harness/audio_server.py.
#
# The counterpart of serve-tts.sh, which runs mlx_audio and answers the same
# two endpoints. Nothing on PyPI answers both here, so this serves Kokoro
# through onnxruntime and faster-whisper through CTranslate2 behind one port.
# harness/audio.py does not know which of the two it is talking to.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
source scripts/versions.sh

LH_LOGS="${LOCALHARNESS_HOME:-$HOME/localharness}/logs"
mkdir -p "$LH_LOGS"

# Kokoro's weights are two files rather than a HuggingFace repo, so they are
# named rather than resolved. Fetch them once:
#   curl -L -o "$KOKORO_MODEL" https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
#   curl -L -o "$KOKORO_VOICES" https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
KOKORO_DIR="${KOKORO_DIR:-${LOCALHARNESS_HOME:-$HOME/localharness}/models/kokoro}"
export KOKORO_MODEL="${KOKORO_MODEL:-$KOKORO_DIR/kokoro-v1.0.onnx}"
export KOKORO_VOICES="${KOKORO_VOICES:-$KOKORO_DIR/voices-v1.0.bin}"
if [ ! -f "$KOKORO_MODEL" ] || [ ! -f "$KOKORO_VOICES" ]; then
  echo "FATAL: Kokoro weights missing." >&2
  echo "       expected $KOKORO_MODEL" >&2
  echo "       and      $KOKORO_VOICES" >&2
  echo "       The two curl commands are in the comment above this check." >&2
  exit 1
fi

# Binds every interface by default, for the same reason as the other services:
# a house LAN, local models, and the point of the machine is that the mini can
# use its GPU. There is NO AUTHENTICATION -- set AUDIO_HOST=127.0.0.1 on an
# untrusted network.
export AUDIO_HOST="${AUDIO_HOST:-0.0.0.0}"
export AUDIO_PORT="${AUDIO_PORT:-8890}"

# THE CUDA RUNTIME HAS TO BE ON PATH BEFORE THE PROCESS STARTS. CTranslate2
# resolves cublas64_12.dll through the OS search path, not through Python's
# loader, so os.add_dll_directory inside the server is too late and reports
# "Library cublas64_12.dll is not found" on a machine that has it installed.
# The nvidia wheels below carry the DLLs, so no system CUDA install is needed.
if [ "${OS:-}" = "Windows_NT" ]; then
  NV_BIN="$(uv run --no-project --with "$NVIDIA_CUBLAS_PIN" --with "$NVIDIA_CUDNN_PIN" \
    python -c "import nvidia, os, glob; print(os.pathsep.join(d for r in nvidia.__path__ for d in glob.glob(os.path.join(r, '*', 'bin')) if os.path.isdir(d)))" 2>/dev/null || true)"
  [ -n "$NV_BIN" ] && export PATH="$NV_BIN;$PATH"
fi

exec uv run --no-project \
  --with "$FASTER_WHISPER_PIN" --with "$KOKORO_ONNX_PIN" \
  --with "$SOUNDFILE_PIN" --with "$NVIDIA_CUBLAS_PIN" --with "$NVIDIA_CUDNN_PIN" \
  --with "$FASTAPI_PIN" --with "$UVICORN_PIN" --with "$MULTIPART_PIN" \
  python -m harness.audio_server
