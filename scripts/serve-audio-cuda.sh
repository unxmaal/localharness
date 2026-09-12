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
# WHERE: beside the other weights, not beside the outputs. LOCALHARNESS_HOME
# is the artifact root -- out/, runs/, logs/ -- and these are ~350MB of model
# that every other weight in this project keeps under HF_ROOT, behind env.sh's
# writability and free-space guard. KOKORO_DIR overrides it.
KOKORO_DIR="${KOKORO_DIR:-${HF_ROOT:-${LOCALHARNESS_HOME:-$HOME/localharness}}/kokoro}"
_KOKORO_WAS="${LOCALHARNESS_HOME:-$HOME/localharness}/models/kokoro"
export KOKORO_MODEL="${KOKORO_MODEL:-$KOKORO_DIR/kokoro-v1.0.onnx}"
export KOKORO_VOICES="${KOKORO_VOICES:-$KOKORO_DIR/voices-v1.0.bin}"
if [ ! -f "$KOKORO_MODEL" ] || [ ! -f "$KOKORO_VOICES" ]; then
  echo "FATAL: Kokoro weights missing." >&2
  echo "       expected $KOKORO_MODEL" >&2
  echo "       and      $KOKORO_VOICES" >&2
  echo "       Set KOKORO_DIR to say where they are, or KOKORO_MODEL and" >&2
  echo "       KOKORO_VOICES individually." >&2
  if [ -f "$_KOKORO_WAS/kokoro-v1.0.onnx" ]; then
    echo >&2
    echo "       They ARE at $_KOKORO_WAS, which is where this looked before" >&2
    echo "       the weights moved under HF_ROOT. Move them, or set" >&2
    echo "       KOKORO_DIR=$_KOKORO_WAS." >&2
  fi
  echo >&2
  echo "       The two curl commands are in the comment above this check." >&2
  exit 1
fi

# Binds every interface by default, for the same reason as the other services:
# a trusted LAN, local models, and the point of the machine is that the Apple Silicon machine can
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
  # CONVERTED, NOT PREPENDED WHOLE. Python joins those directories with the
  # platform separator, which is a semicolon here, and bash splits PATH on
  # colons: prepending the list as it stands leaves PATH beginning with a bare
  # drive letter, after which nothing on it resolves at all. The symptom is
  # this script's own `exec uv` reporting "uv: not found" two lines later, on a
  # machine where uv is plainly on PATH. Same collision as HF_CANDIDATES in
  # scripts/env.sh.
  if [ -n "$NV_BIN" ]; then
    NV_POSIX=""
    while IFS= read -r _dir; do
      [ -n "$_dir" ] || continue
      NV_POSIX="$NV_POSIX$(cygpath -u "$_dir"):"
    done <<< "$(printf '%s' "$NV_BIN" | tr ';' '\n')"
    export PATH="$NV_POSIX$PATH"
  fi
else
  # THE SAME PROBLEM ON LINUX, UNDER A DIFFERENT NAME. CTranslate2 resolves
  # libcublas.so.12 and libcudnn_ops.so through ld.so, and the nvidia wheels
  # put them under site-packages/nvidia/*/lib, which ld.so does not search. The
  # symptom is the same: "Library libcublas.so.12 is not found" on a machine
  # that has it. No cygpath here -- one spelling of a path on this side.
  NV_LIB="$(uv run --no-project --with "$NVIDIA_CUBLAS_PIN" --with "$NVIDIA_CUDNN_PIN" \
    python -c "import nvidia, os, glob; print(os.pathsep.join(d for r in nvidia.__path__ for d in glob.glob(os.path.join(r, '*', 'lib')) if os.path.isdir(d)))" 2>/dev/null || true)"
  if [ -n "$NV_LIB" ]; then
    export LD_LIBRARY_PATH="$NV_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

# UTF-8 REGARDLESS OF THE MACHINE'S CODEPAGE. Python picks its stdio encoding
# from the locale, which is cp1252 on a stock Windows install, and anything
# printing a character outside it dies. LiteLLM's startup banner does exactly
# that, so the gateway exited during startup with a UnicodeEncodeError while
# every one of its own settings was correct. Only visible when output is
# redirected to a file, which is how a service runs.
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

exec uv run --no-project \
  --with "$FASTER_WHISPER_PIN" --with "$KOKORO_ONNX_PIN" \
  --with "$SOUNDFILE_PIN" --with "$NVIDIA_CUBLAS_PIN" --with "$NVIDIA_CUDNN_PIN" \
  --with "$FASTAPI_PIN" --with "$UVICORN_PIN" --with "$MULTIPART_PIN" \
  python -m harness.audio_server
