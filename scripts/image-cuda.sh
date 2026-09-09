#!/usr/bin/env bash
# The image lane's generator on a machine with an NVIDIA card.
#
# harness/engines.py builds the argv; this resolves the environment it runs in.
# Its own venv rather than the project's, for the reason mflux is a `uv tool`
# on the Mac: CUDA torch is 2.5 GB and has no business in a venv the test suite
# creates. Built once, reused after.
#
# The two indexes cannot be collapsed. torch+cu124 exists only on PyTorch's own
# index and diffusers exists only on PyPI, so they are installed in two steps
# rather than by pointing one --index-url at both and getting the CPU build.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
source scripts/versions.sh

VENV="${IMAGE_CUDA_VENV:-${LOCALHARNESS_HOME:-$HOME/localharness}/venvs/image-cuda}"
PY="$VENV/Scripts/python.exe"
[ -x "$PY" ] || PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
  echo "image-cuda: building the generator's environment in $VENV" >&2
  uv venv --python 3.12 "$VENV" >&2
  VIRTUAL_ENV="$VENV" uv pip install --index-url "$TORCH_CUDA_INDEX" \
    "$TORCH_CUDA_PIN" >&2
  VIRTUAL_ENV="$VENV" uv pip install "$DIFFUSERS_PIN" "$TRANSFORMERS_PIN" \
    "$ACCELERATE_PIN" "$SAFETENSORS_PIN" "$PILLOW_PIN" >&2
  PY="$VENV/Scripts/python.exe"
  [ -x "$PY" ] || PY="$VENV/bin/python"
fi

# The generator imports nothing from this project beyond itself, but it lives
# in harness/, so the checkout has to be importable.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m harness.image_cuda "$@"
