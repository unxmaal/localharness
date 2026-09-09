# shellcheck shell=bash
# The environment the CUDA generators run in. Sourced by image-cuda.sh and
# video-cuda.sh; never run directly. Sets $PY to the interpreter to use.
#
# ONE venv for both lanes. They want the same torch and the same diffusers, and
# CUDA torch is 2.5 GB: building it twice would cost five gigabytes to run two
# scripts that import the same module. It is separate from the project's venv
# for the reason mflux is a `uv tool` on the Mac -- the test suite has no need
# of a CUDA torch, and `uv sync` should not drag one in.
#
# The two indexes cannot be collapsed. torch+cu124 is published only on
# PyTorch's own index and diffusers only on PyPI, so pointing a single
# --index-url at both resolves the CPU build, and the lane then runs in minutes
# where the card takes seconds.

DIFFUSERS_VENV="${DIFFUSERS_VENV:-${LOCALHARNESS_HOME:-$HOME/localharness}/venvs/diffusers-cuda}"

_diffusers_python() {
  if [ -x "$DIFFUSERS_VENV/Scripts/python.exe" ]; then
    printf '%s\n' "$DIFFUSERS_VENV/Scripts/python.exe"
  elif [ -x "$DIFFUSERS_VENV/bin/python" ]; then
    printf '%s\n' "$DIFFUSERS_VENV/bin/python"
  fi
}

PY="$(_diffusers_python)"
if [ -z "$PY" ]; then
  echo "diffusers: building the generators' environment in $DIFFUSERS_VENV" >&2
  uv venv --python 3.12 "$DIFFUSERS_VENV" >&2
  VIRTUAL_ENV="$DIFFUSERS_VENV" uv pip install \
    --index-url "$TORCH_CUDA_INDEX" "$TORCH_CUDA_PIN" >&2
  VIRTUAL_ENV="$DIFFUSERS_VENV" uv pip install "$DIFFUSERS_PIN" \
    "$TRANSFORMERS_PIN" "$ACCELERATE_PIN" "$SAFETENSORS_PIN" "$PILLOW_PIN" >&2
  PY="$(_diffusers_python)"
fi
if [ -z "$PY" ]; then
  echo "FATAL: no interpreter in $DIFFUSERS_VENV after building it" >&2
  exit 1
fi

# The generators live in harness/, so the checkout has to be importable. They
# import nothing from this project beyond themselves.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
