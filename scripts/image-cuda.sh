#!/usr/bin/env bash
# The image lane's generator on a machine with an NVIDIA card.
#
# harness/engines.py builds the argv; this resolves the environment it runs in.
# Shares the generators' venv with video-cuda.sh: both want the same torch and
# the same diffusers, and building CUDA torch twice costs five gigabytes to run
# two scripts that import one module. See scripts/diffusers-venv.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
source scripts/versions.sh

source scripts/diffusers-venv.sh

exec "$PY" -m harness.image_cuda "$@"
