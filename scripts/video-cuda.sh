#!/usr/bin/env bash
# The video lane's generator on a machine with an NVIDIA card.
#
# harness/engines.py builds the argv; this resolves the environment. Shares the
# generators' venv with image-cuda.sh, since both want the same torch and the
# same diffusers.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
source scripts/versions.sh
source scripts/diffusers-venv.sh

exec "$PY" -m harness.video_cuda "$@"
