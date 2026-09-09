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

# UTF-8 REGARDLESS OF THE MACHINE'S CODEPAGE. Python picks its stdio encoding
# from the locale, which is cp1252 on a stock Windows install, and anything
# printing a character outside it dies. LiteLLM's startup banner does exactly
# that, so the gateway exited during startup with a UnicodeEncodeError while
# every one of its own settings was correct. Only visible when output is
# redirected to a file, which is how a service runs.
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

exec "$PY" -m harness.image_cuda "$@"
