#!/usr/bin/env bash
# MLX inference engine. Serves any model in the local HF cache; the gateway
# decides which one by the `model` field it sends (mlx_lm.server hot-swaps
# per request, see PLAN.md).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
LH_LOGS="${LOCALHARNESS_HOME:-$HOME/localharness}/logs"
mkdir -p "$LH_LOGS"
# Binds every interface by default. Deliberate: this is a trusted LAN, the models
# are local, and the point of the machine is that other machines on it can use
# the GPU. It is also the shape the M5 Studio needs, with the Studio serving and
# the Apple Silicon machine as a client. There is NO AUTHENTICATION -- set MLX_HOST=127.0.0.1 on an
# untrusted network.
exec uv run mlx_lm.server \
  --model "${BOOT_MODEL:-mlx-community/Qwen2.5-1.5B-Instruct-4bit}" \
  --host "${MLX_HOST:-0.0.0.0}" \
  --port "${MLX_PORT:-8081}"
