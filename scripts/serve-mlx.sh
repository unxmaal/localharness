#!/usr/bin/env bash
# MLX inference engine. Serves any model in the local HF cache; the gateway
# decides which one by the `model` field it sends (mlx_lm.server hot-swaps
# per request, see PLAN.md).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p .logs
exec uv run mlx_lm.server \
  --model "${BOOT_MODEL:-mlx-community/Qwen2.5-1.5B-Instruct-4bit}" \
  --host 127.0.0.1 \
  --port "${MLX_PORT:-8081}"
