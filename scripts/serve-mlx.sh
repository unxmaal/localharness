#!/usr/bin/env bash
# MLX inference engine. Serves any model in the local HF cache; the gateway
# decides which one by the `model` field it sends.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .logs
BOOT_MODEL="${BOOT_MODEL:-mlx-community/Qwen2.5-1.5B-Instruct-4bit}"
PORT="${MLX_PORT:-8081}"
exec uv run mlx_lm.server \
  --model "$BOOT_MODEL" \
  --host 127.0.0.1 \
  --port "$PORT"
