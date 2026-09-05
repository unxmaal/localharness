#!/usr/bin/env bash
# LiteLLM gateway. The only address text clients ever learn.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p .logs

# Opt out of LiteLLM's Responses API adapter. Without it, POST /v1/messages is
# routed to POST /v1/responses upstream, which mlx_lm.server does not implement,
# and every Anthropic-shaped request 404s.
#
# Also set as a litellm_settings: key in gateway/config.yaml. Both work
# independently (controlled A/B, docs/validation-log.md). Belt and braces,
# because the failure mode is a confusing 404 rather than a clear error.
export LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES=1

exec uv run --python 3.12 --with 'litellm[proxy]' \
  litellm --config gateway/config.yaml --port "${GATEWAY_PORT:-4000}"
