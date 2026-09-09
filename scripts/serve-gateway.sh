#!/usr/bin/env bash
# LiteLLM gateway. The only address text clients ever learn.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
source scripts/versions.sh
LH_LOGS="${LOCALHARNESS_HOME:-$HOME/localharness}/logs"
mkdir -p "$LH_LOGS"

# Which config, so one launcher serves both machines: the Mac names
# mlx-community weights, gateway/config.cuda.yaml names GGUF ones.
#
#   GATEWAY_CONFIG=gateway/config.cuda.yaml ./scripts/serve-gateway.sh
#
# Opt out of LiteLLM's Responses API adapter. Without it, POST /v1/messages is
# routed to POST /v1/responses upstream, which mlx_lm.server does not implement,
# and every Anthropic-shaped request 404s.
#
# Also set as a litellm_settings: key in gateway/config.yaml. Both work
# independently (controlled A/B, docs/validation-log.md). Belt and braces,
# because the failure mode is a confusing 404 rather than a clear error.
export LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES=1

# Binds every interface by default. Deliberate: this is a house LAN, the models
# are local, and the point of the machine is that other machines on it can use
# the GPU. It is also the shape the M5 Studio needs, with the Studio serving and
# the mini as a client. There is NO AUTHENTICATION -- set GATEWAY_HOST=127.0.0.1 on an
# untrusted network.
#
# --host is passed explicitly regardless. LiteLLM's own default is 0.0.0.0, so
# omitting the flag would make the binding invisible: every doc in this repo
# once said "127.0.0.1:4000" while lsof said "*:4000". State it, whichever it is.
exec uv run --python 3.12 --with "$LITELLM_PIN" \
  litellm --config "${GATEWAY_CONFIG:-gateway/config.yaml}" \
  --host "${GATEWAY_HOST:-0.0.0.0}" --port "${GATEWAY_PORT:-4000}"
