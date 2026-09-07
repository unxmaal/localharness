#!/usr/bin/env bash
# localharness over MCP, so another machine's agent can use this one's GPU.
#
# Scoped to svg, web, code and image. Video needs job semantics past a queue and
# speech over the LAN was ruled out; both stay reachable locally through `lh`.
#
# Binds every interface by default, like the other services and for the same
# reason: this is a house LAN, the models are local, and the point of the
# machine is that other machines on it can use the GPU. There is NO
# AUTHENTICATION -- set MCP_HOST=127.0.0.1 on an untrusted network.
#
# It shells out to `lh`, so `lh` has to be installed:
#   uv tool install --python 3.12 --editable .
#
# The client entry, on the other machine:
#   claude mcp add --transport http localharness http://styx.local:8899/mcp
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
LH_LOGS="${LOCALHARNESS_HOME:-$HOME/localharness}/logs"
mkdir -p "$LH_LOGS"

command -v lh >/dev/null || {
  echo "lh is not on PATH. Install it: uv tool install --python 3.12 --editable ." >&2
  exit 1
}

HOST="${MCP_HOST:-0.0.0.0}"
PORT="${MCP_PORT:-8899}"
echo "mcp   http://$(scutil --get LocalHostName 2>/dev/null || hostname).local:$PORT/mcp" >&2
exec uv run --group mcp python -m harness.mcp_server \
  --host "$HOST" --port "$PORT" --transport streamable-http \
  ${MCP_ALLOW:+--allow "$MCP_ALLOW"}
