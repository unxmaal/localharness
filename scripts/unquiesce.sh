#!/usr/bin/env bash
# Undo scripts/quiesce.sh using the state it recorded.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
STATE=.logs/quiesce-state.json
[ -f "$STATE" ] || { echo "no $STATE; nothing recorded to restore" >&2; exit 1; }

WANT_DOCKER=$(sed -n 's/.*"docker_desktop":\([a-z]*\).*/\1/p' "$STATE")
CONTAINERS=$(sed -n 's/.*"containers":"\([^"]*\)".*/\1/p' "$STATE")

if [ "$WANT_DOCKER" = "true" ]; then
  echo "starting Docker Desktop"
  open -a Docker
  printf 'waiting for docker daemon'
  for i in $(seq 1 120); do
    docker info >/dev/null 2>&1 && { echo " up after ${i}s"; break; }
    printf '.'; sleep 1
  done
fi

if [ -n "$CONTAINERS" ]; then
  echo "starting containers"
  # shellcheck disable=SC2046  # word splitting is the point: a name list
  docker start $(echo "$CONTAINERS" | tr ',' ' ') 2>&1 | sed 's/^/  /' 
fi

echo ""
echo "localharness servers are NOT auto-restarted; run:"
echo "  ./scripts/serve-mlx.sh &  ./scripts/serve-gateway.sh &"
