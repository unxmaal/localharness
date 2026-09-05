#!/usr/bin/env bash
# Free unified memory before an h3.c generation run.
#
# The M2 Pro reports a 25.0 GiB recommendedMaxWorkingSetSize (tools/h3probe),
# and h3 needs the text-encoder phase to fit inside it. Whatever the rest of the
# system is holding comes off that budget, so this stops the big consumers.
#
# Reversible: running state is recorded to .logs/quiesce-state.json and
# scripts/unquiesce.sh puts it back. GUI apps are NOT touched unless --apps is
# passed, because closing someone's Messages window is not a thing a script
# should decide on its own.
#
# Usage: scripts/quiesce.sh [--apps] [--dry-run]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
mkdir -p .logs
STATE=.logs/quiesce-state.json
APPS=0; DRY=0
for a in "$@"; do
  case "$a" in
    --apps) APPS=1 ;;
    --dry-run) DRY=1 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done
run() { if [ "$DRY" = 1 ]; then echo "  DRY: $*"; else "$@" >/dev/null 2>&1; fi; }

mem() {  # free + inactive pages, in GiB
  vm_stat | awk '
    /page size of/ {ps=$8}
    /Pages free/ {f=$3}
    /Pages inactive/ {i=$3}
    /Pages speculative/ {s=$3}
    END {gsub(/\./,"",f); gsub(/\./,"",i); gsub(/\./,"",s);
         printf "%.1f", (f+i+s)*ps/1024/1024/1024}'
}

echo "available before: $(mem) GiB"
echo ""

# --- localharness servers (they hold MLX model weights resident) ---
echo "stopping localharness servers"
run pkill -f mlx_lm.server
run pkill -f "litellm --config"

# --- docker containers ---
CONTAINERS=$(docker ps --format '{{.Names}}' 2>/dev/null | paste -sd, -)
if [ -n "$CONTAINERS" ]; then
  echo "stopping $(echo "$CONTAINERS" | tr ',' '\n' | wc -l | tr -d ' ') docker containers"
  if [ "$DRY" = 1 ]; then
    echo "  DRY: docker stop $(echo "$CONTAINERS" | tr ',' ' ')"
  else
    printf '{"containers":"%s","docker_desktop":%s,"apps":%s}\n' \
      "$CONTAINERS" \
      "$(pgrep -q -f 'Docker.app/Contents/MacOS/Docker' && echo true || echo false)" \
      "$APPS" > "$STATE"
    # shellcheck disable=SC2046  # word splitting is the point: a name list
    docker stop $(echo "$CONTAINERS" | tr ',' ' ') >/dev/null 2>&1
  fi
else
  echo "no docker containers running"
fi

# --- Docker Desktop itself: the VM reserves 8 GiB (settings-store MemoryMiB) ---
if pgrep -q -f 'Docker.app/Contents/MacOS/Docker'; then
  echo "quitting Docker Desktop (VM reserves 8 GiB)"
  run osascript -e 'quit app "Docker"'
fi

# --- lima VMs ---
if command -v limactl >/dev/null 2>&1; then
  for vm in $(limactl list --format '{{.Name}} {{.Status}}' 2>/dev/null | awk '$2=="Running"{print $1}'); do
    echo "stopping lima VM: $vm"
    run limactl stop "$vm"
  done
fi

# --- optional GUI apps ---
if [ "$APPS" = 1 ]; then
  for app in Spotify WhatsApp Vivaldi Messages; do
    if pgrep -qi "$app"; then echo "quitting $app"; run osascript -e "quit app \"$app\""; fi
  done
else
  echo ""
  echo "not touching GUI apps (pass --apps to also quit Spotify/WhatsApp/Vivaldi/Messages)"
  ps -Ao rss,comm -r 2>/dev/null | awk 'NR>1 && $1>60000 {printf "  holding %6.0f MB  %s\n", $1/1024, $2}' | head -8
fi

if [ "$DRY" = 0 ]; then
  echo ""
  echo "settling..."
  sleep 8
  echo "available after:  $(mem) GiB"
  echo ""
  echo "restore with: scripts/unquiesce.sh"
fi
