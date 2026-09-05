# shellcheck shell=bash
# Shared environment. Sourced by the launchers; never run directly.
#
# Model weights must not land on the internal disk. It runs at 91% and a single
# 70B at 4-bit is ~40GB, while the MiniMax-H3 checkpoint alone is 134GB. Left to
# itself, huggingface_hub silently recreates its cache under $HOME and fills it.
#
# Candidate order is by MEASURED throughput (4 GiB dd, cold read forced by
# unmount/remount so the page cache cannot flatter the number):
#
#   /Volumes/Models  DockCase C1P, ioreg Speed=4 (SuperSpeed+, 10 Gbps), direct
#                    to a host controller. 1013 MB/s write, 959 MB/s cold read.
#   /Volumes/T7      Samsung PSSD T7, ioreg Speed=3 (5 Gbps) because it sits
#                    behind a VIA Labs USB3.0 hub. 422/432 MB/s, 2.2x slower.
#                    The T7 is itself a 10 Gbps device; the hub halves it.
#
# Load time scales directly with this, and mlx_lm.server hot-swaps models per
# request, so the cost is paid on every switch: ~40GB is ~42s at 959 MB/s.
#
# Overrides, both validated the same way rather than trusted:
#   HF_ROOT        pin one location
#   HF_CANDIDATES  colon-separated search order (defaults below)

HF_CANDIDATES="${HF_CANDIDATES:-/Volumes/Models/hf:/Volumes/T7/hf}"

# The volume a path lives on, resolved via its nearest existing ancestor so an
# as-yet-uncreated target still answers. Empty if nothing resolves.
_hf_mountpoint() {
  local p="$1"
  while [ -n "$p" ] && [ ! -e "$p" ]; do
    local parent; parent="$(dirname "$p")"
    [ "$parent" = "$p" ] && break
    p="$parent"
  done
  [ -e "$p" ] || return 1
  df -P "$p" 2>/dev/null | awk 'NR==2 { for (i = 6; i <= NF; i++)
    printf "%s%s", $i, (i < NF ? " " : "") }'
}

# True if a path is on external storage and writable.
#
# Deliberately creates nothing: this is a predicate, and probing candidates
# should not litter empty directories on the volumes that lose the race.
_hf_usable() {
  local cand="$1" mp anchor
  mp="$(_hf_mountpoint "$cand")" || return 1
  [ -n "$mp" ] || return 1

  # The whole point. "/" and the /System/Volumes/* family are the internal
  # disk; anything landing there is the failure this guard exists to prevent.
  case "$mp" in
    /|/System/Volumes/*) return 1 ;;
  esac

  # Writability is checked on the nearest existing ancestor, since the target
  # itself may not exist yet.
  anchor="$cand"
  while [ -n "$anchor" ] && [ ! -e "$anchor" ]; do
    local parent; parent="$(dirname "$anchor")"
    [ "$parent" = "$anchor" ] && break
    anchor="$parent"
  done
  [ -w "$anchor" ] || return 1
  return 0
}

_hf_fatal() {
  echo "FATAL: $1" >&2
  echo "       Refusing to run: weights would land on the internal disk," >&2
  echo "       which is at 91% and cannot hold them." >&2
  return 1 2>/dev/null || exit 1
}

if [ -n "${HF_ROOT:-}" ]; then
  _hf_usable "$HF_ROOT" \
    || _hf_fatal "HF_ROOT=$HF_ROOT is not on writable external storage." \
    || return 1 2>/dev/null || exit 1
else
  HF_ROOT=""
  _hf_saved_ifs="$IFS"; IFS=":"
  for _cand in $HF_CANDIDATES; do
    if _hf_usable "$_cand"; then HF_ROOT="$_cand"; break; fi
  done
  IFS="$_hf_saved_ifs"; unset _hf_saved_ifs _cand
  [ -n "$HF_ROOT" ] \
    || _hf_fatal "no writable external volume found (tried $HF_CANDIDATES)." \
    || return 1 2>/dev/null || exit 1
fi

# Only now, once a location is chosen, do we create anything.
mkdir -p "$HF_ROOT" || _hf_fatal "cannot create $HF_ROOT" \
  || return 1 2>/dev/null || exit 1

export HF_HOME="$HF_ROOT"
echo "hf    HF_HOME=$HF_HOME" >&2
