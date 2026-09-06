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
# The rule is FREE SPACE, not "is this the internal disk". The old rule refused
# "/" and /System/Volumes/* categorically, which encoded this particular
# machine: an M2 mini at 91% full. The M5 Ultra Studio has a large internal SSD
# and may have no external volume attached at all, and a categorical rule would
# refuse it on day one. Nothing about being internal is disqualifying; running
# out of room is, so the threshold says that directly. On THIS machine the
# threshold still refuses the internal disk, for the reason that actually
# matters.
#
# 160GB because MiniMax-H3's FL2VA checkpoint alone is 134GiB and
# fetch-h3-weights.sh demands that much headroom. A smaller threshold would let
# this guard pass and the download fail instead.
#
# Overrides, all validated the same way rather than trusted:
#   HF_ROOT         pin one location
#   HF_CANDIDATES   colon-separated search order (defaults below)
#   HF_MIN_FREE_GB  how much room a location must have

HF_MIN_FREE_GB="${HF_MIN_FREE_GB:-160}"
# Ordered by measured throughput, ending somewhere that exists on any Mac so a
# machine with no external volume still works without hand configuration.
HF_CANDIDATES="${HF_CANDIDATES:-/Volumes/Models/hf:/Volumes/T7/hf:$HOME/.cache/huggingface}"

# The volume a path lives on, resolved via its nearest existing ancestor so an
# as-yet-uncreated target still answers. Empty if nothing resolves.
_hf_mountpoint() {
  # NOTE: every local is declared HERE, never inside the loop. In zsh a bare
  # `local name` PRINTS the parameter once it has a value, so re-declaring it
  # each iteration emits "parent=/Volumes/..." on stdout -- and this function
  # is read through command substitution, so that noise becomes the answer.
  # bash and sh are silent, which is why it survived until a zsh test ran.
  local p="$1" parent
  while [ -n "$p" ] && [ ! -e "$p" ]; do
    parent="$(dirname "$p")"
    [ "$parent" = "$p" ] && break
    p="$parent"
  done
  [ -e "$p" ] || return 1
  df -P "$p" 2>/dev/null | awk 'NR==2 { for (i = 6; i <= NF; i++)
    printf "%s%s", $i, (i < NF ? " " : "") }'
}

# Free space in whole GB on the volume holding a path, via its nearest existing
# ancestor. df -g reports whole gigabytes, which is the resolution wanted here.
_hf_free_gb() {
  local p="$1" parent
  while [ -n "$p" ] && [ ! -e "$p" ]; do
    parent="$(dirname "$p")"
    [ "$parent" = "$p" ] && break
    p="$parent"
  done
  [ -e "$p" ] || return 1
  df -g "$p" 2>/dev/null | awk 'NR==2 { print $4 }'
}

# True if a path is writable and has room for the weights.
#
# Deliberately creates nothing: this is a predicate, and probing candidates
# should not litter empty directories on the volumes that lose the race.
_hf_usable() {
  local cand="$1" mp anchor free parent
  mp="$(_hf_mountpoint "$cand")" || return 1
  [ -n "$mp" ] || return 1

  free="$(_hf_free_gb "$cand")" || return 1
  [ -n "$free" ] || return 1
  [ "$free" -ge "$HF_MIN_FREE_GB" ] || return 1

  # Writability is checked on the nearest existing ancestor, since the target
  # itself may not exist yet.
  anchor="$cand"
  while [ -n "$anchor" ] && [ ! -e "$anchor" ]; do
    parent="$(dirname "$anchor")"
    [ "$parent" = "$anchor" ] && break
    anchor="$parent"
  done
  [ -w "$anchor" ] || return 1
  return 0
}

_hf_fatal() {
  echo "FATAL: $1" >&2
  echo "       A location needs ${HF_MIN_FREE_GB}GB free (HF_MIN_FREE_GB)." >&2
  echo "       Weights are large: MiniMax-H3 alone is 134GiB, and a 70B at" >&2
  echo "       4-bit is ~40GB. Attach a volume, or lower the threshold if you" >&2
  echo "       know what you are fetching." >&2
  return 1 2>/dev/null || exit 1
}

if [ -n "${HF_ROOT:-}" ]; then
  _hf_usable "$HF_ROOT" \
    || _hf_fatal "HF_ROOT=$HF_ROOT is not writable or has under ${HF_MIN_FREE_GB}GB free." \
    || return 1 2>/dev/null || exit 1
else
  # Split the colon list by parameter expansion, NOT by an IFS word-splitting
  # loop. This file is SOURCED, so it runs in whatever shell the user has, and
  # zsh does not word-split unquoted parameters: `for x in $VAR` yields one
  # item there and HF_HOME ends up as the whole "a:b" string, a plausible path
  # that does not exist. Verified against bash, zsh and sh.
  HF_ROOT=""
  _hf_rest="$HF_CANDIDATES"
  while [ -n "$_hf_rest" ]; do
    _cand="${_hf_rest%%:*}"
    if [ "$_hf_rest" = "$_cand" ]; then _hf_rest=""; else _hf_rest="${_hf_rest#*:}"; fi
    [ -n "$_cand" ] || continue
    if _hf_usable "$_cand"; then HF_ROOT="$_cand"; break; fi
  done
  unset _hf_rest _cand
  [ -n "$HF_ROOT" ] \
    || _hf_fatal "no writable location with ${HF_MIN_FREE_GB}GB free (tried $HF_CANDIDATES)." \
    || return 1 2>/dev/null || exit 1
fi

# Only now, once a location is chosen, do we create anything.
mkdir -p "$HF_ROOT" || _hf_fatal "cannot create $HF_ROOT" \
  || return 1 2>/dev/null || exit 1

# Cached weights should not depend on the network. mlx_lm.server issues a HEAD
# to huggingface.co on every model switch even for local files, so a wifi blip
# turns into a model "failure" mid-run. Set HF_HUB_OFFLINE=0 to fetch new ones.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HOME="$HF_ROOT"
echo "hf    HF_HOME=$HF_HOME" >&2
