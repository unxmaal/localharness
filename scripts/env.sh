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

# THE LIST SEPARATOR IS NOT ALWAYS A COLON. A Windows path carries a colon
# after its drive letter, so a colon-separated list splits "C:/models" into "C"
# and "/models" and HF_HOME silently becomes "C". PATH itself is
# semicolon-separated on Windows for exactly this reason.
case "$(uname -s 2>/dev/null)" in
  MINGW*|MSYS*|CYGWIN*) HF_SEP="${HF_SEP:-;}" ;;
  *)                    HF_SEP="${HF_SEP:-:}" ;;
esac
# Ordered by measured throughput, ending somewhere that exists on any Mac so a
# machine with no external volume still works without hand configuration.
if [ "$HF_SEP" = ";" ]; then
  # No /Volumes to search. See the note in harness/env.py.
  HF_CANDIDATES="${HF_CANDIDATES:-$HOME/.cache/huggingface}"
else
  HF_CANDIDATES="${HF_CANDIDATES:-/Volumes/Models/hf:/Volumes/T7/hf:$HOME/.cache/huggingface}"
fi

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
# ancestor. `df -Pk` forces 1024-byte blocks on BSD and GNU alike; plain
# `df -g` is BSD-only and GNU answers "unknown option -- g", which read as
# a machine with no disk rather than as a wrong flag.
_hf_free_gb() {
  local p="$1" parent
  while [ -n "$p" ] && [ ! -e "$p" ]; do
    parent="$(dirname "$p")"
    [ "$parent" = "$p" ] && break
    p="$parent"
  done
  [ -e "$p" ] || return 1
  df -Pk "$p" 2>/dev/null | awk 'NR==2 { print int($4 / 1048576) }'
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

  # READABILITY, which is not implied by any of the above. A launchd agent gets
  # "Operation not permitted" on /Volumes: macOS TCC protects removable volumes
  # and a background job cannot ask for consent. The volume still stats, still
  # reports free space, and still appears in /Volumes, so every other check here
  # passes -- and then mlx_lm hangs forever inside os.listdir. Read one entry.
  _hf_readable "$mp" || return 1

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

# Can we actually list this directory? See the note in _hf_usable.
_hf_readable() {
  ls "$1" >/dev/null 2>&1
}

_hf_fatal() {
  echo "FATAL: $1" >&2
  echo "       A location needs ${HF_MIN_FREE_GB}GB free (HF_MIN_FREE_GB)." >&2
  echo "       Weights are large: MiniMax-H3 alone is 134GiB, and a 70B at" >&2
  echo "       4-bit is ~40GB. Attach a volume, or lower the threshold if you" >&2
  echo "       know what you are fetching." >&2
  echo >&2
  echo "       If the volume IS attached and this still fails, it is probably" >&2
  echo "       macOS TCC. A launchd agent or other background process gets" >&2
  echo "       \"Operation not permitted\" on /Volumes even though the volume" >&2
  echo "       stats fine. Grant Full Disk Access to the program launchd runs" >&2
  echo "       (System Settings > Privacy & Security > Full Disk Access), or" >&2
  echo "       start the services from a terminal that already has it." >&2
  return 1 2>/dev/null || exit 1
}

_hf_explicit=0
[ -n "${HF_ROOT:-}" ] && _hf_explicit=1
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
    _cand="${_hf_rest%%$HF_SEP*}"
    if [ "$_hf_rest" = "$_cand" ]; then _hf_rest=""; else _hf_rest="${_hf_rest#*$HF_SEP}"; fi
    [ -n "$_cand" ] || continue
    if _hf_usable "$_cand"; then HF_ROOT="$_cand"; break; fi
  done
  unset _hf_rest _cand
  [ -n "$HF_ROOT" ] \
    || _hf_fatal "no writable location with ${HF_MIN_FREE_GB}GB free (tried $HF_CANDIDATES)." \
    || return 1 2>/dev/null || exit 1
fi

# WHERE WE LANDED LAST TIME.
#
# A reboot once brought this machine back WITHOUT the weights volume attached.
# The loop above did exactly what it was designed to do: it skipped the missing
# /Volumes/Models, found /Volumes/T7 with room to spare, and the services
# started against an EMPTY CACHE. They listened, served nothing, and said
# nothing about it -- the only trace was one differing line in a log nobody
# reads until something is already wrong.
#
# Falling back is right on a fresh machine and wrong on a machine with 93GB of
# weights sitting on a drive that happens to be unplugged. The difference is
# whether we have been here before, so record it and refuse to move silently.
HF_STATE_FILE="${HF_STATE_FILE:-$HOME/.localharness-hf-root}"
# Only police the AUTO-PICK. An explicit HF_ROOT is the caller saying where
# the weights are, which is the same statement HF_ALLOW_MOVE makes.
if [ "$_hf_explicit" = "0" ] && [ -f "$HF_STATE_FILE" ]; then
  _hf_prev="$(cat "$HF_STATE_FILE" 2>/dev/null)"
  if [ -n "$_hf_prev" ] && [ "$_hf_prev" != "$HF_ROOT" ] && [ "${HF_ALLOW_MOVE:-0}" != "1" ]; then
    echo "FATAL: the weights cache moved." >&2
    echo "       last time: $_hf_prev" >&2
    echo "       this time: $HF_ROOT" >&2
    echo >&2
    if [ ! -d "$_hf_prev" ]; then
      echo "       $_hf_prev is NOT PRESENT. If that is an external drive," >&2
      echo "       it is unplugged, asleep, or failed to mount -- check the" >&2
      echo "       cable and \`diskutil list external\` before doing anything" >&2
      echo "       else. Starting on $HF_ROOT would serve from a cache with" >&2
      echo "       none of your models in it." >&2
    else
      echo "       Both exist, so the candidate order or free space changed." >&2
    fi
    echo >&2
    echo "       To move deliberately:  HF_ALLOW_MOVE=1 (once), or set HF_ROOT." >&2
    unset _hf_prev
    return 1 2>/dev/null || exit 1
  fi
  unset _hf_prev
fi

# Only now, once a location is chosen, do we create anything.
mkdir -p "$HF_ROOT" || _hf_fatal "cannot create $HF_ROOT" \
  || return 1 2>/dev/null || exit 1
printf '%s\n' "$HF_ROOT" > "$HF_STATE_FILE" 2>/dev/null || true
unset _hf_explicit

# Cached weights should not depend on the network. mlx_lm.server issues a HEAD
# to huggingface.co on every model switch even for local files, so a wifi blip
# turns into a model "failure" mid-run. Set HF_HUB_OFFLINE=0 to fetch new ones.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HOME="$HF_ROOT"
echo "hf    HF_HOME=$HF_HOME" >&2
