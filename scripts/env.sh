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
#   HF_ROOT         where the weights live (default: ./hf_root in the checkout)
#   HF_MIN_FREE_GB  how much room a location must have

# Enough for a normal model, not enough for MiniMax-H3. This was 160 for that
# one checkpoint, which made every ordinary machine unusable to guard a
# download that guards itself: fetch-h3-weights.sh demands its own 160GB and
# setup-omnisvg.sh its own 40, where the size is actually known.
HF_MIN_FREE_GB="${HF_MIN_FREE_GB:-20}"

# ONE LOCATION, NOT A SEARCH. This used to walk a colon-separated list of
# /Volumes paths -- one person's Mac written into the repo, which the Windows
# port had to fork here and again in harness/env.py, with a test whose only job
# was to catch the two forks drifting. The default is in the checkout, so it
# exists on any machine with no drive letter and no mount; a machine with a
# fast volume says so by setting HF_ROOT.
#
# FINDING THIS FILE IS SHELL-SPECIFIC. It is sourced, so $0 is the SHELL's name
# under dash and gives no path at all; bash has BASH_SOURCE and zsh has %x, and
# neither exists in the other. Walking up from $PWD for pyproject.toml is the
# portable last resort and is what a plain `sh` gets.
_hf_repo() {
  local here=""
  # shellcheck disable=SC2154
  [ -n "${BASH_SOURCE:-}" ] && here="${BASH_SOURCE[0]}"
  # Through eval because ${(%):-%x} is zsh-only SYNTAX: shellcheck parses this
  # file as sh and rejects it outright, and so would any sh that reached it.
  [ -z "$here" ] && [ -n "${ZSH_VERSION:-}" ] &&
    here="$(eval 'printf %s "${(%):-%x}"' 2>/dev/null)"
  if [ -n "$here" ] && [ -f "$here" ]; then
    (cd "$(dirname "$here")/.." 2>/dev/null && pwd)
    return 0
  fi
  local p="$PWD"
  while [ -n "$p" ] && [ "$p" != "/" ]; do
    if [ -f "$p/pyproject.toml" ]; then printf '%s' "$p"; return 0; fi
    p="$(dirname "$p")"
  done
  printf '%s' "$PWD"
}
_HF_DEFAULT="$(_hf_repo)/hf_root"

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
  # The PARENT must exist. Without this, walking up to the nearest existing
  # ancestor reaches the filesystem root, and any candidate at all is accepted
  # on a machine where the root is writable. harness/env.py carries the same
  # rule; test_the_shipped_candidates_match_the_ones_env_sh_searches is what
  # catches the two drifting apart.
  [ -d "$(dirname "$cand")" ] || return 1
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
  echo "       Point HF_ROOT at a drive with room, or lower the threshold if" >&2
  echo "       you know what you are fetching. The default is ./hf_root in the" >&2
  echo "       checkout; this machine sets HF_ROOT=/Volumes/Models/hf." >&2
  echo >&2
  echo "       If the volume IS attached and this still fails, it is probably" >&2
  echo "       macOS TCC. A launchd agent or other background process gets" >&2
  echo "       \"Operation not permitted\" on /Volumes even though the volume" >&2
  echo "       stats fine. Grant Full Disk Access to the program launchd runs" >&2
  echo "       (System Settings > Privacy & Security > Full Disk Access), or" >&2
  echo "       start the services from a terminal that already has it." >&2
  return 1 2>/dev/null || exit 1
}

# WHAT IS SAID IS STILL CHECKED. An explicit HF_ROOT does not bypass the
# writability and free-space tests -- skipping them was a real bug once, and an
# unwritable and a nonexistent path both passed with exit 0.
HF_ROOT="${HF_ROOT:-$_HF_DEFAULT}"
_hf_usable "$HF_ROOT" \
  || _hf_fatal "HF_ROOT=$HF_ROOT is not writable or has under ${HF_MIN_FREE_GB}GB free." \
  || return 1 2>/dev/null || exit 1

# THERE IS NO SILENT FALLBACK TO GUARD AGAINST ANY MORE.
#
# A reboot once brought this machine back WITHOUT the weights volume attached.
# The candidate loop did exactly what it was designed to do: it skipped the
# missing /Volumes/Models, found /Volumes/T7 with room to spare, and the
# services started against an EMPTY CACHE. They listened, served nothing, and
# said nothing about it. A state file was added to remember where we landed
# last time and refuse to move without HF_ALLOW_MOVE=1.
#
# That guard was aimed at the SEARCH, and the search is gone. One location is
# configured; if it is not there, the block above is FATAL rather than helpful,
# which is the property the state file was reconstructing after the fact.
# Broadening the guard to fire whenever HF_ROOT differs from last time would
# false-alarm on this machine every time the suite (./hf_root) and the services
# (/Volumes/Models/hf) alternate, and a guard people learn to override is worse
# than none. HF_STATE_FILE and HF_ALLOW_MOVE are gone with it.

# Only now, once a location is chosen, do we create anything.
mkdir -p "$HF_ROOT" || _hf_fatal "cannot create $HF_ROOT" \
  || return 1 2>/dev/null || exit 1

# Cached weights should not depend on the network. mlx_lm.server issues a HEAD
# to huggingface.co on every model switch even for local files, so a wifi blip
# turns into a model "failure" mid-run. Set HF_HUB_OFFLINE=0 to fetch new ones.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# NATIVE PATH, NOT AN MSYS ONE. Git Bash reports directories as /d/a/... and
# huggingface_hub is native Python, which cannot resolve that: it would take
# the string literally and build a cache under a directory called "d". The
# default is computed with `pwd`, so on Windows it arrives in the MSYS
# spelling; cygpath is what Git Bash ships to convert it. Nothing to do
# anywhere else, where the two spellings are the same.
if command -v cygpath >/dev/null 2>&1; then
  HF_ROOT="$(cygpath -w "$HF_ROOT")"
fi
export HF_HOME="$HF_ROOT"
echo "hf    HF_HOME=$HF_HOME" >&2
