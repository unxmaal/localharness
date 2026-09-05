# Shared environment. Sourced by the launchers; never run directly.
#
# Model weights live on external storage, not the internal disk: the internal
# volume runs at 92% and a single 70B at 4-bit is ~40GB.
#
# Preference order is by MEASURED throughput (4 GiB dd, cold read forced by
# unmount/remount so the page cache cannot flatter the number):
#
#   /Volumes/Models  DockCase SSD Enclosure C1P, ioreg Speed=4 (SuperSpeed+,
#                    10 Gbps), direct to a host controller.
#                    1013 MB/s write, 959 MB/s cold read. PREFERRED.
#   /Volumes/T7      Samsung PSSD T7, ioreg Speed=3 (SuperSpeed, 5 Gbps) because
#                    it sits behind a VIA Labs USB3.0 hub.
#                    422 MB/s write, 432 MB/s cold read. 2.2x slower.
#                    The T7 is itself a 10 Gbps device: the hub is halving it,
#                    and moving it to a direct port should roughly double it.
#
# Model load time scales directly with this, and the hot-swap design (PLAN.md
# section 4) pays it on every model switch: ~40GB is ~90s at 460 MB/s.
#
# HF_ROOT pins a specific location. It is still validated: an explicit path that
# is unmounted or unwritable is a hard error, never a silent fallback to the
# internal disk.

_hf_usable() {  # $1 = candidate .../hf path
  local cand="$1" vol="${1%/hf}"
  mount | grep -q "on $vol " || return 1
  mkdir -p "$cand" 2>/dev/null || return 1
  [ -w "$cand" ] || return 1
  return 0
}

if [ -n "${HF_ROOT:-}" ]; then
  if ! _hf_usable "$HF_ROOT"; then
    echo "FATAL: HF_ROOT=$HF_ROOT is not a mounted, writable location." >&2
    echo "       Refusing to run: weights would land on the internal disk," >&2
    echo "       which is at 92% and cannot hold them." >&2
    return 1 2>/dev/null || exit 1
  fi
else
  HF_ROOT=""
  for _cand in /Volumes/Models/hf /Volumes/T7/hf; do
    if _hf_usable "$_cand"; then HF_ROOT="$_cand"; break; fi
  done
  if [ -z "$HF_ROOT" ]; then
    echo "FATAL: no writable model volume found (tried /Volumes/Models, /Volumes/T7)." >&2
    echo "       Plug in the external SSD holding the Models volume." >&2
    return 1 2>/dev/null || exit 1
  fi
fi

export HF_HOME="$HF_ROOT"
echo "hf    HF_HOME=$HF_HOME" >&2
