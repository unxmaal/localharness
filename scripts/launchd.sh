#!/usr/bin/env bash
# launchd units for the three services, so the machine comes back up serving
# after a reboot or a crash without anyone remembering three script names.
#
#   ./scripts/launchd.sh generate [DIR]   write the plists (default: $LH_HOME/launchd)
#   ./scripts/launchd.sh install          write them to ~/Library/LaunchAgents and load
#   ./scripts/launchd.sh uninstall        unload and remove them
#   ./scripts/launchd.sh status           what launchd thinks is running
#
# Generated from one loop rather than three hand-written files. The last time
# this repo had three near-identical things the copies disagreed about which
# port they used, and that took a while to find.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
AGENTS="$HOME/Library/LaunchAgents"
# One place for everything this project produces; see harness/paths.py.
LH_HOME="${LOCALHARNESS_HOME:-$HOME/localharness}"
LH_LOGS="$LH_HOME/logs"
PREFIX="com.unxmaal.localharness"
# `discover` is not a server. It is the scheduled sweep, and it is in this
# list because the thing that must survive a reboot is the SCHEDULE. #261.
SERVICES="gateway mlx tts mcp discover"

#: Services that RUN AND EXIT rather than serve, with how often to run them.
#: KeepAlive on one of these restarts a finished sweep at once and the machine
#: discovers in a tight loop; StartInterval is the right key.
declare -a PERIODIC=(discover)
DISCOVER_INTERVAL="${DISCOVER_INTERVAL:-21600}"   # six hours

# launchd starts jobs with PATH=/usr/bin:/bin:/usr/sbin:/sbin and NOTHING else.
# uv, ffmpeg, rsvg-convert and rec all live in /opt/homebrew/bin, so without
# this every service dies on "command not found" -- the same trap a GUI-spawned
# wezterm set for the voice scripts, and it is just as invisible here.
# $HOME/.local/bin is where uv puts `lh`, and serve-mcp.sh shells out to it for
# every tool call. Without it that unit loads, listens, and fails each call with
# "lh: command not found".
JOB_PATH="$HOME/.local/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

# AND PATH IS NOT THE ONLY THING launchd DROPS. env.sh resolves the weights
# cache from HF_ROOT, then an inherited HF_HOME, then ./hf_root in the checkout
# (RULE #222: a default naming one person's volume is not a default). A launchd
# job inherits neither variable, so it took the in-tree default, and mlx_lm's
# /v1/models raised
#
#   CacheNotFound: .../localharness/hf_root/hub
#
# The port stayed bound, the model list came back empty, and every generate
# request hung until the caller gave up. Four text lanes were unrunnable for
# hours and the only symptom anywhere else was models timing out, which reads
# as the model failing. #255.
#
# Resolved HERE rather than written into the plist by hand, so the value is
# this machine's own answer and a second machine gets its own.
_hf_env() {
  local root home
  root="$(cd "$REPO" && . scripts/env.sh >/dev/null 2>&1 && printf '%s' "${HF_ROOT:-}")"
  home="$(cd "$REPO" && . scripts/env.sh >/dev/null 2>&1 && printf '%s' "${HF_HOME:-}")"
  [ -n "$root" ] && printf '    <key>HF_ROOT</key><string>%s</string>\n' "$root"
  [ -n "$home" ] && printf '    <key>HF_HOME</key><string>%s</string>\n' "$home"
}

usage() {
  echo "usage: $0 {generate [DIR]|install|uninstall|status|probe}" >&2
  exit 2
}

# A server is kept alive; a periodic job is run on an interval and allowed to
# finish. Getting this backwards means either a sweep that never repeats or one
# that never stops.
_schedule() {
  local service="$1" p
  for p in "${PERIODIC[@]}"; do
    if [ "$p" = "$service" ]; then
      printf '  <key>StartInterval</key><integer>%s</integer>\n' \
        "$DISCOVER_INTERVAL"
      return
    fi
  done
  printf '  <key>KeepAlive</key><true/>\n'
}

write_plist() {
  local service="$1" dest="$2"
  cat > "$dest/$PREFIX.$service.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$PREFIX.$service</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$REPO/scripts/serve-$service.sh</string>
  </array>
  <key>RunAtLoad</key><true/>
$(_schedule "$service")
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>StandardOutPath</key><string>$LH_LOGS/$service.log</string>
  <key>StandardErrorPath</key><string>$LH_LOGS/$service.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>$JOB_PATH</string>
    <key>HOME</key><string>$HOME</string>
$(_hf_env)  </dict>
  <key>ProcessType</key><string>Interactive</string>
</dict>
</plist>
PLIST
}

generate() {
  local dest="${1:-$LH_HOME/launchd}"
  mkdir -p "$dest" "$LH_LOGS"
  for service in $SERVICES; do
    write_plist "$service" "$dest"
    echo "wrote $dest/$PREFIX.$service.plist"
  done
}

# Can a launchd agent actually read the weights volume?
#
# Checking from THIS shell proves nothing: the terminal already has the Full
# Disk Access the agent lacks. macOS TCC protects /Volumes and a background job
# has no way to ask for consent, so the volume stats fine, reports free space,
# appears in /Volumes -- and every read returns "Operation not permitted".
# mlx_lm turns that into a permanent hang inside os.listdir, with the server
# accepting connections and answering none, which is a genuinely horrible way
# to find out about a permission. So probe from inside launchd first.
preflight() {
  local probe="/tmp/localharness-preflight.$$"
  local label="$PREFIX.preflight"
  local root="${HF_ROOT:-$PWD/hf_root}"
  cat > "$probe.sh" <<PROBE
#!/bin/bash
ls "$root" >/dev/null 2>&1 && echo ok > "$probe.out" || echo denied > "$probe.out"
PROBE
  chmod +x "$probe.sh"
  cat > "$probe.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>$label</string>
<key>ProgramArguments</key><array><string>$probe.sh</string></array>
<key>RunAtLoad</key><true/>
</dict></plist>
PLIST
  launchctl bootout "gui/$UID/$label" 2>/dev/null || true
  launchctl bootstrap "gui/$UID" "$probe.plist" 2>/dev/null || true
  for _ in $(seq 1 20); do [ -f "$probe.out" ] && break; sleep 0.5; done
  local verdict; verdict="$(cat "$probe.out" 2>/dev/null || echo unknown)"
  launchctl bootout "gui/$UID/$label" 2>/dev/null || true
  rm -f "$probe.sh" "$probe.plist" "$probe.out"

  if [ "$verdict" != "ok" ]; then
    cat >&2 <<MSG
FATAL: a launchd agent cannot read $root ($verdict).

  macOS TCC protects /Volumes, and a background job has no way to ask for
  consent. The volume stats fine and shows up in /Volumes, so nothing looks
  wrong until mlx_lm hangs forever inside os.listdir -- serving no requests
  and logging no error.

  To fix, grant Full Disk Access to the program launchd runs:
    System Settings > Privacy & Security > Full Disk Access > +
    add /bin/bash   (this is broad; a narrower option is to point these
                     units at a dedicated interpreter and grant only that)

  Then re-run: ./scripts/launchd.sh install

  Or skip launchd and start the services from a terminal that already has
  the access, which is what ./scripts/serve-*.sh do today.
MSG
    exit 1
  fi
}

install_units() {
  preflight
  mkdir -p "$AGENTS"
  generate "$AGENTS" >/dev/null
  for service in $SERVICES; do
    # bootout first so `install` is re-runnable: bootstrap on an already-loaded
    # label fails, and "already loaded" is the normal state when reinstalling.
    launchctl bootout "gui/$UID/$PREFIX.$service" 2>/dev/null || true
    launchctl bootstrap "gui/$UID" "$AGENTS/$PREFIX.$service.plist"
    echo "loaded $PREFIX.$service"
  done
  echo
  echo "Give them a moment, then: ./scripts/smoke.sh"
}

uninstall_units() {
  for service in $SERVICES; do
    launchctl bootout "gui/$UID/$PREFIX.$service" 2>/dev/null || true
    rm -f "$AGENTS/$PREFIX.$service.plist"
    echo "removed $PREFIX.$service"
  done
}

status() {
  for service in $SERVICES; do
    printf '%-12s ' "$service"
    # First match only: launchctl print reports the job state and then several
    # "active" lines for its endpoints, which read as three answers to one
    # question.
    launchctl print "gui/$UID/$PREFIX.$service" 2>/dev/null \
      | awk '/^\tstate = /{print $3; found=1; exit} END{if(!found) print "not loaded"}'
  done
}

case "${1:-}" in
  generate)  shift; generate "${1:-}" ;;
  probe)     preflight && echo "ok: a launchd agent can read ${HF_ROOT:-$PWD/hf_root}" ;;
  install)   install_units ;;
  uninstall) uninstall_units ;;
  status)    status ;;
  *)         usage ;;
esac
