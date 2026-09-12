#!/usr/bin/env bash
# What a machine needs before it has any lanes at all.
#
# This runs BEFORE uv and lh exist, which is why it is shell and not an lh
# subcommand: half its job is telling you that uv is missing. `lh discover`
# covers the lane-level question once the project is installed; this covers the
# host-level preconditions the setup docs state in prose.
#
# It reports every failure and then exits, rather than stopping at the first.
# Stopping first would mean one apt round-trip per missing package.
#
# NO pipefail HERE, deliberately. Nearly every check is `producer | grep -q`,
# and grep -q exits at the first match, so the producer takes a SIGPIPE and
# pipefail reports the whole pipeline as failed. That turns present things into
# absent ones: it marked DejaVu missing on a machine that had both the fonts
# and fc-list.
set -u

cd "$(dirname "$0")/.." || exit 2

fail=0
section() { printf '\n%s\n' "$1"; }
ok()      { printf '  ok       %s\n' "$1"; }
bad()     { printf '  MISSING  %s\n             -> %s\n' "$1" "$2"; fail=1; }
note()    { printf '  note     %s\n             -> %s\n' "$1" "$2"; }
have()    { command -v "$1" >/dev/null 2>&1; }

need() { # need <command> <how to get it>
  if have "$1"; then ok "$1"; else bad "$1" "$2"; fi
}

uname_s="$(uname -s)"
case "$uname_s" in
  Linux)  apt="sudo apt install -y" ;;
  Darwin) apt="brew install" ;;
  *)      apt="install" ;;
esac

section "accelerator"
if have nvidia-smi; then
  if nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null; then
    ok "nvidia-smi answers"
  else
    bad "nvidia-smi is installed but does not answer" \
        "with Secure Boot on, finish the MOK enrolment on the next boot"
  fi
elif [ "$uname_s" = "Darwin" ]; then
  ok "Apple Silicon: no discrete-accelerator probe needed"
else
  note "no nvidia-smi" "fine on a machine with no NVIDIA card; otherwise: sudo ubuntu-drivers install"
fi

section "packages the docs name"
need git         "$apt git"
need curl        "$apt curl"
need make        "$apt make"
need shellcheck  "$apt shellcheck"
need ffmpeg      "$apt ffmpeg"
need rsvg-convert "$apt librsvg2-bin   # the ink lane silently skips without it"
need zsh         "$apt zsh             # env.sh tests run in three shells"
need sox         "$apt sox             # provides rec, the stt lane's recorder"

# Not just present: harness/proc.py RAISES rather than reporting a zero when
# /usr/bin/time cannot report peak memory, and the shell builtin cannot.
# The two paths below are overridable so the tests can drive this hermetically
# rather than asserting whatever the runner happens to have installed.
time_bin="${LH_PREFLIGHT_TIME_BIN:-/usr/bin/time}"
if [ -x "$time_bin" ] && "$time_bin" -v true 2>&1 | grep -q 'Maximum resident set size'; then
  ok "$time_bin -v reports peak memory"
elif [ -x "$time_bin" ] && "$time_bin" -l true 2>&1 | grep -q 'maximum resident set size'; then
  ok "$time_bin -l reports peak memory"
else
  bad "$time_bin cannot report peak memory" \
      "$apt time   # the shell builtin reports none, and proc.py raises"
fi

if [ "$uname_s" = "Linux" ]; then
  # The file, not fc-list: PIL opens the .ttf directly and fontconfig need not
  # be installed for that to work.
  # shellcheck disable=SC2086  # the default is a deliberate multi-path word list
  font_dirs="${LH_PREFLIGHT_FONT_DIRS:-/usr/share/fonts /usr/local/share/fonts $HOME/.local/share/fonts}"
  dejavu="$(find $font_dirs -iname 'DejaVu*.ttf' -print -quit 2>/dev/null)"
  if [ -n "$dejavu" ]; then
    ok "DejaVu fonts"
  else
    bad "DejaVu fonts" "$apt fonts-dejavu-core   # else PIL's bitmap fallback measures the fixture"
  fi
fi

section "a browser render.py will use"
browser=""
for b in "${LH_CHROME:-}" google-chrome google-chrome-stable chromium chromium-browser; do
  [ -n "$b" ] || continue
  p="$(command -v "$b" 2>/dev/null || true)"
  [ -n "$p" ] || continue
  case "$(readlink -f "$p" 2>/dev/null || echo "$p")" in
    /snap/*) note "$b resolves under /snap" "render.py skips those; install the Chrome .deb" ;;
    *)       browser="$p"; break ;;
  esac
done
if [ -n "$browser" ]; then
  ok "$browser"
else
  bad "no browser outside /snap" \
      "curl -fsSLo /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && $apt /tmp/chrome.deb"
fi

section "the project itself"
need uv "curl -LsSf https://astral.sh/uv/install.sh | sh"
if have lh; then
  ok "lh on PATH"
else
  bad "lh on PATH" 'uv tool install --python 3.12 --editable . && export PATH="$HOME/.local/bin:$PATH"'
fi

section "weights root"
# env.sh owns the guard (writable, enough room). Ask it rather than reimplement.
if hf_out="$(HF_ROOT="${HF_ROOT:-}" bash -c 'source scripts/env.sh >/dev/null 2>&1 && echo "$HF_HOME"')" \
   && [ -n "$hf_out" ]; then
  ok "HF_HOME=$hf_out"
else
  bad "no usable weights root" \
      "export HF_ROOT=/somewhere/with/room   # env.sh checks writability and free space"
fi

section "text lane"
llama="${LLAMACPP_BIN:-llama-server}"
if have "$llama" || [ -x "$llama" ]; then
  ok "$llama"
  # A CPU-only build also exists, also runs, and leaves the card idle. The
  # backend is the thing worth asserting, not the binary.
  cli="$(dirname "$(command -v "$llama" 2>/dev/null || echo "$llama")")/llama-cli"
  if [ -x "$cli" ]; then
    if "$cli" --list-devices 2>/dev/null | grep -qiE 'CUDA|Metal|Vulkan'; then
      ok "backend: $("$cli" --list-devices 2>/dev/null | grep -iE 'CUDA|Metal|Vulkan' | head -1 | sed 's/^ *//')"
    else
      bad "llama.cpp reports no accelerator" \
          "rebuild with -DGGML_CUDA=ON; a CPU-only build serves with the card idle"
    fi
  else
    note "llama-cli not beside llama-server" "cannot confirm the backend is not CPU-only"
  fi
else
  note "no llama-server" "the text lane stays absent until LLAMACPP_BIN points at one"
fi

section "scripts the docs say to run as ./scripts/..."
mode_bad=0
while IFS= read -r line; do
  [ -n "$line" ] || continue
  mode="${line%% *}"
  path="${line#*$'\t'}"
  [ "$(head -c2 "$path" 2>/dev/null)" = "#!" ] || continue
  if [ "$mode" != "100755" ]; then
    bad "$path is $mode" "git update-index --chmod=+x $path"
    mode_bad=1
  fi
done < <(git ls-files -s -- 'scripts/*.sh' 2>/dev/null)
[ "$mode_bad" -eq 0 ] && ok "every script with a shebang is executable"

section "audio out"
player=""
for p in /usr/bin/afplay paplay aplay ffplay; do
  have "$p" && { player="$p"; break; }
done
if [ -n "$player" ]; then
  ok "$player"
else
  note "no audio player" "lh say --play needs one of afplay, paplay, aplay, ffplay"
fi

printf '\n'
if [ "$fail" -eq 0 ]; then
  echo "preflight: ok"
else
  echo "preflight: something above is missing"
fi
exit "$fail"
