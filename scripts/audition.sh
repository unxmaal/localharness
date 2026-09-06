#!/usr/bin/env bash
# Play the French voice artifacts so a human can judge what wer cannot.
#
# Word error rate measures INTELLIGIBILITY. It says nothing about whether a
# cloned voice resembles the speaker it was cloned from, which is the entire
# point of cloning. That needs ears.
#
#   ./scripts/audition.sh              # every candidate on one case
#   ./scripts/audition.sh fr-liaison   # ...on the case you name
#   ./scripts/audition.sh --all        # every candidate on every case
set -euo pipefail

cd "$(dirname "$0")/.."

REFS=/Volumes/Models/corpora/voice-refs
CASE="${1:-fr-pangram}"

say_() { printf '\n\033[1m%s\033[0m\n' "$*"; }

play() {
  [ -f "$1" ] || { printf '  missing: %s\n' "$1" >&2; return 0; }
  printf '  %s\n' "$(basename "$1")"
  afplay "$1"
}

audition_case() {
  local c="$1" ref clone
  say_ "== $c =="

  # The reference first, and every time: judging similarity from memory of a
  # clip played two minutes ago is not judging similarity.
  for n in 1 2 3; do
    ref="$REFS/fleurs-fr-male-$n.wav"
    clone=".logs/fr-voices/Chatterbox-Multilingual-MLX-v2-Q8_fleurs-fr-male-$n--$c.wav"
    [ -f "$clone" ] || clone=".logs/fr-refs/Chatterbox-Multilingual-MLX-v2-Q8_fleurs-fr-male-$n--$c.wav"
    [ -f "$clone" ] || continue
    say_ "-- reference $n, then its clone"
    play "$ref"
    play "$clone"
  done

  say_ "-- Kokoro ff_siwis (no reference: a fixed voice, and the only French one it has)"
  play ".logs/fr-voices/Kokoro-82M-bf16_ff_siwis--$c.wav"
}

if [ "$CASE" = "--all" ]; then
  for c in fr-pangram fr-liaison fr-numbers-and-units fr-long-clause fr-prosody-question; do
    audition_case "$c"
  done
else
  audition_case "$CASE"
fi

say_ "done. The question wer cannot answer: does the clone sound like its reference?"
