#!/usr/bin/env bash
# The reference the music lane's cover case restyles.
#
# GENERATED RATHER THAN COMMITTED. A reference that is itself an ACE-Step
# render holds recording condition constant, which is what RULE #199 says a
# similarity comparison needs; it also keeps a 5 MB wav out of a public repo
# and avoids licensing somebody else's recording.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

: "${ACESTEP_ROOT:?set ACESTEP_ROOT to the ACE-Step checkout}"
OUT="${1:-${LOCALHARNESS_HOME:-$HOME/localharness}/refs/cover-reference.wav}"
mkdir -p "$(dirname "$OUT")"

if [ -f "$OUT" ]; then
  echo "already there: $OUT"
  exit 0
fi

# Fixed caption and seed, so the reference is the same artefact on any machine
# that builds it. Deliberately a DIFFERENT style from the case's own prompt:
# a cover that merely reproduces its reference has not transferred anything.
exec "$ACESTEP_ROOT/.venv/bin/python" scripts/acestep_generate.py \
  --root "$ACESTEP_ROOT" --out "$OUT" \
  --caption "upbeat electronic synthwave with driving arpeggios and a bright analog lead" \
  --instrumental --duration 30 --seed 7 --steps 8
