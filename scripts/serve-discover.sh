#!/usr/bin/env bash
# The sweep, on a schedule. Issue #261.
#
# THIS IS THE PRODUCT'S PREMISE. Obsoletion-proofing that has to be typed by a
# person is a script, not a property of the system: the loop turns only when
# somebody remembers, and what it is meant to protect against is precisely the
# passage of unattended time.
#
# NOT A SERVER, despite living beside the serve-* scripts. launchd runs it on
# an interval and it exits; the plist carries StartInterval rather than
# KeepAlive, or launchd would restart a finished sweep immediately and the
# machine would discover in a tight loop.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

LH_LOGS="${LOCALHARNESS_HOME:-$HOME/localharness}/logs"
mkdir -p "$LH_LOGS"

# WHAT AN UNATTENDED RUN MAY SPEND. A scheduled job that downloads without a
# ceiling fills the disk overnight and is discovered the next morning; the loop
# already refuses to exceed these and they are here so the answer is visible in
# the place that sets the schedule.
TOP="${DISCOVER_TOP:-3}"
BUDGET="${DISCOVER_BUDGET_GIB:-12}"
REPEAT="${DISCOVER_REPEAT:-3}"

echo "=== sweep $(date -u +%Y-%m-%dT%H:%M:%SZ) top=$TOP budget=${BUDGET}GiB ==="
exec uv run lh discover --loop --run \
  --top "$TOP" --budget-gib "$BUDGET" --repeat "$REPEAT"
