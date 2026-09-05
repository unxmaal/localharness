#!/usr/bin/env bash
# Fetch the MiniMax-H3 FL2VA checkpoint for antirez/h3.c.
#
# Only FL2VA is fetched (134.2 GiB of the repo's 464.2 GiB). Ref2VA is optional:
# h3.c probes for Ref2VA/transformer/model.safetensors.index.json and skips the
# whole component when absent, keeping prompt-only FL2VA generation working.
# The repo also ships a flat layout (text_encoder/, transformer/, vae/, ...)
# duplicating the same weights; h3.c wants the task-partitioned form, so the
# flat copy is skipped too.
#
# Resumable: re-running picks up where it left off.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

DEST="${H3_MODEL_DIR:-/Volumes/Models/MiniMax-H3}"
mkdir -p "$DEST"

# ~134 GiB needs headroom; refuse rather than fill the volume.
AVAIL_GB=$(df -g "$DEST" | awk 'NR==2 {print $4}')
if [ "$AVAIL_GB" -lt 160 ]; then
  echo "FATAL: only ${AVAIL_GB}GB free at $DEST, need ~160GB headroom for 134GiB." >&2
  exit 1
fi
echo "dest  $DEST (${AVAIL_GB}GB free)"

exec uv run --no-project --with 'huggingface_hub[hf_transfer]' python - "$DEST" <<'PY'
import os, sys, time
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download

dest = sys.argv[1]
t0 = time.time()
path = snapshot_download(
    repo_id="MiniMaxAI/MiniMax-H3",
    local_dir=dest,
    allow_patterns=["FL2VA/**"],
    max_workers=8,
)
print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {path}")
PY
