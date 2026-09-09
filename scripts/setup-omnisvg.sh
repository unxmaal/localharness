#!/usr/bin/env bash
# Install OmniSVG: the checkout, its own venv, and the two model repos.
#
# It gets a venv of its own because it pins transformers 4.51.3 and wants torch,
# neither of which this project otherwise has, and because its inference script
# imports sibling modules by name and so must run from its own directory.
#
# Resumable: re-running skips whatever is already there.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

DEST="${OMNISVG_HOME:-${LOCALHARNESS_HOME:-$HOME/localharness}/omnisvg}"
REPO="${OMNISVG_REPO:-https://github.com/OmniSVG/OmniSVG.git}"

if [ ! -d "$DEST/.git" ]; then
  echo "clone $REPO -> $DEST"
  git clone --depth 1 "$REPO" "$DEST"
fi

# torch and numpy are unpinned: upstream pins torch==2.3.0, which predates the
# MPS fixes this needs and has no wheel for a current Python.
cat > "$DEST/requirements-mac.txt" <<'REQ'
transformers==4.51.3
accelerate
numpy==2.2.6
Pillow
CairoSVG
einops
ipython
matplotlib
moviepy==1.0.3
networkx
qwen-vl-utils==0.0.11
PyYAML
shapely
torch
torchvision
huggingface_hub
REQ

if [ ! -x "$DEST/.venv/bin/python" ]; then
  uv venv --python 3.12 "$DEST/.venv"
fi
uv pip install --quiet --python "$DEST/.venv/bin/python" -r "$DEST/requirements-mac.txt"

# Both halves are needed: OmniSVG ships the decoder weights only, on top of a
# stock Qwen2.5-VL it does not redistribute.
AVAIL_GB=$(df -Pk "${HF_HOME:-$HOME/.cache/huggingface}" | awk 'NR==2 {print int($4 / 1048576)}')
if [ "$AVAIL_GB" -lt 40 ]; then
  echo "FATAL: only ${AVAIL_GB}GB free at ${HF_HOME:-$HOME/.cache/huggingface}, need ~40GB." >&2
  exit 1
fi

HF_HUB_OFFLINE=0 uv run --no-project --with huggingface_hub python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download

for size, qwen, omni in (("4B", "Qwen/Qwen2.5-VL-3B-Instruct", "OmniSVG/OmniSVG1.1_4B"),):
    print(qwen, snapshot_download(qwen), flush=True)
    print(omni, hf_hub_download(omni, "pytorch_model.bin"), flush=True)
PY

# transformers' resize_token_embeddings defaults to mean_resizing=True, which
# fits a multivariate normal over the old embedding matrix and asks Metal for
# ~704 GiB: an MPS assertion failure and SIGABRT before a token is generated.
# The rows it initialises are overwritten by the OmniSVG checkpoint on the next
# line, so switching it off changes nothing but whether this starts.
uv run --no-project python - "$DEST/decoder.py" <<'MPSPATCH'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
old = "self.transformer.resize_token_embeddings(self.vocab_size)"
new = "self.transformer.resize_token_embeddings(self.vocab_size, mean_resizing=False)"
if new in s:
    sys.exit(0)
if s.count(old) != 1:
    sys.exit(f"decoder.py no longer matches the MPS patch anchor ({s.count(old)} hits)")
p.write_text(s.replace(old, new))
print("patched decoder.py for MPS")
MPSPATCH

echo "ok  OMNISVG_HOME=$DEST"
