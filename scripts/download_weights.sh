#!/usr/bin/env bash
# Download BiSeNet face-parsing weights if not already present.
set -euo pipefail

WEIGHTS_DIR="weights"
BISENET_PATH="${WEIGHTS_DIR}/bisenet_face_parsing.pth"
BISENET_URL="https://huggingface.co/vivym/face-parsing-bisenet/resolve/main/79999_iter.pth"

mkdir -p "${WEIGHTS_DIR}"

if [ -f "${BISENET_PATH}" ]; then
  echo "BiSeNet weights already exist at ${BISENET_PATH}"
else
  echo "Downloading BiSeNet weights to ${BISENET_PATH} ..."
  curl -L -o "${BISENET_PATH}" "${BISENET_URL}"
  echo "Done."
fi
