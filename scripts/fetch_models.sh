#!/usr/bin/env bash
# Fetch the ONNX weights for the opencv face backend (YuNet detector + SFace
# recogniser, ~37 MB total). Only needed if you are not using insightface.
#
# These live behind git-lfs in the OpenCV Zoo, so they must come from the LFS
# media host - raw.githubusercontent.com returns a 130-byte pointer file.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
BASE="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
mkdir -p "$DIR"

fetch() {
  local path="$1" name; name="$(basename "$path")"
  if [ -s "$DIR/$name" ] && [ "$(stat -c%s "$DIR/$name")" -gt 10000 ]; then
    echo "  have $name"; return
  fi
  echo "  fetching $name"
  curl -fsSL -o "$DIR/$name" "$BASE/$path"
}

echo "Fetching OpenCV face models into $DIR"
fetch "face_detection_yunet/face_detection_yunet_2023mar.onnx"
fetch "face_recognition_sface/face_recognition_sface_2021dec.onnx"
echo "Done."
