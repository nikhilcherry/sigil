#!/usr/bin/env bash
# Fetch the ONNX weights for the opencv face backend (YuNet detector + SFace
# recogniser, ~37 MB total). Only needed if you are not using insightface.
#
# These live behind git-lfs in the OpenCV Zoo, so they must come from the LFS
# media host - raw.githubusercontent.com returns a 130-byte pointer file.
#
# Each file's SHA256 is pinned below and verified after download. These weights
# decide every embedding this tool produces, and therefore every match verdict
# and every hash it anchors: a substituted model changes every answer while
# leaving the pipeline, the tests and the chain records all looking correct. A
# project whose whole claim is tamper-evidence should not fetch its own
# decision-making weights on trust.
#
# The pins were taken from a fresh download and confirmed to reproduce. To
# update one deliberately, download it, check the hash against OpenCV Zoo, and
# change it here in the same commit.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
BASE="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
mkdir -p "$DIR"

SHA_face_detection_yunet_2023mar_onnx=8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
SHA_face_recognition_sface_2021dec_onnx=0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79

expected_for() {
  local var="SHA_${1//[.-]/_}"
  printf '%s' "${!var-}"
}

verify() {
  local file="$1" want="$2" got
  got="$(sha256sum "$file" | cut -d' ' -f1)"
  if [ "$got" != "$want" ]; then
    echo "  CHECKSUM MISMATCH for $(basename "$file")" >&2
    echo "    expected $want" >&2
    echo "    got      $got" >&2
    echo "  Refusing to keep it. These weights decide every match this tool" >&2
    echo "  reports, so a wrong file is worse than a missing one." >&2
    rm -f "$file"
    return 1
  fi
}

fetch() {
  local path="$1" name want
  name="$(basename "$path")"
  want="$(expected_for "$name")"
  if [ -z "$want" ]; then
    echo "  no pinned checksum for $name - refusing to fetch it" >&2
    return 1
  fi

  if [ -s "$DIR/$name" ] && verify "$DIR/$name" "$want" 2>/dev/null; then
    echo "  have $name (checksum ok)"
    return
  fi

  echo "  fetching $name"
  curl -fsSL -o "$DIR/$name" "$BASE/$path"
  verify "$DIR/$name" "$want"
  echo "  verified $name"
}

echo "Fetching OpenCV face models into $DIR"
fetch "face_detection_yunet/face_detection_yunet_2023mar.onnx"
fetch "face_recognition_sface/face_recognition_sface_2021dec.onnx"
echo "Done."
