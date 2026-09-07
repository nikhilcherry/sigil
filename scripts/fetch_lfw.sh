#!/usr/bin/env bash
# Fetch Labeled Faces in the Wild (funneled, ~232 MB) for scripts/bench_lfw.py.
#
# Only needed to reproduce the benchmark; nothing in the pipeline uses it.
#
# The canonical host, vis-www.cs.umass.edu, is often unreachable, so this pulls
# from the figshare mirror that scikit-learn also uses. The archive's SHA256 is
# pinned and checked: a benchmark is a claim about this tool's accuracy, and a
# claim measured against an unverified copy of the yardstick is worth less than
# no claim at all.
#
# The mirror throttles hard per connection - about 80 KB/s, or 45 minutes for
# this file - but honours range requests, so the download is split across eight
# of them and takes well under a minute. If that ever starts failing, set
# PARTS=1 for a plain single-stream download and wait it out.
set -euo pipefail

DIR="${LFW_HOME:-$HOME/scikit_learn_data/lfw_home}"
PARTS="${PARTS:-8}"

ARCHIVE_URL="https://ndownloader.figshare.com/files/5976015"
ARCHIVE_BYTES=243346528
ARCHIVE_SHA=b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a
# Cross-checked against the md5 the LFW authors publish for this archive,
# 1b42dfed7d15c9b2dd63d5e5840c86ad, so the mirror is serving the real thing.

# pairs.txt is the protocol itself - which images to compare, and the ten-fold
# split. Without it the dataset is just photographs.
PAIRS_URL="https://ndownloader.figshare.com/files/5976006"
PAIRS_SHA=ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592

mkdir -p "$DIR"
cd "$DIR"

if [ ! -f pairs.txt ]; then
  echo "fetching pairs.txt"
  curl -sSL --retry 3 -o pairs.txt "$PAIRS_URL"
fi
pairs_sha=$(sha256sum pairs.txt | cut -d' ' -f1)
if [ "$pairs_sha" != "$PAIRS_SHA" ]; then
  echo "pairs.txt sha256 $pairs_sha, expected $PAIRS_SHA - refusing" >&2
  exit 1
fi

if [ -d lfw_funneled ]; then
  echo "lfw_funneled/ already present in $DIR - nothing to do"
  exit 0
fi

if [ ! -f lfw-funneled.tgz ]; then
  echo "fetching lfw-funneled.tgz ($((ARCHIVE_BYTES / 1048576)) MB) in $PARTS streams"
  chunk=$(( (ARCHIVE_BYTES + PARTS - 1) / PARTS ))
  for i in $(seq 0 $((PARTS - 1))); do
    start=$((i * chunk)); end=$((start + chunk - 1))
    [ "$end" -ge "$ARCHIVE_BYTES" ] && end=$((ARCHIVE_BYTES - 1))
    curl -sSL --retry 5 --retry-delay 2 -r "${start}-${end}" -o "lfw.part.$i" "$ARCHIVE_URL" &
  done
  wait
  cat $(for i in $(seq 0 $((PARTS - 1))); do echo "lfw.part.$i"; done) > lfw-funneled.tgz
  rm -f lfw.part.*
fi

actual_size=$(stat -c%s lfw-funneled.tgz)
if [ "$actual_size" -ne "$ARCHIVE_BYTES" ]; then
  echo "lfw-funneled.tgz is $actual_size bytes, expected $ARCHIVE_BYTES - refusing" >&2
  exit 1
fi

actual_sha=$(sha256sum lfw-funneled.tgz | cut -d' ' -f1)
if [ "$actual_sha" != "$ARCHIVE_SHA" ]; then
  echo "lfw-funneled.tgz sha256 $actual_sha, expected $ARCHIVE_SHA - refusing" >&2
  exit 1
fi

echo "extracting"
tar xzf lfw-funneled.tgz
echo "ready: $DIR/lfw_funneled ($(find lfw_funneled -name '*.jpg' | wc -l) images)"
echo "now run:  .venv/bin/python scripts/bench_lfw.py"
