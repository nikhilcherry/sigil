#!/usr/bin/env bash
# The full pipeline, one stage per command, for a screen recording.
#
# Every step below hits the live network and a live chain; nothing is stubbed.
# Pass a probe image and a query, or take the defaults.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-.venv/bin/python}"
PROBE="${1:-examples/probe-aoc.jpg}"
QUERY="${2:-AOC}"
run() { echo; echo "\$ sigil $*"; "$PY" -m sigil.cli "$@"; }

echo "=============================================================="
echo " sigil demo - $PROBE / query: $QUERY"
echo "=============================================================="

run chain reset --yes || true
run backends
run scan "$PROBE"
run run "$PROBE" -q "$QUERY"
run verify --probe "$PROBE" --recheck-source
run tamper --field match.text

echo
echo "\$ sigil verify -e artifacts/evidence.tampered.json   # expected to FAIL"
if "$PY" -m sigil.cli verify -e artifacts/evidence.tampered.json; then
  echo "UNEXPECTED: tampered bundle verified"; exit 1
else
  echo
  echo "Tampered bundle rejected, as it should be."
fi

run chain info
