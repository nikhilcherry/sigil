#!/usr/bin/env bash
# The full pipeline, one stage per command, for a screen recording.
#
# Every step below hits the live network and a live chain; nothing is stubbed.
# Pass a probe image and a query, or take the defaults.
#
# For a visual recording instead, run `sigil serve` and drive the same pipeline
# from the browser at http://127.0.0.1:8099.
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

# Face -> name, before any search. Skipped cleanly if no index has been built.
if "$PY" -m sigil.cli index info >/dev/null 2>&1; then
  run index info
  run identify "$PROBE"
  echo
  echo "\$ sigil run $PROBE          # no query: the face names itself"
  "$PY" -m sigil.cli run "$PROBE"
else
  echo
  echo "  (no identity index — build one with 'sigil index build' to run the"
  echo "   pipeline from a face alone; falling back to an explicit query)"
  run run "$PROBE" -q "$QUERY"
fi
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
