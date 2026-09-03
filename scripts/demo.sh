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
#
# The no-query path is the interesting one, and it is also the one that can
# legitimately come up empty: the index covers public figures, and a probe it
# cannot name exits non-zero. Under `set -e` that would kill the recording
# half way through, so it falls back to the explicit query instead of dying.
pipeline_ran=0
if "$PY" -m sigil.cli index info >/dev/null 2>&1; then
  run index info
  run identify "$PROBE" || true
  echo
  echo "\$ sigil run $PROBE          # no query: the face names itself"
  if "$PY" -m sigil.cli run "$PROBE"; then
    pipeline_ran=1
  else
    echo
    echo "  (the index could not name this face — falling back to an explicit query)"
  fi
else
  echo
  echo "  (no identity index — build one with 'sigil index build' to run the"
  echo "   pipeline from a face alone; falling back to an explicit query)"
fi

if [ "$pipeline_ran" -eq 0 ]; then
  if run run "$PROBE" -q "$QUERY"; then
    pipeline_ran=1
  fi
fi

if [ "$pipeline_ran" -eq 0 ]; then
  echo
  echo "No match was found, so there is no evidence bundle to anchor or tamper"
  echo "with. That is the honest outcome, not a crash — but it means the rest of"
  echo "this demo has nothing to show. Try a different probe or a broader query."
  exit 2
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

# The threshold everything above rests on, measured rather than asserted.
# --show reads the cached measurement, so this is instant; a fresh clone has
# not measured one yet, and says so rather than failing the recording.
if "$PY" -m sigil.cli calibrate --show >/dev/null 2>&1; then
  run calibrate --show
else
  echo
  echo "  (no calibration measured yet — run 'sigil calibrate' to measure what"
  echo "   the match threshold actually costs in false accepts and misses)"
fi
