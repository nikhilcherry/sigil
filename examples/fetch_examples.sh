#!/usr/bin/env bash
# Extra demo probes that are not committed here because their licences are not
# MIT-compatible for redistribution. Fetched from Wikimedia Commons on demand.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Jay Graber (Bluesky CEO) - CC BY-SA 4.0, by Jennifer 8. Lee.
curl -fsSL -o probe-jay-graber.jpg \
  "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Jay_Graber_at_the_2025_Knight_Media_Forum_6_%28cropped%29.jpg/1280px-Jay_Graber_at_the_2025_Knight_Media_Forum_6_%28cropped%29.jpg"
echo "fetched probe-jay-graber.jpg  (try: sigil run examples/probe-jay-graber.jpg -q 'jay graber')"
