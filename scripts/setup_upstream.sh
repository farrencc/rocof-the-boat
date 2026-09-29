#!/usr/bin/env bash
# Fetch the read-only upstream simulator into external/ (gitignored).
# Upstream has no LICENSE: never copy its source into this repo; import it via src/upstream.py.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d external/EIEG_Hackathon26/.git ]; then
  echo "external/EIEG_Hackathon26 already present"; git -C external/EIEG_Hackathon26 log -1 --format='upstream @ %H'
  exit 0
fi
mkdir -p external
git clone --depth 1 https://github.com/ciangregg/EIEG_Hackathon26 external/EIEG_Hackathon26
git -C external/EIEG_Hackathon26 log -1 --format='upstream @ %H'
