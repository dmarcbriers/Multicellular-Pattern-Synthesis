#!/bin/bash

set -e
set -o nounset

# Get the path to this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Change to the root of the repository
cd "$DIR/../"

if [ ! -e "$DIR/work" ]; then
  mkdir -p "$DIR/work"
fi

# -p - Map 8888 from guest to host
# -e - Run jupyter lab
# -v - Map the current working directory to the "work" folder
docker run -it --rm \
  -p 8888:8888 \
  -v "${DIR}/work":/home/synth/work:rw \
  -it dmarcbriers/mc-pattern-synthesis:latest \
  /home/synth/.venv/bin/jupyter lab
