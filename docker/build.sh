#!/bin/bash

set -e
set -o nounset

# Get the path to this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Change to the root of the repository
cd "$DIR/../"

# Build the docker image
docker build -f "$DIR/Dockerfile" --tag="dmarcbriers/mc-pattern-synthesis:latest" .
