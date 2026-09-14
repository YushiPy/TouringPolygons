#!/bin/sh
set -eu
cd "$(dirname "$0")"
command -v pandoc >/dev/null 2>&1 || {
  echo "pandoc is required to build the static report" >&2
  exit 1
}
node build.mjs
echo "Wrote $(pwd)/index.html"
