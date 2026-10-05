#!/usr/bin/env bash
# Interactive builder for `tpp.py bench` commands (arrow keys + Enter).
# Prints, copies or runs the command you compose. Uses only the standard library.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${TPP_PYTHON:-python3}" "$ROOT/benchmarks/tpp.py" tui "$@"
