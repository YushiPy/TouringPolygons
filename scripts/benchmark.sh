#!/usr/bin/env sh
# Interactive builder for `tpp.py bench` commands (arrow keys + Enter).
# Prints, copies or runs the command you compose. Uses only the standard library.
# POSIX sh on purpose: it works run directly or through bash, zsh or sh (not sourced).
set -eu

ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
exec "${TPP_PYTHON:-python3}" "$ROOT/benchmarks/tpp.py" tui "$@"
