#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python3 benchmarks/tpp.py build tpp-unordered tpp-unordered-tests
.build/tools/bin/tpp-unordered-tests
.build/tools/bin/tpp-unordered < packages/nonconvex-tpp/cpp/tests/unordered-example.txt
