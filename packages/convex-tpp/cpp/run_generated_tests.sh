#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
generated_dir="$repo_root/.build/convex-generated-tests/tests"

python3 "$repo_root/benchmarks/tpp.py" build tpp-convex-generate-tests tpp-convex-verify-solutions
"$repo_root/.build/tools/bin/tpp-convex-generate-tests" "$generated_dir"
TPP_TEST_DIR="$script_dir/tests" "$repo_root/.build/tools/bin/tpp-convex-verify-solutions"
TPP_TEST_DIR="$generated_dir" "$repo_root/.build/tools/bin/tpp-convex-verify-solutions"
