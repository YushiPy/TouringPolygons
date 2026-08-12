#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
build_dir="$repo_root/build/convex-generated-tests"
generated_dir="$build_dir/tests"

cmake -S "$script_dir" -B "$build_dir" -DTARGET=main-generate_tests
cmake --build "$build_dir"
"$build_dir/tpp-convex" "$generated_dir"

cmake -S "$script_dir" -B "$build_dir" -DTARGET=main-verify_solutions
cmake --build "$build_dir"
TPP_TEST_DIR="$script_dir/tests" "$build_dir/tpp-convex"
TPP_TEST_DIR="$generated_dir" "$build_dir/tpp-convex"
